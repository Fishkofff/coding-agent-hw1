from __future__ import annotations

import copy
import difflib
import json

from agent import prompts
from agent.budget import MIN_RESULT_ALLOWANCE, Budget, fit_result
from agent.faults import Faults
from agent.journal import Journal
from agent.jsonschema import coerce_json_strings, validate
from agent.llm import ModelError
from agent.plan import tasks_from_plan, validate_plan, verify_task
from agent.policy import MODES, ArgumentProblem, PolicyEngine, PolicyViolation, approval_requirement
from agent.report import Reporter, format_changes, shorten
from agent.store import RunStore
from agent.tools import EXECUTION, PLANNING, REGISTRY, WORK_TOOL_NAMES, ToolExecutor, offered_tools
from agent.util import canonical_json, dumps, now_iso, sha256_bytes

TERMINAL_STATUSES = frozenset({"completed", "failed", "aborted", "rolled_back"})
FINAL_CALL_STATUSES = frozenset({"completed", "failed", "rejected", "denied"})
MAX_NUDGES = 3


class UserError(Exception):
    ""

def fingerprint(name: str, arguments_raw: str) -> str:
    try:
        canonical = canonical_json(json.loads(arguments_raw or "{}"))
    except ValueError:
        canonical = arguments_raw
    return sha256_bytes(f"{name}\n{canonical}".encode("utf-8"))


def parse_arguments(raw: str) -> tuple[dict | None, str | None]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        return None, str(exc)
    if not isinstance(value, dict):
        return None, "arguments must be a JSON object"
    return value, None


def err(code: str, message: str, **extra) -> dict:
    return {"ok": False, "error": code, "message": message, **extra}


class AgentRuntime:
    def __init__(self, policy_path, model=None, *, reporter: Reporter | None = None,
                 faults: Faults | None = None, executor: ToolExecutor | None = None):
        self.engine = PolicyEngine(policy_path)
        policy = self.engine.load()
        self.store = RunStore(policy.runs_dir)
        self.model = model
        self.out = reporter or Reporter()
        self.faults = faults or Faults()
        self.executor = executor or ToolExecutor()
        self.state: dict | None = None

    def start(self, task: str, *, mode: str = "safe", plan_mode: bool = True,
              budget: dict | None = None, model_info: dict | None = None) -> str:
        if mode not in MODES:
            raise UserError(f"mode must be one of {MODES}")
        policy = self.engine.load()
        limits = dict(policy.budget)
        limits.update(budget or {})
        run_id = self.store.new_run_id()
        self.state = {
            "version": 1, "run_id": run_id, "created_at": now_iso(), "updated_at": None,
            "task": task, "mode": mode, "plan_mode": plan_mode,
            "phase": PLANNING if plan_mode else EXECUTION,
            "status": "running", "pending": None, "error": None, "final_answer": None,
            "model": model_info,
            "messages": [
                {"role": "system", "content": prompts.system_prompt(policy, mode)},
                {"role": "user", "content": prompts.initial_user_message(task, plan_mode)},
            ],
            "batch": None,
            "planning_calls": {},
            "execution_calls": {},
            "budget": Budget.initial(limits),
            "plan": None, "plan_history": [], "tasks": [], "current_task": None,
            "final_requested": False, "nudges": 0,
            "journal": Journal.initial_state(),
        }
        self.store.create_run_dir(run_id)
        with self.store.lock(run_id):
            self._journal().init_baseline(policy)
            self._save()
        self._event("run_started", mode=mode, plan_mode=plan_mode, task=task)
        self.out.run_started(self.state)
        return run_id

    def run(self, task: str, **kwargs) -> str:
        run_id = self.start(task, **kwargs)
        self.drive()
        return run_id

    def load(self, run_id: str) -> dict:
        self.state = self.store.load(run_id)
        return self.state

    def drive(self) -> str:
        run_id = self.state["run_id"]
        with self.store.lock(run_id):
            self._load_locked(run_id)
            while self.state["status"] == "running":
                if self.state["batch"] is not None:
                    self._process_batch()
                else:
                    self._model_turn()
        self.out.stopped(self.state, self.pending_details())
        return self.state["status"]

    def resume(self, run_id: str) -> str:
        with self.store.lock(run_id):
            self._load_locked(run_id)
            if self.state["status"] == "interrupted":
                self.state["status"] = "running"
                self.state["error"] = None
                self._event("resumed")
                self._save()
        if self.state["status"] in TERMINAL_STATUSES:
            self.out.stopped(self.state)
            return self.state["status"]
        return self.drive()

    def _save(self) -> None:
        self.state["updated_at"] = now_iso()
        self.store.save(self.state)

    def _event(self, event_name: str, **data) -> None:
        self.store.append_event(self.state["run_id"], event_name, **data)

    def _journal(self) -> Journal:
        return Journal(self.store, self.state["run_id"], self.state["journal"])

    def _budget(self) -> Budget:
        return Budget(self.state["budget"])

    def _calls_for(self, phase: str) -> dict:
        return self.state["planning_calls"] if phase == PLANNING else self.state["execution_calls"]

    def all_calls(self) -> list[dict]:
        return list(self.state["planning_calls"].values()) + list(self.state["execution_calls"].values())

    def _find_call(self, call_id: str) -> dict | None:
        return self.state["planning_calls"].get(call_id) or self.state["execution_calls"].get(call_id)

    def _task(self, task_id: str) -> dict | None:
        return next((t for t in self.state["tasks"] if t["id"] == task_id), None)

    def _fail(self, code: str, message: str) -> None:
        self.state["status"] = "failed"
        self.state["error"] = {"code": code, "message": message, "at": now_iso()}
        self._event("run_failed", code=code, message=message)
        self._save()

    def _load_locked(self, run_id: str) -> None:
        self.state = self.store.load(run_id)
        self._reconcile()

    def _reconcile(self) -> None:
        st = self.state
        if st["status"] in TERMINAL_STATUSES:
            return
        changed = False
        for record in self.all_calls():
            if record["status"] != "started":
                continue
            spec = REGISTRY[record["name"]]
            attempt = record["attempts"][-1]
            changed = True
            if not spec.side_effect:
                attempt["status"] = "interrupted"
                record["status"] = "approved" if record.get("decision") else "new"
                self._event("read_call_interrupted", call_id=record["id"])
                continue
            changes = self._journal().changes_since(self.engine.load(), attempt["pre_snapshot"])
            attempt["status"] = "uncertain"
            record["status"] = "uncertain"
            st["status"] = "waiting"
            st["pending"] = {"kind": "recovery", "call_id": record["id"], "changes": changes, "since": now_iso()}
            self._event("uncertain_call", call_id=record["id"], tool=record["name"],
                        changed_paths=[c["path"] for c in changes])
        if changed:
            self._save()

    def _model_turn(self) -> None:
        st = self.state
        if st["phase"] == EXECUTION and st["tasks"]:
            self._schedule_tasks()
            if st["status"] != "running":
                return
        if self.model is None:
            st["status"] = "interrupted"
            st["error"] = {"code": "no_model", "message": "model is not configured (OPENAI_* variables or --scripted)"}
            self._save()
            return
        budget = self._budget()
        if budget.remaining("model_calls") < 1:
            self._fail("budget_exhausted", "model_calls limit reached: " + budget.summary())
            return
        budget.charge("model_calls", 1)
        self._save()
        self.faults.hit("after_model_charge")
        tools = [spec.to_openai() for spec in offered_tools(st["phase"], bool(st["tasks"]))]
        try:
            response = self.model.complete(copy.deepcopy(st["messages"]), tools)
        except ModelError as exc:
            st["status"] = "interrupted"
            st["error"] = {"code": "model_error", "message": str(exc), "at": now_iso()}
            self._event("model_error", message=str(exc))
            self._save()
            return
        self.out.assistant(response.content)
        if response.tool_calls:
            st["nudges"] = 0
            self._register_batch(response)
        else:
            st["messages"].append({"role": "assistant", "content": response.content or ""})
            self._on_text_answer(response.content or "")
        self._save()

    def _register_batch(self, response) -> None:
        st = self.state
        phase = st["phase"]
        task_id = st["current_task"] if phase == EXECUTION else None
        items, wire_calls, new_records, seen = [], [], [], set()
        for index, call in enumerate(response.tool_calls):
            call_id = call.id or f"rt-{len(st['messages'])}-{index}"
            wire_calls.append({"id": call_id, "type": "function",
                               "function": {"name": call.name, "arguments": call.arguments}})
            print_id = fingerprint(call.name, call.arguments)
            if call_id in seen:
                items.append({"id": call_id, "kind": "protocol_error", "result": err(
                    "protocol_error", f"tool_call_id {call_id!r} is used twice in one response; the call was not executed")})
                continue
            seen.add(call_id)
            existing = self._find_call(call_id)
            if existing is not None:
                if existing["fingerprint"] == print_id:
                    items.append({"id": call_id, "kind": "replay"})
                else:
                    items.append({"id": call_id, "kind": "protocol_error", "result": err(
                        "protocol_error", f"tool_call_id {call_id!r} was already used for {existing['name']} "
                                          f"with other arguments; the call was not executed")})
                continue
            arguments, parse_error = parse_arguments(call.arguments)
            new_records.append({
                "id": call_id, "name": call.name, "arguments_raw": call.arguments, "arguments": arguments,
                "parse_error": parse_error, "fingerprint": print_id, "phase": phase, "task_id": task_id,
                "status": "new", "decision": None, "attempts": [], "result": None, "created_at": now_iso(),
            })
            items.append({"id": call_id, "kind": "call"})
        st["messages"].append({"role": "assistant", "content": response.content or "", "tool_calls": wire_calls})

        need_calls = len(new_records)
        need_side = sum(1 for r in new_records if self._is_side_effect(r))
        budget = self._budget()
        shortage = budget.shortfall(tool_calls=need_calls, side_effects=need_side)
        if shortage:
            reason = (f"Budget exceeded: this response has {need_calls} tool call(s) ({need_side} with side effects) "
                      f"but the remaining budget is not enough ({'; '.join(shortage)}). "
                      f"No call from this response was executed.")
            for record in new_records:
                record["status"] = "rejected"
                record["result"] = err("budget_exceeded", reason)
            self._event("batch_rejected_budget", calls=[r["id"] for r in new_records], shortage=shortage)
        elif need_calls:
            budget.charge("tool_calls", need_calls)
        for record in new_records:
            self._calls_for(phase)[record["id"]] = record
            self.out.tool_call(record)
        if shortage:
            self.out.line(f"   x the whole batch was rejected: {'; '.join(shortage)}")
        st["batch"] = {"items": items, "created_at": now_iso()}

    @staticmethod
    def _is_side_effect(record: dict) -> bool:
        spec = REGISTRY.get(record["name"])
        return bool(spec and spec.side_effect and record["phase"] in spec.phases)

    def _on_text_answer(self, text: str) -> None:
        st = self.state
        if st["phase"] == PLANNING:
            self._nudge(prompts.NUDGE_PLAN)
            return
        unfinished = [t for t in st["tasks"] if t["status"] != "done"]
        if unfinished:
            self._nudge(prompts.nudge_task(st["current_task"] or unfinished[0]["id"]))
            return
        st["final_answer"] = text
        st["status"] = "completed"
        st["phase"] = "done"
        st["completed_at"] = now_iso()
        self._event("run_completed")

    def _nudge(self, message: str) -> None:
        st = self.state
        st["nudges"] += 1
        if st["nudges"] > MAX_NUDGES:
            self._fail("model_stuck", f"the model answered with text {MAX_NUDGES} times in a row without progress")
            return
        st["messages"].append({"role": "user", "content": message})

    def _schedule_tasks(self) -> None:
        st = self.state
        if st["current_task"]:
            return
        done = {t["id"] for t in st["tasks"] if t["status"] == "done"}
        pending = [t for t in st["tasks"] if t["status"] == "pending"]
        if not pending:
            if not st["final_requested"]:
                st["final_requested"] = True
                st["messages"].append({"role": "user", "content": prompts.ALL_TASKS_DONE})
                self._save()
            return
        ready = [t for t in pending if all(dep in done for dep in t["depends_on"])]
        if not ready:
            self._fail("tasks_blocked", "no task can start: dependencies are not satisfied")
            return
        task = ready[0]
        task["status"] = "in_progress"
        task["started_at"] = now_iso()
        st["current_task"] = task["id"]
        st["messages"].append({"role": "user", "content": prompts.task_message(task)})
        self._event("task_started", task_id=task["id"])
        self._save()
        self.out.checklist(st["tasks"])

    def _process_batch(self) -> None:
        st = self.state
        batch = st["batch"]
        for item in batch["items"]:
            if item["kind"] == "replay" and "result" not in item:
                original = self._find_call(item["id"])
                item["result"] = {
                    "ok": bool((original.get("result") or {}).get("ok")), "replayed": True,
                    "message": "This tool_call_id was already processed; the stored result is returned and the "
                               "action is NOT executed again.",
                    "original_result": original.get("result"),
                }
                self._event("call_replayed", call_id=item["id"])
                self.out.line(f"-> {original['name']} id={item['id']}: repeated tool_call_id, returning the stored result")
            if item["kind"] != "call":
                continue
            record = self._find_call(item["id"])
            if record["status"] in FINAL_CALL_STATUSES:
                continue
            if self._process_call(record) or st["status"] != "running":
                self._save()
                return
        for item in batch["items"]:
            result = item["result"] if item["kind"] != "call" else self._find_call(item["id"])["result"]
            st["messages"].append({"role": "tool", "tool_call_id": item["id"], "content": dumps(result)})
        st["batch"] = None
        self._save()

    def _process_call(self, record: dict) -> bool:
        spec = REGISTRY.get(record["name"])
        if spec is None:
            return self._reject(record, "unknown_tool",
                                f"Unknown tool {record['name']!r}. Available tools: {', '.join(sorted(REGISTRY))}")
        if record["phase"] not in spec.phases:
            if record["phase"] == PLANNING:
                message = (f"{spec.name} is blocked in PLAN MODE: until the user approves a plan only read-only "
                           f"tools and submit_plan are allowed.")
            else:
                message = f"{spec.name} is not available in the execution phase."
            return self._reject(record, "not_allowed_in_phase", message)
        if record["arguments"] is None:
            return self._reject(record, "invalid_arguments", f"arguments are not a valid JSON object: {record['parse_error']}")
        args = coerce_json_strings(record["arguments"], spec.parameters)
        record["arguments"] = args
        problems = validate(args, spec.parameters)
        if problems:
            return self._reject(record, "invalid_arguments", "; ".join(problems))
        if spec.effect == "control":
            return self._submit_plan(record, args) if spec.name == "submit_plan" else self._complete_task(record, args)

        decision = record.get("decision")
        if decision is not None and not decision["approved"]:
            return self._finish(record, "denied", err(
                "denied_by_user", "The user denied this action. Do not repeat it; choose another approach or finish.",
                reason=decision.get("reason") or "no reason given"))
        policy = self.engine.load()
        try:
            checked = self.engine.check_call(policy, spec.name, args)
        except PolicyViolation as exc:
            return self._reject(record, "policy_denied", str(exc))
        except ArgumentProblem as exc:
            return self._reject(record, "invalid_arguments", str(exc))
        requirement = approval_requirement(spec.effect, self.state["mode"], record["phase"])
        if requirement == "forbidden":
            return self._reject(record, "not_allowed_in_phase", f"{spec.name} is not allowed in phase {record['phase']}")
        if requirement == "confirm" and decision is None:
            record["status"] = "awaiting_confirmation"
            self.state["status"] = "waiting"
            self.state["pending"] = {"kind": "confirmation", "call_id": record["id"], "since": now_iso()}
            self._event("confirmation_requested", call_id=record["id"], tool=spec.name)
            self._save()
            return True
        return self._execute(record, spec, checked, policy)

    def _reject(self, record: dict, code: str, message: str) -> bool:
        return self._finish(record, "rejected", err(code, message))

    def _finish(self, record: dict, status: str, result: dict) -> bool:
        record["status"] = status
        record["result"] = result
        record["finished_at"] = now_iso()
        self._event("call_finished", call_id=record["id"], tool=record["name"], status=status, error=result.get("error"))
        self.out.tool_outcome(record)
        return False

    def _execute(self, record: dict, spec, checked, policy) -> bool:
        st = self.state
        budget = self._budget()
        is_retry = bool(record["attempts"])
        need = {"tool_calls": 1 if is_retry else 0, "side_effects": 1 if spec.side_effect else 0}
        shortage = budget.shortfall(**need)
        if budget.remaining("result_bytes") < MIN_RESULT_ALLOWANCE:
            shortage.append(f"result_bytes: remaining {budget.remaining('result_bytes')} "
                            f"is below the minimum {MIN_RESULT_ALLOWANCE}")
        if shortage:
            return self._reject(record, "budget_exceeded", "The action was not started: " + "; ".join(shortage))

        journal = self._journal()
        pre_snapshot = None
        if spec.side_effect:
            drift, current = journal.detect_drift(policy)
            if drift:
                record["status"] = "conflict"
                st["status"] = "waiting"
                st["pending"] = {"kind": "conflict", "call_id": record["id"], "changes": drift, "since": now_iso()}
                self._event("workspace_conflict", call_id=record["id"], paths=[c["path"] for c in drift])
                self._save()
                return True
            pre_snapshot = journal.save_snapshot(current, f"pre-{record['id']}")

        attempt = {"n": len(record["attempts"]) + 1, "status": "started", "started_at": now_iso(),
                   "pre_snapshot": pre_snapshot}
        for key, amount in need.items():
            if amount:
                budget.charge(key, amount)
        record["attempts"].append(attempt)
        record["status"] = "started"
        self._event("call_started", call_id=record["id"], tool=spec.name, attempt=attempt["n"])
        self._save()
        self.faults.hit("after_tool_start", spec.name)

        allowance = budget.remaining("result_bytes")
        raw_result = self.executor.execute(checked, policy, allowance)
        self.faults.hit("after_tool_execute", spec.name)

        result, size = fit_result(raw_result, allowance)
        budget.charge("result_bytes", size)
        if spec.side_effect:
            entry = journal.record(policy, kind="action", pre_snapshot=pre_snapshot, call_id=record["id"],
                                   attempt=attempt["n"], task_id=record["task_id"], tool=spec.name)
            attempt["post_snapshot"] = entry["post_snapshot"]
            attempt["journal_entry"] = entry["seq"]
            if spec.name == "Bash":
                record["tree_digest_after"] = journal.expected_digest()
        attempt["status"] = "completed"
        attempt["finished_at"] = now_iso()
        record["status"] = "completed"
        record["result"] = result
        record["finished_at"] = attempt["finished_at"]
        self._event("call_completed", call_id=record["id"], tool=spec.name, attempt=attempt["n"], result_bytes=size)
        self._save()
        self.out.tool_outcome(record)
        self.faults.hit("after_result_save", spec.name)
        return False

    def _submit_plan(self, record: dict, plan: dict) -> bool:
        st = self.state
        policy = self.engine.load()
        problems = validate_plan(plan, tool_names=WORK_TOOL_NAMES, engine=self.engine, policy=policy)
        if problems:
            return self._finish(record, "completed", err(
                "invalid_plan", "The runtime did not accept the plan. Fix the problems and call submit_plan again.",
                problems=problems))
        version = len(st["plan_history"]) + 1
        st["plan"] = {"version": version, "status": "awaiting_approval", "submitted_at": now_iso(),
                      "call_id": record["id"], **copy.deepcopy(plan)}
        record["status"] = "awaiting_plan_approval"
        st["status"] = "waiting"
        st["pending"] = {"kind": "plan_approval", "call_id": record["id"], "version": version, "since": now_iso()}
        self._event("plan_submitted", version=version)
        self._save()
        self.out.plan(st["plan"])
        return True

    def _complete_task(self, record: dict, args: dict) -> bool:
        st = self.state
        task_id = args["task_id"]
        if not st["tasks"]:
            return self._finish(record, "completed", err("no_tasks", "This run has no planned tasks."))
        task = self._task(task_id)
        if task is None:
            return self._finish(record, "completed", err("unknown_task", f"Unknown task {task_id!r}."))
        if task["status"] == "done":
            return self._finish(record, "completed", {"ok": True, "task_id": task_id, "already_done": True,
                                                      "message": "The task is already verified."})
        if task_id != st["current_task"]:
            return self._finish(record, "completed", err(
                "not_current_task", f"Task {task_id} is not the current task (current: {st['current_task']}). "
                                    f"A task starts only after its dependencies are verified."))
        ok, evidence, failures = verify_task(task, calls=st["execution_calls"], journal=self._journal(),
                                             engine=self.engine, policy=self.engine.load())
        task["verify_attempts"] += 1
        if not ok:
            task["last_failures"] = failures
            self._event("task_verification_failed", task_id=task_id, failures=failures)
            return self._finish(record, "completed", err(
                "verification_failed", f"Task {task_id} is NOT marked as done: the runtime could not confirm its "
                                       f"checks. Satisfy them and call complete_task again.", failed_checks=failures))
        task.update(status="done", evidence=evidence, summary=args.get("summary", ""), completed_at=now_iso(),
                    completed_by_call=record["id"])
        st["current_task"] = None
        self._event("task_done", task_id=task_id, evidence=evidence)
        self._finish(record, "completed", {"ok": True, "task_id": task_id, "verified": True, "evidence": evidence})
        if all(t["status"] == "done" for t in st["tasks"]):
            self.out.checklist(st["tasks"])
        return False

    def _require_pending(self, *kinds: str) -> dict:
        pending = self.state.get("pending") or {}
        if self.state["status"] != "waiting" or pending.get("kind") not in kinds:
            current = pending.get("kind") or self.state["status"]
            raise UserError(f"run {self.state['run_id']} is not waiting for {' / '.join(kinds)} (now: {current})")
        return pending

    def approve(self, run_id: str, call_id: str | None = None) -> str:
        return self._decide(run_id, True, None, call_id)

    def deny(self, run_id: str, reason: str | None = None, call_id: str | None = None) -> str:
        return self._decide(run_id, False, reason, call_id)

    def _decide(self, run_id: str, approved: bool, reason: str | None, call_id: str | None) -> str:
        with self.store.lock(run_id):
            self._load_locked(run_id)
            pending = self._require_pending("confirmation")
            if call_id and call_id != pending["call_id"]:
                raise UserError(f"pending call is {pending['call_id']}, not {call_id}")
            record = self._find_call(pending["call_id"])
            record["decision"] = {"approved": approved, "reason": reason, "by": "user", "at": now_iso()}
            record["status"] = "approved" if approved else "new"
            self.state["pending"] = None
            self.state["status"] = "running"
            self._event("user_decision", call_id=record["id"], approved=approved, reason=reason)
            self._save()
        return self.drive()

    def approve_plan(self, run_id: str) -> str:
        with self.store.lock(run_id):
            self._load_locked(run_id)
            pending = self._require_pending("plan_approval")
            st = self.state
            plan = st["plan"]
            plan["status"] = "approved"
            plan["approved_at"] = now_iso()
            st["plan_history"].append(copy.deepcopy(plan))
            st["tasks"] = tasks_from_plan(plan)
            st["phase"] = EXECUTION
            st["current_task"] = None
            self._finish(self._find_call(pending["call_id"]), "completed", {
                "ok": True, "approved": True, "version": plan["version"],
                "message": "The user approved the plan. EXECUTION phase started; the runtime assigns tasks one by one."})
            st["pending"] = None
            st["status"] = "running"
            self._event("plan_approved", version=plan["version"])
            self._save()
        return self.drive()

    def reject_plan(self, run_id: str, reason: str | None = None) -> str:
        with self.store.lock(run_id):
            self._load_locked(run_id)
            pending = self._require_pending("plan_approval")
            st = self.state
            plan = st["plan"]
            plan.update(status="rejected", rejected_at=now_iso(), rejection_reason=reason)
            st["plan_history"].append(copy.deepcopy(plan))
            st["plan"] = None
            self._finish(self._find_call(pending["call_id"]), "completed", err(
                "plan_rejected", "The user rejected the plan. Revise it using the feedback and call submit_plan again.",
                reason=reason or "no reason given"))
            st["pending"] = None
            st["status"] = "running"
            self._event("plan_rejected", version=plan["version"], reason=reason)
            self._save()
        return self.drive()

    def recover(self, run_id: str, action: str, changes: str | None = None) -> str:
        """Решение пользователя по неопределённому после сбоя вызову (recovery) или по конфликту дерева.

        action:  retry — повторить (новая попытка того же tool call; только для повторяемых инструментов);
                 fail  — считать шаг неуспешным и продолжить;
                 abort — завершить запуск с ошибкой.
        changes: attribute — обнаруженные изменения сделаны этим запуском (попадут в журнал и откат);
                 external  — изменения посторонние (в журнал запуска не попадают).
        """
        if action not in ("retry", "fail", "abort"):
            raise UserError("action must be retry, fail or abort")
        if changes not in (None, "attribute", "external"):
            raise UserError("changes must be attribute or external")
        with self.store.lock(run_id):
            self._load_locked(run_id)
            st = self.state
            pending = self._require_pending("recovery", "conflict")
            kind = pending["kind"]
            record = self._find_call(pending["call_id"])
            spec = REGISTRY[record["name"]]
            policy = self.engine.load()
            journal = self._journal()
            if kind == "recovery":
                detected = journal.changes_since(policy, record["attempts"][-1]["pre_snapshot"])
            else:
                detected, _ = journal.detect_drift(policy)
            if detected and changes is None:
                paths = ", ".join(f"{c['op']} {c['path']}" for c in detected)
                raise UserError(f"workspace changes detected ({paths}); add --changes attribute (made by this run) "
                                f"or --changes external (made by someone else)")
            if kind == "conflict" and changes == "attribute":
                raise UserError("conflicting changes were made outside the agent; only --changes external is possible")
            if kind == "recovery" and action == "retry" and not spec.retryable:
                raise UserError(f"{spec.name} is not safely retryable; choose --action fail or --action abort")

            if detected and changes == "attribute":
                attempt = record["attempts"][-1]
                entry = journal.record(policy, kind="recovered", pre_snapshot=attempt["pre_snapshot"],
                                       call_id=record["id"], attempt=attempt["n"], task_id=record["task_id"],
                                       tool=spec.name, note="changes of an uncertain call attributed to this run by the user")
                attempt["post_snapshot"] = entry["post_snapshot"]
            elif detected:
                journal.record(policy, kind="external", pre_snapshot=None,
                               note=f"changes accepted as external by the user ({kind})")
            summary = ", ".join(f"{c['op']} {c['path']}" for c in detected) or "no file changes detected"

            if kind == "recovery":
                attempt = record["attempts"][-1]
                attempt["resolution"] = {"action": action, "changes": changes, "at": now_iso()}
                if action == "retry":
                    attempt["status"] = "uncertain_retried"
                    record["status"] = "approved"
                    record["decision"] = {"approved": True, "by": "user", "via": "recovery_retry", "at": now_iso()}
                else:
                    attempt["status"] = "uncertain_failed"
                    self._finish(record, "failed", err(
                        "outcome_unknown", f"The agent process crashed while {spec.name} was running, so its outcome "
                                           f"is unknown; the user marked this call as failed. Workspace: {summary}."))
            else:
                if action == "retry":
                    record["status"] = "approved" if record.get("decision") else "new"
                else:
                    self._finish(record, "rejected", err(
                        "workspace_conflict", f"The workspace was changed outside the agent ({summary}); "
                                              f"the action was not executed."))
            st["pending"] = None
            self._event("recovery_decision", call_id=record["id"], kind=kind, action=action, changes=changes)
            if action == "abort":
                st["status"] = "failed"
                st["error"] = {"code": f"aborted_after_{kind}", "message": f"aborted by the user after {kind}",
                               "at": now_iso()}
                self._save()
                self.out.stopped(st)
                return st["status"]
            st["status"] = "running"
            self._save()
        return self.drive()

    def abort(self, run_id: str, reason: str | None = None) -> str:
        with self.store.lock(run_id):
            self._load_locked(run_id)
            st = self.state
            if st["status"] in TERMINAL_STATUSES:
                raise UserError(f"run is already {st['status']}")
            if (st.get("pending") or {}).get("kind") == "recovery":
                raise UserError("an uncertain call must be resolved first: recover --action abort")
            st["status"] = "aborted"
            st["error"] = {"code": "aborted", "message": reason or "aborted by the user", "at": now_iso()}
            self._event("aborted", reason=reason)
            self._save()
        self.out.stopped(self.state)
        return "aborted"

    def diff_report(self, run_id: str) -> dict:
        state = self.store.load(run_id)
        policy = self.engine.load()
        journal = Journal(self.store, run_id, state["journal"])
        unresolved = [r["id"] for r in list(state["planning_calls"].values()) + list(state["execution_calls"].values())
                      if r["status"] in ("started", "uncertain")]
        return {"run_id": run_id, "files": journal.file_report(policy), "entries": state["journal"]["entries"],
                "rolled_back": state["journal"]["rolled_back"], "unresolved_calls": unresolved}

    def rollback(self, run_id: str) -> dict:
        with self.store.lock(run_id):
            self._load_locked(run_id)
            st = self.state
            if (st.get("pending") or {}).get("kind") == "recovery" or any(
                    r["status"] in ("started", "uncertain") for r in self.all_calls()):
                raise UserError("the run has an uncertain call after a crash: resolve it first with "
                                "`recover ... --changes attribute|external`; rollback is not performed until then")
            policy = self.engine.load()
            result = self._journal().rollback(policy, self.engine)
            st["status"] = "rolled_back"
            st["pending"] = None
            self._event("rolled_back", **result)
            self._save()
        return result

    def pending_details(self) -> str | None:
        st = self.state
        pending = st.get("pending")
        if not pending:
            return None
        record = self._find_call(pending["call_id"])
        where = f" [task {record['task_id']}]" if record.get("task_id") else ""
        kind = pending["kind"]
        if kind == "confirmation":
            return f"Требуется подтверждение: {record['name']}{where} id={record['id']}\n{self._preview(record)}"
        if kind == "plan_approval":
            return f"План v{pending['version']} ждёт решения пользователя (план выведен выше; показать: plan show)."
        if kind == "recovery":
            spec = REGISTRY[record["name"]]
            return (f"Неопределённый вызов после сбоя: {record['name']} id={record['id']}{where}.\n"
                    f"Он начался, но результат не был сохранён. Повтор: "
                    f"{'разрешён (инструмент идемпотентен)' if spec.retryable else 'запрещён (инструмент неповторяем)'}.\n"
                    f"Изменения дерева относительно снимка до вызова:\n{format_changes(pending['changes'])}")
        if kind == "conflict":
            return (f"Конфликт: перед {record['name']} id={record['id']} обнаружены посторонние изменения "
                    f"(действие не выполнялось):\n{format_changes(pending['changes'])}")
        return None

    def _preview(self, record: dict) -> str:
        args = record.get("arguments") or {}
        name = record["name"]
        if name == "Bash":
            return f"  $ {args.get('command')}   (cwd: {args.get('cwd')})"
        if name == "delete_file":
            return f"  удалить файл {args.get('path')}"
        if name == "write_file":
            policy = self.engine.load()
            try:
                path = self.engine.resolve_path(policy, args["path"])
                old = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
            except (PolicyViolation, OSError):
                old = ""
            diff = "".join(difflib.unified_diff(old.splitlines(keepends=True), args["content"].splitlines(keepends=True),
                                                f"a/{args['path']}", f"b/{args['path']}"))
            return "\n".join("  " + line for line in shorten(diff or "(content is unchanged)", 4000).splitlines())
        return f"  {shorten(json.dumps(args, ensure_ascii=False), 400)}"
