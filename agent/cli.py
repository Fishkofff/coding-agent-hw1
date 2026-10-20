"""CLI агента: python3 -m agent <команда>. Список команд: python3 -m agent --help."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from agent.budget import BUDGET_KEYS, Budget
from agent.faults import Faults
from agent.journal import RollbackConflict, RollbackError
from agent.llm import OpenAICompatibleModel, ScriptedModel
from agent.plan import describe_check, render_checklist
from agent.policy import MODES, ArgumentProblem, PolicyEngine, PolicyError, PolicyViolation, approval_requirement
from agent.report import format_args, format_changes, format_plan, shorten
from agent.runtime import AgentRuntime, UserError
from agent.store import RunBusy, RunNotFound
from agent.tools import EXECUTION, PLANNING, REGISTRY


def load_dotenv(path: Path) -> None:
    """Простой загрузчик .env (без зависимостей). Уже заданные переменные окружения не перезаписываются."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def make_model(args, state: dict | None = None):
    scripted = getattr(args, "scripted", None)
    if not scripted and state and (state.get("model") or {}).get("kind") == "scripted":
        scripted = state["model"]["path"]
    if scripted:
        path = str(Path(scripted).resolve())
        return ScriptedModel.from_file(path), {"kind": "scripted", "path": path}
    base_url, model = os.environ.get("OPENAI_BASE_URL"), os.environ.get("OPENAI_MODEL")
    if not base_url or not model:
        return None, None
    info = {"kind": "openai-compatible", "base_url": base_url, "model": model}
    return OpenAICompatibleModel(base_url, os.environ.get("OPENAI_API_KEY", ""), model), info


def make_runtime(args, run_id: str | None = None) -> AgentRuntime:
    faults = Faults(getattr(args, "crash_at", None), hard=True)
    runtime = AgentRuntime(args.policy, None, faults=faults)
    state = runtime.load(run_id) if run_id else None
    runtime.model, _info = make_model(args, state)
    return runtime


def parse_budget(text: str | None) -> dict:
    if not text:
        return {}
    limits = {}
    for part in text.split(","):
        key, _, value = part.partition("=")
        key = key.strip()
        if key not in BUDGET_KEYS or not value.strip().isdigit():
            raise UserError(f"bad budget item {part!r}; expected e.g. model_calls=20,tool_calls=40")
        limits[key] = int(value)
    return limits


def interactive(runtime: AgentRuntime, run_id: str, status: str) -> str:
    """-i: подтверждения прямо в консоли. Состояние уже сохранено, так что можно и прервать процесс."""
    while status == "waiting":
        kind = runtime.state["pending"]["kind"]
        try:
            if kind == "confirmation":
                answer = input("Разрешить действие? [y/N]: ").strip().lower()
                if answer in ("y", "yes", "д", "да"):
                    status = runtime.approve(run_id)
                else:
                    reason = input("Причина отказа (её увидит модель): ").strip()
                    status = runtime.deny(run_id, reason or None)
            elif kind == "plan_approval":
                answer = input("Утвердить план? [y/N]: ").strip().lower()
                if answer in ("y", "yes", "д", "да"):
                    status = runtime.approve_plan(run_id)
                else:
                    reason = input("Что исправить в плане: ").strip()
                    status = runtime.reject_plan(run_id, reason or None)
            else:
                return status
        except EOFError:
            return status
    return status


def cmd_run(args) -> int:
    model, info = make_model(args)
    if model is None:
        raise UserError("model is not configured: set OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL "
                        "(see .env.example) or use --scripted demo/<file>.json")
    runtime = AgentRuntime(args.policy, model, faults=Faults(args.crash_at, hard=True))
    run_id = runtime.start(args.task, mode=args.mode, plan_mode=not args.no_plan,
                           budget=parse_budget(args.budget), model_info=info)
    status = runtime.drive()
    if args.interactive:
        interactive(runtime, run_id, status)
    return 0


def _drive_command(args, action) -> int:
    runtime = make_runtime(args, args.run_id)
    status = action(runtime)
    if getattr(args, "interactive", False):
        interactive(runtime, args.run_id, status)
    return 0


def cmd_approve(args) -> int:
    return _drive_command(args, lambda rt: rt.approve(args.run_id, args.call))


def cmd_deny(args) -> int:
    return _drive_command(args, lambda rt: rt.deny(args.run_id, args.reason, args.call))


def cmd_resume(args) -> int:
    return _drive_command(args, lambda rt: rt.resume(args.run_id))


def cmd_recover(args) -> int:
    return _drive_command(args, lambda rt: rt.recover(args.run_id, args.action, args.changes))


def cmd_plan(args) -> int:
    if args.plan_command == "approve":
        return _drive_command(args, lambda rt: rt.approve_plan(args.run_id))
    if args.plan_command == "reject":
        return _drive_command(args, lambda rt: rt.reject_plan(args.run_id, args.reason))
    state = AgentRuntime(args.policy).load(args.run_id)
    plans = state["plan_history"] + ([state["plan"]] if state["plan"] and state["plan"] not in state["plan_history"] else [])
    if not plans:
        print("План ещё не отправлен.")
    for plan in plans:
        print(format_plan(plan))
        if plan.get("rejection_reason"):
            print(f"Причина отклонения: {plan['rejection_reason']}")
        print()
    return 0


def cmd_abort(args) -> int:
    AgentRuntime(args.policy).abort(args.run_id, args.reason)
    return 0


def cmd_status(args) -> int:
    runtime = AgentRuntime(args.policy)
    st = runtime.load(args.run_id)
    print(f"run_id:   {st['run_id']}")
    print(f"задача:   {shorten(st['task'], 300)}")
    print(f"режим:    {st['mode']}   plan mode: {'on' if st['plan_mode'] else 'off'}   фаза: {st['phase']}")
    status = st["status"]
    if status == "waiting":
        status += f" ({st['pending']['kind']})"
    started = [r["id"] for r in runtime.all_calls() if r["status"] == "started"]
    if started and st["status"] == "running":
        status += f"  — процесс прервался во время вызова {started[0]}; выполните resume"
    print(f"статус:   {status}")
    if st.get("error"):
        print(f"ошибка:   {st['error'].get('code')}: {st['error'].get('message')}")
    print(f"бюджет:   {Budget(st['budget']).summary()}")
    if st["plan"]:
        print(f"план:     v{st['plan']['version']} ({st['plan']['status']})")
    if st["tasks"]:
        print("Задачи:")
        print(render_checklist(st["tasks"]))
    details = runtime.pending_details()
    if details:
        print(details)
    if st.get("final_answer"):
        print(f"ответ:    {st['final_answer']}")
    return 0


def cmd_tasks(args) -> int:
    st = AgentRuntime(args.policy).load(args.run_id)
    if not st["tasks"]:
        print("У запуска нет задач (план ещё не утверждён или запуск без plan mode).")
        return 0
    print("Задачи:")
    print(render_checklist(st["tasks"]))
    for task in st["tasks"]:
        print(f"\n{task['id']} [{task['status']}] {task['description']}")
        print("  проверки: " + "; ".join(describe_check(c) for c in task["checks"]))
        if task.get("evidence"):
            print("  подтверждение runtime: " + json.dumps(task["evidence"], ensure_ascii=False))
        if task.get("last_failures") and task["status"] != "done":
            print("  последняя неудачная проверка: " + "; ".join(f["reason"] for f in task["last_failures"]))
    return 0


def cmd_calls(args) -> int:
    runtime = AgentRuntime(args.policy)
    runtime.load(args.run_id)
    for record in runtime.all_calls():
        where = record["task_id"] or ("planning" if record["phase"] == PLANNING else "-")
        attempts = f" attempts={len(record['attempts'])}" if record["attempts"] else ""
        print(f"{record['id']:<28} {where:<9} {record['status']:<22} {record['name']}({format_args(record)}){attempts}")
    return 0


def cmd_list(args) -> int:
    runtime = AgentRuntime(args.policy)
    runs = runtime.store.list_runs()
    if not runs:
        print("Запусков пока нет.")
    for st in runs:
        print(f"{st['run_id']}  {st['status']:<12} {st['mode']:<5} {shorten(st['task'], 70)}")
    return 0


def cmd_diff(args) -> int:
    report = AgentRuntime(args.policy).diff_report(args.run_id)
    if report["unresolved_calls"]:
        print(f"ВНИМАНИЕ: есть неопределённые после сбоя вызовы {report['unresolved_calls']}; "
              f"их изменения ещё не отнесены к запуску (resume / recover).")
    if report["rolled_back"]:
        print(f"Запуск откатан: {report['rolled_back']['at']}")
    print(f"Изменения файлов запуском {report['run_id']} (before -> after):")
    if not report["files"]:
        print("  (запуск не менял файлы)")
    for item in report["files"]:
        current = "совпадает с after" if not item["externally_modified"] else "ИЗМЕНЁН ИЗВНЕ (откат даст конфликт)"
        print(f"\n{item['op'].upper()} {item['path']}")
        print(f"  before:  {item['before'] or '(не было)'}")
        print(f"  after:   {item['after'] or '(удалён)'}")
        print(f"  сейчас:  {item['current'] or '(нет файла)'}  — {current}")
        print("\n".join("  " + line for line in item["diff"].rstrip("\n").splitlines()))
    if args.entries:
        print("\nЖурнал действий:")
        for entry in report["entries"]:
            label = entry["tool"] or entry["kind"]
            print(f"#{entry['seq']} {entry['kind']} {label} call={entry['call_id']} task={entry['task_id']} at={entry['at']}")
            print(format_changes(entry["changes"]))
    return 0


def cmd_rollback(args) -> int:
    result = AgentRuntime(args.policy).rollback(args.run_id)
    print(f"Откат выполнен. Восстановлены: {result['restored'] or '-'}; удалены созданные запуском: "
          f"{result['deleted'] or '-'}; без чистых изменений: {result['unchanged'] or '-'}")
    return 0


def cmd_policy_check(args) -> int:
    """Показать решение policy для одного вызова без модели: удобно на защите."""
    engine = PolicyEngine(args.policy)
    policy = engine.load()
    spec = REGISTRY.get(args.tool)
    if spec is None:
        raise UserError(f"unknown tool {args.tool!r}; known: {', '.join(REGISTRY)}")
    call_args = json.loads(args.arguments)
    from agent.jsonschema import validate
    problems = validate(call_args, spec.parameters)
    if problems:
        print("schema: ОТКЛОНЕНО — " + "; ".join(problems))
        return 1
    print("schema: ok")
    if spec.effect != "control":
        try:
            checked = engine.check_call(policy, spec.name, call_args)
            print(f"policy: разрешено (путь/cwd: {checked.rel})")
        except (PolicyViolation, ArgumentProblem) as exc:
            print(f"policy: ОТКЛОНЕНО — {exc}")
            return 1
    phase = PLANNING if args.phase == "planning" else EXECUTION
    requirement = "forbidden" if phase not in spec.phases else approval_requirement(spec.effect, args.mode, phase)
    meaning = {"auto": "исполняется сразу", "confirm": "нужно подтверждение пользователя",
               "forbidden": "запрещено в этой фазе"}[requirement]
    print(f"режим {args.mode}, фаза {phase}: {meaning}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--policy", default=os.environ.get("AGENT_POLICY", "policy.json"),
                        help="путь к policy-файлу (по умолчанию ./policy.json)")
    driving = argparse.ArgumentParser(add_help=False)
    driving.add_argument("--scripted", help="JSON-сценарий ответов модели вместо LLM (детерминированное демо)")
    driving.add_argument("--crash-at", help="имитация сбоя: точка[:инструмент], напр. after_tool_execute:write_file")
    driving.add_argument("-i", "--interactive", action="store_true", help="подтверждать действия прямо в консоли")

    parser = argparse.ArgumentParser(prog="python3 -m agent", description="Coding agent с tool calling")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", parents=[common, driving], help="новый запуск агента")
    p.add_argument("task", help="задача для агента")
    p.add_argument("--mode", choices=MODES, default="safe", help="safe (по умолчанию) или auto")
    p.add_argument("--no-plan", action="store_true", help="без Plan Mode (сразу исполнение, без задач)")
    p.add_argument("--budget", help="переопределить лимиты: model_calls=20,tool_calls=40,side_effects=10,result_bytes=50000")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("approve", parents=[common, driving], help="разрешить ожидающее действие")
    p.add_argument("run_id")
    p.add_argument("--call", help="id вызова (для проверки)")
    p.set_defaults(func=cmd_approve)

    p = sub.add_parser("deny", parents=[common, driving], help="отклонить ожидающее действие")
    p.add_argument("run_id")
    p.add_argument("--reason", help="причина (её получит модель)")
    p.add_argument("--call", help="id вызова (для проверки)")
    p.set_defaults(func=cmd_deny)

    p = sub.add_parser("plan", parents=[common], help="Plan Mode: show | approve | reject")
    plan_sub = p.add_subparsers(dest="plan_command", required=True)
    for name, text in (("show", "показать план"), ("approve", "утвердить план"), ("reject", "отклонить план")):
        q = plan_sub.add_parser(name, parents=[common, driving] if name != "show" else [common], help=text)
        q.add_argument("run_id")
        if name == "reject":
            q.add_argument("--reason", help="что исправить (получит модель)")
        q.set_defaults(func=cmd_plan)

    p = sub.add_parser("resume", parents=[common, driving], help="продолжить запуск по run_id")
    p.add_argument("run_id")
    p.set_defaults(func=cmd_resume)

    p = sub.add_parser("recover", parents=[common, driving], help="решение по неопределённому вызову или конфликту")
    p.add_argument("run_id")
    p.add_argument("--action", required=True, choices=("retry", "fail", "abort"))
    p.add_argument("--changes", choices=("attribute", "external"),
                   help="обнаруженные изменения дерева: attribute — сделаны этим запуском, external — посторонние")
    p.set_defaults(func=cmd_recover)

    p = sub.add_parser("abort", parents=[common], help="завершить запуск")
    p.add_argument("run_id")
    p.add_argument("--reason")
    p.set_defaults(func=cmd_abort)

    for name, func, text in (("status", cmd_status, "состояние запуска"), ("tasks", cmd_tasks, "задачи и подтверждения"),
                             ("calls", cmd_calls, "журнал tool calls"), ("rollback", cmd_rollback, "откатить изменения запуска")):
        p = sub.add_parser(name, parents=[common], help=text)
        p.add_argument("run_id")
        p.set_defaults(func=func)

    p = sub.add_parser("diff", parents=[common], help="изменения файлов запуска")
    p.add_argument("run_id")
    p.add_argument("--entries", action="store_true", help="показать также diff каждого действия")
    p.set_defaults(func=cmd_diff)

    p = sub.add_parser("list", parents=[common], help="список запусков")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("policy-check", parents=[common], help="решение policy для одного вызова (без модели)")
    p.add_argument("tool", help="имя инструмента, напр. Bash")
    p.add_argument("arguments", help="JSON аргументов, напр. '{\"command\": \"ls\", \"cwd\": \".\"}'")
    p.add_argument("--mode", choices=MODES, default="safe")
    p.add_argument("--phase", choices=("planning", "execution"), default="execution")
    p.set_defaults(func=cmd_policy_check)
    return parser


def main(argv=None) -> int:
    load_dotenv(Path.cwd() / ".env")
    args = build_parser().parse_args(argv)
    try:
        return args.func(args) or 0
    except (UserError, PolicyError, RunNotFound, RunBusy, RollbackError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    except RollbackConflict as exc:
        print("КОНФЛИКТ: откат не выполнен, ни один файл не перезаписан.", file=sys.stderr)
        for problem in exc.problems:
            print(f"  {problem['path']}: {problem['reason']}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("\nПрервано. Состояние сохранено; продолжить: python3 -m agent resume <run_id>", file=sys.stderr)
        return 130
