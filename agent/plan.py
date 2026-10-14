"""Plan Mode и задачи: схема плана, его проверка, превращение шагов в задачи
и проверка результата задачи самим runtime (а не по словам модели)."""
from __future__ import annotations

from agent import jsonschema
from agent.policy import ArgumentProblem, PolicyViolation
from agent.util import canonical_json, normalize_rel, sha256_bytes

# Виды проверяемого результата задачи и их обязательные поля.
CHECK_TYPES = {
    "file_read": ("path",),           # в рамках задачи файл успешно прочитан read_file
    "file_changed": ("path",),        # файл отличается от состояния на старте запуска
    "file_unchanged": ("path",),      # файл совпадает с состоянием на старте запуска
    "file_contains": ("path", "text"),
    "command_succeeded": ("command",),  # в рамках задачи Bash-команда завершилась с кодом 0, и после неё файлы не менялись
}

CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": list(CHECK_TYPES)},
        "path": {"type": "string", "minLength": 1},
        "text": {"type": "string", "minLength": 1},
        "command": {"type": "string", "minLength": 1},
    },
    "required": ["type"],
    "additionalProperties": False,
}

STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "minLength": 1, "maxLength": 20, "description": "Task id, e.g. T1"},
        "description": {"type": "string", "minLength": 1},
        "depends_on": {"type": "array", "items": {"type": "string"}, "description": "Ids of tasks that must be done first"},
        "tools": {"type": "array", "items": {"type": "string"}, "minItems": 1, "description": "Tools this step needs"},
        "checks": {"type": "array", "items": CHECK_SCHEMA, "minItems": 1,
                   "description": "Verifiable result of the step, checked by the runtime"},
    },
    "required": ["id", "description", "depends_on", "tools", "checks"],
    "additionalProperties": False,
}

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "goal": {"type": "string", "minLength": 1},
        "steps": {"type": "array", "items": STEP_SCHEMA, "minItems": 1, "maxItems": 20},
        "risks": {"type": "array", "items": {"type": "string"}},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "done_criteria": {"type": "string", "minLength": 1},
    },
    "required": ["goal", "steps", "risks", "assumptions", "done_criteria"],
    "additionalProperties": False,
}


def manifest_digest(manifest: dict) -> str:
    return sha256_bytes(canonical_json(manifest).encode("utf-8"))


def validate_plan(plan: dict, *, tool_names, engine, policy) -> list[str]:
    """Ошибки плана: схема, id, зависимости (существуют, без циклов), инструменты, проверки."""
    problems = jsonschema.validate(plan, PLAN_SCHEMA)
    if problems:
        return problems
    steps = plan["steps"]
    ids = [step["id"] for step in steps]
    for dup in sorted({i for i in ids if ids.count(i) > 1}):
        problems.append(f"duplicate step id {dup!r}")
    known = set(ids)
    for step in steps:
        sid = step["id"]
        for dep in step["depends_on"]:
            if dep == sid:
                problems.append(f"step {sid}: depends on itself")
            elif dep not in known:
                problems.append(f"step {sid}: unknown dependency {dep!r}")
        for tool in step["tools"]:
            if tool not in tool_names:
                problems.append(f"step {sid}: unknown tool {tool!r} (available: {', '.join(sorted(tool_names))})")
        for check in step["checks"]:
            need = CHECK_TYPES[check["type"]]
            missing = [key for key in need if not check.get(key)]
            extra = [key for key in check if key != "type" and key not in need]
            if missing:
                problems.append(f"step {sid}: check {check['type']} needs {', '.join(missing)}")
            if extra:
                problems.append(f"step {sid}: check {check['type']} does not take {', '.join(extra)}")
            try:
                if check.get("path") and "path" in need:
                    engine.resolve_path(policy, check["path"])
                if check["type"] == "command_succeeded" and check.get("command"):
                    engine.check_command(policy, check["command"], ".")
            except (PolicyViolation, ArgumentProblem) as exc:
                problems.append(f"step {sid}: check {check['type']} cannot be verified: {exc}")
    if not problems and _has_cycle(steps):
        problems.append("dependencies contain a cycle")
    return problems


def _has_cycle(steps: list[dict]) -> bool:
    remaining = {step["id"]: set(step["depends_on"]) for step in steps}
    while remaining:
        ready = [sid for sid, deps in remaining.items() if not deps & remaining.keys()]
        if not ready:
            return True
        for sid in ready:
            del remaining[sid]
    return False


def tasks_from_plan(plan: dict) -> list[dict]:
    return [
        {
            "id": step["id"],
            "description": step["description"],
            "depends_on": list(step["depends_on"]),
            "tools": list(step["tools"]),
            "checks": list(step["checks"]),
            "status": "pending",          # pending -> in_progress -> done
            "evidence": None,             # подтверждение результата, зафиксированное runtime
            "summary": None,
            "verify_attempts": 0,
            "started_at": None,
            "completed_at": None,
        }
        for step in plan["steps"]
    ]


def verify_task(task: dict, *, calls: dict, journal, engine, policy) -> tuple[bool, list, list]:
    """Проверяет результат задачи по фактическим данным runtime.

    Источники подтверждения: журнал tool calls этой задачи, хеши файлов (снимок
    дерева сейчас против базового снимка запуска) и содержимое файлов.
    """
    current = journal.scan(policy)
    baseline = journal.baseline_manifest()
    task_calls = [r for r in calls.values() if r.get("task_id") == task["id"] and r["status"] == "completed"]
    evidence, failures = [], []

    for check in task["checks"]:
        kind = check["type"]
        path = normalize_rel(check["path"]) if check.get("path") else None
        if kind == "file_read":
            hits = [r for r in task_calls if r["name"] == "read_file" and (r.get("result") or {}).get("ok")
                    and normalize_rel(r["arguments"]["path"]) == path]
            if hits:
                evidence.append({"check": check, "tool_call_id": hits[-1]["id"]})
            else:
                failures.append({"check": check, "reason": f"no successful read_file of {path} within task {task['id']}"})
        elif kind in ("file_changed", "file_unchanged"):
            before, now = baseline.get(path), current.get(path)
            changed = before != now
            if changed == (kind == "file_changed"):
                evidence.append({"check": check, "baseline_sha256": before, "current_sha256": now})
            else:
                state = "differs from" if changed else "is identical to"
                failures.append({"check": check, "reason": f"{path} {state} its state at run start"})
        elif kind == "file_contains":
            try:
                text = engine.resolve_path(policy, check["path"]).read_text(encoding="utf-8", errors="replace")
            except (PolicyViolation, OSError) as exc:
                failures.append({"check": check, "reason": f"cannot read {path}: {exc}"})
                continue
            if check["text"] in text:
                evidence.append({"check": check, "current_sha256": current.get(path)})
            else:
                failures.append({"check": check, "reason": f"{path} does not contain the expected text"})
        elif kind == "command_succeeded":
            wanted = tuple(check["command"].split())
            digest_now = manifest_digest(current)
            runs = [r for r in task_calls if r["name"] == "Bash" and tuple(r["arguments"]["command"].split()) == wanted]
            good = [r for r in runs if r["result"].get("exit_code") == 0 and not r["result"].get("timed_out")]
            fresh = [r for r in good if r.get("tree_digest_after") == digest_now]
            if fresh:
                evidence.append({"check": check, "tool_call_id": fresh[-1]["id"], "exit_code": 0, "tree_sha256": digest_now})
            elif good:
                failures.append({"check": check, "reason": "the command succeeded earlier, but files changed after it; run it again"})
            elif runs:
                failures.append({"check": check, "reason": f"the last run exited with code {runs[-1]['result'].get('exit_code')}"})
            else:
                failures.append({"check": check, "reason": f"Bash command {check['command']!r} was not run within task {task['id']}"})
    return not failures, evidence, failures


def describe_check(check: dict) -> str:
    kind = check["type"]
    if kind == "command_succeeded":
        return f"command_succeeded `{check.get('command')}`"
    if kind == "file_contains":
        return f"file_contains {check.get('path')} \"{check.get('text')}\""
    return f"{kind} {check.get('path')}"


def render_checklist(tasks: list[dict]) -> str:
    """Задачи в формате чеклиста для stdout."""
    lines = []
    for task in tasks:
        box = "[x]" if task["status"] == "done" else "[ ]"
        notes = []
        if task["depends_on"]:
            notes.append("depends on " + ", ".join(task["depends_on"]))
        if task["status"] == "in_progress":
            notes.append("in progress")
        suffix = f"  ({'; '.join(notes)})" if notes else ""
        lines.append(f"- {box} {task['id']}: {task['description']}{suffix}")
    return "\n".join(lines)
