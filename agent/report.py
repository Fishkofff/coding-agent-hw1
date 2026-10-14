"""Вывод в консоль: ход работы агента, план, чеклист задач и подсказки следующих команд."""
from __future__ import annotations

import json
import sys

from agent.budget import Budget
from agent.plan import describe_check, render_checklist

CLI = "python3 -m agent"


def shorten(text, limit: int) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + f"... (+{len(text) - limit} chars)"


def format_args(record: dict) -> str:
    args = record.get("arguments")
    if args is None:
        return shorten(record.get("arguments_raw") or "", 200)
    parts = []
    for key, value in args.items():
        if isinstance(value, str) and (len(value) > 80 or "\n" in value):
            parts.append(f"{key}=<{len(value)} chars>")
        elif isinstance(value, (list, dict)):
            parts.append(f"{key}=<{type(value).__name__}>")
        else:
            parts.append(f"{key}={json.dumps(value, ensure_ascii=False)}")
    return ", ".join(parts)


def summarize_result(record: dict) -> str:
    result = record.get("result") or {}
    name = record["name"]
    if result.get("ok") is False and result.get("error"):
        text = f"x {record['status']}: {result['error']}: {shorten(result.get('message', ''), 400)}"
        if result.get("reason"):
            text += f" (reason: {result['reason']})"
        for item in result.get("problems", []) + result.get("failed_checks", []):
            text += f"\n      - {item if isinstance(item, str) else item.get('reason')}"
        return text
    if name == "Bash":
        flags = (" TIMEOUT" if result.get("timed_out") else "") + (" [output truncated]" if result.get("truncated") else "")
        output = ((result.get("stdout") or "") + (result.get("stderr") or "")).strip().splitlines()
        tail = "\n".join("      | " + line for line in output[-12:])
        return f"<- exit_code={result.get('exit_code')}{flags}" + ("\n" + tail if tail else "")
    if name == "read_file":
        return f"<- {result.get('size')} bytes" + (" [truncated]" if result.get("truncated") else "")
    if name == "list_files":
        return f"<- {len((result.get('entries') or '').splitlines())} entries"
    if name == "write_file":
        return f"<- wrote {result.get('bytes_written')} bytes" + (" (new file)" if result.get("created") else "")
    if name == "delete_file":
        return "<- deleted" if result.get("deleted") else "<- file was already absent"
    if name == "complete_task":
        return f"<- task {result.get('task_id')} verified by the runtime"
    if name == "submit_plan":
        return "<- plan approved by the user"
    return "<- ok"


def format_plan(plan: dict) -> str:
    lines = [f"План v{plan.get('version')} (статус: {plan.get('status')})", f"Цель: {plan['goal']}", "Шаги:"]
    for step in plan["steps"]:
        lines.append(f"  {step['id']}: {step['description']}")
        lines.append(f"      зависит от: {', '.join(step['depends_on']) or '-'}")
        lines.append(f"      инструменты: {', '.join(step['tools'])}")
        lines.append(f"      проверки: {'; '.join(describe_check(c) for c in step['checks'])}")
    lines.append("Риски:")
    lines.extend(f"  - {item}" for item in plan["risks"] or ["-"])
    lines.append("Допущения:")
    lines.extend(f"  - {item}" for item in plan["assumptions"] or ["-"])
    lines.append(f"Критерий готовности: {plan['done_criteria']}")
    return "\n".join(lines)


def format_changes(changes: list[dict], with_diff: bool = True) -> str:
    if not changes:
        return "  (изменений нет)"
    marks = {"created": "A", "modified": "M", "deleted": "D", "unchanged": "="}
    lines = []
    for change in changes:
        lines.append(f"  {marks.get(change['op'], '?')} {change['path']}")
        if with_diff and change.get("diff"):
            lines.extend("      " + line for line in change["diff"].rstrip("\n").splitlines())
    return "\n".join(lines)


class Reporter:
    def __init__(self, stream=None):
        self.stream = stream if stream is not None else sys.stdout

    def line(self, text: str = "") -> None:
        print(text, file=self.stream, flush=True)

    def run_started(self, state: dict) -> None:
        self.line(f"run_id: {state['run_id']}  (mode: {state['mode']}, plan mode: {'on' if state['plan_mode'] else 'off'})")

    def assistant(self, text: str) -> None:
        if text and text.strip():
            self.line("model: " + shorten(text.strip(), 1500))

    def tool_call(self, record: dict) -> None:
        where = f" [task {record['task_id']}]" if record.get("task_id") else (" [planning]" if record["phase"] == "planning" else "")
        self.line(f"-> {record['name']}({format_args(record)}){where}  id={record['id']}")

    def tool_outcome(self, record: dict) -> None:
        self.line("   " + summarize_result(record))

    def checklist(self, tasks: list[dict], title: str = "Tasks") -> None:
        self.line(f"{title}:")
        self.line(render_checklist(tasks))

    def plan(self, plan: dict) -> None:
        self.line(format_plan(plan))

    def stopped(self, state: dict, details: str | None = None) -> None:
        run_id, status = state["run_id"], state["status"]
        pending = state.get("pending") or {}
        title = f"{status} ({pending['kind']})" if status == "waiting" else status
        self.line(f"\n=== run {run_id}: {title} ===")
        if details:
            self.line(details)
        if status == "waiting":
            kind = pending["kind"]
            if kind == "confirmation":
                self.line(f"Разрешить:  {CLI} approve {run_id}")
                self.line(f"Отклонить:  {CLI} deny {run_id} --reason \"...\"")
            elif kind == "plan_approval":
                self.line(f"Утвердить план: {CLI} plan approve {run_id}")
                self.line(f"Отклонить план: {CLI} plan reject {run_id} --reason \"что исправить\"")
            elif kind == "recovery":
                self.line(f"Решение:   {CLI} recover {run_id} --action retry|fail|abort"
                          + (" --changes attribute|external" if pending.get("changes") else ""))
            elif kind == "conflict":
                self.line(f"Решение:   {CLI} recover {run_id} --action retry|fail|abort --changes external")
        elif status == "interrupted":
            self.line(f"Причина: {(state.get('error') or {}).get('message')}")
            self.line(f"Продолжить: {CLI} resume {run_id}")
        elif status == "completed":
            self.line(f"Ответ агента: {state.get('final_answer')}")
            self.line(f"Изменения: {CLI} diff {run_id}    Откат: {CLI} rollback {run_id}")
        elif status == "failed":
            error = state.get("error") or {}
            self.line(f"Ошибка: {error.get('code')}: {error.get('message')}")
            self.line(f"Изменения: {CLI} diff {run_id}    Откат: {CLI} rollback {run_id}")
        self.line("Бюджет: " + Budget(state["budget"]).summary())
