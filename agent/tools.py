"""Инструменты агента: описание для модели (имя, описание, JSON Schema) и исполнитель.

Исполнитель получает только вызовы, уже прошедшие проверки runtime (CheckedCall).
"""
from __future__ import annotations

import os
import signal
import stat
import subprocess
import threading
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

from agent.plan import PLAN_SCHEMA
from agent.policy import CheckedCall, Policy
from agent.util import truncate_text

PLANNING = "planning"
EXECUTION = "execution"
SIDE_EFFECTS = frozenset({"write", "delete", "exec"})


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict
    effect: str          # read | write | delete | exec | control
    retryable: bool      # можно ли повторить вызов, исход которого неизвестен после сбоя
    phases: frozenset    # в каких фазах запуска инструмент доступен

    @property
    def side_effect(self) -> bool:
        return self.effect in SIDE_EFFECTS

    def to_openai(self) -> dict:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


def _object(properties: dict, required: list) -> dict:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


_PATH = {"type": "string", "minLength": 1, "description": "Path relative to the workspace root, e.g. 'order.py'"}

BASH = ToolSpec(
    name="Bash",
    description=("Run exactly one allowlisted command through bash and return stdout, stderr and exit_code. "
                 "The command must match an allowlisted command exactly; pipes, redirects, ';', '&&', '$()', "
                 "quotes and globs are rejected. Use it to run the tests."),
    parameters=_object({
        "command": {"type": "string", "minLength": 1, "maxLength": 1000,
                    "description": "Exact allowlisted command, e.g. 'python3 -m unittest -v'"},
        "cwd": {"type": "string", "minLength": 1,
                "description": "Working directory relative to the workspace root; '.' is the root"},
    }, ["command", "cwd"]),
    # Не повторяем: произвольная команда может быть неидемпотентной (дописать в файл, сходить в сеть).
    effect="exec", retryable=False, phases=frozenset({EXECUTION}),
)
LIST_FILES = ToolSpec(
    name="list_files",
    description="List files and directories (recursively) inside a workspace directory.",
    parameters=_object({"path": {**_PATH, "description": "Directory relative to the workspace root; '.' is the root"}}, []),
    effect="read", retryable=True, phases=frozenset({PLANNING, EXECUTION}),
)
READ_FILE = ToolSpec(
    name="read_file",
    description="Read a UTF-8 text file from the workspace.",
    parameters=_object({"path": _PATH}, ["path"]),
    effect="read", retryable=True, phases=frozenset({PLANNING, EXECUTION}),
)
WRITE_FILE = ToolSpec(
    name="write_file",
    description="Create or overwrite a text file in the workspace with its FULL new content.",
    parameters=_object({"path": _PATH, "content": {"type": "string", "description": "Full new file content"}},
                       ["path", "content"]),
    # Идемпотентен: повтор с тем же content даёт то же итоговое состояние файла.
    effect="write", retryable=True, phases=frozenset({EXECUTION}),
)
DELETE_FILE = ToolSpec(
    name="delete_file",
    description="Delete a file in the workspace. Always requires user confirmation.",
    parameters=_object({"path": _PATH}, ["path"]),
    # Идемпотентен: «файла нет» — одинаковый итог после первой и повторной попытки.
    effect="delete", retryable=True, phases=frozenset({EXECUTION}),
)
SUBMIT_PLAN = ToolSpec(
    name="submit_plan",
    description=("PLAN MODE only. Submit the structured plan for user approval: goal, ordered steps "
                 "(id, description, depends_on, tools, checks), risks, assumptions and done_criteria."),
    parameters=PLAN_SCHEMA,
    effect="control", retryable=True, phases=frozenset({PLANNING}),
)
COMPLETE_TASK = ToolSpec(
    name="complete_task",
    description=("Ask the runtime to mark the current task as done. The runtime verifies the task's checks "
                 "itself; a task is never marked done just because you say so."),
    parameters=_object({
        "task_id": {"type": "string", "minLength": 1},
        "summary": {"type": "string", "description": "What was done"},
    }, ["task_id"]),
    effect="control", retryable=True, phases=frozenset({EXECUTION}),
)

REGISTRY = {spec.name: spec for spec in (BASH, LIST_FILES, READ_FILE, WRITE_FILE, DELETE_FILE, SUBMIT_PLAN, COMPLETE_TASK)}
WORK_TOOL_NAMES = frozenset({"Bash", "list_files", "read_file", "write_file", "delete_file", "complete_task"})


def offered_tools(phase: str, has_tasks: bool) -> list[ToolSpec]:
    """Какие инструменты показать модели. Это подсказка; разрешает или запрещает всё равно runtime."""
    if phase == PLANNING:
        return [LIST_FILES, READ_FILE, SUBMIT_PLAN]
    tools = [BASH, LIST_FILES, READ_FILE, WRITE_FILE, DELETE_FILE]
    if has_tasks:
        tools.append(COMPLETE_TASK)
    return tools


# ---------------------------------------------------------------------------------- исполнитель
class ToolExecutor:
    def __init__(self):
        self.executions: list[tuple[str, dict]] = []   # что реально исполнялось (для тестов и отладки)

    def execute(self, call: CheckedCall, policy: Policy, allowance: int) -> dict:
        """allowance — доступный остаток result_bytes: инструмент не собирает вывод сверх него."""
        self.executions.append((call.tool, dict(call.args)))
        handlers = {
            "Bash": self._bash,
            "list_files": self._list_files,
            "read_file": self._read_file,
            "write_file": self._write_file,
            "delete_file": self._delete_file,
        }
        try:
            return handlers[call.tool](call, policy, allowance)
        except OSError as exc:
            return {"ok": False, "error": "io_error", "message": f"{type(exc).__name__}: {exc}"}

    @staticmethod
    def _list_files(call: CheckedCall, policy: Policy, allowance: int) -> dict:
        entries = []
        for dirpath, dirnames, filenames in os.walk(call.path, followlinks=False):
            here = Path(dirpath)
            visible = []
            for name in sorted(dirnames):
                full = here / name
                if full.is_symlink():
                    entries.append(policy.rel(full) + " (symlink, not followed)")
                elif not policy.is_denied(full) and not _excluded(name, policy):
                    visible.append(name)
                    entries.append(policy.rel(full) + "/")
            dirnames[:] = visible
            for name in sorted(filenames):
                if _excluded(name, policy):
                    continue
                full = here / name
                entries.append(policy.rel(full) + (" (symlink, not followed)" if full.is_symlink() else ""))
            if len(entries) > 2000:
                break
        text, cut = truncate_text("\n".join(sorted(entries)) or "(empty)", allowance)
        return {"ok": True, "path": call.rel, "entries": text, "truncated": cut}

    @staticmethod
    def _read_file(call: CheckedCall, policy: Policy, allowance: int) -> dict:
        fd = os.open(call.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as fh:
            raw = fh.read(policy.max_read_bytes + 1)
        size = call.path.stat().st_size
        cut = len(raw) > policy.max_read_bytes
        raw = raw[:policy.max_read_bytes]
        if b"\x00" in raw:
            return {"ok": False, "error": "binary_file", "message": f"{call.rel} looks binary ({size} bytes)"}
        text, cut_by_allowance = truncate_text(raw.decode("utf-8", errors="replace"), allowance)
        return {"ok": True, "path": call.rel, "size": size, "content": text, "truncated": cut or cut_by_allowance}

    @staticmethod
    def _write_file(call: CheckedCall, policy: Policy, allowance: int) -> dict:
        data = call.args["content"].encode("utf-8")
        target = call.path
        existed = target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.agent-tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0), 0o644)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        if existed:
            os.chmod(tmp, stat.S_IMODE(target.stat().st_mode))
        os.replace(tmp, target)
        return {"ok": True, "path": call.rel, "bytes_written": len(data), "created": not existed}

    @staticmethod
    def _delete_file(call: CheckedCall, policy: Policy, allowance: int) -> dict:
        if not call.path.exists():
            return {"ok": True, "path": call.rel, "deleted": False, "message": "file was already absent"}
        call.path.unlink()
        return {"ok": True, "path": call.rel, "deleted": True}

    @staticmethod
    def _bash(call: CheckedCall, policy: Policy, allowance: int) -> dict:
        cap = max(0, min(policy.bash_max_output_bytes, allowance))
        result = run_bash(call.args["command"], call.cwd, policy.bash_timeout_sec, cap)
        result["cwd"] = call.rel
        return result


def _excluded(name: str, policy: Policy) -> bool:
    return any(fnmatch(name, pattern) for pattern in policy.journal_exclude)


# ---------------------------------------------------------------------------------- Bash
def _bash_env() -> dict:
    """Минимальное окружение: секреты агента (OPENAI_API_KEY и т.п.) в дочерний процесс не попадают."""
    return {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "TERM": "dumb",
        "PYTHONDONTWRITEBYTECODE": "1",   # тесты не создают __pycache__ в workspace
        "PYTHONIOENCODING": "utf-8",
    }


class _Capture(threading.Thread):
    """Читает поток до конца, но хранит не больше cap байт."""

    def __init__(self, stream, cap: int):
        super().__init__(daemon=True)
        self.stream, self.cap = stream, cap
        self.buffer = bytearray()
        self.total = 0

    def run(self) -> None:
        try:
            while True:
                chunk = self.stream.read1(65536)
                if not chunk:
                    break
                self.total += len(chunk)
                room = self.cap - len(self.buffer)
                if room > 0:
                    self.buffer += chunk[:room]
        except (OSError, ValueError):
            pass


def _split(a: int, b: int, cap: int) -> tuple[int, int]:
    if a + b <= cap:
        return a, b
    half = cap // 2
    if a <= half:
        return a, cap - a
    if b <= half:
        return cap - b, b
    return half, cap - half


def run_bash(command: str, cwd: Path, timeout: float, cap: int) -> dict:
    """Запускает уже проверенную простую команду через bash с таймаутом и лимитом вывода."""
    proc = subprocess.Popen(
        ["bash", "--noprofile", "--norc", "-c", command],
        cwd=str(cwd), env=_bash_env(), stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    out, err = _Capture(proc.stdout, cap), _Capture(proc.stderr, cap)
    out.start()
    err.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)   # убиваем всю группу процессов
        except ProcessLookupError:
            pass
        proc.wait()
    out.join(2)
    err.join(2)
    for stream in (proc.stdout, proc.stderr):
        try:
            stream.close()
        except OSError:
            pass
    stdout = out.buffer.decode("utf-8", errors="replace")
    stderr = err.buffer.decode("utf-8", errors="replace")
    keep_out, keep_err = _split(len(out.buffer), len(err.buffer), cap)
    stdout, cut_out = truncate_text(stdout, keep_out)
    stderr, cut_err = truncate_text(stderr, keep_err)
    truncated = cut_out or cut_err or out.total > len(out.buffer) or err.total > len(err.buffer)
    return {
        "ok": proc.returncode == 0 and not timed_out,
        "command": command,
        "exit_code": proc.returncode,
        "timed_out": timed_out,
        "stdout": stdout,
        "stderr": stderr,
        "truncated": truncated,
        "output_bytes_total": out.total + err.total,
    }
