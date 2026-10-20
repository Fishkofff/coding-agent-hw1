"""Policy и проверки безопасности. Модуль не зависит от agent loop.

Любой tool call — недоверенный ввод: модель только предлагает действие. Перед КАЖДЫМ
исполнением runtime заново читает policy-файл и проверяет канонические пути, разрешённые
и запрещённые директории, команду целиком и её аргументы, рабочую директорию и режим.
Запрет сильнее разрешения; всё, что явно не разрешено, отклоняется.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

from agent import jsonschema
from agent.budget import BUDGET_KEYS, DEFAULT_LIMITS
from agent.util import utf8_len

MODES = ("safe", "auto")

SHELL_METACHARS = frozenset("|&;<>()$`\\\"'*?[]{}~!#%^")

DEFAULT_JOURNAL_EXCLUDE = ("__pycache__", "*.pyc", ".pytest_cache", ".DS_Store")


class PolicyError(Exception):
    """Policy-файл некорректен — агент не запускается."""


class PolicyViolation(Exception):
    """Действие запрещено policy."""


class ArgumentProblem(Exception):
    """Policy не против, но действие невыполнимо: файла нет, это каталог и т.п."""


def is_within(path: Path, base: Path) -> bool:
    return path == base or base in path.parents


def canonical(base: Path, value: str) -> Path:
    raw = value if os.path.isabs(value) else os.path.join(base, value)
    return Path(os.path.realpath(raw))


_STR_LIST = {"type": "array", "items": {"type": "string", "minLength": 1}}
_RULES = {
    "type": "object",
    "properties": {"allow": _STR_LIST, "deny": _STR_LIST},
    "required": ["allow", "deny"],
    "additionalProperties": False,
}
POLICY_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "workspace": {"type": "string", "minLength": 1},
        "runs_dir": {"type": "string", "minLength": 1},
        "commands": _RULES,
        "directories": _RULES,
        "bash": {
            "type": "object",
            "properties": {
                "timeout_sec": {"type": "number", "minimum": 0.1, "maximum": 3600},
                "max_output_bytes": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": False,
        },
        "files": {
            "type": "object",
            "properties": {
                "max_read_bytes": {"type": "integer", "minimum": 1},
                "max_write_bytes": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": False,
        },
        "journal": {"type": "object", "properties": {"exclude": _STR_LIST}, "additionalProperties": False},
        "budget": {
            "type": "object",
            "properties": {key: {"type": "integer", "minimum": 0} for key in BUDGET_KEYS},
            "additionalProperties": False,
        },
    },
    "required": ["workspace", "commands", "directories"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Policy:
    source: Path
    workspace: Path
    runs_dir: Path
    allow_dirs: tuple
    deny_dirs: tuple
    allow_commands: tuple
    deny_commands: tuple
    bash_timeout_sec: float
    bash_max_output_bytes: int
    max_read_bytes: int
    max_write_bytes: int
    journal_exclude: tuple
    budget: dict

    def is_denied(self, path: Path) -> bool:
        return any(is_within(path, d) for d in self.deny_dirs)

    def is_allowed(self, path: Path) -> bool:
        return any(is_within(path, a) for a in self.allow_dirs)

    def in_scope(self, path: Path) -> bool:
        return self.is_allowed(path) and not self.is_denied(path)

    def rel(self, path: Path) -> str:
        try:
            return path.relative_to(self.workspace).as_posix()
        except ValueError:
            return str(path)


@dataclass(frozen=True)
class CheckedCall:
    """Вызов, прошедший проверки policy: пути уже канонизированы."""

    tool: str
    args: dict
    path: Path | None = None
    rel: str | None = None
    cwd: Path | None = None
    tokens: tuple = ()


def load_policy(path) -> Policy:
    source = Path(os.path.realpath(path))
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise PolicyError(f"policy file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise PolicyError(f"policy file is not valid JSON: {exc}") from None
    problems = jsonschema.validate(data, POLICY_SCHEMA)
    if problems:
        raise PolicyError("invalid policy: " + "; ".join(problems))

    root = source.parent
    workspace = canonical(root, data["workspace"])
    if not workspace.is_dir():
        raise PolicyError(f"workspace {workspace} does not exist")
    if is_within(source, workspace):
        raise PolicyError("the policy file must live outside the workspace")
    runs_dir = canonical(root, data.get("runs_dir", ".agent_runs"))
    if is_within(runs_dir, workspace) or is_within(workspace, runs_dir):
        raise PolicyError("runs_dir must be outside the workspace")

    allow_dirs = tuple(canonical(root, d) for d in data["directories"]["allow"])
    for directory in allow_dirs:
        if not is_within(directory, workspace):
            raise PolicyError(f"allowed directory {directory} is outside the workspace {workspace}")
    deny_dirs = tuple(canonical(root, d) for d in data["directories"]["deny"])

    allow_commands = []
    for command in data["commands"]["allow"]:
        bad = sorted({ch for ch in command if ch in SHELL_METACHARS or ord(ch) < 32})
        if bad:
            raise PolicyError(f"allowed command {command!r} contains shell syntax {bad}; only simple commands can be allowed")
        allow_commands.append(tuple(command.split()))
    deny_commands = tuple(tuple(command.split()) for command in data["commands"]["deny"] if command.split())

    exclude = tuple(data.get("journal", {}).get("exclude", DEFAULT_JOURNAL_EXCLUDE))
    for pattern in exclude:
        _check_exclude_pattern(pattern, workspace)

    bash = data.get("bash", {})
    files = data.get("files", {})
    return Policy(
        source=source,
        workspace=workspace,
        runs_dir=runs_dir,
        allow_dirs=allow_dirs,
        deny_dirs=deny_dirs,
        allow_commands=tuple(allow_commands),
        deny_commands=deny_commands,
        bash_timeout_sec=float(bash.get("timeout_sec", 30)),
        bash_max_output_bytes=int(bash.get("max_output_bytes", 20_000)),
        max_read_bytes=int(files.get("max_read_bytes", 100_000)),
        max_write_bytes=int(files.get("max_write_bytes", 200_000)),
        journal_exclude=exclude,
        budget={**DEFAULT_LIMITS, **data.get("budget", {})},
    )


def _check_exclude_pattern(pattern: str, workspace: Path) -> None:
    """Исключения журнала не должны прятать исходники и тесты контрольного проекта."""
    if "/" in pattern or pattern.strip("*. ") == "":
        raise PolicyError(f"journal.exclude pattern {pattern!r} is too broad (use a file or directory name)")
    for probe in ("order.py", "test_order.py", "module.py", "test_module.py"):
        if fnmatch(probe, pattern):
            raise PolicyError(f"journal.exclude pattern {pattern!r} would hide Python sources or tests")
    for dirpath, _dirs, filenames in os.walk(workspace):
        for name in filenames:
            if not name.endswith(".py"):
                continue
            rel_parts = (Path(dirpath) / name).relative_to(workspace).parts
            if any(fnmatch(part, pattern) for part in rel_parts):
                raise PolicyError(f"journal.exclude pattern {pattern!r} hides {'/'.join(rel_parts)}")


def approval_requirement(effect: str, mode: str, phase: str) -> str:
    """Матрица режимов. auto — исполнить сразу, confirm — нужен ответ пользователя, forbidden — нельзя.

    safe: чтение сразу; запись и Bash после подтверждения; удаление после подтверждения.
    auto: чтение сразу; запись и Bash сразу (если прошли policy); удаление после подтверждения.
    plan mode (фаза planning): только чтение.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    if effect in ("read", "control"):
        return "auto"
    if phase == "planning":
        return "forbidden"
    if effect == "delete":
        return "confirm"
    if effect in ("write", "exec"):
        return "confirm" if mode == "safe" else "auto"
    return "forbidden"


def _contains_sequence(tokens: tuple, rule: tuple) -> bool:
    size = len(rule)
    return any(tokens[i:i + size] == rule for i in range(len(tokens) - size + 1))


class PolicyEngine:
    """Проверки путей, команд и аргументов. Policy перечитывается при каждой проверке."""

    def __init__(self, policy_path):
        self.policy_path = Path(policy_path)

    def load(self) -> Policy:
        return load_policy(self.policy_path)

    def resolve_path(self, policy: Policy, rel) -> Path:
        """Относительный путь агента -> канонический путь внутри разрешённой директории."""
        if not isinstance(rel, str) or not rel.strip():
            raise PolicyViolation("path must be a non-empty string")
        if "\x00" in rel:
            raise PolicyViolation("path contains a NUL byte")
        if os.path.isabs(rel) or rel.startswith("~"):
            raise PolicyViolation(f"absolute paths are not allowed: {rel!r} (use a path relative to the workspace)")
        if ".." in Path(rel).parts:
            raise PolicyViolation(f"'..' is not allowed in paths: {rel!r}")
        lexical = os.path.normpath(os.path.join(policy.workspace, rel))
        real = os.path.realpath(lexical)
        if real != lexical:
            raise PolicyViolation(f"path {rel!r} goes through a symlink; symlinks are not followed")
        path = Path(real)
        self.check_location(policy, path, rel)
        return path

    def check_location(self, policy: Policy, path: Path, shown: str) -> None:
        if path == policy.source or is_within(path, policy.runs_dir):
            raise PolicyViolation(f"{shown!r} is a protected runtime file")
        if not is_within(path, policy.workspace):
            raise PolicyViolation(f"{shown!r} is outside the workspace")
        for denied in policy.deny_dirs:
            if is_within(path, denied):
                raise PolicyViolation(f"{shown!r} is inside the denied directory {policy.rel(denied)!r}")
        if not policy.is_allowed(path):
            raise PolicyViolation(f"{shown!r} is not inside an allowed directory")

    def check_command(self, policy: Policy, command, cwd) -> tuple[tuple, Path]:
        """Команда целиком, её аргументы и рабочая директория."""
        if not isinstance(command, str) or not command.strip():
            raise PolicyViolation("command is empty")
        if any(ord(ch) < 32 or ord(ch) == 127 for ch in command):
            raise PolicyViolation("control characters (newlines, tabs) are not allowed in commands")
        bad = sorted({ch for ch in command if ch in SHELL_METACHARS})
        if bad:
            raise PolicyViolation(
                f"shell syntax is not allowed ({' '.join(bad)}): only exact simple commands from the allowlist can run")
        tokens = tuple(command.split())
        if "=" in tokens[0]:
            raise PolicyViolation("environment variable assignments are not allowed")
        for rule in policy.deny_commands:
            if _contains_sequence(tokens, rule):
                raise PolicyViolation(f"command matches deny rule {' '.join(rule)!r}")
        if tokens not in policy.allow_commands:
            raise PolicyViolation(
                f"command {' '.join(tokens)!r} is not in the allowlist (everything not explicitly allowed is denied)")
        cwd_path = self.resolve_path(policy, cwd)
        if not cwd_path.is_dir():
            raise ArgumentProblem(f"cwd {cwd!r} is not an existing directory")
        for token in tokens:
            value = token.split("=", 1)[1] if token.startswith("-") and "=" in token else token
            if not value or value.startswith("-"):
                continue
            if os.path.isabs(value) or ".." in Path(value).parts:
                raise PolicyViolation(f"argument {token!r} points outside the allowed directories")
            if "/" in value:
                self.resolve_path(policy, os.path.relpath(os.path.join(cwd_path, value), policy.workspace))
        return tokens, cwd_path

    def check_call(self, policy: Policy, tool: str, args: dict) -> CheckedCall:
        if tool == "Bash":
            tokens, cwd = self.check_command(policy, args["command"], args["cwd"])
            return CheckedCall(tool, args, cwd=cwd, rel=policy.rel(cwd), tokens=tokens)

        rel = args.get("path", ".") if tool == "list_files" else args["path"]
        path = self.resolve_path(policy, rel)
        shown = policy.rel(path)
        if tool == "list_files":
            if not path.is_dir():
                raise ArgumentProblem(f"{rel!r} is not an existing directory")
        elif tool == "read_file":
            if not path.exists():
                raise ArgumentProblem(f"file {rel!r} does not exist")
            if not path.is_file():
                raise ArgumentProblem(f"{rel!r} is not a regular file")
        elif tool == "write_file":
            if path.is_dir():
                raise ArgumentProblem(f"{rel!r} is a directory")
            size = utf8_len(args["content"])
            if size > policy.max_write_bytes:
                raise PolicyViolation(f"content is {size} bytes, limit is {policy.max_write_bytes}")
        elif tool == "delete_file":
            if path.exists() and not path.is_file():
                raise ArgumentProblem(f"{rel!r} is not a regular file; only files can be deleted")
        else:
            raise ArgumentProblem(f"tool {tool!r} has no policy checks defined")
        return CheckedCall(tool, args, path=path, rel=shown)
