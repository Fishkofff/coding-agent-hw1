"""Журнал изменений запуска и безопасный откат.

Снимок дерева workspace = {относительный путь: sha256}; содержимое файлов лежит в blobs/.
Список путей в снимке позволяет увидеть не только изменение, но и создание или удаление файлов.

Перед каждым действием, которое может менять файлы (write_file, delete_file, Bash):
  1. текущее дерево сверяется с последним известным состоянием (expected);
     если оно отличается, действие останавливается с конфликтом;
  2. сохраняется снимок «до».
После действия сохраняются снимок «после», фактический diff и хеши.

Для каждого затронутого запуском файла хранится before (до первого изменения этим запуском)
и after (после последнего изменения). Откат восстанавливает before только для этих файлов,
и только если текущий хеш каждого из них равен after; иначе это конфликт, и не пишется ничего.
"""
from __future__ import annotations

import difflib
import os
import re
from fnmatch import fnmatch
from pathlib import Path

from agent.plan import manifest_digest
from agent.policy import Policy, PolicyViolation
from agent.util import now_iso, sha256_bytes

MAX_DIFF_CHARS = 20_000


class RollbackConflict(Exception):
    def __init__(self, problems: list[dict]):
        self.problems = problems
        details = "; ".join(f"{p['path']}: {p['reason']}" for p in problems)
        super().__init__(f"rollback refused, no files were written: {details}")


class RollbackError(Exception):
    pass


def _safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", label)[:60]


class Journal:
    def __init__(self, store, run_id: str, data: dict):
        self.store, self.run_id, self.data = store, run_id, data

    @staticmethod
    def initial_state() -> dict:
        return {
            "baseline": None,       # имя снимка на старте запуска
            "expected": None,       # последнее известное агенту состояние дерева
            "files": {},            # path -> {"before": sha|None, "after": sha|None, ...}
            "entries": [],          # действия, восстановленные после сбоя, внешние изменения, откат
            "snapshot_count": 0,
            "rolled_back": None,
        }

    def scan(self, policy: Policy) -> dict:
        """Текущее дерево (только зона агента: разрешённые и не запрещённые директории,
        без симлинков и явно исключённых генерируемых файлов). Содержимое -> blobs."""
        manifest = {}
        for dirpath, dirnames, filenames in os.walk(policy.workspace, followlinks=False):
            here = Path(dirpath)
            dirnames[:] = sorted(
                name for name in dirnames
                if not (here / name).is_symlink() and not self._excluded(name, policy) and not policy.is_denied(here / name)
            )
            for name in sorted(filenames):
                full = here / name
                if full.is_symlink() or self._excluded(name, policy) or not policy.in_scope(full):
                    continue
                manifest[full.relative_to(policy.workspace).as_posix()] = self.store.put_blob(self.run_id, full.read_bytes())
        return manifest

    @staticmethod
    def _excluded(name: str, policy: Policy) -> bool:
        return any(fnmatch(name, pattern) for pattern in policy.journal_exclude)

    def save_snapshot(self, manifest: dict, label: str) -> str:
        number = self.data["snapshot_count"]
        self.data["snapshot_count"] = number + 1
        name = f"{number:04d}-{_safe_label(label)}"
        self.store.write_snapshot(self.run_id, name, {"label": label, "created_at": now_iso(), "files": manifest})
        return name

    def load_snapshot(self, name: str) -> dict:
        return self.store.read_snapshot(self.run_id, name)["files"]

    def init_baseline(self, policy: Policy) -> None:
        manifest = self.scan(policy)
        self.data["baseline"] = self.save_snapshot(manifest, "baseline")
        self.data["expected"] = manifest

    def baseline_manifest(self) -> dict:
        return self.load_snapshot(self.data["baseline"])

    def expected_digest(self) -> str:
        return manifest_digest(self.data["expected"])

    def compare(self, old: dict, new: dict) -> list[dict]:
        changes = []
        for path in sorted(set(old) | set(new)):
            before, after = old.get(path), new.get(path)
            if before == after:
                continue
            op = "created" if before is None else "deleted" if after is None else "modified"
            changes.append({"path": path, "op": op, "before": before, "after": after,
                            "diff": self.diff_text(path, before, after)})
        return changes

    def detect_drift(self, policy: Policy) -> tuple[list[dict], dict]:
        """Изменения дерева относительно последнего известного состояния (посторонние правки)."""
        current = self.scan(policy)
        return self.compare(self.data["expected"], current), current

    def changes_since(self, policy: Policy, snapshot: str | None) -> list[dict]:
        old = self.load_snapshot(snapshot) if snapshot else self.data["expected"]
        return self.compare(old, self.scan(policy))

    def record(self, policy: Policy, *, kind: str, pre_snapshot: str | None, call_id=None, attempt=None,
               task_id=None, tool=None, note=None) -> dict:
        """Фиксирует фактический результат: снимок «после», diff, хеши.

        kind: action    — действие агента;
              recovered — изменения неопределённого после сбоя вызова, отнесённые пользователем к запуску;
              external  — посторонние изменения, принятые пользователем (не относятся к запуску);
              rollback  — откат.
        """
        pre = self.load_snapshot(pre_snapshot) if pre_snapshot else self.data["expected"]
        post = self.scan(policy)
        changes = self.compare(pre, post)
        seq = len(self.data["entries"]) + 1
        post_name = self.save_snapshot(post, f"post-{call_id or kind}")
        if kind in ("action", "recovered"):
            for change in changes:
                info = self.data["files"].setdefault(change["path"], {"before": change["before"], "first_seq": seq})
                info["after"] = change["after"]
                info["last_seq"] = seq
        entry = {
            "seq": seq, "kind": kind, "call_id": call_id, "attempt": attempt, "task_id": task_id, "tool": tool,
            "pre_snapshot": pre_snapshot, "post_snapshot": post_name, "changes": changes, "note": note, "at": now_iso(),
        }
        self.data["entries"].append(entry)
        self.data["expected"] = post
        return entry

    def _blob_text(self, digest: str | None) -> str | None:
        if digest is None:
            return ""
        data = self.store.get_blob(self.run_id, digest)
        if b"\x00" in data:
            return None
        return data.decode("utf-8", errors="replace")

    def diff_text(self, path: str, before: str | None, after: str | None) -> str:
        old, new = self._blob_text(before), self._blob_text(after)
        if old is None or new is None:
            return f"Binary file {path} changed\n"
        lines = difflib.unified_diff(
            old.splitlines(keepends=True), new.splitlines(keepends=True),
            fromfile=f"a/{path}" if before else "/dev/null", tofile=f"b/{path}" if after else "/dev/null",
        )
        text = "".join(line if line.endswith("\n") else line + "\n" for line in lines)
        return text if len(text) <= MAX_DIFF_CHARS else text[:MAX_DIFF_CHARS] + "\n...[diff truncated]...\n"

    @staticmethod
    def current_sha(path: Path) -> str | None:
        if path.is_symlink():
            return "symlink"
        if not path.exists():
            return None
        if not path.is_file():
            return "not-a-file"
        return sha256_bytes(path.read_bytes())

    def file_report(self, policy: Policy) -> list[dict]:
        report = []
        for path, info in sorted(self.data["files"].items()):
            before, after = info["before"], info["after"]
            current = self.current_sha(policy.workspace / path)
            if before == after:
                op = "unchanged"
            else:
                op = "created" if before is None else "deleted" if after is None else "modified"
            report.append({
                "path": path, "op": op, "before": before, "after": after, "current": current,
                "externally_modified": current != after,
                "diff": self.diff_text(path, before, after),
            })
        return report

    def rollback(self, policy: Policy, engine) -> dict:
        if self.data.get("rolled_back"):
            raise RollbackError(f"run was already rolled back at {self.data['rolled_back']['at']}")
        targets = self.data["files"]
        problems, resolved = [], {}
        for path in sorted(targets):
            try:
                resolved[path] = engine.resolve_path(policy, path)
            except PolicyViolation as exc:
                problems.append({"path": path, "reason": f"policy: {exc}"})
        for path, info in sorted(targets.items()):
            if path not in resolved:
                continue
            current = self.current_sha(resolved[path])
            if current != info["after"]:
                problems.append({"path": path, "reason": "changed outside the agent after its last action",
                                 "expected_sha256": info["after"], "actual_sha256": current})
        if problems:
            raise RollbackConflict(problems)

        pre_name = self.save_snapshot(self.scan(policy), "pre-rollback")
        restored, deleted, untouched = [], [], []
        staged = []
        try:
            for path, info in sorted(targets.items()):
                if info["before"] == info["after"]:
                    untouched.append(path)
                    continue
                if info["before"] is None:
                    continue
                target = resolved[path]
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name(f".{target.name}.rollback-tmp")
                tmp.write_bytes(self.store.get_blob(self.run_id, info["before"]))
                staged.append((tmp, target, path))
        except OSError:
            for tmp, _target, _path in staged:
                tmp.unlink(missing_ok=True)
            raise
        for tmp, target, path in staged:
            os.replace(tmp, target)
            restored.append(path)
        for path, info in sorted(targets.items()):
            if info["before"] is None and info["after"] is not None:
                resolved[path].unlink(missing_ok=True)
                deleted.append(path)
        entry = self.record(policy, kind="rollback", pre_snapshot=pre_name, note="rollback by user command")
        self.data["rolled_back"] = {"at": now_iso(), "entry": entry["seq"]}
        return {"restored": restored, "deleted": deleted, "unchanged": untouched}
