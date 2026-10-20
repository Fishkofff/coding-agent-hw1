"""Хранилище запусков: .agent_runs/<run_id>/

  state.json      — полное состояние запуска (сообщения, tool calls, решения, бюджет, план,
                    задачи, журнал). Пишется атомарно: tmp + fsync + os.replace.
  blobs/<sha256>  — содержимое файлов для снимков журнала (content-addressed).
  snapshots/*.json— снимки дерева workspace: путь -> sha256.
  events.jsonl    — человекочитаемый аудит-лог (источник истины — state.json).
  lock            — flock: один процесс на запуск; после падения процесса блокировка снимается ОС.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import secrets
from contextlib import contextmanager
from pathlib import Path

from agent.util import atomic_write_bytes, atomic_write_json, now_iso, sha256_bytes

try:
    import fcntl
except ImportError:
    fcntl = None

RUN_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")


class RunNotFound(Exception):
    pass


class RunBusy(Exception):
    pass


class RunStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    @staticmethod
    def new_run_id() -> str:
        return _dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)

    def run_dir(self, run_id: str) -> Path:
        if not isinstance(run_id, str) or not RUN_ID_RE.match(run_id):
            raise RunNotFound(f"invalid run_id {run_id!r}")
        return self.root / run_id

    def create_run_dir(self, run_id: str) -> None:
        self.run_dir(run_id).mkdir(parents=True, exist_ok=False)

    def save(self, state: dict) -> None:
        atomic_write_json(self.run_dir(state["run_id"]) / "state.json", state)

    def load(self, run_id: str) -> dict:
        path = self.run_dir(run_id) / "state.json"
        if not path.exists():
            raise RunNotFound(f"run {run_id} not found in {self.root}")
        return json.loads(path.read_text(encoding="utf-8"))

    def list_runs(self) -> list[dict]:
        if not self.root.exists():
            return []
        runs = []
        for directory in sorted(self.root.iterdir()):
            if RUN_ID_RE.match(directory.name) and (directory / "state.json").exists():
                runs.append(self.load(directory.name))
        return runs

    def put_blob(self, run_id: str, data: bytes) -> str:
        digest = sha256_bytes(data)
        path = self.run_dir(run_id) / "blobs" / digest
        if not path.exists():
            atomic_write_bytes(path, data)
        return digest

    def get_blob(self, run_id: str, digest: str) -> bytes:
        return (self.run_dir(run_id) / "blobs" / digest).read_bytes()

    def write_snapshot(self, run_id: str, name: str, payload: dict) -> None:
        atomic_write_json(self.run_dir(run_id) / "snapshots" / f"{name}.json", payload)

    def read_snapshot(self, run_id: str, name: str) -> dict:
        return json.loads((self.run_dir(run_id) / "snapshots" / f"{name}.json").read_text(encoding="utf-8"))

    def append_event(self, run_id: str, event_name: str, **data) -> None:
        line = json.dumps({"at": now_iso(), "event": event_name, **data}, ensure_ascii=False, default=str)
        with open(self.run_dir(run_id) / "events.jsonl", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def read_events(self, run_id: str) -> list[dict]:
        path = self.run_dir(run_id) / "events.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    @contextmanager
    def lock(self, run_id: str):
        directory = self.run_dir(run_id)
        if not directory.exists():
            raise RunNotFound(f"run {run_id} not found in {self.root}")
        fd = os.open(directory / "lock", os.O_CREAT | os.O_RDWR, 0o644)
        try:
            if fcntl is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise RunBusy(f"run {run_id} is being executed by another process") from None
            yield
        finally:
            if fcntl is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
            os.close(fd)
