"""Мелкие утилиты: атомарная запись, хеши, усечение текста."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import posixpath
from pathlib import Path


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def dumps(obj) -> str:
    """Сериализация результата инструмента в content сообщения role=tool."""
    return json.dumps(obj, ensure_ascii=False)


def utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


def normalize_rel(path: str) -> str:
    """'./a//b.py' -> 'a/b.py': для сравнения путей в проверках задач."""
    return posixpath.normpath(str(path).replace("\\", "/"))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """tmp-файл + fsync + os.replace: после сбоя на диске либо старая версия, либо новая."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def atomic_write_json(path: Path, obj) -> None:
    atomic_write_bytes(path, json.dumps(obj, ensure_ascii=False, indent=1).encode("utf-8"))


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def truncate_text(text: str, max_bytes: int) -> tuple[str, bool]:
    """Обрезает текст до max_bytes байт UTF-8: оставляет начало и конец и явную пометку."""
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text, False
    if max_bytes <= 0:
        return "", True
    marker_len = len(f"\n...[truncated {len(raw)} bytes]...\n".encode("utf-8"))
    if marker_len >= max_bytes:
        return raw[:max_bytes].decode("utf-8", "ignore"), True
    keep = max_bytes - marker_len
    head = keep // 3
    tail = keep - head
    marker = f"\n...[truncated {len(raw) - keep} bytes]...\n"
    tail_part = raw[len(raw) - tail:] if tail else b""
    return raw[:head].decode("utf-8", "ignore") + marker + tail_part.decode("utf-8", "ignore"), True
