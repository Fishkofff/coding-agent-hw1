"""Единый бюджет запуска.

Счётчики живут внутри state.json запуска, поэтому проверка лимита, списание и статус
действия попадают на диск одной атомарной записью. resume, утверждение плана и повтор
после сбоя ничего не обнуляют: счётчики просто читаются из того же state.json.
"""
from __future__ import annotations

from agent.util import dumps, truncate_text, utf8_len

BUDGET_KEYS = ("model_calls", "tool_calls", "side_effects", "result_bytes")
DEFAULT_LIMITS = {"model_calls": 40, "tool_calls": 80, "side_effects": 20, "result_bytes": 300_000}

MIN_RESULT_ALLOWANCE = 256

TEXT_FIELDS = ("stdout", "stderr", "content", "entries", "message")


class Budget:
    """Обёртка над state["budget"] = {"limits": {...}, "used": {...}}."""

    def __init__(self, data: dict):
        self.data = data

    @staticmethod
    def initial(limits: dict) -> dict:
        return {
            "limits": {key: int(limits.get(key, DEFAULT_LIMITS[key])) for key in BUDGET_KEYS},
            "used": {key: 0 for key in BUDGET_KEYS},
        }

    def limit(self, key: str) -> int:
        return self.data["limits"][key]

    def used(self, key: str) -> int:
        return self.data["used"][key]

    def remaining(self, key: str) -> int:
        return max(0, self.limit(key) - self.used(key))

    def shortfall(self, **need: int) -> list[str]:
        """Каких лимитов не хватает для действия (пустой список — хватает всех)."""
        problems = []
        for key, amount in need.items():
            if amount and amount > self.remaining(key):
                problems.append(f"{key}: need {amount}, remaining {self.remaining(key)} of {self.limit(key)}")
        return problems

    def charge(self, key: str, amount: int) -> None:
        if amount < 0 or amount > self.remaining(key):
            raise ValueError(f"budget overcharge for {key}: {amount} > {self.remaining(key)}")
        self.data["used"][key] += amount

    def summary(self) -> str:
        return " | ".join(f"{key} {self.used(key)}/{self.limit(key)}" for key in BUDGET_KEYS)


def fit_result(data: dict, limit: int) -> tuple[dict, int]:
    """Подгоняет результат инструмента под остаток result_bytes.

    Размер результата — байты UTF-8 строки, которая уйдёт модели в сообщении role=tool.
    Если не помещается, обрезаются текстовые поля, а результат помечается
    truncated=true и truncated_reason="result_bytes budget".
    """
    size = utf8_len(dumps(data))
    if size <= limit:
        return data, size
    data = dict(data)
    for _ in range(8):
        fields = [k for k in TEXT_FIELDS if isinstance(data.get(k), str) and data[k]]
        if not fields:
            break
        key = max(fields, key=lambda k: utf8_len(data[k]))
        overflow = size - limit
        data[key], _ = truncate_text(data[key], max(0, utf8_len(data[key]) - overflow - 64))
        data["truncated"] = True
        data["truncated_reason"] = "result_bytes budget"
        size = utf8_len(dumps(data))
        if size <= limit:
            return data, size
    minimal = {
        "ok": bool(data.get("ok")),
        "truncated": True,
        "truncated_reason": "result_bytes budget",
        "message": "output dropped: result budget exhausted",
    }
    return minimal, utf8_len(dumps(minimal))
