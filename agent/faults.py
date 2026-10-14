"""Управляемые точки отказа для тестов и демонстрации восстановления.

  after_model_charge  — бюджет на запрос к модели списан и сохранён, ответа ещё нет;
  after_tool_start    — «вызов начался» и списание сохранены, инструмент ещё не запущен;
  after_tool_execute  — инструмент ВЫПОЛНЕН, результат ещё НЕ сохранён (главная точка);
  after_result_save   — результат сохранён.

Спецификация: "точка" или "точка:инструмент", например "after_tool_execute:write_file".
В тестах бросается SimulatedCrash; в CLI (--crash-at) процесс по-настоящему умирает через os._exit.
"""
from __future__ import annotations

import os
import sys

POINTS = ("after_model_charge", "after_tool_start", "after_tool_execute", "after_result_save")


class SimulatedCrash(BaseException):
    """BaseException, чтобы сбой не перехватили обработчики ошибок инструментов."""


class Faults:
    def __init__(self, spec: str | None = None, *, hard: bool = False):
        self.point = self.tool = None
        if spec:
            point, _, tool = spec.partition(":")
            if point not in POINTS:
                raise ValueError(f"unknown fault point {point!r}; known: {', '.join(POINTS)}")
            self.point, self.tool = point, tool or None
        self.hard = hard
        self.fired = False

    def hit(self, point: str, tool: str | None = None) -> None:
        if self.fired or point != self.point or (self.tool and self.tool != tool):
            return
        self.fired = True
        if self.hard:
            sys.stdout.flush()
            sys.stderr.write(f"\n!!! имитация сбоя: процесс завершается в точке {point} ({tool})\n")
            sys.stderr.flush()
            os._exit(137)
        raise SimulatedCrash(f"{point}:{tool}")
