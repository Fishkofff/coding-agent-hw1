"""Минимальный валидатор JSON Schema (подмножество, нужное для аргументов инструментов и policy)."""
from __future__ import annotations

import json

_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def _type_name(value) -> str:
    for name, check in _TYPE_CHECKS.items():
        if name != "number" and check(value):
            return name
    return type(value).__name__


def validate(instance, schema: dict, path: str = "$") -> list[str]:
    """Возвращает список ошибок; пустой список — данные соответствуют схеме."""
    expected = schema.get("type")
    if expected is not None:
        types = expected if isinstance(expected, list) else [expected]
        if not any(_TYPE_CHECKS[t](instance) for t in types):
            return [f"{path}: expected {expected}, got {_type_name(instance)}"]

    errors: list[str] = []
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: must be one of {schema['enum']}")

    if isinstance(instance, str):
        if len(instance) < schema.get("minLength", 0):
            errors.append(f"{path}: string is shorter than {schema['minLength']}")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errors.append(f"{path}: string is longer than {schema['maxLength']}")

    if _TYPE_CHECKS["number"](instance):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: must be >= {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: must be <= {schema['maximum']}")

    if isinstance(instance, dict):
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in instance:
                errors.append(f"{path}: missing required property '{name}'")
        additional = schema.get("additionalProperties", True)
        for key, value in instance.items():
            if key in properties:
                errors.extend(validate(value, properties[key], f"{path}.{key}"))
            elif additional is False:
                errors.append(f"{path}: unexpected property '{key}'")
            elif isinstance(additional, dict):
                errors.extend(validate(value, additional, f"{path}.{key}"))

    if isinstance(instance, list):
        if len(instance) < schema.get("minItems", 0):
            errors.append(f"{path}: must have at least {schema['minItems']} item(s)")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(f"{path}: must have at most {schema['maxItems']} item(s)")
        if isinstance(schema.get("items"), dict):
            for index, item in enumerate(instance):
                errors.extend(validate(item, schema["items"], f"{path}[{index}]"))
    return errors


def coerce_json_strings(value, schema: dict):
    """Небольшие модели иногда присылают вложенный массив/объект строкой с JSON внутри.
    Если схема ждёт array/object, а пришла строка с корректным JSON, — разбираем её."""
    expected = schema.get("type")
    if isinstance(value, str) and expected in ("array", "object"):
        try:
            value = json.loads(value)
        except ValueError:
            return value
    if isinstance(value, dict) and expected == "object":
        properties = schema.get("properties", {})
        return {k: coerce_json_strings(v, properties[k]) if k in properties else v for k, v in value.items()}
    if isinstance(value, list) and expected == "array" and isinstance(schema.get("items"), dict):
        return [coerce_json_strings(v, schema["items"]) for v in value]
    return value
