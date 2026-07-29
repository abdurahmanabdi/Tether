"""Tool registry: schema validation, retries, errors as values.

A tool call can fail three ways — unknown tool, arguments that violate the
schema, or the tool function raising. All three become a ToolResult with
ok=False that goes back to the agent as an observation. Nothing here raises
into the loop; a crash in a tool is a fact about the world the agent must
react to, not a reason to lose the run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolResult:
    ok: bool
    content: str
    error: str | None = None


@dataclass
class ToolSpec:
    name: str
    description: str
    schema: dict[str, Any]  # JSON schema for the arguments object
    fn: Callable[..., str]
    terminal: bool = False
    answer_arg: str | None = None  # which argument carries the final answer
    retries: int = 0

    def to_schema_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.schema,
        }


def validate_args(schema: dict[str, Any], args: dict[str, Any]) -> str | None:
    """Minimal JSON-schema check: required keys, no unknown keys, basic types.

    Returns an error string, or None if the arguments are valid. Deliberately
    small — the point is that violations are caught *before* execution.
    """
    if not isinstance(args, dict):
        return f"arguments must be an object, got {type(args).__name__}"
    properties: dict[str, Any] = schema.get("properties", {})
    required = schema.get("required", [])
    missing = [k for k in required if k not in args]
    if missing:
        return f"missing required argument(s): {', '.join(sorted(missing))}"
    unknown = [k for k in args if k not in properties]
    if unknown:
        return f"unknown argument(s): {', '.join(sorted(unknown))}"
    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    for key, value in args.items():
        declared = properties.get(key, {}).get("type")
        if declared in type_map and not isinstance(value, type_map[declared]):
            return (
                f"argument {key!r} must be of type {declared}, "
                f"got {type(value).__name__}"
            )
    return None


@dataclass
class ToolRegistry:
    _tools: dict[str, ToolSpec] = field(default_factory=dict)

    def register(
        self,
        name: str,
        description: str,
        schema: dict[str, Any],
        fn: Callable[..., str],
        terminal: bool = False,
        answer_arg: str | None = None,
        retries: int = 0,
    ) -> None:
        if name in self._tools:
            raise ValueError(f"tool {name!r} already registered")
        self._tools[name] = ToolSpec(
            name=name,
            description=description,
            schema=schema,
            fn=fn,
            terminal=terminal,
            answer_arg=answer_arg,
            retries=retries,
        )

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        return [spec.to_schema_dict() for spec in self._tools.values()]

    def execute(self, name: str, args: dict[str, Any]) -> ToolResult:
        spec = self._tools.get(name)
        if spec is None:
            known = ", ".join(sorted(self._tools)) or "(none)"
            return ToolResult(
                ok=False,
                content="",
                error=f"unknown tool {name!r}; available tools: {known}",
            )

        schema_error = validate_args(spec.schema, args)
        if schema_error is not None:
            return ToolResult(
                ok=False,
                content="",
                error=f"invalid arguments for {name!r}: {schema_error}",
            )

        last_error: str | None = None
        for _attempt in range(spec.retries + 1):
            try:
                output = spec.fn(**args)
                return ToolResult(ok=True, content=str(output))
            except Exception as exc:  # errors are values, not crashes
                last_error = f"{type(exc).__name__}: {exc}"
        return ToolResult(
            ok=False,
            content="",
            error=(
                f"tool {name!r} failed after {spec.retries + 1} attempt(s): "
                f"{last_error}"
            ),
        )
