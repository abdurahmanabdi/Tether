"""Tool failures are values the agent sees, never crashes the loop eats."""

from __future__ import annotations

from tether.tools import ToolRegistry


def make_registry(fn=None, retries=0):
    registry = ToolRegistry()
    registry.register(
        name="echo",
        description="Echo the message back.",
        schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
        fn=fn or (lambda message: message),
        retries=retries,
    )
    return registry


def test_unknown_tool_is_an_agent_visible_error_not_a_crash():
    registry = make_registry()
    result = registry.execute("does_not_exist", {})
    assert not result.ok
    assert "unknown tool" in result.error
    assert "echo" in result.error  # the error names the tools that do exist


def test_schema_violation_is_caught_before_execution():
    calls = []

    def spy(message):
        calls.append(message)
        return message

    registry = make_registry(fn=spy)
    missing = registry.execute("echo", {})
    wrong_type = registry.execute("echo", {"message": 42})
    unknown_arg = registry.execute("echo", {"message": "hi", "volume": 11})

    assert not missing.ok and "missing required" in missing.error
    assert not wrong_type.ok and "must be of type string" in wrong_type.error
    assert not unknown_arg.ok and "unknown argument" in unknown_arg.error
    assert calls == []  # the tool function never ran


def test_failing_tool_is_retried_then_reported():
    attempts = []

    def flaky(message):
        attempts.append(message)
        raise ConnectionError("kb backend unreachable")

    registry = make_registry(fn=flaky, retries=2)
    result = registry.execute("echo", {"message": "hi"})

    assert len(attempts) == 3  # initial attempt + 2 retries
    assert not result.ok
    assert "after 3 attempt(s)" in result.error
    assert "kb backend unreachable" in result.error


def test_transient_failure_recovers_within_retry_budget():
    state = {"calls": 0}

    def flaky_then_fine(message):
        state["calls"] += 1
        if state["calls"] == 1:
            raise TimeoutError("blip")
        return f"ok: {message}"

    registry = make_registry(fn=flaky_then_fine, retries=1)
    result = registry.execute("echo", {"message": "hi"})
    assert result.ok
    assert result.content == "ok: hi"
