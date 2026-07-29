"""Run state, persisted to JSONL outside the model.

The model never holds authoritative state. Everything the loop needs to make
a stop/verify decision — which tools ran, what failed, what terminal action
was taken — lives here, and every mutation is appended to a JSONL event log
so a run can be reconstructed without replaying the model.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunState:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    messages: list[dict[str, Any]] = field(default_factory=list)
    tools_called_ok: list[str] = field(default_factory=list)
    tool_errors: list[dict[str, Any]] = field(default_factory=list)
    terminal_action: str | None = None
    final_answer: str | None = None
    iterations: int = 0
    cost_usd: float = 0.0
    status: str | None = None
    jsonl_path: Path | None = None

    def append_message(self, message: dict[str, Any]) -> None:
        self.messages.append(message)
        self.record_event("message", message)

    def record_tool_success(self, tool_name: str) -> None:
        self.tools_called_ok.append(tool_name)
        self.record_event("tool_success", {"tool": tool_name})

    def record_tool_error(self, tool_name: str, error: str) -> None:
        entry = {"tool": tool_name, "error": error}
        self.tool_errors.append(entry)
        self.record_event("tool_error", entry)

    def error_signature_count(self, tool_name: str, error: str) -> int:
        """How many times this exact tool+error pair has occurred."""
        return sum(
            1
            for e in self.tool_errors
            if e["tool"] == tool_name and e["error"] == error
        )

    def record_terminal(self, tool_name: str, answer: str | None) -> None:
        self.terminal_action = tool_name
        self.final_answer = answer
        self.record_event("terminal", {"tool": tool_name, "answer": answer})

    def record_event(self, event_type: str, payload: Any) -> None:
        if self.jsonl_path is None:
            return
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": time.time(),
            "run_id": self.run_id,
            "type": event_type,
            "payload": payload,
        }
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
