from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class StepRecord:
    step: str
    summary: str
    tool: str | None = None
    tool_input: Any = None
    tool_output: Any = None
    status: str = "completed"
    duration_ms: int | None = None
    payload: Any = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class AgentMemory:
    def __init__(self) -> None:
        self._steps: list[StepRecord] = []

    def add(
        self,
        step: str,
        summary: str,
        *,
        tool: str | None = None,
        tool_input: Any = None,
        tool_output: Any = None,
        status: str = "completed",
        duration_ms: int | None = None,
        payload: Any = None,
    ) -> None:
        self._steps.append(
            StepRecord(
                step=step,
                summary=summary,
                tool=tool,
                tool_input=tool_input,
                tool_output=tool_output,
                status=status,
                duration_ms=duration_ms,
                payload=payload,
            )
        )

    def export(self) -> list[dict[str, Any]]:
        return [
            {
                "step": record.step,
                "summary": record.summary,
                "tool": record.tool,
                "tool_input": record.tool_input,
                "tool_output": record.tool_output,
                "status": record.status,
                "duration_ms": record.duration_ms,
                "payload": record.payload,
                "created_at": record.created_at,
            }
            for record in self._steps
        ]
