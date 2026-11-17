"""Multi-turn conversational, working-state, and episodic memory for the OmniForge AI Copilot."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    """Single user, assistant, or tool message stored in copilot history."""

    role: str  # user, assistant, tool, system
    content: str
    tool_name: str | None = None
    tool_result: dict[str, Any] | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class CopilotMemory:
    """Manages bounded dialogue history plus live workspace state references (datasets, models, reports)."""

    def __init__(self, max_turns: int = 30) -> None:
        self.max_turns = max(4, int(max_turns))
        self.history: list[ConversationTurn] = []
        self.working_state: dict[str, Any] = {
            "datasets": {},
            "models": {},
            "reports": {},
            "active_dataset": None,
            "active_model": None,
            "active_domain": None,
        }

    def add_turn(
        self,
        role: str,
        content: str,
        *,
        tool_name: str | None = None,
        tool_result: dict[str, Any] | None = None,
    ) -> ConversationTurn:
        """Append a conversation turn and enforce the sliding memory window."""
        turn = ConversationTurn(
            role=role, content=content, tool_name=tool_name, tool_result=tool_result
        )
        self.history.append(turn)
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns :]
        return turn

    def set_dataset(self, name: str, df: Any) -> None:
        """Register a dataset in working memory and mark it active."""
        self.working_state["datasets"][name] = df
        self.working_state["active_dataset"] = name

    def get_dataset(self, name: str | None = None) -> tuple[str, Any]:
        """Retrieve a named dataset or the active dataset from working memory."""
        target = name or self.working_state.get("active_dataset")
        if not target or target not in self.working_state["datasets"]:
            raise KeyError("No dataset currently loaded in Copilot memory.")
        return str(target), self.working_state["datasets"][target]

    def set_model(self, name: str, model: Any) -> None:
        """Register a trained model in working memory and mark it active."""
        self.working_state["models"][name] = model
        self.working_state["active_model"] = name

    def get_model(self, name: str | None = None) -> tuple[str, Any]:
        """Retrieve a named model or the active model from working memory."""
        target = name or self.working_state.get("active_model")
        if not target or target not in self.working_state["models"]:
            raise KeyError("No trained model currently loaded in Copilot memory.")
        return str(target), self.working_state["models"][target]

    def to_messages(self) -> list[dict[str, str]]:
        """Format conversation history into standard LLM role/content messages."""
        return [{"role": t.role if t.role in ("user", "assistant", "system") else "assistant", "content": t.content} for t in self.history]

    def clear(self) -> None:
        """Reset dialogue history while preserving loaded workspace objects."""
        self.history.clear()
