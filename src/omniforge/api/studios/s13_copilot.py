# mypy: ignore-errors
"""s13_copilot router."""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter, Header

from omniforge.copilot import OmniForgeCopilotAgent

router = APIRouter(tags=["workbench-s13"])


def _wb() -> Any:
    from omniforge.api import workbench as wb
    return wb


@router.post("/s13/chat")
def s13_chat(req: _wb().CopilotReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _wb()._sid(x_session_id)
    state = _wb().get_session_state(sid)
    copilot = state.get("uap_copilot")
    if copilot is None:
        copilot = OmniForgeCopilotAgent(provider="local")
        state["uap_copilot"] = copilot
    raw_df = state.get("uap_df")
    if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
        analysis_df = _wb().get_analysis_dataframe(state, raw_df)
        copilot.load_dataframe(analysis_df, name=str(state.get("uap_dataset_name") or "active_dataset"))
    reply = copilot.chat(req.message)
    dump = reply.model_dump() if hasattr(reply, "model_dump") else {"content": str(reply)}
    ans = dump.get("answer") or dump.get("content") or str(reply)
    history = state.setdefault("uap_chat_history", [])
    history.append({"role": "user", "text": req.message})
    history.append({"role": "assistant", "text": ans})
    payload = _wb().to_json_object(
        {
            "reply": ans,
            "suggested_actions": dump.get("suggested_actions") or [],
            "tool_calls": dump.get("tool_calls") or [],
            "history": history[-60:],
        }
    )
    _wb().save_studio_result(state, "s13_chat", payload)
    return payload


# ---------------------------------------------------------------------------
# Studio 14 — Live Prediction, Reality-vs-Baseline Tolerance & Supervisory Dashboard
# ---------------------------------------------------------------------------
