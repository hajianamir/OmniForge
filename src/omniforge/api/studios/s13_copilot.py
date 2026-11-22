"""Studio 13 (Context-Aware AI Analytics Copilot) router."""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter, Header

from omniforge.api.studios.schemas import CopilotReq
from omniforge.copilot import OmniForgeCopilotAgent
from omniforge.core.utils import to_json_object

router = APIRouter(tags=["workbench-s13"])


def _wb() -> Any:
    from omniforge.api import workbench as wb

    return wb


@router.post("/s13/chat")
def s13_chat(req: CopilotReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
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
    dump: dict[str, Any] = reply.model_dump() if hasattr(reply, "model_dump") else {"reply": str(reply)}
    ans = str(dump.get("reply") or dump.get("answer") or dump.get("content") or str(reply))
    suggested_actions = list(dump.get("suggested_next_steps") or dump.get("suggested_actions") or [])
    tool_calls = list(dump.get("tools_executed") or dump.get("tool_calls") or [])
    provider_used = str(dump.get("provider_used") or getattr(copilot, "provider", "local"))
    history = state.setdefault("uap_chat_history", [])
    history.append({"role": "user", "text": req.message})
    history.append({"role": "assistant", "text": ans})
    payload = to_json_object(
        {
            "reply": ans,
            "provider_used": provider_used,
            "suggested_actions": suggested_actions,
            "tool_calls": tool_calls,
            "history": history[-60:],
        }
    )
    _wb().save_studio_result(state, "s13_chat", payload)
    return payload
