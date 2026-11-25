"""On-Demand Live Professional Chart Hub Router (Catalog, Single-Chart Generation & SVG/PNG/XLSX Export)."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

from omniforge.reporting.chart_exporter import export_live_chart_artifact
from omniforge.reporting.live_charts import generate_live_chart, get_live_chart_catalog

router = APIRouter(tags=["workbench-charts"])


class LiveChartGenerateReq(BaseModel):
    chart_id: str = Field(..., min_length=1, max_length=120)
    studio_id: str = Field(default="all", max_length=16)
    feature_x: str | None = Field(default=None, max_length=256)
    feature_y: str | None = Field(default=None, max_length=256)
    group_col: str | None = Field(default=None, max_length=256)


class LiveChartExportReq(LiveChartGenerateReq):
    format: Literal["svg", "png", "xlsx"] = "xlsx"
    theme: Literal["light", "dark"] = "light"


def _wb() -> Any:
    from omniforge.api import workbench as wb

    return wb


@router.get("/charts/catalog")
def charts_catalog(
    studio_id: str = Query(default="all", max_length=16),
    x_session_id: str | None = Header(default="default"),
) -> dict[str, Any]:
    """Return the live on-demand professional chart catalog with dynamic workspace readiness."""
    sid = _wb()._sid(x_session_id)
    state = _wb().get_session_state(sid)
    return get_live_chart_catalog(state, studio_id=studio_id)


@router.post("/charts/generate")
def charts_generate_single(
    req: LiveChartGenerateReq,
    x_session_id: str | None = Header(default="default"),
) -> dict[str, Any]:
    """Generate ONLY the single user-selected professional chart from live workspace data & models."""
    sid = _wb()._sid(x_session_id)
    state = _wb().get_session_state(sid)
    try:
        return generate_live_chart(
            state,
            chart_id=req.chart_id,
            studio_id=req.studio_id,
            feature_x=req.feature_x or None,
            feature_y=req.feature_y or None,
            group_col=req.group_col or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/charts/export")
def charts_export_single(
    req: LiveChartExportReq,
    x_session_id: str | None = Header(default="default"),
) -> Response:
    """Compute ONLY the selected live chart and stream it as SVG, PNG, or multi-sheet XLSX."""
    sid = _wb()._sid(x_session_id)
    state = _wb().get_session_state(sid)
    try:
        payload = generate_live_chart(
            state,
            chart_id=req.chart_id,
            studio_id=req.studio_id,
            feature_x=req.feature_x or None,
            feature_y=req.feature_y or None,
            group_col=req.group_col or None,
        )
        file_bytes, media_type, filename = export_live_chart_artifact(
            payload,
            fmt=req.format,
            theme=req.theme,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return Response(
        content=file_bytes,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
