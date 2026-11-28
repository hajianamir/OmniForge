"""Shared sample disclosure for the Studio 04 Shapiro-Wilk surfaces."""

from __future__ import annotations

from typing import Literal, TypedDict

SHAPIRO_CAP = 5000


class ShapiroSampleScope(TypedDict):
    source_rows: int
    available_rows: int
    tested_rows: int
    omitted_rows: int
    cap: int
    cap_omitted_rows: int
    cap_reason: str
    selection: Literal["first_non_missing_in_study_order", "evenly_spaced_sorted_ranks"]


def shapiro_sample_disclosure(
    *,
    source_rows: int,
    available_rows: int,
    tested_rows: int,
    cap_omitted_rows: int,
    selection: Literal["first_non_missing_in_study_order", "evenly_spaced_sorted_ranks"],
) -> tuple[ShapiroSampleScope, str]:
    """Describe the input, eligible observations and test separately from any plot thinning."""
    omitted = available_rows - tested_rows
    reason = "Shapiro-Wilk p-value accuracy"
    scope = ShapiroSampleScope(
        source_rows=source_rows,
        available_rows=available_rows,
        tested_rows=tested_rows,
        omitted_rows=omitted,
        cap=SHAPIRO_CAP,
        cap_omitted_rows=cap_omitted_rows,
        cap_reason=reason,
        selection=selection,
    )
    note = f"{available_rows:,} available / {tested_rows:,} tested / {omitted:,} omitted; cap {SHAPIRO_CAP:,} (p-value accuracy)"
    if source_rows != available_rows:
        note += f"; {source_rows:,} source rows"
    return scope, note
