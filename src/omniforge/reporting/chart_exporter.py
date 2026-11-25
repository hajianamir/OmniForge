"""Multi-Format Live Chart Exporter (SVG, PNG, and Native + Visual XLSX Workbooks)."""

from __future__ import annotations

import io
import math
import re
from datetime import UTC, datetime
from typing import Any

import numpy as np
import openpyxl
from openpyxl.chart import BarChart, DoughnutChart, LineChart, RadarChart, Reference, ScatterChart, Series
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from omniforge.reporting.charts import plt
from omniforge.reporting.exporters import sanitize_spreadsheet_value

_PALETTE = [
    "#0284c7",
    "#7c3aed",
    "#059669",
    "#d97706",
    "#dc2626",
    "#db2777",
    "#0d9488",
    "#4f46e5",
    "#ea580c",
    "#9333ea",
]


def _safe_filename(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(raw or "chart")).strip("_")
    return cleaned[:72] or "omniforge_chart"


def _excel_cell_val(val: Any) -> Any:
    if val is None:
        return ""
    if isinstance(val, bool):
        return "TRUE" if val else "FALSE"
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return round(val, 6) if math.isfinite(val) else ""
    return sanitize_spreadsheet_value(str(val))


def render_chart_matplotlib(
    chart_payload: dict[str, Any],
    *,
    fmt: str = "png",
    theme: str = "light",
    dpi: int = 200,
) -> bytes:
    """Render any of the 25 professional chart payloads to vector SVG or high-DPI PNG bytes."""
    is_dark = str(theme).lower() == "dark"
    bg_fig = "#0b1120" if is_dark else "#ffffff"
    bg_ax = "#111b2e" if is_dark else "#f8fafc"
    fg = "#edf2ff" if is_dark else "#0f172a"
    muted = "#94a3b8" if is_dark else "#475569"
    grid_col = "#22314f" if is_dark else "#cbd5e1"

    ctype = str(chart_payload.get("chart_type") or "")
    title = str(chart_payload.get("title") or "OmniForge Analytical Chart")
    subtitle = str(chart_payload.get("subtitle") or "")
    data = chart_payload.get("data") or {}

    if ctype == "radar":
        fig = plt.figure(figsize=(8.6, 5.6), facecolor=bg_fig)
        ax = fig.add_subplot(111, polar=True, facecolor=bg_ax)
    else:
        fig, ax = plt.subplots(figsize=(9.2, 5.2), facecolor=bg_fig)
        ax.set_facecolor(bg_ax)

    if ctype != "radar":
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(grid_col)
        ax.spines["bottom"].set_color(grid_col)
        ax.tick_params(colors=muted, labelsize=8.5)
        ax.grid(True, linestyle="--", alpha=0.45, color=grid_col)
        ax.set_axisbelow(True)

    # Dispatch by chart_type
    if ctype == "gauge":
        ax.grid(False)
        ax.axis("off")
        val = float(data.get("value", 0.0))
        mn = float(data.get("min", 0.0))
        mx = max(mn + 1e-6, float(data.get("max", 100.0)))
        frac = max(0.0, min(1.0, (val - mn) / (mx - mn)))
        theta_bg = np.linspace(np.pi, 0.0, 120)
        theta_val = np.linspace(np.pi, np.pi * (1.0 - frac), max(4, int(120 * frac)))
        ax.plot(np.cos(theta_bg), np.sin(theta_bg), color=grid_col, linewidth=20, solid_capstyle="round")
        arc_col = "#10b981" if frac >= 0.85 else "#0284c7" if frac >= 0.6 else "#f59e0b"
        ax.plot(np.cos(theta_val), np.sin(theta_val), color=arc_col, linewidth=20, solid_capstyle="round")
        for thr in data.get("thresholds") or []:
            tf = max(0.0, min(1.0, (float(thr.get("value", 0.0)) - mn) / (mx - mn)))
            ang = np.pi * (1.0 - tf)
            ax.plot(
                [0.86 * math.cos(ang), 1.14 * math.cos(ang)],
                [0.86 * math.sin(ang), 1.14 * math.sin(ang)],
                color=str(thr.get("color") or "#f59e0b"),
                linewidth=2.5,
            )
        ax.text(0.0, 0.30, f"{val:.1f} {data.get('unit', '')}".strip(), ha="center", va="center", fontsize=24, fontweight="bold", color=fg)
        ax.text(0.0, 0.06, str(data.get("status_label") or ""), ha="center", va="center", fontsize=11, color=muted)
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-0.15, 1.28)

    elif ctype == "radar":
        axes_lbls = [str(a) for a in (data.get("axes") or [])]
        n_ax = max(1, len(axes_lbls))
        angles = np.linspace(0, 2 * np.pi, n_ax, endpoint=False).tolist()
        angles_closed = [*angles, angles[0]]
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles)
        ax.set_xticklabels(axes_lbls, color=fg, fontsize=8.5)
        ax.tick_params(colors=muted, labelsize=7.5)
        ax.spines["polar"].set_color(grid_col)
        ax.grid(color=grid_col, linestyle="--", alpha=0.55)
        max_sc = float(data.get("max_scale", 100.0))
        ax.set_ylim(0, max_sc)
        for idx, s in enumerate(data.get("series") or []):
            vals = [float(v) for v in (s.get("values") or [])]
            if len(vals) == n_ax:
                v_closed = [*vals, vals[0]]
                col = str(s.get("color") or _PALETTE[idx % len(_PALETTE)])
                ax.plot(angles_closed, v_closed, linewidth=2.2, color=col, label=str(s.get("name") or f"Series {idx + 1}"))
                if idx == 0:
                    ax.fill(angles_closed, v_closed, color=col, alpha=0.20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.10), frameon=False, labelcolor=fg, fontsize=8.5)

    elif ctype == "donut":
        ax.grid(False)
        slices = data.get("slices") or []
        vals = [max(0.0, float(s.get("value", 0.0))) for s in slices]
        lbls = [f"{s.get('label', '')} ({s.get('pct', 0)}%)" for s in slices]
        cols = [str(s.get("color") or _PALETTE[i % len(_PALETTE)]) for i, s in enumerate(slices)]
        if sum(vals) > 0:
            wedges, _ = ax.pie(
                vals,
                colors=cols,
                startangle=90,
                wedgeprops={"width": 0.42, "edgecolor": bg_fig, "linewidth": 1.5},
            )
            ax.legend(wedges, lbls, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, labelcolor=fg, fontsize=8.5)
            ax.text(0, 0.05, str(data.get("center_value") or ""), ha="center", va="center", fontsize=15, fontweight="bold", color=fg)
            ax.text(0, -0.14, str(data.get("center_label") or ""), ha="center", va="center", fontsize=8.5, color=muted)

    elif ctype == "waterfall":
        bars = data.get("bars") or []
        xs = np.arange(len(bars))
        for i, b in enumerate(bars):
            kind = str(b.get("kind") or "delta")
            st = float(b.get("start", 0.0))
            en = float(b.get("end", 0.0))
            dl = float(b.get("delta", 0.0))
            bottom = min(st, en) if kind == "delta" else 0.0
            height = abs(dl) if kind == "delta" else en
            col = "#0284c7" if kind in ("start", "total") else ("#10b981" if dl >= 0 else "#ef4444")
            ax.bar(i, height, bottom=bottom, width=0.58, color=col, alpha=0.9)
            if i < len(bars) - 1:
                ax.plot([i + 0.29, i + 0.71], [en, en], linestyle="--", color=muted, linewidth=0.9)
            lbl_txt = f"{dl:+.3g}" if kind == "delta" else f"{en:.4g}"
            ax.text(i, bottom + height, lbl_txt, ha="center", va="bottom", fontsize=7.5, color=fg)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(b.get("label", ""))[:15] for b in bars], rotation=22, ha="right", color=fg)
        ax.axhline(0, color=muted, linewidth=0.9)

    elif ctype == "forest":
        rows = list(reversed(data.get("rows") or []))
        ref = float(data.get("reference_value", 0.0))
        ys = np.arange(len(rows))
        ax.axvline(ref, color=muted, linestyle="--", linewidth=1.3)
        for i, r in enumerate(rows):
            est = float(r.get("estimate", 0.0))
            lo = float(r.get("ci_low", est))
            hi = float(r.get("ci_high", est))
            col = "#0284c7" if est >= ref else "#ef4444"
            ax.plot([lo, hi], [i, i], color=col, linewidth=2.4)
            ax.scatter([est], [i], color=col, s=42, zorder=4)
        ax.set_yticks(ys)
        ax.set_yticklabels([str(r.get("label", ""))[:24] for r in rows], color=fg)
        ax.set_xlabel(str(data.get("x_label") or "Effect Estimate (95% CI)"), color=muted, fontsize=9)

    elif ctype == "pareto":
        items = data.get("items") or []
        xs = np.arange(len(items))
        vals = [float(it.get("value", 0.0)) for it in items]
        cums = [float(it.get("cumulative_pct", 0.0)) for it in items]
        ax.bar(xs, vals, color="#0284c7", alpha=0.88, width=0.64, label=str(data.get("y_left_label") or "Magnitude"))
        ax.set_xticks(xs)
        ax.set_xticklabels([str(it.get("label", ""))[:14] for it in items], rotation=25, ha="right", color=fg)
        ax.set_ylabel(str(data.get("y_left_label") or "Value"), color=muted, fontsize=9)
        ax2 = ax.twinx()
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_color(grid_col)
        ax2.tick_params(colors="#d97706", labelsize=8.5)
        ax2.plot(xs, cums, color="#d97706", marker="o", linewidth=2.1, markersize=4.5, label="Cumulative %")
        cutoff = float(data.get("cutoff_pct", 80.0))
        ax2.axhline(cutoff, color="#10b981", linestyle="--", linewidth=1.2)
        ax2.set_ylim(0, 105)
        ax2.set_ylabel("Cumulative Share (%)", color="#d97706", fontsize=9)

    elif ctype == "ecdf":
        for idx, s in enumerate(data.get("series") or []):
            ecdf_x = s.get("x") or []
            ecdf_y = s.get("y") or []
            if ecdf_x and ecdf_y:
                ax.step(ecdf_x, ecdf_y, where="post", linewidth=2.1, color=str(s.get("color") or _PALETTE[idx % len(_PALETTE)]), label=f"{s.get('name')} (n={s.get('n')})")
        ax.axhline(0.5, color=muted, linestyle=":", linewidth=1.0)
        ax.set_xlabel(str(data.get("x_label") or "Value"), color=muted, fontsize=9)
        ax.set_ylabel("Empirical Cumulative Probability F(x)", color=muted, fontsize=9)
        ax.legend(frameon=False, labelcolor=fg, fontsize=8.5)

    elif ctype == "violin_box":
        groups = data.get("groups") or []
        for i, g in enumerate(groups):
            ky = np.asarray(g.get("kde_y") or [], dtype=float)
            kd = np.asarray(g.get("kde_density") or [], dtype=float) * 0.36
            col = str(g.get("color") or _PALETTE[i % len(_PALETTE)])
            if len(ky) > 1:
                ax.fill_betweenx(ky, i - kd, i + kd, color=col, alpha=0.28, edgecolor=col, linewidth=1.2)
            q25, med, q75 = float(g.get("q25", 0)), float(g.get("median", 0)), float(g.get("q75", 0))
            w_lo, w_hi = float(g.get("whisker_low", q25)), float(g.get("whisker_high", q75))
            ax.plot([i, i], [w_lo, w_hi], color=fg, linewidth=1.2)
            ax.fill([i - 0.08, i + 0.08, i + 0.08, i - 0.08], [q25, q25, q75, q75], color=col, alpha=0.75)
            ax.plot([i - 0.10, i + 0.10], [med, med], color="#fbbf24", linewidth=2.2)
            outs = g.get("outliers") or []
            if outs:
                ax.scatter([i] * len(outs), outs, color="#ef4444", s=12, alpha=0.75, zorder=4)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([str(g.get("label", ""))[:16] for g in groups], rotation=18, ha="right", color=fg)
        ax.set_ylabel(str(data.get("y_label") or "Value"), color=muted, fontsize=9)

    elif ctype == "qq_plot":
        pts = data.get("points") or []
        th = [float(p["theo"]) for p in pts]
        em = [float(p["emp"]) for p in pts]
        rf = [float(p["ref"]) for p in pts]
        env_lo = [float(p["lower"]) for p in pts]
        env_hi = [float(p["upper"]) for p in pts]
        ax.fill_between(th, env_lo, env_hi, color="#0284c7", alpha=0.16, label="95% Order-Stat Envelope")
        ax.plot(th, rf, color="#ef4444", linestyle="--", linewidth=1.6, label="Normal Reference Line")
        ax.scatter(th, em, color="#0284c7", s=16, alpha=0.82, label="Empirical Quantiles")
        ax.set_xlabel("Theoretical Normal Quantiles (z)", color=muted, fontsize=9)
        ax.set_ylabel(f"Sample Quantiles — {data.get('variable_name', '')}", color=muted, fontsize=9)
        ax.legend(frameon=False, labelcolor=fg, fontsize=8.5)

    elif ctype == "regression_ci":
        pts = data.get("points") or []
        band = data.get("band") or []
        if band:
            bx = [float(b["x"]) for b in band]
            ax.fill_between(bx, [float(b["pi_low"]) for b in band], [float(b["pi_high"]) for b in band], color="#94a3b8", alpha=0.13, label="95% Prediction Band")
            ax.fill_between(bx, [float(b["ci_low"]) for b in band], [float(b["ci_high"]) for b in band], color="#0284c7", alpha=0.26, label="95% Confidence Band")
            ax.plot(bx, [float(b["yhat"]) for b in band], color="#ef4444", linewidth=2.1, label=str(data.get("equation") or "OLS Fit"))
        if pts:
            groups = sorted({str(p.get("group", "Obs")) for p in pts})
            for idx, grp in enumerate(groups):
                sub = [p for p in pts if str(p.get("group", "Obs")) == grp]
                ax.scatter([float(p["x"]) for p in sub], [float(p["y"]) for p in sub], s=18, alpha=0.75, color=_PALETTE[idx % len(_PALETTE)], label=grp[:18])
        ax.set_xlabel(str(data.get("x_label") or "X"), color=muted, fontsize=9)
        ax.set_ylabel(str(data.get("y_label") or "Y"), color=muted, fontsize=9)
        ax.legend(frameon=False, labelcolor=fg, fontsize=8)

    elif ctype == "roc_curve":
        pts = data.get("points") or []
        roc_x = [float(p["fpr"]) for p in pts]
        roc_y = [float(p["tpr"]) for p in pts]
        ax.plot([0, 1], [0, 1], linestyle="--", color=muted, linewidth=1.2, label="Random (AUC = 0.50)")
        ax.fill_between(roc_x, roc_y, alpha=0.16, color="#0284c7")
        ax.plot(roc_x, roc_y, color="#0284c7", linewidth=2.4, label=f"ROC (AUC = {data.get('auc', 0):.4f})")
        opt = data.get("optimal_point")
        if isinstance(opt, dict):
            ax.scatter([float(opt["fpr"])], [float(opt["tpr"])], color="#ef4444", s=55, zorder=5, label=f"Youden Opt (thr={opt.get('threshold')})")
        ax.set_xlabel("False Positive Rate (1 - Specificity)", color=muted, fontsize=9)
        ax.set_ylabel("True Positive Rate (Sensitivity)", color=muted, fontsize=9)
        ax.legend(loc="lower right", frameon=False, labelcolor=fg, fontsize=8.5)

    elif ctype == "pr_curve":
        pts = data.get("points") or []
        rec = [float(p["recall"]) for p in pts]
        prec = [float(p["precision"]) for p in pts]
        for f1_iso in (0.2, 0.4, 0.6, 0.8):
            r_g = np.linspace(max(0.01, f1_iso / 1.99), 1.0, 60)
            p_g = (f1_iso * r_g) / np.maximum(2 * r_g - f1_iso, 1e-6)
            valid = (p_g >= 0) & (p_g <= 1.0)
            ax.plot(r_g[valid], p_g[valid], color=grid_col, linestyle=":", linewidth=0.9)
        prev = float(data.get("prevalence", 0.5))
        ax.axhline(prev, linestyle="--", color=muted, linewidth=1.2, label=f"Prevalence ({prev:.3f})")
        ax.plot(rec, prec, color="#d97706", linewidth=2.4, label=f"PR Curve (AP = {data.get('ap', 0):.4f})")
        opt = data.get("optimal_point")
        if isinstance(opt, dict):
            ax.scatter([float(opt["recall"])], [float(opt["precision"])], color="#10b981", s=55, zorder=5, label=f"Peak F1 = {opt.get('f1')}")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Recall", color=muted, fontsize=9)
        ax.set_ylabel("Precision", color=muted, fontsize=9)
        ax.legend(loc="lower left", frameon=False, labelcolor=fg, fontsize=8.5)

    elif ctype == "calibration":
        bins = [b for b in (data.get("bins") or []) if b.get("count", 0) > 0 and b.get("mean_pred") is not None]
        mp = [float(b["mean_pred"]) for b in bins]
        fp = [float(b["frac_pos"]) for b in bins]
        yerr_lo = [max(0.0, float(b["frac_pos"]) - float(b["ci_low"])) for b in bins]
        yerr_hi = [max(0.0, float(b["ci_high"]) - float(b["frac_pos"])) for b in bins]
        ax.plot([0, 1], [0, 1], linestyle="--", color=muted, linewidth=1.3, label="Perfect Calibration")
        ax.errorbar(mp, fp, yerr=[yerr_lo, yerr_hi], fmt="o-", color="#7c3aed", linewidth=2.1, capsize=3, label=f"Model (ECE={data.get('ece', 0):.4f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Mean Predicted Probability", color=muted, fontsize=9)
        ax.set_ylabel("Empirical Positive Fraction", color=muted, fontsize=9)
        ax.legend(loc="upper left", frameon=False, labelcolor=fg, fontsize=8.5)

    elif ctype in ("confusion_matrix", "heatmap_svg"):
        ax.grid(False)
        if ctype == "confusion_matrix":
            lbls = [str(x) for x in (data.get("labels") or [])]
            mat = np.asarray(data.get("row_pcts") or [[0]], dtype=float)
            counts = data.get("counts") or [[0]]
            im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=100, aspect="auto")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row Recall (%)")
            for r in range(mat.shape[0]):
                for c in range(mat.shape[1]):
                    txt_col = "#ffffff" if mat[r, c] > 55 else "#0f172a"
                    ax.text(c, r, f"{counts[r][c]}\n({mat[r, c]:.0f}%)", ha="center", va="center", fontsize=8, fontweight="bold", color=txt_col)
            ax.set_xticks(range(len(lbls)))
            ax.set_yticks(range(len(lbls)))
            ax.set_xticklabels(lbls, rotation=25, ha="right", color=fg)
            ax.set_yticklabels(lbls, color=fg)
            ax.set_xlabel("Predicted Class", color=muted, fontsize=9)
            ax.set_ylabel("Actual Class", color=muted, fontsize=9)
        else:
            lbls = [str(x) for x in (data.get("columns") or [])]
            raw_v = data.get("values") or [[]]
            mat = np.asarray([[float(v) if v is not None else 0.0 for v in row] for row in raw_v], dtype=float)
            im = ax.imshow(mat, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
            for r in range(mat.shape[0]):
                for c in range(mat.shape[1]):
                    v_cell = raw_v[r][c]
                    txt = f"{float(v_cell):.2f}" if v_cell is not None else "—"
                    ax.text(c, r, txt, ha="center", va="center", fontsize=7.5, color="#0f172a" if abs(mat[r, c]) < 0.65 else "#ffffff")
            ax.set_xticks(range(len(lbls)))
            ax.set_yticks(range(len(lbls)))
            ax.set_xticklabels([s[:12] for s in lbls], rotation=32, ha="right", color=fg)
            ax.set_yticklabels([s[:14] for s in lbls], color=fg)

    elif ctype == "gain_lift":
        pts = data.get("points") or []
        pop = [float(p["pop_pct"]) for p in pts]
        gain = [float(p["gain_pct"]) for p in pts]
        wiz = [float(p["wizard_pct"]) for p in pts]
        ax.plot([0, 100], [0, 100], linestyle="--", color=muted, linewidth=1.2, label="Random Baseline")
        ax.plot(pop, wiz, linestyle=":", color="#10b981", linewidth=1.6, label="Wizard (Theoretical Optimal)")
        ax.fill_between(pop, pop, gain, color="#0284c7", alpha=0.15)
        ax.plot(pop, gain, color="#0284c7", linewidth=2.4, label="Model Cumulative Gain")
        ax.set_xlabel("% of Population Targeted", color=muted, fontsize=9)
        ax.set_ylabel("% of Positive Outcomes Captured", color=muted, fontsize=9)
        ax.legend(loc="lower right", frameon=False, labelcolor=fg, fontsize=8.5)

    elif ctype == "residual_diag":
        pts = data.get("points") or []
        tr = data.get("trend") or []
        sig = float(data.get("sigma", 1.0))
        ax.axhline(0, color=muted, linewidth=1.2)
        ax.axhline(2 * sig, color="#ef4444", linestyle="--", linewidth=1.1, label="+2σ Bound")
        ax.axhline(-2 * sig, color="#ef4444", linestyle="--", linewidth=1.1, label="-2σ Bound")
        ax.scatter([float(p["fitted"]) for p in pts], [float(p["residual"]) for p in pts], s=16, alpha=0.72, color="#0284c7")
        if tr:
            ax.plot([float(t["fitted"]) for t in tr], [float(t["mean_resid"]) for t in tr], color="#f59e0b", linewidth=2.2, label="Local Mean Bias")
        ax.set_xlabel("Fitted Value (ŷ)", color=muted, fontsize=9)
        ax.set_ylabel("Residual (Actual - Predicted)", color=muted, fontsize=9)
        ax.legend(frameon=False, labelcolor=fg, fontsize=8.5)

    elif ctype == "bland_altman":
        pts = data.get("points") or []
        bias = float(data.get("bias", 0.0))
        hi = float(data.get("loa_upper", 0.0))
        lo = float(data.get("loa_lower", 0.0))
        ax.axhline(bias, color="#0284c7", linewidth=1.8, label=f"Mean Bias ({bias:+.4g})")
        ax.axhline(hi, color="#ef4444", linestyle="--", linewidth=1.3, label=f"+1.96 SD ({hi:+.4g})")
        ax.axhline(lo, color="#ef4444", linestyle="--", linewidth=1.3, label=f"-1.96 SD ({lo:+.4g})")
        in_pts = [p for p in pts if not p.get("outside")]
        out_pts = [p for p in pts if p.get("outside")]
        if in_pts:
            ax.scatter([float(p["mean"]) for p in in_pts], [float(p["diff"]) for p in in_pts], color="#10b981", s=18, alpha=0.78)
        if out_pts:
            ax.scatter([float(p["mean"]) for p in out_pts], [float(p["diff"]) for p in out_pts], color="#ef4444", s=28, alpha=0.9, label="Outside 95% LoA")
        ax.set_xlabel("Mean of Actual & Predicted ½(y + ŷ)", color=muted, fontsize=9)
        ax.set_ylabel("Difference (y - ŷ)", color=muted, fontsize=9)
        ax.legend(frameon=False, labelcolor=fg, fontsize=8.5)

    elif ctype == "cusum":
        pts = data.get("points") or []
        h_lim = float(data.get("h_limit", 4.0))
        cusum_x = [int(p["step"]) for p in pts]
        cp = [float(p["c_plus"]) for p in pts]
        cm = [float(p["c_minus"]) for p in pts]
        ax.axhline(h_lim, color="#ef4444", linestyle="--", linewidth=1.3, label=f"+H (+{h_lim}σ)")
        ax.axhline(-h_lim, color="#ef4444", linestyle="--", linewidth=1.3, label=f"-H (-{h_lim}σ)")
        ax.axhline(0, color=muted, linewidth=1.0)
        ax.plot(cusum_x, cp, color="#0284c7", linewidth=2.0, label="Upper CUSUM C⁺")
        ax.plot(cusum_x, cm, color="#7c3aed", linewidth=2.0, label="Lower CUSUM C⁻")
        ax.set_xlabel("Sequential Observation Index", color=muted, fontsize=9)
        ax.set_ylabel("Cumulative Sum (σ units)", color=muted, fontsize=9)
        ax.legend(frameon=False, labelcolor=fg, fontsize=8.5)

    elif ctype == "lorenz":
        pts = data.get("points") or []
        px = [float(p["pop_share"]) for p in pts]
        py = [float(p["outcome_share"]) for p in pts]
        ax.plot([0, 1], [0, 1], linestyle="--", color=muted, linewidth=1.3, label="Line of Equality")
        ax.fill_between(px, py, px, color="#10b981", alpha=0.20, label=f"Inequality Gap (Gini = {data.get('gini', 0):.4f})")
        ax.plot(px, py, color="#10b981", linewidth=2.4, label="Lorenz Curve")
        ax.set_xlabel("Cumulative Share of Population", color=muted, fontsize=9)
        ax.set_ylabel("Cumulative Share of Outcome", color=muted, fontsize=9)
        ax.legend(loc="upper left", frameon=False, labelcolor=fg, fontsize=8.5)

    elif ctype == "parallel_coords":
        axes_m = data.get("axes") or []
        xs = np.arange(len(axes_m))
        for ln in data.get("lines") or []:
            ax.plot(xs, ln.get("values") or [], color=str(ln.get("color") or "#0284c7"), alpha=0.18, linewidth=0.9)
        for med in data.get("medians") or []:
            ax.plot(xs, med.get("values") or [], color=str(med.get("color") or "#0284c7"), linewidth=2.8, marker="o", markersize=4, label=f"{med.get('group')} (median)")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{a.get('name', '')[:12]}\n[{a.get('min')}, {a.get('max')}]" for a in axes_m], color=fg, fontsize=7.5)
        ax.set_ylabel("Min-Max Normalized Scale [0, 1]", color=muted, fontsize=9)
        ax.legend(frameon=False, labelcolor=fg, fontsize=8)

    elif ctype == "ridgeline":
        xg = np.asarray(data.get("x_grid") or [], dtype=float)
        ridges = data.get("ridges") or []
        n_r = len(ridges)
        for idx, r in enumerate(ridges):
            base_y = (n_r - 1 - idx) * 0.78
            dens = np.asarray(r.get("density") or [], dtype=float) * 0.68
            col = str(r.get("color") or _PALETTE[idx % len(_PALETTE)])
            ax.fill_between(xg, base_y, base_y + dens, color=col, alpha=0.42)
            ax.plot(xg, base_y + dens, color=col, linewidth=1.6)
        ax.set_yticks([(n_r - 1 - i) * 0.78 for i in range(n_r)])
        ax.set_yticklabels([str(r.get("label", "")) for r in ridges], color=fg)
        ax.set_xlabel(str(data.get("x_label") or "Value"), color=muted, fontsize=9)

    elif ctype == "sankey":
        ax.grid(False)
        ax.axis("off")
        sources = data.get("sources") or []
        targets = data.get("targets") or []
        links = data.get("links") or []
        total = max(1.0, float(data.get("total", 1)))
        gap = 0.03
        s_y: dict[str, tuple[float, float]] = {}
        cur = 0.94
        avail_s = max(0.2, 0.88 - gap * max(0, len(sources) - 1))
        for s in sources:
            h = max(0.025, avail_s * (float(s["total"]) / total))
            s_y[str(s["id"])] = (cur - h, cur)
            ax.add_patch(plt.Rectangle((0.12, cur - h), 0.04, h, color="#0284c7"))
            ax.text(0.10, cur - h / 2, f"{s['label']} ({s['total']})", ha="right", va="center", fontsize=8.5, color=fg)
            cur -= h + gap

        t_y: dict[str, tuple[float, float]] = {}
        cur = 0.94
        avail_t = max(0.2, 0.88 - gap * max(0, len(targets) - 1))
        for i, t in enumerate(targets):
            h = max(0.025, avail_t * (float(t["total"]) / total))
            t_y[str(t["id"])] = (cur - h, cur)
            ax.add_patch(plt.Rectangle((0.84, cur - h), 0.04, h, color=_PALETTE[(i + 2) % len(_PALETTE)]))
            ax.text(0.90, cur - h / 2, f"{t['label']} ({t['total']})", ha="left", va="center", fontsize=8.5, color=fg)
            cur -= h + gap

        s_off: dict[str, float] = {k: v[1] for k, v in s_y.items()}
        t_off: dict[str, float] = {k: v[1] for k, v in t_y.items()}
        xs_bez = np.linspace(0.16, 0.84, 40)
        bez_u = (xs_bez - 0.16) / 0.68
        smooth = 3.0 * (bez_u**2) - 2.0 * (bez_u**3)
        for lnk in links:
            sid, tid = str(lnk["source"]), str(lnk["target"])
            if sid not in s_y or tid not in t_y:
                continue
            s_tot = max(1.0, next((float(s["total"]) for s in sources if str(s["id"]) == sid), 1.0))
            t_tot = max(1.0, next((float(t["total"]) for t in targets if str(t["id"]) == tid), 1.0))
            sh = (s_y[sid][1] - s_y[sid][0]) * (float(lnk["value"]) / s_tot)
            th_h = (t_y[tid][1] - t_y[tid][0]) * (float(lnk["value"]) / t_tot)
            y_top_bez = s_off[sid] * (1.0 - smooth) + t_off[tid] * smooth
            y_bot_bez = (s_off[sid] - sh) * (1.0 - smooth) + (t_off[tid] - th_h) * smooth
            ax.fill_between(xs_bez, y_bot_bez, y_top_bez, color=str(lnk.get("color") or "#0284c7"), alpha=0.35)
            s_off[sid] -= sh
            t_off[tid] -= th_h
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    elif ctype == "funnel":
        ax.grid(False)
        ax.axis("off")
        stages = data.get("stages") or []
        n_st = max(1, len(stages))
        max_v = max((float(s.get("value", 0)) for s in stages), default=1.0) or 1.0
        for idx, st in enumerate(stages):
            frac = max(0.12, float(st.get("value", 0)) / max_v)
            next_frac = max(0.12, float(stages[idx + 1].get("value", 0)) / max_v) if idx + 1 < n_st else frac * 0.86
            y_top = 0.92 - idx * (0.82 / n_st)
            y_bot = y_top - (0.72 / n_st)
            w_t = 0.56 * frac
            w_b = 0.56 * next_frac
            poly_x = [0.5 - w_t / 2, 0.5 + w_t / 2, 0.5 + w_b / 2, 0.5 - w_b / 2]
            poly_y = [y_top, y_top, y_bot, y_bot]
            ax.fill(poly_x, poly_y, color=str(st.get("color") or _PALETTE[idx % len(_PALETTE)]), alpha=0.88)
            ax.text(0.02, 0.5 * (y_top + y_bot), str(st.get("label", "")), ha="left", va="center", fontsize=9, fontweight="bold", color=fg)
            ax.text(0.5, 0.5 * (y_top + y_bot), f"{int(st.get('value', 0)):,}", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#ffffff")
            ax.text(0.98, 0.5 * (y_top + y_bot), f"{st.get('pct_of_initial', 0)}% of initial", ha="right", va="center", fontsize=8.5, color=muted)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    elif ctype == "stacked_bar":
        bars = data.get("bars") or []
        seg_names = [str(s) for s in (data.get("segment_names") or [])]
        ys = np.arange(len(bars))
        lefts = np.zeros(len(bars), dtype=float)
        for s_idx, s_name in enumerate(seg_names):
            widths = np.asarray(
                [next((float(seg["pct"]) for seg in (b.get("segments") or []) if seg.get("name") == s_name), 0.0) for b in bars],
                dtype=float,
            )
            ax.barh(ys, widths, left=lefts, height=0.62, color=_PALETTE[s_idx % len(_PALETTE)], label=s_name[:18], alpha=0.9)
            lefts += widths
        ax.set_yticks(ys)
        ax.set_yticklabels([str(b.get("label", ""))[:18] for b in bars], color=fg)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Composition Share (%)", color=muted, fontsize=9)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=min(5, max(1, len(seg_names))), frameon=False, labelcolor=fg, fontsize=8)

    fig.suptitle(title, fontsize=12.5, fontweight="bold", color=fg, y=0.98)
    if subtitle:
        fig.text(0.5, 0.925, subtitle[:125], ha="center", va="top", fontsize=8.8, color=muted)

    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.90))
    buf = io.BytesIO()
    out_fmt = "svg" if fmt.lower() == "svg" else "png"
    fig.savefig(buf, format=out_fmt, dpi=dpi, facecolor=bg_fig, edgecolor="none")
    plt.close(fig)
    return buf.getvalue()


def _attach_native_openpyxl_chart(
    ws_vis: Any,
    ws_data: Any,
    ctype: str,
    title: str,
    columns: list[str],
    rows: list[list[Any]],
) -> None:
    """Attach a native editable openpyxl chart referencing ``ws_data`` alongside the high-DPI figure."""
    if not rows or len(columns) < 2:
        return
    n_rows = len(rows)
    # Identify numeric columns (1-based indices in ws_data)
    num_col_indices: list[int] = []
    for c_idx in range(1, len(columns) + 1):
        sample_vals = [rows[r][c_idx - 1] for r in range(min(10, n_rows)) if c_idx - 1 < len(rows[r])]
        if sample_vals and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in sample_vals if v is not None and v != ""):
            num_col_indices.append(c_idx)

    if not num_col_indices:
        return

    cats_ref = Reference(ws_data, min_col=1, min_row=2, max_row=n_rows + 1)

    if ctype == "donut":
        chart_obj: Any = DoughnutChart()
        chart_obj.title = title[:48]
        data_ref = Reference(ws_data, min_col=num_col_indices[0], min_row=1, max_row=n_rows + 1)
        chart_obj.add_data(data_ref, titles_from_data=True)
        chart_obj.set_categories(cats_ref)
    elif ctype == "radar":
        chart_obj = RadarChart()
        chart_obj.title = title[:48]
        for nc in num_col_indices[:3]:
            d_ref = Reference(ws_data, min_col=nc, min_row=1, max_row=n_rows + 1)
            chart_obj.add_data(d_ref, titles_from_data=True)
        chart_obj.set_categories(cats_ref)
    elif ctype in ("regression_ci", "qq_plot", "roc_curve", "pr_curve", "residual_diag", "bland_altman", "lorenz") and len(num_col_indices) >= 2:
        chart_obj = ScatterChart()
        chart_obj.title = title[:48]
        x_vals = Reference(ws_data, min_col=num_col_indices[0], min_row=2, max_row=n_rows + 1)
        y_vals = Reference(ws_data, min_col=num_col_indices[1], min_row=2, max_row=n_rows + 1)
        ser = Series(y_vals, x_vals, title=str(columns[num_col_indices[1] - 1]))
        chart_obj.series.append(ser)
    elif ctype in ("ecdf", "cusum", "gain_lift", "calibration"):
        chart_obj = LineChart()
        chart_obj.title = title[:48]
        for nc in num_col_indices[:2]:
            d_ref = Reference(ws_data, min_col=nc, min_row=1, max_row=n_rows + 1)
            chart_obj.add_data(d_ref, titles_from_data=True)
        chart_obj.set_categories(cats_ref)
    else:
        chart_obj = BarChart()
        chart_obj.title = title[:48]
        d_ref = Reference(ws_data, min_col=num_col_indices[0], min_row=1, max_row=n_rows + 1)
        chart_obj.add_data(d_ref, titles_from_data=True)
        chart_obj.set_categories(cats_ref)

    chart_obj.width = 16.5
    chart_obj.height = 10.5
    ws_vis.add_chart(chart_obj, "B36")


def build_chart_xlsx(chart_payload: dict[str, Any]) -> bytes:
    """Build an enterprise multi-sheet .xlsx workbook containing the rendered chart, native Excel chart, and underlying data."""
    wb = openpyxl.Workbook()
    ws_vis = wb.active
    assert ws_vis is not None
    ws_vis.title = "Chart_Visual"
    ws_data = wb.create_sheet("Chart_Data")
    ws_meta = wb.create_sheet("Statistical_Metadata")

    hdr_fill = PatternFill(start_color="0F172A", end_color="0F172A", fill_type="solid")
    sub_fill = PatternFill(start_color="F1F5F9", end_color="F1F5F9", fill_type="solid")
    hdr_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    thin_border = Border(
        left=Side(style="thin", color="CBD5E1"),
        right=Side(style="thin", color="CBD5E1"),
        top=Side(style="thin", color="CBD5E1"),
        bottom=Side(style="thin", color="CBD5E1"),
    )

    title = sanitize_spreadsheet_value(str(chart_payload.get("title") or "OmniForge Chart"))
    subtitle = sanitize_spreadsheet_value(str(chart_payload.get("subtitle") or ""))
    prov = sanitize_spreadsheet_value(str(chart_payload.get("provenance") or ""))
    now_utc = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    # 1. Populate Chart_Data sheet with genuine numeric & text cells
    cols = [str(c) for c in (chart_payload.get("table_columns") or ["Metric", "Value"])]
    rows = list(chart_payload.get("table_rows") or [])

    ws_data.freeze_panes = "A2"
    for c_i, col_name in enumerate(cols, start=1):
        cell = ws_data.cell(row=1, column=c_i, value=sanitize_spreadsheet_value(col_name))
        cell.fill = hdr_fill
        cell.font = hdr_font
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = thin_border

    for r_i, row_vals in enumerate(rows, start=2):
        for c_i in range(1, len(cols) + 1):
            raw_v = row_vals[c_i - 1] if c_i - 1 < len(row_vals) else ""
            c_obj = ws_data.cell(row=r_i, column=c_i, value=_excel_cell_val(raw_v))
            c_obj.border = thin_border
            if r_i % 2 == 1:
                c_obj.fill = sub_fill

    for c_i in range(1, len(cols) + 1):
        col_letter = get_column_letter(c_i)
        max_len = max(
            [len(str(cols[c_i - 1]))]
            + [len(str(rows[r][c_i - 1])) for r in range(min(80, len(rows))) if c_i - 1 < len(rows[r])]
        )
        ws_data.column_dimensions[col_letter].width = min(38, max(13, max_len + 4))

    # 2. Populate Statistical_Metadata sheet
    ws_meta.freeze_panes = "A2"
    for c_i, h_txt in enumerate(["Property / Metric", "Value"], start=1):
        cell = ws_meta.cell(row=1, column=c_i, value=h_txt)
        cell.fill = hdr_fill
        cell.font = hdr_font

    meta_pairs: list[tuple[str, Any]] = [
        ("Chart ID", chart_payload.get("chart_id")),
        ("Chart Type", chart_payload.get("chart_type")),
        ("Category", chart_payload.get("category")),
        ("Studio Context", chart_payload.get("studio_id")),
        ("Chart Title", chart_payload.get("title")),
        ("Statistical Subtitle", chart_payload.get("subtitle")),
        ("Dataset & Provenance", chart_payload.get("provenance")),
        ("Export Timestamp (UTC)", now_utc),
    ]
    for k, v in (chart_payload.get("kpis") or {}).items():
        meta_pairs.append((f"KPI · {k}", v))

    for r_i, (mk, mv) in enumerate(meta_pairs, start=2):
        c1 = ws_meta.cell(row=r_i, column=1, value=sanitize_spreadsheet_value(str(mk)))
        c2 = ws_meta.cell(row=r_i, column=2, value=_excel_cell_val(mv))
        c1.font = Font(name="Calibri", size=10.5, bold=True, color="0F172A")
        c1.border = thin_border
        c2.border = thin_border
    ws_meta.column_dimensions["A"].width = 32
    ws_meta.column_dimensions["B"].width = 64

    # 3. Populate Chart_Visual executive sheet + high-DPI PNG + native openpyxl chart
    ws_vis.column_dimensions["A"].width = 3
    ws_vis.column_dimensions["B"].width = 96
    ws_vis.cell(row=2, column=2, value=title).font = Font(name="Calibri", size=15, bold=True, color="0F172A")
    ws_vis.cell(row=3, column=2, value=subtitle).font = Font(name="Calibri", size=10.5, italic=True, color="334155")
    ws_vis.cell(row=4, column=2, value=f"Provenance: {prov}  |  Exported: {now_utc}").font = Font(name="Calibri", size=9.5, color="64748B")

    kpis = chart_payload.get("kpis") or {}
    if kpis:
        kpi_line = "   ·   ".join(f"{k}: {v}" for k, v in list(kpis.items())[:6])
        ws_vis.cell(row=5, column=2, value=sanitize_spreadsheet_value(kpi_line)).font = Font(name="Calibri", size=10, bold=True, color="0284C7")

    try:
        from openpyxl.drawing.image import Image as XLImage

        png_bytes = render_chart_matplotlib(chart_payload, fmt="png", theme="light", dpi=200)
        xl_img = XLImage(io.BytesIO(png_bytes))
        target_w = 780
        if xl_img.width and xl_img.height:
            ratio = target_w / float(xl_img.width)
            xl_img.width = target_w
            xl_img.height = int(xl_img.height * ratio)
        ws_vis.add_image(xl_img, "B7")
    except Exception:
        pass

    try:
        _attach_native_openpyxl_chart(
            ws_vis,
            ws_data,
            str(chart_payload.get("chart_type") or "bar"),
            str(chart_payload.get("title") or "Chart"),
            cols,
            rows,
        )
    except Exception:
        pass

    out_buf = io.BytesIO()
    wb.save(out_buf)
    return out_buf.getvalue()


def export_live_chart_artifact(
    chart_payload: dict[str, Any],
    *,
    fmt: str,
    theme: str = "light",
) -> tuple[bytes, str, str]:
    """Return ``(file_bytes, media_type, download_filename)`` for ``svg``, ``png``, or ``xlsx``."""
    fmt_clean = fmt.strip().lower()
    stem = _safe_filename(f"omniforge_{chart_payload.get('chart_id', 'chart')}")
    if fmt_clean == "svg":
        content = render_chart_matplotlib(chart_payload, fmt="svg", theme=theme)
        return content, "image/svg+xml; charset=utf-8", f"{stem}.svg"
    if fmt_clean == "png":
        content = render_chart_matplotlib(chart_payload, fmt="png", theme=theme, dpi=200)
        return content, "image/png", f"{stem}.png"
    if fmt_clean == "xlsx":
        content = build_chart_xlsx(chart_payload)
        return (
            content,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            f"{stem}.xlsx",
        )
    raise ValueError(f"Unsupported chart export format {fmt!r}. Choose from: svg, png, xlsx.")
