"""Causal Drift Root-Cause Attribution Engine (C5 — Future Developments II).

Connects :class:`~omniforge.mlops.drift.DriftDetector` feature shifts (PSI / Wasserstein)
with model prediction-shift attribution (Delta-SHAP per drifted feature) and structural
DAG upstream-ancestor traversal to distinguish *root-cause upstream sensor/source shifts*
from *downstream correlated symptoms*. Emits concrete remediation prescriptions for
Studios 11 (MLOps) and 14 (Supervisory).

All components run offline with zero third-party LLM dependency.
"""

from __future__ import annotations

from datetime import UTC, datetime
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

from omniforge.core.logging import get_logger
from omniforge.core.types import DriftStatus
from omniforge.evaluation.interpretability import InterpretabilityEngine
from omniforge.mlops.drift import DriftDetector, DriftReport, FeatureDriftResult

logger = get_logger(__name__)


# --------------------------------------------------------------------------- data models


class DriftedFeature(BaseModel):
    """A single drifted feature with attribution and root-cause classification."""

    column: str
    status: DriftStatus
    psi: float | None = None
    wasserstein_distance: float | None = None
    js_divergence: float | None = None
    delta_prediction: float = 0.0  # prediction-shift attribution (Delta-SHAP style)
    abs_delta_prediction: float = 0.0
    is_root_cause: bool = False
    is_symptom: bool = False
    upstream_ancestors: list[str] = Field(default_factory=list)
    downstream_descendants: list[str] = Field(default_factory=list)
    remediation: str = ""


class RootCauseReport(BaseModel):
    """Dataset-level causal drift root-cause attribution."""

    overall_status: DriftStatus
    n_reference: int = 0
    n_current: int = 0
    drifted_features: list[DriftedFeature] = Field(default_factory=list)
    root_causes: list[str] = Field(default_factory=list)
    symptoms: list[str] = Field(default_factory=list)
    dag_edges: list[list[str]] = Field(default_factory=list)  # [parent, child] pairs
    prediction_shift: float = 0.0
    total_attribution: float = 0.0
    remediation_prescriptions: list[str] = Field(default_factory=list)
    evaluated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


# --------------------------------------------------------------------------- causal discovery


class CausalDiscovery:
    """Constraint-based structural DAG discovery (PC-style skeleton + orientation).

    Learns a parent-set DAG from observational tabular data using first-order partial
    correlation conditional-independence tests (PC skeleton), then orients edges using
    a temporal column when available or deterministic stability/naming heuristics.
    The resulting parent map supports upstream-ancestor traversal for root-cause
    isolation (root cause = drifted node with no drifted ancestors).
    """

    def __init__(
        self,
        *,
        alpha: float = 0.05,
        max_cond_set_size: int = 2,
        random_seed: int = 42,
    ) -> None:
        self.alpha = float(np.clip(alpha, 0.001, 0.2))
        self.max_cond_set_size = max(0, int(max_cond_set_size))
        self.random_seed = int(random_seed)

    # -- partial correlation CI test -------------------------------------------
    @staticmethod
    def _partial_corr(
        df: pd.DataFrame, x: str, y: str, cond_cols: list[str]
    ) -> tuple[float, float]:
        """Return (partial_correlation, p_value) of x vs y controlling for cond_cols."""
        sub = df[[x, y] + list(cond_cols)].dropna()
        n = len(sub)
        if n < 4 or len(cond_cols) >= n - 2:
            return 0.0, 1.0
        X = sub[[x] + list(cond_cols)].to_numpy(dtype=float)
        Y = sub[[y] + list(cond_cols)].to_numpy(dtype=float)
        # Regress x on cond, y on cond; correlate residuals.
        cx = X[:, 0]
        cy = Y[:, 0]
        if len(cond_cols) > 0:
            C = X[:, 1:]
            # Ordinary least squares residuals via lstsq
            bx, _, _, _ = np.linalg.lstsq(C, cx, rcond=None)
            by, _, _, _ = np.linalg.lstsq(C, cy, rcond=None)
            rx = cx - C @ bx
            ry = cy - C @ by
        else:
            rx, ry = cx, cy
        sx = float(np.std(rx, ddof=1))
        sy = float(np.std(ry, ddof=1))
        if sx <= 1e-12 or sy <= 1e-12:
            return 0.0, 1.0
        r = float(np.corrcoef(rx, ry)[0, 1])
        r = float(np.clip(r, -1.0, 1.0))
        # t-statistic for partial correlation
        df_resid = n - len(cond_cols) - 2
        if df_resid <= 1:
            return r, 1.0
        t_stat = r * np.sqrt(df_resid / max(1.0 - r * r, 1e-12))
        p_val = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df_resid))
        return r, float(p_val)

    @staticmethod
    def _rolling_window_drift_onset(
        df: pd.DataFrame, time_col: str, col: str, n_windows: int = 5
    ) -> int | None:
        """Return the earliest time-window index where ``col`` first drifts vs baseline.

        Splits the temporally-sorted frame into ``n_windows`` consecutive windows,
        computes PSI of each later window against the first (baseline) window, and
        returns the index of the first window whose PSI >= 0.10. Returns ``None`` if
        no window drifts (or if there are too few rows). Lower index = earlier shift
        = more likely upstream root cause.
        """
        sdf = df.sort_values(time_col)
        vals = pd.to_numeric(sdf[col], errors="coerce").dropna()
        if len(vals) < max(10, n_windows * 2):
            return None
        bounds = np.linspace(0, len(vals), n_windows + 1).astype(int)
        baseline = vals.iloc[bounds[0] : bounds[1]].to_numpy(dtype=float)
        det = DriftDetector()
        for w in range(1, n_windows):
            cur = vals.iloc[bounds[w] : bounds[w + 1]].to_numpy(dtype=float)
            if len(cur) < 3:
                continue
            rep = det.detect(baseline.reshape(-1, 1), cur.reshape(-1, 1))
            if rep.features and rep.features[0].psi is not None and rep.features[0].psi >= 0.10:
                return w
        return None

    def discover(
        self, df: pd.DataFrame, time_col: str | None = None
    ) -> dict[str, set[str]]:
        """Return ``{child: set of direct parents}`` adjacency (parents = upstream causes).

        Skeleton: start complete, drop edges where X ⟂ Y | S for some small S.
        Orientation: temporal column (rolling drift onset) if provided, otherwise
        deterministic stability + naming heuristics (sensor/process -> derived/KPI).
        """
        numeric_cols = [
            c
            for c in df.select_dtypes(include=[np.number]).columns
            if df[c].dropna().nunique() > 1
        ]
        if len(numeric_cols) < 2:
            return {c: set() for c in numeric_cols}

        rng = np.random.default_rng(self.random_seed)
        cols = list(numeric_cols)

        # ---- PC skeleton -------------------------------------------------------
        edges: set[tuple[str, str]] = set()
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                edges.add((a, b) if a < b else (b, a))

        for cond_size in range(self.max_cond_set_size + 1):
            to_remove: set[tuple[str, str]] = set()
            for a, b in list(edges):
                others = [c for c in cols if c not in (a, b)]
                independent = False
                if len(others) >= cond_size:
                    for cond_set in combinations(others, cond_size):
                        _, p_val = self._partial_corr(df, a, b, list(cond_set))
                        if p_val >= self.alpha:
                            independent = True
                            break
                if independent:
                    to_remove.add((a, b))
            edges -= to_remove

        # ---- Orientation -------------------------------------------------------
        # Sensor/process upstream keywords -> derived/KPI downstream keywords.
        upstream_hints = (
            "temp", "pressure", "flow", "voltage", "current", "level", "ph",
            "ph_value", "rate", "speed", "vibration", "load", "torque", "signal",
            "raw", "sensor", "input", "feed", "inlet", "outlet",
        )
        downstream_hints = (
            "ratio", "diff", "delta", "change", "score", "index", "kpi",
            "efficiency", "yield", "quality", "output", "result", "prediction",
            "target", "label", "flag", "alert", "anomaly",
        )

        def upstream_score(col: str) -> float:
            c = col.lower()
            score = 0.0
            for hint in upstream_hints:
                if hint in c:
                    score += 1.0
            for hint in downstream_hints:
                if hint in c:
                    score -= 1.0
            return score

        # Temporal drift onset: earlier shifting feature is upstream (root cause).
        onset: dict[str, int | None] = {}
        if time_col and time_col in df.columns:
            for c in cols:
                onset[c] = self._rolling_window_drift_onset(df, time_col, c)

        # Stability score: lower drift magnitude -> more likely upstream source.
        drift: dict[str, float] = {}
        if not (time_col and time_col in df.columns):
            drift = {c: float(df[c].std(ddof=1)) if df[c].notna().sum() > 1 else 0.0 for c in cols}
        else:
            try:
                sdf = df.sort_values(time_col)
                n = len(sdf)
                third = max(1, n // 3)
                early = sdf.iloc[:third]
                late = sdf.iloc[-third:]
                det = DriftDetector()
                for c in cols:
                    rep = det.detect(early[[c]], late[[c]])
                    drift[c] = rep.features[0].psi if rep.features else 0.0
            except Exception:  # pragma: no cover - defensive
                drift = {c: float(df[c].std()) for c in cols}

        parents: dict[str, set[str]] = {c: set() for c in cols}
        for a, b in edges:
            sa, sb = upstream_score(a), upstream_score(b)
            oa, ob = onset.get(a), onset.get(b)
            da, db = drift.get(a, 0.0), drift.get(b, 0.0)
            # 1) Earlier drift onset wins (temporal causality).
            if oa is not None and ob is not None and oa != ob:
                parent, child = (a, b) if oa < ob else (b, a)
            elif oa is not None and ob is None:
                parent, child = a, b  # a shifts, b does not -> a upstream
            elif ob is not None and oa is None:
                parent, child = b, a
            # 2) Fall back to naming heuristics / stability.
            elif sa > sb or (sa == sb and da < db):
                parent, child = a, b
            elif sb > sa or (sa == sb and db < da):
                parent, child = b, a
            else:
                # Exact tie: deterministic by column order (lower index = upstream).
                parent, child = (a, b) if cols.index(a) < cols.index(b) else (b, a)
            parents[child].add(parent)
            logger.debug(
                "root_cause_edge_oriented", parent=parent, child=child,
                upstream_score_a=sa, upstream_score_b=sb, onset_a=oa, onset_b=ob,
                drift_a=round(da, 4), drift_b=round(db, 4),
            )
        return parents

    # -- ancestor traversal ------------------------------------------------------
    @staticmethod
    def upstream_ancestors(
        parents: dict[str, set[str]], node: str, visited: set[str] | None = None
    ) -> set[str]:
        """All upstream ancestors of ``node`` in the parent map."""
        if visited is None:
            visited = set()
        anc: set[str] = set()
        stack = list(parents.get(node, set()))
        while stack:
            p = stack.pop()
            if p in visited or p in anc:
                continue
            visited.add(p)
            anc.add(p)
            stack.extend(parents.get(p, set()))
        return anc

    @staticmethod
    def downstream_descendants(
        parents: dict[str, set[str]], node: str, visited: set[str] | None = None
    ) -> set[str]:
        """All downstream descendants of ``node`` (reverse traversal)."""
        if visited is None:
            visited = set()
        # Build children map once
        children: dict[str, set[str]] = {}
        for child, par_set in parents.items():
            for p in par_set:
                children.setdefault(p, set()).add(child)
        desc: set[str] = set()
        stack = list(children.get(node, set()))
        while stack:
            c = stack.pop()
            if c in visited or c in desc:
                continue
            visited.add(c)
            desc.add(c)
            stack.extend(children.get(c, set()))
        return desc


# --------------------------------------------------------------------------- root-cause analyzer


class DriftRootCauseAnalyzer:
    """End-to-end causal drift root-cause attribution.

    Pipeline:
      1. DriftDetector on reference vs current -> per-feature drift (PSI/Wasserstein)
      2. Prediction-shift attribution (Delta-SHAP style feature replacement)
      3. CausalDiscovery DAG + upstream-ancestor traversal
      4. Classify each drifted feature as root-cause (source, no drifted ancestors)
         or symptom (downstream, has drifted ancestors)
      5. Emit remediation prescriptions
    """

    def __init__(
        self,
        *,
        random_seed: int = 42,
        psi_warning: float = 0.10,
        psi_critical: float = 0.25,
        max_cond_set_size: int = 2,
    ) -> None:
        self.random_seed = int(random_seed)
        self.psi_warning = psi_warning
        self.psi_critical = psi_critical
        self.max_cond_set_size = max_cond_set_size

    def analyze(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        *,
        model: Any | None = None,
        feature_cols: list[str] | None = None,
        target_col: str | None = None,
        time_col: str | None = None,
    ) -> RootCauseReport:
        """Run the full drift -> attribution -> root-cause pipeline.

        Returns a :class:`RootCauseReport` with root-cause / symptom classification
        and remediation prescriptions.
        """
        rng = np.random.default_rng(self.random_seed)

        # ---- feature selection -----------------------------------------------
        all_cols = [c for c in reference_df.columns if c in current_df.columns]
        if feature_cols is not None:
            feat_cols = [c for c in feature_cols if c in all_cols]
        else:
            feat_cols = [c for c in all_cols if c != target_col and c != time_col]
        if not feat_cols:
            raise ValueError("No shared feature columns between reference and current.")

        ref = reference_df[feat_cols].copy()
        cur = current_df[feat_cols].copy()

        # ---- Step 1: Drift detection -----------------------------------------
        det = DriftDetector(psi_warning=self.psi_warning, psi_critical=self.psi_critical)
        drift_report: DriftReport = det.detect(ref, cur)
        drift_map: dict[str, FeatureDriftResult] = {f.column: f for f in drift_report.features}

        # ---- Step 2: Prediction-shift attribution (exact Delta-SHAP) ---------
        attributions: dict[str, float] = {}
        if model is not None and hasattr(model, "predict"):
            try:
                interp = InterpretabilityEngine(random_seed=self.random_seed)
                # Mean SHAP attribution per feature on reference vs current samples.
                n_attr = max(3, min(len(ref), 8))
                ref_sample = ref.sample(n=n_attr, random_state=self.random_seed)
                cur_sample = cur.sample(n=n_attr, random_state=self.random_seed + 1)

                def _mean_attrs(sample_df: pd.DataFrame) -> dict[str, float]:
                    agg: dict[str, list[float]] = {c: [] for c in feat_cols}
                    for _, row in sample_df.iterrows():
                        inst = row.to_frame().T
                        shap_local = interp.explain_shap_local(
                            model, ref, inst, n_samples=32
                        )
                        for col, val in shap_local.get("attributions", {}).items():
                            if col in agg:
                                agg[col].append(float(val))
                    return {c: float(np.mean(v)) if v else 0.0 for c, v in agg.items()}

                attrs_ref = _mean_attrs(ref_sample)
                attrs_cur = _mean_attrs(cur_sample)
                for col in feat_cols:
                    if col not in drift_map or drift_map[col].status == DriftStatus.HEALTHY:
                        continue
                    attributions[col] = round(
                        float(attrs_cur.get(col, 0.0) - attrs_ref.get(col, 0.0)), 6
                    )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("root_cause_attribution_skipped", reason=str(exc))

        # ---- Step 3: CausalDiscovery DAG -------------------------------------
        cd = CausalDiscovery(
            alpha=0.05, max_cond_set_size=self.max_cond_set_size, random_seed=self.random_seed
        )
        parents = cd.discover(pd.concat([ref, cur], ignore_index=True), time_col=time_col)

        dag_edges: list[list[str]] = []
        for child, par_set in parents.items():
            for p in par_set:
                dag_edges.append([p, child])

        # ---- Step 4: Root-cause / symptom classification ---------------------
        drifted_cols = [f.column for f in drift_report.features if f.status != DriftStatus.HEALTHY]
        drifted_set = set(drifted_cols)

        feature_details: list[DriftedFeature] = []
        for fres in drift_report.features:
            col = fres.column
            if col not in feat_cols:
                continue
            ancestors = CausalDiscovery.upstream_ancestors(parents, col)
            drifted_ancestors = ancestors & drifted_set
            descendants = CausalDiscovery.downstream_descendants(parents, col)
            drifted_descendants = descendants & drifted_set

            is_root = (
                fres.status != DriftStatus.HEALTHY
                and len(drifted_ancestors) == 0
                and len(drifted_descendants) > 0
            )
            is_symptom = (
                fres.status != DriftStatus.HEALTHY
                and len(drifted_ancestors) > 0
            )
            # A drifted source with no descendants is still a root cause (isolated source).
            if fres.status != DriftStatus.HEALTHY and len(drifted_ancestors) == 0 and len(drifted_descendants) == 0:
                is_root = True

            d = DriftedFeature(
                column=col,
                status=fres.status,
                psi=fres.psi,
                wasserstein_distance=fres.wasserstein_distance,
                js_divergence=fres.js_divergence,
                delta_prediction=round(attributions.get(col, 0.0), 6),
                abs_delta_prediction=round(abs(attributions.get(col, 0.0)), 6),
                is_root_cause=is_root,
                is_symptom=is_symptom,
                upstream_ancestors=sorted(ancestors),
                downstream_descendants=sorted(drifted_descendants),
                remediation="",
            )
            feature_details.append(d)

        root_causes = sorted(f.column for f in feature_details if f.is_root_cause)
        symptoms = sorted(f.column for f in feature_details if f.is_symptom)

        # ---- Step 5: Remediation prescriptions -------------------------------
        prescriptions: list[str] = []
        if not drifted_cols:
            prescriptions.append("No material drift detected; no remediation required.")
        else:
            for rc in root_causes:
                fres = drift_map.get(rc)
                psi_str = f"PSI={fres.psi:.3f}" if fres and fres.psi is not None else "PSI=n/a"
                attr = attributions.get(rc, 0.0)
                prescriptions.append(
                    f"ROOT CAUSE '{rc}' ({psi_str}): investigate upstream sensor/source "
                    f"feed for '{rc}' (attributed prediction shift {attr:+.4f}); "
                    f"check calibration, sensor drift, or upstream ETL change."
                )
            for sym in symptoms:
                fres = drift_map.get(sym)
                anc = ", ".join(
                    sorted(CausalDiscovery.upstream_ancestors(parents, sym) & drifted_set)
                )
                prescriptions.append(
                    f"SYMPTOM '{sym}': downstream effect of drifted upstream cause(s) "
                    f"[{anc}]; do not remediate '{sym}' directly — fix the upstream root cause(s)."
                )
            if not root_causes and drifted_cols:
                prescriptions.append(
                    "Drift detected but no clear upstream root cause isolated; "
                    "inspect all drifted features for shared upstream source changes."
                )

        total_attribution = float(sum(abs(v) for v in attributions.values()))
        # Net signed prediction shift attributed to drift (Delta-SHAP).
        prediction_shift = (
            round(float(sum(attributions.values())), 6) if attributions else 0.0
        )

        return RootCauseReport(
            overall_status=drift_report.overall_status,
            n_reference=int(len(ref)),
            n_current=int(len(cur)),
            drifted_features=feature_details,
            root_causes=root_causes,
            symptoms=symptoms,
            dag_edges=dag_edges,
            prediction_shift=prediction_shift,
            total_attribution=round(total_attribution, 6),
            remediation_prescriptions=prescriptions,
        )

    def analyze_workbench_state(
        self,
        state: dict[str, Any],
        *,
        current_df: pd.DataFrame | None = None,
        target_col: str | None = None,
    ) -> RootCauseReport:
        """Convenience: analyze drift using the workbench reference dataset and active model.

        Uses ``state['uap_df']`` as reference, ``current_df`` (or state's analysis frame) as
        current, and the active champion model for prediction-shift attribution.
        """
        ref_df = state.get("uap_df")
        if not isinstance(ref_df, pd.DataFrame) or ref_df.empty:
            raise ValueError("No reference dataset in workbench state (load Studio 01 first).")
        if current_df is None:
            from omniforge.api import workbench as wb

            current_df = wb.get_analysis_dataframe(state, ref_df)
        model = state.get("uap_active_model")
        feat_cols = list(state.get("uap_active_feature_cols") or [])
        if not feat_cols:
            feat_cols = [c for c in ref_df.columns if c != target_col]
        return self.analyze(
            ref_df, current_df, model=model, feature_cols=feat_cols, target_col=target_col
        )
