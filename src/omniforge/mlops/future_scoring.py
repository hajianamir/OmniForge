"""Operational Future Target Prediction & Incoming Data Scoring Engine (Manual, File, and SQL).

Connects new/future observations from manual scenario tables, uploaded files (CSV/Excel/Parquet/JSON),
or live SQL queries to all fitted OmniForge models (Baselines, Neural Networks, and Advanced Engines),
synthesizing future target forecasts with statistical distribution-shift checks, risk tail thresholds,
and discovered cluster assignments from prior OmniForge analyses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from omniforge.core.exceptions import DataIngestionError, DependencyMissingError, SecurityError
from omniforge.core.logging import get_logger
from omniforge.core.types import TaskType
from omniforge.data.connectors import validate_readonly_sql
from omniforge.data.ingestion import DataIngestionEngine

logger = get_logger(__name__)


def _redact_uri_secret_in_error(err_text: str, raw_uri: str) -> str:
    """Scrub any password embedded in ``raw_uri`` from downstream driver exception text."""
    from urllib.parse import urlparse

    try:
        pw = urlparse(raw_uri).password
    except ValueError:
        pw = None
    if pw:
        return err_text.replace(pw, "***")
    return err_text


def _require_sqlalchemy() -> tuple[Any, Any]:
    """Import SQLAlchemy lazily so core-only ``omniforge.mlops`` imports work without ``omniforge[data]``."""
    try:
        from sqlalchemy import create_engine, text
    except ImportError as exc:
        raise DependencyMissingError(
            "Install 'sqlalchemy' (or 'omniforge[data]') for SQL-backed future target scoring.",
            suggestions=["pip install 'omniforge[data]'"],
        ) from exc
    return create_engine, text


@dataclass
class FuturePredictionResult:
    """Complete output of operational future-target scoring with analytical context."""

    scored_df: pd.DataFrame
    target_summaries: dict[str, dict[str, Any]]
    schema_report: dict[str, Any]
    feature_shift_table: list[dict[str, Any]] = field(default_factory=list)
    model_comparison_rows: list[dict[str, Any]] = field(default_factory=list)
    executive_outlook: list[str] = field(default_factory=list)
    source_type: str = "manual"
    n_rows_scored: int = 0


class FutureTargetPredictor:
    """Predicts future target outcomes for new incoming data using trained OmniForge models and prior analyses."""

    @staticmethod
    def build_manual_template(
        reference_df: pd.DataFrame,
        feature_cols: list[str],
        n_rows: int = 3,
    ) -> pd.DataFrame:
        """Build a schema-accurate editable template populated with realistic baseline, low, and high scenarios."""
        quantiles = [0.50, 0.25, 0.75]
        rows: list[dict[str, Any]] = []
        for idx in range(max(1, int(n_rows))):
            q = quantiles[idx % len(quantiles)]
            row: dict[str, Any] = {}
            for col in feature_cols:
                if col not in reference_df.columns:
                    row[col] = 0.0
                    continue
                s = reference_df[col].dropna()
                if len(s) == 0:
                    row[col] = 0.0
                elif pd.api.types.is_numeric_dtype(s):
                    val = float(s.quantile(q))
                    row[col] = int(round(val)) if pd.api.types.is_integer_dtype(s) else round(val, 4)
                else:
                    modes = s.mode()
                    uniq = s.astype(str).unique().tolist()
                    if idx == 0 and len(modes) > 0:
                        row[col] = str(modes.iloc[0])
                    else:
                        row[col] = str(uniq[idx % len(uniq)]) if uniq else "unknown"
            rows.append(row)
        return pd.DataFrame(rows, columns=feature_cols)

    @staticmethod
    def create_sample_sqlite_db(
        reference_df: pd.DataFrame,
        feature_cols: list[str],
        db_path: Path | str,
        *,
        table_name: str = "incoming_future_data",
        n_rows: int = 12,
        seed: int = 42,
    ) -> tuple[str, str, pd.DataFrame]:
        """Create a realistic sample SQLite database matching the analyzed feature schema for instant SQL testing."""
        create_engine, _ = _require_sqlalchemy()
        clean_table = (
            "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in table_name.strip())
            or "incoming_future_data"
        )
        rng = np.random.default_rng(seed)
        target_path = Path(db_path).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, Any]] = []
        n_gen = max(1, int(n_rows))
        for i in range(n_gen):
            row: dict[str, Any] = {"record_id": i + 1}
            for col in feature_cols:
                if col not in reference_df.columns:
                    row[col] = 0.0
                    continue
                s = reference_df[col].dropna()
                if len(s) == 0:
                    row[col] = 0.0
                elif pd.api.types.is_numeric_dtype(s):
                    base_val = float(s.iloc[int(rng.integers(0, len(s)))])
                    std_v = float(s.std()) if len(s) > 1 else 0.0
                    # Small realistic perturbation around observed values (+ one intentional tail scenario on last row)
                    jitter = rng.normal(0.0, 0.25 * std_v) if i < n_gen - 1 else 2.2 * std_v
                    val = base_val + float(jitter)
                    row[col] = int(round(val)) if pd.api.types.is_integer_dtype(s) else round(val, 4)
                else:
                    vals = s.astype(str).to_numpy()
                    row[col] = str(vals[int(rng.integers(0, len(vals)))])
            rows.append(row)

        sample_df = pd.DataFrame(rows)
        uri = f"sqlite:///{target_path.as_posix()}"
        engine = create_engine(uri)
        try:
            sample_df.to_sql(clean_table, engine, if_exists="replace", index=False)
        finally:
            engine.dispose()
        query = f'SELECT * FROM "{clean_table}"'
        return uri, query, sample_df

    @staticmethod
    def load_from_sql(connection_uri: str, query: str) -> pd.DataFrame:
        """Load incoming future feature records from any SQLAlchemy-supported SQL database."""
        raw_uri = connection_uri.strip()
        if not raw_uri:
            raise DataIngestionError("SQL connection URI cannot be empty.")
        if not query.strip():
            raise DataIngestionError("SQL query cannot be empty.")
        validated_sql = validate_readonly_sql(query.strip())
        create_engine, text = _require_sqlalchemy()
        engine: Any = None
        try:
            engine = create_engine(raw_uri)
            with engine.connect() as conn:
                df = pd.read_sql_query(text(validated_sql), conn)
        except SecurityError:
            raise
        except Exception as exc:
            safe_uri = DataIngestionEngine._sanitize_uri(raw_uri)
            safe_err = _redact_uri_secret_in_error(str(exc), raw_uri)
            raise DataIngestionError(
                f"Failed to execute SQL query on '{safe_uri}': {safe_err}",
                cause=exc,
            ) from exc
        finally:
            if engine is not None:
                engine.dispose()
        return df

    @staticmethod
    def write_predictions_to_sql(
        predictions_df: pd.DataFrame,
        connection_uri: str,
        *,
        table_name: str = "omniforge_future_predictions",
        if_exists: Literal["append", "replace"] = "replace",
    ) -> dict[str, Any]:
        """Persist scored future target predictions back into a SQL database table."""
        raw_uri = connection_uri.strip()
        if not raw_uri:
            raise DataIngestionError("SQL connection URI cannot be empty.")
        create_engine, _ = _require_sqlalchemy()
        clean_table = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in table_name.strip()) or "omniforge_future_predictions"
        engine: Any = None
        try:
            engine = create_engine(raw_uri)
            predictions_df.to_sql(clean_table, engine, if_exists=if_exists, index=False)
        except Exception as exc:
            safe_uri = DataIngestionEngine._sanitize_uri(raw_uri)
            safe_err = _redact_uri_secret_in_error(str(exc), raw_uri)
            raise DataIngestionError(
                f"Failed to persist predictions to '{safe_uri}' (table '{clean_table}'): {safe_err}",
                cause=exc,
            ) from exc
        finally:
            if engine is not None:
                engine.dispose()
        return {
            "connection_uri": DataIngestionEngine._sanitize_uri(raw_uri),
            "table_name": clean_table,
            "rows_written": int(len(predictions_df)),
            "columns_written": int(predictions_df.shape[1]),
        }

    @staticmethod
    def align_incoming_schema(
        incoming_df: pd.DataFrame,
        reference_df: pd.DataFrame,
        feature_cols: list[str],
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Align incoming records to exact training feature columns, imputing missing columns with historical statistics."""
        matched = [c for c in feature_cols if c in incoming_df.columns]
        missing = [c for c in feature_cols if c not in incoming_df.columns]
        extra = [c for c in incoming_df.columns if c not in feature_cols]

        aligned = pd.DataFrame(index=incoming_df.index)
        imputed_defaults: dict[str, Any] = {}
        for col in feature_cols:
            ref_s = reference_df[col] if col in reference_df.columns else pd.Series(dtype=float)
            is_ref_num = pd.api.types.is_numeric_dtype(ref_s) and len(ref_s) > 0
            if col in incoming_df.columns:
                ser = incoming_df[col].copy()
                if is_ref_num:
                    ser = pd.to_numeric(ser, errors="coerce")
                    fill_v = float(ref_s.median()) if ref_s.notna().any() else 0.0
                    if ser.isna().any():
                        ser = ser.fillna(fill_v)
                else:
                    modes = ref_s.dropna().mode()
                    fill_s = str(modes.iloc[0]) if len(modes) > 0 else "unknown"
                    ser = ser.fillna(fill_s).astype(str)
                aligned[col] = ser
            else:
                if is_ref_num:
                    fill_v = float(ref_s.median()) if ref_s.notna().any() else 0.0
                    aligned[col] = fill_v
                    imputed_defaults[col] = round(fill_v, 4)
                else:
                    modes = ref_s.dropna().mode()
                    fill_s = str(modes.iloc[0]) if len(modes) > 0 else "unknown"
                    aligned[col] = fill_s
                    imputed_defaults[col] = fill_s

        schema_report = {
            "required_features": len(feature_cols),
            "matched_features": len(matched),
            "missing_features": missing,
            "extra_columns_preserved": extra,
            "imputed_defaults": imputed_defaults,
            "coverage_pct": round(len(matched) / max(len(feature_cols), 1) * 100.0, 1),
        }
        return aligned, schema_report

    @classmethod
    def predict_future(
        cls,
        incoming_df: pd.DataFrame,
        reference_df: pd.DataFrame,
        feature_cols: list[str],
        target_models: dict[str, Any],
        *,
        secondary_models_by_target: dict[str, dict[str, Any]] | None = None,
        clustering_result: Any | None = None,
        risk_report: dict[str, Any] | None = None,
        source_type: str = "manual",
    ) -> FuturePredictionResult:
        """Predict future target outcomes across all available models with distribution-shift and risk synthesis.

        Args:
            incoming_df: New incoming feature DataFrame from Manual Entry, File Upload, or SQL Query.
            reference_df: Historical analyzed dataset used during diagnostics and model training.
            feature_cols: Ordered list of predictor columns expected by the models.
            target_models: Mapping ``{target_col: primary_champion_model}``.
            secondary_models_by_target: Optional mapping ``{target_col: {model_name: fitted_model}}``
                covering Baselines, Neural Networks, and Classical/Ensemble engines for consensus comparison.
            clustering_result: Optional :class:`ClusteringResult` from Studio 02 to assign future rows to segments.
            risk_report: Optional risk dictionary from Studio 03 for tail/VaR alerting on predicted targets.
            source_type: Origin tag (``"manual"``, ``"file"``, or ``"sql"``).

        Returns:
            A :class:`FuturePredictionResult` containing scored rows, multi-model consensus, shift alerts,
            and executive analytical outlook.
        """
        if incoming_df is None or len(incoming_df) == 0:
            raise DataIngestionError("Incoming data has 0 rows — provide at least 1 row to predict future targets.")
        if not target_models:
            raise DataIngestionError("No trained models or baselines are available yet for future target prediction.")

        X_aligned, schema_report = cls.align_incoming_schema(incoming_df, reference_df, feature_cols)
        out = incoming_df.reset_index(drop=True).copy()
        X_aligned = X_aligned.reset_index(drop=True)

        # 1. Per-row Distribution Shift & Out-of-Range Audit against historical training data
        num_feats = [
            c for c in feature_cols if c in reference_df.columns and pd.api.types.is_numeric_dtype(reference_df[c])
        ]
        n_rows = len(X_aligned)
        ood_counts = np.zeros(n_rows, dtype=int)
        max_z = np.zeros(n_rows, dtype=float)
        flagged_per_row: list[list[str]] = [[] for _ in range(n_rows)]
        feature_shift_table: list[dict[str, Any]] = []

        for col in num_feats:
            ref_vals = pd.to_numeric(reference_df[col], errors="coerce").dropna().to_numpy(dtype=float)
            inc_vals = pd.to_numeric(X_aligned[col], errors="coerce").to_numpy(dtype=float)
            if len(ref_vals) < 3:
                continue
            r_mean = float(np.mean(ref_vals))
            r_std = float(np.std(ref_vals, ddof=1))
            r_min = float(np.min(ref_vals))
            r_max = float(np.max(ref_vals))
            q1, q3 = float(np.quantile(ref_vals, 0.25)), float(np.quantile(ref_vals, 0.75))
            iqr = max(q3 - q1, 1e-12)
            lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr

            z_vals = np.abs(inc_vals - r_mean) / max(r_std, 1e-12)
            max_z = np.maximum(max_z, z_vals)
            is_ood = (inc_vals < r_min) | (inc_vals > r_max) | (z_vals > 3.0) | (inc_vals < lo_fence) | (inc_vals > hi_fence)
            for idx_r in np.where(is_ood)[0]:
                ood_counts[idx_r] += 1
                flagged_per_row[idx_r].append(f"{col} (z={z_vals[idx_r]:.1f})")

            inc_mean = float(np.nanmean(inc_vals))
            shift_z = (inc_mean - r_mean) / max(r_std, 1e-12)
            feature_shift_table.append(
                {
                    "Feature": col,
                    "Historical Mean": round(r_mean, 4),
                    "Historical Range": f"[{r_min:.3g}, {r_max:.3g}]",
                    "Incoming Mean": round(inc_mean, 4),
                    "Mean Shift (sigma)": round(shift_z, 3),
                    "Out-of-Range Rows": int(np.sum(is_ood)),
                    "Shift Status": "High Shift" if abs(shift_z) >= 1.5 or int(np.sum(is_ood)) > 0 else "Aligned",
                }
            )

        out["ood_flagged_features"] = ood_counts
        out["max_feature_zscore"] = np.round(max_z, 2)
        out["analytical_data_risk"] = [
            "Out-of-Distribution" if (cnt >= 2 or z >= 3.5) else ("Elevated Shift" if (cnt == 1 or z >= 2.2) else "In-Distribution")
            for cnt, z in zip(ood_counts, max_z, strict=False)
        ]
        out["shift_details"] = [", ".join(fl[:4]) if fl else "Within historical bounds" for fl in flagged_per_row]

        # 2. Nearest Discovered Cluster Assignment (if Studio 02 Clustering was performed)
        if clustering_result is not None and getattr(clustering_result, "cluster_profiles", None):
            try:
                raw_c_feats = (
                    getattr(clustering_result, "feature_columns", None)
                    or getattr(clustering_result, "feature_cols", None)
                    or []
                )
                c_feats = [c for c in raw_c_feats if c in X_aligned.columns and c in reference_df.columns]
                profiles = [p for p in clustering_result.cluster_profiles if not str(p.get("Cluster", "")).lower().startswith("noise")]
                if c_feats and profiles:
                    ref_means = np.array([float(pd.to_numeric(reference_df[c], errors="coerce").mean()) for c in c_feats], dtype=float)
                    ref_stds = np.array([max(float(pd.to_numeric(reference_df[c], errors="coerce").std()), 1e-12) for c in c_feats], dtype=float)
                    centroids = []
                    c_labels = []
                    for prof in profiles:
                        c_labels.append(str(prof.get("Cluster", "Cluster")))
                        vec = [float(prof.get(f"{c} (mean)", ref_means[i])) for i, c in enumerate(c_feats)]
                        centroids.append((np.array(vec, dtype=float) - ref_means) / ref_stds)
                    cent_mat = np.vstack(centroids)
                    inc_mat = (X_aligned[c_feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float) - ref_means) / ref_stds
                    dists = np.linalg.norm(inc_mat[:, None, :] - cent_mat[None, :, :], axis=2)
                    nearest_idx = np.argmin(dists, axis=1)
                    out["assigned_segment"] = [c_labels[int(i)] for i in nearest_idx]
            except Exception as exc:
                logger.debug("future_cluster_assignment_skipped", error=str(exc))

        # 3. Predict Future Target(s) with Champion + Baselines + Neural Networks + Consensus
        target_summaries: dict[str, dict[str, Any]] = {}
        model_comparison_rows: list[dict[str, Any]] = []
        executive_outlook: list[str] = []

        for t_col, primary_model in target_models.items():
            task = getattr(primary_model, "task_type", TaskType.REGRESSION)
            is_clf = task in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)

            primary_preds = np.asarray(primary_model.predict(X_aligned))
            out[f"future_{t_col}"] = primary_preds

            primary_conf: np.ndarray | None = None
            if is_clf:
                try:
                    proba = np.asarray(primary_model.predict_proba(X_aligned), dtype=float)
                    primary_conf = np.max(proba, axis=1)
                    out[f"confidence_{t_col}_pct"] = np.round(primary_conf * 100.0, 2)
                except Exception:
                    primary_conf = None

            # Evaluate secondary models (Baselines, Deep Learning, Classical/Ensemble engines)
            sec_map = (secondary_models_by_target or {}).get(t_col, {})
            all_engine_preds: dict[str, np.ndarray] = {
                str(getattr(primary_model, "name", "Champion")): primary_preds
            }
            baseline_preds: np.ndarray | None = None
            baseline_name_used: str | None = None
            neural_preds: np.ndarray | None = None
            neural_name_used: str | None = None

            for m_name, m_obj in sec_map.items():
                if m_obj is primary_model:
                    continue
                try:
                    p_sec = np.asarray(m_obj.predict(X_aligned))
                    if len(p_sec) != n_rows:
                        continue
                    all_engine_preds[m_name] = p_sec
                    fam = str(getattr(m_obj, "model_family", "")).lower()
                    lname = m_name.lower()
                    if ("baseline" in fam or "baseline" in lname) and (
                        baseline_preds is None or "ridge" in lname or "logistic" in lname
                    ):
                        baseline_preds = p_sec
                        baseline_name_used = m_name
                    if ("deep" in fam or "deep" in lname or "mlp" in lname or "cnn" in lname or "transformer" in lname or "gcn" in lname) and neural_preds is None:
                        neural_preds = p_sec
                        neural_name_used = m_name
                except Exception as exc:
                    logger.debug("secondary_future_predict_skipped", target=t_col, model=m_name, error=str(exc))

            if baseline_preds is not None:
                out[f"baseline_{t_col}"] = baseline_preds
            if neural_preds is not None:
                out[f"neural_{t_col}"] = neural_preds

            # Historical target reference stats & synthesis
            hist_s = reference_df[t_col].dropna() if t_col in reference_df.columns else pd.Series(dtype=float)

            if is_clf:
                # Multi-model consensus vote & agreement %
                pred_matrix = np.column_stack([arr.astype(str) for arr in all_engine_preds.values()])
                consensus_cls: list[str] = []
                agreement_pct: list[float] = []
                for row_i in range(n_rows):
                    vals_r, counts_r = np.unique(pred_matrix[row_i], return_counts=True)
                    best_i = int(np.argmax(counts_r))
                    consensus_cls.append(str(vals_r[best_i]))
                    agreement_pct.append(round(float(counts_r[best_i]) / pred_matrix.shape[1] * 100.0, 1))
                out[f"consensus_{t_col}"] = consensus_cls
                out[f"engine_agreement_{t_col}_pct"] = agreement_pct

                pred_counts = pd.Series(primary_preds).astype(str).value_counts(normalize=True).to_dict()
                hist_counts = hist_s.astype(str).value_counts(normalize=True).to_dict() if len(hist_s) > 0 else {}
                dominant_cls = max(pred_counts, key=pred_counts.get) if pred_counts else "N/A"
                dominant_share = float(pred_counts.get(dominant_cls, 0.0)) * 100.0
                hist_dom_share = float(hist_counts.get(dominant_cls, 0.0)) * 100.0
                avg_conf = float(np.mean(primary_conf) * 100.0) if primary_conf is not None else None

                for m_name, p_arr in all_engine_preds.items():
                    vc = pd.Series(p_arr).astype(str).value_counts(normalize=True)
                    top_c = str(vc.index[0]) if len(vc) else "N/A"
                    model_comparison_rows.append(
                        {
                            "Target": t_col,
                            "Engine": m_name,
                            "Role": (
                                "Champion"
                                if m_name == str(getattr(primary_model, "name", "Champion"))
                                else ("Baseline" if "baseline" in m_name.lower() else ("Neural Network" if any(k in m_name.lower() for k in ("deep", "mlp", "cnn", "transformer", "gcn")) else "Advanced Engine"))
                            ),
                            "Predicted Dominant Outcome": f"{top_c} ({float(vc.iloc[0]) * 100.0:.1f}%)",
                            "Class Breakdown": ", ".join(f"{k}: {v * 100:.1f}%" for k, v in vc.head(4).items()),
                        }
                    )

                target_summaries[t_col] = {
                    "task_type": "classification",
                    "dominant_future_class": dominant_cls,
                    "dominant_future_share_pct": round(dominant_share, 2),
                    "historical_share_pct": round(hist_dom_share, 2),
                    "avg_confidence_pct": round(avg_conf, 2) if avg_conf is not None else None,
                    "avg_model_agreement_pct": round(float(np.mean(agreement_pct)), 1),
                    "predicted_distribution": {k: round(v * 100.0, 2) for k, v in pred_counts.items()},
                    "historical_distribution": {k: round(v * 100.0, 2) for k, v in hist_counts.items()},
                    "baseline_model_used": baseline_name_used,
                    "neural_model_used": neural_name_used,
                    "engines_compared": len(all_engine_preds),
                }
                executive_outlook.append(
                    f"Target `{t_col}` (Classification): Most likely future outcome is `{dominant_cls}` "
                    f"({dominant_share:.1f}% of incoming records vs. {hist_dom_share:.1f}% historical base rate), "
                    + (f"mean confidence {avg_conf:.1f}%, " if avg_conf is not None else "")
                    + f"cross-engine agreement {float(np.mean(agreement_pct)):.1f}% across {len(all_engine_preds)} model(s)."
                )
            else:
                prim_f = np.asarray(primary_preds, dtype=float)
                num_matrix = np.column_stack([np.asarray(arr, dtype=float) for arr in all_engine_preds.values()])
                consensus_mean = np.mean(num_matrix, axis=1)
                spread_std = np.std(num_matrix, axis=1) if num_matrix.shape[1] > 1 else np.zeros(n_rows, dtype=float)
                # Use model spread or 10% of historical target std as conservative uncertainty half-width
                hist_vals = pd.to_numeric(hist_s, errors="coerce").dropna().to_numpy(dtype=float)
                hist_mean = float(np.mean(hist_vals)) if len(hist_vals) else 0.0
                hist_median = float(np.median(hist_vals)) if len(hist_vals) else 0.0
                hist_std = float(np.std(hist_vals, ddof=1)) if len(hist_vals) > 1 else 0.0
                half_w = np.maximum(1.96 * spread_std, 0.25 * hist_std)

                out[f"consensus_{t_col}"] = np.round(consensus_mean, 4)
                out[f"future_{t_col}_low"] = np.round(prim_f - half_w, 4)
                out[f"future_{t_col}_high"] = np.round(prim_f + half_w, 4)
                if len(hist_vals) > 0:
                    pct_ranks = [round(float(np.mean(hist_vals <= v) * 100.0), 1) for v in prim_f]
                    out[f"historical_pct_rank_{t_col}"] = pct_ranks
                    # Tail risk comparison using 5% / 95% empirical quantiles (aligned with Studio 03 VaR)
                    tail_a = float((risk_report or {}).get("tail_alpha", 0.05))
                    q_lo, q_hi = float(np.quantile(hist_vals, tail_a)), float(np.quantile(hist_vals, 1.0 - tail_a))
                    out[f"tail_alert_{t_col}"] = [
                        "Upper Tail Risk" if v >= q_hi else ("Lower Tail Risk" if v <= q_lo else "Normal Range")
                        for v in prim_f
                    ]

                if baseline_preds is not None:
                    out[f"delta_vs_baseline_{t_col}"] = np.round(prim_f - np.asarray(baseline_preds, dtype=float), 4)

                fut_mean = float(np.mean(prim_f))
                fut_median = float(np.median(prim_f))
                pct_delta = ((fut_mean - hist_mean) / max(abs(hist_mean), 1e-12) * 100.0) if len(hist_vals) else 0.0

                for m_name, p_arr in all_engine_preds.items():
                    pf = np.asarray(p_arr, dtype=float)
                    model_comparison_rows.append(
                        {
                            "Target": t_col,
                            "Engine": m_name,
                            "Role": (
                                "Champion"
                                if m_name == str(getattr(primary_model, "name", "Champion"))
                                else ("Baseline" if "baseline" in m_name.lower() else ("Neural Network" if any(k in m_name.lower() for k in ("deep", "mlp", "cnn", "transformer", "gcn")) else "Advanced Engine"))
                            ),
                            "Predicted Mean": round(float(np.mean(pf)), 4),
                            "Predicted Median": round(float(np.median(pf)), 4),
                            "Predicted Min": round(float(np.min(pf)), 4),
                            "Predicted Max": round(float(np.max(pf)), 4),
                        }
                    )

                target_summaries[t_col] = {
                    "task_type": "regression",
                    "future_mean": round(fut_mean, 4),
                    "future_median": round(fut_median, 4),
                    "future_min": round(float(np.min(prim_f)), 4),
                    "future_max": round(float(np.max(prim_f)), 4),
                    "historical_mean": round(hist_mean, 4),
                    "historical_median": round(hist_median, 4),
                    "delta_vs_historical_pct": round(pct_delta, 2),
                    "avg_uncertainty_half_width": round(float(np.mean(half_w)), 4),
                    "baseline_model_used": baseline_name_used,
                    "neural_model_used": neural_name_used,
                    "engines_compared": len(all_engine_preds),
                }
                direction = "above" if pct_delta >= 0 else "below"
                executive_outlook.append(
                    f"Target `{t_col}` (Regression): Expected future mean is **{fut_mean:.4f}** "
                    f"(range [{float(np.min(prim_f)):.4f}, {float(np.max(prim_f)):.4f}]), which is "
                    f"**{abs(pct_delta):.1f}% {direction}** the analyzed historical mean ({hist_mean:.4f}) "
                    f"across {len(all_engine_preds)} engine(s)."
                )

        n_ood_rows = int(np.sum(ood_counts > 0))
        if n_ood_rows > 0:
            executive_outlook.append(
                f"Distribution-Shift Warning: {n_ood_rows}/{n_rows} incoming record(s) have one or more feature values "
                "outside the historical training range or >3σ from the historical mean — interpret predictions with caution."
            )
        else:
            executive_outlook.append(
                f"Statistical Alignment: All {n_rows} incoming record(s) fall within the analyzed historical feature distributions."
            )

        return FuturePredictionResult(
            scored_df=out,
            target_summaries=target_summaries,
            schema_report=schema_report,
            feature_shift_table=feature_shift_table,
            model_comparison_rows=model_comparison_rows,
            executive_outlook=executive_outlook,
            source_type=source_type,
            n_rows_scored=n_rows,
        )
