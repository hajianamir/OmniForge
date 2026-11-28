"""Leakage-Safe Data Splitting, Cross-Validation & sklearn Pipeline/ColumnTransformer Builder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PowerTransformer, RobustScaler, StandardScaler

from omniforge.core.exceptions import DataValidationError
from omniforge.core.logging import get_logger
from omniforge.core.task_inference import infer_supervised_task
from omniforge.core.types import TaskType
from omniforge.data.cpcv import iter_cpcv_splits

logger = get_logger(__name__)

SplitStrategyType = Literal["auto", "stratified", "random", "group", "time_series"]


class SplitConfig(BaseModel):
    """Configuration governing leakage-safe train/validation/test splitting and cross-validation."""

    strategy: SplitStrategyType = "auto"
    test_size: float = Field(default=0.20, gt=0.05, lt=0.60)
    val_size: float = Field(default=0.0, ge=0.0, lt=0.40)
    cv_folds: int = Field(default=5, ge=2, le=20)
    group_col: str | None = None
    time_col: str | None = None
    gap: int = Field(default=0, ge=0)
    regression_stratify_bins: int = Field(default=5, ge=2, le=20)
    strict_duplicates: bool = False
    random_seed: int = 42


class LeakageAuditReport(BaseModel):
    """Formal verification report proving absence (or exact source) of split-level data leakage."""

    passed: bool
    strategy_used: str
    n_train: int
    n_val: int = 0
    n_test: int
    index_overlap_count: int = 0
    group_overlap_count: int = 0
    temporal_order_valid: bool = True
    temporal_boundary: str = ""
    exact_duplicate_leak_count: int = 0
    checks: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@dataclass
class SplitResult:
    """Immutable split bundle returned by :class:`LeakageSafeSplitter`."""

    strategy_used: str
    train_indices: np.ndarray
    test_indices: np.ndarray
    val_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    train_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    test_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    val_df: pd.DataFrame | None = None
    audit_report: LeakageAuditReport | None = None
    config: SplitConfig = field(default_factory=SplitConfig)


class LeakageSafeSplitter:
    """Enterprise data splitter enforcing Stratified, Group, Chronological Time-Series, and Random protocols."""

    def __init__(self, config: SplitConfig | None = None) -> None:
        self.config = config or SplitConfig()

    def resolve_strategy(
        self,
        df: pd.DataFrame,
        target_col: str | None = None,
    ) -> str:
        """Resolve ``'auto'`` to the most statistically appropriate leakage-safe split strategy."""
        if self.config.strategy != "auto":
            return self.config.strategy
        if self.config.time_col and self.config.time_col in df.columns:
            return "time_series"
        if self.config.group_col and self.config.group_col in df.columns:
            return "group"
        if target_col and target_col in df.columns:
            ser = df[target_col].dropna()
            if len(ser) > 0:
                is_discrete = infer_supervised_task(ser) != TaskType.REGRESSION
                if is_discrete and int(ser.value_counts().min()) >= 2:
                    return "stratified"
        return "random"

    def split(
        self,
        df: pd.DataFrame,
        *,
        target_col: str | None = None,
    ) -> SplitResult:
        """Split ``df`` into disjoint Train / (optional Val) / Test partitions and run full leakage verification."""
        if not isinstance(df, pd.DataFrame) or len(df) < 5:
            raise DataValidationError("LeakageSafeSplitter requires a pandas DataFrame with at least 5 rows.")

        work = df.reset_index(drop=True)
        n = len(work)
        strategy = self.resolve_strategy(work, target_col=target_col)
        all_pos = np.arange(n, dtype=int)

        train_idx: np.ndarray
        val_idx = np.array([], dtype=int)
        test_idx: np.ndarray

        if strategy == "time_series":
            if self.config.time_col and self.config.time_col in work.columns:
                order = pd.to_datetime(work[self.config.time_col], errors="coerce", format="mixed")
                if order.notna().sum() >= n * 0.8:
                    sorted_pos = np.argsort(order.to_numpy(), kind="mergesort")
                else:
                    sorted_pos = np.argsort(work[self.config.time_col].to_numpy(), kind="mergesort")
            else:
                sorted_pos = all_pos

            n_test = max(1, int(round(n * self.config.test_size)))
            gap = min(self.config.gap, max(0, n - n_test - 2))
            test_start = n - n_test
            train_end = max(2, test_start - gap)

            if self.config.val_size > 0.0 and train_end >= 6:
                n_val = max(1, int(round(train_end * self.config.val_size)))
                val_start = max(2, train_end - n_val)
                train_idx = sorted_pos[: max(2, val_start - gap)]
                val_idx = sorted_pos[val_start:train_end]
            else:
                train_idx = sorted_pos[:train_end]
            test_idx = sorted_pos[test_start:]

        elif strategy == "group":
            gcol = self.config.group_col
            if not gcol or gcol not in work.columns:
                raise DataValidationError(
                    "Group split requires a valid 'group_col' column in the dataset.",
                    suggestions=["Select a patient_id, customer_id, machine_id, or entity grouping column."],
                )
            groups = work[gcol].fillna("__MISSING_GROUP__").astype(str).to_numpy()
            if len(np.unique(groups)) < 2:
                raise DataValidationError(
                    f"Group column '{gcol}' has fewer than 2 unique groups; cannot perform Group split."
                )
            gss = GroupShuffleSplit(n_splits=1, test_size=self.config.test_size, random_state=self.config.random_seed)
            tr_raw, te_raw = next(gss.split(all_pos, groups=groups))
            train_idx = np.asarray(tr_raw, dtype=int)
            test_idx = np.asarray(te_raw, dtype=int)
            if self.config.val_size > 0.0 and len(np.unique(groups[train_idx])) >= 2:
                rel_val = min(0.45, self.config.val_size / max(1.0 - self.config.test_size, 0.2))
                gss_v = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=self.config.random_seed + 1)
                sub_tr, sub_va = next(gss_v.split(train_idx, groups=groups[train_idx]))
                val_idx = train_idx[sub_va]
                train_idx = train_idx[sub_tr]

        elif strategy == "stratified":
            strat_vec: np.ndarray | None = None
            if target_col and target_col in work.columns:
                t_ser = work[target_col]
                if pd.api.types.is_numeric_dtype(t_ser) and t_ser.nunique(dropna=True) > 15:
                    # Quantile-binned stratification for continuous regression targets
                    filled = pd.to_numeric(t_ser, errors="coerce").fillna(float(pd.to_numeric(t_ser, errors="coerce").median()))
                    try:
                        bins = pd.qcut(filled, q=min(self.config.regression_stratify_bins, max(2, len(work) // 10)), duplicates="drop")
                        cand = bins.astype(str).to_numpy()
                        if pd.Series(cand).value_counts().min() >= 2:
                            strat_vec = cand
                    except Exception:
                        strat_vec = None
                else:
                    cand = t_ser.fillna("__NA__").astype(str).to_numpy()
                    if pd.Series(cand).value_counts().min() >= 2:
                        strat_vec = cand

            if strat_vec is not None:
                n_test_est = max(1, int(np.floor(n * self.config.test_size)))
                if n_test_est < len(np.unique(strat_vec)):
                    strat_vec = None

            tr_raw, te_raw = train_test_split(
                all_pos,
                test_size=self.config.test_size,
                random_state=self.config.random_seed,
                stratify=strat_vec,
            )
            train_idx = np.asarray(tr_raw, dtype=int)
            test_idx = np.asarray(te_raw, dtype=int)
            if self.config.val_size > 0.0 and len(train_idx) >= 6:
                rel_val = min(0.45, self.config.val_size / max(1.0 - self.config.test_size, 0.2))
                sub_strat = strat_vec[train_idx] if strat_vec is not None else None
                if sub_strat is not None and (
                    pd.Series(sub_strat).value_counts().min() < 2
                    or int(np.floor(len(train_idx) * rel_val)) < len(np.unique(sub_strat))
                ):
                    sub_strat = None
                sub_tr, sub_va = train_test_split(
                    train_idx, test_size=rel_val, random_state=self.config.random_seed + 1, stratify=sub_strat
                )
                train_idx = np.asarray(sub_tr, dtype=int)
                val_idx = np.asarray(sub_va, dtype=int)
        else:
            tr_raw, te_raw = train_test_split(
                all_pos,
                test_size=self.config.test_size,
                random_state=self.config.random_seed,
            )
            train_idx = np.asarray(tr_raw, dtype=int)
            test_idx = np.asarray(te_raw, dtype=int)
            if self.config.val_size > 0.0 and len(train_idx) >= 6:
                rel_val = min(0.45, self.config.val_size / max(1.0 - self.config.test_size, 0.2))
                sub_tr, sub_va = train_test_split(train_idx, test_size=rel_val, random_state=self.config.random_seed + 1)
                train_idx = np.asarray(sub_tr, dtype=int)
                val_idx = np.asarray(sub_va, dtype=int)

        train_df = work.iloc[train_idx].copy()
        test_df = work.iloc[test_idx].copy()
        val_df = work.iloc[val_idx].copy() if len(val_idx) > 0 else None

        audit = self.verify_no_leakage(
            work,
            train_idx=train_idx,
            test_idx=test_idx,
            val_idx=val_idx,
            strategy_used=strategy,
        )
        logger.info(
            "leakage_safe_split_created",
            strategy=strategy,
            n_train=len(train_idx),
            n_val=len(val_idx),
            n_test=len(test_idx),
            audit_passed=audit.passed,
        )
        return SplitResult(
            strategy_used=strategy,
            train_indices=train_idx,
            test_indices=test_idx,
            val_indices=val_idx,
            train_df=train_df,
            test_df=test_df,
            val_df=val_df,
            audit_report=audit,
            config=self.config,
        )

    def verify_no_leakage(
        self,
        df: pd.DataFrame,
        *,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        val_idx: np.ndarray | None = None,
        strategy_used: str = "random",
        strict_duplicates: bool | None = None,
    ) -> LeakageAuditReport:
        """Run 4 formal anti-leakage audits across Train, Validation, and Test partitions."""
        tr_set = set(int(i) for i in train_idx)
        te_set = set(int(i) for i in test_idx)
        va_set = set(int(i) for i in (val_idx if val_idx is not None else []))

        idx_overlap = len(tr_set & te_set) + len(tr_set & va_set) + len(va_set & te_set)
        checks: list[dict[str, Any]] = [
            {
                "check": "Disjoint Row Indices (Train ∩ Val ∩ Test = ∅)",
                "passed": idx_overlap == 0,
                "details": f"Overlapping indices: {idx_overlap}",
            }
        ]
        warnings: list[str] = []

        # Group isolation check across Train, Validation, and Test
        group_overlap = 0
        if self.config.group_col and self.config.group_col in df.columns:
            tr_g = set(df.iloc[train_idx][self.config.group_col].fillna("__MISSING_GROUP__").astype(str))
            te_g = set(df.iloc[test_idx][self.config.group_col].fillna("__MISSING_GROUP__").astype(str))
            va_g = (
                set(df.iloc[val_idx][self.config.group_col].fillna("__MISSING_GROUP__").astype(str))
                if val_idx is not None and len(val_idx) > 0
                else set()
            )
            group_overlap = len(tr_g & te_g) + len(tr_g & va_g) + len(va_g & te_g)
            checks.append(
                {
                    "check": f"Entity Group Isolation ('{self.config.group_col}')",
                    "passed": group_overlap == 0,
                    "details": f"Shared groups across Train/Val/Test: {group_overlap}",
                }
            )
            if group_overlap > 0:
                warnings.append(
                    f"Group leakage detected: {group_overlap} entity group(s) in '{self.config.group_col}' appear across split boundaries."
                )

        # Temporal non-inversion check across Train, Validation, and Test
        temporal_valid = True
        temporal_boundary = ""
        if self.config.time_col and self.config.time_col in df.columns:
            t_tr = pd.to_datetime(df.iloc[train_idx][self.config.time_col], errors="coerce", format="mixed").dropna()
            t_te = pd.to_datetime(df.iloc[test_idx][self.config.time_col], errors="coerce", format="mixed").dropna()
            t_va = (
                pd.to_datetime(df.iloc[val_idx][self.config.time_col], errors="coerce", format="mixed").dropna()
                if val_idx is not None and len(val_idx) > 0
                else pd.Series(dtype="datetime64[ns]")
            )
            if not t_tr.empty and not t_te.empty:
                max_tr, min_te = t_tr.max(), t_te.min()
                if not t_va.empty:
                    temporal_valid = bool((max_tr <= t_va.min()) and (t_va.max() <= min_te))
                    temporal_boundary = f"Train max={max_tr} | Val=[{t_va.min()}, {t_va.max()}] | Test min={min_te}"
                else:
                    temporal_valid = bool(max_tr <= min_te)
                    temporal_boundary = f"Train max={max_tr} | Test min={min_te}"
                checks.append(
                    {
                        "check": f"Chronological Order ('{self.config.time_col}': max(Train) <= min(Val) <= min(Test))",
                        "passed": temporal_valid,
                        "details": temporal_boundary,
                    }
                )
                if not temporal_valid:
                    warnings.append(f"Temporal look-ahead leakage: {temporal_boundary}.")

        # Cross-boundary exact duplicate row check across ALL split pairs (train∩val, train∩test, val∩test)
        dup_leaks = 0
        try:
            tr_hashes = set(pd.util.hash_pandas_object(df.iloc[train_idx], index=False).tolist())
            va_list = (
                pd.util.hash_pandas_object(df.iloc[val_idx], index=False).tolist()
                if (val_idx is not None and len(val_idx) > 0)
                else []
            )
            va_hashes = set(va_list)
            te_list = pd.util.hash_pandas_object(df.iloc[test_idx], index=False).tolist()

            dup_tr_va = sum(1 for h in va_list if h in tr_hashes)
            dup_tr_te = sum(1 for h in te_list if h in tr_hashes)
            dup_va_te = sum(1 for h in te_list if h in va_hashes)
            dup_leaks = int(dup_tr_va + dup_tr_te + dup_va_te)
            checks.append(
                {
                    "check": "Zero Cross-Split Exact Duplicate Rows",
                    "passed": dup_leaks == 0,
                    "details": (
                        f"Cross-split identical rows: total={dup_leaks} "
                        f"(train∩val={dup_tr_va}, train∩test={dup_tr_te}, val∩test={dup_va_te})"
                    ),
                }
            )
            if dup_leaks > 0:
                warnings.append(
                    f"{dup_leaks} cross-split duplicate row(s) detected "
                    f"(train∩val={dup_tr_va}, train∩test={dup_tr_te}, val∩test={dup_va_te}); "
                    "run deduplication before splitting."
                )
        except Exception as exc:
            logger.debug("duplicate_hash_audit_skipped", error=str(exc))

        enforce_strict_dups = self.config.strict_duplicates if strict_duplicates is None else bool(strict_duplicates)
        passed = (
            (idx_overlap == 0)
            and (group_overlap == 0)
            and temporal_valid
            and (dup_leaks == 0 if enforce_strict_dups else True)
        )
        return LeakageAuditReport(
            passed=passed,
            strategy_used=strategy_used,
            n_train=len(train_idx),
            n_val=len(va_set),
            n_test=len(test_idx),
            index_overlap_count=idx_overlap,
            group_overlap_count=group_overlap,
            temporal_order_valid=temporal_valid,
            temporal_boundary=temporal_boundary,
            exact_duplicate_leak_count=dup_leaks,
            checks=checks,
            warnings=warnings,
        )

    def build_cv_splitter(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        *,
        is_classification: bool,
        groups: np.ndarray | None = None,
    ) -> tuple[Any, np.ndarray | None]:
        """Construct the appropriate leakage-safe scikit-learn CV splitter matching ``config.strategy``."""
        n = len(X)
        strategy = self.config.strategy
        if strategy == "auto":
            if self.config.time_col and self.config.time_col in X.columns:
                strategy = "time_series"
            elif groups is not None:
                strategy = "group"
            elif is_classification:
                strategy = "stratified"
            else:
                strategy = "random"

        if strategy == "time_series":
            n_splits = max(2, min(self.config.cv_folds, max(2, n // 10)))
            gap = min(self.config.gap, max(0, (n // (n_splits + 1)) - 2))
            return TimeSeriesSplit(n_splits=n_splits, gap=gap), None

        if strategy == "group":
            if groups is None:
                raise DataValidationError("Group cross-validation requested (strategy='group') but groups=None.")
            uniq_g = len(np.unique(groups))
            n_splits = max(2, min(self.config.cv_folds, uniq_g))
            return GroupKFold(n_splits=n_splits), groups

        if strategy == "stratified" and is_classification:
            min_cls = int(pd.Series(y).value_counts().min()) if len(y) > 0 else 0
            n_splits = max(2, min(self.config.cv_folds, min_cls)) if min_cls >= 2 else 2
            if min_cls >= 2:
                return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.config.random_seed), None

        n_splits = max(2, min(self.config.cv_folds, max(2, n // 3)))
        return KFold(n_splits=n_splits, shuffle=True, random_state=self.config.random_seed), None


class LeakageSafePipelineBuilder:
    """Constructs genuine sklearn Pipeline + ColumnTransformer bundles guaranteeing zero fold leakage."""

    @staticmethod
    def build_preprocessor(
        X: pd.DataFrame,
        *,
        numeric_imputation: str = "median",
        categorical_imputation: str = "most_frequent",
        scaler: Literal["standard", "robust", "none"] = "standard",
        power_transform: bool = False,
        categorical_encoding: Literal["onehot", "ordinal"] = "onehot",
        max_onehot_categories: int = 25,
    ) -> ColumnTransformer:
        """Build a fold-safe ``ColumnTransformer`` handling numeric & categorical features cleanly."""
        num_cols = list(X.select_dtypes(include=[np.number]).columns)
        cat_cols = [c for c in X.columns if c not in num_cols and not pd.api.types.is_datetime64_any_dtype(X[c])]

        transformers: list[tuple[str, Any, list[str]]] = []
        if num_cols:
            num_steps: list[tuple[str, Any]] = [
                ("imputer", SimpleImputer(strategy=numeric_imputation if numeric_imputation in ("mean", "median", "most_frequent", "constant") else "median"))
            ]
            if power_transform:
                num_steps.append(("power", PowerTransformer(method="yeo-johnson", standardize=False)))
            if scaler == "standard":
                num_steps.append(("scaler", StandardScaler()))
            elif scaler == "robust":
                num_steps.append(("scaler", RobustScaler()))
            transformers.append(("num", Pipeline(num_steps), num_cols))

        if cat_cols:
            cat_imp_strat = categorical_imputation if categorical_imputation in ("most_frequent", "constant") else "most_frequent"
            if categorical_encoding == "onehot":
                encoder: Any = OneHotEncoder(
                    handle_unknown="ignore",
                    max_categories=max_onehot_categories,
                    sparse_output=False,
                )
            else:
                encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            cat_pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy=cat_imp_strat, fill_value="MISSING")),
                    ("encoder", encoder),
                ]
            )
            transformers.append(("cat", cat_pipe, cat_cols))

        if not transformers:
            raise DataValidationError("No usable numeric or categorical predictor columns found for ColumnTransformer.")

        return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)

    @classmethod
    def build_ml_pipeline(
        cls,
        X: pd.DataFrame,
        estimator: Any,
        *,
        task_type: TaskType,
        numeric_imputation: str = "median",
        scaler: Literal["standard", "robust", "none"] = "standard",
        power_transform: bool = False,
        select_k_features: int | None = None,
        selection_score_func: Literal["mutual_info", "f_stat"] = "f_stat",
    ) -> Pipeline:
        """Assemble a complete deployable sklearn ``Pipeline`` (ColumnTransformer -> optional SelectKBest -> Estimator)."""
        preprocessor = cls.build_preprocessor(
            X,
            numeric_imputation=numeric_imputation,
            scaler=scaler,
            power_transform=power_transform,
        )
        steps: list[tuple[str, Any]] = [("preprocess", preprocessor)]

        if select_k_features is not None and select_k_features > 0:
            is_clf = task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)
            if selection_score_func == "mutual_info":
                s_fn = mutual_info_classif if is_clf else mutual_info_regression
            else:
                s_fn = f_classif if is_clf else f_regression
            steps.append(("select", SelectKBest(score_func=s_fn, k=min(select_k_features, max(1, X.shape[1])))))

        steps.append(("model", estimator))
        return Pipeline(steps)

    @classmethod
    def cross_validate_leakage_free(
        cls,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        estimator: Any,
        *,
        task_type: TaskType,
        split_config: SplitConfig | None = None,
        groups: np.ndarray | None = None,
        scaler: Literal["standard", "robust", "none"] = "standard",
        select_k_features: int | None = None,
    ) -> dict[str, Any]:
        """Run strict fold-isolated cross-validation where preprocessing & feature selection fit inside every fold."""
        cfg = split_config or SplitConfig()
        splitter = LeakageSafeSplitter(cfg)
        is_clf = task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)
        cv_obj, cv_groups = splitter.build_cv_splitter(X, y, is_classification=is_clf, groups=groups)

        pipe = cls.build_ml_pipeline(
            X,
            estimator,
            task_type=task_type,
            scaler=scaler,
            select_k_features=select_k_features,
        )
        scoring = "f1_weighted" if is_clf else "neg_root_mean_squared_error"
        res = cross_validate(
            pipe,
            X,
            y,
            cv=cv_obj,
            groups=cv_groups,
            scoring=scoring,
            return_train_score=True,
            error_score="raise",
        )
        raw_test = np.asarray(res["test_score"], dtype=float)
        raw_train = np.asarray(res["train_score"], dtype=float)
        if not is_clf:
            # Flip neg_rmse back to positive RMSE
            raw_test = -raw_test
            raw_train = -raw_train

        return {
            "metric": "f1_weighted" if is_clf else "rmse",
            "cv_strategy": type(cv_obj).__name__,
            "n_splits": int(len(raw_test)),
            "test_scores": [round(float(v), 6) for v in raw_test],
            "test_mean": round(float(np.mean(raw_test)), 6),
            "test_std": round(float(np.std(raw_test)), 6),
            "train_mean": round(float(np.mean(raw_train)), 6),
            "overfit_gap": round(float(np.mean(raw_train) - np.mean(raw_test)) if is_clf else float(np.mean(raw_test) - np.mean(raw_train)), 6),
            "pipeline_template": pipe,
        }


class LegacyCombinatorialPurgedKFold:
    """Historical list-returning CPCV adapter; new callers should use evaluation.validation.

    Preserves the legacy purge-plus-embargo policy and can return empty training sets.

    Generates all ``C(n_splits, n_test_splits)`` combinations of test folds,
    applying a symmetric ``purge_gap`` around each test block and an ``embargo_pct``
    buffer immediately following each test block to prevent temporal label leakage.
    """

    def __init__(
        self,
        n_splits: int = 6,
        n_test_splits: int = 2,
        purge_gap: int = 0,
        embargo_pct: float = 0.0,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2.")
        if n_test_splits < 1 or n_test_splits >= n_splits:
            raise ValueError("n_test_splits must satisfy 1 <= n_test_splits < n_splits.")
        if purge_gap < 0:
            raise ValueError("purge_gap must be >= 0.")
        if not 0.0 <= embargo_pct < 1.0:
            raise ValueError("embargo_pct must be in [0.0, 1.0).")
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct

    def get_n_splits(
        self,
        X: Any = None,
        y: Any = None,
        groups: Any = None,
    ) -> int:
        from math import comb

        return comb(self.n_splits, self.n_test_splits)

    def split(
        self,
        X: Any,
        y: Any = None,
        groups: Any = None,
        *,
        eval_times: np.ndarray | pd.Series | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Compatibility adapter for the historical symmetric purge and list return type."""
        if len(X) < self.n_splits:
            raise ValueError(f"Cannot have n_splits={self.n_splits} greater than n_samples={len(X)}.")
        embargo = int(np.ceil(len(X) * self.embargo_pct))
        return list(iter_cpcv_splits(
            X, n_splits=self.n_splits, n_test_groups=self.n_test_splits, purge_gap=self.purge_gap,
            post_buffer=self.purge_gap + embargo, eval_times=eval_times, include_empty=True,
        ))


# Backward-compatible import (including existing serialized class references).
CombinatorialPurgedKFold = LegacyCombinatorialPurgedKFold
