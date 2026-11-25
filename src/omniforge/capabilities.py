"""OmniForge capability packs — the single source of truth for the Windows installer.

Three tiers drive every downstream artifact (no duplication, no drift):

``SIMPLE``      → what the beginner user gets. Studio 01-05 + Universal Diagnostics work;
                  every heavy backend uses its documented NumPy / scikit-learn fallback.
``ADVANCED``    → SIMPLE + the full modeling, MLOps, reporting and connector stack.
``EVERYTHING``  → ADVANCED + the optional native backends (torch, xgboost/catboost, CV, NLP, ...).

Consumers:
  * ``installer/provision.py``      (venv bootstrap + ``uv pip install``)
  * ``omniforge.capabilities``      (runtime probe → ``/api/system/capabilities``)
  * ``frontend``                    (badges + "Install this pack" buttons)
  * CI                              (``locks/*.lock`` generation per tier)
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Final, Literal

Tier = Literal["simple", "advanced", "everything"]

# ---------------------------------------------------------------------------
# Packs. ``dists`` are *PyPI distribution names* (not import names).
# ``imports`` are used for the runtime availability probe.
# ``unlocks`` is human-facing copy for the installer wizard / capability badges.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pack:
    """One installable dependency pack."""

    id: str
    title: str
    description: str
    dists: tuple[str, ...]
    imports: tuple[str, ...]
    unlocks: str = ""
    extras: tuple[str, ...] = field(default_factory=tuple)
    # ``core`` packs are mandatory for their tier, ``optional`` ones are selectable.
    kind: Literal["core", "optional"] = "core"
    # Rough installed weight shown to the user (MB, unpacked) — refreshed by CI.
    approx_mb: int = 0


CORE_RUNTIME = Pack(
    id="core-runtime",
    title="Core runtime (required)",
    description=(
        "Scientific Python stack, settings/logging, CLI and — in Advanced — the FastAPI workbench server. "
        "Without this nothing runs."
    ),
    dists=(
        "numpy>=2.5.5",
        "pandas>=3.1.1",
        "scipy>=1.18.1",
        "scikit-learn>=1.9.1",
        "joblib>=1.6.0",
        "python-dateutil>=4.0.0",
        "networkx>=3.7",
        "pydantic>=2.14.3",
        "pydantic-settings>=2.16.3",
        "PyYAML>=6.0.3",
        "structlog>=26.1.0",
        "psutil>=7.2.2",
        "httpx>=0.28.1",
        "Jinja2>=3.1.6",
        "typer>=0.29.1",
        "rich>=15.0.0",
        "plotly>=7.1.0",
        "pillow>=13.0.0",
        "defusedxml>=0.7.1",
    ),
    imports=(
        "numpy", "pandas", "scipy", "sklearn", "joblib", "dateutil", "networkx",
        "pydantic", "pydantic_settings", "yaml", "structlog", "psutil", "httpx",
        "jinja2", "typer", "rich", "plotly", "PIL", "defusedxml",
    ),
    unlocks="Data ingestion, cleaning, 6-D quality, EDA, feature engineering, clustering, "
            "baselines, classical ML and the AI Copilot (offline planner).",
    approx_mb=430,
)


API_SERVER = Pack(
    id="api-server",
    title="Workbench server (required in Advanced)",
    description="FastAPI + uvicorn + auth/crypto: serves the React SPA, REST API and the CLI server modes.",
    dists=(
        "fastapi>=0.163.1",
        "uvicorn[standard]>=0.54.0",
        "python-multipart>=0.0.33",
        "pyjwt>=2.15.1",
        "cryptography>=50.0.2",
        "prometheus-client>=0.26.0",
    ),
    imports=("fastapi", "uvicorn", "multipart", "jwt", "cryptography", "prometheus_client"),
    unlocks="The full 14-studio React workbench, REST API, JWT/RBAC, encrypted artifacts.",
    approx_mb=90,
)


STATISTICS = Pack(
    id="statistics",
    title="Forecasting & advanced statistics",
    description="statsmodels: ARIMA/SARIMAX/ETS, hierarchical forecasting, causal inference estimators.",
    dists=("statsmodels>=0.15.0",),
    imports=("statsmodels",),
    unlocks="Studio 08 forecasting, causal inference (DML/IV-2SLS), hypothesis tests that need statsmodels.",
    approx_mb=60,
)


DATA_IO = Pack(
    id="data-io",
    title="Fast data connectors",
    description="pyarrow (Parquet/Feather/ORC/Avro), duckdb, SQLAlchemy (SQL/PG/MySQL/MSSQL), lxml, fsspec.",
    dists=(
        "pyarrow>=26.0.1",
        "duckdb>=2.0.0",
        "sqlalchemy>=2.1.8",
        "fsspec>=2026.9.0",
        "lxml>=6.1.3",
    ),
    imports=("pyarrow", "duckdb", "sqlalchemy", "fsspec", "lxml"),
    unlocks="Parquet/ORC/Avro/XML ingestion, the demo SQL workspace, live SQL scoring in Studio 14.",
    approx_mb=200,
)


REPORTING = Pack(
    id="reporting",
    title="Executive reporting & export",
    description="python-docx, python-pptx, reportlab, nbformat, openpyxl, matplotlib chart rendering.",
    dists=(
        "openpyxl>=3.1.5",
        "matplotlib>=3.11.2",
        "python-docx>=1.2.0",
        "python-pptx>=1.0.2",
        "reportlab>=5.1.0",
        "nbformat>=5.11.1",
    ),
    imports=("openpyxl", "matplotlib", "docx", "pptx", "reportlab", "nbformat"),
    unlocks="Studio 12: HTML/PDF/DOCX/PPTX/XLSX/IPYNB export plus the Docker bundle.",
    approx_mb=110,
)


BOOSTING = Pack(
    id="boosting",
    title="Gradient boosting (XGBoost / LightGBM / CatBoost)",
    description="Native boosted-tree libraries. Without them the same models run on scikit-learn's HistGradientBoosting.",
    dists=("xgboost>=3.4.1", "lightgbm>=4.7.0", "catboost>=1.2.10; python_version < '3.15'"),
    imports=("xgboost", "lightgbm", "catboost"),
    unlocks="Native XGB/LGBM/CatBoost speed & accuracy in Studio 03/06 (fallback: scikit-learn HGB).",
    kind="optional",
    approx_mb=260,
)


DEEP = Pack(
    id="deep",
    title="Deep learning (PyTorch)",
    description="PyTorch + Lightning. Without it every neural model uses the shipped NumPy implementations.",
    dists=("torch>=2.15.0", "lightning>=2.6.6"),
    imports=("torch", "lightning"),
    unlocks="GPU deep learning, TabNet, ResNet-tabular, FT-Transformer, torch forecasting backends "
            "(fallback: omniforge_numpy_* implementations, same API, CPU only).",
    kind="optional",
    approx_mb=2800,
)


VISION = Pack(
    id="vision",
    title="Computer vision",
    description="OpenCV + scikit-image: real image decoding, resizing, filters for Studio 09 Vision.",
    dists=("opencv-python>=5.0.0.93", "scikit-image>=0.26.0; python_version < '3.15'"),
    imports=("cv2", "skimage"),
    unlocks="Native image pipelines; without it the vision studio uses the Pillow/NumPy fallback.",
    kind="optional",
    approx_mb=180,
)


NLP = Pack(
    id="nlp",
    title="NLP toolkit",
    description="NLTK + spaCy + gensim: tokenization, embeddings, topic modeling.",
    dists=("nltk>=3.10.4", "spacy>=3.8.18", "gensim>=4.4.0; python_version < '3.14'"),
    imports=("nltk", "spacy", "gensim"),
    unlocks="Richer Studio 09 NLP; without it a scikit-learn TF-IDF path is used.",
    kind="optional",
    approx_mb=220,
)


FOUNDATION = Pack(
    id="foundation",
    title="Foundation models (transformers, CLIP, YOLO)",
    description="transformers, timm, sentence-transformers, open-clip-torch, ultralytics. Requires the Deep pack.",
    dists=(
        "transformers>=5.20.0",
        "timm>=1.0.31",
        "sentence-transformers>=6.3.0",
        "open-clip-torch>=4.2.0",
        "ultralytics>=8.4.0",
    ),
    imports=("transformers", "timm", "sentence_transformers", "open_clip", "ultralytics"),
    unlocks="Zero-shot vision, LLM summarization/NER, semantic embeddings.",
    kind="optional",
    approx_mb=1500,
)


AUTOML_BACKENDS = Pack(
    id="automl",
    title="Extra AutoML backends",
    description="Optuna, FLAML, AutoGluon, TPOT — additional search strategies behind the same AutoML API.",
    dists=("optuna>=5.0.0", "flaml>=2.8.0"),
    imports=("optuna", "flaml"),
    unlocks="Bayesian/FLAML search in Studio 06 (default GP-UCB + random/grid already included).",
    kind="optional",
    approx_mb=40,
)


RESPONSIBLE = Pack(
    id="responsible",
    title="Explainability & carbon tracking",
    description="SHAP for exact Shapley values, codecarbon for measured (not estimated) energy/CO2.",
    dists=("shap>=0.54.1", "codecarbon>=3.3.2"),
    imports=("shap", "codecarbon"),
    unlocks="SHAP plots in Studio 07 and measured carbon numbers (fallback: permutation importance + "
            "hardware-estimated carbon).",
    kind="optional",
    approx_mb=90,
)


ALL_PACKS: Final[tuple[Pack, ...]] = (
    CORE_RUNTIME, API_SERVER, STATISTICS, DATA_IO, REPORTING,
    BOOSTING, DEEP, VISION, NLP, FOUNDATION, AUTOML_BACKENDS, RESPONSIBLE,
)

PACKS_BY_ID: Final[dict[str, Pack]] = {p.id: p for p in ALL_PACKS}


# ---------------------------------------------------------------------------
# Tiers — what the installer preselects.
# ---------------------------------------------------------------------------

#: Simple profile: everything that Studio 01-05 + diagnostics needs, nothing that can be
#: replaced by a shipped fallback. Served by the bundled SPA, so no FastAPI needed.
TIER_SIMPLE: Final[tuple[str, ...]] = ("core-runtime", "statistics", "data-io", "reporting")

#: Advanced profile: adds the workbench server + the remaining required stack.
TIER_ADVANCED: Final[tuple[str, ...]] = TIER_SIMPLE + ("api-server",)

#: Everything: Advanced + every optional native backend.
TIER_EVERYTHING: Final[tuple[str, ...]] = TIER_ADVANCED + (
    "boosting", "deep", "vision", "nlp", "foundation", "automl", "responsible",
)

TIERS: Final[dict[str, tuple[str, ...]]] = {
    "simple": TIER_SIMPLE,
    "advanced": TIER_ADVANCED,
    "everything": TIER_EVERYTHING,
}

#: Packs that the user may additionally tick in the wizard (never required).
OPTIONAL_PACKS: Final[tuple[str, ...]] = tuple(p.id for p in ALL_PACKS if p.kind == "optional")

#: Packs the runtime cannot start without, per tier.
REQUIRED_BY_TIER: Final[dict[str, tuple[str, ...]]] = {
    "simple": ("core-runtime",),
    "advanced": ("core-runtime", "api-server"),
    "everything": ("core-runtime", "api-server"),
}

#: Feature → pack that upgrades it (drives the "Install X" buttons inside the app).
FEATURE_UPGRADES: Final[dict[str, tuple[str, str]]] = {
    "deep_learning": ("deep", "Native PyTorch deep learning"),
    "gradient_boosting": ("boosting", "Native XGBoost / LightGBM / CatBoost"),
    "computer_vision": ("vision", "Native OpenCV image pipeline"),
    "nlp": ("nlp", "spaCy / gensim NLP toolkit"),
    "foundation_models": ("foundation", "Transformers / CLIP / YOLO"),
    "shap_explainability": ("responsible", "SHAP explanations"),
    "measured_carbon": ("responsible", "codecarbon measured energy"),
    "extra_automl": ("automl", "Optuna / FLAML search"),
    "executive_reports": ("reporting", "PDF / DOCX / PPTX / XLSX export"),
    "fast_data_io": ("data-io", "Parquet / DuckDB / SQL connectors"),
}


def packs_for_tier(tier: Tier) -> tuple[Pack, ...]:
    """Return the ``Pack`` objects preselected for ``tier``."""
    return tuple(PACKS_BY_ID[pid] for pid in TIERS[tier])


def resolve_selection(tier: Tier, extra_packs: list[str] | tuple[str, ...] = ()) -> tuple[Pack, ...]:
    """Union of the tier's packs and any individually selected optional packs (deduplicated, stable order)."""
    chosen = list(TIERS[tier])
    for pid in extra_packs:
        if pid not in PACKS_BY_ID:
            raise KeyError(f"Unknown capability pack: {pid!r}")
        if pid not in chosen:
            chosen.append(pid)
    return tuple(PACKS_BY_ID[pid] for pid in chosen)


def requirements_for(packs: tuple[Pack, ...] | list[Pack]) -> list[str]:
    """Flatten packs to a deduplicated requirement list (first pin wins)."""
    seen: dict[str, str] = {}
    for pack in packs:
        for dist in pack.dists:
            name = _dist_name(dist)
            seen.setdefault(name, dist)
    return list(seen.values())


def imports_for(packs: tuple[Pack, ...] | list[Pack]) -> list[str]:
    """Flatten packs to the import names used by the runtime availability probe."""
    out: list[str] = []
    for pack in packs:
        for imp in pack.imports:
            if imp not in out:
                out.append(imp)
    return out


def _dist_name(spec: str) -> str:
    """Extract the normalized distribution name from a requirement specifier."""
    for sep in ("[", "<", ">", "=", "!", "~", ";", " "):
        spec = spec.split(sep, 1)[0]
    return spec.strip().lower().replace("_", "-")


def profile_summary(tier: Tier, extra_packs: list[str] | tuple[str, ...] = ()) -> dict[str, object]:
    """Human/CI-facing summary of a selection (ids, approvals, MB and import list)."""
    packs = resolve_selection(tier, extra_packs)
    return {
        "tier": tier,
        "packs": [p.id for p in packs],
        "approx_mb": sum(p.approx_mb for p in packs),
        "requirements": requirements_for(packs),
        "imports": imports_for(packs),
        "unlocks": {p.id: p.unlocks for p in packs if p.unlocks},
    }


def probe_runtime_capabilities() -> dict[str, object]:
    """Probe installed capability packs without eagerly importing heavy native libraries."""
    pack_status: dict[str, dict[str, object]] = {}
    available_packs: list[str] = []
    for pack in ALL_PACKS:
        missing_imports = [
            mod_name for mod_name in pack.imports if importlib.util.find_spec(mod_name) is None
        ]
        installed = len(missing_imports) == 0
        if installed:
            available_packs.append(pack.id)
        pack_status[pack.id] = {
            "id": pack.id,
            "title": pack.title,
            "description": pack.description,
            "kind": pack.kind,
            "approx_mb": pack.approx_mb,
            "installed": installed,
            "missing_imports": missing_imports,
            "unlocks": pack.unlocks,
        }

    features: dict[str, dict[str, object]] = {}
    for feat_key, (pack_id, label) in FEATURE_UPGRADES.items():
        p_info = pack_status[pack_id]
        features[feat_key] = {
            "pack_id": pack_id,
            "label": label,
            "native_available": bool(p_info["installed"]),
            "description": p_info["description"],
        }

    return {
        "available_packs": available_packs,
        "packs": pack_status,
        "features": features,
    }
