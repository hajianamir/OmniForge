"""Extensible Industry Domain Plugin SDK with entry-point discovery, feature engineering, KPIs, and compliance."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field

from omniforge.core.exceptions import DomainPluginError
from omniforge.core.logging import get_logger
from omniforge.data.validation import ValidationRule

logger = get_logger(__name__)


class DomainMetadata(BaseModel):
    """Metadata describing an industry domain plugin."""

    name: str
    display_name: str
    description: str
    version: str = "1.0.0"
    compliance_frameworks: list[str] = Field(default_factory=list)
    kpis: list[str] = Field(default_factory=list)
    recommended_models: list[str] = Field(default_factory=list)
    default_target: str = "target"
    config_file: str | None = None


class BaseDomainPlugin(ABC):
    """Abstract base class for all OmniForge industry plugins."""

    metadata: DomainMetadata

    def load_config(self, config_dir: str | Path = "configs/domains") -> dict[str, Any]:
        """Load the domain's YAML configuration if present on disk."""
        candidates = [
            Path(config_dir) / f"{self.metadata.name}.yaml",
            Path(__file__).resolve().parents[3] / "configs" / "domains" / f"{self.metadata.name}.yaml",
        ]
        for cand in candidates:
            if cand.exists():
                with cand.open("r", encoding="utf-8") as fh:
                    return yaml.safe_load(fh) or {}
        return {"domain": self.metadata.name}

    @abstractmethod
    def engineer_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific derived features safely without mutating the caller's DataFrame."""

    @abstractmethod
    def compute_domain_kpis(
        self, df: pd.DataFrame, *, predictions: np.ndarray | None = None
    ) -> dict[str, float]:
        """Compute industry-standard KPI metrics from ``df`` and optional model predictions."""

    def get_validation_rules(self) -> list[ValidationRule]:
        """Return industry-specific data validation expectations."""
        return []

    @abstractmethod
    def generate_sample_dataset(self, n_samples: int = 200, random_state: int = 42) -> pd.DataFrame:
        """Generate a realistic domain-specific synthetic dataset for onboarding, demos, and testing."""

    def summary(self) -> dict[str, Any]:
        """Return JSON-serializable plugin metadata summary."""
        return self.metadata.model_dump()


class DomainPluginRegistry:
    """Central registry managing built-in and externally installed industry domain plugins."""

    def __init__(self) -> None:
        self._plugins: dict[str, BaseDomainPlugin] = {}

    def register(self, plugin: BaseDomainPlugin) -> None:
        """Register a domain plugin instance by its canonical slug."""
        self._plugins[plugin.metadata.name.lower()] = plugin

    def get(self, name: str) -> BaseDomainPlugin:
        """Retrieve a registered domain plugin by slug."""
        key = name.lower().strip()
        if key not in self._plugins:
            raise DomainPluginError(
                f"Domain plugin '{name}' is not registered.",
                suggestions=[f"Available domains: {sorted(self._plugins.keys())}"],
            )
        return self._plugins[key]

    def list_plugins(self) -> list[BaseDomainPlugin]:
        """Return all registered domain plugins sorted alphabetically by slug."""
        return [self._plugins[k] for k in sorted(self._plugins.keys())]

    def list_names(self) -> list[str]:
        """Return sorted list of registered domain slugs."""
        return sorted(self._plugins.keys())

    def discover_entry_points(
        self, group: str = "omniforge.domains", *, skip_existing: bool = False
    ) -> int:
        """Dynamically load external plugins registered under setuptools entry points."""
        loaded = 0
        try:
            eps = importlib_metadata.entry_points(group=group)
            for ep in eps:
                try:
                    if skip_existing and ep.name.lower().strip() in self._plugins:
                        continue
                    plugin_obj = ep.load()
                    if (
                        inspect.isclass(plugin_obj)
                        and issubclass(plugin_obj, BaseDomainPlugin)
                        and plugin_obj is not BaseDomainPlugin
                        and not inspect.isabstract(plugin_obj)
                    ):
                        cls_meta = getattr(plugin_obj, "metadata", None)
                        if (
                            skip_existing
                            and isinstance(cls_meta, DomainMetadata)
                            and cls_meta.name.lower() in self._plugins
                        ):
                            continue
                        plugin_obj = plugin_obj()
                    if isinstance(plugin_obj, BaseDomainPlugin):
                        if skip_existing and plugin_obj.metadata.name.lower() in self._plugins:
                            continue
                        self.register(plugin_obj)
                        loaded += 1
                except Exception as exc:
                    logger.warning("domain_entrypoint_load_failed", entry_point=ep.name, reason=str(exc))
        except Exception as exc:
            logger.debug("domain_entrypoint_discovery_skipped", group=group, error=str(exc))
        return loaded


default_domain_registry = DomainPluginRegistry()
