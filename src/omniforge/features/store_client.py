"""Unified Feature Store Client supporting LocalFeatureStore and external Feast repositories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from omniforge.core.logging import get_logger
from omniforge.core.utils import is_package_available
from omniforge.data.feature_store import FeatureEntity, FeatureViewSpec, LocalFeatureStore

logger = get_logger(__name__)


class FeatureStoreClient:
    """High-level client bridging OmniForge feature pipelines with local or Feast feature stores."""

    def __init__(
        self,
        repo_path: str | Path = "./.omniforge_workspace/feature_store",
        *,
        use_feast_backend: bool = False,
    ) -> None:
        self.repo_path = Path(repo_path)
        self._local = LocalFeatureStore(self.repo_path)
        self._feast: Any = None
        if use_feast_backend and is_package_available("feast"):
            try:
                from feast import FeatureStore

                self._feast = FeatureStore(repo_path=str(self.repo_path))
                logger.info("feast_backend_connected", repo_path=str(self.repo_path))
            except Exception as exc:
                logger.warning("feast_backend_fallback", reason=str(exc))

    def register_entity(self, name: str, join_key: str, description: str = "") -> FeatureEntity:
        """Register an entity join key."""
        return self._local.register_entity(
            FeatureEntity(name=name, join_key=join_key, description=description)
        )

    def publish_features(
        self,
        view_name: str,
        df: pd.DataFrame,
        *,
        join_keys: list[str],
        timestamp_column: str | None = None,
        description: str = "",
        materialize_online: bool = True,
    ) -> FeatureViewSpec:
        """Register and persist an engineered feature set into offline & online stores."""
        return self._local.register_feature_view(
            name=view_name,
            df=df,
            join_keys=join_keys,
            timestamp_column=timestamp_column,
            description=description,
            materialize_online=materialize_online,
        )

    def fetch_training_features(
        self,
        entity_df: pd.DataFrame,
        feature_refs: list[str],
        *,
        entity_timestamp_col: str | None = None,
    ) -> pd.DataFrame:
        """Retrieve point-in-time historical features for model training."""
        if self._feast is not None:
            return self._feast.get_historical_features(
                entity_df=entity_df, features=feature_refs
            ).to_df()
        return self._local.get_historical_features(
            entity_df=entity_df,
            feature_refs=feature_refs,
            entity_timestamp_col=entity_timestamp_col,
        )

    def fetch_online_features(
        self,
        view_name: str,
        entity_rows: list[dict[str, Any]],
        features: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve low-latency online features for inference."""
        if self._feast is not None and features:
            refs = [f"{view_name}:{f}" for f in features]
            resp = self._feast.get_online_features(features=refs, entity_rows=entity_rows).to_dict()
            keys = list(resp.keys())
            n = len(resp[keys[0]]) if keys else 0
            return [{k: resp[k][i] for k in keys} for i in range(n)]
        return self._local.get_online_features(view_name, entity_rows, features=features)

    def list_views(self) -> list[FeatureViewSpec]:
        """List all available feature views."""
        return self._local.list_feature_views()
