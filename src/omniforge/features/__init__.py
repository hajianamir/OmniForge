"""Feature engineering, selection, dense embeddings, and feature store integration."""

from omniforge.features.embeddings import EmbeddingGenerator
from omniforge.features.engineering import FeatureConfig, FeatureEngineeringPipeline
from omniforge.features.selection import FeatureSelectionReport, FeatureSelector
from omniforge.features.store_client import FeatureStoreClient

__all__ = [
    "EmbeddingGenerator",
    "FeatureConfig",
    "FeatureEngineeringPipeline",
    "FeatureSelectionReport",
    "FeatureSelector",
    "FeatureStoreClient",
]
