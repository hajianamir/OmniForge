"""Unit tests for Data Ingestion, Cleaning, Validation, Quality, Versioning, Feature Store, Synthetic & Multimodal."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from omniforge.data.cleaning import CleaningConfig, DataCleaner
from omniforge.data.feature_store import LocalFeatureStore
from omniforge.data.ingestion import DataIngestionEngine
from omniforge.data.multimodal import MultimodalProcessor
from omniforge.data.quality import DataQualityProfiler
from omniforge.data.synthetic import SyntheticDataGenerator
from omniforge.data.validation import DataValidator, ValidationRule
from omniforge.data.versioning import DatasetVersionManager


def test_ingestion_formats_and_sql(classification_df: pd.DataFrame, workspace_tmp: Path) -> None:
    eng = DataIngestionEngine()
    csv_p = workspace_tmp / "data.csv"
    pq_p = workspace_tmp / "data.parquet"
    json_p = workspace_tmp / "data.json"
    db_p = workspace_tmp / "test.db"

    classification_df.to_csv(csv_p, index=False)
    classification_df.to_parquet(pq_p, index=False)
    classification_df.to_json(json_p, orient="records")
    with sqlite3.connect(db_p) as conn:
        classification_df.to_sql("customers", conn, index=False, if_exists="replace")

    for src in (csv_p, pq_p, json_p, classification_df):
        res = eng.ingest(src)
        assert res.metadata.n_rows == len(classification_df)
        assert len(res.metadata.checksum_sha256) == 64

    sql_res = eng.ingest(f"sqlite:///{db_p}", query="SELECT * FROM customers LIMIT 25")
    assert len(sql_res.data) == 25


def test_excel_multisheet_discovery_single_concat_and_merge(workspace_tmp: Path) -> None:
    """Verify multi-sheet Excel listing, single-sheet import, row stacking, and key-column merging."""
    eng = DataIngestionEngine()
    xlsx_p = workspace_tmp / "multi_sheet.xlsx"

    sheet_sales = pd.DataFrame({"customer_id": [1, 2, 3], "revenue": [100.0, 200.0, 300.0]})
    sheet_q2 = pd.DataFrame({"customer_id": [4, 5], "revenue": [400.0, 500.0]})
    sheet_profile = pd.DataFrame({"customer_id": [1, 2, 3], "segment": ["SMB", "ENT", "SMB"]})

    with pd.ExcelWriter(xlsx_p, engine="openpyxl") as writer:
        sheet_sales.to_excel(writer, sheet_name="Sales_Q1", index=False)
        sheet_q2.to_excel(writer, sheet_name="Sales_Q2", index=False)
        sheet_profile.to_excel(writer, sheet_name="Profiles", index=False)

    raw_bytes = xlsx_p.read_bytes()
    sheets_from_path = eng.list_excel_sheets(xlsx_p)
    sheets_from_bytes = eng.list_excel_sheets(raw_bytes)
    assert sheets_from_path == ["Sales_Q1", "Sales_Q2", "Profiles"]
    assert sheets_from_bytes == sheets_from_path

    # 1. Single chosen sheet
    res_single = eng.ingest_excel_sheets(raw_bytes, sheets=["Sales_Q2"], combine_mode="single")
    assert len(res_single.data) == 2
    assert list(res_single.data["customer_id"]) == [4, 5]

    # 2. Concat / Stack multiple sheets vertically with origin sheet tag
    res_concat = eng.ingest_excel_sheets(
        raw_bytes, sheets=["Sales_Q1", "Sales_Q2"], combine_mode="concat", add_sheet_column=True
    )
    assert len(res_concat.data) == 5
    assert "_source_sheet" in res_concat.data.columns
    assert set(res_concat.data["_source_sheet"]) == {"Sales_Q1", "Sales_Q2"}

    # 3. Merge / Join multiple sheets horizontally on shared key column
    res_merge = eng.ingest_excel_sheets(
        raw_bytes, sheets=["Sales_Q1", "Profiles"], combine_mode="merge", merge_on="customer_id", merge_how="inner"
    )
    assert len(res_merge.data) == 3
    assert {"customer_id", "revenue", "segment"}.issubset(set(res_merge.data.columns))

    # 4. Direct ingest with sheet_name=None safely returns a stacked DataFrame (never raw dict crash)
    res_all = eng.ingest(xlsx_p, sheet_name=None)
    assert isinstance(res_all.data, pd.DataFrame)
    assert "_source_sheet" in res_all.data.columns
    assert len(res_all.data) == 8


def test_cleaning_validation_quality_versioning_feature_store(
    classification_df: pd.DataFrame, workspace_tmp: Path
) -> None:
    dirty = pd.concat([classification_df, classification_df.iloc[[0]]], ignore_index=True)
    dirty.loc[1, "feature_0"] = 9999.0

    quality = DataQualityProfiler().profile(dirty)
    assert 0.0 <= quality.overall_score <= 100.0
    assert set(quality.dimensions.keys()) == {
        "completeness",
        "uniqueness",
        "validity",
        "consistency",
        "timeliness",
        "accuracy",
    }

    cleaner = DataCleaner(CleaningConfig(numeric_imputation="median", outlier_method="iqr", outlier_action="clip"))
    cleaned, rep = cleaner.fit_transform(dirty)
    assert rep.exact_duplicates_removed >= 1
    assert rep.missing_values_after == 0
    assert cleaned["feature_0"].max() < 9999.0

    transformed, _ = cleaner.transform(classification_df.head(15))
    assert int(transformed.isna().sum().sum()) == 0

    rules = [
        ValidationRule(name="target_binary", rule_type="in_set", column="target", params={"values": [0, 1]}),
        ValidationRule(name="feature_0_non_null", rule_type="not_null", column="feature_0"),
    ]
    vrep = DataValidator(rules).validate(cleaned)
    assert vrep.passed is True

    vm = DatasetVersionManager(store_dir=workspace_tmp / "versions")
    v1 = vm.commit(cleaned, "customers", message="clean baseline")
    loaded_df, v_meta = vm.checkout("customers", v1.version_id)
    assert len(loaded_df) == len(cleaned)
    assert v_meta.checksum_sha256 == v1.checksum_sha256

    fs = LocalFeatureStore(repo_path=workspace_tmp / "fs_repo")
    fs_df = cleaned.head(20).copy()
    fs_df["entity_id"] = [f"cust_{i}" for i in range(len(fs_df))]
    fs.register_feature_view("customer_features", fs_df, join_keys=["entity_id"])
    online = fs.get_online_features("customer_features", [{"entity_id": "cust_0"}, {"entity_id": "cust_1"}])
    assert len(online) == 2


def test_synthetic_copula_and_multimodal(classification_df: pd.DataFrame) -> None:
    syn = SyntheticDataGenerator(random_seed=7).fit_generate_copula(classification_df.dropna(), n_samples=60)
    assert len(syn) == 60
    assert list(syn.columns) == list(classification_df.columns)

    mp = MultimodalProcessor()
    img_feats = mp.extract_image_features(np.random.default_rng(0).integers(0, 255, size=(32, 32, 3), dtype=np.uint8))
    assert "img_width" in img_feats and img_feats["img_width"] == 32.0
    aud_feats = mp.extract_audio_features(np.sin(np.linspace(0, 20 * np.pi, 1600)), sample_rate=16000)
    assert "rms_energy" in aud_feats or len(aud_feats) >= 3
    geo_feats = mp.extract_geospatial_features(pd.DataFrame({"latitude": [50.11, 52.52], "longitude": [8.68, 13.40]}))
    assert len(geo_feats) >= 2
