"""Phase 4 verification: statistical mining FDR, feature-store TTL/erasure, and CycloneDX SBOM provenance."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from omniforge.core.sbom import generate_cyclonedx_sbom, verify_lockfiles_provenance
from omniforge.data.feature_store import LocalFeatureStore
from omniforge.models.mining.miner import AssociationRuleMiner


def _strong_association_tx() -> pd.DataFrame:
    """Transactions where eggs => bread is a strong, statistically significant rule.

    eggs co-occurs with bread in 60/100 txns; bread appears in 80/100, eggs in 60/100.
    Rule {eggs} => {bread}: conf = 1.0, lift = 1.25, hypergeometric p << 0.05.
    """
    rows: list[dict[str, object]] = []
    for tid in range(1, 61):  # 60: bread+milk+eggs
        rows.extend(
            [
                {"tid": tid, "item": "bread"},
                {"tid": tid, "item": "milk"},
                {"tid": tid, "item": "eggs"},
            ]
        )
    for tid in range(61, 81):  # 20: bread+milk only (no eggs)
        rows.extend([{"tid": tid, "item": "bread"}, {"tid": tid, "item": "milk"}])
    for tid in range(81, 101):  # 20: unrelated basket
        rows.extend([{"tid": tid, "item": "coffee"}, {"tid": tid, "item": "sugar"}])
    return pd.DataFrame(rows)


def test_association_rule_miner_fdr_fields() -> None:
    """mine_transactions returns hypergeometric p-values with BH FDR q-values and significance flags."""
    tx = _strong_association_tx()
    report = AssociationRuleMiner().mine_transactions(
        tx, transaction_col="tid", item_col="item", min_support=0.05, min_confidence=0.5, min_lift=1.0
    )
    assert report["n_rules"] > 0
    rules = report["rules"]
    # Every rule carries the Webb (2007) hypergeometric p-value and BH FDR outputs.
    for rule in rules:
        assert "p_value" in rule and isinstance(rule["p_value"], float)
        assert "fdr_q_value" in rule and isinstance(rule["fdr_q_value"], float)
        assert "significant" in rule and isinstance(rule["significant"], bool)
        assert 0.0 <= rule["fdr_q_value"] <= 1.0


def test_association_rule_miner_strong_rule_significant() -> None:
    """A strong eggs => bread rule surfaces as significant (BH FDR < 0.05)."""
    tx = _strong_association_tx()
    rules = AssociationRuleMiner().mine_transactions(
        tx, transaction_col="tid", item_col="item", min_support=0.05, min_confidence=0.5, min_lift=1.0
    )["rules"]
    # mine_transactions emits single-item antecedents/consequents (lists).
    egg_to_bread = [
        r
        for r in rules
        if r["antecedent"] == ["eggs"] and r["consequent"] == ["bread"]
    ]
    assert egg_to_bread, "expected a significant eggs => bread rule"
    top = egg_to_bread[0]
    assert top["confidence"] >= 0.99
    assert top["lift"] > 1.0
    assert top["significant"] is True
    assert top["p_value"] < 0.05
    assert top["fdr_q_value"] < 0.05
    assert top["fdr_q_value"] < 0.05


def test_feature_store_ttl_enforcement(tmp_path: Path) -> None:
    """get_online_features honors spec.ttl_seconds via event_timestamp freshness (RR-014)."""
    store = LocalFeatureStore(repo_path=tmp_path / "fs")
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    df = pd.DataFrame(
        {
            "entity_id": ["c1", "c2"],
            "event_ts": [t0, t0],
            "score": [0.1, 0.2],
        }
    )
    store.register_feature_view(
        "ttl_view", df, join_keys=["entity_id"], timestamp_column="event_ts", ttl_seconds=60, materialize_online=True
    )

    # Fresh relative to t0 -> feature present.
    fresh = store.get_online_features("ttl_view", [{"entity_id": "c1"}], enforce_ttl=True, as_of=t0)
    assert fresh[0]["score"] == 0.1

    # 120s later (> 60s TTL) -> feature expired, returns None.
    stale = store.get_online_features(
        "ttl_view", [{"entity_id": "c1"}], enforce_ttl=True, as_of=t0 + timedelta(seconds=120)
    )
    assert stale[0]["score"] is None

    # Disabling enforcement returns the stored value regardless of age.
    raw = store.get_online_features("ttl_view", [{"entity_id": "c1"}], enforce_ttl=False, as_of=t0 + timedelta(seconds=120))
    assert raw[0]["score"] == 0.1


def test_feature_store_purge_expired_features(tmp_path: Path) -> None:
    """purge_expired_features removes expired offline/online rows (RR-014)."""
    store = LocalFeatureStore(repo_path=tmp_path / "fs2")
    old = datetime(2026, 1, 1, tzinfo=UTC)
    recent = datetime(2026, 2, 1, tzinfo=UTC)
    df = pd.DataFrame(
        {
            "entity_id": ["e_old", "e_recent", "e_old2"],
            "event_ts": [old, recent, old],
            "score": [0.1, 0.2, 0.3],
        }
    )
    store.register_feature_view(
        "purge_view", df, join_keys=["entity_id"], timestamp_column="event_ts", ttl_seconds=86400 * 15, materialize_online=True
    )

    rep = store.purge_expired_features(as_of=recent + timedelta(days=1))
    assert rep["offline_rows_purged"] >= 2
    assert rep["online_rows_purged"] >= 2
    assert rep["views_processed"] >= 1

    # Remaining (non-expired) row survives.
    offline_path = store.get_feature_view("purge_view").offline_path
    offline = pd.read_parquet(offline_path)
    assert (offline["entity_id"] == "e_recent").any()


def test_feature_store_delete_entity_data(tmp_path: Path) -> None:
    """delete_entity_data purges a matching entity offline and online (GDPR Art. 17, RR-014)."""
    store = LocalFeatureStore(repo_path=tmp_path / "fs3")
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    df = pd.DataFrame(
        {
            "entity_id": ["victim", "survivor"],
            "event_ts": [ts, ts],
            "score": [0.1, 0.2],
        }
    )
    store.register_feature_view(
        "erase_view", df, join_keys=["entity_id"], timestamp_column="event_ts", ttl_seconds=86400, materialize_online=True
    )

    rep = store.delete_entity_data({"entity_id": "victim"})
    assert rep["offline_rows_deleted"] >= 1
    assert rep["online_rows_deleted"] >= 1

    offline_path = store.get_feature_view("erase_view").offline_path
    offline = pd.read_parquet(offline_path)
    assert (offline["entity_id"] == "victim").sum() == 0
    assert (offline["entity_id"] == "survivor").sum() == 1


def test_sbom_generate_verify_deterministic() -> None:
    """CycloneDX 1.5 SBOM generation is deterministic and lockfiles are fully hash-pinned (RR-022)."""
    rep = verify_lockfiles_provenance()
    assert rep["verified"] is True
    assert rep["lockfile_count"] >= 1
    assert rep["unpinned_packages"] == []

    doc_a = generate_cyclonedx_sbom(tier="everything")
    doc_b = generate_cyclonedx_sbom(tier="everything")
    assert doc_a == doc_b  # byte-deterministic (timestamp derived from lockfile hash)
    assert doc_a["bomFormat"] == "CycloneDX"
    assert doc_a["specVersion"] == "1.5"
    assert doc_a["metadata"]["component"]["version"] == "4.0.0"

    props = {p["name"]: p["value"] for p in doc_a["metadata"]["properties"]}
    assert props["omniforge:lockfile:name"] == "everything.lock"
    assert len(props["omniforge:lockfile:sha256"]) == 64

    for comp in doc_a["components"]:
        assert comp["type"] == "library"
        assert comp["purl"].startswith("pkg:pypi/")
        assert len(comp["hashes"]) >= 1
        assert all(h["alg"] == "SHA-256" and len(h["content"]) == 64 for h in comp["hashes"])

    # Round-trip through JSON is stable.
    serialized = json.dumps(doc_a, sort_keys=True)
    assert json.loads(serialized)["specVersion"] == "1.5"
