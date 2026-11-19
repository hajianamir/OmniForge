from omniforge.capabilities import (
    OPTIONAL_PACKS,
    PACKS_BY_ID,
    TIERS,
    imports_for,
    packs_for_tier,
    profile_summary,
    requirements_for,
    resolve_selection,
)


def test_tiers_are_cumulative():
    assert set(TIERS["simple"]) < set(TIERS["advanced"]) < set(TIERS["everything"])


def test_every_tier_references_real_packs():
    for ids in TIERS.values():
        for pid in ids:
            assert pid in PACKS_BY_ID


def test_optional_packs_are_reachable_from_wizard():
    assert "deep" in OPTIONAL_PACKS and "boosting" in OPTIONAL_PACKS
    sel = resolve_selection("simple", ["deep"])
    assert "deep" in [p.id for p in sel]


def test_requirements_are_deduplicated_and_pinned():
    reqs = requirements_for(packs_for_tier("everything"))
    names = [r.split(">=")[0].split("==")[0].split("[")[0].lower() for r in reqs]
    assert len(names) == len(set(names))
    assert all(any(op in r for op in (">=", "==", "~=")) for r in reqs)


def test_simple_tier_excludes_every_heavy_backend():
    imports = set(imports_for(packs_for_tier("simple")))
    for heavy in ("torch", "xgboost", "catboost", "cv2", "fastapi", "transformers"):
        assert heavy not in imports


def test_advanced_tier_adds_the_server_but_no_optional_backends():
    imports = set(imports_for(packs_for_tier("advanced")))
    assert "fastapi" in imports and "uvicorn" in imports
    assert "torch" not in imports and "xgboost" not in imports


def test_profile_summary_reports_sizes_and_unlocks():
    summary = profile_summary("advanced")
    assert summary["tier"] == "advanced"
    assert summary["approx_mb"] > 0
    assert "api-server" in summary["packs"]
    assert summary["unlocks"]["core-runtime"]


def test_unknown_pack_is_rejected():
    try:
        resolve_selection("simple", ["does-not-exist"])
    except KeyError:
        return
    raise AssertionError("unknown pack id must raise")
