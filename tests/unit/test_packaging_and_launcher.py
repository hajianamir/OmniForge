"""Unit tests for the OmniForge desktop launcher, Windows .bat scripts, and .exe packaging suite."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from omniforge.launcher import (
    build_parser,
    find_available_port,
    is_frozen,
    load_dotenv_file,
    resolve_resource_path,
    run_environment_check,
)
from omniforge.launcher import (
    main as launcher_main,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_launcher_environment_check_and_helpers(tmp_path: Path, capsys) -> None:
    """Verify environment self-check, resource lookup, dotenv loader, and port finder."""
    assert is_frozen() is False

    cfg = resolve_resource_path("configs/default.yaml")
    assert cfg.is_file()

    env_file = tmp_path / ".env"
    env_file.write_text("# comment\nOMNIFORGE_TEST_LAUNCHER_VAR=enterprise_ok\n", encoding="utf-8")
    loaded = load_dotenv_file(env_file)
    assert loaded.get("OMNIFORGE_TEST_LAUNCHER_VAR") == "enterprise_ok"

    port = find_available_port("127.0.0.1", 18501)
    assert isinstance(port, int) and port >= 18501

    report = run_environment_check()
    assert report["domains"]["count"] == 18
    assert report["resources"]["ui_app_exists"] is True
    assert report["resources"]["default_config_exists"] is True
    assert report["required_modules"]["fastapi"] is True
    assert report["required_modules"]["uvicorn"] is True
    assert report["required_modules"]["pyarrow"] is True
    assert report["required_modules"]["duckdb"] is True

    parser = build_parser()
    parsed = parser.parse_args(["--mode", "check", "--json"])
    assert parsed.mode == "check" and parsed.json is True

    rc = launcher_main(["--mode", "check", "--json"])
    captured = capsys.readouterr()
    parsed_json = json.loads(captured.out)
    assert parsed_json["domains"]["count"] == 18
    assert rc in (0, 1)


def test_packaging_icon_specs_and_dry_run(tmp_path: Path) -> None:
    """Verify Windows icon generator, PyInstaller specs, version info, and build dry-run."""
    packaging_dir = PROJECT_ROOT / "packaging"
    assert (packaging_dir / "omniforge_version_info.txt").is_file()
    assert (packaging_dir / "entry_desktop.py").is_file()
    assert (packaging_dir / "entry_cli.py").is_file()
    assert (packaging_dir / "pyinstaller" / "omniforge.spec").is_file()
    assert (packaging_dir / "pyinstaller" / "omniforge_cli.spec").is_file()
    assert (packaging_dir / "inno_setup" / "omniforge_installer.iss").is_file()
    assert (PROJECT_ROOT / "requirements.txt").is_file()
    assert (PROJECT_ROOT / "requirements-build.txt").is_file()

    sys.path.insert(0, str(packaging_dir))
    try:
        from build_exe import main as build_main
        from build_exe import validate_build_environment, write_checksums
        from generate_icon import generate_omniforge_icon

        ico_path, png_path = generate_omniforge_icon(tmp_path)
        assert ico_path.is_file() and ico_path.stat().st_size > 1000
        assert png_path.is_file() and png_path.stat().st_size > 1000

        checks = validate_build_environment(require_pyinstaller=False)
        assert checks["omniforge_spec_exists"] is True
        assert checks["omniforge_cli_spec_exists"] is True

        assert build_main(["--mode", "onedir", "--target", "all", "--dry-run"]) == 0
        assert build_main(["--mode", "onefile", "--target", "desktop", "--dry-run"]) == 0

        fake_exe = tmp_path / "OmniForge.exe"
        fake_exe.write_bytes(b"MZ-test-payload")
        manifest = write_checksums(tmp_path)
        assert "OmniForge.exe" in manifest.read_text(encoding="utf-8")
    finally:
        if str(packaging_dir) in sys.path:
            sys.path.remove(str(packaging_dir))


def test_serving_batch_onnx_health_monitor_and_historical_store(tmp_path: Path) -> None:
    """Verify MLOps batch scoring, ONNX/edge export, /api/v1/health, CLI monitor, and point-in-time feature join."""
    import pandas as pd
    from fastapi.testclient import TestClient
    from typer.testing import CliRunner

    from omniforge.api.main import create_app
    from omniforge.cli.main import cli
    from omniforge.core.types import TaskType
    from omniforge.data.feature_store import LocalFeatureStore
    from omniforge.domains import default_domain_registry
    from omniforge.mlops.serving import ModelServingEngine
    from omniforge.models.classical import ClassicalMLModel

    # 1. Batch serving + ONNX/portable edge export
    fin_df = default_domain_registry.get("finance").generate_sample_dataset(n_samples=80, random_state=7)
    X = fin_df.drop(columns=["default_flag"])
    y = fin_df["default_flag"]
    model = ClassicalMLModel(
        algorithm="random_forest", task_type=TaskType.BINARY_CLASSIFICATION, hyperparameters={"n_estimators": 20}
    ).fit(X, y)

    serving = ModelServingEngine()
    serving.deploy_model("credit_risk_v1", model, version="1.0.0")
    scored = serving.predict_batch("credit_risk_v1", X.head(25), batch_size=10, return_probabilities=True)
    assert len(scored) == 25
    assert "prediction" in scored.columns
    assert "prediction_probability" in scored.columns

    exported = serving.export_onnx("credit_risk_v1", tmp_path / "credit_risk.onnx", X.head(5))
    assert Path(exported["path"]).is_file()

    # 2. Docker healthcheck endpoint /api/v1/health
    client = TestClient(create_app())
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200 and resp.json()["status"] == "healthy"

    # 3. Worker CLI monitor command
    runner = CliRunner()
    mon = runner.invoke(cli, ["monitor"])
    assert mon.exit_code == 0 and "Drift Status" in mon.output

    # 4. Point-in-time historical feature join (verifies numpy import fix)
    fs = LocalFeatureStore(repo_path=tmp_path / "feature_repo")
    ts = pd.date_range("2026-01-01", periods=4, freq="D")
    feat_df = pd.DataFrame({"entity_id": [1, 1, 2, 2], "event_ts": ts, "score": [0.1, 0.4, 0.7, 0.9]})
    fs.register_feature_view("risk_view", feat_df, join_keys=["entity_id"], timestamp_column="event_ts")
    entity_df = pd.DataFrame({"entity_id": [1, 2], "query_ts": [ts[1], ts[3]]})
    joined = fs.get_historical_features(entity_df, ["risk_view:score"], entity_timestamp_col="query_ts")
    assert len(joined) == 2 and "score" in joined.columns


def test_computer_vision_without_cv2_fallback(tmp_path: Path, monkeypatch) -> None:
    """Verify ComputerVisionEngine works completely via Pillow+SciPy+NumPy when cv2 is absent."""
    import numpy as np
    from PIL import Image

    from omniforge.models.computer_vision import vision as vision_mod

    monkeypatch.setattr(vision_mod, "_HAS_CV2", False)
    monkeypatch.setattr(vision_mod, "cv2", None)

    cv = vision_mod.ComputerVisionEngine(image_size=(32, 32))
    img_a = np.zeros((48, 48, 3), dtype=np.uint8)
    img_a[10:34, 10:34] = [240, 240, 240]
    img_b = np.full((48, 48, 3), 30, dtype=np.uint8)
    img_b[4:16, 4:16] = [200, 60, 40]

    img_file = tmp_path / "sample.png"
    Image.fromarray(img_a).save(img_file)

    loaded = cv.load_image(img_file)
    assert loaded.shape == (32, 32, 3) and loaded.dtype == np.uint8

    desc = cv.extract_descriptor(img_file)
    assert desc.shape == (38,) and np.all(np.isfinite(desc))

    dets = cv.detect_objects(img_a, min_area=10)
    assert len(dets) >= 1 and len(dets[0]["bbox"]) == 4

    aug = cv.augment(img_a, flip_horizontal=True, brightness_delta=10.0)
    assert aug.shape == img_a.shape and aug.dtype == np.uint8

    cv.fit_classifier([img_a, img_a, img_b, img_b], [0, 0, 1, 1])
    preds = cv.predict_classifier([img_a, img_b])
    assert len(preds) == 2


def test_windows_batch_scripts_integrity_and_crlf() -> None:
    """Validate install.bat, start.bat, and build_exe.bat syntax, CRLF, labels, and English-only text."""
    for script_name in ("install.bat", "start.bat", "build_exe.bat"):
        path = PROJECT_ROOT / script_name
        assert path.is_file(), f"Missing Windows script: {script_name}"

        raw = path.read_bytes()
        assert b"\r\n" in raw, f"{script_name} must use Windows CRLF line endings"
        assert b"\n" not in raw.replace(b"\r\n", b""), f"{script_name} has mixed LF/CRLF line endings"

        text = raw.decode("utf-8")
        # Strictly ASCII/English batch content
        assert all(ord(ch) < 128 for ch in text), f"{script_name} contains non-ASCII characters"

        lines = text.splitlines()
        labels = {
            m.group(1).lower()
            for line in lines
            if (m := re.match(r"^\s*:([A-Za-z0-9_]+)\s*$", line))
        }
        for line in lines:
            for m in re.finditer(r"(?:goto|call)\s+:([A-Za-z0-9_]+)", line, re.I):
                target = m.group(1).lower()
                assert target in labels or target == "eof", (
                    f"{script_name} references undefined label :{target}"
                )
            # Ensure for /f ('...') commands never contain nested single quotes
            if re.search(r"for\s+/f\b", line, re.I) and "in ('" in line:
                inner = line.split("in ('", 1)[1].rsplit("')", 1)[0]
                assert "'" not in inner, f"{script_name} has nested single quotes in for /f: {line}"

        # Validate cmd.exe parenthesized block safety: no unescaped '(' or ')' in echo inside if/else/for blocks
        depth = 0
        for lineno, raw_line in enumerate(lines, 1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("::") or stripped.lower().startswith("rem "):
                continue
            if re.match(r"^\)\s*(else\b.*)?$", stripped, re.I):
                depth -= 1
                assert depth >= 0, f"{script_name}:{lineno} unmatched closing ')'"
                if stripped.rstrip().endswith("("):
                    depth += 1
                continue
            in_q = False
            esc = False
            open_inline = 0
            last_ns = ""
            for i, ch in enumerate(raw_line):
                if esc:
                    esc = False
                    continue
                if ch == "^" and not in_q:
                    esc = True
                    continue
                if ch == '"':
                    in_q = not in_q
                    continue
                if not in_q:
                    if not ch.isspace():
                        last_ns = ch
                    if depth > 0:
                        if ch == "(":
                            if raw_line[i + 1 :].strip() == "":
                                depth += 1
                            elif stripped.lower().startswith("echo"):
                                raise AssertionError(
                                    f"{script_name}:{lineno} unescaped '(' in echo inside block: {stripped}"
                                )
                            else:
                                open_inline += 1
                        elif ch == ")":
                            if stripped.lower().startswith("echo"):
                                raise AssertionError(
                                    f"{script_name}:{lineno} fatal unescaped ')' in echo inside block: {stripped}"
                                )
                            if open_inline > 0:
                                open_inline -= 1
                            else:
                                raise AssertionError(
                                    f"{script_name}:{lineno} fatal stray unescaped ')' inside block: {stripped}"
                                )
            assert not in_q, f"{script_name}:{lineno} unclosed double quote"
            if depth == 0 and last_ns == "(" and re.match(r"^(if|for)\b", stripped, re.I):
                depth += 1
        assert depth == 0, f"{script_name} unbalanced parenthesized block depth={depth}"

    install_text = (PROJECT_ROOT / "install.bat").read_text(encoding="utf-8")
    assert "Rebuilding incompatible .venv" in install_text
    assert "Active .venv Python" in install_text
    assert "--no-build-isolation" in install_text
    assert 'set "PYYAML_FORCE_LIBYAML=0"' in install_text
    assert "import omniforge.launcher" in install_text

    start_text = (PROJECT_ROOT / "start.bat").read_text(encoding="utf-8")
    assert "sys.version_info >= (3, 12)" in start_text
    assert 'set "PYTHONPATH=%ROOT_DIR%\\src;%PYTHONPATH%"' in start_text
    assert "import omniforge.launcher, fastapi, uvicorn, pandas, numpy, sklearn, pyarrow" in start_text
    assert ':run_full\nif not exist "%VENV_PY%" goto :run_exe\ncall :ensure_venv' in start_text.replace("\r\n", "\n")

    build_text = (PROJECT_ROOT / "build_exe.bat").read_text(encoding="utf-8")
    assert 'set "PYTHONPATH=%ROOT_DIR%\\src;%PYTHONPATH%"' in build_text
    assert 'set "PYYAML_FORCE_LIBYAML=0"' in build_text


def test_pyinstaller_specs_submodule_hygiene_and_frozen_assets() -> None:
    """Verify PyInstaller specs cleanly collect all uap submodules and compiled React SPA assets."""
    import pkgutil

    import omniforge

    # 1. Verify every directory under src/omniforge has __init__.py and pkgutil discovers all omniforge submodules cleanly
    omniforge_root = PROJECT_ROOT / "src" / "omniforge"
    for dir_path in omniforge_root.rglob("*"):
        if dir_path.is_dir() and dir_path.name != "__pycache__" and any(dir_path.glob("*.py")):
            assert (dir_path / "__init__.py").is_file(), f"Missing __init__.py in {dir_path}"

    discovered = [m.name for m in pkgutil.walk_packages(omniforge.__path__, prefix="omniforge.")]
    assert len(discovered) >= 140
    assert "omniforge.models.clustering.clusterer" in discovered
    assert "omniforge.api.workbench" in discovered
    assert "omniforge.automl.backends" in discovered
    assert "omniforge.models.deep.tabnet" in discovered
    assert "omniforge.models.deep.resnet_tabular" in discovered
    assert "omniforge.models.computer_vision.foundation" in discovered
    assert "omniforge.models.nlp.advanced_nlp" in discovered
    assert "omniforge.models.graphs.link_prediction" in discovered

    # 2. Verify both PyInstaller specs fix all submodule/runtime-asset regressions
    for spec_name in ("omniforge.spec", "omniforge_cli.spec"):
        spec_text = (PROJECT_ROOT / "packaging" / "pyinstaller" / spec_name).read_text(encoding="utf-8")
        assert "sys.path.insert(0, str(SRC_DIR))" in spec_text
        assert 'collect_submodules("omniforge"' in spec_text
        assert "pkg_resources" not in spec_text, f"{spec_name} must not reference removed pkg_resources"
        assert '"unittest"' not in spec_text, f"{spec_name} must not exclude stdlib unittest"
        assert "_keep_submodule" in spec_text
        assert "sklearn.externals.array_api_compat" in spec_text
        for required_data_pkg in (
            "rfc3987_syntax",
            "lark",
            "jsonschema",
            "jsonschema_specifications",
            "tzdata",
            "certifi",
        ):
            assert f'"{required_data_pkg}"' in spec_text, f"{spec_name} missing runtime data pkg {required_data_pkg}"

    uap_spec_text = (PROJECT_ROOT / "packaging" / "pyinstaller" / "omniforge.spec").read_text(encoding="utf-8")
    assert '"fastapi"' in uap_spec_text
    assert '"omniforge/ui/static_app"' in uap_spec_text
    assert (PROJECT_ROOT / "src" / "omniforge" / "ui" / "static_app" / "index.html").is_file()

