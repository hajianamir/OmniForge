"""Fallbacks stay available while operational failures are observable and data-safe."""

from __future__ import annotations

import ast
import contextlib
import io
import json
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from omniforge.models.computer_vision import foundation
from omniforge.models.mining import miner
from omniforge.models.nlp import advanced_nlp
from omniforge.ui import app as ui_app

_PRIVATE = "private-user-content-never-in-logs"
_TEXTS = ["Quarterly revenue grew 18 percent.", "Customer trust supports revenue growth."]


def _fail(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError(_PRIVATE)


def _exercise_nlp(suite: advanced_nlp.AdvancedNLPSuite, operation: str) -> str:
    if operation == "zero_shot_classification":
        result = suite.zero_shot_classify(_TEXTS, ["finance", "weather"])
        assert len(result) == 2 and result[0]["top_label"] in {"finance", "weather"}
        return str(result[0]["backend"])
    if operation == "question_answering":
        result = suite.answer_question("What grew?", " ".join(_TEXTS))
        assert result["answer"]
        return str(result["backend"])
    if operation == "embeddings":
        embeddings, backend = suite.encode_embeddings(_TEXTS)
        assert embeddings.shape[0] == 2 and np.isfinite(embeddings).all()
        return backend
    result = suite.summarize(" ".join(_TEXTS))
    assert result["summary"]
    return str(result["backend"])


_NLP_CASES = [
    ("zero_shot_classification", "transformers", "native_semantic_tfidf"),
    ("question_answering", "transformers", "native_extractive_qa"),
    ("embeddings", "sentence_transformers", "native_tfidf_svd"),
    ("summarization", "transformers", "native_textrank"),
]


@pytest.mark.parametrize(("operation", "backend", "fallback"), _NLP_CASES)
@pytest.mark.parametrize("failure_phase", ["initialization", "inference"])
def test_nlp_failure_logs_and_preserves_native_fallback(
    monkeypatch: pytest.MonkeyPatch, operation: str, backend: str, fallback: str, failure_phase: str
) -> None:
    def pipeline(*args: Any, **kwargs: Any) -> Any:
        if failure_phase == "initialization":
            return _fail()
        return _fail

    class BrokenSentenceTransformer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if failure_phase == "initialization":
                _fail()

        def encode(self, *args: Any, **kwargs: Any) -> Any:
            return _fail()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=pipeline))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=BrokenSentenceTransformer))
    monkeypatch.setattr(advanced_nlp, "is_package_available", lambda name: name in {"transformers", "sentence_transformers"})
    log = Mock()
    monkeypatch.setattr(advanced_nlp, "logger", log)

    assert _exercise_nlp(advanced_nlp.AdvancedNLPSuite(use_pretrained=True), operation) == fallback
    log.warning.assert_called_once_with(
        "nlp_backend_failed", operation=operation, backend=backend, error_type="RuntimeError", fallback=fallback
    )
    assert _PRIVATE not in repr(log.warning.call_args)


def test_nlp_warning_is_emitted_as_structured_log_without_exception_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=_fail))
    monkeypatch.setattr(advanced_nlp, "is_package_available", lambda name: name == "transformers")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    actual_logger = logging.getLogger("omniforge.models.nlp.advanced_nlp")
    actual_logger.addHandler(handler)
    try:
        result = advanced_nlp.AdvancedNLPSuite(use_pretrained=True).zero_shot_classify(_TEXTS, ["finance", "weather"])
        assert result[0]["backend"] == "native_semantic_tfidf"
    finally:
        actual_logger.removeHandler(handler)
        handler.close()
    record = json.loads(stream.getvalue())
    assert record["event"] == "nlp_backend_failed"
    assert record["level"] == "warning"
    assert record["backend"] == "transformers" and record["error_type"] == "RuntimeError"
    assert record["fallback"] == "native_semantic_tfidf"
    assert _PRIVATE not in stream.getvalue()
    assert all(text not in stream.getvalue() for text in _TEXTS)


@pytest.mark.parametrize("enabled", [False, True])
def test_nlp_absent_or_disabled_optional_backends_do_not_warn(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    available = Mock(return_value=False)
    log = Mock()
    monkeypatch.setattr(advanced_nlp, "is_package_available", available)
    monkeypatch.setattr(advanced_nlp, "logger", log)
    suite = advanced_nlp.AdvancedNLPSuite(use_pretrained=enabled)
    for operation, _, fallback in _NLP_CASES:
        assert _exercise_nlp(suite, operation) == fallback
    log.warning.assert_not_called()
    if not enabled:
        available.assert_not_called()


def test_successful_nlp_backends_do_not_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    def pipeline(task: str, **kwargs: Any) -> Any:
        if task == "zero-shot-classification":
            return lambda text, **options: {"labels": ["finance", "weather"], "scores": [0.8, 0.2]}
        if task == "question-answering":
            return lambda **options: {"answer": "revenue", "score": 0.9, "start": 10, "end": 17}
        return lambda text, **options: [{"summary_text": "Revenue grew."}]

    class SentenceTransformer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
            return np.ones((len(texts), 3))

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=pipeline))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=SentenceTransformer))
    monkeypatch.setattr(advanced_nlp, "is_package_available", lambda name: name in {"transformers", "sentence_transformers"})
    log = Mock()
    monkeypatch.setattr(advanced_nlp, "logger", log)
    suite = advanced_nlp.AdvancedNLPSuite(use_pretrained=True)
    assert _exercise_nlp(suite, "zero_shot_classification") == "transformers_nli"
    assert _exercise_nlp(suite, "question_answering") == "transformers_qa"
    assert _exercise_nlp(suite, "embeddings") == "sentence_transformers"
    assert _exercise_nlp(suite, "summarization") == "transformers:distilbart-cnn-12-6"
    log.warning.assert_not_called()


def test_binning_failure_logs_index_and_continues_mining(monkeypatch: pytest.MonkeyPatch) -> None:
    private_header = f"customer-{_PRIVATE}"
    frame = pd.DataFrame({
        private_header: np.arange(20),
        "good_numeric": np.arange(20) + 100,
        "category": ["alpha", "beta"] * 10,
    })
    original_qcut = pd.qcut

    def qcut(values: pd.Series, *args: Any, **kwargs: Any) -> Any:
        if values.name == private_header:
            raise ValueError(_PRIVATE)
        return original_qcut(values, *args, **kwargs)

    log = Mock()
    monkeypatch.setattr(miner.pd, "qcut", qcut)
    monkeypatch.setattr(miner, "logger", log)
    report = miner.AssociationRuleMiner().mine(frame, min_support=0.1, min_lift=0.0)
    assert report["n_transactions"] == 20 and report["n_items_evaluated"] > 0
    itemsets = [item["itemset"] for item in report["frequent_itemsets"]]
    assert any("good_numeric=" in item for item in itemsets)
    assert all(private_header not in item for item in itemsets)
    log.warning.assert_called_once_with(
        "association_binning_failed", column_index=0, error_type="ValueError", fallback="continue_mining"
    )
    assert _PRIVATE not in repr(log.warning.call_args)


def test_binning_failure_keeps_already_built_items(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame({
        "category": ["alpha", "beta"] * 10,
        "bad_numeric": np.arange(20),
        "good_numeric": np.arange(20) + 100,
    })
    original_qcut = pd.qcut

    class PartiallyFailingBins:
        cat = SimpleNamespace(categories=["Low", "High"])

        def __eq__(self, level: object) -> pd.Series:  # type: ignore[override]
            if level == "High":
                raise ValueError(_PRIVATE)
            return pd.Series([True] * 10 + [False] * 10)

    def qcut(values: pd.Series, *args: Any, **kwargs: Any) -> Any:
        if values.name == "bad_numeric":
            return SimpleNamespace(cat=SimpleNamespace(
                categories=[0, 1], rename_categories=lambda labels: PartiallyFailingBins()
            ))
        return original_qcut(values, *args, **kwargs)

    log = Mock()
    monkeypatch.setattr(miner.pd, "qcut", qcut)
    monkeypatch.setattr(miner, "logger", log)
    report = miner.AssociationRuleMiner().mine(frame, min_support=0.1, min_lift=0.0)
    itemsets = [item["itemset"] for item in report["frequent_itemsets"]]
    assert any("bad_numeric=Low" in item for item in itemsets)
    assert any("good_numeric=" in item for item in itemsets)
    log.warning.assert_called_once_with(
        "association_binning_failed", column_index=1, error_type="ValueError", fallback="continue_mining"
    )


def test_successful_binning_does_not_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    log = Mock()
    monkeypatch.setattr(miner, "logger", log)
    report = miner.AssociationRuleMiner().mine(pd.DataFrame({"numeric": np.arange(20), "category": ["a", "b"] * 10}))
    assert report["n_transactions"] == 20
    log.warning.assert_not_called()


def _install_broken_vision_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = ModuleType("torch")
    fake_timm = SimpleNamespace(create_model=_fail)
    fake_models = SimpleNamespace(resnet18=_fail)
    fake_torchvision = ModuleType("torchvision")
    fake_torchvision.models = fake_models  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "timm", fake_timm)
    monkeypatch.setitem(sys.modules, "torchvision", fake_torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.models", fake_models)
    monkeypatch.setitem(sys.modules, "open_clip", SimpleNamespace(create_model_and_transforms=_fail))
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=_fail))


@pytest.mark.parametrize(
    ("backend", "operation", "fallback"),
    [
        ("timm", "foundation_embedding", "torchvision_or_native_spatial_pyramid"),
        ("torchvision", "foundation_embedding", "native_spatial_pyramid"),
        ("open_clip", "zero_shot_classification", "native_visual_semantic_fallback"),
        ("ultralytics", "object_detection", "native_otsu_morphological"),
    ],
)
def test_vision_failure_logs_and_keeps_fallback(
    monkeypatch: pytest.MonkeyPatch, backend: str, operation: str, fallback: str
) -> None:
    _install_broken_vision_backends(monkeypatch)
    monkeypatch.setattr(foundation, "is_package_available", lambda name: name in {backend, "torch"})
    log = Mock()
    monkeypatch.setattr(foundation, "logger", log)
    hub = foundation.VisionFoundationHub(image_size=(24, 24))
    image = np.full((24, 24, 3), 128, dtype=np.uint8)
    if operation == "foundation_embedding":
        result = hub.extract_foundation_embedding(image, pretrained=False)
        assert result["backend"] == "native_spatial_pyramid" and result["embedding_dim"] > 0
    elif operation == "zero_shot_classification":
        result = hub.zero_shot_classify(image, ["sky", "forest"])
        assert result["backend"] == fallback and result["top_label"] in {"sky", "forest"}
    else:
        result = hub.detect_objects(image)
        assert result["backend"] == fallback and isinstance(result["detections"], list)
    log.warning.assert_called_once_with(
        "vision_backend_failed", operation=operation, backend=backend, error_type="RuntimeError", fallback=fallback
    )
    assert _PRIVATE not in repr(log.warning.call_args)


def test_both_vision_embedding_backends_fail_before_native_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_broken_vision_backends(monkeypatch)
    monkeypatch.setattr(foundation, "is_package_available", lambda name: name in {"torch", "timm", "torchvision"})
    log = Mock()
    monkeypatch.setattr(foundation, "logger", log)
    image = np.full((24, 24, 3), 128, dtype=np.uint8)
    result = foundation.VisionFoundationHub(image_size=(24, 24)).extract_foundation_embedding(image, pretrained=False)
    assert result["backend"] == "native_spatial_pyramid"
    assert [call.kwargs["backend"] for call in log.warning.call_args_list] == ["timm", "torchvision"]
    assert [call.kwargs["fallback"] for call in log.warning.call_args_list] == [
        "torchvision_or_native_spatial_pyramid", "native_spatial_pyramid"
    ]


def test_absent_vision_backends_do_not_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(foundation, "is_package_available", lambda name: False)
    log = Mock()
    monkeypatch.setattr(foundation, "logger", log)
    hub = foundation.VisionFoundationHub(image_size=(24, 24))
    image = np.full((24, 24, 3), 128, dtype=np.uint8)
    assert hub.extract_foundation_embedding(image)["backend"] == "native_spatial_pyramid"
    assert hub.zero_shot_classify(image, ["sky", "forest"])["backend"] == "native_visual_semantic_fallback"
    assert hub.detect_objects(image)["backend"] == "native_otsu_morphological"
    log.warning.assert_not_called()


def test_successful_vision_backend_does_not_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    class Tensor:
        def permute(self, *args: Any) -> Tensor:
            return self

        def unsqueeze(self, *args: Any) -> Tensor:
            return self

        def cpu(self) -> Tensor:
            return self

        def numpy(self) -> np.ndarray:
            return np.array([[3.0, 4.0]], dtype=np.float32)

    class Model:
        def eval(self) -> Model:
            return self

        def __call__(self, *args: Any) -> Tensor:
            return Tensor()

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(from_numpy=lambda array: Tensor(), no_grad=contextlib.nullcontext))
    monkeypatch.setitem(sys.modules, "timm", SimpleNamespace(create_model=lambda *args, **kwargs: Model()))
    monkeypatch.setattr(foundation, "is_package_available", lambda name: name in {"timm", "torch"})
    log = Mock()
    monkeypatch.setattr(foundation, "logger", log)
    hub = foundation.VisionFoundationHub(image_size=(24, 24))
    image = np.full((24, 24, 3), 128, dtype=np.uint8)
    result = hub.extract_foundation_embedding(image, pretrained=False)
    assert result["backend"] == "timm:resnet18"
    np.testing.assert_allclose(result["embedding"], [0.6, 0.8])
    log.warning.assert_not_called()


@pytest.mark.parametrize("behavior", ["raise", "unavailable", "success", "disabled"])
def test_browser_open_failure_is_observable_without_stopping_server(
    monkeypatch: pytest.MonkeyPatch, behavior: str
) -> None:
    run = Mock()
    opened = Mock(side_effect=_fail) if behavior == "raise" else Mock(return_value=behavior == "success")
    log = Mock()

    class ImmediateThread:
        def __init__(self, *, target: Any, **kwargs: Any) -> None:
            self.target = target

        def start(self) -> None:
            self.target()

    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=run))
    monkeypatch.setattr(ui_app, "create_workbench_app", lambda: "app")
    monkeypatch.setattr(ui_app, "threading", SimpleNamespace(Thread=ImmediateThread))
    monkeypatch.setattr(ui_app, "time", SimpleNamespace(sleep=lambda seconds: None))
    monkeypatch.setattr(ui_app, "webbrowser", SimpleNamespace(open=opened))
    monkeypatch.setattr(ui_app, "logger", log)
    ui_app.run_workbench_server(host="0.0.0.0", port=8750, open_browser=behavior != "disabled")
    run.assert_called_once_with("app", host="0.0.0.0", port=8750, log_level="info")
    if behavior == "raise":
        log.warning.assert_called_once_with(
            "workbench_browser_open_failed", reason="browser_open_exception", error_type="RuntimeError", fallback="server_only"
        )
    elif behavior == "unavailable":
        log.warning.assert_called_once_with(
            "workbench_browser_open_failed", reason="browser_open_returned_false", fallback="server_only"
        )
    else:
        log.warning.assert_not_called()
    if behavior == "disabled":
        opened.assert_not_called()
    else:
        opened.assert_called_once_with("http://127.0.0.1:8750")
    assert _PRIVATE not in repr(log.warning.call_args)


def test_no_bare_exception_pass_in_reviewed_files() -> None:
    root = Path(__file__).resolve().parents[2]
    files = [
        "src/omniforge/ui/app.py",
        "src/omniforge/models/mining/miner.py",
        "src/omniforge/models/nlp/advanced_nlp.py",
        "src/omniforge/models/computer_vision/foundation.py",
    ]
    offenders: list[str] = []
    for name in files:
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ExceptHandler)
                and isinstance(node.type, ast.Name)
                and node.type.id == "Exception"
                and len(node.body) == 1
                and isinstance(node.body[0], ast.Pass)
            ):
                offenders.append(f"{name}:{node.lineno}")
    assert not offenders, f"Silent exception handlers found: {offenders}"
