"""Contract tests: restricted deserialization, StrEnum pickle-by-value, and model joblib round-trip.

Regression coverage for SEC-001 (CWE-502): native StrEnum pickling on Python < 3.12 serialized
by *name* via ``builtins.getattr``, which the restricted unpickler correctly rejected — breaking
artifact round-trips on 3.11. All OmniForge enums now serialize by *value*, and the restricted
unpickler is verified to reject both module-confusion payloads and ``builtins.getattr`` abuse.
"""

from __future__ import annotations

import io
import pickle

import joblib
import numpy as np
import pytest

from omniforge.core.exceptions import ErrorSeverity
from omniforge.core.security import SecurityError, safe_joblib_load_bytes
from omniforge.core.types import (
    DataModality,
    DriftStatus,
    ModelStage,
    TaskType,
    ValidationSeverity,
)

_ALL_ENUMS = (TaskType, DataModality, ModelStage, ValidationSeverity, DriftStatus, ErrorSeverity)


@pytest.mark.parametrize("enum_cls", _ALL_ENUMS)
def test_str_enums_pickle_roundtrip_by_value(enum_cls: type) -> None:
    for member in enum_cls:
        payload = pickle.dumps(member)
        # Must not route through builtins.getattr (by-name pickling of 3.11 StrEnums).
        assert b"getattr" not in payload
        restored = pickle.loads(payload)
        assert restored is member
        assert restored == member


@pytest.mark.parametrize("protocol", [0, 2, 4, 5])
def test_str_enums_pickle_protocols(protocol: int) -> None:
    member = next(iter(TaskType))
    restored = pickle.loads(pickle.dumps(member, protocol=protocol))
    assert restored is member


def test_classical_ml_model_joblib_roundtrip_through_restricted_loader() -> None:
    """End-to-end: fit -> joblib dump -> restricted load must preserve predictive behavior."""
    from omniforge.models.classical import ClassicalMLModel

    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = ClassicalMLModel("random_forest", task_type=TaskType.BINARY_CLASSIFICATION, n_estimators=5, random_state=42)
    model.fit(X, y)

    buf = io.BytesIO()
    joblib.dump(model, buf, compress=0)
    blob = buf.getvalue()
    assert blob

    restored = safe_joblib_load_bytes(blob, filename="model.joblib")
    np.testing.assert_allclose(model.predict(X), restored.predict(X))

    # Ensure the wrapped scikit-learn estimator itself survives restricted loading intact.
    assert restored._estimator is not None


def test_restricted_unpickler_rejects_os_system_payload() -> None:
    payload = b"cos\nsystem\n(Vecho pwned\\n\ntR."
    with pytest.raises(SecurityError):
        safe_joblib_load_bytes(payload, filename="evil.pkl")


def test_restricted_unpickler_rejects_builtins_getattr_payload() -> None:
    """The exact primitive abused by Python 3.11 by-name StrEnum pickling must stay blocked."""

    class _Evil:
        def __reduce__(self) -> tuple:
            return (getattr, (dict, "get"))

    payload = pickle.dumps(_Evil())
    with pytest.raises(SecurityError):
        safe_joblib_load_bytes(payload, filename="evil_getattr.pkl")


def test_restricted_unpickler_rejects_unknown_module() -> None:
    class _Evil:
        def __reduce__(self) -> tuple:
            return (__import__("http.client", fromlist=["HTTPConnection"]).HTTPConnection, ())

    payload = pickle.dumps(_Evil())
    with pytest.raises(SecurityError):
        safe_joblib_load_bytes(payload, filename="evil_module.pkl")
