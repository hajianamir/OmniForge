"""Rigorous regression tests for OmniForge v4.0.0 audit remediations."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import multiprocessing
import os
import socket
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import numpy as np
import pandas as pd
import pytest

from omniforge.core.exceptions import ModelSerializationError, SecurityError, SimulationError
from omniforge.core.security import AuditCheckpointAnchor, AuditLogger, JWTManager
from omniforge.core.types import Principal, TaskType
from omniforge.data.connectors import resolve_and_validate_outbound_http_url
from omniforge.data.ingestion import DataIngestionEngine
from omniforge.models.base import BaseOmniForgeModel
from omniforge.models.classical import ClassicalMLModel
from omniforge.simulation.closed_loop import ClosedLoopMPCController, DigitalTwinEngine, build_default_twin


def _multiprocess_audit_worker(args: tuple[str, str, str, int, int]) -> int:
    log_path, anchor_path, key_dir, worker_id, events_per_worker = args
    os.environ["OMNIFORGE_KEY_DIR"] = key_dir
    audit = AuditLogger(
        log_path=log_path,
        anchor_path=anchor_path,
        checkpoint_key="mp-checkpoint-secret-key-at-least-32-bytes!!",
        anchor_every=1,
    )
    for idx in range(events_per_worker):
        audit.log_event(
            action="mp.write",
            resource_type="worker",
            resource_id=f"w{worker_id}-{idx}",
            actor=f"worker-{worker_id}",
            details={"worker": worker_id, "seq": idx},
        )
    return events_per_worker


# ---------------------------------------------------------------------------
# P0-1: ClosedLoopMPCController exact gradients, initial-state dynamics, slack
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("horizon", [1, 3, 6])
def test_mpc_objective_grad_and_dynamics_jacobian_match_finite_differences(horizon: int) -> None:
    rng = np.random.default_rng(42 + horizon)
    n_x, n_u = 3, 2
    twin = DigitalTwinEngine(
        A=np.array([[0.95, 0.12, 0.0], [-0.05, 0.88, 0.07], [0.02, -0.04, 0.91]]),
        B=np.array([[0.4, -0.1], [0.15, 0.35], [-0.05, 0.25]]),
        H=np.eye(n_x),
        Q=np.eye(n_x) * 1e-3,
        R=np.eye(n_x) * 1e-2,
        x0=np.zeros(n_x),
    )
    # Deliberately test non-symmetric positive-definite Q and R as well as symmetric weights.
    Q = np.array([[2.0, 0.4, -0.1], [0.1, 1.5, 0.2], [0.0, -0.1, 1.2]])
    R = np.array([[0.5, 0.15], [-0.05, 0.8]])
    ctrl = ClosedLoopMPCController(twin, horizon=horizon, Q=Q, R=R)

    z = rng.normal(size=horizon * (n_u + n_x))
    r_path = rng.normal(size=(horizon, n_x))
    x0 = rng.normal(size=n_x)

    analytic_grad = ctrl._objective_grad(z, r_path)
    eps = 1e-6
    fd_grad = np.zeros_like(z)
    for i in range(z.size):
        zp = z.copy()
        zm = z.copy()
        zp[i] += eps
        zm[i] -= eps
        fd_grad[i] = (ctrl._objective(zp, r_path) - ctrl._objective(zm, r_path)) / (2.0 * eps)
    np.testing.assert_allclose(analytic_grad, fd_grad, rtol=1e-6, atol=1e-7)

    c0 = ctrl._dynamics_constraints(z, x0)
    assert c0.shape == (horizon * n_x,)
    analytic_jac = ctrl._dynamics_jacobian(z, x0)
    assert analytic_jac.shape == (horizon * n_x, z.size)
    fd_jac = np.zeros_like(analytic_jac)
    for i in range(z.size):
        zp = z.copy()
        zm = z.copy()
        zp[i] += eps
        zm[i] -= eps
        fd_jac[:, i] = (ctrl._dynamics_constraints(zp, x0) - ctrl._dynamics_constraints(zm, x0)) / (2.0 * eps)
    np.testing.assert_allclose(analytic_jac, fd_jac, rtol=1e-7, atol=1e-8)


def test_mpc_enforces_initial_step_dynamics_and_tracks_from_nonzero_x0() -> None:
    twin = build_default_twin(n_x=2, n_u=1, dt=1.0)
    ctrl = ClosedLoopMPCController(
        twin,
        horizon=5,
        u_min=np.array([-5.0]),
        u_max=np.array([5.0]),
        x_min=np.array([-20.0, -20.0]),
        x_max=np.array([20.0, 20.0]),
    )
    x0 = np.array([4.0, -2.0])
    setpoint = np.zeros((5, 2))
    res = ctrl.solve(x0=x0, r_path=setpoint)

    assert res.success is True
    assert res.slack_used == 0.0
    # Verify exact dynamics rollout across the entire predicted horizon including k=0 (x_1 = A x_0 + B u_0)!
    expected_x = x0.copy()
    for k in range(ctrl.horizon):
        expected_x = twin.A @ expected_x + twin.B @ res.full_control_sequence[k]
        np.testing.assert_allclose(res.state_trajectory[k + 1], expected_x, atol=1e-6)
    # Non-zero control must be applied at t=0 to drive x0=[4,-2] toward origin (previously u_0 was 0.0!).
    assert np.linalg.norm(res.optimal_control) > 1e-2


def test_mpc_soft_state_constraints_converge_when_hard_state_bounds_infeasible() -> None:
    twin = build_default_twin(n_x=2, n_u=1, dt=1.0)
    ctrl = ClosedLoopMPCController(
        twin,
        horizon=6,
        u_min=np.array([-1.0]),
        u_max=np.array([1.0]),
        x_min=np.array([-5.0, -5.0]),
        x_max=np.array([5.0, 5.0]),
    )
    # x0 = [25, 0] cannot reach [-5, 5] in 1 step when |u| <= 1, so hard QP is infeasible.
    x0 = np.array([25.0, 0.0])
    res = ctrl.solve(x0=x0, r_path=np.zeros((6, 2)))

    assert res.success is True
    assert res.slack_used > 0.0
    assert np.isfinite(res.cost)
    # Hard actuator bounds must still hold strictly.
    assert np.all(res.full_control_sequence >= ctrl.u_min - 1e-6)
    assert np.all(res.full_control_sequence <= ctrl.u_max + 1e-6)
    # Predicted states must still obey physical plant dynamics from x0.
    roll = x0.copy()
    for k in range(6):
        roll = twin.A @ roll + twin.B @ res.full_control_sequence[k]
        np.testing.assert_allclose(res.state_trajectory[k + 1], roll, atol=1e-5)
    # Original state-bound violations are transparently reported.
    assert any(v.startswith("x[") for v in res.constraint_violations)


def test_mpc_validates_inverted_bounds_and_persists_warm_start_and_weights() -> None:
    twin = build_default_twin(n_x=2, n_u=1, dt=1.0)
    with pytest.raises(SimulationError, match="u_min cannot exceed u_max"):
        ClosedLoopMPCController(twin, u_min=np.array([2.0]), u_max=np.array([1.0]))
    with pytest.raises(SimulationError, match="x_min cannot exceed x_max"):
        ClosedLoopMPCController(twin, x_min=np.array([0.0, 5.0]), x_max=np.array([1.0, 4.0]))

    custom_Q = np.diag([3.5, 1.25])
    custom_R = np.array([[0.75]])
    ctrl = ClosedLoopMPCController(twin, horizon=4, Q=custom_Q, R=custom_R)
    state: dict[str, Any] = {}
    ctrl.run_closed_loop([np.array([1.0, 0.5]), np.array([0.8, 0.3])], setpoint=np.zeros(2), state=state, state_key="mpc")
    saved = state["mpc"]
    assert saved["warm_start"] is not None
    assert len(saved["warm_start"]) == 4 * (1 + 2)
    restored_ctrl, ws = ClosedLoopMPCController.restore(twin, saved)
    assert ws is not None and ws.size == 12
    np.testing.assert_allclose(restored_ctrl.Q, custom_Q)
    np.testing.assert_allclose(restored_ctrl.R, custom_R)


# ---------------------------------------------------------------------------
# P0-2: AuditLogger & AuditCheckpointAnchor multi-instance / thread / process
# ---------------------------------------------------------------------------


def test_audit_logger_multi_instance_and_multithreaded_no_chain_fork(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_dir = tmp_path / "keys"
    monkeypatch.setenv("OMNIFORGE_KEY_DIR", str(key_dir))
    log_path = tmp_path / "audit.jsonl"
    anchor_path = tmp_path / "anchor.jsonl"
    ckpt_key = "unit-test-checkpoint-key-at-least-32-chars!!"

    # Two independent AuditLogger instances initialized BEFORE any writes (previously forked at record 1!)
    a1 = AuditLogger(log_path=log_path, anchor_path=anchor_path, checkpoint_key=ckpt_key, anchor_every=1)
    a2 = AuditLogger(log_path=log_path, anchor_path=anchor_path, checkpoint_key=ckpt_key, anchor_every=1)
    a1.log_event(action="e1", resource_type="ds", resource_id="1")
    a2.log_event(action="e2", resource_type="ds", resource_id="2")

    ok_chain, count = AuditLogger.verify_file(log_path)
    assert ok_chain is True and count == 2
    ok_anchor, checked, detail = AuditCheckpointAnchor.verify_against_log(log_path, anchor_path, ckpt_key)
    assert ok_anchor is True and checked == 2, detail

    # Hammer the same log + anchor concurrently across 8 threads with separate instances.
    def _thread_worker(tid: int) -> None:
        inst = AuditLogger(log_path=log_path, anchor_path=anchor_path, checkpoint_key=ckpt_key, anchor_every=1)
        for j in range(10):
            inst.log_event(action="thread.op", resource_type="item", resource_id=f"{tid}-{j}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_thread_worker, range(8)))

    ok_chain, count = AuditLogger.verify_file(log_path)
    assert ok_chain is True and count == 82
    ok_anchor, checked, detail = AuditCheckpointAnchor.verify_against_log(log_path, anchor_path, ckpt_key)
    assert ok_anchor is True and checked == 82, detail


def test_audit_logger_multiprocess_concurrency_and_tamper_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_dir = tmp_path / "mp_keys"
    monkeypatch.setenv("OMNIFORGE_KEY_DIR", str(key_dir))
    seed_log = tmp_path / "audit_mp.jsonl"
    anchor_log = tmp_path / "anchor_mp.jsonl"

    # Reject misconfigured anchor_path == log_path upfront.
    with pytest.raises(SecurityError, match="distinct from log_path"):
        AuditLogger(log_path=seed_log, anchor_path=seed_log, checkpoint_key="mp-checkpoint-secret-key-at-least-32-bytes!!")

    # Cold-start: key_dir does NOT exist yet — all 4 spawned worker processes race to initialize
    # signing keyfiles, audit log, and checkpoint anchor simultaneously without pre-seeding!
    assert not key_dir.exists()
    tasks = [(str(seed_log), str(anchor_log), str(key_dir), w, 8) for w in range(4)]
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
        totals = list(pool.map(_multiprocess_audit_worker, tasks))
    assert sum(totals) == 32

    ok_chain, count = AuditLogger.verify_file(seed_log)
    assert ok_chain is True and count == 32
    ok_anchor, checked, detail = AuditCheckpointAnchor.verify_against_log(
        seed_log, anchor_log, "mp-checkpoint-secret-key-at-least-32-bytes!!"
    )
    assert ok_anchor is True and checked == 32, detail

    active_logger = AuditLogger(
        log_path=seed_log,
        anchor_path=anchor_log,
        checkpoint_key="mp-checkpoint-secret-key-at-least-32-bytes!!",
        anchor_every=1,
    )
    active_logger.log_event(action="post_workers", resource_type="init", resource_id="32")

    # Tamper with an HISTORICAL first record (same byte length edit!) and verify active logger fails closed.
    lines = seed_log.read_text(encoding="utf-8").splitlines(keepends=True)
    rec0 = json.loads(lines[0])
    orig_action = str(rec0["action"])
    rec0["action"] = ("X" if orig_action[0] != "X" else "Y") + orig_action[1:]
    lines[0] = json.dumps(rec0) + "\n"
    seed_log.write_text("".join(lines), encoding="utf-8")

    with pytest.raises(SecurityError):
        active_logger.log_event(action="post_tamper", resource_type="init", resource_id="99")


# ---------------------------------------------------------------------------
# P0-3: Atomic BaseOmniForgeModel.save() & zero plaintext exposure on disk
# ---------------------------------------------------------------------------


def test_model_save_encrypted_never_writes_plaintext_and_preserves_existing_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OMNIFORGE_KEY_DIR", str(tmp_path / "keys"))
    X = pd.DataFrame({"f1": [0.1, 0.2, 0.8, 0.9, 0.15, 0.85], "f2": [1.0, 0.9, 0.1, 0.2, 0.95, 0.05]})
    y = pd.Series([0, 0, 1, 1, 0, 1])
    model = ClassicalMLModel(
        "random_forest",
        task_type=TaskType.BINARY_CLASSIFICATION,
        n_estimators=8,
        random_state=7,
    )
    model.fit(X, y)

    enc_key = base64.urlsafe_b64encode(b"0123456789abcdef0123456789abcdef").decode("ascii")
    enc_target = tmp_path / "encrypted_rf.joblib"

    written_payloads: list[bytes] = []
    orig_write_temp = BaseOmniForgeModel._write_temp_fsync

    def _spy_write_temp(parent: Path, prefix: str, payload: bytes) -> Path:
        if enc_target.name in prefix and not prefix.endswith(".meta.json."):
            written_payloads.append(payload)
        return orig_write_temp(parent, prefix, payload)

    with patch.object(BaseOmniForgeModel, "_write_temp_fsync", side_effect=_spy_write_temp):
        model.save(enc_target, encrypt=True, encryption_key=enc_key)

    assert len(written_payloads) == 1
    # Every byte written to disk for the model artifact must be urlsafe-base64 AES-256-GCM ciphertext (OF2GCM\x01), never pickle/joblib plaintext.
    assert base64.urlsafe_b64decode(written_payloads[0]).startswith(b"OF2GCM\x01")
    assert b"ClassicalMLModel" not in written_payloads[0]
    loaded = BaseOmniForgeModel.load(enc_target, encrypted=True, encryption_key=enc_key)
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))

    # Now simulate an OS failure on the second os.replace (swapping the artifact after meta_path swap)
    # and prove that the existing valid model + metadata sidecar remain 100% intact and loadable!
    model2 = ClassicalMLModel(
        "random_forest",
        task_type=TaskType.BINARY_CLASSIFICATION,
        n_estimators=12,
        random_state=19,
    )
    model2.fit(X, 1 - y)
    real_replace = os.replace
    call_counter = {"n": 0}

    def _fail_second_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        call_counter["n"] += 1
        if call_counter["n"] == 2:
            raise OSError("Simulated disk I/O failure during artifact swap")
        real_replace(src, dst)

    with patch("omniforge.models.base.os.replace", side_effect=_fail_second_replace):
        with pytest.raises(ModelSerializationError):
            model2.save(enc_target, encrypt=True, encryption_key=enc_key)

    # Zero leftover .tmp files, and original model still passes HMAC verification & predicts original labels!
    assert list(tmp_path.glob("*.tmp")) == []
    still_valid = BaseOmniForgeModel.load(enc_target, encrypted=True, encryption_key=enc_key)
    np.testing.assert_array_equal(still_valid.predict(X), model.predict(X))


# ---------------------------------------------------------------------------
# P1-1: SSRF DNS-rebinding protection via per-hop IP pinning + SNI/Host
# ---------------------------------------------------------------------------


def test_http_safe_pins_resolved_ip_against_dns_rebinding_and_preserves_host_and_sni(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookup_count = {"calls": 0}

    def _rebinding_getaddrinfo(host: str, port: Any, *args: Any, **kwargs: Any) -> list[Any]:
        lookup_count["calls"] += 1
        if lookup_count["calls"] == 1:
            # First lookup returns vetted public IP.
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port or 443))]
        # Second lookup (DNS rebinding TTL=0 attack!) attempts to return AWS IMDS 169.254.169.254!
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", port or 443))]

    monkeypatch.setattr(socket, "getaddrinfo", _rebinding_getaddrinfo)

    seen_requests: list[httpx.Request] = []

    def _transport_handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(200, headers={"content-type": "text/csv"}, content=b"a,b\n1,2\n3,4\n")

    ingestor = DataIngestionEngine()
    df = ingestor._ingest_http_file(
        "https://rebinding.example.com:8443/dataset.csv",
        fmt="csv",
        _transport=httpx.MockTransport(_transport_handler),
    )
    assert len(df) == 2
    assert lookup_count["calls"] == 1
    assert len(seen_requests) == 1
    req = seen_requests[0]
    # Transport connected to the vetted pinned IP, NOT the hostname (so httpx cannot re-resolve to 169.254.169.254)!
    assert req.url.host == "93.184.216.34"
    assert req.url.port == 8443
    assert req.headers["Host"] == "rebinding.example.com:8443"
    assert req.extensions.get("sni_hostname") == "rebinding.example.com"


def test_http_safe_falls_back_across_multiple_vetted_ips_and_repins_redirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dns_map = {
        "primary.example.com": ["198.51.100.10", "198.51.100.11"],  # Wait: TEST-NET-2 is reserved? Check public IPs!
    }
    _ = dns_map
    # Use globally routable public IPs (93.184.216.34, 1.1.1.1, 8.8.8.8) that pass ip.is_global.
    host_ips = {
        "origin.example.com": ["93.184.216.34", "1.1.1.1"],
        "cdn.example.net": ["8.8.8.8"],
    }

    def _fake_getaddrinfo(host: str, port: Any, *args: Any, **kwargs: Any) -> list[Any]:
        ips = host_ips[host]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 443)) for ip in ips]

    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo)
    attempts: list[tuple[str, str, str | None]] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        attempts.append((request.url.host, request.headers.get("Host", ""), request.headers.get("Authorization")))
        if request.url.host == "93.184.216.34":
            raise httpx.ConnectError("first A record unreachable")
        if request.url.host == "1.1.1.1":
            return httpx.Response(302, headers={"location": "https://cdn.example.net/final.csv"})
        assert request.url.host == "8.8.8.8"
        return httpx.Response(200, headers={"content-type": "text/csv"}, content=b"x,y\n10,20\n")

    ingestor = DataIngestionEngine()
    df = ingestor._ingest_http_file(
        "https://origin.example.com/start.csv",
        fmt="csv",
        headers={"Authorization": "Bearer top-secret"},
        _transport=httpx.MockTransport(_handler),
    )
    assert df.to_dict(orient="list") == {"x": [10], "y": [20]}
    assert attempts == [
        ("93.184.216.34", "origin.example.com", "Bearer top-secret"),
        ("1.1.1.1", "origin.example.com", "Bearer top-secret"),
        ("8.8.8.8", "cdn.example.net", None),
    ]

    with pytest.raises(SecurityError, match="Backslashes"):
        resolve_and_validate_outbound_http_url("https://example.com\\@169.254.169.254/latest")


# ---------------------------------------------------------------------------
# P1-2a: JWTManager requires exp and sub in both PyJWT and stdlib fallback
# ---------------------------------------------------------------------------


def test_jwt_manager_rejects_tokens_without_exp_or_sub_in_both_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jwt as pyjwt

    import omniforge.core.security as sec_mod

    secret = "jwt-remediation-secret-key-at-least-32-bytes!!"
    mgr = JWTManager(secret_key=secret)

    # 1. Valid token works in both PyJWT and stdlib fallback modes.
    valid_token = mgr.create_token(Principal(subject="alice", roles=["analyst"], tenant_id="t1"))
    assert mgr.verify_token(valid_token).subject == "alice"

    # 2. Signed token missing 'exp' MUST be rejected in PyJWT mode.
    no_exp_token = pyjwt.encode({"sub": "alice", "roles": ["admin"]}, secret, algorithm="HS256")
    with pytest.raises(SecurityError):
        mgr.verify_token(no_exp_token)

    # 3. Signed token missing 'sub' MUST be rejected in PyJWT mode.
    no_sub_token = pyjwt.encode({"exp": 4102444800, "roles": ["admin"]}, secret, algorithm="HS256")
    with pytest.raises(SecurityError):
        mgr.verify_token(no_sub_token)

    # 4. Stdlib fallback mode must enforce the exact same requirements (including rejecting bool exp).
    monkeypatch.setattr(sec_mod, "_HAS_PYJWT", False)
    assert mgr.verify_token(valid_token).subject == "alice"
    with pytest.raises(SecurityError):
        mgr.verify_token(no_exp_token)
    with pytest.raises(SecurityError):
        mgr.verify_token(no_sub_token)

    header_b64 = JWTManager._b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode("utf-8"))
    bool_exp_payload = JWTManager._b64url(json.dumps({"sub": "alice", "exp": True}).encode("utf-8"))
    sig = JWTManager._b64url(
        hmac.new(secret.encode("utf-8"), f"{header_b64}.{bool_exp_payload}".encode("ascii"), hashlib.sha256).digest()
    )
    with pytest.raises(SecurityError):
        mgr.verify_token(f"{header_b64}.{bool_exp_payload}.{sig}")


# ---------------------------------------------------------------------------
# Loop 2: Analytical LQR/KKT exactness, live TLS loopback SNI, RBAC, Plugins,
#         Copilot studio routes, Capabilities probe, and Config parity
# ---------------------------------------------------------------------------


def test_mpc_unconstrained_matches_closed_form_batch_lqr_optimum() -> None:
    """When bounds are inactive (-inf, +inf), MPC must match the exact closed-form batch LQ optimum."""
    A = np.array([[0.9, 0.25], [-0.1, 0.85]])
    B = np.array([[0.5, 0.1], [0.05, 0.4]])
    Q = np.array([[2.5, 0.3], [0.3, 1.8]])
    R = np.array([[0.6, 0.05], [0.05, 0.9]])
    H = 5
    n_x, n_u = 2, 2
    twin = DigitalTwinEngine(A=A, B=B, H=np.eye(n_x), Q=np.eye(n_x) * 1e-3, R=np.eye(n_x) * 1e-2, x0=np.zeros(n_x))
    ctrl = ClosedLoopMPCController(
        twin,
        horizon=H,
        Q=Q,
        R=R,
        u_min=np.full(n_u, -np.inf),
        u_max=np.full(n_u, np.inf),
        x_min=np.full(n_x, -np.inf),
        x_max=np.full(n_x, np.inf),
    )
    x0 = np.array([3.2, -1.7])
    r_path = np.array([[1.0, 0.5], [0.8, 0.4], [0.5, 0.2], [0.2, 0.1], [0.0, 0.0]])

    # Build batch prediction X_vec = S_x x0 + S_u U_vec where X_vec = [x_1; ...; x_H].
    S_x = np.zeros((H * n_x, n_x))
    S_u = np.zeros((H * n_x, H * n_u))
    A_pow = np.eye(n_x)
    for k in range(H):
        A_pow = A @ A_pow
        S_x[k * n_x : (k + 1) * n_x, :] = A_pow
        for j in range(k + 1):
            term = np.linalg.matrix_power(A, k - j) @ B
            S_u[k * n_x : (k + 1) * n_x, j * n_u : (j + 1) * n_u] = term

    Q_bar = np.kron(np.eye(H), Q)
    R_bar = np.kron(np.eye(H), R)
    lhs = S_u.T @ Q_bar @ S_u + R_bar
    rhs = S_u.T @ Q_bar @ (r_path.ravel() - S_x @ x0)
    u_exact = np.linalg.solve(lhs, rhs)

    res = ctrl.solve(x0=x0, r_path=r_path)
    assert res.success is True
    assert res.slack_used == 0.0
    np.testing.assert_allclose(res.full_control_sequence.ravel(), u_exact, atol=1e-5, rtol=1e-5)


def test_model_unfitted_save_and_checksum_tamper_and_clean_dump_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from omniforge.core.exceptions import ModelNotFittedError

    monkeypatch.setenv("OMNIFORGE_KEY_DIR", str(tmp_path / "keys"))
    unfitted = ClassicalMLModel("random_forest", task_type=TaskType.BINARY_CLASSIFICATION, n_estimators=4, random_state=1)
    with pytest.raises(ModelNotFittedError):
        unfitted.save(tmp_path / "never.joblib")

    X = pd.DataFrame({"a": [0.0, 1.0, 0.1, 0.9], "b": [1.0, 0.0, 0.8, 0.2]})
    y = pd.Series([0, 1, 0, 1])
    unfitted.fit(X, y)

    fresh_target = tmp_path / "failed_dump.joblib"
    with patch("omniforge.models.base.joblib.dump", side_effect=RuntimeError("boom")):
        with pytest.raises(ModelSerializationError):
            unfitted.save(fresh_target)
    assert not fresh_target.exists()
    assert not fresh_target.with_suffix(".joblib.meta.json").exists()
    assert list(tmp_path.glob("*.tmp")) == []

    good_target = tmp_path / "signed_rf.joblib"
    unfitted.save(good_target)
    meta_path = good_target.with_suffix(".joblib.meta.json")
    meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
    meta_obj["artifact_checksum"] = "0" * 64
    meta_path.write_text(json.dumps(meta_obj), encoding="utf-8")
    with pytest.raises(SecurityError, match="checksum"):
        BaseOmniForgeModel.load(good_target)


def test_live_https_loopback_tls_sni_and_host_header_smuggling_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real TCP+TLS socket test proving pinned-IP HTTPS sends SNI and validates X.509 SAN."""
    import datetime as dt
    import ssl
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "rebinding.example.test")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(dt.datetime.now(dt.UTC) - dt.timedelta(days=1))
        .not_valid_after(dt.datetime.now(dt.UTC) + dt.timedelta(days=2))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("rebinding.example.test")]), critical=False)
        .sign(key, hashes.SHA256())
    )
    cert_file = tmp_path / "cert.pem"
    key_file = tmp_path / "key.pem"
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )

    observed: dict[str, str | None] = {"sni": None, "host": None}

    class _CSVHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            observed["host"] = self.headers.get("Host")
            body = b"col1,col2\n7,8\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/csv")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_ctx.load_cert_chain(certfile=str(cert_file), keyfile=str(key_file))

    def _sni_cb(ssl_sock: ssl.SSLSocket, server_name: str | None, _ctx: ssl.SSLContext) -> None:
        observed["sni"] = server_name

    server_ctx.sni_callback = _sni_cb

    httpd = HTTPServer(("127.0.0.1", 0), _CSVHandler)
    port = httpd.server_port
    httpd.socket = server_ctx.wrap_socket(httpd.socket, server_side=True)
    srv_thread = threading.Thread(target=httpd.handle_request, daemon=True)
    srv_thread.start()

    monkeypatch.setenv("OMNIFORGE_ALLOW_PRIVATE_HTTP_HOSTS", "true")
    monkeypatch.setenv("SSL_CERT_FILE", str(cert_file))

    orig_getaddrinfo = socket.getaddrinfo
    dns_hosts: list[str] = []

    def _rebinding_loopback_dns(host: str, p: Any, *args: Any, **kwargs: Any) -> list[Any]:
        dns_hosts.append(str(host))
        if host == "rebinding.example.test":
            if dns_hosts.count("rebinding.example.test") == 1:
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", p or port))]
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", p or port))]
        return orig_getaddrinfo(host, p, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", _rebinding_loopback_dns)

    client_ctx = ssl.create_default_context(cafile=str(cert_file))
    transport = httpx.HTTPTransport(verify=client_ctx)
    try:
        df = DataIngestionEngine()._ingest_http_file(
            f"https://rebinding.example.test:{port}/live.csv",
            fmt="csv",
            headers={"Host": "evil-smuggled-host.internal"},
            timeout=5.0,
            _transport=transport,
        )
    finally:
        httpd.server_close()
        srv_thread.join(timeout=2.0)

    assert df.to_dict(orient="list") == {"col1": [7], "col2": [8]}
    assert dns_hosts.count("rebinding.example.test") == 1
    assert "127.0.0.1" in dns_hosts
    assert observed["sni"] == "rebinding.example.test"
    assert observed["host"] == f"rebinding.example.test:{port}"


def test_jwt_algorithm_allowlist_and_auth_settings_wiring(monkeypatch: pytest.MonkeyPatch) -> None:
    import omniforge.core.security as sec_mod
    from omniforge.api.auth import _build_jwt_manager
    from omniforge.config.settings import get_settings

    secret = "jwt-alg-test-secret-key-with-high-entropy-123456"
    with pytest.raises(SecurityError, match="Unsupported or insecure JWT algorithm"):
        JWTManager(secret_key=secret, algorithm="none")
    with pytest.raises(SecurityError, match="Unsupported or insecure JWT algorithm"):
        JWTManager(secret_key=secret, algorithm="RS256")
    with pytest.raises(SecurityError, match="expiration_minutes"):
        JWTManager(secret_key=secret, expiration_minutes=0)

    # Verify HS384 round-trips in both PyJWT and stdlib fallback, and header alg mismatch is rejected.
    mgr384 = JWTManager(secret_key=secret, algorithm="HS384", expiration_minutes=15)
    tok384 = mgr384.create_token(Principal(subject="bob", roles=["mlops_engineer"], tenant_id="t1"))
    assert mgr384.verify_token(tok384).subject == "bob"

    monkeypatch.setattr(sec_mod, "_HAS_PYJWT", False)
    assert mgr384.verify_token(tok384).subject == "bob"
    mgr256 = JWTManager(secret_key=secret, algorithm="HS256")
    with pytest.raises(SecurityError):
        mgr256.verify_token(tok384)
    monkeypatch.setattr(sec_mod, "_HAS_PYJWT", True)

    monkeypatch.setenv("OMNIFORGE_SECURITY__JWT_SECRET_KEY", secret)
    monkeypatch.setenv("OMNIFORGE_SECURITY__JWT_ALGORITHM", "HS384")
    monkeypatch.setenv("OMNIFORGE_SECURITY__JWT_EXPIRATION_MINUTES", "45")
    get_settings.cache_clear()
    try:
        wired = _build_jwt_manager()
        assert wired._algorithm == "HS384"
        assert wired._expiration_minutes == 45
    finally:
        get_settings.cache_clear()


def test_rbac_model_promote_enforced_on_api_v1_stage_and_copilot_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from omniforge.api.auth import _request_principal_ctx, jwt_manager
    from omniforge.api.main import app
    from omniforge.copilot.memory import CopilotMemory
    from omniforge.copilot.tools import CopilotToolRegistry
    from omniforge.core.exceptions import AuthorizationError

    monkeypatch.setenv("OMNIFORGE_KEY_DIR", str(tmp_path / "keys"))
    client = TestClient(app)

    # Seed a model in tenant 'rbac-test-tenant' using mlops_engineer / data_scientist.
    ds_principal = Principal(subject="ds-user", roles=["data_scientist"], tenant_id="rbac-test-tenant")
    mlops_principal = Principal(subject="mlops-user", roles=["mlops_engineer"], tenant_id="rbac-test-tenant")
    viewer_principal = Principal(subject="viewer-user", roles=["viewer"], tenant_id="rbac-test-tenant")

    ds_headers = {"Authorization": f"Bearer {jwt_manager.create_token(ds_principal)}"}
    mlops_headers = {"Authorization": f"Bearer {jwt_manager.create_token(mlops_principal)}"}

    records = [
        {"f1": float(i % 5) + 0.1 * i, "f2": float((i * 3) % 7), "target": int(i % 2)}
        for i in range(40)
    ]
    r_train = client.post(
        "/api/v1/automl/train",
        json={
            "records": records,
            "target_column": "target",
            "time_budget_seconds": 5,
            "register_as": "rbac_guarded_model",
        },
        headers=ds_headers,
    )
    assert r_train.status_code == 200, r_train.text

    # data_scientist has model:deploy -> Archiving succeeds (200), but promoting to Production MUST be 403!
    r_arch = client.post(
        "/api/v1/registry/rbac_guarded_model/v1/stage",
        json={"target_stage": "Archived"},
        headers=ds_headers,
    )
    assert r_arch.status_code == 200, r_arch.text

    r_prod_denied = client.post(
        "/api/v1/registry/rbac_guarded_model/v1/stage",
        json={"target_stage": "Production"},
        headers=ds_headers,
    )
    assert r_prod_denied.status_code == 403, r_prod_denied.text

    # mlops_engineer has BOTH model:deploy and model:promote -> promoting to Production succeeds (200)!
    r_prod_ok = client.post(
        "/api/v1/registry/rbac_guarded_model/v1/stage",
        json={"target_stage": "Production"},
        headers=mlops_headers,
    )
    assert r_prod_ok.status_code == 200, r_prod_ok.text
    assert r_prod_ok.json()["stage"] == "Production"

    # Copilot tool_promote_model must ALSO enforce RBAC when a request principal is bound!
    tool_reg = CopilotToolRegistry(CopilotMemory())
    tok_viewer = _request_principal_ctx.set(viewer_principal)
    try:
        with pytest.raises(AuthorizationError):
            tool_reg.invoke("promote_model", {"model_name": "rbac_guarded_model", "version": 1, "stage": "Staging"})
    finally:
        _request_principal_ctx.reset(tok_viewer)

    tok_ds = _request_principal_ctx.set(ds_principal)
    try:
        with pytest.raises(AuthorizationError):
            tool_reg.invoke("promote_model", {"model_name": "rbac_guarded_model", "version": 1, "stage": "Production"})
    finally:
        _request_principal_ctx.reset(tok_ds)


def test_copilot_list_studios_routes_all_exist_in_openapi_and_capabilities_endpoint_works() -> None:
    from fastapi.testclient import TestClient

    from omniforge.api.auth import jwt_manager
    from omniforge.api.main import app
    from omniforge.config.settings import SecuritySettings
    from omniforge.copilot.memory import CopilotMemory
    from omniforge.copilot.tools import CopilotToolRegistry

    openapi_paths = set(app.openapi()["paths"].keys())
    reg = CopilotToolRegistry(CopilotMemory())
    res = reg.invoke("list_studios", {})
    assert res["count"] == 15
    assert [s["id"] for s in res["studios"]] == [f"s{i:02d}" for i in range(15)]
    for studio in res["studios"]:
        assert studio["route"] in openapi_paths, f"Studio {studio['id']} route {studio['route']} missing from OpenAPI!"
        assert len(studio["endpoints"]) >= 1
        for ep in studio["endpoints"]:
            assert ep in openapi_paths, f"Discovered endpoint {ep} missing from OpenAPI!"

    ov_s08 = reg.invoke("studio_overview", {"studio_id": "s08"})
    assert ov_s08["route"] in openapi_paths

    # /api/system/capabilities must return 401 unauthenticated and 200 with valid token.
    client = TestClient(app)
    assert client.get("/api/system/capabilities").status_code == 401
    assert client.get("/api/v1/system/capabilities").status_code == 401

    tok = jwt_manager.create_token(Principal(subject="probe", roles=["viewer"], tenant_id="default-tenant"))
    r_cap = client.get("/api/system/capabilities", headers={"Authorization": f"Bearer {tok}"})
    assert r_cap.status_code == 200
    body = r_cap.json()
    assert "core-runtime" in body["available_packs"]
    assert "packs" in body and "features" in body

    # .env.example must align with SecuritySettings().auto_bootstrap_local_ui == False.
    env_example = (Path(__file__).resolve().parents[2] / ".env.example").read_text(encoding="utf-8")
    assert "OMNIFORGE_SECURITY__AUTO_BOOTSTRAP_LOCAL_UI=false" in env_example
    assert SecuritySettings().auto_bootstrap_local_ui is False


def test_domain_plugin_registry_discovers_all_18_class_entry_points_and_skips_abstract() -> None:
    from omniforge.domains.plugin_sdk import BaseDomainPlugin, DomainMetadata, DomainPluginRegistry

    reg = DomainPluginRegistry()
    loaded = reg.discover_entry_points()
    assert loaded == 18
    assert len(reg.list_names()) == 18
    assert "finance" in reg.list_names() and "healthcare" in reg.list_names()

    class _AbstractCustomPlugin(BaseDomainPlugin):
        metadata = DomainMetadata(name="abs_dom", display_name="Abstract", description="Abstract")

    class _ConcreteCustomPlugin(BaseDomainPlugin):
        metadata = DomainMetadata(name="custom_aerospace", display_name="Aerospace", description="Aerospace telemetry")

        def engineer_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
            return df.copy()

        def compute_domain_kpis(self, df: pd.DataFrame, *, predictions: np.ndarray | None = None) -> dict[str, float]:
            _ = predictions
            return {"rows": float(len(df))}

        def generate_sample_dataset(self, n_samples: int = 200, random_state: int = 42) -> pd.DataFrame:
            _ = random_state
            return pd.DataFrame({"altitude": np.zeros(n_samples)})

    class _FakeEP:
        def __init__(self, name: str, target: Any) -> None:
            self.name = name
            self._target = target

        def load(self) -> Any:
            return self._target

    custom_reg = DomainPluginRegistry()
    fake_eps = [
        _FakeEP("abs", _AbstractCustomPlugin),
        _FakeEP("cls", _ConcreteCustomPlugin),
        _FakeEP("inst", _ConcreteCustomPlugin()),
    ]
    with patch("omniforge.domains.plugin_sdk.importlib_metadata.entry_points", return_value=fake_eps):
        n = custom_reg.discover_entry_points(group="custom.group")
    assert n == 2
    assert custom_reg.list_names() == ["custom_aerospace"]

    # skip_existing=True skips already-registered slugs without calling load()/re-instantiating.
    load_calls = {"n": 0}

    class _CountingEP:
        def __init__(self, name: str, target: Any) -> None:
            self.name = name
            self._target = target

        def load(self) -> Any:
            load_calls["n"] += 1
            return self._target

    with patch(
        "omniforge.domains.plugin_sdk.importlib_metadata.entry_points",
        return_value=[_CountingEP("custom_aerospace", _ConcreteCustomPlugin)],
    ):
        skipped = custom_reg.discover_entry_points(group="custom.group", skip_existing=True)
    assert skipped == 0
    assert load_calls["n"] == 0


# ---------------------------------------------------------------------------
# Loop 3: UI import decoupling, HTTP error response sanitization, and hygiene
# ---------------------------------------------------------------------------


def test_importing_omniforge_ui_does_not_build_fastapi_app_or_touch_disk(tmp_path: Path) -> None:
    """Fresh interpreter check: importing omniforge.ui.* must not import omniforge.api.main."""
    import subprocess
    import sys

    code = (
        "import sys; "
        "import omniforge.ui; "
        "import omniforge.ui.state; "
        "import omniforge.ui.app; "
        "assert 'omniforge.api.main' not in sys.modules, 'api.main eagerly imported!'; "
        "assert len(omniforge.ui.STUDIOS) == 14"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"STDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
    assert not (tmp_path / ".omniforge_workspace").exists()


def test_omniforge_error_http_handler_omits_internal_cause_and_context() -> None:
    from fastapi.testclient import TestClient

    from omniforge.api.auth import jwt_manager
    from omniforge.api.main import app
    from omniforge.core.exceptions import DataIngestionError

    err = DataIngestionError(
        "Safe user-facing message.",
        context={"internal_path": "/srv/secret/db.sqlite"},
        cause=RuntimeError("driver credentials leak"),
    )
    # Internal Python telemetry retains context and cause by default.
    internal_dict = err.to_dict()
    assert "context" in internal_dict and "cause" in internal_dict
    # External HTTP payload omits both.
    public_dict = err.to_dict(include_internal=False)
    assert "context" not in public_dict and "cause" not in public_dict

    client = TestClient(app)
    tok = jwt_manager.create_token(Principal(subject="u1", roles=["analyst"], tenant_id="default-tenant"))
    r_err = client.get(
        "/api/v1/domains/nonexistent_domain_xyz/sample",
        headers={"Authorization": f"Bearer {tok}"},
    )
    assert r_err.status_code == 400
    body = r_err.json()
    assert body["error_type"] == "DomainPluginError"
    assert "context" not in body
    assert "cause" not in body


def test_repo_hygiene_makefile_readme_ci_and_build_headers() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    assert "omniferge" not in readme
    assert "/s14/root-cause" not in readme
    assert "omniforge automl finance.csv" in readme

    req_build = (repo_root / "requirements-build.in").read_text(encoding="utf-8")
    assert "OmniForge v4.0.0" in req_build

    makefile = (repo_root / "Makefile").read_text(encoding="utf-8")
    assert "HOST ?= 127.0.0.1" in makefile

    ci_yml = (repo_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "git status --porcelain -- src/omniforge/ui/static_app" in ci_yml


def test_ipv6_dns_pinning_concurrent_audit_reader_and_e2e_copilot_http_403(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from omniforge.api.auth import jwt_manager
    from omniforge.api.main import app

    # 1. IPv6 AAAA record DNS-rebinding pinning formats [2606:4700:4700::1111] cleanly and sets Host + SNI.
    def _ipv6_dns(host: str, port: Any, *args: Any, **kwargs: Any) -> list[Any]:
        assert host == "v6.example.com"
        return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2606:4700:4700::1111", port or 443, 0, 0))]

    monkeypatch.setattr(socket, "getaddrinfo", _ipv6_dns)
    seen: list[httpx.Request] = []

    def _v6_handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, headers={"content-type": "text/csv"}, content=b"k,v\n1,9\n")

    df = DataIngestionEngine()._ingest_http_file(
        "https://v6.example.com:8443/v6.csv",
        fmt="csv",
        _transport=httpx.MockTransport(_v6_handler),
    )
    assert df.to_dict(orient="list") == {"k": [1], "v": [9]}
    assert seen[0].url.host == "2606:4700:4700::1111"
    assert seen[0].headers["Host"] == "v6.example.com:8443"
    assert seen[0].extensions.get("sni_hostname") == "v6.example.com"

    # 2. Concurrent AuditLogger writer + verify_with_anchor reader linearizability.
    monkeypatch.setenv("OMNIFORGE_KEY_DIR", str(tmp_path / "keys"))
    log_file = tmp_path / "linearizable.jsonl"
    anchor_file = tmp_path / "linearizable_anchor.jsonl"
    ck = "linearizable-checkpoint-secret-key-32b!!"
    writer = AuditLogger(log_path=log_file, anchor_path=anchor_file, checkpoint_key=ck, anchor_every=1)
    writer.log_event(action="init", resource_type="r", resource_id="0")

    def _run_job(idx: int) -> bool:
        if idx % 2 == 0:
            writer.log_event(action="write", resource_type="r", resource_id=str(idx))
            return True
        ok, _cnt, detail = writer.verify_with_anchor()
        assert ok is True, detail
        return ok

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(_run_job, range(40)))
    assert all(results)

    # 3. End-to-end HTTP /api/v1/copilot/chat RBAC enforcement:
    #    - viewer cannot train AutoML (403)
    #    - data_scientist can train + register to Staging (200), but cannot promote to Production (403)
    #    - viewer cannot register model (403)
    client = TestClient(app)
    viewer_tok = jwt_manager.create_token(Principal(subject="v1", roles=["viewer"], tenant_id="cop-rbac-t1"))
    ds_tok = jwt_manager.create_token(Principal(subject="v1", roles=["data_scientist"], tenant_id="cop-rbac-t1"))
    records = [{"x1": float(i), "x2": float(i % 3), "target": int(i % 2)} for i in range(30)]

    r_viewer_train = client.post(
        "/api/v1/copilot/chat",
        json={"message": "train automl target=target", "records": records},
        headers={"Authorization": f"Bearer {viewer_tok}"},
    )
    assert r_viewer_train.status_code == 403, r_viewer_train.text

    r_train = client.post(
        "/api/v1/copilot/chat",
        json={"message": "train automl target=target", "records": records},
        headers={"Authorization": f"Bearer {ds_tok}"},
    )
    assert r_train.status_code == 200, r_train.text

    r_forbid = client.post(
        "/api/v1/copilot/chat",
        json={"message": "register model"},
        headers={"Authorization": f"Bearer {viewer_tok}"},
    )
    assert r_forbid.status_code == 403, r_forbid.text

    r_ds_reg = client.post(
        "/api/v1/copilot/chat",
        json={"message": "register model"},
        headers={"Authorization": f"Bearer {ds_tok}"},
    )
    assert r_ds_reg.status_code == 200, r_ds_reg.text

    r_ds_promote = client.post(
        "/api/v1/copilot/chat",
        json={"message": "promote model"},
        headers={"Authorization": f"Bearer {ds_tok}"},
    )
    assert r_ds_promote.status_code == 403, r_ds_promote.text


