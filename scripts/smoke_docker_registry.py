"""Build-independent smoke test for registry persistence across container replacement."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import time
from typing import Any
from uuid import uuid4

# Run inside each container so the smoke test requires no published ports.
_PROBE = r"""
import json
import os
import sys
import urllib.request
from datetime import date, timedelta
from pathlib import Path

from omniforge.models.base import BaseOmniForgeModel

mode = sys.argv[1]
port = sys.argv[2]
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
root = f"http://127.0.0.1:{port}"
key = os.environ["OMNIFORGE_SECURITY__API_KEY"]

def call(path, payload=None, *, bearer=None, content_type="application/json"):
    headers = {"Authorization": f"Bearer {bearer}"} if bearer else {"X-API-Key": key}
    headers["X-Session-ID"] = "registry-smoke"
    if payload is not None:
        headers["Content-Type"] = content_type
        if not isinstance(payload, bytes):
            payload = json.dumps(payload).encode("utf-8")
    with opener.open(urllib.request.Request(root + path, payload, headers), timeout=90) as response:
        return json.load(response)

token = call("/api/v1/auth/token", {
    "username": "registry-smoke", "role": "admin", "tenant_id": "persistence-test",
})["access_token"]
rest_models = "/api/v1/registry/models"
workbench = "/api/workbench"

if mode == "write":
    assert call(rest_models, bearer=token)["versions"] == []
    assert call(workbench + "/s11/overview", bearer=token)["registered_models"] == []
    rows = [{"x": i, "y": float(i * 10.5 + i % 3)} for i in range(64)]
    call("/api/v1/automl/train", {
        "records": rows, "target_column": "y", "register_as": "rest-smoke", "time_budget_seconds": 5,
    }, bearer=token)

    # Controlled fixture: each consecutive block contains both outcomes in both
    # groups. Reserve the last five complete blocks for the promotion audit.
    outcomes_and_groups = ((0, "A"), (1, "A"), (0, "B"), (1, "B"))
    csv = "observed_at,x,target,group\n" + "".join(
        f"{date(2025, 1, 1) + timedelta(days=i)},{10 if label else -10},{label},{group}\n"
        for i, (label, group) in enumerate(outcomes_and_groups * 20)
    )
    boundary = "omniforge-smoke-boundary"
    payload = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="sample.csv"\r\n'
               f'Content-Type: text/csv\r\n\r\n{csv}\r\n--{boundary}--\r\n').encode()
    call(workbench + "/s01/upload", payload, bearer=token, content_type=f"multipart/form-data; boundary={boundary}")
    call(workbench + "/targets", {"targets": ["target"]}, bearer=token)
    call(workbench + "/exclusions", {"excluded_columns": ["group"]}, bearer=token)
    call(workbench + "/split-lock", {
        "strategy": "time_series", "time_col": "observed_at", "target_col": "target", "test_size": 0.25,
    }, bearer=token)
    training = call(workbench + "/s14/quick-train", {"method": "random_forest", "target": "target"}, bearer=token)
    assert training["evaluation"] == {"partition": "test", "source": "split_lock", "n_train": 60, "n_test": 20}, training["evaluation"]
    fairness = call(workbench + "/s07/fairness", {"sensitive_col": "group"}, bearer=token)
    assert fairness["fairness"]["passed"], fairness["fairness"]
    assert fairness["promotion_evidence"]["n_holdout_rows"] == 20
    call(workbench + "/s11/register", {"model_name": "workbench-smoke", "stage": "Production"}, bearer=token)
else:
    promoted = call(workbench + "/s11/promote", {
        "model_name": "workbench-smoke", "version": 1, "stage": "Production",
    }, bearer=token)
    assert promoted["updated"]["fairness_evidence"]["passed"]

rest = call(rest_models, bearer=token)["versions"]
wb = call(workbench + "/s11/overview", bearer=token)["registered_models"]
expected = [("rest-smoke", 1, "Staging"), ("workbench-smoke", 1, "Production")]
assert [(v["model_name"], v["version"], v["stage"]) for v in rest] == expected
assert [(v["model_name"], v["version"], v["stage"]) for v in wb] == expected
assert wb[1]["fairness_evidence"]["artifact_checksum"] == wb[1]["checksum_sha256"]
assert wb[1]["fairness_evidence"]["version"] == 1
assert rest == wb
for version in rest:
    path = Path(version["artifact_path"])
    assert path.is_relative_to("/var/lib/omniforge/tenants"), path
    assert BaseOmniForgeModel.load(
        path, encrypted=version["encrypted_at_rest"], expected_artifact_checksum=version["checksum_sha256"],
    ).is_fitted_
print(json.dumps([v["checksum_sha256"] for v in rest] + [wb[1]["fairness_evidence"]["holdout_fingerprint"]]))
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Tag of the built OmniForge image")
    config = parser.parse_args()

    prefix = f"omniforge-registry-smoke-{uuid4().hex[:12]}"
    volume = f"{prefix}-data"
    settings = {
        f"OMNIFORGE_SECURITY__{name}": secrets.token_urlsafe(40)
        for name in ("API_KEY", "ARTIFACT_SIGNING_KEY", "ENCRYPTION_KEY", "JWT_SECRET_KEY")
    }
    settings.update(OMNIFORGE_REQUIRE_FAIRNESS_FOR_PRODUCTION="true", OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2")
    env_args = [arg for name in settings for arg in ("--env", name)]
    env = {**os.environ, **settings}

    def docker(*args: str, **options: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["docker", *args], env=env, text=True, **options)

    def phase(name: str, mode: str, port: str, action: str) -> tuple[str, list[str]]:
        container_id = docker(
            "run", "-d", "--name", name, "--mount", f"source={volume},target=/var/lib/omniforge",
            *env_args, config.image, mode, check=True, stdout=subprocess.PIPE,
        ).stdout.strip()
        try:
            deadline = time.monotonic() + 60
            while True:
                ready = docker("exec", name, "python", "/app/healthcheck.py", port, stdout=subprocess.DEVNULL)
                if ready.returncode == 0:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"{mode} container did not become healthy")
                time.sleep(1)
            result = docker(
                "exec", "-i", name, "python", "-", action, port, input=_PROBE, check=True, stdout=subprocess.PIPE,
            )
            return container_id, json.loads(result.stdout.strip().splitlines()[-1])
        except Exception:
            docker("logs", "--tail", "25", name)
            raise
        finally:
            docker("rm", "-f", name, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    docker("volume", "create", volume, check=True, stdout=subprocess.DEVNULL)
    try:
        writer, before = phase(f"{prefix}-api", "api", "8000", "write")
        reader, after = phase(f"{prefix}-ui", "ui", "8501", "read")
        if writer == reader or before != after:
            raise RuntimeError("Registry versions, artifacts, or evidence changed after container replacement.")
        print("Verified the shared tenant registry through both APIs, signed artifacts, and bound evidence after container replacement.")
    finally:
        docker("volume", "rm", volume, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
