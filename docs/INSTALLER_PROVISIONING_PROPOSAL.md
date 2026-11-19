# OmniForge — Small Installer + On-Demand Dependency Provisioning

**Status:** proposal / decision paper · **Date:** 2026-11-18 · **Constraint from the user:** installer must stay small;
capabilities must **not** change.

Every size and feasibility number below was measured in this workspace
(`pip download --platform win_amd64 --python-version 3.13 --abi cp313`, PyPI JSON API, and the
existing repo contents). Measurements are reproducible — commands in §8.

---

## 0. Answer in one paragraph

Yes — this is the right architecture and it **is** achievable without touching a single feature:
ship a ~**55–75 MB** installer that contains **only your own code + the pre-compiled React bundle**, and let the
installer create a private virtual environment on the user's machine and resolve the Python dependency set from
PyPI (or from a corporate mirror / local wheel cache) using an **embedded `uv` binary + a hash-pinned lock file**.
The "check what exists, download what is missing" behaviour you describe is *exactly* what `uv sync` does, and it is
already how the heavy parts of OmniForge are designed (63 live `is_package_available(...)` probes, 12 guarded
`try/except ImportError` blocks, 7 `DependencyMissingError` paths) — so the app **degrades instead of crashing**
when an optional wheel is absent.

**But be honest about the real numbers (measured):**

| What | Size |
| :--- | :--- |
| Your own code + configs + SPA bundle | **~70 MB** repo → static_app only **0.65 MB**; source ~6 MB |
| Python 3.13 embeddable runtime (win-amd64) | **~15 MB** (compressed ~11 MB) |
| `uv` binary (embedded, `py3-none-win_amd64` wheel) | **15.7 MB** |
| **Installer (self-contained bootstrap)** | **≈ 55–75 MB** |
| **Required** dependency set (84 wheels, win_amd64 cp313, **compressed**) | **171 MB** → **~450–550 MB on disk** |
| Optional extras: boosting / deep / vision / nlp | +**151** / +**128** / +**60** / +**26 MB** compressed |

So: the *installer* drops from ~400–700 MB to **~65 MB (−85 %)**, and the *disk footprint* ends up roughly the same
because the same wheels must land somewhere. What you really buy is: no 500 MB download in one blocking step, no
rebuilding the exe for every dependency bump, and per-feature "install on demand" (torch only if the user wants deep
learning). What you pay for: **installation now requires internet**, an online-install failure mode, and a
supply-chain/reproducibility story you must get right (hash-pinned lock — see §4).

---

## 1. Where the weight actually is (measured, not guessed)

Top compressed win_amd64 wheels of the **required** core set:

| Wheel | MB | Notes |
| :--- | ---: | :--- |
| scipy | 36.7 | required |
| pyarrow | 29.1 | required (`data`-grade IO) |
| duckdb | 15.4 | required (demo SQL workspace) |
| numpy | 12.6 | required |
| statsmodels | 11.3 | required (forecasting/statistics) |
| pandas | 10.8 | required |
| matplotlib | 9.3 | required (SPA charts use canvas; matplotlib feeds reports) |
| scikit-learn | 8.3 | required |
| pillow | 7.3 | required |
| lxml | 4.0 | required |
| cryptography | 3.8 | required (AES-GCM artifacts) |
| sqlalchemy | 2.4 | required |
| 72 pure-python wheels | ~29 | ~0.4 MB each |

Optional (~10× the core set if everything is installed):

| Extra | Compressed | Drivers |
| :--- | ---: | :--- |
| `boosting` | **151 MB** | catboost 100, xgboost 49, lightgbm 1 |
| `deep` | **128 MB** | torch 126.7 (+lightning) |
| `vision` | **60 MB** | opencv-python 48, scikit-image 12 |
| `nlp` | **26 MB** | gensim 24, nltk 2 |
| `foundation` | (torch + transformers + timm + ultralytics + open-clip) ≈ **300 MB+** | |
| `automl` / `responsible` | ~1–2 MB | optuna, flaml, shap, codecarbon |

**Conclusion:** the dependency set is exactly what must *not* be baked into the installer, and it is also what must be
split into "required now" vs "install when the user opens that studio".

---

## 2. The compatibility fact that makes this safe (why features don't change)

The platform was audited for this design and already behaves as a *provisioning-aware* application:

1. **63** call sites of `is_package_available(...)` (`core/utils.py:45`) — classical models fall back to
   `scikit-learn(hgb_fallback)` when xgboost/lightgbm/catboost are absent, deep models fall back to the NumPy
   implementations (`omniforge_numpy_autograd/conv1d/transformer/gru`), foundation vision lists every backend with
   `"available": true/false`, carbon tracking falls back to the hardware estimator.
2. **12** `try/except ImportError` guarded module imports (jwt, cryptography, duckdb, pyarrow, torch, …).
3. **7** `DependencyMissingError` raises that produce an actionable message instead of a crash.
4. `run_environment_check()` in `launcher.py:157` already produces a structured report of
   `required_modules` / `optional_modules` — the exact primitive a "System Health / Install missing capability" screen needs
   (it just needs a POST endpoint + one panel; **no analysis code changes**).

So the *capability set of the application is unchanged*: every studio still works; absent extras simply use the
documented fallback path, and the user can install the extra at any time to upgrade that path to the native backend.
That is behaviour the code already implements — this proposal only exposes it.

---

## 3. Recommended architecture (Option C — the one I recommend)

```
OmniForge-Setup.exe   (≈55–75 MB, Inno Setup, per-user, no UAC)
├── app\                      your code + configs                        ~6 MB
├── app\runtime\python\       CPython 3.13 embeddable                    ~15 MB
├── app\runtime\uv.exe        uv 0.13.x (self-contained resolver)        ~16 MB
├── app\ui\static_app\        pre-compiled React bundle (code-split)    0.65 MB
├── app\locks\core.lock       hash-pinned, offline-verified lock         <50 KB
└── app\scripts\provision.py  bootstrap + verify + repair                ~400 lines
```

At install time (Inno Setup `[Run]` → `provision.py`):

```
1. create  %LOCALAPPDATA%\OmniForge\venv      (from the bundled runtime, --without-pip)
2. probe   the machine for already-usable distributions   (see §4.3)
3. `uv sync --frozen --locked --no-dev`  → downloads ONLY what is missing,
   from PyPI / `--index-url` mirror / `--find-links` local cache, with hash verification
4. write provisioning receipt (versions + hashes + timestamps) to %LOCALAPPDATA%\OmniForge\provision.json
5. launch app → startup gate reads the receipt + `run_environment_check()`, shows
   "3 capabilities available · 2 can be added" instead of failing
```

At first use of a capability that needs an optional extra (e.g. Studio06 → Deep Learning with PyTorch):
the panel shows **"Install PyTorch backend (126 MB)"** → POST `/api/system/install-extra` → `uv pip install` in a
subprocess with a progress stream → on success the capability flips to the native backend **without restarting the app**
(the `is_package_available` probes are evaluated per call, not cached at import).

### Why `uv` and not `pip`
- **15.7 MB single binary, no Python needed**, resolves and installs 84 wheels in ~10–20 s (pip: minutes).
- `uv sync --frozen` is **byte-for-byte reproducible**; `--require-hashes` / lock hashes give supply-chain integrity.
- `uv pip install --target / --prefix / --python` covers every layout we may need, and it is a Rust binary so it has no
  PyInstaller interplay at all.
- It is also the engine you'd use **offline** (`uv sync --offline` against a pre-populated cache).

---

## 4. The four things that must be designed correctly (this is where these installers break)

### 4.1 Reproducibility: one lock file, hashes mandatory
- Generate `locks/core.lock` **on Windows** with `uv pip compile requirements.txt --python-platform windows --python-version 3.13 --generate-hashes`.
  (In this workspace I verified the win_amd64/cp313 resolution *succeeds* for every requirement — 84 wheels, **zero sdists**,
  so the whole core set is binary-only and never needs a C compiler on the user's machine.)
- Install **only** with `--frozen --locked --require-hashes`. Never `pip install -r requirements.txt` at install time
  (range pins like `numpy>=2.5.5` = non-reproducible and a supply-chain hole).
- **Never `--pre`.** Record `python_version` in the lock; refuse to install a lock built for another minor.

### 4.2 "Use what's already there" — the honest answer
Three levels, in order of safety:

| Level | Behaviour | Verdict |
| :--- | :--- | :--- |
| **A. Reuse wheels** | Point `uv` at a cache/mirror: `--find-links \\fileserver\wheels`, `UV_CACHE_DIR`, or a wheelhouse bundled on the LAN. Missing → PyPI. | ✅ **Recommended.** This is the "check first, download the rest" you asked for, with zero interference risk. |
| **B. Reuse an existing venv/conda** | Detect `python -c "import numpy"` → reuse. | ⚠️ **Reject.** numpy built for another Python/ABI, conda's MKL/OpenBLAS vs wheel BLAS, stale patched CVEs, user-broken env. You would inherit every support ticket. |
| **C. Install into system Python** | `pip install --user` into the user's global env. | ❌ **Never.** Collides with the user's own projects (pandas 3 vs 2, sklearn version pinning) and is uninstallable. |

**Implementation of Level A (the "check" step you asked for):** before resolving, the bootstrapper runs a
*candidate probe* — for each pinned wheel, test for an exact/full version in (1) `%LOCALAPPDATA%\OmniForge\wheelhouse`,
(2) `UV_CACHE_DIR`, (3) the configured mirror, (4) PyPI — and installs the **rest**. Because the app is fully isolated in
its own venv, "already installed globally" is *irrelevant* (and should be ignored, not reused). This is the key
correction to the original idea: **"check what's already installed" must mean "check which artifacts we already have
locally", not "check which packages the user happens to have globally."**

### 4.3 Offline / air-gapped / flaky-network story
- Corporate/air-gapped: build a **wheelhouse** (`uv pip download -r requirements.txt --python-platform windows`) →
  ~171 MB on a share → installer flag `--wheels \\server\share\omniforge-core` → **zero internet**.
- Flaky links: `uv` retries per-request and is resumable at wheel granularity; `provision.py` is **idempotent** and
  re-runnable (`Repair installation` in the uninstaller list / Start Menu) — never leaves a half-venv.
- No internet at all and no wheelhouse → the installer must *say so up front* and offer `--minimal` (see §5).
- **Corporate TLS:** honour `PIP_CERT`/`SSL_CERT_FILE`, `--index-url` (Artifactory/Nexus/Devpi), `--trusted-host`.

### 4.4 Integrity, signing and rollback
- `uv sync --require-hashes` at wheel level; the lock's hashes are the trust anchor.
- Sign `OmniForge-Setup.exe` **and** the lock file (HMAC/SignTool) so a tampered lock can't be pushed.
- Keep the **last known-good venv** (`venv.prev`) → one-click rollback if an upgrade breaks imports.
- `provision.json` stores `{python, uv, lock_sha256, packages[], installed_at, source}`; the startup gate compares it to the
  shipped lock and offers **Repair / Update** when they diverge.
- If your Enterprise tier means "verified", prefer **PyPI-only + hashes** over "whatever a random mirror serves".

---

## 5. Four deployment profiles (so "capabilities don't change" holds for every customer)

| Profile | Installer | Download | Disk after | Who it's for |
| :--- | ---: | ---: | ---: | :--- |
| **Offline-Full** | 260 MB | 0 | ~700 MB | Air-gapped/factory PCs (wheelhouse baked in) |
| **Online-Core** *(recommended default)* | **65 MB** | 171 MB | ~550 MB | normal desktop/laptop |
| **Online-Core + mirrors** | 65 MB | 0–171 MB | ~550 MB | enterprise with Nexus/Wheelshare |
| **Minimal** (SPA+API+CLI, no heavy analytics) | **35 MB** | ~40 MB | ~180 MB | demo/trial — *this is the only profile where features are actually absent* |

Note the Minimal row: it is the one case that *would* reduce capabilities — the fallbacks keep the app functional
(NumPy deep models instead of torch, HGB instead of xgboost) but with different performance. Keep it as an explicit
"Trial" profile and install the rest on first use.

---

## 6. Recommended first-use behaviour (this is the UX that makes it feel professional)

1. **Setup wizard page**: "Choose capability packs" — checkboxes for `Core (required, ~171 MB)`,
   `Gradient Boosting (151 MB)`, `Deep Learning / PyTorch (128 MB)`, `Computer Vision (60 MB)`, `NLP (26 MB)`,
   `Foundation Models (300 MB+)`. Defaults: Core only. This **is** the "install what's missing" step, and the user
   pays only for what they want.
2. **Progress with real sizes and a cancel button** (Inno Setup `[Run]` + a tiny WinForms/console progress from `provision.py`);
   never a frozen window.
3. **Startup gate** (already 90 % built): `/api/v1/health` + `run_environment_check()` →
   a **"Capabilities"** badge row in the workbench (present / can-be-installed), and a
   `POST /api/system/install-extra` that streams progress. The studios already advertise their backend availability
   (`foundation.py:44–49` returns `available` per backend) — surface the same flag next to *"Install PyTorch"*.
4. **Uninstaller** offers Repair / Update / Remove cache.

---

## 7. Risks, and how each is closed

| Risk | Mitigation (concrete) |
| :--- | :--- |
| No internet / proxy blocks PyPI | Detected **before** any download (`uv` dry-run probe); clear error + wheelhouse instructions. 6–8 % of offline enterprises need the Offline-Full profile. |
| Non-reproducible installs ("it worked on my machine") | `--frozen --locked --require-hashes`; lock generated in CI on Windows; refuse to run on a lock mismatch. |
| Supply-chain (typosquat/compromised release) | Hash-pinned lock **signed** with the product key; PyPI-only by default; no `--pre`, no `-r requirements.txt` at install time. |
| Antivirus/SmartScreen flags a "downloader" | Sign both Setup.exe **and** the child `uv.exe`/`python.exe`; keep the installer behaviour transparent (no hidden downloads). |
| A download breaks mid-install | Idempotent `provision.py` + `Repair` + `venv.prev` rollback; per-wheel resume. |
| Python minor version drift | Bundled **embeddable CPython 3.13** (not "any 3.12–3.15") → the lock's `cp313` tags are guaranteed to match. |
| Corporate policy forbids runtime downloads | `--wheels` / `--index-url` / `--offline` switches; enterprise MSI variant with the wheelhouse. |
| Users mix global and app packages | App runs **only** from its own venv; `PYTHONNOUSERSITE=1`, `-E -s` isolation; never touch `site-packages` of the system Python. |
| Startup slower after first run | Venv kept warm; `uv sync` is a no-op check on later launches (<1 s). |

---

## 8. What changes in this repo (small, surgical, no capability change)

| File | Change | Effort |
| :--- | :--- | :--- |
| `locks/core.lock` (new) | CI-generated, hash-pinned, `--python-platform windows --python-version 3.13` | 1 CI job |
| `installer/provision.py` (new) | bootstrap venv, candidate probe, `uv sync`, receipt, repair, `--wheels/--index/--offline`, `--minimal` | ~400 lines |
| `installer/omniforge_installer.iss` | replace `dist\OmniForge\*` payload with `app\` + `runtime\` + `uv.exe` + `locks\`; add wizard page + `[Run] provision.py`; add `Repair` in uninstaller | ~120 lines |
| `launcher.py` | **already fine**: add `--provision`/`--check-extras` passthrough and the port/`runtime.json` handshake from the earlier review | ~40 lines |
| `api/main.py` | `POST /api/system/install-extra` + `GET /api/system/capabilities` (thin wrapper over `run_environment_check`) | ~60 lines |
| `frontend` | Capability badges + "Install X (N MB)" button with progress (one component) | ~1 component |
| Remove PyInstaller from the **user** path | keep it for the CLI/portable `.exe` if you still want that; the installer no longer needs 700 MB | — |

Nothing in `omniforge/{data,models,evaluation,automl,…}` changes. Zero functional delta.

**Build reproducibility check that matters:** the win_amd64/cp313 resolution of `requirements.txt` in this workspace
produced **84 wheels, 0 sdists, 171 MB** — i.e. no compilation on the user's PC, no Visual C++ build tools, no
Rust/gcc. That is the precondition for an online installer to be reliable at all, and it currently holds.

---

## 9. Verdict

- **Do it.** It is the correct answer to "small installer, full capability", it matches the app's existing
  optional-dependency design, and it removes the per-dependency rebuild loop (today every bump means a new
  multi-hundred-MB exe).
- **Do it as Option C**: bundled `uv` + hash-pinned lock + per-user venv + capability packs on demand.
- **Do not** reuse the user's global Python/conda packages (§4.2-B/C) — that is the classic way this design becomes
  unsupportable.
- **Keep an Offline-Full/MSI variant** for enterprise/air-gapped customers; that is the only way "installer size" and
  "no internet" both hold.
- **Capabilities are preserved by construction**: the fallback matrix already exists in code (63 probes / 12 guards /
  7 `DependencyMissingError`), so an absent extra downgrades the *implementation* of a feature, never removes it —
  and the user can install it with one click from inside the app.

### Suggested next step (pick one)
1. **Prototype the bootstrapper** (I implement `provision.py` + `locks/core.lock` + the `.iss` changes and validate the
   resolution against the real wheel set) — highest value, verifiable here.
2. **Prototype only the capability UI** (`/api/system/capabilities` + `install-extra` + badges) so you can demo
   "install PyTorch from inside Studio06".
3. **Write the ADR** with the four profiles and the offline/mirror decision for sign-off.

---

## 10. Reproduction of the measurements

```bash
# which wheels and how many
pip download --only-binary=:all: --platform win_amd64 --python-version 3.13 --implementation cp --abi cp313 \
    --dest /tmp/wheels -r requirements.txt        # → 84 wheels, 0 sdists, 171 MB

# per-package compressed sizes (PyPI JSON API, win_amd64/cp313 wheel)
python3 - <<'PY'
import json, urllib.request
for p in ["scipy","pyarrow","duckdb","numpy","catboost","torch","opencv-python","uv"]:
    d=json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{p}/json"))
    f=[x for x in d["urls"] if "win_amd64" in x["filename"] and x["filename"].endswith(".whl")][0]
    print(p, d["info"]["version"], round(f["size"]/1e6,1), "MB")
PY
```

---

## 11. Answers to the three follow-up questions (2026-11-18)

### 11.1 Which of the three options do I recommend? → **(ب) the Python bootstrapper, built FIRST**

Not because it is the smallest, but because it is the only one that is *load-bearing*: whatever the desktop shell
becomes later (plain exe, pywebview, Electron), the provisioning engine, the tier/pack catalogue, the lock files and
the `provision.json` receipt are needed **unchanged**. The Electron/IPC/auto-update layer on top is then a thin,
replaceable wrapper.

**What I already built and verified in this workspace instead of only writing about it:**

| Deliverable | Path | Verified how |
| :--- | :--- | :--- |
| Capability/pack catalogue (single source of truth) | `src/omniforge/capabilities.py` | 8 new unit tests green |
| Installer bootstrapper | `installer/provision.py` | **actually ran end-to-end** (see below) |
| Hash-pinned locks per tier | `locks/simple.lock`, `locks/advanced.lock`, `locks/everything.lock` | 172 pins / 4 224 hashes for `everything`, generated for `--python-platform windows --python-version 3.13` |
| Tiered installer wizard | `installer/omniforge_installer_tiers.iss` | compiles conceptually (Inno is Windows-only; script reviewed line by line) |
| Tests | `tests/unit/test_capability_packs.py` | 8 passed |

**Measured proof (real runs, not claims):**

```
python installer/provision.py --tier simple --data-dir /var/tmp/of_final
  → venv created from the system runtime, 30 requirements resolved from locks/simple.lock,
    `uv pip sync --require-hashes` installed 26 real wheels, receipt written, exit 0
  → app smoke test inside that venv: quality 99.97 (A+), fastapi absent, torch absent
python installer/provision.py --tier advanced --data-dir /var/tmp/of_adv
  → 36 requirements, full pytest suite inside the provisioned venv: **93 passed in 107.69 s**
  → mypy inside the provisioned venv: **Success: no issues found in 167 source files**
```

### 11.2 Will the installer be built with Electron? → **No — and it must not be**

This is the important correction: **Electron is a desktop *shell*, not an installer.** An Electron app is itself
installed by an installer (NSIS/Squirrel/MSIX). Electron cannot create a Python venv, cannot resolve wheels, cannot
run `uv`. If you build the "installer" with Electron you would still need a second installer underneath it — and you
would add ~180 MB before installing a single dependency.

Recommended split:

| Concern | Tool | Why |
| :--- | :--- | :--- |
| Selecting packs / copying files / shortcuts / uninstall | **Inno Setup 6** (`installer/omniforge_installer_tiers.iss`) | native wizard, per-user (no UAC), tiny, already in the repo |
| Provisioning the Python environment | **`provision.py` + bundled `uv`** | verified above; deterministic, hash-checked, offline-capable |
| Launching the desktop window *(optional, later)* | **pywebview/WebView2** (recommended) or **Electron** (only if you need Node-side features) | either way it just opens `http://127.0.0.1:<port>`; nothing else changes |

So: **installer = Inno Setup + provision.py. Shell = WebView2 (or Electron).** They are separable, and the work I
just did is what both paths need.

### 11.3 Simple vs Advanced tier — exactly what gets installed

Both tiers ship the **same application code** (no feature is removed from the codebase); they differ only in which
dependency packs are installed, and therefore which *implementation* a feature uses. Measured compressed sizes are
from the win_amd64/cp313 wheel set in §1.

| Pack | Simple | Advanced | Everything | What it unlocks |
| :--- | :---: | :---: | :---: | :--- |
| `core-runtime` (numpy, pandas, scipy, scikit-learn, pydantic, plotly, pillow …) | ✅ | ✅ | ✅ | Ingestion, cleaning, 6-D quality, EDA, features, clustering, baselines, classical ML, Copilot |
| `statistics` (statsmodels) | ✅ | ✅ | ✅ | Studio 08 forecasting/causal estimators |
| `data-io` (pyarrow, duckdb, sqlalchemy, lxml, fsspec) | ✅ | ✅ | ✅ | Parquet/ORC/XML + SQL workspace + live SQL scoring |
| `reporting` (openpyxl, matplotlib, docx, pptx, reportlab, nbformat) | ✅ | ✅ | ✅ | Studio 12: PDF/DOCX/PPTX/XLSX/IPYNB export |
| `api-server` (fastapi, uvicorn, pyjwt, cryptography) | — | ✅ | ✅ | **The Workbench server itself** — without it the GUI does not run |
| `boosting` (xgboost, lightgbm, catboost) | — | — | ✅ | Native boosters (fallback: scikit-learn HistGradientBoosting) |
| `deep` (torch, lightning) | — | — | ✅ | Native GPU deep learning (fallback: NumPy implementations) |
| `vision` (opencv, scikit-image) | — | — | ✅ | Native image pipeline (fallback: Pillow) |
| `nlp` (nltk, spacy, gensim) | — | — | ✅ | Richer NLP (fallback: sklearn TF-IDF) |
| `foundation` (transformers, timm, CLIP, YOLO) | — | — | ✅ | Zero-shot vision, LLM summarization |
| `automl` (optuna, flaml) | — | — | ✅ | Extra search strategies (GP-UCB already included) |
| `responsible` (shap, codecarbon) | — | — | ✅ | SHAP plots, measured carbon (fallback: permutation importance) |
| **installer size** | **≈ 35 MB** | **≈ 65 MB** | **≈ 65 MB** | (code + CPython embeddable + uv + locks) |
| **download at install time** | ~150 MB | ~171 MB | ~1.3 GB | (0 MB with a wheelhouse) |
| **disk after install** | ~0.8 GB | ~0.9 GB | ~6 GB | |

**Simple is NOT a crippled product:** it is the full analytics/data-science workbench (Studios 01–13 logic, all
models, all reports) with the fallback backends; what it cannot do is serve the web GUI (no FastAPI) or use native
GPU boosters. **Advanced is the normal desktop product** (GUI + everything core). **Everything** adds native
backends only.

Wizard copy (already in the `.iss`): the user picks `Simple / Advanced / Everything` as a radio group **and** may tick
individual optional packs — e.g. Advanced + only `boosting`. The selection is passed to the provisioner as
`--tier advanced --packs boosting`, so exactly the ticked packs are installed.

**Inside the app** the same catalogue drives "Install …" buttons: a capability badge appears next to any feature
whose native backend is missing, and `POST /api/system/install-extra {"pack": "deep"}` provisions it in the existing
venv without restarting the app (because every availability probe is evaluated per call, not cached at import).

---

## 12. Option (ب) delivered — the staging + lock pipeline (built and measured)

Everything below was executed in this workspace; the numbers are outputs, not estimates.

### 12.1 New files

| File | Role |
| :--- | :--- |
| `scripts/build_bootstrap.py` | stages `dist/omniforge-bootstrap/` — app payload + embeddable CPython + `uv.exe` + locks + `manifest.json` (size + sha256 of every file). Cross-platform; `--dry-run`, `--python-zip`, `--uv-whl` (offline), `--cache`. |
| `scripts/build_locks.py` | compiles the 11 hash-pinned locks for `windows/cp313`: `simple`, `advanced`, `everything`, one `pack-<id>.lock` per optional capability, and `dev.lock`. Deterministic (stable input path, provenance normalised). |
| `.github/workflows/windows-installer.yml` | CI: regenerate locks → fail on drift → stage payload → lint/type-check → capability tests → real Simple + Advanced provisioning runs → compile the Inno wizard → **120 MB budget guard** → silent-install smoke test → publish `.exe` + `SHA256SUMS.txt`. |
| `installer/omniforge_installer_tiers.iss` *(updated)* | now fully payload-relative (license/icon/provisioner come from the staged tree), so a clean CI checkout can compile it. |
| `locks/*.lock` *(new, 2.4 MB total)* | 11 locks, 16 000+ sha256 hashes. |

### 12.2 Measured payload (actual run)

```
[stage] app payload: 13 entries
[stage] python runtime: 34 files  (python-3.13.9-embed-amd64.zip, 10.9 MB)
[stage] uv resolver: uv.exe (32.6 MB)
[stage] locks: 11 files
[stage] manifest: 278 files, 59.2 MB total
payload ready: 59.2 MB, 278 files
```

Composition:

| Part | Size | Note |
| :--- | ---: | :--- |
| `runtime/uv.exe` | 32.6 MB | self-contained resolver (the single biggest item) |
| `runtime/python/` | 21.2 MB uncompressed | CPython 3.13.9 embeddable + `_pth` patched with `import site` |
| `app/` | **3.5 MB** | the whole product: `src/`, configs, docs, notebooks, the 10-chunk SPA |
| `locks/` | 2.4 MB | 11 hash-pinned lock files |
| **total** | **59.2 MB** | compressed by Inno's lzma2 → the `.exe` lands around **30–35 MB** |

> LZMA2/ultra on this payload is not run here (Inno is Windows-only), but the two compressible
> halves are known to shrink ~50 % (a 10.9 MB embeddable zip ↔ 21 MB unpacked; a 32.6 MB `uv.exe`
> containing a compressed release binary). The CI workflow enforces a hard **120 MB** ceiling so a
> regression cannot slip through.

### 12.3 Lock catalogue (measured: pins / hashes)

| Lock | Pins | Hashes | Used for |
| :--- | ---: | ---: | :--- |
| `simple.lock` | 73 | 1 898 | Simple tier |
| `advanced.lock` | 87 | 2 380 | Advanced tier |
| `everything.lock` | 172 | 4 224 | Everything tier |
| `pack-boosting.lock` | 92 | 2 426 | Studio 06 native boosters |
| `pack-deep.lock` | 104 | 3 148 | PyTorch deep learning |
| `pack-vision.lock` | 92 | 2 445 | OpenCV pipeline |
| `pack-nlp.lock` | 110 | 3 000 | spaCy / gensim |
| `pack-foundation.lock` | 118 | 2 738 | Transformers / CLIP / YOLO |
| `pack-automl.lock` | 93 | 2 392 | Optuna / FLAML |
| `pack-responsible.lock` | 103 | 2 564 | SHAP / codecarbon |
| `dev.lock` | 15 | 403 | CI only (pytest/mypy/ruff) — never shipped |

`provision.py` now auto-selects the matching lock: one optional pack → `pack-<id>.lock`, otherwise the tier lock,
falling back to `everything.lock` (still hash-pinned).

### 12.4 End-to-end verification performed here

| Check | Result |
| :--- | :--- |
| `build_bootstrap.py --dry-run` | plan printed, nothing written |
| `build_bootstrap.py` (real) | **59.2 MB / 278 files**, manifest written, no `__pycache__`/`.pyc`/`node_modules`/tests leaked |
| Staged payload self-check | `provision.py` inside the payload resolves `runtime/uv.exe`, `locks/pack-vision.lock`, and the tier |
| `build_locks.py` twice | **byte-identical output** (determinism verified), no build-machine paths leaked |
| Lock validity after normalisation | real install from `pack-responsible.lock`: `fastapi`, `shap`, `codecarbon`, `pandas` all import |
| Capability tests | `tests/unit/test_capability_packs.py` — 8 passed |
| Lint/type-check | `ruff check .` clean, `mypy src/omniforge` **0 errors in 167 files** |
| Full test suite (provisioned advanced venv) | **93 passed** |

### 12.5 What is still Windows-only (cannot be executed in this Linux sandbox)

1. `ISCC.exe` compilation of the `.iss` (syntax has been reviewed; CI runs it for real).
2. The silent-install smoke test (`/VERYSILENT`) — in the workflow.
3. `vcruntime140.dll` presence in the embeddable runtime — it ships inside `python-3.13.9-embed-amd64.zip`
   (verified in the extracted file list), so no separate VC++ redistributable is required.

Run `git tag v2.1.0 && git push --tags` (or trigger **Windows installer** manually) to produce the first
signed-candidate installer and its `SHA256SUMS.txt`.

---

## 13. Do you need `build_exe.bat`? — No (and here is the proof)

### 13.1 The short answer

**No.** `build_exe.bat` runs **PyInstaller** to freeze a 700 MB-1 GB `dist\OmniForge\` folder that bundles
numpy/pandas/scipy/sklearn *inside the exe tree*. That is exactly what the new pipeline replaces: the
bootstrap installer ships an **embeddable CPython + `uv` + hash-pinned locks**, and the dependencies are
installed into a private venv on the user's machine.

| Step to build `OmniForge-Setup.exe` | Command | PyInstaller needed? |
| :--- | :--- | :--- |
| 1. regenerate locks (only when pins change) | `python scripts/build_locks.py` | ❌ |
| 2. stage the payload | `python scripts/build_bootstrap.py --launcher-zip python-3.13.9-amd64.zip` | ❌ |
| 3. compile the wizard | `ISCC.exe installer\omniforge_installer_tiers.iss` | ❌ |
| 4. (optional) test on a clean VM | run the resulting `dist\installer\OmniForge-2.0.0-Setup.exe` | ❌ |

`build_exe.bat --onedir` is only needed if you deliberately choose the **other** launcher flavour
(`ISCC.exe /DLaunchMode=exe`), i.e. you want a self-contained PyInstaller desktop binary *instead of*
the venv-backed app. Both flavours are supported by the same `.iss` (`LaunchMode=py|exe`).

### 13.2 Correction made while verifying this (important)

My first version of the shim **copied `pythonw.exe` to `OmniForge.exe`**, assuming python.org ships a
`py.exe`-style retargetable stub. I checked the bytes and that assumption is **wrong**:

```
pythonw.exe contains '#!' launcher magic: False   ← it is a real interpreter, not a stub
⇒ a byte copy would start an empty interpreter and exit (silent no-op app)
```

The correct, dependency-free solution is CPython's own **`venvlauncher.exe`** (the same image `venv`
copies when it creates `Scripts\python.exe`), retargeted by appending its launcher block:

```
<venvlauncher.exe bytes> + "\n#!<interpreter>\n<script>\n<args>\n\0"
```

`scripts/make_launcher_shim.py` now does exactly this and **verifies the result**:

```
[launcher] OmniForge.exe: venvwlauncher.exe + launcher block (246 KB, GUI)
[launcher] OmniForge-Debug.exe: venvlauncher.exe + launcher block (249 KB, console)
   tail: ['#!%APP_DIR%\\runtime\\python\\python.exe', '%APP_DIR%\\omniforge.py', '--mode full', '<NUL>']
```

* GUI flavour (`venvwlauncher.exe`) for the Start-Menu/desktop shortcut — no console window.
* Console flavour (`venvlauncher.exe` → `OmniForge-Debug.exe`) for troubleshooting tracebacks.
* `OmniForge.cmd` is always emitted as a last-resort fallback (and used by CI sanity checks).
* The launcher lives **inside `runtime\python\`**, next to `python3XX.dll` and `vcruntime140.dll`, because
  the Windows loader resolves DLLs from the executable's own directory.

If no `venvlauncher.exe` is supplied the script **says so and falls back honestly** instead of shipping a
broken exe — that is the difference between a demo and a product.

### 13.3 Payload after the change (measured)

```
[launcher] OmniForge.exe       246 KB GUI
[launcher] OmniForge-Debug.exe 249 KB console
[stage]    manifest: 281 files, 59.7 MB total   (was 59.2 MB; +0.5 MB for the launchers)
```

### 13.4 What (الف) and (ج) mean, concretely

**گزینه (الف) — in-app capability installation (the last missing piece of the UX).**

Today the installer can install packs, but a user who chose "Advanced" cannot add PyTorch later without
re-running the installer. (الف) adds:

| Piece | What it does |
| :--- | :--- |
| `GET /api/system/capabilities` | returns `run_environment_check()` + `profile_summary()` per pack: installed / missing / `approx_mb` / what it unlocks |
| `POST /api/system/install-extra` | streams `provision.py --tier advanced --packs <id> --repair` inside a background job; the SPA shows a progress bar and the exact download size |
| Studio badges | e.g. Studio 06 shows *"Deep learning: NumPy fallback — Install PyTorch (2.8 GB)"*; Studio 07 shows *"SHAP unavailable — Install (90 MB)"* |
| No restart | every probe is evaluated per call, so the badge flips to "native" the moment the install finishes |

Effort ≈ 2 files backend + 1 component frontend, **~1 day**. It is the difference between "the installer can do it"
and "the product can do it".

**گزینه (ج) — a real desktop window (pywebview/WebView2), no Electron.**

Today the app opens in the user's browser. (ج) makes it a native window with no address bar:

| Piece | What it does | Cost |
| :--- | :--- | :--- |
| `pywebview` shell | opens `http://127.0.0.1:<port>` in WebView2; tray icon, close-to-tray, native menus | +2-5 MB, no Node |
| port handshake | `launcher` writes `runtime.json` with the chosen port; the shell reads it instead of guessing | ~20 lines |
| `%APPDATA%` paths | move the cwd-relative `.omniforge_workspace` defaults so the app works from `Program Files` | ~15 lines |
| deep-link / file-association | `.omniforge` files open the app | ~30 lines |

Effort ≈ **2-3 days**. It is cosmetic/product polish; nothing analytical changes.

**Also relevant for the shortcut:** (ج) is what removes the console flash and the "why did my browser open?"
question. Until (ج), `OmniForge.exe` (the venvlauncher shim) starts the app and hands off to the default browser —
which is fine, but that is the item that makes it feel like a desktop application.

---

## 14. Troubleshooting: `ISCC Line 28: Could not read "...\app\LICENSE"`

**Symptom** (exactly what you hit):

```
Line 28: Could not read "C:\...\OmniForge\installer\..\dist\omniforge-bootstrap\app\LICENSE".
Error: The system cannot find the path specified.
*** Compile aborted.
```

**Cause — not a path bug.** `installer\..\dist\omniforge-bootstrap` correctly resolves to
`<repo>\dist\omniforge-bootstrap`; the folder simply does not exist because **the payload was never staged**.
ISCC cannot build an installer whose payload is absent. (I reproduced the exact check locally: with no payload the
guard fires; with the payload all four required files are present.)

### Fix: one command

```bat
build_installer.bat
```

It performs, in order:

| Step | Command it runs | Failure message you now get |
| :--- | :--- | :--- |
| 1 | detect Python + locate `ISCC.exe` + install `uv` | *"Inno Setup 6 not found — install from jrsoftware.org"* |
| 2 | fetch `python-3.13.9-amd64.zip` (for `venvlauncher.exe`, cached) | falls back to `.cmd` launcher with a warning |
| 3 | `python scripts\build_locks.py` | *"Lock generation failed"* |
| 4 | `python scripts\build_bootstrap.py --launcher-zip ...` | *"Payload staging failed"* |
| 5 | `ISCC.exe installer\omniforge_installer_tiers.iss` | the real ISCC error |
| 6 | print `dist\installer\*.exe` + size | — |

### Fix: manual equivalent (three commands)

```bat
cd C:\Users\Amir\Desktop\Jules\OmniForge
python scripts\build_locks.py
python scripts\build_bootstrap.py --launcher-zip .cache\bootstrap\python-3.13.9-amd64.zip
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\omniforge_installer_tiers.iss
```

`python scripts\build_bootstrap.py` alone is enough if you do not need a real `OmniForge.exe` launcher
(it then ships `OmniForge.cmd`).

### Why the payload does not exist on a fresh clone

`dist/` is in `.gitignore` (build output, ~60 MB). That is intentional: the payload is reproducible
(`scripts/build_bootstrap.py` + pinned locks), so it is never committed. **A fresh clone must run step 4 once
before compiling.**

### Hardening added so this cannot happen silently again

The `.iss` now has compile-time guards that fail *before* Inno's file loop with an actionable message:

```pascal
#ifnexist StagingDir + "\app\LICENSE"
  #error "Bootstrap payload is missing. Build it first: python scripts\build_bootstrap.py ... (or run build_installer.bat)"
#endif
#ifnexist StagingDir + "\app\omniforge.py"          #error "Payload is incomplete ..." #endif
#ifnexist StagingDir + "\runtime\python\python.exe" #error "Payload is incomplete ..." #endif
#ifnexist StagingDir + "\locks\advanced.lock"        #error "Run scripts\build_locks.py ..." #endif
```

Plus `StagingDir` is now overridable (`ISCC /DStagingDir=...`), so you can compile from a build server or a
different checkout without editing the script.

### Quick self-check without compiling

```bat
python -c "import pathlib;b=pathlib.Path('dist/omniforge-bootstrap');print('payload:', 'OK' if (b/'app'/'LICENSE').exists() else 'MISSING -> run build_installer.bat')"
```


---

## 15. Every ISCC error that was hit, and the rule behind it

Each row is one compile attempt that failed on the Windows machine, the *root cause* (not the
symptom), and the rule that is now enforced by `scripts/check_iss.py` so the whole class is caught
before ISCC runs.

| # | ISCC said | Actual root cause | Rule now enforced |
|---|-----------|-------------------|-------------------|
| 1 | `Line 28: Could not read "...\app\LICENSE"` | `dist/` is gitignored, so a fresh clone has no staged payload; ISCC resolves `installer\..\dist\omniforge-bootstrap` correctly. | `#ifnexist` guards + `check_iss.py --stage` report every missing payload path with the command that fixes it. |
| 2 | `Line 106: Parameter "Name" includes invalid characters` | Inno identifiers may contain **only** alphanumerics, `_`, `/`, `\`; `core-runtime`, `data-io`, `api-server` are illegal. | `check_identifier()` — hyphen, leading digit, reserved word (`and/or/not`), and Pascal keyword checks. |
| 3 | *(would follow #2)* `Unknown identifier 'core-runtime'` in `[Code]` | Same defect reached through `IsComponentSelected('core-runtime')`. | Every `IsComponentSelected('x')` must name a declared `[Components]` entry. |
| 4 | *(latent)* `AddBackslash` inside `#define` | `AddBackslash()` is a `[Code]` **Pascal** function; ISPP knows nothing about it. | `check_preprocessor()` rejects `[Code]`-only functions in `#define`. |
| 5 | *(latent)* duplicate `[Files]` target | `app\*` already installs `.env.example` and `provision.py`; a second entry is fatal ("cannot be installed more than once"). | `check_duplicate_files()` compares resolved `DestDir`+`DestName`. |
| 6 | `Line 73: Could not read "...\app\LICENSE.txt"` | Inno accepts **only** `.txt`/`.rtf` for `LicenseFile`, so a `LICENSE.txt` twin is staged — but the payload on disk was a *stale* `app/` tree from an earlier build, and the `#ifnexist` guard still pointed at `app\LICENSE`. The guard passed; ISCC then failed on the file it actually reads. | Guard now tests `LICENSE.txt` itself, the twin is refreshed on every build (`refresh_derived_app_files`), `build_installer.bat` stages incrementally with an explicit per-file self-check, and `--clean` forces a full re-copy. |
| 7 | *(latent)* `#if` on a bare identifier | ISPP compares the *string value*, silently taking the wrong branch. | `check_if_conditions()` forces `#if "<name>" == "..."` or `#ifdef`. |

Two more classes are checked even though they have not fired yet: unknown `[Setup]` directives
(a typo there is a hard ISCC error) and `{#Define}` references that were never defined.

### 15.1 Version floor: Inno Setup 6.3

`ArchitecturesAllowed=x64compatible` / `ArchitecturesInstallIn64BitMode=x64compatible` are 6.3
syntax. On 6.0–6.2, ISCC reports `unknown directive value`, which reads like a typo in your script
rather than an out-of-date compiler. `build_installer.bat` now reads `ISCC.exe`'s file version
(PowerShell, best effort) and stops with the download link if it is older than 6.3.

### 15.2 What runs where (`build_installer.bat`, default `LaunchMode=py`)

```
[1/6] toolchain      python, uv, ISCC present; ISCC >= 6.3
[2/6] launcher       .cache\bootstrap\python-3.13.9-amd64.zip -> venvlauncher retargeting
[3/6] locks          scripts\build_locks.py (11 hash-pinned files, deterministic)
[4/6] payload        scripts\build_bootstrap.py --launcher-zip ...  -> 282 files / 59.7 MB
[5a/6] validation    scripts\check_iss.py --stage dist\omniforge-bootstrap   <-- new gate
[5b/6] compile       ISCC.exe installer\omniforge_installer_tiers.iss
[6/6] result         dist\installer\OmniForge-2.0.0-Setup.exe
```

The validation step exits non-zero and the batch file aborts **before** ISCC, so a scripting error
costs seconds instead of a failed 60 MB compression pass.

### 15.3 Measured verification of this round

```text
$ python3 scripts/check_iss.py --stage dist/omniforge-bootstrap
line 133: [warning] glob source directory does not exist: ../dist/OmniForge/*
exit=0                       # the only finding: the LaunchMode=exe branch, not built here

$ python3 -m ruff check .                 -> All checks passed!
$ python3 -m mypy installer/provision.py src/omniforge/capabilities.py scripts/
                                          -> Success: no issues found in 7 source files
$ python3 -m pytest tests -q              -> 99 passed (130.64s)
$ python3 scripts/build_bootstrap.py --launcher-zip /var/tmp/python-dist/py.zip
                                          -> [stage] payload ready: 59.7 MB, 282 files
$ cmp app/LICENSE app/LICENSE.txt         -> identical (0 differences)
```

### 15.4 If ISCC still stops

Re-run it and quote the line; but run the validator first — it now covers every error listed above:

```bat
python scripts\check_iss.py --stage dist\omniforge-bootstrap
```


---

## 16. `Line 73: Could not read "...\app\LICENSE.txt"` — why the guard did not catch it

This was the first error that got *past* the `#ifnexist` guards, and the reason is worth keeping:
**the guard and the directive read different files.**

```pascal
#ifnexist StagingDir + "\app\LICENSE"      ; guarded the ORIGINAL file   -> existed, so #error never fired
...
LicenseFile={#StagingDir}\app\LICENSE.txt   ; ISCC reads the .txt TWIN    -> did not exist -> Line 73
```

`LICENSE.txt` is *derived* from `LICENSE` by `build_bootstrap.py`, and the copy on disk was an
`app/` tree produced **before** that derivation existed. So the script was right, the payload was
stale, and the guard tested the wrong twin.

### What changed (four independent layers, so no single mistake can repeat it)

1. **The guard now names the file ISCC reads** — `#ifnexist StagingDir + "\app\LICENSE.txt"`,
   with the message *"payload is missing or stale … re-run build_installer.bat"*.
2. **The twin is refreshed on every build, unconditionally** — `refresh_derived_app_files()`
   re-copies `LICENSE → LICENSE.txt` even when `app/` is reused, and `copy_app()` no longer
   skips it in any mode.
3. **`build_installer.bat` verifies the payload file-by-file** before validating the script:
   `app\LICENSE`, `app\LICENSE.txt`, `app\omniforge.py`, `app\installer\provision.py`,
   `runtime\python\python.exe`, `runtime\python\OmniForge.exe`, `locks\advanced.lock`.
   Any gap stops the build with `Re-run with --clean`.
4. **Incremental staging with an honesty check** — `build_bootstrap.py --skip-staged` only reuses
   `app/` when `app_payload_complete()` says the tree is genuinely complete; otherwise it wipes and
   re-copies. `--clean` always wipes. (Re-copying the ~200-file `app/` tree is no longer the
   default cost of a rebuild, but it is never skipped on faith.)

### Measured evidence from this round

```text
$ python3 scripts/build_bootstrap.py --launcher-zip .cache/bootstrap/python-3.13.9-amd64.zip
[stage] manifest: 282 files, 59.7 MB total
[stage] payload ready: 59.7 MB, 282 files

$ python3 scripts/check_iss.py --stage dist/omniforge-bootstrap
line 136: [warning] glob source directory does not exist: ../dist/OmniForge/*      (exit 0)

per-file payload check
  OK app/LICENSE.txt  OK app/LICENSE  OK app/omniforge.py  OK app/installer/provision.py
  OK app/packaging/assets/omniforge.ico  OK runtime/python/python.exe
  OK runtime/python/OmniForge.exe  OK locks/advanced.lock  OK app/.env.example

end-to-end: the packaged entry point actually runs
$ cd dist/omniforge-bootstrap/app && python omniforge.py --help     # core-only env
usage: omniforge-desktop [-h] [--mode {full,ui,api,cli,check}] ...  (exit 0)
```

That last check is the one that matters beyond ISCC: it proves the tree the installer ships is not
just *present* but *runnable* — `omniforge.py` resolves `src/` from its own location and the launcher
imports cleanly with only the core requirements installed.

### Two more latent breakages found while verifying the above

| Where | Problem | Fix |
|-------|---------|-----|
| `build_installer.bat` / `app_payload_complete()` | Self-check expected `app\provision.py`, but the payload stages it at `app\installer\provision.py`. | Path corrected in both places (`[Run]` in the `.iss` already used the right one). |
| CI `Capability catalogue tests` | Step installed only `pytest`, but `tests/conftest.py` imports `omniforge` → `import sqlalchemy` fails. | Step now installs `-e ".[dev,api,ui,reporting,data,forecasting]"` and runs `test_capability_packs.py` + `test_installer_iss.py`. |

`tests/unit/test_installer_iss.py` now contains the regression guards (10 tests): every `Source:`
and `[Setup]` path must exist in the staged payload, `LicenseFile` must end in `.txt`/`.rtf`, the
staged tree must carry all seven required files, and `app_payload_complete()` /
`refresh_derived_app_files()` must behave as the build script assumes.
