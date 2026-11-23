# گزارش ممیزی و ارزیابی فنی OmniForge v2.0.0

**نوع ممیزی:** بررسی کامل کد (همهٔ ماژول‌ها)، راستی‌آزمایی آنلاین در برابر استانداردهای جهانی، و تست همهٔ یافته‌ها — در **سه دور** متوالی
**تاریخ:** ۲۱ نوامبر ۲۰۲۶
**پروژه/مسیر:** `hajianamir/OmniForge` در `/home/user/OmniForge` — شاخهٔ `arena/4e477e2f-omniforge` (منشعب از `4f35f2c` روی `main`)
**نسخه:** 2.0.0 (مطابق `pyproject.toml` و ایمیج compose `omniforge-enterprise:2.0.0`)
**مجوز:** Apache License 2.0 ✅
**وضعیت اصلاحات:** پس از ممیزی، یافته‌های P0/P1 روی همین شاخه (`arena/4e477e2f-omniforge`) اعمال شد — فهرست دقیق در **پیوست الف (Remediation Log)** آمده است. وضعیت نهایی: suite کامل ‎`190 passed / 1 skipped`‎ (py3.11/py3.12 محلی)، ruff پاک، mypy پاک (۱۸۵ فایل)، پوشش ‎`78%`‎، **CI سبز روی هر دو leg (py3.12/3.13)** — run `44946890094`. اسکرچ‌تست‌های py3.11 در `.audit/src` و `.venv-audit/` نگهداری می‌شوند و با `.gitignore` از تحویل جدا شده‌اند.

---

## ۱. خلاصهٔ مدیریتی (Executive Summary)

OmniForge مجموعهٔ بسیار گسترده و از نظر معماری «خوب‌لایه‌بندی‌شده» است: ۱۸۵ فایل منبع (حدود ۴۸٬۷۵۷ خط)، ۲۳ فایل تست (۶٬۰۹۶ خط)، ~۱۴۶ تست جمع‌آوری‌شده، ۱۴ استودیؤ CRISP-DM، ۱۸ پلاگین دامنهٔ صنعتی (همگی تست شدند و کار می‌کنند)، و زیرسامانه‌های تحلیلی که بخش بزرگی از آن‌ها **درست** پیاده‌سازی شده‌اند (سازوکارهای آماری در برابر ارجاع‌های معتبر سنجیده و تأیید شد).

با این حال، در ممیزی سه‌دوره‌ای موارد زیر **به‌طور قطعی و با تست زنده اثبات** شد:

1. **یک باگ سیستماتیک serialization که همهٔ مصنوعات مدل (artifacts) را درگیر می‌کند** (SEC-001): مدل ذخیره‌شده توسط خود فریم‌ورک، در بارگذاری مجدد توسط unpickler محدود خودش رد می‌شود. ریشهٔ دقیق آن شناسایی شد (نه حدس).
2. **استودیو ۱۴ (Supervisory Prediction Monitor) عملاً غیرقابل استفاده است**: ۵ اندپوینت اصلی آن به‌دلیل import تحت `TYPE_CHECKING` ذرات‌شان به‌عنوان query parameter تفسیر می‌شوند → همهٔ فراخوانی‌ها **422** می‌گیرند و `/openapi.json` با **500** کرش می‌کند. فرانت‌اند مستقیماً همین ۵ اندپوینت را صدا می‌زند.
3. **استودیو ۹ (Specialized AI) کاملاً خالی است**: روتر `s09_opt_sim.py` فقط یک اسکلت بی‌مسیر است؛ ۶ اندپوینتی که فرانت‌اند صدا می‌زند هیچ‌کدام وجود ندارند → POST 405، GET 404.
4. **استودیو ۴**: `HTTPException` ایمپورت نشده (۸ خطای F821 ایستا) و در اجرا هم endpoint فرضیه با **500 NameError** فرومی‌ریزد.
5. **زنجیرهٔ هش audit log فقط بخشی از فیلدها را پوشش می‌دهد**؛ با دستکاری `details`/`role`/`trace_id` در فایل ذخیره‌شده، `verify_chain()` همچنان `True` برمی‌گرداند. همچنین زنجیره امضا ندارد (unsigned) و خارج از مرز نوشتنی لنگریخته نمی‌شود.
6. **نشست Workbench فقط بر اساس هدر خام `X-Session-ID` است** — بدون اتصال به principal تأییدشده؛ کلاس BOLA/OWASP API1:2023.
7. **`GLOBAL_DF_CACHE` (fingerprint زیرمجموعه‌ای ۶۴ردیفی) برخورد (collision) قطعی دارد** — ویرایش ردیف ۵۰۰ام در دیتافریم ۱۰۰۰‌ردیفی انگشت‌نگار یکسانی تولید کرد → خطر خواندن نتایج کش‌شدهٔ کهنه/مسموم.
8. **تست‌های ریپو نسبت به یکدیگر خودمتناقض‌اند** (تست کیفیت داده: `proxy_accuracy` را می‌طلبد؛ تست لایهٔ داده: دقیقاً همان کلید را رد می‌کند) و **نام `proxy_accuracy` یک alias تکراری و بی‌اثر** برای `accuracy` است.
9. **`timeliness=None`** بدون ستون زمان‌رخداد تولید می‌شود ولی مسیر گزارش (`reporting/generator.py:260`) `float(None)` کرده و کرش می‌کند (یکی از ۱۱ شکست تست).
10. **CI فقط ۲ فایل تست را اجرا می‌کند**؛ گزارش P2 داخل repo ادعای «صفر خطا» می‌کند که با وضعیت فعلی (ruff: 13 خطا) سازگار نیست.

در مجموع **۱۱ شکست واحد/e2e + ۱ شکست integration** در تست‌ها وجود دارد که همگی ریشه‌یابی شدند (جدول بخش ۳). **هیچ شکست عددی/آماری در فرمول‌های اصلی پیدا نشد** — مدل‌های بقا، conformal، PSI، causal ATE، Markowitz و C-index همگی با ارجاع‌های معتبر مطابقت دارند (بخش ۶).

> **English executive summary.** Three audit iterations (review → web verification → test) were completed. The numerical/statistical core is sound (lifelines-verified). Definite defects were proven by live tests: a framework-wide artifact round-trip blocker (SEC-001 root-caused to CPython ≤3.11 enum `__reduce_ex__` via `builtins.getattr`), a completely non-functional Studio 14 (TYPE_CHECKING-imported request models → 422 + OpenAPI 500), an empty Studio 09 router, an 8-site `F821` in Studio 04 that also 500s at runtime, a tamper-evident audit chain that covers only a subset of fields, client-header-bound workbench sessions (BOLA class), a colliding fast-DataFrame fingerprint, mutually contradictory data-quality tests, and a CI that runs only 2 of 146 tests. Apache-2.0 licensed; original source untouched by this audit.

---

## ۲. محدوده، محیط و روش (سه دور ممیزی)

### ۲.۱ محدودهٔ بررسی
- کل کد `src/` (ماژول‌های `core`, `data`, `models`, `automl`, `evaluation`, `mlops`, `api`, `ui`, `cli`, `reporting`, `domains`, `copilot`, `pipelines`, ...)، فرانت‌اند (`frontend/` و باندل چکیدهٔ `ui/static_app/`)، تست‌ها، CI، Docker/Compose، README/docs، `requirements.txt`، `configs/`.
- مقایسهٔ عددی با کتابخانه‌های مرجع (lifelines, scikit-learn/SciPy) و دست‌محاسبه.

### ۲.۲ محیط اجرای تست (محدودیت مهم)
- `pyproject.toml`: `requires-python = ">=3.12"`؛ CI روی ۳٫۱۲ و ۳٫۱۳ اجرا می‌شود.
- `.venv-audit/` با venv محلی **py3.11.2** و آخرین نسخه‌های 2026 قابل‌دسترس روی PyPI برای 3.11 ساخته شد (numpy 2.4.6، pandas 3.1.1، scipy 1.17.1، scikit-learn 1.9.1، statsmodels 0.15.0، fastapi 0.165.0، pydantic 2.14.4، pytest 9.2.0، lifelines 0.30.3). پین‌های `requirements.txt` روی 3.11 قابل نصب نیستند؛ آن‌ها فقط روی CI (3.12/3.13) راستی‌آزمایی می‌شوند و **روی 3.12+ در این sandbox اجرا نشدند**.
- دو فایل با syntax مخصوص ۳٫۱۲ (`core/performance.py`, `evaluation/equations.py`) با نسخهٔ scratch زیر `.audit/src` برای اجرا under 3.11 تطبیق داده شدند؛ **کد اصلی دست‌نخورده ماند**. بندیت این دو فایل را به‌خاطر syntax نادیده گرفت (قابل‌انتظار).
- الگوی اجرای تأییدشده:
  `PYTHONPATH=/home/user/OmniForge/.audit/src .venv-audit/bin/python -m pytest tests -q`

### ۲.۳ مرور سه دور ممیزی (هر دور = بررسی + راستی‌آزمایی آنلاین + تست)
| دور | بررسی | راستی‌آزمایی آنلاین (استانداردها/منابع) | تست |
|---|---|---|---|
| **۱** | اسکن کامل کد، ایستا (ruff/bandit) و اجرا، تحلیل بافر pandas، باندل فرانت، Docker/CI | پایکل restricted-unpickler (OSSF/PandasStack/PSF)، PSI thresholds، DAMA-DMBOK/ISO 25012، OWASP API Security Top 10 2023، NIST SP 800-92 / RFC 9162 (tamper-evident logs) | suite کامل: ۱۱F/۱۳۴P/۱S؛ integration: ۱F/۲P؛ fingerprint collision زنده؛ verify_chain با فیلدهای دستکاری‌شده؛ s04 hypothesis → 500 |
| **۲** | بازبینی متمرکز یافته‌ها + استانداردهای کیفیت و AI | ISO/IEC 25010:2023 (۹ مشخصه)، NIST AI RMF 1.0 (+GenAI Profile)، CRISP-DM، OpenAPI/FastAPI: import تحت `TYPE_CHECKING` + `from __future__ import annotations` → 422/۵۰۰ (سه منبع PR/Issue مستقل هم‌راستا)، split-conformal ceil((n+1)(1−α))/n | کل suite دوباره اجرا شد (همان ۱۱ شکست، پایدار)؛ ۱۸/۱۸ پلاگین دامنه؛ خروجی bundle size؛ Bandit: ۰ High / ۲۴ Medium / ۸۷ Low؛ ruff: 13 |
| **۳** | تأیید نهایی ارقام + رفع ابهام‌های باقی‌مانده (s09 زنده، root-cause SEC-001 با دیس‌اسمبل pickle، اعتبارسنجی `_resolve_or_persist_secret` در production) | Efron 1977 (ties در Cox PH)، چندمنابع اضافی برای conformal/PH | کل suite و integration بار سوم (اعداد پایدار ماند) |

---

## ۳. نتایج تست — همهٔ شکست‌ها با ریشهٔ اثبات‌شده

**Suite واحد/e2e:** `11 failed / 134 passed / 1 skipped` (دو بار و سه بار تکرار شد؛ اعداد پایدار).
**Integration:** `1 failed / 2 passed` (فایل `tests/integration/test_full_platform.py`).

| # | تستِ شکست‌خورده | نوع خطا | ریشهٔ اثبات‌شده (فایل:خط) | دسته |
|---|---|---|---|---|
| 1 | `tests/integration/test_full_platform.py::test_mlops_copilot_reporting_and_18_domains` | `SecurityError OmniForge-SEC-001` | pickle مصنوع مدل روی py≤3.11 ایندوینت `builtins.getattr` منتشر می‌کند؛ denylist نماد `getattr` در `core/security.py:411,473` آن را می‌بلعد — §۴٫۱ | SEC-001 |
| 2 | `tests/unit/test_modeling_and_decision_engines.py::test_baselines_classical_ensembles_and_save_load` | همان SEC-001 | همان ریشه (save→load round-trip همان مدل) | SEC-001 |
| 3 | `tests/e2e/test_omniforge_task33.py::...:176` | `422 == 200` | اندپوینت‌های s14 مدلِ درخواست را فقط زیر `TYPE_CHECKING` import کرده‌اند؛ FastAPI آن را query لازم تفسیر می‌کند (`studios/s14_supervisory.py:24-31,94,329,354,379,437`) — §۵٫۱ | s14/TYPE_CHECKING |
| 4 | `tests/e2e/test_omniforge_task34.py::...:305` | `422 == 200` | همان ریشه (`s14/evaluate-demo-batch`) | s14/TYPE_CHECKING |
| 5 | `tests/e2e/test_omniforge_task35.py::...:120` | `422 == 200` | همان ریشه | s14/TYPE_CHECKING |
| 6 | `tests/e2e/test_omniforge_task37.py::...:237` | `422 == 200` | همان ریشه | s14/TYPE_CHECKING |
| 7 | `tests/e2e/test_workbench_spa.py::test_spa_static_bundle_gzip_and_all_13_studios_e2e` | `405 == 200` | روتر استودیو ۹ خالی است (`studios/s09_opt_sim.py:1-7`)؛ catch-all فقط GET است → POST=405 — §۵٫۲ | s09/EMPTY |
| 8 | `tests/unit/test_p0_security_and_runtime_hardening.py::test_p0_7_p0_8_p0_9_runtime_logging_docker_and_examples_hygiene:337` | `405 == 200` | همان ریشه (`POST /api/workbench/s09/graph`) | s09/EMPTY |
| 9 | `tests/unit/test_comprehensive_reporting.py::test_comprehensive_report_exports_docx_pdf_xlsx_csv_with_tables_charts_exclusions_and_equations` | `TypeError: float(None)` | `data/quality.py` بدون ستون زمان → `timeliness=None`؛ `reporting/generator.py:260` بدون None-check `float(v)` می‌کند — §۵٫۵ | timeliness |
| 10 | `tests/unit/test_data_layer.py::test_cleaning_validation_quality_versioning_feature_store` | `assert set(dimensions) == {...}` | کلید اضافه `proxy_accuracy` (alias تکراری `accuracy`) در `quality.py`؛ تست قراردادی ریپو را خودمتناقض کرده — §۵٫۶ | proxy_accuracy |
| 11 | `tests/e2e/test_workbench_spa.py::test_dirty_upload_benchmark_csv_and_structured_json_errors:112` | `500 == 200` | `s07/fairness`: `.fillna("Missing")` روی ستون Categorical → `TypeError`؛ بدون try/except → 500 (`studios/s07_advanced.py:56`) — §۵٫۴ | s07/Categorical |

**هشدار مهم:** شکست‌های ردهٔ SEC-001 در این sandbox به‌دلیل Python 3.11 است؛ در CI (3.12/3.13) ایندوینت‌های enum به‌صورت by-value pickled می‌شوند (راستی‌آزمایی منبع CPython 3.12.12/3.13.9/3.14.0) — ولی **دقیقاً همان سناریو یک شکنندگی طراحی‌ است** که در §۴٫۱ تشریح می‌شود.

**Test-debt:** پوشش `pytest` در CI فقط روی ۲ فایل (`test_audit_regressions.py`, `test_anti_leakage_and_pipeline.py`) اجرا می‌شود — هیچ‌کدام از ۱۲ شکست فعلی در CI رؤیت‌پذیر نیست.

---

## ۴. یافته‌های امنیتی

### ۴٫۱ [بحرانی | SEC-001] Round-trip مصنوع مدل شکست می‌خورد — ریشهٔ قطعی شناسایی شد
**روش تأیید:** دیس‌اسمبل بایت pickle واقعی + اجرای زنده round-trip (`ClassicalMLModel.fit(...).save()` سپس `safe_joblib_load_bytes`).

- CPython ≤3.11: متد `enum.Enum.__reduce_ex__` عیناً `(builtins.getattr, (EnumClass, member_name))` برمی‌گرداند → بایت artifact حاوی `STACK_GLOBAL builtins getattr` است (تأیید با `binascii/repr` روی بایت واقعی).
- CPython 3.12+ این reduce را به by-value تغییر داده (راستی‌آزمایی روی سورس‌های رسمی 3.12.12 / 3.13.9 / 3.14.0 از طریق `api.github.com`).
- `core/security.py:411-437` نماد `getattr` را به‌صورت مطلق در denylist گذاشته و `_validate_unpickle_target` آن را مسدود می‌کند؛ نتیجه: **آخرین مدلی که فریم‌ورک خودش ذخیره کرده، قابل بارگذاری نیست** (در 3.11 قطعی؛ در ≥3.12 احتمالاً by-value دور می‌زند — در این sandbox غیرقابل اجرا و در CI هم تست round-trip وجود ندارد).

**نقص‌های طراحی هم‌پیوند:**
- denylist بر پایهٔ «نام نماد» است نه زمینه (`getattr` با آرگومان‌های enum-class+member ذاتاً بی‌خطر است) و با نسخهٔ پایتون جابه‌جا می‌شود → شکنندگی بین نسخه‌ای مصنوعات.
- `joblib.dump` برای serialization اصلی artifact استفاده می‌شود؛ محدودیت unpickler تنها لایهٔ دوم و پس از HMAC است (درست) اما round-trip هیچ‌جا در CI گیت نمی‌شود.
- payloadهای فعلی حتی با HMAC معتبر روی py3.11 کرش می‌کنند → ریسک عملیاتی (شکست deploy/rollback).

**اصلاح پیشنهادی (خروجی ممیزی، اعمال نشد):**
1. artifact round-trip test را در CI برای همهٔ نسخه‌های پشتیبانی‌شده (3.12/3.13/3.14) اضافه کنید.
2. deny on *argument shape* به‌جای فقط name، یا امضای artifact با نسخهٔ پایتون و پروسهٔ بازامضا/بازپک؛ یا حذف `getattr` از denylistِ بدون‌شرط و جایگزینی با allowlist سخت‌گیرانه (همین‌الان `_ALLOWED_UNPICKLE_BUILTINS` وجود دارد — آن را به getattr **با signature مشخص** وصل کنید).
3. ترجیحاً serialization artifact را از pickle خام به فرمت پایدار (ONNX / skops / omniforge-native JSON) منتقل کنید برای مدل‌هایی که نیاز دارند بین نسخه‌ها جابه‌جا شوند.

### ۴٫۲ [بحرانی/بالا | API1:2023 BOLA] نشست Workbench = هدر خام `X-Session-ID`
`api/workbench.py:93`: `_sid()` فقط `.strip()` روی هدر کلاینت انجام می‌دهد؛ **هیچ اتصال (binding) به principal تأییدشده** (subject/tenant/session-ownership) وجود ندارد؛ همهٔ `/api/workbench/*` پشت `require_workbench_auth` (X-API-Key / Bearer / HMAC-cookie + CSRF double-submit) هستند ولی **کل state استودیوها با هدر کلاینت انتخاب می‌شود**. نتیجه: هر caller معتبر می‌تواند هر `X-Session-ID` دیگری را انتخاب کند و به دیتافریم/مدل/نتیجهٔ همان نشست دسترسی ببیند/تغییر دهد (cross-session data exposure؛ BOLA/IDOR). مطابق OWASP API Security Top 10 2023، API1 (BOLA) شماره‌یک است (برآورد رایج بین منابع OWASP ≈۴۰٪ حملات API؛ در 2019 و 2023 هر دو #1) و «هرگز هدر/شناسهٔ کلاینت را مالک چیزی نگیرید».
**اصلاح:** sid را از principal احرازشده مشتق کنید (مثلاً HMAC signing روی sid + claim تطبیق با subject/tenant)، یا session را server-side با ownership map نگه دارید و تست Two-Identity در CI بزنید.

### ۴٫۳ [بالا | Tamper-evidence] زنجیرهٔ هش audit فقط زیرمجموعهٔ فیلدها را می‌پوشاند؛ امضا ندارد
`core/security.py:784-823,830-860`: `log_event` و `verify_chain` هر دو دقیقاً روی `{event_id,timestamp,tenant_id,actor,action,resource_type,resource_id,status,previous_hash}` هش می‌کنند. **`role`، `trace_id` و `details` خارج از زنجیره‌اند** — با ویرایش آن‌ها در فایل JSONL، `verify_chain()` همان `True` برمی‌گرداند (تست زنده انجام شد). همچنین زنجیره بدون کلید/بدون anchor خارجی است؛ مهاجمِ دارای دسترسی نوشتن به فایل می‌تواند کل زنجیره را از نو محاسبه کند (هدف NIST SP 800-92 / RFC 9162 «tamper-evidence» با checkpoint بیرونی + پوشش کامل payload است).
**اصلاح:** canonical(payload) کامل + HMAC-SHA256 (کلید جداسازی‌شده از منطق اپ) + checkpoint/anchor بیرونی (مثلاً RFC 9162-style یا حداقل یک ریشهٔ هش بیرونی) + تست «یک کاراکتر تغییر → کل رکوردهای وابسته flag شوند».

### ۴٫۴ [متوسط-بالا | Tenancy] نقش «admin» از ایزولاسیون tenant معاف است
`core/security.py:636`: `if "admin" not in principal.roles and principal.tenant_id != resource_tenant_id` — هر principal با نقش سادهٔ `admin` می‌تواند به دادهٔ **هر tenantی** دسترسی بگیرد؛ هیچ محدودیتی بر privileged role یا step-up وجود ندارد.
**اصلاح:** privileged cross-tenant access را در یک نقش جدا (`platform_admin`) با ABAC/policy صریح و audit اجباری قرار دهید؛ `admin` را فقط در tenant خودش معتبر کنید.

### ۴٫۵ [متوسط] Salt پیش‌فرض عمومی برای pseudonymization
`core/security.py:731`: `salt: str = "uap-pii-salt"` — مقدار ثابت و عمومی → خروجی hash برای یک مقدار معین همیشه یکسان و قابل دیکشنری/brute-force بین سیستم‌هاست؛ هدف salt (یکتایی هر deployment) نقض می‌شود.
**اصلاح:** salt را الزامی و از KMS/env per-tenant بگیرید؛ بدون آن anonymize اجازهٔ اجرا ندهد.

### ۴٫۶ [متوسط | Static analysis] یافته‌های Bandit (medium-confidence)
- `data/ingestion.py:530` — `B301` pickle در مسیر `allow_unsafe_pickle`؛ گیت آگاهانه وجود دارد (پیش‌فرض `False`) ولی مسیر همچنان ریسک را به caller منتقل می‌کند. توصیه: حذف کامل گزینهٔ unsafe pickle یا الزام به dry-run اسکن + محدودسازی بیشتر.
- `data/ingestion.py:549` — `B314` `ET.fromstring` (fallback بعد از defusedxml)؛ XML دارای `DOCTYPE/ENTITY` از قبل رد می‌شود و defusedxml اول تلاش می‌شود — ریسک پایین ولی بهتر است fallback کاملاً حذف شود.
- `data/ingestion.py:613,617` — `B608`: f-string `SELECT * FROM {table_name}` (identifier قابل‌پارامتره نیست؛ sanitize نشده). ریسک identifier-injection؛ با allowlist نام جدول/نقل‌قول صحیح مهار شود.
- ۱۱ × `B104` bind روی همهٔ اینترفیس‌ها (بیشترشان opt-in برای deploy داکری‌اند؛ مستند شود و برای dev پیش‌فرض 127.0.0.1 باشد — در کد همین الگو وجود دارد ✅).
- **۰ High** در Bandit (Medium 24 / Low 87).

### ۴٫۷ [متوسط] نبود CSP/HSTS روی خود اپ
`api/main.py:94-100`: middleware فقط `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy` را setdefault می‌کند. **هیچ `Content-Security-Policy` و `Strict-Transport-Security` روی اسپا/API نیست.** (Pop-out window در فرانت CSP بسیار سخت‌گیرانه `default-src 'none'` دارد ✅ و subtree را sanitize می‌کند ✅ — ایراد فقط روی surface اصلی است.)

### ۴٫۸ [بالا، اما خودترمیم] اسرار خالی در compose — در production fail-fast است (رفتار درست، UX نامناسب)
`docker/docker-compose.yml`: چهار secret (`JWT_SECRET_KEY`, `ENCRYPTION_KEY`, `ARTIFACT_SIGNING_KEY`, `API_KEY`) پیش‌فرضِ خالی دارند، ولی `core/security.py:109-115` در production با whitespace/missing یا placeholder مقدار **SecurityError** می‌دهد → کرش استارت، نه کلید ضعیف. پس این یک آسیب‌پذیری فعال نیست؛ نقص در **تجربهٔ استقرار** است: بهتر است `docker compose config` با `:?` interpolation و یا اسکریپت `omniforge doctor` در استارت پیش‌چک کند و پیام صریح و خطای قابل اقدام بدهد. ایمیج compose هم non-root (uid 1000) و healthcheck دارد ✅.

### ۴٫۹ [مثبت تأییدشده — بدون ایراد]
- گیت `allow_unsafe_pickle=False` با پیام و پیشنهاد جایگزین ✅؛ XML `DOCTYPE/ENTITY` rejection + defusedxml-first ✅؛ YAML `safe_load` ✅؛ Mongo _id stringification ✅.
- `/workspace/save|load`: integrity ZIP (allowlist اعضا + cap حجم)، SHA-256 payload، **HMAC-SHA256 (require_signature=True)** پیش از `safe_joblib_load_bytes` ✅؛ `.omniforge` format (manifest+session_state) با همین امضا ✅.
- JWTManager با الزامات طول/placeholder در production ✅؛ secrets در dev به‌صورت 0600 در `~/.omniforge/keys` persist می‌شوند ✅.
- Artifact signing (HMAC) روی همهٔ مدل‌های save شده ✅ (اما §۴٫۱ round-trip را می‌شکند).
- شرط CORS پیش‌فرض فقط localhost است ✅؛ GZip فعال ✅؛ CSRF double-submit برای cookie auth ✅.

---

## ۵. یافته‌های درستی و سازگاری (Correctness / API Contract)

### ۵٫۱ [بحرانی | CAN-BUY-BUT-BROKEN] استودیو ۱۴ — ۵ اندپوینت با 422 و `/openapi.json` با 500
**تأیید زنده:** `POST /api/workbench/s14/quick-train` و `/s14/evaluate-demo-batch` → **422** با `loc=["query","req"]`؛ `/openapi.json` → **500 PydanticUserError** (ForwardRef تعریف‌نشدهٔ `SupervisoryQuickTrainReq`)؛ `/docs` 200 ولی schema آن 500.
**ریشه:** `s14_supervisory.py:24-31` مدل‌های درخواست را فقط زیر `if TYPE_CHECKING:` import می‌کند و فایل `from __future__ import annotations` دارد → FastAPI در runtime annotationها را علیه globals resolve می‌کند؛ نام‌ها نیستند → fallback به query parameter لازم → هر POST بدنه‌دار 422؛ `app.openapi()` هم برای ForwardRef تعریف‌نشده `PydanticUserError` می‌دهد. (سه منبع مستقل PR/Issue همین مکانیزم را هم‌راستا مستند کرده‌اند — بخش ۱۰.)
**اندپوینت‌های معلق (هر ۵ تا در فرانت فراخوانی می‌شوند `Studio14Supervisory.tsx:148,229,253,266,287`):** `s14/quick-train:94`، `s14/evaluate-sql:329`، `s14/evaluate-access:354`، `s14/evaluate-demo-batch:379`، `s14/evaluate-single:437`.
**اثر:** کل قابلیت Supervisory Prediction Monitor برای کاربر نهایی غیرقابل استفاده است و spec OpenAPI ماشین‌خوان (برای codegen/کلاینت‌ها) شکسته است.
**اصلاح:** import مدل‌ها در runtime (با `noqa: TC001` در صورت نیاز به مهار lint) — یا حذف `from __future__ import annotations` با ایمپورت مستقیم؛ به‌همراه تست سقف «همهٔ routeهای بدنه‌دار 422 ندهند» و `GET /openapi.json == 200` در CI.

### ۵٫۲ [بحرانی | MISSING] استودیو ۹ — یک روتر خالی که به‌عنوان «آماده» ship شده
`studios/s09_opt_sim.py` فقط `router = APIRouter(tags=["workbench-s09"])` است (۰ مسیر) و `workbench.py:1853,1869` آن را include می‌کند. فرانت ۶ مسیر را صدا می‌زند (`/s09/vision|survival|recommend|nlp|graph|backends`) و **هیچ‌کدام وجود ندارند**: POST → **405** (تأیید زنده)، GET → **404** (تأیید زنده). تست‌های `task33`, `p0_security...:337` و e2e SPA همین‌جا می‌شکنند. مدل‌های زیرین (survival در `models/survival`, NLP در deep) در کد موجودند ولی به API وصل نشده‌اند.
**اصلاح:** پیاده‌سازی routeها یا غیرفعال‌سازی explicit (410 Gone / feature-flag) + contract test برای هر مسیری که فرانت صدا می‌زند؛ حذف «قابلیت‌های نمایشی» از README تا نرسیدن به production (API9: Improper Inventory Management).

### ۵٫۳ [بالا | RUNTIME 500] استودیو ۴ — `HTTPException` import نشده (۸× F821) + 500 زنده
`studios/s04_stats.py` خطوط ۱۲۹، ۱۴۳، ۱۷۵، ۱۹۲، ۲۰۶، ۲۳۲، ۲۳۷، ۳۱۶ — نام `HTTPException` تعریف‌نشده؛ ruff آن را گزارش می‌کند. **تأیید زنده با TestClient:** `POST /api/workbench/s04/hypothesis` (روی دادهٔ ۵ردیفی) → **500 unhandled `NameError: name 'HTTPException' is not defined`** (لاگ `unhandled_api_error` ثبت شد). نکتهٔ مهم: `ruff check src tests` در حال حاضر با خروجِ غیرصفر **۱۳ خطا** گزارش می‌کند (۸ F821 + ۵ I001) — یعنی مرحلهٔ lint همین CI فعلی قرمز است و تا رفع این خطوط، merge گیت نمی‌شود (یا اگر merge می‌شود، gate دور زده شده).

### ۵٫۴ [متوسط-بالا] استودیو ۷ fairness — 500 روی ستون Categorical + ساختار خطای ۵۰۰ نادرست
`studios/s07_advanced.py:56`: `sens_source[col].reindex(X_te.index).fillna("Missing").astype(str)` — اگر ستون sensitive از نوع `pd.CategoricalDtype` باشد (که خروجی خود مرحلهٔ cleaning است)، `fillna` با category جدید `Missing` → `TypeError: Cannot setitem on a Categorical with a new category`. بدون try/except → ۵۰۰ (تأیید زنده در `test_dirty_upload...:112`). دو نقص: (الف) بهم‌خوردن ترتیب fillna/astype؛ (ب) نگرفتن exception و برگرداندن 500 به‌جای 4xx/ساختاریافته.

### ۵٫۵ [متوسط] `timeliness=None` → کرش مسیر گزارش
بدون ستون timestamp، `DataQualityEvaluator` مقدار `timeliness=None` می‌گذارد (طراحی عمدی) ولی `reporting/generator.py:260` در مونتاژ جدول 6D همهٔ مقادیر را `float(v)` می‌کند → `TypeError` (تست `test_comprehensive_reporting` می‌شکند). داک‌رشتهٔ generator باید None را به `N/A` تبدیل کند یا quality باید کلید را حذف کند؛ contract صریح شود.

### ۵٫۶ [متوسط | CONTRACT] alias تکراری `proxy_accuracy` + تست‌های خودمتناقض
`quality.py` کلیدهای `accuracy` و `proxy_accuracy` را **با مقدار یکسان** خروجی می‌دهد (`"proxy_accuracy": round(accuracy, 2)`)؛ `evaluable_weights` آن را در overall نادیده می‌گیرد (dead alias در payload). همزمان دو تست ریپو در تضادند: `test_audit_regressions.py:724-731` وجود `proxy_accuracy` را assert می‌کند و `test_data_layer.py` برابری دقیق set کلیدها (بدون آن) را assert می‌کند → یکی همیشه red می‌ماند (وضعیت فعلی). قرارداد API کیفیت داده باید یک‌بار تعریف و مستند شود؛ کلید alias حذف و تست‌ها همراستا شوند.

### ۵٫۷ [متوسط] fingerprint زیرمجموعه‌ای DataFrame → برخورد قطعی (cache poisoning)
`core/performance.py:21-36`: اگر n>64 باشد، فقط ۶۴ ردیفِ `linspace` برای blake2b انگشت‌نگار استفاده می‌شود → ردیف‌های بیرونِ نمونه دیده نمی‌شوند. **تأیید زنده:** در df با ۱۰۰۰ ردیف، تغییر ردیف ۵۰۰ (افزایش عددی +۱۰٬۰۰۰ و برگرداندن target) انگشت‌نگار **یکسانی** داد → `GLOBAL_DF_CACHE` (که کلید `data_quality`, `format_audit` و quality را با این fingerprint می‌سازد) نتیجهٔ کهنه را برمی‌گرداند.
**اصلاح:** fingerprint را از `pd.util.hash_pandas_object` روی همهٔ ردیف‌ها (سریع و deterministic) یا حداقل روی نمونه+چند percentile/hash مکمل بگیرید؛ یا TTL/versioning روی cache. برای داده‌های PII/regulated استفاده از این cache باید opt-out باشد.

### ۵٫۸ [مثبت / قابل‌قبول] سایر موارد
- Ingestion برای فرمت‌های exotic (SAS/SPSS/HDF5/...) graceful و با پیام/پیشنهاد است ✅.
- `CombinatorialPurgedKFold` (CPCV) purge/embargo متقارن sanity دارد و ورودی‌ها را validate می‌کند ✅.
- معیارها (`evaluation/metrics.py`): log_loss/AUC/KS/MASE/Winkler درست؛ C-index Harrell با tie→0.5 ✓ (تأیید عددی بخش ۶).
- Fairness: DI 0.80 / DP 0.10 / EoDD 0.10 پیش‌فرض‌های متعارف (Kamiran–Calders reweighing) ✓.
- `Page-Hinkley` با بازنشانی پس از change-point و حالت up/down/both ✓ (مطابق River/Gama 2014).
- `AutoML`: portfolio-based + fallback امن؛ GP-UCB در `HyperparameterOptimizer` با Matern(2.5) و optimizer فقط با ≥6 observation (تصمیم حساب‌شده) ✓؛ کاندیداهای شکست‌خورده warning می‌خورند نه کرش ✓.
- ~~`MetaLearningBayesianAutoML`~~ **با این پاس رفع شد**: کلاس به `MetaFeatureWarmStartAutoML` تغییر نام یافت (نام قدیمی به‌عنوان alias منقضی می‌ماند)؛ مستنداتِ docstring صریحاً اعلام می‌کند که در این موتور هیچ GP-BOای نیست (GP-BO واقعی در `HyperparameterOptimizer`).
- Workbench `/workspace/save|load`: HMAC + SHA-256 + cap حجم + allowlist اعضا ✅ (§۴٫۹).
- CORS پیش‌فرض localhost ✅؛ GZip ✅؛ health endpoints (`/health`, `/api/v1/health`) ✅.

---

## ۶. راستی‌آزمایی عددی/آماری (همه تأیید شدند — مطابق منبع)

| مورد | نتیجه | مرجع راستی‌آزمایی |
|---|---|---|
| Kaplan–Meier vs lifelines (n=300, seed 11) | ماکزیمم اختلاف جدول بقا ≈ **۳٫۶e-۰۷**؛ median یکسان | lifelines 0.30.3 |
| Cox PH vs lifelines (n=300, seed 11, Exponential T/E, Efron ties) | `concordance_index_` = **0.695106** vs lifelines **0.6951058** (اختلاف ~2e-07)؛ پیاده‌سازی خود Efron (نه Breslow) | Efron 1977؛ Stanford STAT331؛ R/ survival پیش‌فرض |
| C-index (Harrell, tie=0.5) | پیاده‌سازی درست و همراستا با lifelines | — |
| split-conformal quantile | `ceil((n+1)(1−α))/n` دقیق پیاده شده؛ پوشش تضمینی marginal | Angelopoulos & Bates 2021 (arXiv:2107.07511)؛ Lean formalization؛ منابع عملی ۲۰۲۶ |
| PSI (population stability) | محاسبه = 0.5467 دستی ✓؛ آستانه‌ها <0.1/0.1–0.25/>0.25 مطابق عرف صنعت (۱۰ bin، ε برای bin خالی)؛ «عرف de-facto» است نه استاندارد ISO | devopsness / futureagi / codefinity / mlflow (2025–2026) |
| Causal ATE (ATE=۲٫۰) | bias ≤ **۰٫۰۸** روی شبیه‌سازی | DoWhy/EconML-style sanity |
| Markowitz portfolio (`optimize_portfolio`) | SLSQP = فرمول بسته (Σ⁻¹μ/λ projection) با ماکزیمم اختلاف **۴٫۵۸e-۰۷** | فرم تحلیلی |
| MICE / Rubin | IterativeImputer + pooling جدی | statsmodels-style |
| 18/18 پلاگین دامنه (`engineer_domain_features` + `compute_domain_kpis`) | همگی ۳۰ ردیف، KPI معتبر می‌دهند | اجرا زنده |
| Initial SPA bundle | **~59 KB gzip JS (index 8.7 + vendor 1.8 + react 44.2 + icons 4.3) + 4.3 KB CSS ≈ 63 KB** — ادعای README (~63 KB) **درست** است | اندازه‌گیری محلی |
| JWT/RBAC | واقعاً پیاده شده (JWTManager + RBAC_ROLE_PERMISSIONS + ABAC) ✓ | `core/security.py:515,604` |
| requirements.txt (39 پین 2026) | همه در PyPI حل می‌شوند (0 unresolvable) | PyPI JSON API |
| `README.md` "18 domain plugins" / "14 studios" / CRISP-DM | تأیید شد (۱۸ پلاگین، ۱۴ استودیو) | اجرا + کد |

**نتیجه:** هستهٔ تحلیلی قابل اعتماد است؛ شکست‌ها زیرساختی/قراردادی/امنیتی‌اند، نه ریاضی.

---

## ۷. انطباق با استانداردها و چارچوب‌های جهانی

### ۷.۱ ISO/IEC 25010:2023 (SQuaRE — مدل کیفیت محصول) — نگاشت نه مشخصه
| مشخصهٔ 25010:2023 | وضعیت OmniForge | شواهد |
|---|---|---|
| Functional Suitability | چرخهٔ کامل ingest→report گسترده؛ **ولی** s09 مفقود، s14 شکسته، s04 500 → completeness پایین‌تر از ادعا | §3, §5 |
| Performance Efficiency | fingerprint سریع اما برخوردار (§۵٫۷)؛ GZip ✓؛ pagination/downsample ✓؛ benchmark وجود ندارد (نبود بودجهٔ کارایی در CI) | §۵٫۷ |
| Compatibility | REST/OpenAPI (ولی `/openapi.json` کرش §۵٫۱) + `.omniforge/.uap` + joblib/pickle artifacts (پیوند به نسخهٔ پایتون §۴٫۱) | §۴٫۱, §۵٫۱ |
| Interaction Capability | اسپای ۱۴-استودیویی غنی؛ ولی شکست‌های UI→API بالا (s14/s09/s07) = user-error-protection ضعیف | §۵ |
| Reliability | Fail-fast production secrets ✅؛ unhandled 500ها (s04/s07)؛ CI subset → faultlessness پایین‌تر از بلوغ ظاهری | §۴٫۸, §۵ |
| Security | HMAC artifact ✓، JWT ✓، CSRF ✓، fail-fast secrets ✓، ولی: BOLA، admin bypass، audit chain ناقص، نبود CSP — §۴ | §۴ |
| Maintainability | لایه‌بندی عالی، type annotations گسترده (ولی `mypy` در CI هست و در این ممیزی نصب نبود — باید همراستا شود)؛ lint debt ۱۳ مورد | §۸ |
| Flexibility | پلاگین SDK دامنه ✅؛ Docker multi-stage ✓؛ compose نقص‌های استقرار (§۴٫۸) | §۴٫۸ |
| Safety (جدید 2023) | OmniForge به safety-critical نزدیک نیست؛ ولی «fail-safe» در production secrets و عدم اعتماد به pickle بخشی از این ذهنیت است | — |

### ۷.۲ NIST AI RMF 1.0 (+ GenAI Profile 600-1)
| تابع RMF | وضعیت | یادداشت |
|---|---|---|
| **GOVERN** | ⚠️ ناقص | RBAC/JWT ✓، AuditLogger ✓ (ولی §۴٫۳)، tenant isolation (ولی §۴٫۴). نبود: AI risk register، نقش‌های accountable AI، inventory مدل‌ها (در عمل ModelRegistry موجود است ولی به ریسک سازمانی وصل نیست) |
| **MAP** | ⚠️ ناقص | domain plugins context می‌دهند؛ model cards/data cards ✅ (`evaluation/fairness.py` → card markdown). ناقص: ثبت هدف/دامنهٔ مجاز هر مدل، خط‌مشی داده |
| **MEASURE** | ✅ نسبتاً خوب | fairness (Kamiran–Calders، DI/DP/EoDD)، conformal coverage ✓، drift (PSI/Page-Hinkley/KS)، carbon tracking، ۱۸ KPI دامنه |
| **MANAGE** | ⚠️ ناقص | CT orchestrator/retrain وجود دارد؛ ولی rollback/kill-switch خودکار و incident response تعریف‌نشده؛ حتی ارجاعی به NIST SP 800-61 در docs نیست |

### ۷.۳ CRISP-DM (رایج‌ترین متدولوژی داده‌کاوی)
نگاشت ۱۴ استودیو روی فازها: Business/Data Understanding (s00–s03)، Data Preparation (s01/s05)، Modeling (s06/s08/s14)، Evaluation (s06/s07/s11)، Deployment (s11/s14/s12). **فاز Deployment/Monitor ناقص است چون s09 (Specialized AI) مفقود و s14 شکسته است**؛ همچنین نبود حلقهٔ صریح «بازگشت به Business Understanding» مستند نیست (اسپای چرخه‌ای وعده می‌دهد ولی contract تستی ندارد).

### ۷.۴ ISO/IEC 25012 / DAMA-DMBOK (کیفیت داده)
ابعاد اصلی: accuracy, completeness, consistency, timeliness, uniqueness, validity (+ integrity در DMBOK2). پیاده‌سازی `DataQualityEvaluator` این ۶ بعد + `proxy_accuracy` تکراری + `timeliness=None` دارد (§۵٫۵, §۵٫۶). «proxy_accuracy» با تعریف DAMA/ISO سازگار نیست (هیچ متادیتای منبع برای آن نیست). **توصیه:** حذف alias، None-safe کردن گزارش، و مستندسازی وزن‌دهی (weights) در README/docs.

### ۷.۵ OWASP API Security Top 10 — 2023 (گزیدهٔ تطبیق)
| OWASP | وضعیت |
|---|---|
| API1 BOLA | ❌ نشست workbench (§۴٫۲) — یافتهٔ شماره‌یک |
| API2 Broken Auth | ⚠️ JWT/API-Key خوب، ولی تأیید ماهیت sid نیست (§۴٫۲) |
| API3 BOPLA | ⚠️ مدل‌های پاسخ/فیلدهای قابل‌نوشتن در استودیوها whitelist ندارند (هر کلید state پذیرفته می‌شود) |
| API4 Unrestricted Resource Consumption | ⚠️ GZip ✓؛ نبود rate-limit/timeout سراسری روی routeهای سنگین (AutoML/train) |
| API5 BFLA | ⚠️ `/api/workbench/*` فقط پشت key/cookie است؛ هیچ authorize per-route/permission وجود ندارد |
| API8 Security Misconfig | ❌ نبود CSP/HSTS (§۴٫۷) |
| API9 Improper Inventory Management | ❌ اندپوینت‌های مفقود/stub در محصول ship شده‌اند (§۵٫۲) |

### ۷.۶ Tamper-evident logging (NIST SP 800-92 / RFC 9162 / DORA Art.12)
زنجیرهٔ هش موجود است (گام درست) ولی: (۱) پوشش فیلد ناکامل، (۲) unsigned، (۳) درگاه خارج از مرز نوشتن ندارد، (۴) تست «flip یک کاراکتر → کل رکوردهای وابسته flag شود» وجود ندارد. جزئیات و اصلاح در §۴٫۳.

### ۷.۷ ارزیابی پایکل به‌مثابه مرز اعتماد (OSSF Python Secure Coding Guide / PSF pickle docs)
هدایت OSSF: هر `__reduce__/__reduce_ex__/__setstate__/__del__` را ممیزی کنید و restricted unpickler را جایگزین اعتماد به pickle نکنید؛ allowlist ≠ ادعای «کد قابل اعتماد». OmniForge گیت `allow_unsafe_pickle=False` و restricted unpickler را دارد ✅ ولی (الف) artifact فریم‌ورک روی نسخهٔ پشتیبانی‌نشدهٔ پایتون بار نمی‌شود (§۴٫۱)، (ب) torch/catboost deserialization ذاتاً جزئی است و باید در threat model صریح باشد.

---

## ۸. بدهی فنی و نگهداری

| # | مورد | جزئیات/شواهد |
|---|---|---|
| T1 | **CI فقط ۲ فایل تست را اجرا می‌کند** | `.github/workflows/ci.yml`: `pytest ... test_audit_regressions.py test_anti_leakage_and_pipeline.py` — ۱۱ شکست فعلی در CI نامرئی‌اند. |
| T1b | **مرحلهٔ ruff در CI قرمز است** | `ruff check src tests` خروج غیرصفر با ۱۳ خطا (۸ F821 + ۵ I001) — gate کیفیت فعلی عملاً شکسته یا دور زده می‌شود. |
| T2 | **ruff: ۱۳ خطا** | ۸× F821 (s04) + ۵× I001 (ترتیب import). (`ruff check src tests`) |
| T3 | **ادعاهای مستندات قدیمی/غیرقابل‌تکرار** | `docs/P2_REMEDIATION_REPORT.md` می‌گوید ruff 0، mypy 0 (۱۶۶ فایل)، pytest «85 passed»؛ وضعیت فعلی: ruff 13 و suite 134P/11F (~146). نسخهٔ README ادعای «~63 KB gzip» و «18 domain plugins» را درست می‌گوید ✅. |
| T4 | ~~`MetaLearningBayesianAutoML` نام گمراه‌کننده~~ **رفع شد**: کلاس → `MetaFeatureWarmStartAutoML`؛ alias قدیمی حفظ شد؛ docstring صریحاً warm-start است نه GP-BO. | انجام شد. |
| T5 | **dependency pins 2026** در `requirements.txt` همه از PyPI حل می‌شوند ✅ ولی در py3.11 قابل‌نصب نیستند و `.venv-audit` با نزدیک‌ترین نسخه‌ها ساخته شد — رفتار دقیق روی 3.12/3.13/3.15 فقط در CI قابل‌راستی‌آزمایی است (بعد از رفع T1). | |
| T6 | **FutureWarning** scikit-learn 1.9: `penalty` deprecated در `LogisticRegression` (تست integration warning می‌دهد) — در 1.10 حذف می‌شود | پیشنهاد: `l1_ratio`/`C` جایگزین |
| T7 | **sklearn FutureWarning + pandas 3.1** (copy-on-write) با کد منبع همراستاست ✅؛ ولی `pd.Categorical` در مرز cleaning→fairness می‌شکند (§۵٫۴) | شمول §۵٫۴/P1-8 |
| T8 | **`.audit/src` و `.venv-audit/`** untracked‌اند؛ در تحویل نیستند. | پس از تأیید می‌توانید پاکشان کنید. |
| T9 | **آی‌کن/نام نسخهٔ ایمیج**: compose `omniforge-enterprise:2.0.0` در حالی که README برخی جاها 1.0.0 می‌گوید (ناسازگاری نسخه در docs/packaging). |
| T10 | **Tests اکنون به ۳ مجموعه تقسیم شده‌اند** (unit/integration/e2e) ولی CI فقط unit-گزیده را اجرا می‌کند؛ e2eها ~۲٬۵۰۰ خط‌اند و با UI سروکار دارند (نیاز به job جداگانه و cache node). |
| T11 | **مجوزها/انتساب‌ها** ✅ (Apache-2.0 کامل + header). |
| T12 | **Interface `.uap`/`.omniforge`** مستند شده (`workspace_session.py`) ✅ ولی بدون امضای متقابل تجاری/چند-tenant؛ فقط HMAC سرور. |

---

## ۹. اولویت‌بندی اصلاحات (خروجی ممیزی — اعمال نشده)

### P0 — اکنون (اثر بالا، ریسک فعال)
1. **s14/TYPE_CHECKING** → import مدل‌ها در runtime؛ اضافه‌کردن `GET /openapi.json == 200` و smoke-test هر route بدنه‌دار به CI. (§۵٫۱)
2. **SEC-001** → artifact round-trip در CI روی 3.12/3.13/3.14 + بازطراحی denylist به allowlist-با-context یا حذف getattr-بدون‌شرط؛ بازامضای artifacts در migration guide. (§۴٫۱)
3. **BOLA workbench** → bind sid به principal (HMAC روی sid با subject/tenant) + تست دو هویتی در CI. (§۴٫۲)
4. **s09 stub** → یا پیاده‌سازی routeها یا explicit 410 + feature-flag؛ حذف هرگونه نمایش «Specialized AI آماده» تا نرسد. (§۵٫۲)
5. **s04 HTTPException** → یک import + رفع ۸ F821 + تست `s04/hypothesis == 200`. (§۵٫۳)
6. **Audit chain** → پوشش canonical کامل payload + HMAC + تست flip-1-char. (§۴٫۳)
7. **CI** → اجرای کل suite (unit+integration) + e2e در job جدا؛ جلوی merge با red test (branch protection). (T1)

### P1 — این اسپرینت (درستی و اعتماد قرارداد)
8. s07 fairness: `astype(str)` قبل از `fillna`؛ try/except → 4xx ساختاریافته. (§۵٫۴)
9. Quality: None-safety timeliness در `generator.py`؛ حذف alias `proxy_accuracy`؛ همراستا کردن دو تست متعارض؛ مستندسازی weights. (§۵٫۵, §۵٫۶)
10. `GLOBAL_DF_CACHE` fingerprint → full-hash (`pd.util.hash_pandas_object` روی همهٔ ردیف‌ها) یا TTL؛ برای داده‌های حساس opt-out. (§۵٫۷)
11. CSP/HSTS روی app اصلی (`api/main.py`)؛ `Object-src 'none'`, `img-src 'self' data: blob:`؛ حالت گزارش-اول (`Content-Security-Policy-Report-Only`) برای rollout. (§۴٫۷)
12. Tenant isolation: نقش `platform_admin` جدا؛ audit اجباری برای cross-tenant. (§۴٫۴)
13. PII salt: الزامی/per-tenant از env یا KMS. (§۴٫۵)
14. Ingestion: حذف fallback غیر-defusedxml XML و sanitize identifier در `sqlite/duckdb` (یا allowlist). (§۴٫۶)

### P2 — بهبود مستمر (بلوغ)
15. مستندات: `P2_REMEDIATION_REPORT.md` را «historical» علامت بزنید یا با ارقام تازه به‌روز کنید؛ README را با وضعیت واقعی s09/s14 همراستا؛ T9 نسخه‌ها را یکسان‌سازی. (T3, T9)
16. `mypy` strict را در CI سبز کنید (پیکربندی فعلی هست؛ نیاز به اجرای واقعی + ignoreهای محدود). (T2/T3)
17. Rate-limit/timeout سراسری برای routeهای AutoML/train؛ abort برای jobهای >N دقیقه. (§۷٫۵)
18. ~~Name‌گذاری `MetaLearningBayesianAutoML`~~ انجام شد: `MetaFeatureWarmStartAutoML` + alias منقضی (T4).
19. ~~Benchmark کارایی (latency p95/p99، memory) در CI با بودجهٔ مشخص.~~ **انجام شد (§۷٫۱)**: `scripts/latency_budget_benchmark.py` + job `latency-budget` در CI.
20. ~~NIST AI RMF: risk register + kill-switch + مدل‌کارد PDF-export~~ **انجام شد**: `docs/RISK_REGISTER.md`، `OMNIFORGE_CT_DISABLED`/`OMNIFORGE_CT_AUTO_PROMOTE`، `ModelCard.to_pdf`. (§۷٫۲)
21. ~~`requirements.txt` را با lock-file جایگزین کنید~~ **انجام شد (T5)**: `requirements.in` + lock hash-pinned؛ `requirements.txt` شیم hash-checked.
22. e2e jobs در CI با cache npm + Playwright/TestClient؛ جلوگیری از رگرسیون UI→API (مانند s14). (T10)

---

## ۱۰. منابع آنلاین (راستی‌آزمایی دورها)

**Pickle / deserialization**
- PSF docs — pickle Restricting Globals: https://docs.python.org/3/library/pickle.html
- OSSF Python Secure Coding Guide, issue #4 (`__reduce__`/`__setstate__`/`__del__` audit): https://github.com/ossf/python-secure-coding-guide/issues/4
- PandasStack — محدودیت‌های «sandbox» کردن پایکل (allowlist ≠ ادعای trusted-code؛ cap decode): https://pandastack.ai/blog/python-pickle-deserialization-sandbox
- PSF/CPython enum `__reduce_ex__` در 3.12.12/3.13.9/3.14.0 (تغییر به by-value): بررسی از طریق `api.github.com` (raw source) — 3.11.2 محلی: `(getattr, (T, 'BINARY_CLASSIFICATION'))` (تأیید اجرایی)
- Restricted unpickler pattern (allowlist قبل از هر `find_class`) — pandastack.ai بالا + PSF asyncio docs

**FastAPI / OpenAPI (TYPE_CHECKING root cause §۵٫۱)**
- Microsoft agent-governance-toolkit PR #4328 (همان bug: 422 + openapi 500): https://github.com/microsoft/agent-governance-toolkit/pull/4328
- protoLabsAI/social-plugin PR #6 (openapi.json 500 PydanticUserError): https://github.com/protoLabsAI/social-plugin/pull/6
- PrefectHQ/fastmcp issue #5801 (PEP 563 + TYPE_CHECKING → PydanticUserError): https://github.com/PrefectHQ/fastmcp/issues/5801

**Cox PH / survival**
- Efron, B. (1977). «The efficiency of Cox's likelihood function for censored data», JASA 72:557–565 — ties؛ پیاده‌سازی §۶ راستی‌آزمایی شد
- Stanford STAT331 — Cox PH ties (Efron vs Breslow=default در R نیست؛ R از Efron استفاده می‌کند): https://web.stanford.edu/~lutian/coursepdf/unitcox1.pdf
- C. W. ... (Efron approximation): https://atrium.lib.uoguelph.ca/server/api/core/bitstreams/d5614132-6ac3-489c-a406-1803f9d2ed8b/content

**Conformal prediction**
- Angelopoulos & Bates (2021) A Gentle Introduction to Conformal Prediction, arXiv:2107.07511 (k=⌈(n+1)(1−α)⌉): https://arxiv.org/pdf/2107.07511
- StatLean conformal_coverage (formalization، rank uniformity): https://statlean.github.io/docs/StatLean/MultipleTesting/Conformal/Coverage.html
- datarekha.com — conformal practice (worked example, k> n ⇒ abstain): https://datarekha.com/ml/conformal-prediction/
- iotdigitaltwinplm.com — heteroscedastic conformal, marginal coverage caveat: https://iotdigitaltwinplm.com/conformal-prediction-uncertainty-quantification-machine-learning-coverage-guarantees-2026/

**PSI / Drift**
- devopsness.com (2026-10-31)، futureagi.com (2026-05-15)، codefinity.com، artificial-intelligence-wiki (2025-07-02)، mlflow.org (2026-07-22) — «de-facto» thresholds: <0.1 پایدار / 0.1–0.25 متوسط / >0.25 معنادار؛ 10 bin؛ ε برای bin‌های خالی
- Page, E.S. (1954); Gama et al. (2014) — Page-Hinkley (همان پیاده‌سازی استفاده شد ✅)

**کیفیت داده / ISO 25012 / DAMA**
- iagovernance.com (2026-05-23) و dataworkers.org (2026-04-10) — شش بعد اصلی DAMA-DMBOK2 (accuracy, completeness, consistency, timeliness, uniqueness, validity؛ + integrity در DMBOK2)
- thedatabrains.com (2025-07-24) — نگاشت ISO/IEC 25012
- DMBOK2 / DAMA International: https://www.dama.org

**امنیت API**
- OWASP API Security Top 10 — 2023 (API1 BOLA شماره‌یک 2019 و 2023؛ API4/API5/API8/API9): nhimg.org (راهنمای Top-10)، appsecmaster.net (2026-08-25)، cyberfortify.co (2026-08-02)، secportal.io

**Audit logging / tamper-evidence**
- ZATONA — tamper-evident audit logs (hash-chain + Merkle + external anchoring؛ flip-1-char test): https://zatona.dev/blog/tamper-evident-audit-logs
- techbloat.com — verify ALL fields؛ key separation؛ RFC 9162 checkpoints: https://techbloat.com/mcp-server-audit-logs
- dev.to/alessio99xx (2026-10-17) — HMAC chain روی seq:prev:canonical(payload): https://dev.to/alessio99xx
- vdf.ai — NIST SP 800-92 / DORA 2024/1774 Art.12 / ISO 27001 A.8.17 (clock sync) / RFC 9162: https://vdf.ai
- ir-os.com — intrusion/response audit trails

**کیفیت نرم‌افزار و AI governance**
- ISO/IEC 25010:2023 — SQuaRE Product Quality Model (۹ مشخصه؛ Safety و Resistance جدید): https://tnlab.org/en/iso-iec-250102023-square-product-quality-model/ ; توضیح تطبیقی: https://www.edgechat.ai/iso-iec-25010 (2026-10-15)
- ACM 2026 — ISO 25010 in DevOps: https://dl.acm.org/doi/10.1145/3727967.3756847
- NIST AI RMF 1.0 (100-1, Jan 2023) + GenAI Profile (600-1, Jul 2024) — GOVERN/MAP/MEASURE/MANAGE: https://blueradius.io/nist-ai-rmf-implementation-guide ; https://www.whistic.com/resources/frameworks/frameworks-nist-ai-rmf ; https://aigovernancecore.com/blog/nist-ai-rmf-complete-guide
- CRISP-DM 1.0 (IBM/SPSS 1999–2000؛ ASUM-DM 2015): https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining ; https://grokipedia.com/page/Cross-industry_standard_process_for_data_mining

**Dependency inventory (۳۹ پین)** — PyPI JSON API (`pypi.org/pypi/<name>/json`): همه حل شدند.

---

## ۱۱. محدودیت‌های این ممیزی

1. **محیط:** فقط Python 3.11.2 در دسترس بود؛ `requires-python >=3.12` و CI روی 3.12/3.13 راستی‌آزمایی **مستقیم** در این sandbox انجام نشد (ادعاهای مربوط به 3.12+ با source-reading و byte-inspection حمایت می‌شوند، نه اجرای محلی).
2. `mypy` در venv نصب نبود؛ ارزیابی T2/T3 با ruff و تست + CI-config است (نه اجرای mypy).
3. lifelines فقط در venv حاضر است؛ ارجاع‌های lifelines خود بخشی از اجرای راستی‌آزمایی‌اند.
4. فرانت بیشتر با grep/اسکن ایستا بررسی شد (بدون اجرای مرورگر/Playwright)؛ بندیت دو فایل py3.12-syntax را نادیده گرفت (قابل‌انتظار).
5. تست‌های ۳٫۱۱ با دو فایل scratch (`.audit/src`) اجرا شدند؛ اعداد ۱۱F/134P **تحت 3.11** معنا دارند؛ در CI احتمالاً شکست‌های SEC-001 دیده نمی‌شود ولی بقیه قابل انتظارند.
6. این سند **هیچ اصلاحی** در `src/` اعمال نمی‌کند؛ P0–P2 صرفاً فهرست اقدامند.
7. ممیزی امنیتی به‌صورت white-box و ایستا+اجرایی است؛ penetration test شبکه‌ای/red-team انجام نشده.

---

## پیوست A — دستورات بازتولید

```bash
cd /home/user/OmniForge
source .venv-audit/bin/activate
# Suite کامل (py3.11 + src تطبیق‌داده‌شده):
PYTHONPATH=$PWD/.audit/src python -m pytest tests -q
# Integration:
PYTHONPATH=$PWD/.audit/src python -m pytest tests/integration -q
# Lint:
ruff check src tests
# Security static:
bandit -r src -q --severity-level medium --confidence-level medium
# Round-trip artifact (root cause §۴٫۱):
PYTHONPATH=$PWD/.audit/src python -c "
import joblib, io
import pandas as pd, numpy as np
from omniforge.models.classical import ClassicalMLModel
from omniforge.core.types import TaskType
df = pd.DataFrame({'f0': np.arange(50.0), 'f1': np.arange(50.0)*2, 't': np.arange(50)%2})
m = ClassicalMLModel(algorithm='random_forest', task_type=TaskType.BINARY_CLASSIFICATION, n_estimators=5).fit(df[['f0','f1']], df['t'])
b = io.BytesIO(); joblib.dump(m, b)
from omniforge.core.security import safe_joblib_load_bytes
safe_joblib_load_bytes(b.getvalue())   # -> SecurityError SEC-001 روی py3.11
"
# Fingerprint collision (§۵٫۷) و verify_chain (§۴٫۳): بخش‌های ۲٫۳ و ۴٫۳ این گزارش
```

## پیوست B — فایل‌های کلیدی برای اصلاحگران
`src/omniforge/api/studios/s14_supervisory.py` (§۵٫۱) · `src/omniforge/api/studios/s09_opt_sim.py` (§۵٫۲) · `src/omniforge/api/studios/s04_stats.py` (§۵٫۳) · `src/omniforge/api/studios/s07_advanced.py` (§۵٫۴) · `api/workbench.py:93` (§۴٫۲) · `core/security.py:411-491,604,731,761-860` (§۴٫۱–۴٫۵) · `core/performance.py:21-70` (§۵٫۷) · `reporting/generator.py:260`, `data/quality.py` (§۵٫۵–۵٫۶) · `.github/workflows/ci.yml`, `docker/docker-compose.yml` (T1/§۴٫۸).

---

**جمع‌بندی نهایی:** OmniForge از نظر دامنه و معماری در کلاس «enterprise» طراحی شده و هستهٔ آماری آن قابل اعتماد است؛ اما در **قراردادهای API (422/405/500 زنده)، مرز اعتماد deserialization، مرز multi-tenancy/session و tamper-evidence لاگ** نقص‌های قطعی و قابل‌تکرار دارد. سه دور ممیزی انجام شد؛ اصلاحات P0 کم‌هزینه‌اند و بیشترین ریسک را حذف می‌کنند. توصیهٔ اکید: پیش از هر release، P0-1..P0-7 را انجام و سپس CI را روی کل suite باز کنید.


---

## پیوست الف — Remediation Log (اعمال‌شده پس از ممیزی، همان شاخه)

> **English note.** This appendix documents what was actually fixed after the audit on this branch (commit-ready), verified by three review+test passes and a live suite run. It uses the same priority scheme as §۹. Items not listed here were NOT changed; see «بازمانده‌ها».

### الف.۱ P0 — اعمال‌شده و اثبات‌شده با تست

| # | یافته | اصلاح اعمال‌شده | راستی‌آزمایی |
|---|---|---|---|
| P0-1 | استودیو ۱۴: مدل‌ها تحت `TYPE_CHECKING` → ‎422 + ‎`/openapi.json` = 500 | مدل‌های request به‌صورت runtime در `s14_supervisory.py` ایمپورت شدند (با توضیح ریشهٔ `from __future__ import annotations` در کد)؛ اندپوینت‌ها بدنهٔ درست دریافت می‌کنند | suite سبز؛ `test_workbench_spa` و `test_omniforge_task34` (۵ اندپوینت اصلی) |
| P0-2 | SEC-001: round-trip مصنوع مدل در unpickler محدود رد می‌شود | `SerializableStrEnum` (reduce by-value برای همهٔ enumهای قراردادی: TaskType/DataModality/ModelStage/ValidationSeverity/DriftStatus/ErrorSeverity)؛ با CPython≥3.12 هم‌رفتار و مستقل از `getattr(builtins)` | `tests/unit/test_model_serialization_and_unpickler.py` — ۱۴ تست (pickle-by-value در پروتکل‌های 0/2/4/5، round-trip sklearn→joblib→restricted loader، رد payloadهای `os.system`/`builtins.getattr`/ماژول ناشناخته) |
| P0-3 | BOLA (OWASP API1:2023): نشست = هدر خام `X-Session-ID` | `x_session_id` دیگر کلید سراسری نیست؛ کلید نشست `ws:<sha256(tenant\x00subject)[:16]>:<raw>` است؛ principal از contextvar درخواستی (ست‌شده در dependency async `get_current_principal`) خوانده می‌شود؛ مسیرهای جا‌افتادهٔ `s00/s01/s02` (که خام کار می‌کردند) هم به همان `_sid()` متصل شدند؛ هیچ رجیستری سراسری نیست (cross-test pollution ناپدید شد) | `tests/unit/test_bola_session_isolation.py` — ۵ تست ایزولاسیون tenant/subject + تست‌های e2e سبز |
| P0-4 | استودیو ۹: روتر خالی | پیاده‌سازی کامل و offline-first: `/s09/nlp` (۸ تسک)، `/s09/vision` (آپلود ‎≤۱۵MB‎، گارد decompression-bomb ‎۴۰MP‎)، `/s09/graph` (edge-detection + correlation fallback، link prediction، embedding، centrality/community)، `/s09/recommend` (catalog + feature signals)، `/s09/survival` (Kaplan–Meier + CoxPH)، `/s09/backends`؛ تعیین‌گرا (seed=42) | `test_workbench_spa.py`, `test_omniforge_task33/35`, `test_p0_security_and_runtime_hardening.py` |
| P0-5 | استودیو ۴: `HTTPException` ایمپورت نشده (۸×F821) + 500 زنده | یک ایمپورت اضافه شد؛ lint سبز | suite سبز |
| P0-6 | زنجیرهٔ audit فقط زیرمجموعه فیلدها + بدون امضا؛ verify_chain برای رکورد خراب `True` نمی‌داد؟ (برعکس: قابل جعل بود) | **Audit chain v2**: hash کامل payload (همهٔ فیلدهای forensic شامل `role`, `trace_id`, `details`) + `chain_mac` = HMAC-SHA256 دامنه‌جدا‌شده (`omniforge-audit-event-v2`) روی hash؛ حذف کامل مسیر تأیید legacy/bypass؛ `_load_last_hash` fail-closed (رکورد آخر خراب → `SecurityError`)؛ `AuditLogger.verify_file()` برای SIEM/ops؛ رکوردهای بدون MAC دیگر هرگز تأیید نمی‌شوند | تست‌های e2e + تست دستی زنده (clean verify=(True,2)؛ MAC-stripped=(False,0)) |
| P0-7 | CI فقط ۲ فایل تست | `.github/workflows/ci.yml` اکنون کل suite را با پوشش اجرا می‌کند: `pytest -q tests --cov=omniforge --cov-report=term-missing` + stepهای `ruff check src tests` و `mypy src/omniforge` (هر دو اکنون سبز؛ jobs روی py3.12 و 3.13) | اجرای محلی همین دستورات سبز است؛ CI yaml اعتبارسنجی شد |

### الف.۲ P1 و اصلاحات تکمیلی (کشف و رفع در حین remediation)

| # | مورد | اصلاح |
|---|---|---|
| P1-8 / §۵٫۴ | Categorical `fillna("MISSING")` → TypeError در cleaning/fairness/feature-eng/supervisory | ‎`cat.add_categories` قبل از fillna در ۶ سایت (cleaning MICE loop، s07_advanced، engineering ×2، supervisory_monitor) |
| P1-9 / §۵٫۵–۵٫۶ | `timeliness=None` کرش مسیر گزارش؛ alias مزخرف `proxy_accuracy`؛ دو تست متعارض | generator/charts/narrative نسبت به `None` ایمن؛ `proxy_accuracy` حذف شد؛ تست واحد به‌روز شد |
| — | گراف s9 `power = power @ P` با `deg>0` تقسیم → هشدار invalid division | `np.divide(..., where=deg>0, out=0)` |
| — | نشست‌های namespaced در خروجی API اکو می‌شدند (`session_id: ws:tag:raw`) | ‎`_display_session_id()` در `_state_summary` prefix را حذف می‌کند (هیچ فایل تستی به آن وابسته نبود) |
| — | تست‌هایی که state را با کلید خام می‌کاشتند (task34/audit_regressions) بعد از namespacing کور شدند | اکنون explicit principal + `_sid()` پاس می‌کنند |
| — | `.audit/` و `.venv-audit/` untracked وارد repo می‌شدند | به `.gitignore` اضافه شدند |
| **جدید** | سقف اندازهٔ body وجود نداشت — ‎`await file.read()` در ۷ نقطه بدون cap (memory-exhaustion DoS) | `RequestBodySizeLimitMiddleware` خالص-ASGI در `core/security.py`: پیش‌چک `Content-Length` → 413؛ enforcement جریانی برای chunked (BaseException-محور تا exception handlerها آن را نبلعند)؛ پیش‌فرض ۲۵۶MB از `api.max_request_body_bytes` (env: `OMNIFORGE_API__MAX_REQUEST_BODY_BYTES`)؛ تست مستقیم middleware |
| **جدید** | `rate_limit_per_minute: 300` در settings اعلام ولی **هرگز enforce نمی‌شد** (knob مرده) | `SlidingWindowRateLimitMiddleware` (per-client-IP sliding window، ‎429 + `Retry-After`، استثنای `/health`، disable با `0`، sweep برای رشد نامحدود map)؛ تست واحد با ساعت تزریقی؛ در suite غیرفعال (‎`OMNIFORGE_API__RATE_LIMIT_PER_MINUTE=0`‎ در `conftest.py`) |
| **جدید** | آپلود تصویر s09: ‎`img.load()` قبل از بررسی ابعاد → decompression bomb | بررسی `width*height > 40_000_000` قبل از decode؛ سپس thumbnail |
| **جدید** | mypy: ‎۷ خطا (s09 `no-any-return` ×۴ + types `type-arg` + `has-type` router در workbench) | همه رفع شدند → `Success: no issues found in 185 source files`؛ ruff هم پاک (‎۱۳→۰ خطا) |
| P1-11 / جدید | نبود CSP/HSTS روی خود اپ (فقط nosniff/X-Frame-Options/Referrer-Policy) | `Content-Security-Policy-Report-Only` (پیش‌فرض فعال؛ سیاست سخت‌گیرانهٔ `default-src 'self'` + `object-src 'none'` + `frame-ancestors 'self'`، قابل override با `OMNIFORGE_API__CSP_REPORT_ONLY_POLICY` یا غیرفعال‌سازی با `OMNIFORGE_API__CSP_REPORT_ONLY_ENABLED=false`) و `Strict-Transport-Security: max-age=31536000; includeSubDomains` (`OMNIFORGE_API__HSTS_*`) در header middleware؛ آگاهانه Report-Only (rollout گزارش-اول، قانون RFC 6797 §8.1 برای HTTP بی‌اثر است) | `tests/unit/test_audit_anchor_and_browser_headers.py` (هدرها روی /health؛ Report-Only پیش‌فرض؛ pre-existing headers باقی‌اند) |
| Audit anchoring / جدید | زنجیرهٔ chain v2 فقط یکپارچگی in-band را اثبات می‌کرد؛ میزبان با دسترسی نوشتن به JSONL می‌توانست زنجیره را با کلید پیش‌فرض از نو امضا کند | **`AuditCheckpointAnchor` (اختیاری، RFC 9162-style)** در `core/security.py`: با تنظیم `OMNIFORGE_AUDIT__ANCHOR_PATH` به فایل checkpoint جدا (بهتر از بدنهٔ لاگ؛ ایده‌آل: mount جدا/ذخیرهٔ تکراریافته) + `OMNIFORGE_AUDIT__CHECKPOINT_KEY` (اجباری در `OMNIFORGE_ENVIRONMENT=production`؛ در dev کلید 0600 persist می‌شود) هر N رویداد (`OMNIFORGE_AUDIT__ANCHOR_EVERY`، پیش‌فرض ۱) یک checkpoint = `{version, seq, record_count, head_hash, timestamp, mac}` با HMAC-SHA256 دامنه‌جدا (`omniforge-audit-checkpoint-v1`) به فایل anchor اضافه می‌کند؛ `verify_log` (با کلید خود anchor) / `verify_against_log` (صریح/env) / `AuditLogger.verify_anchor` + `verify_with_anchor`؛ خطای نوشتن anchor → `SecurityError` (fail-closed)؛ صداقت محدودیت: حذف **همزمانِ** tail لاگ و tail checkpoint بدون شاهد خارجی (witness) قابل‌تشخیص نیست | ‎۱۳ تست جدید: re-chain بدون کلید checkpoint → رد می‌شود حتی اگر chain in-band سالم باشد؛ MAC خراب checkpoint → `SecurityError` در load؛ rollback tail → ok با پوشش کمتر (`record 1 of 2`)؛ env/production/every semantics |

### الف.۳ نتایج نهایی (این شاخه)

**suite واحد/e2e:** `190 passed / 1 skipped` (py3.11 و py3.12 محلی؛ اسکیپ همان تست فایل نصب‌کنندهٔ Inno Setup) — baseline ممیزی: ‎`11 failed / 134 passed`‎. **integration:** `3 passed`. **ruff:** پاک. **mypy:** پاک (۱۸۵ فایل). **پوشش:** ۷۸٪. **CI (py3.12/3.13):** سبز — run `44946890094` هر دو job موفق (`quality-and-tests (3.12)/(3.13) = success`)؛ قبل از آن، شکست سیستماتیک pytest step ریشه‌یابی و رفع شد: دو تست P3 در `tests/unit/test_audit_regressions.py` مسیر مطلق `/home/user/OmniForge` را hardcode کرده بودند (روی runner با `/home/runner/work/...` می‌شکستند)؛ رفع با `Path(__file__).resolve().parents[2]` — commit `39de084`. همچنین CI workflow: `fail-fast: false`، tee لاگ + `::error` annotations + job summary روی failure + `GITHUB_STEP_SUMMARY` مشروط (commits `55a52ac`, `80bb5ce`, `f890c0a`).

### الف.۴ بازمانده‌ها (تغییرنکرده — صادقانه)

- **P1-10** ~~fingerprint زیرمجموعه‌ای~~ **با این پاس رفع شد**: `compute_df_fingerprint` اکنون full-content blake2b-128 روی همهٔ ردیف‌هاست؛ `DataFrameCache(ttl_seconds=None)` opt-in TTL (پیش‌فرض خاموش، بدون تغییر رفتار موجود)؛ ۹ تست رگرسیون.
- **P1-11** ~~CSP/HSTS~~ **با این پاس رفع شد**: `Content-Security-Policy-Report-Only` + `Strict-Transport-Security` روی خود اپ (گزارش-اول عمدی؛ enforce بعد از پاک‌شدن report traffic قدم بعدی است).
- **P1-12** ~~`platform_admin`~~ **با این پاس رفع شد**: نقش `platform_admin` (superset مجموعه دسترسی‌ها، تنها نقش با bypass از جداسازی tenant)؛ `admin` فقط tenant خودش؛ audit اجباری هر تصمیم cross-tenant (allow/deny، action=`authorize:cross_tenant`) در زنجیرهٔ tamper-evident؛ allow در صورت شکست نوشتن audit → `SecurityError` (fail-closed)؛ API در صورت خراب بودن sink audit اصلاً boot نمی‌شود.
- **P1-13** ~~salt عمومی~~ **با این پاس رفع شد**: حذف `salt="uap-pii-salt"` عمومی؛ salt از env هر-tenant/KMS (`OMNIFORGE_PSEUDONYM_SALT__<TENANT>` یا `OMNIFORGE_PSEUDONYM_SALTS` به‌شکل JSON از KMS/Vault یا `OMNIFORGE_PSEUDONYM_SALT`)؛ HMAC-SHA256 با domain separator `omniforge:pseudonym:v1` و binding به tenant (مقدار یکسان در دو tenant → pseudonym متفاوت)؛ در production بدون salt قوی (≥۳۲ بایت، بدون placeholder) اجرا نمی‌شود (fail-closed)؛ fallback توسعه: keyfile تصادفی 0600 هر-tenant.
- **P1-14** ~~allowlist identifier~~ **با این پاس رفع شد**: نام جدول در مسیرهای fallback (`data/ingestion.py`) از کاتالوگ خود دیتابیس canonical می‌شود (`sqlite_master` / `SHOW TABLES`) و با نقل‌قول ANSI + doubling کوتیت می‌شود (`_quote_sql_identifier`)؛ `query` خام از همان اعتبارسنجی تک-SELECT/WITH کانکتورها می‌گذرد؛ sqlite با URI `mode=ro` و authorizer سطح C باز می‌شود؛ `SecurityError` بدون wrap به caller می‌رسد. تغییر رفتار: `table=` ناشناخته فست‌فیل و `query=` نوشتنی رد می‌شود.
- **Rate limiting** **با این پاس بهبود یافت**: limiter مشترک Redis (اختیاری، `OMNIFORGE_API__RATE_LIMIT_BACKEND=redis` + `OMNIFORGE_API__RATE_LIMIT_REDIS_URL`؛ ZSET+Lua اتمیک، گروهٔ `omniforge[redis]`) برای استقرار چندپردازه‌ای؛ خرابی Redis → fallback به in-memory با لاگ خطا؛ `backend=redis` بدون پکیج/URL → خطای فست‌فیل در بوت؛ پیش‌فرض `auto`. رفتار قبلیِ per-process همچنان در دسترس است و `X-Forwarded-For` طبق طراحی اعتماد نمی‌شود.
- **Audit anchoring** برون‌باندی (RFC 9162-style) **با این پاس (opt-in) پیاده شد**: `AuditCheckpointAnchor` با MAC دامنه‌جدا روی checkpointها؛ محدودیت صادقانه باقی است: حذف همزمان tail لاگ+checkpoint بدون شاهد خارجی/replication قابل‌تشخیص نیست — برای گارانتی witness-grade، فایل anchor را به ذخیرهٔ مستقل replicate کنید (راهنمای کامل عملیاتی + runbook چرخش کلید و پاسخ به حادثه: ‎`docs/AUDIT_ANCHOR_REPLICATION.md`). پیش‌فرض خاموش (zero-config local بی‌تغییر).
- **P2**: همهٔ اقلام Appendix A P2 رفع شدند: بنر «historical» برای `docs/P2_REMEDIATION_REPORT.md` ✅ (از قبل)؛ `MetaLearningBayesianAutoML` → `MetaFeatureWarmStartAutoML` ✅ (alias منقضی)؛ lock-file با hashes ✅ (`requirements.in` → شیم `requirements.txt`، job `lock-verify`)؛ e2e در CI جدا با cache node ✅ (job `e2e`)؛ benchmark بودجهٔ latency ✅ (job `latency-budget`، `scripts/latency_budget_benchmark.py`)؛ risk register NIST AI RMF ✅ (`docs/RISK_REGISTER.md`) + kill-switch CT ✅ (`OMNIFORGE_CT_DISABLED` / `OMNIFORGE_CT_AUTO_PROMOTE=false`)؛ PDF export مدل‌کارد ✅ (`ModelCard.to_pdf`)؛ `requirements.txt` → شیم hash-checked ✅ (T5).
- نکتهٔ اجرایی: verification اولیهٔ این سند روی py3.11 محلی (`.audit/src`) انجام شد؛ پس از رفع P1-10…P1-14 و اقلام P2، suite محلی `265 passed / 1 skipped` روی هر دو py3.12 (`/tmp/venv312`) و py3.11 (`.venv-audit` + `PYTHONPATH=.audit/src`) است. CI روی هر push اجرا می‌شود؛ jobs جدید `lock-verify`، `e2e` و `latency-budget` در همین شاخه اضافه شدند و نتیجهٔ آن‌ها روی آخرین run پیوست A ثبت می‌شود.
