# گزارش ممیزی جامع آخرین `main` پروژهٔ OmniForge (نسخهٔ 4.0.0)

- **شاخه و کامیت بررسی‌شده:** `main` — SHA `eb8f7954233489edabb0738ab7982a1c8d5a4fe3` (`2026-11-23T16:36:14Z`)
- **محیط آزمون:** کپی دست‌نخورده در `/tmp/omniforge-eb8f795/source`؛ Python 3.12.9 (ساخته‌شده از منبع + ماژول‌های `zlib`, `_sqlite3`, `_ctypes`, `_bz2`, `_lzma`, `_ssl`, `_hashlib`)؛ وابستگی‌های قفل‌شده از `requirements.lock` (`numpy 2.5.5`, `pandas 3.1.1`, `scipy 1.18.1`, `scikit-learn 1.9.1`, `fastapi 0.165.0`, `pydantic 2.14.4`).
- **محدودیت‌ها:** هیچ تغییری در کد پروژه یا checkout فعال (`93a367...`) داده نشده است. ابزار `web_search` برای انطباق با استانداردهای بین‌المللی (scikit-learn، Fairlearn، OWASP API Security Top 10 2023، NIST SP 800-162، Hyndman MASE) استفاده شد.

---

## ۱. خلاصهٔ اجرایی و وضعیت تست کلی

- **تست‌های واحد و یکپارچه (`pytest`):** `359 passed, 1 skipped, 4 warnings in 271.58s` — پوشش کد: **۸۰٪** (۳۰٬۵۲۶ خط، ۶٬۲۰۲ خط پوشش‌داده‌نشده).
- **ساخت فرانت‌اند (`npm ci && npm run build`):** موفق در `3.34s` بدون خطا.
- **جمع‌بندی تطبیق یافته‌های پیشین:** بخش مهمی از باگ‌های آماری و امنیتیِ گزارش‌شده در ممیزی‌های قبلی (مانند فرمول `dCor`، خطای `Friedman` برای ۲ مدل، ماتریس کوواریانس نامعتبر در بهینه‌سازی سبد، روش ناشناخته در استنباط علّی، نشت jobهای کاربران دیگر، فرار مسیر `../../` در registry، هم‌پوشانی holdout در `AutoMLPipelineSearch`، حفظ `target` در پاک‌سازی و تبدیل `datetime`) **اصلاح شده‌اند**. با این حال، **۲۳ مورد باز** در حوزه‌های امنیت (BFLA)، MLOps، جداسازی داده (Split Lock)، امتیازدهی Workbench، عدالت (Fairness)، پیش‌بینی سری زمانی و شبیه‌سازی همچنان در `eb8f795` بازتولید شدند.

---

## ۲. یافته‌های اصلاح‌شده (تأییدشده با اجرای عملی روی `eb8f795`)

| موضوع | نتیجهٔ اجرا روی `eb8f795` | مرجع |
|---|---|---|
| **Distance Correlation (`dcor_scale`)** | `1.0` برای هر دو مقیاس (`x_x` و `tiny_tiny`) | [Székely et al., 2007](https://projecteuclid.org/journals/annals-of-statistics/volume-35/issue-6/Measuring-and-testing-dependence-by-correlation-of-distances/10.1214/009053607000000505.pdf) |
| **آزمون Friedman با ۲ مدل** | خطای صریح `ValueError: At least 3 samples must be given for Friedman test, got 2.` | [SciPy `friedmanchisquare`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html) |
| **کوواریانس نامعتبر در سبد سهام** | خطای صریح `OptimizationError: Covariance matrix is not positive semi-definite.` | — |
| **روش ناشناخته در Causal** | خطای صریح `CausalInferenceError: Unsupported causal estimation method 'not-a-method'.` | — |
| **جداسازی فهرست Jobها (`global_jobs`)** | `probe_job_visible: false` (کاربر دیگر job را نمی‌بیند) | [OWASP API1:2023 BOLA](https://owasp.org/API-Security/editions/2023/en/0xa1-broken-object-level-authorization/) |
| **فرار مسیر Registry (`../../escape`)** | خطای صریح `SecurityError: [OmniForge-SEC-001] Invalid model_name` | — |
| **برچسب Canary (`canary_mislabel`)** | `variant: "canary"`, `version: "primary-v1-canary"` | — |
| **هم‌پوشانی Holdout در AutoML بومی** | `cv_holdout_overlap: [0, 0, 0, 0]` | [scikit-learn Common Pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) |
| **حفظ `target` در Cleaning** | `target_after: [NaN, Infinity, 9999.0]` بدون تغییر | — |
| **تبدیل `datetime` خام** | `is_missing: true`, `known: 1577836800.0` (ثانیهٔ دقیق) | — |
| **تشخیص پرت `elliptic`** | `detector_fitted: true` | — |
| **طبقه‌بندی با کلاس غایب** | `accuracy: 1.0`, `balanced_accuracy: 1.0`, `f1_macro: 1.0` | — |
| **Group CV بدون `groups`** | خطای صریح `DataValidationError` | — |

---

## ۳. موارد باز و باقی‌مانده (بازتولیدشده روی `eb8f795`)

### الف) امنیت و کنترل دسترسی (P0)
1. **کنترل نقش در ارتقای مدل (`viewer_can_promote` — بحرانی):** کاربر با نقش `viewer` در JWT همچنان به `POST /api/workbench/s11/promote` پاسخ `200` می‌گیرد و `transition_stage` فراخوانی می‌شود.
   - *انطباق:* نقض [OWASP API5:2023 Broken Function Level Authorization](https://api-security.owasp.org/editions/2023/en/0xa5-broken-function-level-authorization/) و اصل کنترل دسترسی مبتنی بر نقش/ویژگی در [NIST SP 800-162](https://nhimg.org/authentication-vs-authorisation-guide).

### ب) MLOps، رجیستری و صف کار (P0/P1)
2. **عدم رمزگشایی در Registry (`encrypted_registry` — بالا):** `register_model(..., encrypt=True)` نسخه را ثبت می‌کند، اما `load_model` با `ModelSerializationError: Failed to load model artifact` شکست می‌خورد.
3. **ارتقای خودکار Challenger هنگام خطای Champion (`failed_champion_promotes` — بالا):** وقتی بارگذاری champion با `RuntimeError('signature verification failed')` خطا می‌دهد، ارکستراتور همچنان challenger را به `Production` ارتقا می‌دهد (`promoted: true`).
4. **محدودیت `max_history` در `BackgroundJobManager` (`job_history_unbounded` — متوسط تا بالا):** با `max_history=4` همچنان `9` job در حافظه نگه داشته می‌شوند.

### پ) جداسازی داده، نشت و Workbench (P0/P1)
5. **عدم پایداری Split Lock با Index غیراستاندارد (`lock_nonrange`, `lock_overlapping_labels` — بحرانی):** با index غیر RangeIndex خطای `ValueError: Active Split Lock is invalid or corrupted` رخ می‌دهد و با برچسب‌های تکراری index، عضویت train/test جابه‌جا می‌شود.
6. **نادیده‌گرفتن Split Lock در `s06_run_automl` (`automl_boundary` — بحرانی):** با وجود قفل ۳۰/۳۰، همچنان هر ۶۰ سطر به `AutoMLBackendHub.run` ارسال می‌شوند.
7. **امتیازدهی تکی و دسته‌ای مقادیر دسته‌ای را صفر می‌کند (`workbench_scoring` — بحرانی):** در `s06_predict_single` و `s06_predict_batch` مقدار `'blue'` به `0.0` تبدیل می‌شود.
8. **ناهمخوانی فضای انتخاب ویژگی (`feature_selection_commit` — بالا):** پس از انتخاب در فضای dummy، `excluded_raw_columns: []` باقی می‌ماند.
9. **ورود `target` در خوشه‌بندی و ناهنجاری (`cluster`, `explicit_anomaly` — بالا):** `s02_run_clustering` همچنان `target` را در `features_used` می‌آورد و `s02_run_anomaly` با درخواست صریح `features=['x', 'target']` هر دو را به مدل می‌دهد.

### ت) پاک‌سازی، کیفیت داده و Feature Store (P1)
10. **روش `percentile` بدون fit (`clean_percentile` — بالا):** `detector_fitted: false`.
11. **تکثیر سطر در `ffill` با Index تکراری (`ffill_duplicate_index` — بالا):** ورودی ۵ سطری به خروجی ۷ سطری تبدیل می‌شود.
12. **گزارش ناقص `missforest` (`partial_missforest` — بالا):** `b_missing_after: 4` و `report_imputed_columns: {}`.
13. **کش نادرست کیفیت داده (`quality_cache` — بالا):** با تغییر قوانین، `accuracy` همچنان `100.0` برمی‌گردد (به‌جای `33.33`).
14. **مقدار گمشده در `FeatureStore` (`feature_store_ttl`):** خروجی `online` برابر `null` و `historical` برابر `NaN` است.

### ث) عدالت، پیش‌بینی سری زمانی، استنباط علّی و بهینه‌سازی (P1/P2)
15. **Fairness با دادهٔ خالی یا کلاس غایب (`fairness_missing_classes`, `fairness_empty` — بالا):** هر دو `passed: true` برمی‌گردانند.
16. **توصیهٔ متناقض در Fairness (`fairness_dp_only` — بالا):** `passed: false` با نقض Demographic Parity، اما متن `recommendations` می‌گوید مدل همهٔ گیت‌ها را پاس کرده است.
    - *انطباق:* [Fairlearn Metrics](https://fairlearn.org/v0.14/user_guide/assessment/perform_fairness_assessment.html) برای معیارهای تعریف‌نشده یا تک‌گروهی نباید عبور کاذب گزارش کند.
17. **خطای صفر برای مقادیر واقعی گمشده در Backtest (`backtest_missing_actuals` — بالا):** `rmse: 0.0, mae: 0.0, smape: 0.0, mase: 0.0`.
    - *انطباق:* طبق [Hyndman (2006)](https://robjhyndman.com/papers/foresight.pdf)، ارزیابی بدون مشاهدهٔ واقعی معتبر یا با مخرج صفر تعریف‌نشده است.
18. **عدم تطابق `gap` در Rolling CV (`forecast_gap_alignment` — بالا):** با `gap=3` و `horizon=4`، همچنان `requested_horizon=4` فراخوانی می‌شود.
    - *انطباق:* [scikit-learn `TimeSeriesSplit`](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html).
19. **عدم تطابق `Baseline_naive` در Refit (`forecast_baseline_mismatch`):** پیش‌بینی `[3.0, 4.0]` به‌جای `[4.0, 4.0]`.
20. **بازهٔ معکوس در پیش‌بینی (`interval_invalid`):** `winkler_score: 78.0` بدون خطای اعتبارسنجی.
21. **DiD تک‌مشاهده‌ای (`did_singletons` — بالا):** `std_error: 0.0`, `ci_lower: 1.0`, `ci_upper: 1.0`, `p_value: 0.0`.
22. **حذف بی‌صدای وزن در Portfolio (`portfolio_missing_asset_names` — بالا):** با نام دارایی ناقص، فقط یک وزن `0.475` برمی‌گردد.
23. **افق شبیه‌ساز صف (`queue_horizon` — متوسط):** با `horizon_time=1` همچنان `completed_entities: 6` تا تخلیه پردازش می‌شود. (در `cpcv_small_data` نیز به‌جای کرش، `[]` برمی‌گردد.)
