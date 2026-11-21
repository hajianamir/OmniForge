# OmniForge v3 Migration Guide (`uap_*` -> `omniforge_*`)

In OmniForge v2.x, internal Workbench session state keys use the `uap_*` prefix (e.g., `uap_df`, `uap_raw_df`, `uap_split_lock`, `uap_targets`, `uap_excluded`) for backward compatibility.

## Planned v3.0 Key Mapping

| Legacy Key (`v2.x`) | Target Key (`v3.0`) | Purpose |
|---|---|---|
| `uap_df` | `omniforge_active_df` | Active working DataFrame |
| `uap_raw_df` | `omniforge_raw_df` | Immutable raw audit DataFrame |
| `uap_split_lock` | `omniforge_split_lock` | Train/Test/Gap `row_id` contract |
| `uap_targets` | `omniforge_targets` | Selected target columns |
| `uap_excluded` | `omniforge_excluded` | Excluded/role columns |
| `__uap_automl_target__` | `__omniforge_automl_target__` | Internal AutoML target column |
