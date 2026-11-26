# Workbench evaluation access and evidence

Workbench requests authenticate and resolve a tenant- and principal-scoped session before reading or changing its state. Cookie requests that mutate state also pass CSRF validation. Action-specific rules apply in addition to session access.

| Operation | Required permission | Scope and side effects |
| --- | --- | --- |
| Workspace archive export | `report:generate` | Current session only. The signed archive can contain fitted models and records. |
| Workspace restore or clear | `data:write` | Current session only. Restore validates the archive and replaces the session dataset and derived state. |
| Studio 03 diagnostic preview | `data:read` | Current study. A diagnostic-only selection preserves the existing fitted baseline catalog. |
| Studio 03 baseline fitting and Studio 14 quick-train | `model:train` | Publish fitted entries only to the current session. Quick-train rejects a changed study scope at publication. |
| Studio 06 scoring | `model:read` | Current session’s fitted model. Batch scoring attaches the current result to that session. |
| Studio 05 preview, commit | `data:read`, plus `data:write` for commit | A preview caches analysis only. Commit changes the active study. |
| Studio 09 twin control | `mlops:write` | May update the session checkpoint. |
| Studio 11 drift and Studios 11/14 cohort diagnostics | `mlops:read` | Read current study or named session cohorts and cache the diagnostic in the same session. |
| Studio 14 evaluation from file, SQL or Access | `data:write` | Bounded uploads or guarded connectors. Results update the session telemetry. |
| Studio 14 simulation or single reading | `mlops:write` | Update the session’s latest evaluation and history. |
| Studio 14 result export and chart export | `report:generate` | Stream a file computed from the current session. |
| Copilot demo generation and Split Lock tool | `data:write` when called via HTTP | Change only the current Copilot memory. SDK callers use their own local memory. |
| Theme and job cancellation | Session owner | Theme changes a UI preference. Cancellation applies to jobs owned by the calling session. |

## Evaluation behavior

- Chart inference applies the specific fit’s preprocessor. An active champion holdout must match the fit, outcome, ordered row index and saved digest. Legacy workspace archives can require retraining to establish that link.
- Supervisory evaluation uses the chosen model’s outcome. Explicit comparison models must use the same outcome and task. A missing baseline leaves comparative metrics unavailable, while primary-model metrics can still use valid observations. Comparisons use the rows with valid observations and predictions from both models.
- A residual chart may show observed residuals without control limits. The control limits require held-out rows for the same fit and a finite, nonzero reference variation. Incoming data does not set the control limits. A missing or invalid observation cannot count toward tolerance compliance.
- Cohort diagnostics require a reference and a distinct current cohort. Defaults use the active training partition when a Split Lock is present and the input columns shared by the latest supervisory evaluation and active study. Named Copilot datasets are resolved within the current session. Drift, prediction differences and feature attributions are observational.
- PostgreSQL libpq connectors retain the requested hostname for TLS verification and use a validated address for the connection. MySQL and MariaDB read-only connectors request a read-only transaction default. Use restricted database accounts and verify driver behavior and network controls in the deployment environment.
