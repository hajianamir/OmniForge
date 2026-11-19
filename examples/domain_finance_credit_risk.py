"""Finance Industry Accelerator Example: Credit Default Risk, Basel KPIs, Validation & AutoML."""

from __future__ import annotations

import omniforge


def main() -> None:
    plugin = omniforge.default_domain_registry.get("finance")
    raw_df = plugin.generate_sample_dataset(n_samples=350, random_state=7)
    feat_df = plugin.engineer_domain_features(raw_df)

    val_report = omniforge.DataValidator(plugin.get_validation_rules()).validate(feat_df)
    print(f"Validation passed: {val_report.passed} ({val_report.passed_count}/{val_report.total_expectations} rules)")

    X = feat_df.drop(columns=[plugin.metadata.default_target])
    y = feat_df[plugin.metadata.default_target]
    res = omniforge.AutoMLPipelineSearch(time_budget_seconds=20.0, tune_trials=3).fit(X, y)
    probs = res.best_model.predict_proba(X)[:, 1]
    kpis = plugin.compute_domain_kpis(feat_df, predictions=probs)
    print("Champion model:", res.best_model_name, "Score:", res.best_score)
    print("Finance Domain KPIs:", kpis)


if __name__ == "__main__":
    main()
