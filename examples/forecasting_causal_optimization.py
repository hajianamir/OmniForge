"""Decision Intelligence Example: Time-Series Forecasting, Causal Inference, Optimization & Monte Carlo."""

from __future__ import annotations

import omniforge


def main() -> None:
    gen = uap.SyntheticDataGenerator(random_state=42)

    # 1. Probabilistic Forecasting Tournament
    ts_df = gen.generate_time_series_dataset(n_periods=120)
    fc = uap.ForecastingEngine(include_deep=False).forecast(ts_df, target_col="y", time_col="timestamp", horizon=14)
    print(f"[Forecast] Champion: {fc.champion_model} | Leaderboard[0]: {fc.leaderboard[0]}")

    # 2. Causal Effect Estimation with DoWhy-style Refutations
    cdf = gen.generate_causal_dataset(n_samples=300, true_ate=3.5)
    causal = uap.CausalInferenceEngine(bootstrap_rounds=30).estimate_ate(
        cdf, treatment_col="treatment", outcome_col="outcome", confounders=["confounder_1", "confounder_2"], method="aipw"
    )
    print(f"[Causal AIPW] Estimated ATE: {causal.ate:.3f} (95% CI [{causal.ci_lower:.2f}, {causal.ci_upper:.2f}])")

    # 3. Portfolio Optimization (Markowitz) & Monte Carlo VaR
    opt = uap.OptimizationEngine().optimize_portfolio(
        expected_returns=[0.08, 0.14, 0.06, 0.11],
        cov_matrix=[
            [0.04, 0.01, 0.005, 0.01],
            [0.01, 0.09, 0.002, 0.02],
            [0.005, 0.002, 0.015, 0.004],
            [0.01, 0.02, 0.004, 0.05],
        ],
    )
    print(f"[Optimization] Portfolio weights: {opt.solution} | Sharpe: {opt.details['sharpe_ratio']}")

    mc_res, _ = uap.MonteCarloSimulator(n_simulations=3000, random_seed=42).run(
        {"annual_return": {"dist": "normal", "mean": opt.details["expected_return"], "std": opt.details["volatility"]}},
        lambda s: 1_000_000.0 * (1.0 + s["annual_return"]),
    )
    print(f"[Monte Carlo] Mean portfolio value: ${mc_res.mean:,.2f} | VaR 5%: ${mc_res.var_95:,.2f}")


if __name__ == "__main__":
    main()
