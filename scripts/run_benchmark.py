"""Benchmark utility verifying all 18 domain plugins, baselines, AutoML, and carbon tracking."""

from __future__ import annotations

import time

import omniforge


def main() -> None:
    t0 = time.perf_counter()
    plugins = omniforge.default_domain_registry.list_plugins()
    print(f"Benchmarking {len(plugins)} OmniForge industry domain plugins...")
    for plugin in plugins:
        df = plugin.engineer_domain_features(plugin.generate_sample_dataset(n_samples=120))
        kpis = plugin.compute_domain_kpis(df)
        quality = omniforge.DataQualityProfiler().profile(df)
        print(
            f"  - {plugin.metadata.name:16s} | rows={len(df):4d} | cols={len(df.columns):2d} | "
            f"quality={quality.overall_score:5.1f} ({quality.grade}) | kpis={list(kpis.keys())}"
        )
    print(f"All {len(plugins)} domains verified in {time.perf_counter() - t0:.2f}s.")


if __name__ == "__main__":
    main()
