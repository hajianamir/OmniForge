"""Enterprise Typer CLI for OmniForge (`omniforge` console entry point)."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from omniforge.version import get_version_info

cli = typer.Typer(
    name="omniforge",
    help="OmniForge Enterprise CLI — From Raw Data to Deployed Decision Intelligence.",
    no_args_is_help=True,
)
console = Console()


@cli.command("version")
def cmd_version() -> None:
    """Display installed OmniForge version and edition."""
    info = get_version_info()
    console.print(Panel.fit(f"[bold cyan]{info['name']}[/bold cyan]\nVersion: {info['version']} ({info['edition']})"))


@cli.command("domains")
def cmd_domains(
    export_sample: str | None = typer.Option(None, "--sample", help="Generate sample CSV for a domain slug."),
    out: Path = typer.Option(Path("domain_sample.csv"), "--out", help="Destination CSV path for sample."),
    rows: int = typer.Option(250, "--rows", help="Number of sample rows to generate."),
) -> None:
    """List all 18 enterprise industry domain plugins or generate a domain sample dataset."""
    from omniforge.domains import default_domain_registry

    if export_sample:
        if rows < 10:
            console.print("[red]Error: --rows must be at least 10.[/red]")
            raise typer.Exit(code=1)
        try:
            plugin = default_domain_registry.get(export_sample)
        except Exception as exc:
            console.print(f"[red]Unknown domain plugin '{export_sample}': {exc}[/red]")
            raise typer.Exit(code=1) from exc
        from omniforge.core.utils import sanitize_spreadsheet_frame

        df = plugin.generate_sample_dataset(n_samples=rows)
        out.parent.mkdir(parents=True, exist_ok=True)
        sanitize_spreadsheet_frame(df).to_csv(out, index=False)
        console.print(f"[green]Generated {len(df)} rows for domain '{export_sample}' -> {out}[/green]")
        return

    table = Table(title="OmniForge — 18 Industry Domain Plugins")
    table.add_column("Slug", style="cyan")
    table.add_column("Industry Vertical", style="bold")
    table.add_column("Default Target", style="magenta")
    table.add_column("Compliance Frameworks", style="green")
    for p in default_domain_registry.list_plugins():
        table.add_row(
            p.metadata.name,
            p.metadata.display_name,
            p.metadata.default_target,
            ", ".join(p.metadata.compliance_frameworks),
        )
    console.print(table)


@cli.command("quality")
def cmd_quality(source: str = typer.Argument(..., help="Input dataset file path or URI.")) -> None:
    """Profile 6-dimensional enterprise data quality score for any supported dataset."""
    from omniforge.data.ingestion import DataIngestionEngine
    from omniforge.data.quality import DataQualityProfiler

    ing = DataIngestionEngine().ingest(source)
    rep = DataQualityProfiler().profile(ing.data)
    console.print(
        Panel.fit(
            f"[bold]Overall Quality Score:[/bold] [cyan]{rep.overall_score}/100 (Grade {rep.grade})[/cyan]\n"
            f"Dimensions: {json.dumps(rep.dimensions, indent=2)}\n"
            f"Recommendations: {'; '.join(rep.recommendations)}"
        )
    )


@cli.command("clean")
def cmd_clean(
    source: str = typer.Argument(..., help="Raw dataset path."),
    out: Path = typer.Option(Path("cleaned.csv"), "--out", help="Output path for cleaned dataset."),
) -> None:
    """Clean missing values, duplicates, strings, and outliers and save cleaned data."""
    from omniforge.core.utils import sanitize_spreadsheet_frame
    from omniforge.data.cleaning import DataCleaner
    from omniforge.data.ingestion import DataIngestionEngine

    ing = DataIngestionEngine().ingest(source)
    cleaned, rep = DataCleaner().fit_transform(ing.data)
    out.parent.mkdir(parents=True, exist_ok=True)
    sanitize_spreadsheet_frame(cleaned).to_csv(out, index=False)
    console.print(
        f"[green]Cleaned dataset saved to {out}[/green] | Rows: {rep.input_shape[0]} -> {rep.output_shape[0]}"
    )


@cli.command("eda")
def cmd_eda(
    source: str = typer.Argument(..., help="Input dataset path."),
    target: str | None = typer.Option(None, "--target", "-t", help="Optional target column."),
    html_out: Path = typer.Option(Path("omniforge_eda_report.html"), "--html", help="Standalone HTML report output path."),
) -> None:
    """Run Automated EDA and export an interactive standalone HTML report."""
    from omniforge.data.ingestion import DataIngestionEngine
    from omniforge.eda.automated_eda import AutomatedEDA

    ing = DataIngestionEngine().ingest(source)
    if target is not None and target not in ing.data.columns:
        console.print(
            f"[red]Target column '{target}' not found in dataset. Available columns: {list(ing.data.columns)}[/red]"
        )
        raise typer.Exit(code=1)
    rep = AutomatedEDA().run(ing.data, target_column=target, generate_figures=True)
    rep.export_html(html_out)
    console.print(f"[green]EDA report written to {html_out}[/green]\n\n{rep.narrative}")


@cli.command("automl")
def cmd_automl(
    source: str = typer.Argument(..., help="Input training dataset path."),
    target: str = typer.Option(..., "--target", "-t", help="Target column name."),
    budget: float = typer.Option(30.0, "--budget", help="Time budget in seconds."),
    save_model: Path | None = typer.Option(None, "--save-model", help="Optional model artifact destination."),
) -> None:
    """Run AutoML with mandatory strong baseline benchmarks, HPO, and carbon tracking."""
    from omniforge.automl.pipeline_search import AutoMLPipelineSearch
    from omniforge.data.ingestion import DataIngestionEngine

    if budget <= 0:
        console.print("[red]Error: --budget must be positive (seconds).[/red]")
        raise typer.Exit(code=1)

    ing = DataIngestionEngine().ingest(source)
    df = ing.data
    if target not in df.columns:
        console.print(
            f"[red]Target column '{target}' not found in dataset. Available columns: {list(df.columns)}[/red]"
        )
        raise typer.Exit(code=1)
    X = df.drop(columns=[target])
    y = df[target]
    res = AutoMLPipelineSearch(time_budget_seconds=budget).fit(X, y)

    table = Table(title=f"OmniForge AutoML Leaderboard ({res.task_type.value})")
    table.add_column("Rank", style="cyan")
    table.add_column("Model", style="bold")
    table.add_column("Kind")
    table.add_column(res.primary_metric, style="green")
    for idx, row in enumerate(res.leaderboard[:10], start=1):
        table.add_row(str(idx), str(row["model"]), str(row["kind"]), str(row["score"]))
    console.print(table)
    lift = res.lift_over_best_baseline
    console.print(f"Lift over best baseline: {lift:+.4f}" if lift is not None else "Lift over best baseline: unavailable")

    if save_model:
        res.best_model.save(save_model)
        console.print(f"[green]Champion model saved to {save_model}[/green]")


@cli.command("drift")
def cmd_drift(
    reference: str = typer.Argument(..., help="Reference dataset path."),
    current: str = typer.Argument(..., help="Current production dataset path."),
) -> None:
    """Evaluate PSI, KS, Wasserstein, and JS distribution drift between two datasets."""
    from omniforge.data.ingestion import DataIngestionEngine
    from omniforge.mlops.drift import DriftDetector

    ref_df = DataIngestionEngine().ingest(reference).data
    cur_df = DataIngestionEngine().ingest(current).data
    rep = DriftDetector().detect(ref_df, cur_df)
    console.print(
        Panel.fit(
            f"Overall Drift Status: [bold]{rep.overall_status.value.upper()}[/bold]\n"
            f"Drifted Features: {rep.drifted_feature_count}/{rep.total_features} ({rep.drifted_feature_ratio*100:.1f}%)\n"
            f"Recommendations: {'; '.join(rep.recommendations)}"
        )
    )


@cli.command("copilot")
def cmd_copilot(
    prompt: str = typer.Argument(..., help="Natural language English instruction for OmniForge Copilot."),
    data: str | None = typer.Option(None, "--data", "-d", help="Optional dataset path to load."),
    provider: str = typer.Option("local", "--provider", help="LLM provider: local|openai|anthropic|azure|ollama."),
) -> None:
    """Send a natural-language instruction to the OmniForge AI Copilot."""
    from omniforge.copilot.agent import OmniForgeCopilotAgent
    from omniforge.data.ingestion import DataIngestionEngine

    agent = OmniForgeCopilotAgent(provider=provider)
    if data:
        agent.load_dataframe(DataIngestionEngine().ingest(data).data, name=Path(data).stem)
    resp = agent.chat(prompt)
    console.print(Panel(resp.reply, title=f"OmniForge AI Copilot ({resp.provider_used})"))


@cli.command("check")
def cmd_check(
    as_json: bool = typer.Option(False, "--json", help="Print diagnostic output as JSON."),
) -> None:
    """Verify Python runtime, required packages, 18 domain plugins, and bundled resources."""
    from omniforge.launcher import _print_check_report, run_environment_check

    report = run_environment_check()
    _print_check_report(report, as_json=as_json)
    if report["status"] != "ok":
        raise typer.Exit(code=1)


@cli.command("desktop")
def cmd_desktop(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host address."),
    api_port: int = typer.Option(8000, "--api-port", help="FastAPI backend port."),
    ui_port: int = typer.Option(8501, "--ui-port", help="FastAPI + React workbench port."),
    no_browser: bool = typer.Option(False, "--no-browser", help="Disable automatic browser launch."),
) -> None:
    """Launch both the FastAPI REST API and OmniForge FastAPI + React Enterprise Workbench together."""
    from omniforge.launcher import launch_full

    launch_full(host=host, api_port=api_port, ui_port=ui_port, open_browser=not no_browser)


@cli.command("monitor")
def cmd_monitor(
    reference: str | None = typer.Option(None, "--reference", help="Reference baseline dataset path."),
    current: str | None = typer.Option(None, "--current", help="Current production batch dataset path."),
    interval: int = typer.Option(0, "--interval", help="Polling interval in seconds (0 = single-shot check)."),
    max_cycles: int = typer.Option(1, "--max-cycles", help="Maximum monitoring cycles when interval > 0."),
) -> None:
    """Run MLOps drift & SLA monitoring check (single-shot or periodic worker loop)."""
    import time

    from omniforge.data.ingestion import DataIngestionEngine
    from omniforge.domains import default_domain_registry
    from omniforge.mlops.drift import DriftDetector

    if (reference is None) != (current is None):
        console.print("[red]Error: Provide both --reference and --current together, or omit both for synthetic demo check.[/red]")
        raise typer.Exit(code=1)

    detector = DriftDetector()
    cycles = max(1, max_cycles) if interval > 0 else 1

    for cycle in range(1, cycles + 1):
        if reference and current:
            ref_df = DataIngestionEngine().ingest(reference).data
            cur_df = DataIngestionEngine().ingest(current).data
        else:
            plugin = default_domain_registry.get("finance")
            ref_df = plugin.generate_sample_dataset(n_samples=150, random_state=42)
            cur_df = plugin.generate_sample_dataset(n_samples=150, random_state=42 + cycle)

        rep = detector.detect(ref_df, cur_df)
        console.print(
            f"[cyan][Cycle {cycle}/{cycles}][/cyan] Drift Status: [bold]{rep.overall_status.value.upper()}[/bold] | "
            f"Drifted Features: {rep.drifted_feature_count}/{rep.total_features}"
        )
        if interval > 0 and cycle < cycles:
            time.sleep(interval)


@cli.command("sbom")
def cmd_sbom(
    out: Path = typer.Option(Path("omniforge.sbom.cdx.json"), "--out", help="Output CycloneDX 1.5 JSON path."),
    tier: str = typer.Option("everything", "--tier", help="Lockfile tier (simple, advanced, everything, dev)."),
    verify_only: bool = typer.Option(False, "--verify", help="Verify SHA-256 hash pinning across all lockfiles."),
) -> None:
    """Generate a CycloneDX 1.5 Software Bill of Materials (SBOM) or verify lockfile SHA-256 provenance."""
    from omniforge.core.sbom import verify_lockfiles_provenance, write_cyclonedx_sbom

    if verify_only:
        rep = verify_lockfiles_provenance()
        status_str = "[green]VERIFIED[/green]" if rep["verified"] else "[red]FAILED[/red]"
        console.print(
            f"Lockfile Provenance: {status_str} | Lockfiles: {rep['lockfile_count']} | "
            f"Unpinned Packages: {len(rep['unpinned_packages'])}"
        )
        if not rep["verified"]:
            raise typer.Exit(code=1)
        return

    written = write_cyclonedx_sbom(out, tier=tier)
    console.print(f"[green]CycloneDX 1.5 SBOM written to {written}[/green]")


@cli.command("serve")
def cmd_serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
) -> None:
    """Start the FastAPI enterprise REST API server (defaults to loopback-only binding)."""
    from omniforge.launcher import launch_api

    launch_api(host=host, port=port)


@cli.command("ui")
def cmd_ui(
    port: int = typer.Option(8501, "--port"),
    host: str = typer.Option("127.0.0.1", "--host"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Disable automatic browser launch."),
) -> None:
    """Launch the OmniForge FastAPI + React Enterprise Analytics Workbench (defaults to loopback-only binding)."""
    from omniforge.launcher import launch_ui

    launch_ui(host=host, port=port, open_browser=not no_browser)


def main() -> None:
    """Console script entry point."""
    cli()


if __name__ == "__main__":
    main()
