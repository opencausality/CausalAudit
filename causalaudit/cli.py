"""CausalAudit CLI — Typer-based command-line interface.

Commands:
    audit      Run a full causal drift audit (baseline vs current CSV)
    baseline   Build and save a baseline causal graph from historical data
    explain    Generate an LLM narrative explanation for a saved report
    providers  List available LLM providers and status
    serve      Run the optional REST API server
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from causalaudit import __version__

app = typer.Typer(
    name="causalaudit",
    help="Causal drift detection for deployed ML models.",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()
logger = logging.getLogger("causalaudit.cli")

# ── Callbacks ─────────────────────────────────────────────────────────────────


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"[bold cyan]CausalAudit[/] v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """CausalAudit — detect causal drift in deployed ML models."""
    from causalaudit.config import configure_logging, get_settings

    settings = get_settings()
    if verbose:
        settings.log_level = "DEBUG"
    configure_logging(settings)


# ── audit ─────────────────────────────────────────────────────────────────────


@app.command()
def audit(
    baseline_path: Path = typer.Option(
        ...,
        "--baseline",
        "-b",
        help="Path to the baseline inference log CSV.",
        exists=True,
        readable=True,
    ),
    current_path: Path = typer.Option(
        ...,
        "--current",
        "-c",
        help="Path to the current inference log CSV.",
        exists=True,
        readable=True,
    ),
    target: str = typer.Option(
        ...,
        "--target",
        "-t",
        help="Name of the label / outcome column.",
    ),
    explain: bool = typer.Option(
        False,
        "--explain",
        "-e",
        help="Use LLM to generate a narrative explanation of detected breaks.",
    ),
    show: bool = typer.Option(
        False,
        "--show",
        "-s",
        help="Open an interactive graph visualisation after the audit.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Save the full DriftReport as JSON to this path.",
    ),
    significance: float = typer.Option(
        0.05,
        "--significance",
        help="P-value significance threshold for edge detection (0.0–1.0).",
        min=0.0,
        max=1.0,
    ),
    threshold: float = typer.Option(
        0.5,
        "--threshold",
        help="Minimum edge strength to include (0.0–1.0).",
        min=0.0,
        max=1.0,
    ),
) -> None:
    """Run a full causal drift audit comparing baseline and current inference logs.

    Detects structural changes in causal relationships between features and
    the target label, producing a DriftReport with severity scores and
    actionable recommendations.
    """
    from causalaudit.attribution.engine import attribute_degradation
    from causalaudit.config import get_settings
    from causalaudit.data.loader import load_inference_logs
    from causalaudit.data.schema import DriftReport
    from causalaudit.drift.classifier import classify_root_cause, compute_degradation_score
    from causalaudit.drift.detector import detect_drift
    from causalaudit.exceptions import CausalAuditError
    from causalaudit.monitoring.baseline import build_baseline
    from causalaudit.monitoring.tracker import build_current_graph

    settings = get_settings()
    settings.significance_level = significance
    settings.confidence_threshold = threshold

    # ── Load data ──────────────────────────────────────────────────────────
    console.print(f"[dim]Loading baseline from {baseline_path}...[/]")
    try:
        baseline_log = load_inference_logs(baseline_path, target)
    except CausalAuditError as exc:
        console.print(f"[red]Failed to load baseline:[/] {exc}")
        raise typer.Exit(code=1)

    console.print(f"[dim]Loading current data from {current_path}...[/]")
    try:
        current_log = load_inference_logs(current_path, target)
    except CausalAuditError as exc:
        console.print(f"[red]Failed to load current data:[/] {exc}")
        raise typer.Exit(code=1)

    console.print(
        f"[dim]Baseline: {baseline_log.n_rows} rows | "
        f"Current: {current_log.n_rows} rows | "
        f"Target: '{target}'[/]"
    )

    # ── Build graphs ───────────────────────────────────────────────────────
    console.print("\n[cyan]Building baseline causal graph...[/]")
    baseline_graph = build_baseline(baseline_log, settings)

    console.print("[cyan]Building current causal graph...[/]")
    current_graph = build_current_graph(current_log, settings)

    # ── Detect drift ───────────────────────────────────────────────────────
    console.print("[cyan]Detecting causal drift...[/]")
    breaks = detect_drift(baseline_graph, current_graph, settings)
    degradation_score = compute_degradation_score(breaks, baseline_graph)
    root_cause = classify_root_cause(breaks, baseline_graph)

    # ── LLM explanation ────────────────────────────────────────────────────
    model_used = ""
    if explain and breaks:
        console.print("[cyan]Generating LLM narrative explanation...[/]")
        try:
            from causalaudit.llm.adapter import LLMAdapter
            from causalaudit.llm.prompts import DRIFT_EXPLANATION_PROMPT, SYSTEM_PROMPT

            adapter = LLMAdapter(settings)
            breaks_desc = "\n".join(
                f"- {b.break_type}: '{b.cause}' → '{b.effect}' "
                f"(significance={b.significance:.2f}, "
                f"baseline={b.baseline_strength}, current={b.current_strength})"
                for b in breaks[:10]
            )
            prompt = DRIFT_EXPLANATION_PROMPT.format(
                breaks_description=breaks_desc,
                target=target,
                baseline_period=baseline_graph.timestamp,
                current_period=current_graph.timestamp,
            )
            llm_response = adapter.complete(prompt, system=SYSTEM_PROMPT)
            model_used = adapter.provider_info()["model"]
            console.print(
                Panel(llm_response, title="[bold cyan]LLM Explanation[/]", border_style="cyan")
            )
        except Exception as exc:
            console.print(f"[yellow]LLM explanation unavailable:[/] {exc}")

    # ── Build full report ──────────────────────────────────────────────────
    draft_report = DriftReport(
        baseline_graph=baseline_graph,
        current_graph=current_graph,
        breaks=breaks,
        degradation_score=degradation_score,
        root_cause=root_cause,
        recommendations=[],
        model_used=model_used,
    )
    recommendations = attribute_degradation(draft_report)
    report = DriftReport(
        baseline_graph=baseline_graph,
        current_graph=current_graph,
        breaks=breaks,
        degradation_score=degradation_score,
        root_cause=root_cause,
        recommendations=recommendations,
        model_used=model_used,
    )

    # ── Display results ────────────────────────────────────────────────────
    _display_audit_results(report)

    # ── Save output ────────────────────────────────────────────────────────
    if output:
        output.write_text(
            json.dumps(report.model_dump(), indent=2), encoding="utf-8"
        )
        console.print(f"\n[green]Report saved[/] → [bold]{output}[/]")

    # ── Visualise ──────────────────────────────────────────────────────────
    if show:
        from causalaudit.graph.visualizer import render_graph

        vis_path = render_graph(current_graph, "causalaudit_graph.html", breaks=breaks)
        console.print(f"[green]Graph visualisation[/] → [bold]{vis_path}[/]")
        typer.launch(str(vis_path))

    # Exit non-zero if degradation is severe
    if degradation_score >= 0.6:
        raise typer.Exit(code=2)


# ── baseline ──────────────────────────────────────────────────────────────────


@app.command()
def baseline(
    data: Path = typer.Option(
        ...,
        "--data",
        "-d",
        help="Path to historical inference log CSV.",
        exists=True,
        readable=True,
    ),
    target: str = typer.Option(
        ...,
        "--target",
        "-t",
        help="Name of the label / outcome column.",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Where to save the baseline graph JSON.",
    ),
    significance: float = typer.Option(
        0.05,
        "--significance",
        help="P-value threshold for edge significance.",
        min=0.0,
        max=1.0,
    ),
) -> None:
    """Build and save a baseline causal graph from historical inference logs.

    The saved baseline can later be loaded by the ``audit`` command for
    incremental drift monitoring without re-processing historical data.
    """
    from causalaudit.config import get_settings
    from causalaudit.data.loader import load_inference_logs
    from causalaudit.exceptions import CausalAuditError
    from causalaudit.monitoring.baseline import build_baseline, save_baseline

    settings = get_settings()
    settings.significance_level = significance

    console.print(f"[dim]Loading data from {data}...[/]")
    try:
        log = load_inference_logs(data, target)
    except CausalAuditError as exc:
        console.print(f"[red]Failed to load data:[/] {exc}")
        raise typer.Exit(code=1)

    console.print(
        f"[dim]Loaded {log.n_rows} rows, {len(log.feature_columns)} features, "
        f"target='{target}'[/]"
    )
    console.print("[cyan]Discovering baseline causal structure...[/]")

    graph = build_baseline(log, settings)
    save_baseline(graph, output)

    _display_graph_summary(graph, title="Baseline Causal Graph")
    console.print(f"\n[green]Baseline saved[/] → [bold]{output}[/]")


# ── explain ───────────────────────────────────────────────────────────────────


@app.command()
def explain(
    report_path: Path = typer.Argument(
        ...,
        help="Path to a JSON DriftReport file produced by ``audit --output``.",
        exists=True,
    ),
) -> None:
    """Generate an LLM-powered narrative explanation for a saved drift report.

    Reads a ``DriftReport`` JSON file and sends the detected breaks to the
    configured LLM for a plain-English reliability analysis.
    """
    from causalaudit.data.schema import DriftReport
    from causalaudit.exceptions import ProviderError
    from causalaudit.llm.adapter import LLMAdapter
    from causalaudit.llm.prompts import DRIFT_EXPLANATION_PROMPT, SYSTEM_PROMPT

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        report = DriftReport.model_validate(payload)
    except Exception as exc:
        console.print(f"[red]Failed to read report:[/] {exc}")
        raise typer.Exit(code=1)

    if not report.breaks:
        console.print(
            Panel(
                "No causal breaks in this report — the model's causal structure is "
                "consistent with the baseline.",
                title="[green]No Drift Detected[/]",
                border_style="green",
            )
        )
        return

    console.print(f"[dim]Explaining {len(report.breaks)} causal breaks...[/]")

    breaks_desc = "\n".join(
        f"- {b.break_type}: '{b.cause}' → '{b.effect}' "
        f"(significance={b.significance:.2f}, "
        f"baseline_strength={b.baseline_strength}, "
        f"current_strength={b.current_strength})\n"
        f"  {b.explanation}"
        for b in report.breaks[:10]
    )

    prompt = DRIFT_EXPLANATION_PROMPT.format(
        breaks_description=breaks_desc,
        target=report.baseline_graph.nodes[-1] if report.baseline_graph.nodes else "target",
        baseline_period=report.baseline_graph.timestamp,
        current_period=report.current_graph.timestamp,
    )

    try:
        adapter = LLMAdapter()
        response = adapter.complete(prompt, system=SYSTEM_PROMPT)
        console.print(
            Panel(
                response,
                title=f"[bold cyan]LLM Explanation[/] (model: {adapter.provider_info()['model']})",
                border_style="cyan",
            )
        )
    except ProviderError as exc:
        console.print(
            f"[red]LLM provider unavailable:[/] {exc}\n"
            "[dim]Check ``causalaudit providers`` for connection status.[/]"
        )
        raise typer.Exit(code=1)


# ── providers ─────────────────────────────────────────────────────────────────


@app.command()
def providers() -> None:
    """List available LLM providers and their connection status."""
    from causalaudit.config import LLMProvider, get_settings

    settings = get_settings()

    table = Table(title="CausalAudit — LLM Providers", border_style="cyan")
    table.add_column("Provider", style="bold")
    table.add_column("Active", justify="center")
    table.add_column("Model")
    table.add_column("Status")

    for provider in LLMProvider:
        is_active = provider == settings.llm_provider
        marker = "[green]●[/]" if is_active else "[dim]○[/]"
        model = settings.resolved_model if is_active else "—"
        status = _check_provider_status(provider) if is_active else "[dim]not configured[/]"
        table.add_row(provider.value, marker, model, status)

    console.print(table)
    console.print(
        "\n[dim]Set CAUSALAUDIT_LLM_PROVIDER and model via .env or environment variables.[/]"
    )


# ── serve ─────────────────────────────────────────────────────────────────────


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Listen address."),
    port: int = typer.Option(8000, "--port", "-p", help="Listen port."),
) -> None:
    """Run the CausalAudit REST API server."""
    import uvicorn

    console.print(
        f"[bold cyan]CausalAudit API[/] starting on [bold]http://{host}:{port}[/]\n"
        f"[dim]Docs at http://{host}:{port}/docs[/]"
    )
    uvicorn.run(
        "causalaudit.api.server:create_app",
        host=host,
        port=port,
        factory=True,
        reload=False,
    )


# ── Helper functions ──────────────────────────────────────────────────────────


def _display_audit_results(report: "object") -> None:
    """Render audit results to the console in a structured way."""
    from causalaudit.data.schema import DriftReport

    assert isinstance(report, DriftReport)

    # Degradation score panel
    score = report.degradation_score
    if score >= 0.6:
        score_colour = "red"
        score_label = "CRITICAL"
    elif score >= 0.3:
        score_colour = "yellow"
        score_label = "MODERATE"
    else:
        score_colour = "green"
        score_label = "LOW"

    console.print(
        Panel(
            f"Degradation Score: [{score_colour}]{score:.3f}[/] — [{score_colour}]{score_label}[/]\n\n"
            f"{report.root_cause}",
            title="[bold]Causal Drift Report[/]",
            border_style=score_colour,
        )
    )

    # Graph summary
    console.print(
        f"\n[bold]Graph Summary:[/]\n"
        f"  Baseline: [cyan]{len(report.baseline_graph.edges)}[/] edges, "
        f"[cyan]{len(report.baseline_graph.nodes)}[/] nodes\n"
        f"  Current:  [cyan]{len(report.current_graph.edges)}[/] edges, "
        f"[cyan]{len(report.current_graph.nodes)}[/] nodes"
    )

    # Breaks table
    if report.breaks:
        table = Table(title="Detected Causal Breaks", border_style="yellow")
        table.add_column("Type", style="bold")
        table.add_column("Cause", style="yellow")
        table.add_column("→", justify="center")
        table.add_column("Effect", style="magenta")
        table.add_column("Baseline", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("Significance", justify="right")

        type_colours = {
            "PROXY_COLLAPSE": "red",
            "REMOVED_EDGE": "orange3",
            "WEAKENED": "yellow",
            "STRENGTHENED": "green",
            "NEW_EDGE": "cyan",
        }

        for b in report.breaks:
            colour = type_colours.get(b.break_type, "white")
            baseline_str = f"{b.baseline_strength:.3f}" if b.baseline_strength is not None else "—"
            current_str = f"{b.current_strength:.3f}" if b.current_strength is not None else "—"
            sig_colour = "red" if b.significance >= 0.7 else ("yellow" if b.significance >= 0.3 else "green")
            table.add_row(
                f"[{colour}]{b.break_type}[/]",
                b.cause,
                "→",
                b.effect,
                baseline_str,
                current_str,
                f"[{sig_colour}]{b.significance:.3f}[/]",
            )

        console.print(table)
    else:
        console.print("\n[green]No causal breaks detected.[/] Causal structure is stable.")

    # Recommendations
    if report.recommendations:
        console.print("\n[bold]Recommendations:[/]")
        for i, rec in enumerate(report.recommendations, 1):
            rec_colour = "red" if "CRITICAL" in rec or "RETRAIN" in rec else "white"
            console.print(f"  [{rec_colour}]{i}. {rec[:120]}{'...' if len(rec) > 120 else ''}[/]")


def _display_graph_summary(graph: "object", title: str = "Causal Graph") -> None:
    """Display a compact summary table for a single graph."""
    from causalaudit.data.schema import CausalGraph

    assert isinstance(graph, CausalGraph)

    table = Table(title=title, border_style="cyan")
    table.add_column("Cause", style="bold yellow")
    table.add_column("→", justify="center")
    table.add_column("Effect", style="bold magenta")
    table.add_column("Strength", justify="right")
    table.add_column("P-Value", justify="right")
    table.add_column("Significant", justify="center")

    for edge in sorted(graph.edges, key=lambda e: e.strength, reverse=True):
        strength_colour = (
            "green" if edge.strength >= 0.6 else ("yellow" if edge.strength >= 0.3 else "red")
        )
        sig_marker = "[green]✓[/]" if edge.is_significant else "[red]✗[/]"
        table.add_row(
            edge.cause,
            "→",
            edge.effect,
            f"[{strength_colour}]{edge.strength:.3f}[/]",
            f"{edge.p_value:.4f}",
            sig_marker,
        )

    console.print(table)
    console.print(
        f"  [dim]{len(graph.nodes)} nodes, {len(graph.edges)} edges "
        f"({sum(1 for e in graph.edges if e.is_significant)} significant)[/]"
    )


def _check_provider_status(provider: "object") -> str:
    """Quick check whether a provider is reachable."""
    from causalaudit.config import LLMProvider

    assert isinstance(provider, LLMProvider)

    if provider == LLMProvider.OLLAMA:
        try:
            import ollama

            ollama.list()
            return "[green]connected[/]"
        except Exception:
            return "[red]not running — install from ollama.com[/]"

    return "[green]configured[/]"
