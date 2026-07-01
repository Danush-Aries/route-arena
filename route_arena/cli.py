"""route-arena command line interface.

Three subcommands:

  * route "<prompt>"   - show what each strategy would pick for a prompt.
  * bench              - print the offline quality leaderboard + a routing
                         benchmark across sample prompts and strategies.
  * arena "<prompt>"   - live A/B: route with two strategies, invoke the
                         chosen models, and compare projected vs realized.

Pretty output uses `rich` when it is installed; otherwise a clean plain-text
fallback is used so the tool works with zero optional dependencies.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from route_arena import __version__
from route_arena.arena import ArenaResult, run_arena
from route_arena.eval.prompts import EVAL_SET
from route_arena.models import CATEGORIES, all_models
from route_arena.providers import is_mock_mode
from route_arena.router.policy import STRATEGIES, Router, RoutingDecision

# --------------------------------------------------------------------------- #
# Optional rich rendering
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - presence depends on the environment
    from rich.console import Console
    from rich.markup import escape as _rich_escape
    from rich.table import Table

    _RICH = True
    _console = Console()
except Exception:  # pragma: no cover
    _RICH = False
    _console = None

    def _rich_escape(text: str) -> str:
        return text


def _print(msg: str = "") -> None:
    if _RICH:
        _console.print(msg)
    else:
        # Strip the handful of rich markup tags we use for the plain path.
        for tag in ("[bold]", "[/bold]", "[cyan]", "[/cyan]", "[green]",
                    "[/green]", "[dim]", "[/dim]", "[yellow]", "[/yellow]"):
            msg = msg.replace(tag, "")
        print(msg)


def _mode_banner() -> str:
    return "mock (offline, no API keys)" if is_mock_mode() else "live (API keys detected)"


# --------------------------------------------------------------------------- #
# Table helpers (rich table, or aligned plain text)
# --------------------------------------------------------------------------- #


def _render_table(title: str, columns: List[str], rows: List[List[str]]) -> None:
    if _RICH:
        table = Table(title=title, title_style="bold cyan", header_style="bold")
        for col in columns:
            table.add_column(col)
        for row in rows:
            table.add_row(*row)
        _console.print(table)
        return

    # Plain-text aligned table.
    widths = [len(c) for c in columns]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    line = "  ".join(c.ljust(widths[i]) for i, c in enumerate(columns))
    print(f"\n{title}")
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


# --------------------------------------------------------------------------- #
# route
# --------------------------------------------------------------------------- #


def _print_decision(decision: RoutingDecision) -> None:
    _print(
        f"  [bold]{decision.strategy:<13}[/bold] -> [green]{decision.chosen:<14}[/green] "
        f"| q={decision.projected_quality:>6.0f} "
        f"| ${decision.projected_cost_usd:.5f} "
        f"| {decision.projected_latency_ms:>5.0f}ms"
    )
    _print(f"      [dim]{decision.rationale}[/dim]")


def cmd_route(args: argparse.Namespace) -> int:
    router = Router()
    prompt = args.prompt
    category = router.route(prompt, strategy="balanced").category

    _print(f"[bold]Prompt:[/bold] {prompt}")
    _print(f"[bold]Category:[/bold] [cyan]{category}[/cyan]   [dim]mode: {_mode_banner()}[/dim]\n")

    strategies = [args.strategy] if args.strategy else list(STRATEGIES)
    for strategy in strategies:
        _print_decision(router.route(prompt, strategy=strategy))

    # Show the Pareto frontier for context.
    frontier = router.route(prompt, strategy="balanced").frontier
    _print(f"\n[dim]Pareto frontier (non-dominated): {', '.join(frontier)}[/dim]")
    return 0


# --------------------------------------------------------------------------- #
# bench
# --------------------------------------------------------------------------- #


def cmd_bench(args: argparse.Namespace) -> int:
    router = Router()

    # 1) Quality leaderboard from the offline eval (results.json).
    rows: List[List[str]] = []
    for spec in all_models():
        cells = [spec.name]
        cats = [router.quality_for(spec.name, c) for c in CATEGORIES]
        cells += [f"{q:.0f}" for q in cats]
        cells.append(f"{sum(cats) / len(cats):.0f}")
        cells.append(f"${spec.input_price:.2f}/${spec.output_price:.2f}")
        rows.append(cells)
    rows.sort(key=lambda r: float(r[-2]), reverse=True)
    _render_table(
        "Quality leaderboard (ELO by category, from offline eval)",
        ["model", *CATEGORIES, "avg", "in/out $/M"],
        rows,
    )

    # 2) Routing benchmark: what each strategy picks across sample prompts.
    sample_prompts = _bench_prompts()
    brows: List[List[str]] = []
    for prompt in sample_prompts:
        category = router.route(prompt, strategy="balanced").category
        cells = [_truncate(prompt, 34), category]
        for strategy in STRATEGIES:
            d = router.route(prompt, strategy=strategy)
            cells.append(f"{d.chosen} (${d.projected_cost_usd:.4f})")
        brows.append(cells)
    _render_table(
        "Routing benchmark (chosen model + projected cost per strategy)",
        ["prompt", "category", *STRATEGIES],
        brows,
    )

    # 3) Aggregate cost comparison across strategies.
    totals = {s: 0.0 for s in STRATEGIES}
    for prompt in sample_prompts:
        for strategy in STRATEGIES:
            totals[strategy] += router.route(prompt, strategy=strategy).projected_cost_usd
    trows = [[s, f"${totals[s]:.5f}"] for s in STRATEGIES]
    baseline = totals["quality_first"] or 1.0
    for row, s in zip(trows, STRATEGIES):
        row.append(f"{(1 - totals[s] / baseline) * 100:+.0f}% vs quality_first")
    _render_table(
        f"Total projected cost over {len(sample_prompts)} prompts",
        ["strategy", "total cost", "savings"],
        trows,
    )
    _print(f"\n[dim]mode: {_mode_banner()}[/dim]")
    return 0


def _bench_prompts() -> List[str]:
    """One representative prompt per category from the eval set."""
    picked: List[str] = []
    seen: set = set()
    for item in EVAL_SET:
        if item.category not in seen:
            seen.add(item.category)
            picked.append(item.prompt)
    return picked


# --------------------------------------------------------------------------- #
# arena
# --------------------------------------------------------------------------- #


def _print_side(label: str, result: ArenaResult) -> None:
    side = result.sides[label]
    d = side.decision
    rq = "n/a" if side.realized_quality is None else f"{side.realized_quality:.2f}"
    _print(
        f"  [bold]Side {label}[/bold] [cyan]{d.strategy}[/cyan] -> "
        f"[green]{d.chosen}[/green]"
    )
    _print(
        f"    projected: q={d.projected_quality:.0f}  ${d.projected_cost_usd:.5f}  "
        f"{d.projected_latency_ms:.0f}ms   [dim]realized: {side.completion.latency_ms:.0f}ms  "
        f"score={rq}[/dim]"
    )
    preview = side.completion.text.strip().splitlines()
    head = preview[0] if preview else ""
    _print(f"    [dim]response: {_rich_escape(_truncate(head, 70))}[/dim]")


def cmd_arena(args: argparse.Namespace) -> int:
    result = run_arena(args.prompt, strategy_a=args.a, strategy_b=args.b)
    _print(f"[bold]Arena[/bold]  [dim]mode: {result.mode}[/dim]")
    _print(f"[bold]Prompt:[/bold] {result.prompt}")
    _print(f"[bold]Category:[/bold] [cyan]{result.category}[/cyan]\n")
    _print_side("A", result)
    _print("")
    _print_side("B", result)
    _print(f"\n[bold yellow]Verdict:[/bold yellow] {result.verdict}"
           if _RICH else f"\nVerdict: {result.verdict}")
    return 0


# --------------------------------------------------------------------------- #
# misc
# --------------------------------------------------------------------------- #


def _truncate(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="route-arena",
        description="A cost/quality/latency LLM router with an offline eval and a live A/B arena.",
    )
    parser.add_argument("--version", action="version", version=f"route-arena {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_route = sub.add_parser("route", help="route a prompt and show each strategy's pick")
    p_route.add_argument("prompt", help="the prompt to route")
    p_route.add_argument(
        "-s", "--strategy", choices=sorted(STRATEGIES),
        help="only show this strategy (default: all three)",
    )
    p_route.set_defaults(func=cmd_route)

    p_bench = sub.add_parser("bench", help="print the quality leaderboard and routing benchmark")
    p_bench.set_defaults(func=cmd_bench)

    p_arena = sub.add_parser("arena", help="live A/B two strategies on one prompt")
    p_arena.add_argument("prompt", help="the prompt to test")
    p_arena.add_argument("--a", default="balanced", choices=sorted(STRATEGIES),
                         help="strategy for side A (default: balanced)")
    p_arena.add_argument("--b", default="quality_first", choices=sorted(STRATEGIES),
                         help="strategy for side B (default: quality_first)")
    p_arena.set_defaults(func=cmd_arena)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
