"""CLI smoke tests: the subcommands run end-to-end in mock mode."""

from __future__ import annotations

from route_arena.cli import main


def test_cli_route_smoke(capsys):
    rc = main(["route", "Write a python function to reverse a linked list."])
    out = capsys.readouterr().out
    assert rc == 0
    assert "cost_optimal" in out
    assert "quality_first" in out
    assert "balanced" in out


def test_cli_route_single_strategy(capsys):
    rc = main(["route", "Summarize this text.", "--strategy", "cost_optimal"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "cost_optimal" in out


def test_cli_bench_smoke(capsys):
    rc = main(["bench"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "leaderboard" in out.lower()
    # every model should appear in the leaderboard
    for model in ("gpt-4o", "claude-sonnet", "llama-3-70b"):
        assert model in out


def test_cli_arena_smoke(capsys):
    rc = main(["arena", "write a python function to reverse a linked list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Verdict" in out
    assert "Side A" in out and "Side B" in out


def test_cli_arena_custom_strategies(capsys):
    rc = main(["arena", "Summarize the report.", "--a", "balanced", "--b", "cost_optimal"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "balanced" in out
