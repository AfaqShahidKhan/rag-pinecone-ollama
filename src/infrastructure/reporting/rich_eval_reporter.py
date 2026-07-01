"""
src/infrastructure/reporting/rich_eval_reporter.py

IEvalReporter adapter backed by `rich`. This is the only file responsible
for how evaluation diagnostics/results are rendered to a console.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.domain.entities import EvalResult, PromptPackage, SearchResult
from src.domain.interfaces import IEvalReporter


class RichEvalReporter(IEvalReporter):
    def __init__(self, console: Console) -> None:
        self._console = console

    def report_debug_query(
        self, question: str, results: list[SearchResult], package: PromptPackage
    ) -> None:
        self._console.rule(f"[bold cyan]Debug: {question}[/bold cyan]")

        table = Table(title=f"Retrieved Chunks (top {len(results)})", show_lines=True, expand=True)
        table.add_column("Rank", style="dim", justify="right", width=4)
        table.add_column("Score", style="yellow", justify="right", width=7)
        table.add_column("Page", style="magenta", justify="right", width=4)
        table.add_column("Chunk", style="cyan", justify="right", width=5)
        table.add_column("Text", style="white")

        for i, r in enumerate(results):
            table.add_row(
                str(i + 1),
                f"{r.score:.4f}",
                str(r.page),
                str(r.chunk_index),
                r.text[:200].replace("\n", " "),
            )

        self._console.print(table)
        self._console.print(Panel(
            package.prompt,
            title="[bold]Full Prompt[/bold]",
            border_style="dim",
            expand=True,
        ))

    def report_case_result(self, index: int, total: int, result: EvalResult) -> None:
        self._console.print(f"[cyan][{index}/{total}][/cyan] {result.case.question}")
        status = "[bold green]PASS[/bold green]" if result.passed else "[bold red]FAIL[/bold red]"
        self._console.print(f"  {status} — Answer: [italic]{result.answer[:80]}[/italic]")
        if result.matched_keywords:
            self._console.print(f"  Matched: {result.matched_keywords}")
        self._console.print()

    def report_summary(self, results: list[EvalResult]) -> None:
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        avg_latency = sum(r.latency_seconds for r in results) / total if total else 0
        avg_score = sum(r.top_score for r in results) / total if total else 0

        table = Table(title="Evaluation Summary", show_lines=True)
        table.add_column("Question", style="cyan", max_width=45)
        table.add_column("Result", justify="center", width=6)
        table.add_column("Answer", style="white", max_width=35)
        table.add_column("Top Score", style="yellow", justify="right", width=9)
        table.add_column("Latency", style="dim", justify="right", width=8)

        for r in results:
            status = "✓" if r.passed else "✗"
            style = "green" if r.passed else "red"
            table.add_row(
                r.case.question,
                f"[{style}]{status}[/{style}]",
                r.answer[:60],
                f"{r.top_score:.4f}",
                f"{r.latency_seconds:.1f}s",
            )

        self._console.print(table)
        pct = (passed / total * 100) if total else 0.0
        color = "green" if passed == total else "yellow"
        self._console.print(
            f"\n[bold]Score: {passed}/{total} ([{color}]{pct:.0f}%[/{color}])[/bold] | "
            f"Avg latency: {avg_latency:.1f}s | "
            f"Avg top retrieval score: {avg_score:.4f}"
        )
