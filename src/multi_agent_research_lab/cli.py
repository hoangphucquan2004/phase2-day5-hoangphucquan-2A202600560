"""Command-line entrypoint for the lab starter."""

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline using one LLM call."""

    _init()
    from multi_agent_research_lab.services.llm_client import LLMClient

    request = ResearchQuery(query=query)
    state = ResearchState(request=request)

    llm = LLMClient()
    response = llm.complete(
        system_prompt="You are a research assistant. Answer the query thoroughly in ~500 words.",
        user_prompt=f"Query: {query}",
    )
    state.final_answer = response.content
    console.print(Panel.fit(state.final_answer, title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow skeleton."""

    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    console.print(result.model_dump_json(indent=2))


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Benchmark single-agent vs multi-agent and print a comparison table."""

    _init()
    from rich.table import Table

    from multi_agent_research_lab.evaluation.benchmark import run_benchmark
    from multi_agent_research_lab.services.llm_client import LLMClient

    # --- single-agent runner ---
    def single_runner(q: str) -> ResearchState:
        llm = LLMClient()
        state = ResearchState(request=ResearchQuery(query=q))
        response = llm.complete(
            system_prompt="You are a research assistant. Answer the query thoroughly in ~500 words.",
            user_prompt=f"Query: {q}",
        )
        state.final_answer = response.content
        # ghi token vào trace để estimate cost
        state.add_trace_event("single_agent_done", {
            "input_tokens": response.input_tokens or 0,
            "output_tokens": response.output_tokens or 0,
        })
        return state

    # --- multi-agent runner ---
    def multi_runner(q: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=q))
        return MultiAgentWorkflow().run(state)

    console.print("\n[bold]Running single-agent baseline...[/bold]")
    _, m_single = run_benchmark("single-agent", query, single_runner)

    console.print("[bold]Running multi-agent workflow...[/bold]\n")
    _, m_multi = run_benchmark("multi-agent", query, multi_runner)

    table = Table(title="Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Single-Agent", justify="right")
    table.add_column("Multi-Agent", justify="right")

    table.add_row("Latency (s)", str(m_single.latency_seconds), str(m_multi.latency_seconds))
    table.add_row("Est. Cost (USD)", str(m_single.estimated_cost_usd), str(m_multi.estimated_cost_usd))
    table.add_row("Quality (heuristic)", str(m_single.quality_score), str(m_multi.quality_score))
    table.add_row("Notes", m_single.notes, m_multi.notes)

    console.print(table)


if __name__ == "__main__":
    app()
