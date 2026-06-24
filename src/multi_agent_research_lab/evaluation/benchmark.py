"""Benchmark for single-agent vs multi-agent."""

from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState


Runner = Callable[[str], ResearchState]

# gpt-4o-mini pricing (USD per 1M tokens, as of 2024)
_COST_PER_1M_INPUT = 0.15
_COST_PER_1M_OUTPUT = 0.60


def _estimate_cost(state: ResearchState) -> float:
    total_input = sum(e["payload"].get("input_tokens", 0) for e in state.trace if "input_tokens" in e.get("payload", {}))
    total_output = sum(e["payload"].get("output_tokens", 0) for e in state.trace if "output_tokens" in e.get("payload", {}))
    return (total_input / 1_000_000) * _COST_PER_1M_INPUT + (total_output / 1_000_000) * _COST_PER_1M_OUTPUT


def run_benchmark(run_name: str, query: str, runner: Runner) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, token cost, and basic quality signal."""

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    cost = _estimate_cost(state)
    answer_len = len(state.final_answer or "")
    # Heuristic quality: longer + no errors = higher score (peer review sẽ cho điểm thật)
    quality = min(10.0, (answer_len / 200) + (2.0 if not state.errors else 0.0))

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=round(latency, 2),
        estimated_cost_usd=round(cost, 6),
        quality_score=round(quality, 1),
        notes=f"iterations={state.iteration}, errors={len(state.errors)}, sources={len(state.sources)}",
    )
    return state, metrics
