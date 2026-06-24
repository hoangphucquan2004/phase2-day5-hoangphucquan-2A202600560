"""Analyst agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""

        if not state.research_notes:
            state.errors.append("analyst: no research_notes to analyse")
            return state

        try:
            system = (
                "You are a critical analyst. Given research notes, extract the key claims, "
                "compare viewpoints, identify strengths and weaknesses in evidence, "
                "and flag anything uncertain or missing."
            )
            user = (
                f"Query: {state.request.query}\n\n"
                f"Research notes:\n{state.research_notes}\n\n"
                "Provide structured analysis with: key claims, evidence quality, open questions."
            )
            response = self._llm.complete(system, user)
            state.analysis_notes = response.content
            state.add_trace_event("analyst_done", {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            })
        except Exception as exc:
            state.errors.append(f"analyst error: {exc}")

        return state
