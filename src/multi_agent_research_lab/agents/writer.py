"""Writer agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""

        try:
            sources_block = ""
            if state.sources:
                lines = [f"  [{i+1}] {s.title}" + (f" — {s.url}" if s.url else "") for i, s in enumerate(state.sources)]
                sources_block = "\nSources:\n" + "\n".join(lines)

            system = (
                f"You are a professional writer producing content for {state.request.audience}. "
                "Write a clear, well-structured answer that synthesises the research and analysis. "
                "Include inline source references like [1] where relevant."
            )
            user = (
                f"Query: {state.request.query}\n\n"
                f"Research notes:\n{state.research_notes or 'N/A'}\n\n"
                f"Analysis:\n{state.analysis_notes or 'N/A'}"
                f"{sources_block}\n\n"
                "Write a ~500-word answer."
            )
            response = self._llm.complete(system, user)
            state.final_answer = response.content
            state.add_trace_event("writer_done", {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            })
        except Exception as exc:
            state.errors.append(f"writer error: {exc}")

        return state
