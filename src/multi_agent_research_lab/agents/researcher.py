"""Researcher agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self) -> None:
        self._search = SearchClient()
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""

        try:
            sources = self._search.search(
                state.request.query,
                max_results=state.request.max_sources,
            )
            state.sources.extend(sources)

            snippets = "\n\n".join(
                f"[{i+1}] {s.title}\n{s.snippet}" for i, s in enumerate(sources)
            )
            system = (
                "You are a research assistant. Summarize the provided sources into "
                "clear, concise notes that capture key facts, claims, and gaps. "
                "Audience: technical learners."
            )
            user = (
                f"Query: {state.request.query}\n\n"
                f"Sources:\n{snippets}\n\n"
                "Write research notes (bullet points preferred)."
            )
            response = self._llm.complete(system, user)
            state.research_notes = response.content
            state.add_trace_event("researcher_done", {
                "sources_found": len(sources),
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            })
        except Exception as exc:
            state.errors.append(f"researcher error: {exc}")

        return state
