"""Search client abstraction for ResearcherAgent."""

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient:
    """Provider-agnostic search client. Uses Tavily if key is set, else returns mock data."""

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""

        settings = get_settings()
        if settings.tavily_api_key:
            return self._tavily_search(query, max_results, settings.tavily_api_key)
        return self._mock_search(query, max_results)

    def _tavily_search(self, query: str, max_results: int, api_key: str) -> list[SourceDocument]:
        from tavily import TavilyClient  # type: ignore[import-untyped]

        client = TavilyClient(api_key=api_key)
        results = client.search(query=query, max_results=max_results)
        return [
            SourceDocument(
                title=r.get("title", "Untitled"),
                url=r.get("url"),
                snippet=r.get("content", ""),
            )
            for r in results.get("results", [])
        ]

    def _mock_search(self, query: str, max_results: int) -> list[SourceDocument]:
        """Return static mock documents for offline / no-key development."""

        mocks = [
            SourceDocument(
                title="Overview of the topic",
                url=None,
                snippet=f"This is a mock result about: {query}. It covers the main concepts and recent developments.",
            ),
            SourceDocument(
                title="State of the art survey",
                url=None,
                snippet=f"A survey of recent advances related to: {query}. Multiple approaches are compared.",
            ),
            SourceDocument(
                title="Practical applications",
                url=None,
                snippet=f"Real-world use cases and applications of: {query}. Includes case studies.",
            ),
        ]
        return mocks[:max_results]
