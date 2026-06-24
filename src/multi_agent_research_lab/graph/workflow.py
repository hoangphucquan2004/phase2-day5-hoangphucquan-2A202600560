"""LangGraph workflow."""

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.state import ResearchState


def _route(state: ResearchState) -> str:
    """Return the last recorded route for LangGraph conditional edge."""
    return state.route_history[-1] if state.route_history else "done"


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph."""

    def build(self) -> object:
        """Create a compiled LangGraph graph."""

        supervisor = SupervisorAgent()
        researcher = ResearcherAgent()
        analyst = AnalystAgent()
        writer = WriterAgent()

        graph: StateGraph = StateGraph(ResearchState)

        graph.add_node("supervisor", supervisor.run)
        graph.add_node("researcher", researcher.run)
        graph.add_node("analyst", analyst.run)
        graph.add_node("writer", writer.run)

        graph.set_entry_point("supervisor")

        graph.add_conditional_edges(
            "supervisor",
            _route,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "done": END,
            },
        )

        # Sau mỗi worker, quay về supervisor để quyết định bước tiếp theo
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")

        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state."""

        compiled = self.build()
        result = compiled.invoke(state)
        if isinstance(result, ResearchState):
            return result
        return ResearchState(**result)
