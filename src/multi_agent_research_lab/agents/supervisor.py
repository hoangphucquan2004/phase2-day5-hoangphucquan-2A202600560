"""Supervisor / router skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route."""

        settings = get_settings()

        # Fallback nếu vượt quá số vòng lặp tối đa
        if state.iteration >= settings.max_iterations:
            state.errors.append("Max iterations reached, forcing stop.")
            state.record_route("done")
            return state

        # Fallback nếu có quá nhiều lỗi
        if len(state.errors) >= 3:
            state.record_route("done")
            return state

        # Routing logic: tuần tự researcher → analyst → writer → done
        if state.research_notes is None:
            route = "researcher"
        elif state.analysis_notes is None:
            route = "analyst"
        elif state.final_answer is None:
            route = "writer"
        else:
            route = "done"

        state.record_route(route)
        state.add_trace_event("supervisor_route", {"route": route, "iteration": state.iteration})
        return state
