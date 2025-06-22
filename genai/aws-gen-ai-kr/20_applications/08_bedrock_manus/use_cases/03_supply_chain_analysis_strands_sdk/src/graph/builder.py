from langgraph.graph import StateGraph, START, END

from .types import State
from .nodes import (
    clarification_node,
    human_feedback_node,
    planner_node,
    supervisor_node,
    research_node,
    code_node,
    reporter_node,
    
)
from .scm_nodes import (
    scm_researcher_node,
    scm_insight_analyzer_node,
    scm_impact_analyzer_node,
    scm_correlation_analyzer_node,
    scm_mitigation_planner_node,
)

def build_graph():
    """Build and return the agent workflow graph."""
    builder = StateGraph(State)
    builder.add_node("clarifier", clarification_node)
    builder.add_node("human_feedback", human_feedback_node)
    builder.add_node("planner", planner_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", research_node)
    builder.add_node("coder", code_node)
    builder.add_node("reporter", reporter_node)
    builder.add_edge(START, "clarifier")
    return builder.compile()


def build_graph():
#def build_scm_graph():
    """Build and return the SCM specialized workflow graph."""
    builder = StateGraph(State)
    
    # Add all SCM nodes
    builder.add_node("scm_researcher", scm_researcher_node)
    builder.add_node("scm_insight_analyzer", scm_insight_analyzer_node)
    #builder.add_node("planner", planner_node)
    #builder.add_node("supervisor", supervisor_node)
    #builder.add_node("scm_impact_analyzer", scm_impact_analyzer_node)
    #builder.add_node("scm_correlation_analyzer", scm_correlation_analyzer_node)
    #builder.add_node("scm_mitigation_planner", scm_mitigation_planner_node)
    #builder.add_node("reporter", reporter_node)
    
    # SCM workflow: scm_researcher → scm_insight_analyzer → planner → supervisor → [scm analyzers] → reporter
    builder.add_edge(START, "scm_researcher")
    builder.add_edge("scm_insight_analyzer", END)
    
    return builder.compile()
