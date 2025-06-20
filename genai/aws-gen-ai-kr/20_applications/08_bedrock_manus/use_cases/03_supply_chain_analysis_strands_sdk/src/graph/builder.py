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
