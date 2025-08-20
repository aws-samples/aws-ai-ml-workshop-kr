from strands.multiagent import GraphBuilder
#from langgraph.graph import StateGraph, START, END

#from .types import State
from .nodes import (
    FunctionNode,
    supervisor_node,
    #code_node,
    coordinator_node,
    #reporter_node,
    planner_node,
    should_handoff_to_planner,
)

def build_graph():
    """Build and return the agent workflow graph."""
    builder = GraphBuilder()

    # Nodes
    coordinator = FunctionNode(func=coordinator_node, name="coordinator")
    planner = FunctionNode(func=planner_node, name="planner")
    supervisor = FunctionNode(func=supervisor_node, name="supervisor")
    
    builder.add_node(coordinator, "coordinator")
    builder.add_node(planner, "planner")
    builder.add_node(supervisor, "supervisor")

    # Set entry points (optional - will be auto-detected if not specified)
    builder.set_entry_point("coordinator")
    
    # Add conditional edge - only go to planner if handoff is requested
    builder.add_edge("coordinator", "planner", condition=should_handoff_to_planner)
    
    # Add edge - planner to supervisor (no condition needed)
    builder.add_edge("planner", "supervisor")

    # Build the graph
    return builder.build()
