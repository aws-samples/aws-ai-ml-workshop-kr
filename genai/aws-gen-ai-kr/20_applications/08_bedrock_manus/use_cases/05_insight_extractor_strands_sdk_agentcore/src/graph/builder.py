from strands.multiagent import GraphBuilder
#from langgraph.graph import StateGraph, START, END

#from .types import State
from .nodes import (
    FunctionNode,
    #supervisor_node,
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
    builder.add_node(coordinator, "coordinator")
    builder.add_node(planner, "planner")

    # Set entry points (optional - will be auto-detected if not specified)
    builder.set_entry_point("coordinator")
    
    # Add conditional edge - only go to planner if handoff is requested
    builder.add_edge("coordinator", "planner", condition=should_handoff_to_planner)

    # Build the graph
    return builder.build()


    #builder = StateGraph(State)
    #builder.add_node("coordinator", coordinator_node)
    #builder.add_node("planner", planner_node)
    #builder.add_node("supervisor", supervisor_node)
    #builder.add_node("coder", code_node)
    #builder.add_node("reporter", reporter_node)
    #builder.add_edge(START, "coordinator")
    #return builder.compile()
