from strands.multiagent import GraphBuilder
#from langgraph.graph import StateGraph, START, END

#from .types import State
from .nodes import (
    FunctionNode,
    StreamingFunctionNode,
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

class StreamingGraphWrapper:
    """Wrapper for graph that supports streaming execution"""
    
    def __init__(self, graph):
        self.graph = graph
        self.streaming_nodes = {}
    
    def add_streaming_node(self, node_name, node_func):
        """Add a node that supports streaming"""
        self.streaming_nodes[node_name] = StreamingFunctionNode(node_func, node_name)
    
    async def invoke_async_streaming(self, task=None, **kwargs):
        """Execute graph with streaming support using original graph routing logic"""
        from .nodes import _global_node_states
        
        # Clear global state
        _global_node_states.clear()
        
        # Start from graph entry point
        entry_point = list(self.graph.entry_points)[0] if self.graph.entry_points else None
        current_node_id = entry_point.node_id if entry_point else None
        
        while current_node_id and current_node_id != "__end__":
            # Check if current node supports streaming
            if current_node_id in self.streaming_nodes:
                streaming_node = self.streaming_nodes[current_node_id]
                
                # Execute streaming node and yield events
                async for event in streaming_node.invoke_async_streaming(task=task, **kwargs):
                    yield event
                
                # Find next node using original graph logic
                current_node_id = self._get_next_node(current_node_id)
                
            else:
                print(f"[WARNING] Node '{current_node_id}' not found in streaming nodes!")
                current_node_id = "__end__"
                
    def _get_next_node(self, current_node_id):
        """Get next node using original graph's edge conditions"""
        from .nodes import _global_node_states
        
        # Find outgoing edges from current node
        for edge in self.graph.edges:
            if edge.from_node.node_id == current_node_id:
                # Check edge condition if it exists
                if edge.condition is None:
                    # No condition, always take this edge
                    return edge.to_node.node_id
                else:
                    # Check condition with shared state
                    shared_state = _global_node_states.get('shared', {})
                    if edge.condition(shared_state):
                        return edge.to_node.node_id
        
        # No valid edge found, end workflow
        return "__end__"

def build_streaming_graph():
    """Build and return a streaming-enabled graph wrapper."""
    base_graph = build_graph()
    streaming_wrapper = StreamingGraphWrapper(base_graph)
    
    # Add streaming nodes
    streaming_wrapper.add_streaming_node("coordinator", coordinator_node)
    streaming_wrapper.add_streaming_node("planner", planner_node)
    streaming_wrapper.add_streaming_node("supervisor", supervisor_node)
    
    return streaming_wrapper
