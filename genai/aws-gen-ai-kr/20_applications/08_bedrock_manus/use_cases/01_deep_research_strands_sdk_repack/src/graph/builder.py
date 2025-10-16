
import asyncio
from strands.multiagent import GraphBuilder
from src.utils.strands_sdk_utils import FunctionNode
from src.utils.event_queue import has_events, get_event
from .nodes import (
    clarification_node,
    human_feedback_node,
    planner_node,
    supervisor_node,
    #should_handoff_to_planner,
)

class StreamableGraph:
    """Graph wrapper that adds streaming capability to Strands graphs."""
    
    def __init__(self, graph):
        self.graph = graph
    
    async def invoke_async(self, task):
        """Original non-streaming invoke method."""
        return await self.graph.invoke_async(task)
    
    async def _cleanup_workflow(self, workflow_task):
        """Handle workflow completion and cleanup."""
        if not workflow_task.done():
            try:
                await asyncio.wait_for(workflow_task, timeout=1.0)
            except asyncio.TimeoutError:
                workflow_task.cancel()
                try: 
                    await workflow_task
                except asyncio.CancelledError: 
                    pass
    
    async def _yield_pending_events(self):
        """Yield any pending events from queue."""
        while has_events():
            event = get_event()
            if event: 
                yield event
    
    async def stream_async(self, task):
        """Stream events from graph execution using background task + event queue pattern."""
        
        # Step 1: Run graph backgound and put event into the global queue
        async def run_workflow():
            try:
                return await self.graph.invoke_async(task)
            except Exception as e:
                print(f"Workflow error: {e}")
                raise
        
        workflow_task = asyncio.create_task(run_workflow())
        
        # Step 2: Consuming event in the global queue
        try:
            while not workflow_task.done():
                async for event in self._yield_pending_events():
                    yield event
                await asyncio.sleep(0.005)
        finally:
            await self._cleanup_workflow(workflow_task)
            async for event in self._yield_pending_events():
                yield event
        
        yield {"type": "workflow_complete", "message": "All events processed through global queue"}

def build_graph():
    """Build and return the agent workflow graph with streaming capability."""
    builder = GraphBuilder()

    # Add nodes
    
    clarifier = FunctionNode(func=clarification_node, name="clarifier")
    human_feedback = FunctionNode(func=human_feedback_node, name="human_feedback")
    planner = FunctionNode(func=planner_node, name="planner")
    supervisor = FunctionNode(func=supervisor_node, name="supervisor")

    builder.add_node(clarifier, "clarifier")
    builder.add_node(human_feedback, "human_feedback")
    builder.add_node(planner, "planner")
    builder.add_node(supervisor, "supervisor")
    
    #builder.add_node(coordinator, "coordinator")
    

    # Set entry point and edges
    builder.set_entry_point("clarifier")
    builder.add_edge("clarifier", "human_feedback")
    builder.add_edge("human_feedback", "planner")
    builder.add_edge("planner", "supervisor")
    #builder.add_edge("coordinator", "planner", condition=should_handoff_to_planner)
    

    # Return graph with streaming capability
    return StreamableGraph(builder.build())



    # builder.add_node("clarifier", clarification_node)
    # builder.add_node("human_feedback", human_feedback_node)
    # builder.add_node("planner", planner_node)
    # builder.add_node("supervisor", supervisor_node)
    # builder.add_node("researcher", research_node)
    # builder.add_node("coder", code_node)
    # builder.add_node("reporter", reporter_node)
