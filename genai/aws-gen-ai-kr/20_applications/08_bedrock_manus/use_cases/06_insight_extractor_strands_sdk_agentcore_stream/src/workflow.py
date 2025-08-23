import logging
from textwrap import dedent
from src.graph import build_graph
from src.graph.builder import build_streaming_graph


def enable_debug_logging():
    """Enable debug level logging for more detailed execution information."""
    logging.getLogger(__name__).setLevel(logging.DEBUG)

# 로거 설정을 전역으로 한 번만 수행
logger = logging.getLogger(__name__)
logger.propagate = False
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
handler = logging.StreamHandler()
formatter = logging.Formatter('\n%(levelname)s [%(name)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)  # 기본 레벨은 INFO로 설정

class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# Create the graph
graph = build_graph()

def get_graph():
    return graph


async def run_agent_workflow_streaming(user_input: str, debug: bool = False):
    """Run the agent workflow with streaming support using callback mechanism.

    Args:
        user_input: The user's query or request
        debug: If True, enables debug level logging

    Yields:
        Streaming events from the workflow execution
    """
    if not user_input:
        raise ValueError("Input could not be empty")

    if debug:
        enable_debug_logging()

    # Clear global state at workflow start
    from src.graph.nodes import _global_node_states, set_streaming_callback
    _global_node_states.clear()

    logger.info(f"{Colors.GREEN}===== Starting streaming workflow ====={Colors.END}")
    logger.info(f"{Colors.GREEN}\nuser input: {user_input}{Colors.END}")
    
    # Prepare input as dictionary for flexible parameter passing
    user_prompts = dedent(
        '''
        Here is a user request: <user_request>{user_request}</user_request>
        '''
    )
    context = {"user_request": user_input}
    user_prompts = user_prompts.format(**context)

    # Create a list to collect streaming events
    streaming_events = []
    
    def streaming_callback(event):
        """Callback to collect streaming events"""
        streaming_events.append(event)
    
    # Set the streaming callback
    set_streaming_callback(streaming_callback)

    try:
        # Execute the workflow
        result = await graph.invoke_async(
            task={
                "request": user_input,
                "request_prompt": user_prompts
            }
        )
        
        # Yield all collected streaming events
        for event in streaming_events:
            yield event
            
        # Yield final result
        yield {"type": "workflow_complete", "result": result}
        
    finally:
        # Clean up callback
        set_streaming_callback(None)
        
    logger.debug(f"{Colors.RED}Final workflow state: {result}{Colors.END}")
    logger.info(f"{Colors.GREEN}===== Streaming workflow completed successfully ====={Colors.END}")


async def run_graph_streaming_workflow(user_input: str, debug: bool = False):
    """Full graph streaming workflow that maintains graph structure.
    
    Args:
        user_input: The user's query or request  
        debug: If True, enables debug level logging
        
    Yields:
        Streaming events from graph execution in real-time
    """
    if not user_input:
        raise ValueError("Input could not be empty")

    if debug:
        enable_debug_logging()

    logger.info(f"{Colors.GREEN}===== Starting graph streaming workflow ====={Colors.END}")
    logger.info(f"{Colors.GREEN}\nuser input: {user_input}{Colors.END}")
    
    # Prepare input as dictionary
    user_prompts = dedent(
        '''
        Here is a user request: <user_request>{user_request}</user_request>
        '''
    )
    context = {"user_request": user_input}
    user_prompts = user_prompts.format(**context)

    # Build streaming graph
    streaming_graph = build_streaming_graph()
    
    # Execute graph with streaming
    async for event in streaming_graph.invoke_async_streaming(
        task={
            "request": user_input,
            "request_prompt": user_prompts
        }
    ):
        yield event
        
    logger.info(f"{Colors.GREEN}===== Graph streaming workflow completed ====={Colors.END}")


