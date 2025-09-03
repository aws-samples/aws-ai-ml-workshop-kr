
import logging
from src.graph.builder import build_graph

# Simple logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Colors:
    GREEN = '\033[92m'
    END = '\033[0m'

async def run_graph_streaming_workflow(user_input: str):
    """Full graph streaming workflow that maintains graph structure.

    Args:
        user_input: The user's query or request  

    Returns:
        The result of the workflow execution
    """
    if not user_input:
        raise ValueError("Input could not be empty")

    logger.info(f"\n{Colors.GREEN}Starting graph streaming workflow{Colors.END}")

    # Prepare user prompt
    user_prompts = f"Here is a user request: <user_request>{user_input}</user_request>"


    #########################
    ## modification START  ##
    #########################

    # Build and execute graph
    graph = build_graph()
    result = await graph.invoke_async(
        task={
            "request": user_input,
            "request_prompt": user_prompts
        }
    )
    #########################
    ## modification END    ##
    #########################

    logger.info(f"\n{Colors.GREEN}Graph streaming workflow completed{Colors.END}")
    return result
