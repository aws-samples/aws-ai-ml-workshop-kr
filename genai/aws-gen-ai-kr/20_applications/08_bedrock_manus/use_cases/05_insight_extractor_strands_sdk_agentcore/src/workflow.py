import logging
from textwrap import dedent
#from src.config import TEAM_MEMBERS
from src.graph import build_graph
from src.utils.common_utils import get_message_from_string

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Default level is INFO
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

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

def run_agent_workflow(user_input: str, debug: bool = False):
    """Run the agent workflow with the given user input.

    Args:
        user_input: The user's query or request
        debug: If True, enables debug level logging

    Returns:
        The final state after the workflow completes
    """
    if not user_input:
        raise ValueError("Input could not be empty")

    if debug:
        enable_debug_logging()

    #logger.info(f"Starting workflow with user input: {user_input}")
    # Clear global state at workflow start
    from src.graph.nodes import _global_node_states
    _global_node_states.clear()
    
    logger.info(f"{Colors.GREEN}===== Starting workflow ====={Colors.END}")
    logger.info(f"{Colors.GREEN}\nuser input: {user_input}{Colors.END}")
    
    # Prepare input as dictionary for flexible parameter passing
    user_prompts = dedent(
        '''
        Here is a user request: <user_request>{user_request}</user_request>
        '''
    )
    context = {"user_request": user_input}
    user_prompts = user_prompts.format(**context)

    # Pass dictionary to graph for **kwargs support
    input_data = {
        "request": user_input,
        "request_prompt": user_prompts
    }
    result = graph(input_data)
        
    # result = graph.invoke(
    #     input={
    #         # Constants
    #         "TEAM_MEMBERS": TEAM_MEMBERS,
    #         # Runtime Variables
    #         "messages": messages,
    #         #"deep_thinking_mode": True,
    #         #"search_before_planning": False,
    #         "request": user_input,
    #         "request_prompt": user_prompts
    #     },
    #     config={
    #         "recursion_limit": 100
    #     }
    # )
    logger.debug(f"{Colors.RED}Final workflow state: {result}{Colors.END}")
    logger.info(f"{Colors.GREEN}===== Workflow completed successfully ====={Colors.END}")
    return result


if __name__ == "__main__":
    run_agent_workflow(
        user_input="안녕 나는 장동진이야"
    )
    #print(graph.get_graph().draw_mermaid())
