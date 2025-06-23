import logging
from textwrap import dedent
from src.config import SCM_TEAM_MEMBERS
from src.graph import build_graph
from src.graph.builder import build_graph
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
    """Run the SCM specialized workflow with the given user input.

    Args:
        user_input: The user's SCM-related query or request
        debug: If True, enables debug level logging

    Returns:
        The final state after the SCM workflow completes
    """

    if not user_input:
        raise ValueError("Input could not be empty")

    if debug:
        enable_debug_logging()

    logger.info(f"{Colors.GREEN}===== Starting SCM workflow ====={Colors.END}")
    logger.info(f"{Colors.GREEN}\nSCM user input: {user_input}{Colors.END}")
    
    user_prompts = dedent(
        '''
        Here is a SCM-related user request: <user_request>{user_request}</user_request>
        
        This appears to be a supply chain management query.
        Please analyze this for supply chain disruptions, logistics issues, or other SCM-related impacts.
        '''
    )
    context = {"user_request": user_input}
    user_prompts = user_prompts.format(**context)
    messages = [get_message_from_string(role="user", string=user_prompts, imgs=[])]
    
    result = graph.invoke(
        input={
            # Constants
            "SCM_TEAM_MEMBERS": SCM_TEAM_MEMBERS,
            # Runtime Variables
            "messages": messages,
            "search_before_planning": False,
            "request": user_input,
            "request_prompt": user_prompts
        },
        config={
            "recursion_limit": 100
        }
    )
    logger.debug(f"{Colors.RED}Final SCM workflow state: {result}{Colors.END}")
    logger.info(f"{Colors.GREEN}===== SCM Workflow completed successfully ====={Colors.END}")
    
    # Show artifacts summary
    from src.utils.scm_file_utils import get_artifacts_summary
    summary = get_artifacts_summary()
    logger.info(f"{Colors.BLUE}Generated {summary['total_files']} analysis files in ./artifacts/{Colors.END}")
    
    return result


if __name__ == "__main__":
    print("Regular workflow graph:")
    #print(get_graph().get_graph().draw_mermaid())
    print("\nSCM workflow graph:")
    print(graph.get_graph().draw_mermaid())
