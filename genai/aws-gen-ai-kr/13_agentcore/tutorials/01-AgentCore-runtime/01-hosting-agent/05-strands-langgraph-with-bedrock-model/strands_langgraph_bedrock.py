
import sys, os
#module_path = "../../../.."
#module_path = "../../../.."
#sys.path.append(os.path.abspath(module_path))

import json
import logging
import argparse
from textwrap import dedent
from graph import build_graph
from src.utils.common_utils import get_message_from_string
from src.utils.strands_sdk_utils import strands_utils
from src.prompts.template import apply_prompt_template

from bedrock_agentcore.runtime import BedrockAgentCoreApp

app = BedrockAgentCoreApp()

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
    logger.info(f"{Colors.GREEN}===== Starting workflow ====={Colors.END}")
    logger.info(f"{Colors.GREEN}\nuser input: {user_input}{Colors.END}")

    user_prompts = dedent(
        '''
        Here is a user request: <user_request>{user_request}</user_request>
        '''
    )
    context = {"user_request": user_input}
    user_prompts = user_prompts.format(**context)
    messages = [get_message_from_string(role="user", string=user_prompts, imgs=[])]


    result = graph.invoke(
        input={
            # Runtime Variables
            "messages": messages,
            "request": user_input,
            "request_prompt": user_prompts
        },
        config={
            "recursion_limit": 100
        }
    )
    logger.debug(f"{Colors.RED}Final workflow state: {result}{Colors.END}")
    logger.info(f"{Colors.GREEN}===== Workflow completed successfully ====={Colors.END}")
    return result

@app.entrypoint
def strands_langgraph_bedrock(payload):
    """
    Invoke the agent with a payload
    """
    user_query = payload.get("prompt")

    result = run_agent_workflow(
        user_input=user_query,
        debug=False
    )

    # Create the input in the format expected by LangGraph
    #response = agent.invoke({"messages": [HumanMessage(content=user_input)]})

    # Extract the final message content
    #return response["messages"][-1].content
    return result

if __name__ == "__main__":
    app.run()

    #parser = argparse.ArgumentParser()
    #parser.add_argument("payload", type=str)
    #args = parser.parse_args()
    #response = strands_langgraph_bedrock(json.loads(args.payload))
    #print(response)

