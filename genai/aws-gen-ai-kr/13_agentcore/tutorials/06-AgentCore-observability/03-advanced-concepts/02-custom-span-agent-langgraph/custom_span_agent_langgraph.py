
import sys, os
import json
import logging
import argparse
from textwrap import dedent
from graph import build_graph
from src.utils.common_utils import get_message_from_string
from src.utils.strands_sdk_utils import strands_utils
from src.prompts.template import apply_prompt_template

import argparse
from opentelemetry import baggage, context, trace

###########################
####   Session info    ####
###########################

def set_session_context(session_id):
    """Set the session ID in OpenTelemetry baggage for trace correlation"""
    ctx = baggage.set_baggage("session.id", session_id)
    token = context.attach(ctx)
    logging.info(f"Session ID '{session_id}' attached to telemetry context")
    return token

def set_session_context(session_id, user_type=None, experiment_id=None):

    ctx = baggage.set_baggage("session.id", str(session_id))
    logging.info(f"Session ID '{session_id}' attached to telemetry context")

    if user_type:
        ctx = baggage.set_baggage("user.type", user_type, context=ctx)
        logging.info(f"user Type '{user_type}' attached to telemetry context")
    if experiment_id:
        ctx = baggage.set_baggage("experiment.id", experiment_id, context=ctx)
        logging.info(f"Experiment ID '{experiment_id}' attached to telemetry context")

    return context.attach(ctx)

###########################
#### Agent Code below: ####
###########################

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

def parse_arguments():
    parser = argparse.ArgumentParser(description='Agent with Custom Span Creation')
    parser.add_argument(
        '--session-id', 
        type=str,
        required=True,
        help='Session ID to associate with this agent run'
    )
    return parser.parse_args()

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
    context_info = {"user_request": user_input}
    user_prompts = user_prompts.format(**context_info)
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

def strands_langgraph_bedrock(payload):
    """
    Invoke the agent with a payload
    """

    session_id = payload.get("session_id")

    print ("session-id", session_id)
    context_token = set_session_context(payload.get("session-id"))


    try:
        user_query = payload.get("prompt")

        # Get tracer for main application
        tracer = trace.get_tracer("data_analysis_agent", "1.0.0")
        with tracer.start_as_current_span("data_analysis_session") as main_span:
            result = run_agent_workflow(
                user_input=user_query,
                debug=False
            )

            print("Result:", result)

            main_span.add_event( # add_event 사용 시 (CloudWatch)GenAI Observability의 trace의 event에서 바로 확인 x, "Traces"의 event에서는 확인 가능
                "user_query",
                {"user-query": str(user_query)}
            ) # attribute.name에 _(under bar)가 들어가면 안된다. 
            main_span.add_event( # add_event 사용 시 (CloudWatch)GenAI Observability의 trace의 event에서 바로 확인 x, "Traces"의 event에서는 확인 가능
                "data_analysis_completed",
                {
                    "result": str(result),
                    "success": True
                }
            )

    finally:
        context.detach(context_token)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=str)
    args, _ = parser.parse_known_args()
    response = strands_langgraph_bedrock(json.loads(args.payload))
    print(response)
