import logging
import asyncio
from typing import Any, Annotated
from strands.types.tools import ToolResult, ToolUse
from src.utils.strands_sdk_utils import strands_utils
from src.prompts.template import apply_prompt_template
from src.utils.common_utils import get_message_from_string
from src.tools import python_repl_tool, bash_tool


# Simple logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TOOL_SPEC = {
    "name": "coder_agent_tool",
    "description": "Execute Python code and bash commands using a specialized coder agent. This tool provides access to a coder agent that can execute Python code for data analysis and calculations, run bash commands for system operations, and handle complex programming tasks.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The coding task or question that needs to be executed by the coder agent."
                }
            },
            "required": ["task"]
        }
    }
}

RESPONSE_FORMAT = "Response from {}:\n\n<response>\n{}\n</response>\n\n*Please execute the next step.*"
FULL_PLAN_FORMAT = "Here is full plan :\n\n<full_plan>\n{}\n</full_plan>\n\n*Please consider this to select the next step.*"
CLUES_FORMAT = "Here is clues from {}:\n\n<clues>\n{}\n</clues>\n\n"

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    END = '\033[0m'

def handle_coder_agent_tool(task: Annotated[str, "The coding task or question that needs to be executed by the coder agent."]):
    """
    Execute Python code and bash commands using a specialized coder agent.

    This tool provides access to a coder agent that can:
    - Execute Python code for data analysis and calculations
    - Run bash commands for system operations
    - Handle complex programming tasks

    Args:
        task: The coding task or question that needs to be executed

    Returns:
        The result of the code execution or analysis
    """
    print()  # Add newline before log
    logger.info(f"\n{Colors.GREEN}Coder Agent Tool starting task{Colors.END}")

    # Try to extract shared state from global storage
    from src.graph.nodes import _global_node_states
    shared_state = _global_node_states.get('shared', None)

    if not shared_state:
        logger.warning("No shared state found")
        return "Error: No shared state available"

    request_prompt, full_plan = shared_state.get("request_prompt", ""), shared_state.get("full_plan", "")
    clues, messages = shared_state.get("clues", ""), shared_state.get("messages", [])

    # Create coder agent with specialized tools using consistent pattern
    coder_agent = strands_utils.get_agent(
        agent_name="coder",
        system_prompts=apply_prompt_template(prompt_name="coder", prompt_context={"USER_REQUEST": request_prompt, "FULL_PLAN": full_plan}),
        agent_type="claude-sonnet-4", # claude-sonnet-3-5-v-2, claude-sonnet-3-7, claude-sonnet-4
        enable_reasoning=False,
        tools=[python_repl_tool, bash_tool],
        streaming=True  # Enable streaming for consistency
    )

    # Prepare message with context if available
    message = '\n\n'.join([messages[-1]["content"][-1]["text"], clues])

    # Process streaming response and collect text in one pass
    async def process_coder_stream():
        full_text = ""
        async for event in strands_utils.process_streaming_response_yield(
            coder_agent, message, agent_name="coder", source="coder_tool"
        ):
            if event.get("event_type") == "text_chunk": full_text += event.get("data", "")
        return {"text": full_text}

    response = asyncio.run(process_coder_stream())
    result_text = response['text']

    # Update clues
    clues = '\n\n'.join([clues, CLUES_FORMAT.format("coder", response["text"])])

    # Update history
    history = shared_state.get("history", [])
    history.append({"agent":"coder", "message": response["text"]})

    # Update shared state
    shared_state['messages'] = [get_message_from_string(role="user", string=RESPONSE_FORMAT.format("coder", response["text"]), imgs=[])]
    shared_state['clues'] = clues
    shared_state['history'] = history

    logger.info(f"\n{Colors.GREEN}Coder Agent Tool completed successfully{Colors.END}")
    return result_text

# Function name must match tool name
def coder_agent_tool(tool: ToolUse, **_kwargs: Any) -> ToolResult:
    tool_use_id = tool["toolUseId"]
    task = tool["input"]["task"]

    # Use the existing handle_coder_agent_tool function
    result = handle_coder_agent_tool(task)

    # Check if execution was successful based on the result string
    if "Error in coder agent tool" in result:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": result}]
        }
    else:
        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": result}]
        }
