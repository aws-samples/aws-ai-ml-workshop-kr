import logging
import asyncio
from typing import Any, Annotated
from strands.types.tools import ToolResult, ToolUse
from src.utils.strands_sdk_utils import strands_utils
from src.prompts.template import apply_prompt_template
from src.utils.common_utils import get_message_from_string
{{TOOL_IMPORTS}}


# Simple logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TOOL_SPEC = {
    "name": "{{TOOL_NAME}}",
    "description": "{{TOOL_DESCRIPTION}}",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "{{TASK_DESCRIPTION}}"
                }
            },
            "required": ["task"]
        }
    }
}

RESPONSE_FORMAT = "Response from {{AGENT_NAME}}:\n\n<response>\n{{}}\n</response>\n\n*Please execute the next step.*"
CLUES_FORMAT = "Here is clues from {{AGENT_NAME}}:\n\n<clues>\n{{}}\n</clues>\n\n"

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    END = '\033[0m'

def handle_{{TOOL_NAME}}(task: Annotated[str, "{{TASK_DESCRIPTION}}"]):
    """
    {{TOOL_DESCRIPTION}}

    Args:
        task: {{TASK_DESCRIPTION}}

    Returns:
        The result of the agent's execution
    """
    print()  # Add newline before log
    logger.info(f"\n{{Colors.GREEN}}{{AGENT_NAME}} Agent Tool starting task{{Colors.END}}")

    # Try to extract shared state from global storage
    from src.graph.nodes import _global_node_states
    shared_state = _global_node_states.get('shared', None)

    if not shared_state:
        logger.warning("No shared state found")
        return "Error: No shared state available"

    request_prompt, full_plan = shared_state.get("request_prompt", ""), shared_state.get("full_plan", "")
    clues, messages = shared_state.get("clues", ""), shared_state.get("messages", [])

    # Create agent with specialized tools
    agent = strands_utils.get_agent(
        agent_name="{{AGENT_NAME}}",
        system_prompts=apply_prompt_template(prompt_name="{{AGENT_NAME}}", prompt_context={{"USER_REQUEST": request_prompt, "FULL_PLAN": full_plan}}),
        agent_type="{{MODEL_TYPE}}",
        enable_reasoning={{ENABLE_REASONING}},
        prompt_cache_info={{PROMPT_CACHE_INFO}},
        tools=[{{AGENT_TOOLS}}],
        streaming=True
    )

    # Prepare message with context
    message = '\n\n'.join([messages[-1]["content"][-1]["text"], clues])

    # Process streaming response
    async def process_stream():
        full_text = ""
        async for event in strands_utils.process_streaming_response_yield(
            agent, message, agent_name="{{AGENT_NAME}}", source="{{TOOL_NAME}}"
        ):
            if event.get("event_type") == "text_chunk":
                full_text += event.get("data", "")
        return {{"text": full_text}}

    response = asyncio.run(process_stream())
    result_text = response['text']

    # Update clues
    clues = '\n\n'.join([clues, CLUES_FORMAT.format("{{AGENT_NAME}}", response["text"])])

    # Update history
    history = shared_state.get("history", [])
    history.append({{"agent": "{{AGENT_NAME}}", "message": response["text"]}})

    # Update shared state
    shared_state['messages'] = [get_message_from_string(role="user", string=RESPONSE_FORMAT.format("{{AGENT_NAME}}", response["text"]), imgs=[])]
    shared_state['clues'] = clues
    shared_state['history'] = history

    logger.info(f"\n{{Colors.GREEN}}{{AGENT_NAME}} Agent Tool completed successfully{{Colors.END}}")
    return result_text

# Function name must match tool name
def {{TOOL_NAME}}(tool: ToolUse, **_kwargs: Any) -> ToolResult:
    tool_use_id = tool["toolUseId"]
    task = tool["input"]["task"]

    # Use the existing handle function
    result = handle_{{TOOL_NAME}}(task)

    # Check if execution was successful
    if "Error in {{TOOL_NAME}}" in result:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{{"text": result}}]
        }
    else:
        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{{"text": result}}]
        }
