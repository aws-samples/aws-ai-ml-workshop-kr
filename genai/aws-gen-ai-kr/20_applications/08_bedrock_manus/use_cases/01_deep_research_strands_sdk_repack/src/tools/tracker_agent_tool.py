import logging
import asyncio
from typing import Any, Annotated
from strands.types.tools import ToolResult, ToolUse
from src.utils.strands_sdk_utils import strands_utils
from src.prompts.template import apply_prompt_template
from src.utils.common_utils import get_message_from_string

# Simple logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TOOL_SPEC = {
    "name": "tracker_agent_tool",
    "description": "Track and update task completion status based on agent results. This tool monitors progress and updates checklists from [ ] to [x] format based on completed work.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "completed_agent": {
                    "type": "string",
                    "description": "The name of the agent that just completed its task (e.g., 'coder', 'reporter')."
                },
                "completion_summary": {
                    "type": "string", 
                    "description": "Summary of what was completed by the agent to help identify which tasks to mark as done."
                }
            },
            "required": ["completed_agent", "completion_summary"]
        }
    }
}

RESPONSE_FORMAT = "Updated task tracking from {}:\n\n<tracking_update>\n{}\n</tracking_update>\n\n*Task status has been updated.*"
CLUES_FORMAT = "Here is updated tracking status:\n\n<tracking_clues>\n{}\n</tracking_clues>\n\n"

class Colors:
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    END = '\033[0m'

def handle_tracker_agent_tool(completed_agent: Annotated[str, "The name of the agent that just completed its task"], 
                             completion_summary: Annotated[str, "Summary of what was completed by the agent"]):
    """
    Track and update task completion status based on agent results.
    
    This tool provides access to a tracker agent that can:
    - Monitor task progress and update completion status
    - Update checklists from [ ] to [x] format
    - Maintain accurate task tracking
    
    Args:
        completed_agent: The name of the agent that just completed its task
        completion_summary: Summary of what was completed by the agent
        
    Returns:
        Updated task tracking status
    """
    print()  # Add newline before log
    logger.info(f"\n{Colors.GREEN}Tracker Agent Tool starting{Colors.END}")
    
    # Try to extract shared state from global storage
    from src.graph.nodes import _global_node_states
    shared_state = _global_node_states.get('shared', None)
    
    if not shared_state:
        logger.warning("No shared state found")
        return "Error: No shared state available" 
                    
    request_prompt = shared_state.get("request_prompt", "")
    full_plan = shared_state.get("full_plan", "")
    clues = shared_state.get("clues", "")
    messages = shared_state.get("messages", [])
    
    # Create tracker agent - uses reasoning LLM like planner and supervisor
    tracker_agent = strands_utils.get_agent(
        agent_name="tracker",
        system_prompts=apply_prompt_template(
            prompt_name="tracker",
            prompt_context={
                "USER_REQUEST": request_prompt,
                "FULL_PLAN": full_plan
            }
        ),
        agent_type="claude-sonnet-4-5", # claude-sonnet-3-5-v-2, claude-sonnet-3-7
        enable_reasoning=False,
        prompt_cache_info=(True, None),  # reasoning agent uses prompt caching
        tools=[],  # tracker doesn't need additional tools
        streaming=True
    )
    
    # Prepare tracking message with context
    tracking_message = f"Agent '{completed_agent}' has completed its task. Here's what was accomplished:\n\n{completion_summary}\n\nPlease update the task completion status accordingly."
    
    # Add context from previous messages and clues if available
    if messages:
        tracking_message = '\n\n'.join([messages[-1]["content"][-1]["text"], clues, tracking_message])
    
    # Process streaming response and collect text in one pass
    async def process_tracker_stream():
        full_text = ""
        async for event in strands_utils.process_streaming_response_yield(
            tracker_agent, tracking_message, agent_name="tracker", source="tracker_tool"
        ):
            if event.get("event_type") == "text_chunk": 
                full_text += event.get("data", "")
        return {"text": full_text}
    
    response = asyncio.run(process_tracker_stream())
    
    result_text = response['text']
    
    # Update clues with tracking information
    clues = '\n\n'.join([clues, CLUES_FORMAT.format(response["text"])])
    
    # Update history
    history = shared_state.get("history", [])
    history.append({"agent": "tracker", "message": response["text"]})
    
    # Update shared state with tracking results
    shared_state['messages'] = [get_message_from_string(role="user", string=RESPONSE_FORMAT.format("tracker", response["text"]), imgs=[])]
    shared_state['clues'] = clues
    shared_state['history'] = history
    
    # Update the full_plan with the tracked version if the response contains an updated plan
    if "# Plan" in response["text"]:
        shared_state['full_plan'] = response["text"]
        logger.info(f"{Colors.BLUE}Updated full_plan with tracking results{Colors.END}")
    
    logger.info(f"\n{Colors.GREEN}Tracker Agent Tool completed{Colors.END}")

    return result_text

# Function name must match tool name
def tracker_agent_tool(tool: ToolUse, **_kwargs: Any) -> ToolResult:
    tool_use_id = tool["toolUseId"]
    completed_agent = tool["input"]["completed_agent"]
    completion_summary = tool["input"]["completion_summary"]
    
    # Use the existing handle_tracker_agent_tool function
    result = handle_tracker_agent_tool(completed_agent, completion_summary)
    
    # Check if execution was successful based on the result string
    if "Error" in result:
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