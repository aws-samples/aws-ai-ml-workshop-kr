import pprint
import logging
import asyncio
from typing import Any, Annotated
from strands.types.tools import ToolResult, ToolUse
from src.utils.strands_sdk_utils import strands_utils
from src.prompts.template import apply_prompt_template
from src.utils.common_utils import get_message_from_string

from src.tools import python_repl_tool, bash_tool


# Initialize logger
logger = logging.getLogger(__name__)
logger.propagate = False  # 상위 로거로 메시지 전파 중지
for handler in logger.handlers[:]: 
    logger.removeHandler(handler)
handler = logging.StreamHandler()
formatter = logging.Formatter('\n%(levelname)s [%(name)s] %(message)s')  # 로그 레벨이 동적으로 표시되도록 변경
handler.setFormatter(formatter)
logger.addHandler(handler)
# DEBUG와 INFO 중 원하는 레벨로 설정
logger.setLevel(logging.INFO)  # DEBUG 이상 모든 로그 표시

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
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
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
    logger.info(f"{Colors.GREEN}===== Coder Agent Tool starting task ====={Colors.END}")
    
    #try:
        # Import tools here to avoid circular imports
        

        # Try to extract shared state from global storage (optional for context)
    from src.graph.nodes import _global_node_states
    shared_state = _global_node_states.get('shared', None)

    logger.info(f"{Colors.YELLOW}===== Successfully retrieved shared state from global storage ====={Colors.END}")
    logger.info(f"\n{Colors.YELLOW}Shared state:\n{pprint.pformat(shared_state, indent=2, width=100)}{Colors.END}") 
                    
    request_prompt, full_plan = shared_state.get("request_prompt", ""), shared_state.get("full_plan", "")
    clues, messages = shared_state.get("clues", ""), shared_state.get("messages", [])
    
    # Create coder agent with specialized tools using consistent pattern
    coder_agent = strands_utils.get_agent(
        agent_name="coder",
        system_prompts=apply_prompt_template(prompt_name="coder", prompt_context={"USER_REQUEST": request_prompt, "FULL_PLAN": full_plan}),
        agent_type="basic",  # coder uses basic LLM
        prompt_cache_info=(False, None),  # basic agent doesn't use prompt caching
        tools=[python_repl_tool, bash_tool],
        streaming=True  # Enable streaming for consistency
    )
    
    # Prepare message with context if available
    message = '\n\n'.join([messages[-1]["content"][-1]["text"], clues])
    
    # Import event queue for streaming
    from src.utils.event_queue import put_event
    
    # Process streaming response and send events to global queue
    async def process_coder_stream():
        streaming_events = []
        async for event in strands_utils.process_streaming_response_yield(coder_agent, message, agent_name="coder"):
            # Add source metadata to distinguish from supervisor events
            event["source"] = "coder_tool"
            event["tool_context"] = "coder_agent_tool"
            
            # Put event in global queue for main.py to process
            put_event(event)
            
            # Also print directly for immediate feedback
            #if event.get("event_type") == "text_chunk":
            #    print(f"\033[96m{event.get('data', '')}\033[0m", end="", flush=True)
            
            # Also keep for local processing
            streaming_events.append(event)
            
        # Reconstruct response from streaming events for return value
        response = {"text": ""}
        for event in streaming_events:
            if event.get("event_type") == "text_chunk":
                response["text"] += event.get("data", "")
        
        return coder_agent, response
    
    coder_agent, response = asyncio.run(process_coder_stream())
    
    # Extract text from response
    if isinstance(response, dict) and 'text' in response:
        result_text = response['text']
    elif hasattr(response, 'message') and 'content' in response.message:
        result_text = response.message['content'][-1]['text']
    else:
        result_text = str(response)
    
    logger.debug(f"\n{Colors.RED}Coder - current state messages:\n{pprint.pformat(shared_state.get('messages', []), indent=2, width=100)}{Colors.END}")
    logger.debug(f"\n{Colors.RED}Coder response:\n{pprint.pformat(response["text"], indent=2, width=100)}{Colors.END}")

    # Update clues
    clues = '\n\n'.join([clues, CLUES_FORMAT.format("coder", response["text"])])
    
    # Update history
    history = shared_state.get("history", [])
    history.append({"agent":"coder", "message": response["text"]})
    
    # Update shared state
    shared_state['messages'] = [get_message_from_string(role="user", string=RESPONSE_FORMAT.format("coder", response["text"]), imgs=[])]
    shared_state['clues'] = clues
    shared_state['history'] = history
    
    logger.info(f"{Colors.GREEN}===== Updated shared state with coder results ====={Colors.END}")
    logger.info(f"{Colors.GREEN}===== Coder Agent Tool completed successfully ====={Colors.END}")

    print ("result_text", result_text)

    return result_text
        
    #except Exception as e:
    #    error_msg = f"Error in coder agent tool: {str(e)}"
    #    logger.error(f"{Colors.RED}{error_msg}{Colors.END}")
    #    return error_msg

# Function name must match tool name
def coder_agent_tool(tool: ToolUse, **kwargs: Any) -> ToolResult:
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