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
    "name": "reporter_agent_tool",
    "description": "Generate comprehensive reports based on analysis results using a specialized reporter agent. This tool provides access to a reporter agent that can read analysis results from artifacts, create structured reports with visualizations, and generate output in multiple formats (HTML, PDF, Markdown).",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The reporting task or instruction for generating the report (e.g., 'Create a comprehensive analysis report', 'Generate PDF report with all findings')."
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

def handle_reporter_agent_tool(task: Annotated[str, "The reporting task or instruction for generating the report."]):
    """
    Generate comprehensive reports based on analysis results using a specialized reporter agent.
    
    This tool provides access to a reporter agent that can:
    - Read analysis results from artifacts directory
    - Create structured reports with executive summaries, key findings, and detailed analysis
    - Generate reports in multiple formats (HTML, PDF, Markdown)
    - Include visualizations and charts in reports
    - Process accumulated analysis results from all_results.txt
    
    Args:
        task: The reporting task or instruction for generating the report
        
    Returns:
        The generated report content and confirmation of file creation
    """
    logger.info(f"{Colors.GREEN}===== Reporter Agent Tool starting task ====={Colors.END}")
    
    #try:
    # Try to extract shared state from global storage (optional for context)
    from src.graph.nodes import _global_node_states
    shared_state = _global_node_states.get('shared', None)

    logger.info(f"{Colors.YELLOW}===== Successfully retrieved shared state from global storage ====={Colors.END}")
    logger.info(f"\n{Colors.YELLOW}Shared state:\n{pprint.pformat(shared_state, indent=2, width=100)}{Colors.END}") 
                    
    request_prompt, full_plan = shared_state.get("request_prompt", ""), shared_state.get("full_plan", "")
    clues, messages = shared_state.get("clues", ""), shared_state.get("messages", [])
    
    # Create reporter agent with specialized tools using consistent pattern
    reporter_agent = strands_utils.get_agent(
        agent_name="reporter",
        system_prompts=apply_prompt_template(prompt_name="reporter", prompt_context={"USER_REQUEST": request_prompt, "FULL_PLAN": full_plan}),
        agent_type="reasoning",  # reporter uses reasoning LLM
        prompt_cache_info=(True, None),  # reasoning agent uses prompt caching
        tools=[python_repl_tool, bash_tool],
        streaming=True  # Enable streaming for consistency
    )
    
    # Prepare message with context if available
    
    print ("messages", messages)
    print ("==")
    print ('\n\n'.join([messages[-1]["content"][-1]["text"], clues]))
    message = '\n\n'.join([messages[-1]["content"][-1]["text"], clues])


    

    reporter_agent, response = asyncio.run(strands_utils.process_streaming_response(reporter_agent, message))
    
    # Extract text from response
    if isinstance(response, dict) and 'text' in response:
        result_text = response['text']
    elif hasattr(response, 'message') and 'content' in response.message:
        result_text = response.message['content'][-1]['text']
    else:
        result_text = str(response)
    
    logger.debug(f"\n{Colors.RED}Reporter - current state messages:\n{pprint.pformat(shared_state.get('messages', []), indent=2, width=100)}{Colors.END}")
    logger.debug(f"\n{Colors.RED}Reporter response:\n{pprint.pformat(response["text"], indent=2, width=100)}{Colors.END}")

    # Update clues
    clues = '\n\n'.join([clues, CLUES_FORMAT.format("reporter", response["text"])])
    
    # Update history
    history = shared_state.get("history", [])
    history.append({"agent":"reporter", "message": response["text"]})
    
    # Update shared state
    shared_state['messages'] = [get_message_from_string(role="user", string=RESPONSE_FORMAT.format("reporter", response["text"]), imgs=[])]
    shared_state['clues'] = clues
    shared_state['history'] = history
    
    logger.info(f"{Colors.GREEN}===== Updated shared state with reporter results ====={Colors.END}")
    logger.info(f"{Colors.GREEN}===== Reporter Agent Tool completed successfully ====={Colors.END}")
    return result_text
        
    #except Exception as e:
    #    error_msg = f"Error in reporter agent tool: {str(e)}"
    #    logger.error(f"{Colors.RED}{error_msg}{Colors.END}")
    #    return error_msg

# Function name must match tool name
def reporter_agent_tool(tool: ToolUse, **kwargs: Any) -> ToolResult:
    tool_use_id = tool["toolUseId"]
    task = tool["input"]["task"]
    
    # Use the existing handle_reporter_agent_tool function
    result = handle_reporter_agent_tool(task)
    
    # Check if execution was successful based on the result string
    if "Error in reporter agent tool" in result:
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