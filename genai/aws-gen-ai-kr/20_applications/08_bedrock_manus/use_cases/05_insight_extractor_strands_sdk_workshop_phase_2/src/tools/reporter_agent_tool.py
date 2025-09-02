import os
import logging
import asyncio
from typing import Any, Annotated
from strands.types.tools import ToolResult, ToolUse
from src.utils.strands_sdk_utils import strands_utils
from src.prompts.template import apply_prompt_template
from src.utils.common_utils import get_message_from_string

from src.tools import python_repl_tool, bash_tool
from strands_tools import file_read

# Observability
from dotenv import load_dotenv
from opentelemetry import trace
from src.utils.agentcore_observability import add_span_event
load_dotenv()

# Simple logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
    GREEN = '\033[92m'
    END = '\033[0m'

def handle_reporter_agent_tool(_task: Annotated[str, "The reporting task or instruction for generating the report."]):
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
    tracer = trace.get_tracer(
        instrumenting_module_name=os.getenv("TRACER_MODULE_NAME", "insight_extractor_agent"),
        instrumenting_library_version=os.getenv("TRACER_LIBRARY_VERSION", "1.0.0")
    )
    with tracer.start_as_current_span("reporter_agent_tool") as span:
        print()  # Add newline before log
        logger.info(f"\n{Colors.GREEN}Reporter Agent Tool starting{Colors.END}")
        
        # Try to extract shared state from global storage
        from src.graph.nodes import _global_node_states
        shared_state = _global_node_states.get('shared', None)
        
        if not shared_state:
            logger.warning("No shared state found")
            return "Error: No shared state available" 
                        
        request_prompt, full_plan = shared_state.get("request_prompt", ""), shared_state.get("full_plan", "")
        clues, messages = shared_state.get("clues", ""), shared_state.get("messages", [])
        
        # Create reporter agent with specialized tools using consistent pattern
        reporter_agent = strands_utils.get_agent(
            agent_name="reporter",
            system_prompts=apply_prompt_template(prompt_name="reporter", prompt_context={"USER_REQUEST": request_prompt, "FULL_PLAN": full_plan}),
            agent_type="claude-sonnet-3-7", # claude-sonnet-3-5-v-2, claude-sonnet-3-7
            enable_reasoning=False,
            prompt_cache_info=(True, None), # reasoning agent uses prompt caching
            tools=[python_repl_tool, bash_tool, file_read],
            streaming=True  # Enable streaming for consistency
        )
        
        # Prepare message with context if available
        message = '\n\n'.join([messages[-1]["content"][-1]["text"], clues])
        
        # Process streaming response
        async def process_reporter_stream():
            streaming_events = []
            async for event in strands_utils.process_streaming_response_yield(
                reporter_agent, message, agent_name="reporter", source="reporter_tool"
            ):
                streaming_events.append(event)
                
            # Reconstruct response from streaming events for return value
            response = {"text": ""}
            for event in streaming_events:
                if event.get("event_type") == "text_chunk":
                    response["text"] += event.get("data", "")
            
            return reporter_agent, response
        
        reporter_agent, response = asyncio.run(process_reporter_stream())
        result_text = response['text']
        
        # Update clues
        clues = '\n\n'.join([clues, CLUES_FORMAT.format("reporter", response["text"])])
        
        # Update history
        history = shared_state.get("history", [])
        history.append({"agent":"reporter", "message": response["text"]})
        
        # Update shared state
        shared_state['messages'] = [get_message_from_string(role="user", string=RESPONSE_FORMAT.format("reporter", response["text"]), imgs=[])]
        shared_state['clues'] = clues
        shared_state['history'] = history
        
        logger.info(f"\n{Colors.GREEN}Reporter Agent Tool completed{Colors.END}")

        # Add Event
        add_span_event(span, "input_message", {"message": str(request_prompt)})
        add_span_event(span, "response", {"response": str(response["text"])})

        return result_text

# Function name must match tool name
def reporter_agent_tool(tool: ToolUse, **_kwargs: Any) -> ToolResult:
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