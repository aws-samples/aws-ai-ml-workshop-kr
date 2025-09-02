import os
import logging
import subprocess
from typing import Any, Annotated
from strands.types.tools import ToolResult, ToolUse
from src.tools.decorators import log_io

# Observability
from dotenv import load_dotenv
from opentelemetry import trace
from src.utils.agentcore_observability import add_span_event
load_dotenv()

# Simple logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TOOL_SPEC = {
    "name": "bash_tool",
    "description": "Use this to execute bash command and do necessary operations.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "cmd": {
                    "type": "string",
                    "description": "The bash command to be executed."
                }
            },
            "required": ["cmd"]
        }
    }
}

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    END = '\033[0m'

@log_io
def handle_bash_tool(cmd: Annotated[str, "The bash command to be executed."]):
    """Use this to execute bash command and do necessary operations."""

    tracer = trace.get_tracer(
        instrumenting_module_name=os.getenv("TRACER_MODULE_NAME", "insight_extractor_agent"),
        instrumenting_library_version=os.getenv("TRACER_LIBRARY_VERSION", "1.0.0")
    )
    with tracer.start_as_current_span("bash_tool") as span:
        print()  # Add newline before log
        logger.info(f"\n{Colors.GREEN}Executing Bash: {cmd}{Colors.END}")
        try:
            # Execute the command and capture output
            result = subprocess.run(
                cmd, shell=True, check=True, text=True, capture_output=True
            )
            # Return stdout as the result
            results = "||".join([cmd, result.stdout])

            # Add Event
            add_span_event(span, "command", {"cmd": str(cmd)})
            add_span_event(span, "result", {"response": str(result.stdout)})

            return results + "\n"
            
        except subprocess.CalledProcessError as e:
            # If command fails, return error information
            error_message = f"Command failed with exit code {e.returncode}.\nStdout: {e.stdout}\nStderr: {e.stderr}"
            logger.error(f"{Colors.RED}Command failed: {e.returncode}{Colors.END}")

            # Add Event
            add_span_event(span, "code", {"code": str(cmd)})
            add_span_event(span, "result", {"response": repr(e)})

            return error_message
        
        except Exception as e:
            # Catch any other exceptions
            error_message = f"Error executing command: {str(e)}"
            logger.error(f"{Colors.RED}Error: {str(e)}{Colors.END}")

            # Add Event
            add_span_event(span, "code", {"code": str(cmd)})
            add_span_event(span, "result", {"response": repr(e)})

            return error_message

# Function name must match tool name
def bash_tool(tool: ToolUse, **_kwargs: Any) -> ToolResult:
    tool_use_id = tool["toolUseId"]
    cmd = tool["input"]["cmd"]
    
    # Use the existing handle_bash_tool function
    result = handle_bash_tool(cmd)
    
    # Check if execution was successful based on the result string
    if "Command failed" in result or "Error executing command" in result:
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

if __name__ == "__main__":
    # Test example using the handle_bash_tool function directly
    print(handle_bash_tool("ls -all"))