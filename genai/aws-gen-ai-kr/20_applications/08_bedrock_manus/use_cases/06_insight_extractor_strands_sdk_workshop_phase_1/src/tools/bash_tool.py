import logging
import subprocess
from typing import Any, Annotated
from strands.types.tools import ToolResult, ToolUse
from src.tools.decorators import log_io


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

    print()  # Add newline before log
    logger.info(f"\n{Colors.GREEN}Executing Bash: {cmd}{Colors.END}")
    try:
        # Execute the command and capture output
        result = subprocess.run(
            cmd, shell=True, check=True, text=True, capture_output=True
        )
        # Return stdout as the result
        results = "||".join([cmd, result.stdout])
        return results + "\n"

    except subprocess.CalledProcessError as e:
        # If command fails, return error information
        error_message = f"Command failed with exit code {e.returncode}.\nStdout: {e.stdout}\nStderr: {e.stderr}"
        logger.error(f"{Colors.RED}Command failed: {e.returncode}{Colors.END}")
        return error_message

    except Exception as e:
        # Catch any other exceptions
        error_message = f"Error executing command: {str(e)}"
        logger.error(f"{Colors.RED}Error: {str(e)}{Colors.END}")
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