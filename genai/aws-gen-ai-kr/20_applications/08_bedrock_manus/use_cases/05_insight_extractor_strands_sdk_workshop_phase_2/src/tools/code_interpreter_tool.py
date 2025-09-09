import json
import logging
from typing import Any, Annotated
from strands.types.tools import ToolResult, ToolUse
from bedrock_agentcore.tools.code_interpreter_client import code_session

# Simple logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TOOL_SPEC = {
    "name": "code_interpreter_tool",
    "description": "Execute Python code in AWS Bedrock Code Interpreter sandbox. Use this for secure code execution with data analysis capabilities.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute in the sandbox."
                },
                "description": {
                    "type": "string", 
                    "description": "Optional description of what the code does."
                }
            },
            "required": ["code"]
        }
    }
}

class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

def handle_code_interpreter_tool(
        code: Annotated[str, "The Python code to execute in the sandbox."], 
        description: Annotated[str, "Optional description of what the code does."] = ""
    ):
    """Execute Python code in AWS Bedrock Code Interpreter sandbox."""
    
    print()  # Add newline before log
    logger.info(f"{Colors.GREEN}===== Executing Code in Bedrock Sandbox ====={Colors.END}")
    
    try:
        if description: code = f"# {description}\n{code}"
        
        print(f"\n{Colors.YELLOW}Generated Code:{Colors.END}\n{code}\n")
        
        with code_session("us-west-2") as code_client:
            response = code_client.invoke("executeCode", {
                "code": code,
                "language": "python",
                "clearContext": False
            })
        
        # Process the response stream
        result = None
        for event in response["stream"]:
            result = event["result"]
            break
        
        if result:
            result_str = f"Successfully executed in Bedrock sandbox:\n||{code}||{json.dumps(result, indent=2)}"
            logger.info(f"{Colors.GREEN}===== Sandbox execution successful ====={Colors.END}")
            return result_str
        else:
            error_msg = "No result received from Bedrock Code Interpreter"
            logger.error(f"{Colors.RED}{error_msg}{Colors.END}")
            return f"Failed to execute. Error: {error_msg}"
            
    except Exception as e:
        error_msg = f"Failed to execute in Bedrock sandbox. Error: {repr(e)}"
        logger.error(f"{Colors.RED}{error_msg}{Colors.END}")
        return error_msg

# Function name must match tool name
def code_interpreter_tool(tool: ToolUse, **kwargs: Any) -> ToolResult:
    tool_use_id = tool["toolUseId"]
    code = tool["input"]["code"]
    description = tool["input"].get("description", "")
    
    # Use the existing handle function
    result = handle_code_interpreter_tool(code, description)
    
    # Check if execution was successful based on the result string
    if "Failed to execute" in result:
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

# Keep the original function for backward compatibility
def execute_python(code: str, description: str = "") -> str:
    """Execute Python code in the sandbox."""
    return handle_code_interpreter_tool(code, description)