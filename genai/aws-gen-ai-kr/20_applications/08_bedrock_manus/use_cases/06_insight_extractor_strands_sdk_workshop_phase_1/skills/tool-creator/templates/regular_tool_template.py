import logging
from typing import Any, Annotated
from strands.types.tools import ToolResult, ToolUse
from src.tools.decorators import log_io


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
                {{INPUT_PROPERTIES}}
            },
            "required": {{REQUIRED_FIELDS}}
        }
    }
}

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    END = '\033[0m'

@log_io
def handle_{{TOOL_NAME}}({{FUNCTION_PARAMETERS}}):
    """{{TOOL_DESCRIPTION}}"""

    print()  # Add newline before log
    logger.info(f"\n{{Colors.GREEN}}Executing {{TOOL_NAME}}{{Colors.END}}")

    try:
        {{IMPLEMENTATION_LOGIC}}

        logger.info(f"{{Colors.GREEN}}{{TOOL_NAME}} completed successfully{{Colors.END}}")
        return result

    except Exception as e:
        error_message = f"Error in {{TOOL_NAME}}: {{str(e)}}"
        logger.error(f"{{Colors.RED}}Error: {{str(e)}}{{Colors.END}}")
        return error_message

# Function name must match tool name
def {{TOOL_NAME}}(tool: ToolUse, **_kwargs: Any) -> ToolResult:
    tool_use_id = tool["toolUseId"]
    {{EXTRACT_INPUTS}}

    # Use the existing handle function
    result = handle_{{TOOL_NAME}}({{CALL_PARAMETERS}})

    # Check if execution was successful
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

if __name__ == "__main__":
    # Test example
    print(handle_{{TOOL_NAME}}({{TEST_PARAMETERS}}))
