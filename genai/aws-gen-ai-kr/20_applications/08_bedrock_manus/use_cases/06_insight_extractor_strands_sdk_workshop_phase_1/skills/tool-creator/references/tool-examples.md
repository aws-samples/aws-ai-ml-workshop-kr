# Tool Implementation Examples

This document provides complete, real-world examples of both tool types to serve as reference during tool creation.

## Example 1: Regular Tool - Bash Tool

**Purpose**: Execute bash commands for system operations

**File**: `src/tools/bash_tool.py`

```python
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
```

**Key Features:**
- `@log_io` decorator for automatic I/O logging
- Subprocess execution with error handling
- Color-coded console output
- Result format: `"cmd||output"`
- Test section in `__main__`

---

## Example 2: Regular Tool - Python REPL Tool

**Purpose**: Execute Python code for data analysis and calculations

**File**: `src/tools/python_repl_tool.py`

```python
import sys
import logging
import subprocess
from typing import Any, Annotated
from strands.types.tools import ToolResult, ToolUse
from src.tools.decorators import log_io


# Simple logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TOOL_SPEC = {
    "name": "python_repl_tool",
    "description": "Use this to execute python code and do data analysis or calculation. If you want to see the output of a value, you should print it out with `print(...)`. This is visible to the user.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The python code to execute to do further analysis or calculation."
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
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class PythonREPL:
    def __init__(self):
        pass

    def run(self, command):
        try:
            # Execute the command
            result = subprocess.run(
                [sys.executable, "-c", command],
                capture_output=True,
                text=True,
                timeout=600  # Timeout setting
            )
            # Return result
            if result.returncode == 0:
                return result.stdout
            else:
                return f"Error: {result.stderr}"
        except Exception as e:
            return f"Exception: {str(e)}"

repl = PythonREPL()

@log_io
def handle_python_repl_tool(code: Annotated[str, "The python code to execute to do further analysis or calculation."]):
    """
    Use this to execute python code and do data analysis or calculation. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user.
    """
    print()  # Add newline before log
    logger.info(f"{Colors.GREEN}===== Executing Python code ====={Colors.END}")
    try:
        result = repl.run(code)
    except BaseException as e:
        error_msg = f"Failed to execute. Error: {repr(e)}"
        logger.debug(f"{Colors.RED}Failed to execute. Error: {repr(e)}{Colors.END}")
        return error_msg

    # Truncate code to first 7 lines for context efficiency
    code_lines = code.split('\n')
    if len(code_lines) > 7:
        code_preview = '\n'.join(code_lines[:7])
        code_summary = f"{code_preview}\n... ({len(code_lines) - 7} more lines omitted)"
    else:
        code_summary = code

    result_str = f"Successfully executed:\n||{code_summary}||{result}"
    logger.info(f"{Colors.GREEN}===== Code execution successful ====={Colors.END}")
    return result_str

# Function name must match tool name
def python_repl_tool(tool: ToolUse, **kwargs: Any) -> ToolResult:
    tool_use_id = tool["toolUseId"]
    code = tool["input"]["code"]

    # Use the existing handle_python_repl_tool function
    result = handle_python_repl_tool(code)

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
```

**Key Features:**
- Dedicated `PythonREPL` class for code execution
- Code truncation for long outputs (context efficiency)
- Timeout protection (600 seconds)
- Result format: `"Successfully executed:\n||code||output"`

---

## Example 3: Agent-as-a-Tool - Coder Agent Tool

**Purpose**: Execute Python and bash commands using a specialized coder agent

**File**: `src/tools/coder_agent_tool.py`

```python
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
        agent_type="claude-sonnet-3-7",
        enable_reasoning=False,
        tools=[python_repl_tool, bash_tool],
        streaming=True
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
```

**Companion System Prompt**: `src/prompts/coder.md`

**Key Features:**
- Global state integration (`_global_node_states`)
- Streaming async pattern
- Context propagation (USER_REQUEST, FULL_PLAN)
- State updates (clues, history, messages)
- Sub-tools access (python_repl_tool, bash_tool)
- Response formatting with XML tags

---

## Example 4: Agent-as-a-Tool - Reporter Agent Tool

**Purpose**: Generate comprehensive reports using a specialized reporter agent

**File**: `src/tools/reporter_agent_tool.py`

```python
import logging
import asyncio
from typing import Any, Annotated
from strands.types.tools import ToolResult, ToolUse
from src.utils.strands_sdk_utils import strands_utils
from src.prompts.template import apply_prompt_template
from src.utils.common_utils import get_message_from_string

from src.tools import python_repl_tool, bash_tool
from strands_tools import file_read

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

    Args:
        task: The reporting task or instruction for generating the report

    Returns:
        The generated report content and confirmation of file creation
    """
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

    # Create reporter agent with specialized tools
    reporter_agent = strands_utils.get_agent(
        agent_name="reporter",
        system_prompts=apply_prompt_template(prompt_name="reporter", prompt_context={"USER_REQUEST": request_prompt, "FULL_PLAN": full_plan}),
        agent_type="claude-sonnet-3-7",
        enable_reasoning=False,
        prompt_cache_info=(True, None),
        tools=[python_repl_tool, bash_tool, file_read],
        streaming=True
    )

    # Prepare message with context
    message = '\n\n'.join([messages[-1]["content"][-1]["text"], clues])

    # Process streaming response
    async def process_reporter_stream():
        full_text = ""
        async for event in strands_utils.process_streaming_response_yield(
            reporter_agent, message, agent_name="reporter", source="reporter_tool"
        ):
            if event.get("event_type") == "text_chunk": full_text += event.get("data", "")
        return {"text": full_text}

    response = asyncio.run(process_reporter_stream())
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
    return result_text

# Function name must match tool name
def reporter_agent_tool(tool: ToolUse, **_kwargs: Any) -> ToolResult:
    tool_use_id = tool["toolUseId"]
    task = tool["input"]["task"]

    # Use the existing handle function
    result = handle_reporter_agent_tool(task)

    # Check if execution was successful
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
```

**Key Features:**
- Prompt caching enabled: `prompt_cache_info=(True, None)`
- External tool integration: `file_read` from strands_tools
- Same state management pattern as coder
- Domain-specific tools (python, bash, file_read)

---

## Pattern Comparison

### Regular Tool Pattern

**Structure:**
1. TOOL_SPEC definition
2. Colors class for logging
3. `@log_io` decorated handler function
4. Direct operation (subprocess, API call, etc.)
5. ToolResult wrapper function
6. Optional `__main__` test section

**When to use:**
- Simple, direct operations
- No need for reasoning or multi-step logic
- Deterministic behavior
- Fast execution

### Agent Tool Pattern

**Structure:**
1. TOOL_SPEC definition
2. Response format constants (RESPONSE_FORMAT, CLUES_FORMAT)
3. Colors class for logging
4. Handler function with:
   - Global state extraction
   - Agent creation with `strands_utils.get_agent()`
   - Context preparation
   - Async streaming processing
   - State updates (clues, history, messages)
5. ToolResult wrapper function

**When to use:**
- Complex, multi-step operations
- Requires reasoning or decision-making
- Needs access to other tools
- Domain-specific expertise

---

## Common Code Patterns

### Pattern 1: Global State Access (Agent Tools)

```python
from src.graph.nodes import _global_node_states
shared_state = _global_node_states.get('shared', None)

if not shared_state:
    logger.warning("No shared state found")
    return "Error: No shared state available"

request_prompt = shared_state.get("request_prompt", "")
full_plan = shared_state.get("full_plan", "")
clues = shared_state.get("clues", "")
messages = shared_state.get("messages", [])
```

### Pattern 2: Agent Creation (Agent Tools)

```python
agent = strands_utils.get_agent(
    agent_name="agent_name",
    system_prompts=apply_prompt_template(
        prompt_name="agent_name",
        prompt_context={"USER_REQUEST": request_prompt, "FULL_PLAN": full_plan}
    ),
    agent_type="claude-sonnet-3-7",
    enable_reasoning=False,
    prompt_cache_info=(True, None),  # Optional
    tools=[tool1, tool2],
    streaming=True
)
```

### Pattern 3: Streaming Processing (Agent Tools)

```python
async def process_stream():
    full_text = ""
    async for event in strands_utils.process_streaming_response_yield(
        agent, message, agent_name="name", source="source"
    ):
        if event.get("event_type") == "text_chunk":
            full_text += event.get("data", "")
    return {"text": full_text}

response = asyncio.run(process_stream())
```

### Pattern 4: State Update (Agent Tools)

```python
# Update clues
clues = '\n\n'.join([clues, CLUES_FORMAT.format("agent_name", response["text"])])

# Update history
history = shared_state.get("history", [])
history.append({"agent": "agent_name", "message": response["text"]})

# Update shared state
shared_state['messages'] = [get_message_from_string(
    role="user",
    string=RESPONSE_FORMAT.format("agent_name", response["text"]),
    imgs=[]
)]
shared_state['clues'] = clues
shared_state['history'] = history
```

### Pattern 5: Error Handling (Both Types)

```python
try:
    # Operation
    result = perform_operation()
    return result
except SpecificException as e:
    error_message = f"Error: {str(e)}"
    logger.error(f"{Colors.RED}Error: {str(e)}{Colors.END}")
    return error_message
except Exception as e:
    error_message = f"Unexpected error: {str(e)}"
    logger.error(f"{Colors.RED}Error: {str(e)}{Colors.END}")
    return error_message
```

---

## Usage in Agent Systems

### Importing Tools

```python
# Regular tools
from src.tools.bash_tool import bash_tool
from src.tools.python_repl_tool import python_repl_tool

# Agent tools
from src.tools.coder_agent_tool import coder_agent_tool
from src.tools.reporter_agent_tool import reporter_agent_tool
```

### Using in Agent Creation

```python
# Supervisor with agent tools
supervisor_agent = strands_utils.get_agent(
    agent_name="supervisor",
    system_prompts=supervisor_prompt,
    tools=[coder_agent_tool, reporter_agent_tool],
    streaming=True
)

# Coder agent with regular tools
coder_agent = strands_utils.get_agent(
    agent_name="coder",
    system_prompts=coder_prompt,
    tools=[python_repl_tool, bash_tool],
    streaming=True
)
```

---

## Key Takeaways

1. **Regular tools** are simpler, faster, and deterministic
2. **Agent tools** are more powerful but more complex
3. Both follow consistent patterns for integration
4. State management is critical for agent tools
5. Streaming is the standard for agent interactions
6. Error handling should be graceful and informative
7. Logging provides visibility into tool execution
8. Templates variables must be properly escaped in system prompts
