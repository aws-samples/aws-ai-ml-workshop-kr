---
name: tool-creator
description: This skill should be used when users want to create a new tool for the Strands SDK agent system. It supports creating both agent-as-a-tool (complex agents wrapped as tools) and regular tools (simple function-based tools). Use this skill when users request to create, build, or add a new tool.
---

# Tool Creator Skill

This skill provides comprehensive guidance for creating effective tools for the Strands SDK-based agent system. It supports two types of tools: **Agent-as-a-Tool** (agents wrapped as tools) and **Regular Tools** (function-based tools).

## About Tools in This System

Tools extend agent capabilities by providing:
1. **Agent-as-a-Tool**: Specialized agents with their own prompts, models, and sub-tools
2. **Regular Tools**: Direct function execution for system operations, API calls, or data processing

### Tool Anatomy

Every tool in `src/tools/` consists of:

```python
# Required components
TOOL_SPEC = {
    "name": "tool_name",
    "description": "What the tool does",
    "inputSchema": {"json": {...}}
}

def handle_tool_name(param: Annotated[type, "description"]):
    """Implementation logic"""
    pass

def tool_name(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Strands SDK tool wrapper"""
    pass
```

## Tool Creation Process

Follow these steps to create a tool. The process supports both full specification upfront and interactive information gathering.

### Step 1: Determine Tool Type

**Automatic Detection:**
- If user mentions "agent tool", "agent-as-a-tool", or describes complex multi-step operations → Agent-as-a-Tool
- If user mentions "simple tool", "regular tool", or describes direct operations → Regular Tool
- If ambiguous → Ask user

**Question to ask if ambiguous:**
```
Which type of tool would you like to create?

1. **Agent-as-a-Tool**: A specialized agent with its own prompt, model, and sub-tools (e.g., coder_agent_tool, reporter_agent_tool)
   - Use when: Complex reasoning, multi-step operations, or domain expertise needed

2. **Regular Tool**: A simple function-based tool (e.g., bash_tool, python_repl_tool)
   - Use when: Direct operations like API calls, file operations, or system commands
```

### Step 2: Gather Basic Tool Information

Collect the following information. If user provided some details already, only ask for missing information.

**Required for Both Types:**

1. **Tool Name** (if not provided)
   - Question: "What should the tool be named? (Use snake_case, e.g., 'data_analyzer_tool')"
   - Validation: Must end with '_tool', use snake_case

2. **Tool Description** (if not provided)
   - Question: "What does this tool do? Provide a clear description of its purpose and capabilities."
   - This becomes the tool's description field that helps other agents decide when to use it

3. **Input Parameters** (if not provided)
   - Question: "What input parameters does this tool need?"
   - For agent tools, typically: `task` (string describing what to do)
   - For regular tools: specific parameters (e.g., `cmd` for bash, `code` for python)

### Step 3: Gather Type-Specific Information

#### For Regular Tools:

Collect these details (skip if already provided):

1. **Implementation Logic**
   - Question: "What operation should this tool perform? (e.g., execute subprocess, call API, read file)"
   - Common patterns: subprocess execution, HTTP requests, file operations, data transformations

2. **Error Handling**
   - Question: "What errors should be handled? (Default: try/except with error logging)"

3. **External Dependencies** (optional)
   - Question: "Does this tool require external libraries? If yes, which ones?"

#### For Agent-as-a-Tool:

Collect these details (skip if already provided):

1. **Agent's Purpose and Role**
   - Question: "What is the agent's primary purpose? What role does it play in the system?"
   - This informs the system prompt creation

2. **Agent Model Type**
   - Question: "Which LLM model should the agent use?"
   - Options:
     - `claude-sonnet-3-7` (recommended for most tasks)
     - `claude-sonnet-4` (advanced reasoning)
     - `claude-sonnet-3-5-v-2` (legacy)

3. **Reasoning Capability**
   - Question: "Should this agent use extended thinking/reasoning? (True/False)"
   - Default: False
   - Use True for: complex analysis, planning, strategic decisions

4. **Prompt Caching**
   - Question: "Should prompt caching be enabled? (Recommended: True for agents called frequently)"
   - Default: (True, None)

5. **Sub-tools** (if not provided)
   - Question: "Which tools should this agent have access to?"
   - Common options: `python_repl_tool`, `bash_tool`, `file_read`
   - Reference existing tools in `src/tools/`

6. **System Prompt Creation**
   - **IMPORTANT**: For system prompt creation, refer to `references/system-prompt-guidelines.md`
   - If user hasn't provided a system prompt, ask: "Do you want to create a custom system prompt for this agent?"
   - If yes: Use system-prompt-writer guidelines from references to create an effective prompt
   - If no: Create a basic prompt based on the agent's purpose

### Step 4: Create the Tool File

Generate the tool file in `src/tools/` using the appropriate template:

- **Regular Tool**: Use `templates/regular_tool_template.py`
- **Agent-as-a-Tool**: Use `templates/agent_tool_template.py`

**File Creation Steps:**

1. Load the appropriate template
2. Replace template variables with gathered information
3. If creating system prompt:
   - Create prompt file in `src/prompts/[tool_name_without_tool].md`
   - Follow system-prompt-writer guidelines from `references/system-prompt-guidelines.md`
   - Use proper template variable escaping (double braces `{{}}` for code samples)
4. Write the tool file to `src/tools/[tool_name].py`
5. Inform user of file locations

### Step 5: Validation and Next Steps

After creating the tool:

1. **Verify File Creation**
   - Confirm tool file exists at `src/tools/[tool_name].py`
   - If agent tool with prompt, confirm prompt file at `src/prompts/[name].md`

2. **Integration Guidance**
   - Inform user how to import and use the new tool:
     ```python
     from src.tools.[tool_name] import [tool_name]

     # Use in agent
     agent = strands_utils.get_agent(
         agent_name="example",
         tools=[tool_name, other_tool],
         ...
     )
     ```

3. **Testing Recommendations**
   - Suggest testing the tool in isolation
   - For agent tools: Test with sample tasks
   - For regular tools: Test with sample inputs

## Key Design Principles

### For All Tools

1. **Clear Naming**: Tool names should be descriptive and end with `_tool`
2. **Comprehensive Descriptions**: Description should clearly state what the tool does and when to use it
3. **Annotated Parameters**: Use `Annotated[type, "description"]` for all parameters
4. **Consistent Error Handling**: Return error messages, don't raise exceptions
5. **Logging**: Use color-coded logging for visibility

### For Agent-as-a-Tool

1. **Global State Integration**: Always access `_global_node_states` for shared context
2. **Streaming Support**: Use async streaming pattern with `process_streaming_response_yield`
3. **State Updates**: Update clues, history, and messages in shared state
4. **Response Format**: Use standard response format templates
5. **Prompt Templates**: Use `apply_prompt_template()` with proper context variables

### For Regular Tools

1. **Simplicity**: Keep logic straightforward and focused
2. **Decorator Usage**: Use `@log_io` decorator for input/output logging
3. **Subprocess Safety**: Set timeouts and handle errors for subprocess calls
4. **Result Formatting**: Return results in consistent format (e.g., `"cmd||output"`)

## Common Patterns

### Pattern 1: Agent Tool with Analysis Capabilities
```python
# Agent for data analysis tasks
- Model: claude-sonnet-3-7
- Reasoning: False
- Tools: [python_repl_tool, bash_tool]
- Purpose: Execute data analysis and calculations
```

### Pattern 2: Agent Tool for Report Generation
```python
# Agent for creating reports
- Model: claude-sonnet-3-7
- Reasoning: False
- Tools: [python_repl_tool, bash_tool, file_read]
- Purpose: Generate formatted reports from analysis results
```

### Pattern 3: Simple Execution Tool
```python
# Tool for direct command execution
- Type: Regular Tool
- Operation: subprocess.run()
- Error Handling: Capture stderr, return error messages
```

## References

- **System Prompt Creation**: See `references/system-prompt-guidelines.md` for comprehensive prompt writing guidance
- **Template Files**: See `templates/` for tool code templates
- **Example Tools**: See `references/tool-examples.md` for complete real-world examples

## Iteration and Improvement

After creating the initial tool:

1. **Test with Real Scenarios**: Try the tool with actual use cases
2. **Gather Feedback**: Identify what works and what doesn't
3. **Refine Prompts**: For agent tools, improve system prompts based on behavior
4. **Optimize Parameters**: Adjust input schemas if needed
5. **Update Documentation**: Keep descriptions accurate

The goal is creating **effective, reliable tools** that seamlessly integrate with the Strands SDK agent system.
