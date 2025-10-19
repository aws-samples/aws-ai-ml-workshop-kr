# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bedrock-Manus is an AI automation framework optimized for Amazon Bedrock that implements a hierarchical multi-agent system using the Strands SDK. The system orchestrates specialized agents in a streaming workflow to accomplish complex tasks like data analysis, code generation, and report creation.

## Common Development Commands

### Environment Setup
```bash
# Create and activate UV environment with dependencies (recommended)
cd setup/
./create-uv-env.sh bedrock-manus-agentcore 3.12

# Run from root directory (UV creates symlinks automatically)
cd ..
uv run python main.py

# Test Korean font installation (required for PDF reports)
python setup/test_korean_font.py
```

### Running the Application
```bash
# CLI execution with custom query
python main.py --user_query "Your analysis request"

# Run with default predefined query
python main.py

# Streamlit UI
cd app/
streamlit run app.py

# Jupyter notebook interface
jupyter lab main.ipynb
```

### Development Dependencies
The project uses UV for dependency management. Key dependencies in `setup/pyproject.toml`:
- **Core Framework**: `strands-agents>=1.12.0`, `mcp>=1.13.0`
- **LLM Integration**: `boto3>=1.40.51`, `langchain>=0.3.27`
- **Data Analysis**: `pandas>=2.3.1`, `numpy>=1.26.4`, `scikit-learn>=1.7.1`
- **Visualization**: `matplotlib>=3.10.7`, `seaborn>=0.13.2`, `plotly>=6.3.0`, `lovelyplots>=1.0.2`
- **Document Generation**: `weasyprint>=66.0` for PDF reports, `koreanize-matplotlib>=0.1.1`
- **UI**: `streamlit==1.48.1`

## Architecture Overview

### Three-Agent Streaming Workflow
The framework implements a **Strands-based streaming workflow** with three core agents:

1. **Coordinator** (`src/graph/nodes.py:coordinator_node`) - Entry point that handles user requests and determines if handoff to planner is needed
2. **Planner** (`src/graph/nodes.py:planner_node`) - Creates detailed execution plans using reasoning capabilities
3. **Supervisor** (`src/graph/nodes.py:supervisor_node`) - Orchestrates specialized tool agents (Coder, Reporter, Tracker, Validator) to execute the plan

### Agent Tool Integration via Supervisor
The Supervisor agent has access to specialized tool agents defined in `src/tools/`:
- **Coder Agent Tool** (`coder_agent_tool.py`) - Python execution, bash commands, data analysis
- **Reporter Agent Tool** (`reporter_agent_tool.py`) - Report generation and formatting
- **Tracker Agent Tool** (`tracker_agent_tool.py`) - Progress monitoring and workflow state
- **Validator Agent Tool** (`validator_agent_tool.py`) - Validation of outputs and results

### LLM Configuration System
Agent-LLM mapping in `src/graph/nodes.py`:
- **Coordinator**: `claude-sonnet-4` (fast responses, no reasoning)
- **Planner**: `claude-sonnet-4` (reasoning enabled, prompt caching disabled)
- **Supervisor**: `claude-sonnet-4` (no reasoning, prompt caching enabled as "default")
- **Tool Agents** (Coder/Reporter/Tracker/Validator): Configurable per agent

Model selection is handled by `strands_utils.get_model()` in `src/utils/strands_sdk_utils.py` which supports:
- `claude-sonnet-3-7` - Maps to Claude V3.7 Sonnet
- `claude-sonnet-4` - Maps to Claude V4 Sonnet
- `claude-sonnet-3-5-v-2` - Maps to Claude V3.5 V2 Sonnet

### Streaming Architecture
The system uses `StreamableGraph` (`src/graph/builder.py`) which:
- Wraps Strands GraphBuilder with streaming capability
- Uses background task execution with event queue pattern (`src/utils/event_queue.py`)
- Provides real-time event streaming via `graph.stream_async()`
- Events are placed in a global queue and consumed asynchronously

### Global State Management
The workflow uses a **global state system** (`_global_node_states` in `src/graph/nodes.py`):
- Shared state storage accessible across all nodes
- Maintains conversation history, messages, plans, and clues
- Enables stateful communication across the agent workflow
- Key state fields:
  - `messages` - Current conversation messages
  - `request` and `request_prompt` - User's original request
  - `full_plan` - Planner's execution plan
  - `clues` - Accumulated context from agent executions
  - `history` - List of agent interactions with format `{"agent": "name", "message": "text"}`

## Key File Locations

### Core Architecture
- `main.py` - Main entry point with streaming execution using `graph_streaming_execution()`
- `src/graph/builder.py` - StreamableGraph and LangGraph construction with `build_graph()`
- `src/graph/nodes.py` - Agent node implementations and global state management
- `src/utils/event_queue.py` - Event streaming infrastructure with thread-safe queue

### Agent Infrastructure
- `src/utils/strands_sdk_utils.py` - Strands SDK integration utilities including:
  - `get_agent()` - Agent creation with model, prompts, tools
  - `get_model()` - LLM model initialization with caching and reasoning settings
  - `process_streaming_response_yield()` - Streaming event processing

### Specialized Tools
- `src/tools/python_repl_tool.py` - Python code execution environment
- `src/tools/bash_tool.py` - System command execution
- `src/tools/coder_agent_tool.py` - Coder agent wrapper as tool
- `src/tools/reporter_agent_tool.py` - Reporter agent wrapper as tool
- `src/tools/tracker_agent_tool.py` - Tracker agent wrapper as tool
- `src/tools/validator_agent_tool.py` - Validator agent wrapper as tool
- `src/tools/decorators.py` - Logging and instrumentation decorators

### Prompts System
- `src/prompts/template.py` - Prompt template engine with variable substitution
- `src/prompts/coordinator.md` - Coordinator agent system prompt
- `src/prompts/planner.md` - Planner agent system prompt with reasoning
- `src/prompts/supervisor.md` - Supervisor agent system prompt
- `src/prompts/coder.md` - Coder tool agent system prompt
- `src/prompts/reporter.md` - Reporter tool agent system prompt
- `src/prompts/tracker.md` - Tracker tool agent system prompt
- `src/prompts/validator.md` - Validator tool agent system prompt

### Configuration and Environment
- `.env.example` - Required environment variables template
- `setup/pyproject.toml` - UV dependency management
- `setup/create-uv-env.sh` - Environment setup script with symlink creation

## Development Patterns

### Agent Implementation Pattern
All agents in `src/graph/nodes.py` follow this pattern:
```python
# 1. Extract shared state from global storage
global _global_node_states
shared_state = _global_node_states.get('shared', {})

# 2. Create agent via Strands SDK
agent = strands_utils.get_agent(
    agent_name="agent_name",
    system_prompts=apply_prompt_template(prompt_name="agent_name", prompt_context={}),
    agent_type="claude-sonnet-4",  # Model selection
    enable_reasoning=True/False,
    prompt_cache_info=(True/False, "default"),  # (enable, cache_type)
    tools=[...],  # Optional tools
    streaming=True
)

# 3. Stream processing
full_text = ""
async for event in strands_utils.process_streaming_response_yield(
    agent, message, agent_name="agent_name", source="node_name"
):
    if event.get("event_type") == "text_chunk":
        full_text += event.get("data", "")
response = {"text": full_text}

# 4. Update global state
shared_state['messages'] = agent.messages
shared_state['history'].append({"agent": "name", "message": response["text"]})
```

### Tool Agent Implementation Pattern
Tool agents (in `src/tools/*_agent_tool.py`) follow this pattern:
```python
TOOL_SPEC = {
    "name": "tool_name",
    "description": "Tool description",
    "inputSchema": {"json": {...}}
}

def handle_tool_function(task: str):
    # 1. Access global state
    from src.graph.nodes import _global_node_states
    shared_state = _global_node_states.get('shared', {})

    # 2. Create specialized agent with tools
    agent = strands_utils.get_agent(...)

    # 3. Process streaming in async context
    async def process_stream():
        full_text = ""
        async for event in strands_utils.process_streaming_response_yield(...):
            if event.get("event_type") == "text_chunk":
                full_text += event.get("data", "")
        return {"text": full_text}

    response = asyncio.run(process_stream())

    # 4. Update global state (clues, history)
    shared_state['clues'] = '...'
    shared_state['history'].append(...)

    return result_text
```

### Workflow Control Flow
The workflow uses conditional edges defined in `src/graph/builder.py`:
- `should_handoff_to_planner()` determines Coordinator → Planner transition
  - Returns `True` if coordinator's response contains `'handoff_to_planner'`
- Fixed edge from Planner → Supervisor
- Supervisor executes tool agents and completes workflow

### Event Streaming Pattern
The event queue system in `src/utils/event_queue.py` provides:
```python
from src.utils.event_queue import put_event, get_event, has_events, clear_queue

# Producer: Put events in queue
put_event({"type": "text_chunk", "data": "..."})

# Consumer: Retrieve events
while has_events():
    event = get_event()

# Cleanup: Clear queue
clear_queue()
```

## Environment Configuration

### AWS Configuration
- Uses AWS credentials from environment/CLI for Bedrock access
- Set `AWS_REGION` and `AWS_DEFAULT_REGION` (default: us-west-2)
- Optional `BEDROCK_MODEL_ID` for custom model selection

### Required Configuration
Based on `.env.example`:
- `AWS_REGION=us-west-2`
- `AWS_DEFAULT_REGION=us-west-2`
- `BEDROCK_MODEL_ID` (optional, defaults configured in code)

## Output and Artifacts

- `./artifacts/` directory - Generated reports and analysis results (auto-cleaned on each run by `remove_artifact_folder()` in `main.py`)
- PDF report generation requires Korean font installation via `setup/install_korean_font.sh`
- Sample outputs in `assets/` directory including demo.gif and example reports
- Reports generated in multiple formats: PDF, HTML, Markdown
