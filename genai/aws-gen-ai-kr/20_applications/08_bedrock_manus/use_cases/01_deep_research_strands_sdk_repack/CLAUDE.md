# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bedrock-Manus is an AI automation framework optimized for Amazon Bedrock that implements a hierarchical multi-agent system using the Strands SDK. The system orchestrates three core agents in a streaming workflow to accomplish complex tasks like data analysis, code generation, and report creation.

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
python main.py --user_query "Your analysis request" --session_id "session-1"

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
- **Core Framework**: `strands-agents>=1.7.0`, `bedrock-agentcore>=0.1.2`, `bedrock-agentcore-starter-toolkit>=0.1.8`
- **LLM Integration**: `boto3>=1.40.10`, `langchain>=0.3.27`, `mcp>=1.13.0`
- **Data Analysis**: `pandas>=2.3.1`, `numpy>=1.26.4`, `scikit-learn>=1.7.1`
- **Visualization**: `matplotlib>=3.10.5`, `seaborn>=0.13.2`, `plotly>=6.3.0`, `lovelyplots>=1.0.2`
- **Document Generation**: `weasyprint>=66.0` for PDF reports, `koreanize-matplotlib>=0.1.1`
- **UI**: `streamlit==1.48.1`

### Testing
```bash
# Run available test files
python test_stream_graph.py
python setup/test_korean_font.py
```

## Architecture Overview

### Three-Agent Streaming Workflow
The framework implements a **Strands-based streaming workflow** with three core agents:

1. **Coordinator** (`src/graph/nodes.py:coordinator_node`) - Entry point that handles user requests and determines if handoff to planner is needed
2. **Planner** (`src/graph/nodes.py:planner_node`) - Creates detailed execution plans using reasoning capabilities
3. **Supervisor** (`src/graph/nodes.py:supervisor_node`) - Orchestrates specialized tools (Coder, Reporter, Tracker) to execute the plan

### Agent Tool Integration via Supervisor
The Supervisor agent has access to three specialized tools:
- **Coder Agent Tool** (`src/tools/coder_agent_tool.py`) - Python execution, bash commands, data analysis
- **Reporter Agent Tool** (`src/tools/reporter_agent_tool.py`) - Report generation and formatting
- **Tracker Agent Tool** (`src/tools/tracker_agent_tool.py`) - Progress monitoring and workflow state

### LLM Configuration System
Agent-LLM mapping in actual codebase:
- **Coordinator**: `claude-sonnet-3-5-v-2` (fast responses, no reasoning)
- **Planner**: `claude-sonnet-3-7` (reasoning enabled, prompt caching disabled)
- **Supervisor**: `claude-sonnet-3-7` (no reasoning, prompt caching enabled as "default")

### Streaming Architecture
The system uses `StreamableGraph` (`src/graph/builder.py`) which:
- Wraps Strands GraphBuilder with streaming capability
- Uses background task execution with event queue pattern (`src/utils/event_queue.py`)
- Provides real-time event streaming via `graph.stream_async()`

### Global State Management
The workflow uses a **global state system** (`_global_node_states` in `src/graph/nodes.py`):
- Shared state storage between all nodes
- Maintains conversation history, messages, plans, and clues
- Enables stateful communication across the agent workflow

## Key File Locations

### Core Architecture
- `main.py` - Main entry point with streaming execution
- `src/graph/builder.py` - StreamableGraph and LangGraph construction
- `src/graph/nodes.py` - Agent nodes and global state management
- `src/utils/event_queue.py` - Event streaming infrastructure

### Agent Infrastructure
- `src/agents/llm.py` - Bedrock LLM initialization
- `src/utils/strands_sdk_utils.py` - Strands SDK integration and utilities

### Specialized Tools
- `src/tools/python_repl_tool.py` - Python code execution environment
- `src/tools/bash_tool.py` - System command execution
- `src/tools/code_interpreter_tool.py` - Enhanced code interpretation
- `src/tools/decorators.py` - Logging and instrumentation decorators

### Configuration and Environment
- `.bedrock_agentcore.yaml` - Bedrock AgentCore runtime configuration
- `setup/pyproject.toml` - UV dependency management
- `.env.example` - Required environment variables template

## Development Patterns

### Agent Implementation Pattern
All agents in `src/graph/nodes.py` follow this pattern:
```python
# 1. Create agent via Strands SDK
    
# 2. Create agent via Strands SDK
agent = strands_utils.get_agent(
    agent_name="agent_name",
    system_prompts=apply_prompt_template(...),
    agent_type="claude-sonnet-3-7",
    enable_reasoning=True/False,
    prompt_cache_info=(True/False, "default"),
    tools=[...] if needed
)

# 3. Stream processing
async for event in strands_utils.process_streaming_response_yield(...):
    # Process streaming events
    
# 4. Update global state
_global_node_states['shared'].update(...)
```

### Tool Development Pattern
Custom tools should implement the Strands tool specification:
1. Use `@log_io` decorator from `src/tools/decorators.py` for instrumentation
2. Follow patterns in existing tools (`python_repl_tool.py`, `bash_tool.py`)
3. Include proper error handling and logging

### Workflow Control Flow
The workflow uses conditional edges:
- `should_handoff_to_planner()` determines Coordinator → Planner transition
- Fixed edge from Planner → Supervisor
- Supervisor executes tools and completes workflow

## Environment Configuration

### AWS Configuration
- Uses AWS credentials from environment/CLI for Bedrock access

### Required Configuration
Based on `.env.example`:
- `AWS_REGION` and `AWS_DEFAULT_REGION` (us-west-2)
- `BEDROCK_MODEL_ID` (default: anthropic.claude-3-haiku-20240307-v1:0)

## Output and Artifacts

- `./artifacts/` directory: Generated reports and analysis results (auto-cleaned on each run)
- PDF report generation requires Korean font installation via `setup/install_korean_font.sh`
- Sample outputs in `assets/` directory including demo.gif and report.pdf
- Reports generated in multiple formats: PDF, HTML, Markdown

## AgentCore Integration

The project integrates with **Amazon Bedrock AgentCore**:
- Runtime configuration in `.bedrock_agentcore.yaml`
- Containerized deployment support with Docker
- AWS execution role and ECR repository integration
- CodeBuild project for automated builds