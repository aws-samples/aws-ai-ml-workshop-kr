# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bedrock-Manus is an AI automation framework optimized for Amazon Bedrock that implements a hierarchical multi-agent system using LangGraph. The system orchestrates specialized agents to accomplish complex tasks like data analysis, code generation, and report creation.

## Common Development Commands

### Environment Setup
```bash
# Create and activate UV environment
cd setup/
./create-uv-env.sh bedrock-manus

# Install Korean fonts (required for PDF generation)
cd setup/
./install_korean_font.sh
```

### Running the Application
```bash
# Run via Python script
python main.py

# Run via Jupyter notebook
jupyter lab main.ipynb

# Run Streamlit UI
cd app/
streamlit run app.py
```

### Development Dependencies
The project uses UV for dependency management. Main dependencies are defined in `setup/pyproject.toml`:
- **Core**: `strands-agents`, `bedrock-agentcore`, `boto3`
- **UI**: `streamlit`
- **Data Processing**: `matplotlib`, `seaborn`
- **Document Generation**: `pandoc`, `texlive-xetex` (system packages)

## Architecture Overview

### Multi-Agent System Design
The framework implements a LangGraph-based workflow with five specialized agents:

1. **Coordinator** (`src/graph/nodes.py:coordinator_node`) - Entry point that handles initial interactions and routes tasks
2. **Planner** (`src/graph/nodes.py:planner_node`) - Analyzes tasks and creates execution strategies using reasoning LLM
3. **Supervisor** (`src/graph/nodes.py:supervisor_node`) - Oversees and manages execution of other agents using reasoning LLM
4. **Coder** (`src/graph/nodes.py:code_node`) - Handles Python code execution and bash commands via custom tools
5. **Reporter** (`src/graph/nodes.py:reporter_node`) - Generates reports and summaries using reasoning LLM

### LLM Tier System
Agent-LLM mapping is configured in `src/config/agents.py`:
- **Basic LLM**: Coordinator, Coder
- **Reasoning LLM**: Planner, Supervisor, Reporter  
- **Vision LLM**: Browser agent (available but not used in current workflow)

### Prompt System
Each agent uses role-specific prompts from `src/prompts/*.md` files:
- Template engine in `src/prompts/template.py` handles variable substitution
- Supports prompt caching for reasoning agents to improve performance

### Custom Tools Integration
The system provides specialized tools via Strands SDK:
- **Python REPL** (`src/tools/python_repl_tool.py`) - Code execution environment
- **Bash Tool** (`src/tools/bash_tool.py`) - System command execution
- **Web Crawling** (`src/crawler/`) - Content extraction using Jina API

## Key File Locations

### Core Workflow
- `src/workflow.py` - Main workflow execution entry point
- `src/graph/builder.py` - LangGraph construction and node connections
- `src/graph/types.py` - State management and type definitions

### Agent Configuration  
- `src/config/agents.py` - Agent-LLM mapping and caching configuration
- `src/config/tools.py` - Tool specifications and registration
- `src/agents/llm.py` - LLM initialization using Bedrock models

### Utilities
- `src/utils/strands_sdk_utils.py` - Strands SDK integration helpers
- `src/utils/bedrock.py` - AWS Bedrock client configuration
- `src/utils/common_utils.py` - Message formatting and common utilities

## Development Patterns

### Agent Implementation
New agents should follow the pattern in `src/graph/nodes.py`:
1. Use `strands_utils().get_agent_by_name()` to create agent with proper LLM
2. Apply prompt template from `src/prompts/template.py`
3. Handle state updates and message passing according to LangGraph conventions

### Tool Registration
Custom tools should:
1. Implement the Strands tool specification format
2. Include proper logging using the established logger pattern
3. Use the `@log_io` decorator from `src/tools/decorators.py`

### State Management
The workflow uses a shared state object defined in `src/graph/types.py` that includes:
- Message history
- Team member information
- Request context
- Intermediate results

## Environment Configuration

### Required Environment Variables
- `TAVILY_API_KEY` - For web search functionality
- `JINA_API_KEY` - For content extraction
- `CHROME_INSTANCE_PATH` - For browser automation (optional)
- `BROWSER_HEADLESS` - Browser mode configuration

### AWS Configuration
The framework automatically uses AWS credentials from the environment or AWS CLI configuration for Bedrock access.

## Output and Artifacts

- Generated reports and analysis results are saved to `./artifacts/` directory
- The system automatically cleans this directory on each run
- PDF reports require Korean font installation for proper rendering