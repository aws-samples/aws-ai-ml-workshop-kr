# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bedrock-Manus is an AI automation framework optimized for Amazon Bedrock that implements a hierarchical multi-agent system using LangGraph. The system orchestrates specialized agents to accomplish complex tasks like data analysis, code generation, and report creation.

## Common Development Commands

### Environment Setup
```bash
# Create and activate UV environment with dependencies
cd setup/
./create-uv-env.sh bedrock-manus

# Alternative: Traditional conda environment (as shown in README)
./create_conda_virtual_env.sh bedrock-manus
conda activate bedrock-manus

# Korean fonts are automatically installed by the UV script
# Manual installation if needed:
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
- **Core**: `strands-agents>=1.4.0`, `bedrock-agentcore==0.1.2`, `boto3>=1.40.10`
- **UI**: `streamlit==1.48.1`
- **Data Processing**: `matplotlib>=3.10.5`, `seaborn>=0.13.2`, `lovelyplots>=1.0.2`
- **Document Generation**: `pandoc`, `texlive-xetex`, `poppler-utils` (system packages, auto-installed)
- **Testing**: Basic test files available (`test_*.py` files in root directory)

### Testing
```bash
# Run individual test files
python test_coordinator_integration.py
python test_state_system.py

# Test Korean font installation
python setup/test_korean_font.py
```

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
- **Basic LLM**: Coordinator, Coder (optimized for fast responses)
- **Reasoning LLM**: Planner, Supervisor, Reporter (supports prompt caching)
- **Vision LLM**: Browser agent (available but not used in current workflow)
- **Model Support**: All Amazon Bedrock models (Nova, Claude, DeepSeek, Llama, etc.)

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
1. Implement the Strands tool specification format (see `src/config/tools.py`)
2. Include proper logging using the established logger pattern
3. Use the `@log_io` decorator from `src/tools/decorators.py`
4. Follow patterns in existing tools: `python_repl_tool.py`, `bash_tool.py`

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
- Sample outputs available in `assets/` directory (report.pdf, demo.gif)

## Development Environment

### Tested Environments
- Amazon SageMaker AI Studio (CodeEditor and JupyterLab)
- Local development with UV package manager

### File Structure Notes
- `test.py` - Basic integration testing example
- `main.py` - CLI entry point  
- `main.ipynb` - Jupyter notebook interface
- `app/app.py` - Streamlit web interface
- Configuration files use both UV (`setup/pyproject.toml`) and traditional Python patterns