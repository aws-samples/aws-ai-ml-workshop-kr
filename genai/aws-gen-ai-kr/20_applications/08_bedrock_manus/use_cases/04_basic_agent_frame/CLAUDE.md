# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a basic agent framework for AWS Bedrock using Strands Agent SDK. The project provides a foundation for building AI agents that interact with Amazon Bedrock services, specifically using Claude models through the Strands framework.

## Development Setup

### Environment Setup
Use UV for Python environment management:
```bash
cd setup/
chmod +x create-uv-env.sh
./create-uv-env.sh agent_frame 3.12
source .venv/bin/activate
```

### Required Dependencies
The project uses:
- `strands-agents==1.0.1` - Core agent framework
- `bedrock-agentcore==0.1.0` - Bedrock integration
- `strands-agents-tools==0.2.1` - Agent tools
- `boto3==1.39.9` - AWS SDK
- `langgraph==0.5.3` - Graph-based agent orchestration
- `mcp==1.12.0` - Model Context Protocol
- `streamlit==1.47.0` - Web UI framework

### Running the Project
- Start Jupyter environment: `uv run jupyter lab`
- Run Python scripts: `uv run python <script_name>`
- Execute main notebook: Open `main.ipynb` in Jupyter Lab with the registered kernel

## Architecture

### Core Components

**Agent Configuration (`src/config/agents.py`)**:
- Defines agent-to-LLM type mappings
- Manages prompt caching configuration
- Supports multiple agent types: clarifier, planner, researcher, coder, reporter, SCM specialists

**LLM Management (`src/agents/llm.py`)**:
- Provides `get_llm_by_type()` function for LLM instantiation
- Supports "basic" (Claude-3.5-Sonnet) and "reasoning" (Claude-3.7-Sonnet) models
- Configures Bedrock models with appropriate parameters and caching

**Bedrock Utilities (`src/utils/bedrock.py`)**:
- Contains `bedrock_info` class with model mappings
- Provides helper functions for Bedrock client creation
- Includes conversation API utilities and output parsing

### Agent Types and Model Mapping
- **Basic agents**: Use Claude-3.5-V-2-Sonnet for straightforward tasks
- **Reasoning agents**: Use Claude-3.7-Sonnet with thinking capabilities enabled
- **SCM specialists**: Dedicated agents for supply chain management tasks

### Prompt and Tool Management
- Prompts stored in `src/prompt/` directory (currently empty structure)
- Tool configuration in `src/config/tools.py`
- Agent utilities in `src/utils/strands_sdk_utils.py`

## Key Files
- `main.ipynb`: Primary notebook demonstrating agent usage
- `setup/pyproject.toml`: Project dependencies and configuration
- `src/config/agents.py`: Agent configuration and LLM mappings
- `src/agents/llm.py`: LLM instantiation and configuration
- `src/utils/bedrock.py`: Bedrock service utilities and model management

## Usage Patterns

### Creating an Agent
```python
from src.agents.llm import get_llm_by_type
from strands import Agent

# Get appropriate LLM for agent type
llm = get_llm_by_type("reasoning", cache_type="default", enable_reasoning=True)

# Create agent with system prompt and tools
agent = Agent(
    model=llm,
    system_prompt=system_prompts,
    tools=tools,
    callback_handler=None
)
```

### Model Selection
- Use "basic" type for standard tasks requiring Claude-3.5-Sonnet
- Use "reasoning" type for complex tasks requiring Claude-3.7-Sonnet with thinking
- Enable prompt caching for frequently used agents (planner, reporter, correlation analyzer)

## Development Notes
- The project requires AWS credentials for Bedrock access
- Korean font support is included in the setup process
- Uses UV for modern Python dependency management
- Jupyter kernels are automatically registered during environment setup