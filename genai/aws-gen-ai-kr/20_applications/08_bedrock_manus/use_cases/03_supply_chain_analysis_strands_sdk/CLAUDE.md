# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a supply chain analysis use case that combines two AI frameworks:
1. **Strands SDK** - Multi-agent research and analysis workflow (from 01_deep_research_strands_sdk)
2. **OpenSearch Integration** - Supply chain data storage and querying system

The project enables AI-powered supply chain analysis through natural language queries, combining structured data analysis with research capabilities.

## Architecture

### Multi-Agent Workflow (Strands SDK)
The system uses a **LangGraph-based state machine** with specialized AI agents:

**Core Agent Flow**: `clarifier` → `planner` → `supervisor` → [`researcher`, `coder`, `reporter`] → `FINISH`

**Agent Responsibilities**:
- **clarifier**: Processes and clarifies user requests
- **planner**: Creates execution plans using reasoning LLM
- **supervisor**: Routes tasks to appropriate agents using reasoning LLM  
- **researcher**: Conducts web research using Tavily API
- **coder**: Executes Python code and data analysis
- **reporter**: Generates final reports and visualizations using reasoning LLM

**LLM Assignment**:
- `reasoning` LLM: planner, supervisor, reporter (complex decision-making)
- `basic` LLM: clarifier, researcher, coder (straightforward tasks)

### Supply Chain Data Layer (OpenSearch)
**Data Indices**:
- `shipment_tracking`: Maritime shipping data with ports, costs, lead times
- `order_fulfillment`: Customer order processing and delivery tracking  
- `inventory_levels`: Material inventory and stock management
- `supplier_performance`: Supplier quality and delivery metrics
- `ira_compliance`: IRA (Inflation Reduction Act) compliance tracking

**Integration**: MCP (Model Control Protocol) server provides OpenSearch access to Strands agents

### State Management
The `State` TypedDict tracks:
- `messages`: LangGraph message flow
- `history`: Agent conversation history
- `artifacts`: Generated files and visualizations
- `request`: Original user query
- `next`: Current routing decision

## Development Commands

### Environment Setup
```bash
# 1. Create UV environment
chmod +x setup/create-uv-env.sh
./setup/create-uv-env.sh supply-chain-analysis 3.12
cd setup && source .venv/bin/activate

# 2. Install dependencies
uv pip install -r requirements.txt
```

### OpenSearch Setup (Optional - for data analysis)
```bash
# Deploy OpenSearch cluster
chmod +x setup/create-opensearch.sh
./setup/create-opensearch.sh -v 2.19 -d <domain> -u <user> -p <pass> -m prod

# Index supply chain data
python setup/os_indexing.py

# Start MCP server
chmod +x setup/execution-os-mcp-server.sh
./setup/execution-os-mcp-server.sh
```

### Running the System
```bash
# Command line execution
python main.py "Analyze Q4 supply chain performance trends"

# Jupyter notebook (interactive)
jupyter notebook main.ipynb

# With debugging
python main.py "Your query here" --debug
```

## Key Configuration

### Environment Variables (.env)
```bash
TAVILY_API_KEY=your_key                    # Required for web research
JINA_API_KEY=your_key                      # Optional for content extraction
CHROME_INSTANCE_PATH=/path/to/chrome       # For browser automation
BROWSER_HEADLESS=False                     # Browser display mode
```

### Agent-LLM Mapping
Defined in `src/config/agents.py`:
- Reasoning tasks: `planner`, `supervisor`, `reporter`
- Basic tasks: `clarifier`, `researcher`, `coder`
- Prompt caching enabled for: `planner`, `reporter`

### Team Configuration
`TEAM_MEMBERS = ["researcher", "coder", "reporter"]` - Agents available for supervisor routing

## Working with the Code

### Agent Workflow Execution
The main workflow runs through `src/workflow.py`:
```python
from src.workflow import run_agent_workflow
result = run_agent_workflow(user_input="Your query", debug=False)
```

### Strands SDK Integration
Agents are created using `strands_utils.get_agent()` with:
- Dynamic LLM assignment based on agent type
- Streaming response handling
- Prompt template application
- Tool integration

### Graph Structure
Built in `src/graph/builder.py` using LangGraph:
- Nodes: Individual agent functions
- Edges: Flow between agents
- State: Shared context across all agents

### OpenSearch Operations
Use `utils/opensearch.py` for:
- AWS OpenSearch client creation
- Bulk data indexing operations
- Search and analytics queries
- MCP server integration

### Artifact Generation
The system automatically creates:
- `./artifacts/` folder for generated files
- PDF reports, charts, and analysis results
- Conversation history and intermediate outputs

## Common Patterns

### Adding New Agents
1. Create prompt template in `src/prompts/`
2. Add agent function in `src/graph/nodes.py`
3. Register in `src/config/agents.py` with LLM type
4. Update `TEAM_MEMBERS` if supervisor-routable

### Custom Tools
Tools are defined in `src/tools/` and can be:
- Python functions with `@tool` decorator
- Browser automation tools
- External API integrations
- File management utilities

### Prompt Engineering
Prompts use template system in `src/prompts/template.py`:
- Context injection from current state
- Dynamic content based on workflow stage
- Structured output formatting