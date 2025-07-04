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
# 1. Create UV environment (from setup directory)
cd setup
chmod +x create-uv-env.sh
./create-uv-env.sh supply-chain-analysis 3.12

# 2. Activate environment and install dependencies
source .venv/bin/activate
uv pip install -r requirements.txt

# 3. Install additional prerequisites (optional)
chmod +x install-prerequisites.sh
./install-prerequisites.sh
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
# Command line execution (regular workflow)
python main.py "Analyze Q4 supply chain performance trends"

# SCM specialized workflow
python main.py "Analyze Chicago port strike impact" --scm

# Jupyter notebook (interactive)
jupyter notebook main.ipynb

# With debugging
python main.py "Your query here" --debug
```

### Testing
```bash
# Run SCM workflow test
python test_scm_workflow.py

# Test Korean font rendering (for charts)
python setup/test_korean_font.py
```

**Note**: The project currently uses manual testing via direct script execution. No formal testing framework (pytest, unittest) is configured.

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
- **Reasoning tasks**: `planner`, `supervisor`, `reporter`, `scm_correlation_analyzer`, `scm_mitigation_planner`
- **Basic tasks**: `clarifier`, `researcher`, `coder`, `scm_researcher`, `scm_data_analyzer`, `scm_impact_analyzer`
- **Prompt caching enabled for**: `planner`, `reporter`, `scm_correlation_analyzer`, `scm_mitigation_planner`

### Team Configuration
- **SCM workflow**: `SCM_TEAM_MEMBERS = ["scm_impact_analyzer", "scm_correlation_analyzer", "scm_mitigation_planner", "planner", "reporter"]` - Agents available for supervisor routing
- **Standard workflow**: Currently uses the same SCM workflow configuration

## Working with the Code

### Agent Workflow Execution
Two main workflows are available:

**Standard Workflow** (`src/workflow.py`):
```python
from src.workflow import run_agent_workflow
result = run_agent_workflow(user_input="Your query", debug=False)
```

**SCM Specialized Workflow** (`src/workflow.py`):
```python
from src.workflow import run_agent_workflow
result = run_agent_workflow(user_input="Supply chain query", debug=False)
```

**Note**: The codebase currently only implements a single SCM workflow function (`run_agent_workflow`) which handles all supply chain analysis. The system auto-detects SCM-related queries using keywords: `["supply chain", "scm", "port", "shipping", "logistics", "disruption", "strike", "transportation"]`

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
1. Create prompt template in `src/prompts/` (e.g., `new_agent.md`)
2. Add agent function in `src/graph/scm_nodes.py` (currently only SCM nodes are implemented)
3. Register in `src/config/agents.py`:
   - Add to `AGENT_LLM_MAP` with LLM type (`basic`/`reasoning`/`vision`)
   - Add to `AGENT_PROMPT_CACHE_MAP` with caching preference
4. Update workflow routing logic in `src/graph/builder.py`
5. Update `SCM_TEAM_MEMBERS` if supervisor-routable

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

**Available Prompt Templates**:
- Standard agents: `clarifier.md`, `planner.md`, `supervisor.md`, `researcher.md`, `coder.md`, `reporter.md`
- SCM agents: `scm_researcher.md`, `scm_data_analyzer.md`, `scm_impact_analyzer.md`, `scm_correlation_analyzer.md`, `scm_mitigation_planner.md`
- Utility prompts: `browser.md`, `human_feedback.md`, `coordinator.md`

## Key Dependencies
- **Strands SDK**: `strands-agents==0.1.7`, `strands-agents-tools==0.1.5`
- **LangGraph**: `langgraph==0.4.8` for workflow orchestration
- **AWS Services**: `boto3==1.38.36`, `langchain-aws==0.2.25`
- **Research Tools**: `tavily-python==0.7.6` for web research
- **Visualization**: `matplotlib==3.10.1`, `seaborn==0.13.2`, `lovelyplots==1.0.2`
- **MCP Integration**: `mcp==1.9.4` for OpenSearch communication
- **Data Processing**: `pandas==2.3.0`, `numpy==2.3.0`
- **Web Automation**: `playwright==1.52.0` for browser automation
- **Document Processing**: `weasyprint==65.1` for PDF generation