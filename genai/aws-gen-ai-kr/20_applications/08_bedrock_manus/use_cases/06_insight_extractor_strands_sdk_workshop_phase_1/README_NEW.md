<div align="center">
  <h1>Deep Insight</h1>

  <h2>A model-driven approach to building customizable reporting agents with Amazon Bedrock</h2>

  <div align="center">
    <a href="https://github.com/aws-samples/aws-ai-ml-workshop-kr/graphs/commit-activity"><img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/m/aws-samples/aws-ai-ml-workshop-kr"/></a>
    <a href="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/LICENSE"><img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green"/></a>
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.12+-blue.svg"/></a>
  </div>

  <p>
    <a href="#quick-start">Quick Start</a>
    ◆ <a href="#installation">Installation</a>
    ◆ <a href="#features-at-a-glance">Features</a>
    ◆ <a href="#architecture">Architecture</a>
    ◆ <a href="#usage">Usage</a>
  </p>
</div>

## Feature Overview

Deep Insight transforms weeks of manual reporting work into minutes using hierarchical multi-agent systems built on Strands SDK and Amazon Bedrock.

- **🔧 Full Customization**: Deploy and modify multi-agent workflows in your AWS VPC with complete code access
- **🤖 Multi-Agent System**: Hierarchical workflow with Coordinator, Planner, Supervisor, and specialized tool agents
- **📊 Verifiable Insights**: Reports with calculation methods, sources, and transparent reasoning processes
- **🔗 Extensible Integration**: Connect external data sources via AgentCore Gateway and MCP protocol
- **🚀 Beyond Reporting**: Extend framework to any agent use case—shopping, support, log analysis, and more
- **🔒 Enterprise Security**: Complete VPC isolation for sensitive data with single-tenant deployment
- **⚡ Rapid Execution**: Transform waterfall sequential workflows into parallel multi-agent collaboration

## Quick Start

```bash
# Clone repository
git clone https://github.com/aws-samples/aws-ai-ml-workshop-kr.git
cd aws-ai-ml-workshop-kr/genai/aws-gen-ai-kr/20_applications/08_bedrock_manus/use_cases/06_insight_extractor_strands_sdk_workshop_phase_1

# Create UV environment (automatically installs dependencies)
cd setup/
./create-uv-env.sh deep-insight 3.12

# Run the framework
cd ..
uv run python main.py
```

**Expected output:**
```
✓ Environment initialized
✓ Agents loaded: Coordinator, Planner, Supervisor
✓ Starting workflow...
[Real-time streaming output will appear here]
```

> **Note**: Requires Python 3.12+, AWS credentials configured, and Claude model access enabled in Amazon Bedrock (us-west-2 region recommended).

## Installation

Ensure you have Python 3.12+ installed, then:

```bash
# Clone the repository
git clone https://github.com/aws-samples/aws-ai-ml-workshop-kr.git
cd aws-ai-ml-workshop-kr/genai/aws-gen-ai-kr/20_applications/08_bedrock_manus/use_cases/06_insight_extractor_strands_sdk_workshop_phase_1
```

### Option 1: UV Environment (Recommended)

UV provides fast and reliable dependency management:

```bash
# Navigate to setup directory
cd setup/

# Create UV environment with Python 3.12
./create-uv-env.sh deep-insight 3.12

# Return to project root and run
cd ..
uv run python main.py
```

### Option 2: Conda Environment

```bash
# Create conda environment
cd setup/
./create_conda_virtual_env.sh deep-insight 3.12

# Activate environment
source .venv/bin/activate

# Run from project root
cd ..
python main.py
```

### Configure AWS Credentials

```bash
# Option 1: AWS CLI configuration
aws configure

# Option 2: Environment variables
export AWS_REGION=us-west-2
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key

# Option 3: Use .env file
cp .env.example .env
# Edit .env with your AWS credentials and settings
```

### Enable Bedrock Model Access

1. Navigate to [AWS Bedrock Console](https://console.aws.amazon.com/bedrock/)
2. Go to "Model access" section
3. Request access for Claude Sonnet models
4. Wait for approval (usually instant)

### Verify Installation

```bash
# Check Python version
python --version  # Should show Python 3.12.x

# Test framework import
python -c "from src.graph.builder import build_graph; print('✓ Installation successful!')"

# (Optional) Test Korean font for PDF reports
python setup/test_korean_font.py
```

## Features at a Glance

### Hierarchical Multi-Agent System

Built on Strands SDK with three-tier agent hierarchy:

```python
# Coordinator handles user requests
# Planner creates execution plans with reasoning
# Supervisor orchestrates specialized tool agents

from src.graph.builder import build_graph

graph = build_graph()
result = await graph.stream_async("Analyze sales data in ./data/sales.csv")
```

**Agent Workflow:**
- **Coordinator**: Routes queries and handles simple requests directly
- **Planner**: Creates detailed execution plans using reasoning capabilities
- **Supervisor**: Delegates tasks to Coder, Reporter, Tracker, and Validator agents
- **Tool Agents**: Execute specialized tasks (data analysis, report generation, validation)

### Streaming Execution

Real-time progress updates with event-based streaming:

```python
from src.utils.event_queue import get_event, has_events

# Events stream in real-time
while has_events():
    event = get_event()
    if event.get("event_type") == "text_chunk":
        print(event.get("data"), end="", flush=True)
```

**Key features:**
- Background task execution with event queue pattern
- Thread-safe global state management
- Live UI updates via `StreamableGraph`

### Multi-Model Support

Leverage all Amazon Bedrock models with intelligent routing:

```python
from src.utils.strands_sdk_utils import strands_utils

# Configure different models per agent
coordinator_agent = strands_utils.get_agent(
    agent_name="coordinator",
    agent_type="claude-sonnet-4",  # Fast responses
    enable_reasoning=False
)

planner_agent = strands_utils.get_agent(
    agent_name="planner",
    agent_type="claude-sonnet-4",  # Reasoning enabled
    enable_reasoning=True
)
```

**Supported models:**
- Anthropic Claude (Sonnet 4, Sonnet 3.7, Sonnet 3.5, Opus, Haiku)
- Amazon Nova (Nova Pro, Nova Lite)
- Meta Llama, Mistral AI, Cohere

### Professional Report Generation

Automated report creation with visualizations:

```python
# Reports generated automatically with:
# - Executive summaries
# - Statistical analysis
# - Data visualizations
# - Methodology explanations
# - Multi-format output (PDF, HTML, Markdown)

# Output saved to ./artifacts/
# - analysis_report.pdf
# - visualizations/*.png
# - data_summary.json
```

**Visualization capabilities:**
- Line charts, bar charts, scatter plots, heatmaps
- Korean language support with custom fonts
- Publication-ready formatting

### Extensible Tool System

Add custom tools to agents:

```python
# Create custom tool in src/tools/custom_tool.py
TOOL_SPEC = {
    "name": "custom_analyzer",
    "description": "Performs custom analysis",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "data": {"type": "string", "description": "Data to analyze"}
            },
            "required": ["data"]
        }
    }
}

def handle_custom_analyzer(data: str):
    # Your custom logic
    return analysis_result

# Register tool with agent
from src.tools.custom_tool import TOOL_SPEC, handle_custom_analyzer
agent = strands_utils.get_agent(tools=[TOOL_SPEC])
```

### Alternative Interfaces

Multiple ways to interact with the framework:

```bash
# CLI with custom query
python main.py --user_query "Analyze customer churn patterns"

# Jupyter Notebook
jupyter lab main.ipynb

# Streamlit Web UI
cd app/
streamlit run app.py
```

## Architecture

### System Overview

```
User Query + Data File → Coordinator → Planner → Supervisor
                                                     ├─→ Coder (Data Analysis & Execution)
                                                     ├─→ Validator (Quality & Verification)
                                                     ├─→ Reporter (Report Generation)
                                                     └─→ Tracker (Process Transparency)
                                                            ↓
                                                   AgentCore Gateway (MCP Tools)
                                                            ↓
                                                   PDF Report + Verification Files
```

### Three-Tier Agent Hierarchy

```
┌─────────────────────────────────────────────────────────┐
│                     User Input                          │
│              (Natural Language Query)                    │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  COORDINATOR (Entry Point)                              │
│  • Handles initial user requests                        │
│  • Routes simple queries directly                       │
│  • Hands off complex tasks to Planner                   │
│  Model: Claude Sonnet 4 (no reasoning)                  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  PLANNER (Strategic Thinking)                           │
│  • Analyzes task complexity                             │
│  • Creates detailed execution plan                      │
│  • Uses reasoning for step-by-step strategy             │
│  Model: Claude Sonnet 4 (reasoning enabled)             │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  SUPERVISOR (Task Orchestrator)                         │
│  • Delegates tasks to specialized agents                │
│  • Monitors progress and coordinates workflow           │
│  • Aggregates results from tool agents                  │
│  Model: Claude Sonnet 4 (prompt caching enabled)        │
└──────────┬──────────┬──────────┬──────────┬─────────────┘
           │          │          │          │
     ┌─────┘    ┌─────┘    ┌─────┘    └─────┐
     ▼          ▼          ▼                 ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────────┐
│  CODER  │ │REPORTER │ │TRACKER  │ │  VALIDATOR   │
│         │ │         │ │         │ │              │
│ Python  │ │ Report  │ │Progress │ │ Quality      │
│ Bash    │ │ Format  │ │Monitor  │ │ Validation   │
│ Analysis│ │ Generate│ │ State   │ │ Verification │
└─────────┘ └─────────┘ └─────────┘ └──────────────┘
```

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| **Graph Builder** | `src/graph/builder.py` | StreamableGraph construction and workflow orchestration |
| **Agent Nodes** | `src/graph/nodes.py` | Coordinator, Planner, Supervisor implementations |
| **Strands Utils** | `src/utils/strands_sdk_utils.py` | Agent creation, model selection, streaming processing |
| **Event Queue** | `src/utils/event_queue.py` | Thread-safe event streaming infrastructure |
| **Tool Agents** | `src/tools/*_agent_tool.py` | Specialized agents wrapped as tools |
| **Prompts** | `src/prompts/*.md` | System prompts for each agent |

### Global State Management

The framework uses a shared state system (`_global_node_states` in `src/graph/nodes.py`):

- **messages**: Current conversation messages for each agent
- **request**: User's original request
- **full_plan**: Planner's execution plan
- **clues**: Accumulated context from agent executions
- **history**: List of agent interactions with format `{"agent": "name", "message": "text"}`

This enables stateful communication across the entire workflow.

## Usage

### Basic Execution

Run with default predefined query:

```bash
python main.py
```

Run with custom query:

```bash
python main.py --user_query "Analyze sales trends for Q4 2024"
```

### Configuration

Edit `.env` file for custom settings:

```bash
# AWS Configuration
AWS_REGION=us-west-2
AWS_DEFAULT_REGION=us-west-2

# Model Configuration
BEDROCK_MODEL_ID=claude-sonnet-4
```

### Output Files

Results are saved to `./artifacts/` directory:

```
artifacts/
├── analysis_report.pdf       # Final PDF report
├── analysis_report.html      # HTML version
├── analysis_report.md        # Markdown version
├── data_summary.json         # Structured results
└── visualizations/           # Generated charts
    ├── trend_chart.png
    └── correlation_matrix.png
```

**Note**: The `artifacts/` directory is automatically cleaned at the start of each run. Backup important results before running again.

### Advanced Usage Examples

**Example 1: Batch Processing**

```python
from src.graph.builder import build_graph

graph = build_graph()

queries = [
    "Analyze Q1 sales data",
    "Analyze Q2 sales data",
    "Analyze Q3 sales data",
    "Analyze Q4 sales data"
]

for query in queries:
    result = await graph.stream_async(query)
    # Results automatically saved to ./artifacts/
```

**Example 2: Custom Model Selection**

```python
import os
os.environ['BEDROCK_MODEL_ID'] = 'claude-opus'

# Run with premium model for complex reasoning
python main.py --user_query "Perform advanced statistical analysis"
```

**Example 3: Jupyter Notebook**

```bash
# Launch Jupyter Lab
jupyter lab main.ipynb

# Execute cells to:
# 1. Initialize framework
# 2. Submit query
# 3. View streaming results
# 4. Access generated artifacts
```

**Example 4: Streamlit Web UI**

```bash
cd app/
streamlit run app.py

# Web interface provides:
# - Text input for queries
# - Real-time streaming output
# - Download links for artifacts
# - Workflow visualization
```

## Demo

### Amazon Sales Data Analysis

> **Task**: "I would like to analyze Amazon product sales data. The target file is `./data/Amazon_Sale_Report.csv`. Please conduct comprehensive analysis to extract marketing insights—explore data attributes, product trends, variable relationships, and combinations. Include detailed analysis with supporting charts and save the final report as PDF."

[![Demo](./assets/demo.gif)](https://youtu.be/DwWICGLEv14)

[▶️ Watch Full Demo on YouTube](https://youtu.be/DwWICGLEv14)

### Sample Outputs

- 📄 [English Report (6 pages)](./assets/report_en.pdf)
- 📄 [Korean Report (10 pages)](./assets/report.pdf)
- 📊 Dataset: [Amazon Sale Report from Kaggle](https://www.kaggle.com/datasets/thedevastator/unlock-profits-with-e-commerce-sales-data)

### What Happened Behind the Scenes

1. **Coordinator** received the natural language query
2. **Planner** created a 7-step execution plan
3. **Supervisor** delegated tasks to:
   - **Coder Agent**: Loaded CSV, performed statistical analysis, created visualizations
   - **Reporter Agent**: Compiled findings into structured report with charts
   - **Validator Agent**: Verified data quality and analysis correctness
4. Final PDF report generated with executive summary, trend analysis, and recommendations

**Time**: Completed in ~15 minutes (traditional manual process: 2-3 days)

## When to Choose Deep Insight

### Deep Insight is Best For:

- **Full Customization Needs**: Modify agent behavior, prompts, and workflows
- **Regulatory Compliance**: Deploy in single-tenant VPC with complete data isolation
- **Domain-Specific Workflows**: Build specialized agent systems beyond general-purpose tools
- **Competitive Differentiation**: Develop proprietary AI capabilities
- **Strategic Ownership**: Control infrastructure evolution as strategic asset
- **Multi-Use Cases**: Extend framework beyond reporting to other agent applications

### Managed Services are Best For:

- **Immediate Deployment**: Zero setup, vendor-managed infrastructure
- **General-Purpose Tasks**: Standard analytical workflows
- **Limited Customization**: Pre-built agents meet requirements
- **Multi-Tenant SaaS**: Shared infrastructure is acceptable

### Strategic Benefits

1. **Knowledge Accumulation**: Build internal expertise in Strands SDK and AgentCore
2. **Competitive Edge**: Develop capabilities competitors can't replicate
3. **Cost Predictability**: Control infrastructure costs in your AWS account
4. **Data Sovereignty**: Maintain complete control over sensitive data
5. **Innovation Speed**: Rapidly prototype new agent use cases

## Troubleshooting

### Installation Issues

**Problem**: `ModuleNotFoundError: No module named 'strands'`

**Solution**: Ensure UV environment is activated and dependencies are installed:
```bash
cd setup/
./create-uv-env.sh deep-insight 3.12
cd ..
uv run python main.py
```

---

**Problem**: Python version mismatch

**Solution**: This project requires Python 3.12+. Check your version:
```bash
python --version
```

If older version, install Python 3.12 or use conda:
```bash
conda create -n deep-insight python=3.12
conda activate deep-insight
```

### AWS & Bedrock Issues

**Problem**: `AccessDeniedException: Model access denied`

**Solution**: Enable Claude models in Amazon Bedrock console:
1. Navigate to AWS Bedrock console
2. Go to "Model access" section
3. Request access for Claude Sonnet models
4. Wait for approval (usually instant)

---

**Problem**: `CredentialsError: Unable to locate credentials`

**Solution**: Configure AWS credentials:
```bash
# Option 1: AWS CLI
aws configure

# Option 2: Environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_REGION=us-west-2
```

### Runtime Issues

**Problem**: PDF report generation fails

**Solution**: Install Korean font (if using Korean text):
```bash
cd setup/
./install_korean_font.sh
python test_korean_font.py
```

---

**Problem**: Workflow hangs or times out

**Solution**: Check event queue and clear if needed:
```python
from src.utils.event_queue import clear_queue
clear_queue()
```

### Getting Help

- **Documentation**: See [CLAUDE.md](CLAUDE.md) for developer documentation
- **Issues**: [Report bugs or request features](https://github.com/aws-samples/aws-ai-ml-workshop-kr/issues)
- **Debugging**: Enable debug logging with `export LOG_LEVEL=DEBUG`

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for details on:
- Reporting bugs & requesting features
- Development setup and workflow
- Submitting Pull Requests
- Code style guidelines
- Security issue reporting

### Quick Start for Contributors

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/aws-ai-ml-workshop-kr.git
cd aws-ai-ml-workshop-kr/genai/aws-gen-ai-kr/20_applications/08_bedrock_manus/use_cases/06_insight_extractor_strands_sdk_workshop_phase_1

# Create development environment
cd setup/
./create-uv-env.sh deep-insight 3.12

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and test
cd ..
uv run python main.py

# Commit and push
git add .
git commit -m "Add feature: description"
git push origin feature/your-feature-name
```

### Contribution Areas

- **New Agent Types**: Add specialized agents for specific domains
- **Tool Integration**: Create new tools for agent capabilities
- **Model Support**: Add support for additional LLM providers
- **Documentation**: Improve guides, examples, and tutorials
- **Bug Fixes**: Fix issues and improve stability
- **Performance**: Optimize streaming, caching, and execution

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for information on reporting security issues.

## Acknowledgments

### Open Source Foundation

Deep Insight is built on the shoulders of giants:

- **[LangManus](https://github.com/Darwin-lfl/langmanus)** - Original open-source framework that inspired this project
- **[Strands Agent SDK](https://github.com/strands-agents/sdk-python)** - Agent orchestration and LLM integration
- **[AgentCore](https://aws.amazon.com/agentcore/)** - MCP server integration and tool gateway
- **[Amazon Bedrock](https://aws.amazon.com/bedrock/)** - Managed LLM service

### Key Libraries

- **Data Analysis**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Document Generation**: WeasyPrint, ReportLab
- **UI**: Streamlit

### Philosophy

> **"Come From Open Source, Back to Open Source"**

We believe in the power of open collaboration. Deep Insight takes the excellent work of the LangManus community and extends it with AWS-native capabilities, then contributes those enhancements back to the community.

## Contributors

- **Dongjin Jang, Ph.D.** - AWS AI/ML Specialist Solutions Architect
  - [Email](mailto:dongjinj@amazon.com) | [LinkedIn](https://www.linkedin.com/in/dongjin-jang-kr/) | [GitHub](https://github.com/dongjin-ml) | [Hugging Face](https://huggingface.co/Dongjin-kr)

---

<div align="center">
  <p>
    <strong>Built with ❤️ by the AWS AI/ML team</strong><br>
    <sub>Empowering enterprises to build customizable agentic AI systems</sub>
  </p>
</div>
