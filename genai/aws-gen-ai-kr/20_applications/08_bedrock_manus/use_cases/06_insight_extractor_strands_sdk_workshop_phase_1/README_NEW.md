# Deep Insight

<p align="center">
    <a href="https://github.com/aws-samples">
        <img alt="Build" src="https://img.shields.io/badge/Contribution-Welcome-blue">
    </a>
    <a href="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green">
    </a>
    <a href="https://www.python.org/downloads/">
        <img alt="Python" src="https://img.shields.io/badge/python-3.12+-blue.svg">
    </a>
</p>

<p align="center">
    <strong>Customizable Reporting Agent Platform built on AgentCore and Strands Agent</strong>
</p>

---

## 🎯 The Big Picture

> **When Customization and Control Matter Most**
>
> Not all enterprises can rely on managed services. When you need deep workflow customization, regulatory compliance within your AWS VPC, or domain-specific agent behavior that goes beyond standard use cases, **Deep Insight** provides the open, customizable framework you need to build exactly what your unique situation demands.

### What is Deep Insight?

Deep Insight is a **customizable reporting agent platform** built on Strands Agent and AgentCore that enables enterprises to deploy, modify, and extend multi-agent systems within their own AWS environment. Unlike managed services that prioritize speed and simplicity, Deep Insight empowers organizations with unique workflows and stringent requirements to build and evolve their own agent infrastructure while maintaining full ownership of data, compute resources, and competitive differentiation.

### Why Deep Insight?

**The Problem**: Traditional reporting processes follow waterfall methodology, requiring significant time and manpower from topic selection to final report creation. Enterprises face unique challenges:
- Unique workflows requiring deep modification of agent behavior and logic
- Regulatory or security mandates requiring data/compute to remain within AWS environment
- Domain-specific requirements beyond standard analytical use cases
- Strategic need to own and evolve agent infrastructure as competitive differentiator

**The Solution**: Deep Insight transforms sequential workflows into multi-agent collaboration:
1. Describe your analysis needs in natural language
2. Multi-agents automatically perform data exploration to advanced statistical analysis
3. Generate professional reports with charts, calculations, and data sources for verification
4. Deploy and customize within your AWS VPC with full control over prompts, agents, and workflows

### Key Highlights

- **🔧 Full Customization**: Deploy open-source code in your AWS VPC and freely modify prompts, agents, and workflows for your specific needs
- **🤖 Hierarchical Multi-Agent System**: Coordinator, Planner, Supervisor orchestrate specialized agents (Coder, Validator, Reporter, Tracker) for systematic task execution
- **📊 Verifiable Insights**: Generated reports include calculation methods and original data sources for analysis reliability verification
- **🔗 Extensible Integration**: AgentCore Gateway enables connected analysis by combining with external data sources (e.g., YouTube API via MCP)
- **🚀 Beyond Reporting**: Extend the codebase to other agent use cases—customer service, shopping assistance, log analysis, and more
- **🔒 Enterprise Security**: Entire runtime operates within AWS Customer VPC, isolating data for sensitive workloads
- **⚡ Rapid Cycle**: Dramatically reduce waterfall sequential work structure through multi-agent collaboration

### Architecture at a Glance

```
User Query + Data File → Coordinator → Planner → Supervisor
                                                     ├─→ Coder (Data Analysis & Code Execution)
                                                     ├─→ Validator (Quality & Verification)
                                                     ├─→ Reporter (Professional Report Generation)
                                                     └─→ Tracker (Process Transparency)
                                                            ↓
                                                   AgentCore Gateway (MCP Tools)
                                                            ↓
                                                   PDF Report + Verification Files
```

### Deep Insight vs. Managed Services

| Aspect | Managed Services (e.g., QuickSight Q) | Deep Insight |
|--------|--------------------------------|--------------|
| **Deployment** | SaaS, managed platform | Open-source in your AWS VPC |
| **Customization** | Pre-built agents, limited modification | Full code access, deep customization |
| **Security** | Enterprise-grade, multi-tenant | Single-tenant VPC isolation |
| **Use Case** | General-purpose business tasks | Domain-specific, regulatory-sensitive workloads |
| **Integration** | 50+ connectors, MCP/OpenAPI | Custom integration, extensible to any agent use case |
| **Ownership** | Vendor-managed infrastructure | Full ownership of data, compute, and evolution |
| **Best For** | Speed and simplicity | Control and competitive differentiation |

---

## 🚀 Quick Start

Get up and running in 5 minutes:

```bash
# Clone the repository
git clone https://github.com/aws-samples/aws-ai-ml-workshop-kr.git
cd aws-ai-ml-workshop-kr/genai/aws-gen-ai-kr/20_applications/08_bedrock_manus/use_cases/06_insight_extractor_strands_sdk_workshop_phase_1

# Create UV environment (automatically installs dependencies)
cd setup/
./create-uv-env.sh deep-insight 3.12

# Run the framework (from project root)
cd ..
uv run python main.py
```

**Expected output:**
```
✓ Environment initialized
✓ Agents loaded: Coordinator, Planner, Supervisor
✓ Starting workflow...
[Streaming output will appear here in real-time]
```

<details>
<summary><strong>📖 Alternative Installation Methods</strong></summary>

### Option 2: Traditional Conda Environment

```bash
# Create and activate conda environment
cd setup/
./create_conda_virtual_env.sh deep-insight 3.12

# Activate environment
source .venv/bin/activate

# Run from project root
cd ..
python main.py
```

### Option 3: Jupyter Notebook

```bash
# Launch Jupyter Lab
jupyter lab main.ipynb
```

### Option 4: Streamlit UI

```bash
# Launch web interface
cd app/
streamlit run app.py
```

</details>

---

## 🎬 See It In Action

<details open>
<summary><strong>📹 Demo: Amazon Sales Data Analysis</strong></summary>

### Task Description

> "I would like to analyze Amazon product sales data. The target file is `./data/Amazon_Sale_Report.csv`. Please conduct comprehensive analysis to extract marketing insights—explore data attributes, product trends, variable relationships, and combinations. Include detailed analysis with supporting charts and save the final report as PDF."

### Demo Video

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

</details>

---

## 🏗️ Architecture

<details>
<summary><strong>📐 System Architecture Diagram</strong></summary>

![Deep Insight Architecture](./assets/architecture.png)

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

### Agent Responsibilities

| Agent | Role | Key Capabilities | Model Configuration |
|-------|------|-----------------|-------------------|
| **Coordinator** | Entry point | Understands user prompts, routes tasks | Claude Sonnet 4, no reasoning |
| **Planner** | Strategic planner | Establishes analysis plans, task decomposition | Claude Sonnet 4, reasoning enabled |
| **Supervisor** | Workflow orchestrator | Orchestrates overall workflow, agent coordination | Claude Sonnet 4, prompt caching |
| **Coder** | Code execution | Generates and executes data analysis code | Configurable |
| **Validator** | Quality assurance | Verifies analysis results, manages quality | Configurable |
| **Reporter** | Document generation | Generates final reports with professional formatting | Configurable |
| **Tracker** | Process transparency | Tracks entire process, ensures transparency at each stage | Configurable |

</details>

<details>
<summary><strong>⚙️ Key Technical Components</strong></summary>

### Streaming Architecture

The framework uses **event-based streaming** for real-time progress updates:

- `StreamableGraph` wraps Strands GraphBuilder with streaming capability
- Background task execution with thread-safe event queue
- Real-time event emission via `graph.stream_async()`
- Events consumed asynchronously for live UI updates

### Global State Management

Shared state system (`_global_node_states` in `src/graph/nodes.py`):
- Maintains conversation history across all agents
- Stores execution plans and accumulated context ("clues")
- Enables stateful communication in the agent workflow
- Thread-safe access for concurrent agent operations

### LLM Configuration

Flexible model selection via `strands_utils.get_model()`:
- **Claude Sonnet 4**: Latest high-performance model
- **Claude Sonnet 3.7**: Previous generation
- **Claude Sonnet 3.5 V2**: Alternative version
- Configurable reasoning, prompt caching, and streaming per agent

### Workflow Control Flow

Conditional edges define agent transitions:
1. `should_handoff_to_planner()` determines Coordinator → Planner
2. Fixed edge from Planner → Supervisor
3. Supervisor executes tool agents based on task requirements
4. Workflow completes when Supervisor marks task done

</details>

---

## ✨ Features

<details>
<summary><strong>🤖 LLM Integration</strong></summary>

### Multi-Model Support

Deep Insight integrates with all models available in Amazon Bedrock:

- **Anthropic Claude**: Sonnet 4, Sonnet 3.7, Sonnet 3.5, Opus, Haiku
- **Amazon Nova**: Nova Pro, Nova Lite
- **Meta Llama**: Llama 3.1, Llama 3.2
- **Mistral AI**: Mistral Large, Mistral Small
- **Cohere**: Command R, Command R+

### Intelligent Model Routing

Different agents use different models based on task requirements:
- **Coordinator**: Fast model for quick routing (no reasoning needed)
- **Planner**: Reasoning-enabled model for strategic thinking
- **Supervisor**: Balanced model with prompt caching for efficiency
- **Tool Agents**: Configurable per agent based on task complexity

### Prompt Caching

Reduce costs and latency with intelligent prompt caching:
- Automatic caching of system prompts
- Cache hit rates improve with repeated similar queries
- Configurable cache types per agent

</details>

<details>
<summary><strong>🐍 Python Integration</strong></summary>

### Built-in Python REPL

Execute Python code dynamically during workflow:

```python
# Coder agent can execute arbitrary Python
import pandas as pd
df = pd.read_csv('data.csv')
df.describe()
```

### Safe Code Execution

- Isolated execution environment
- Automatic dependency management
- Error handling and recovery
- Result capture and streaming

### Data Analysis Libraries

Pre-configured with popular data science libraries:
- **Data Manipulation**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Machine Learning**: Scikit-learn
- **Statistical Analysis**: SciPy, Statsmodels

</details>

<details>
<summary><strong>📊 Visualization & Reporting</strong></summary>

### Automated Report Generation

Generate professional reports with:
- Executive summaries
- Data visualizations (charts, graphs, heatmaps)
- Statistical tables
- Methodology explanations
- Recommendations and insights

### Multi-Format Output

Export reports in multiple formats:
- **PDF**: Publication-ready reports with Korean font support
- **HTML**: Interactive web-based reports
- **Markdown**: Developer-friendly documentation

### Visualization Capabilities

Create various chart types automatically:
- Line charts for trends
- Bar charts for comparisons
- Scatter plots for relationships
- Heatmaps for correlations
- Custom visualizations with Plotly

### Korean Language Support

Full support for Korean content:
- Korean font installation script (`setup/install_korean_font.sh`)
- Matplotlib Korean rendering (`koreanize-matplotlib`)
- UTF-8 encoding for all text processing

</details>

<details>
<summary><strong>🔄 Workflow Management</strong></summary>

### TODO-Based Planning

Planner creates structured task lists:
1. Break down complex queries into steps
2. Prioritize tasks based on dependencies
3. Track completion status
4. Adapt plan based on intermediate results

### Workflow Graph Visualization

Visualize agent interactions and task flow:
- Node-based graph representation
- Edge transitions showing control flow
- State snapshots at each step

### Multi-Agent Orchestration

Supervisor coordinates multiple agents:
- Parallel task execution where possible
- Sequential execution for dependent tasks
- Result aggregation and synthesis
- Error recovery and retry logic

### Progress Monitoring

Real-time tracking via Tracker agent:
- Current task status
- Completed steps
- Remaining work
- Estimated completion

</details>

---

## 🛠️ Setup

<details>
<summary><strong>📋 Prerequisites</strong></summary>

### System Requirements

- **Python**: 3.12 or higher
- **Operating System**: Linux, macOS, or Windows WSL2
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Disk Space**: 2GB for dependencies and artifacts

### Required Accounts & Credentials

- **AWS Account**: For Amazon Bedrock access
- **AWS Credentials**: Configured via AWS CLI or environment variables
- **Bedrock Model Access**: Ensure Claude Sonnet models are enabled in your AWS region

### Tested Environments

This framework has been tested on:
- Amazon SageMaker Studio Code Editor
- Amazon SageMaker Studio JupyterLab
- Ubuntu 22.04 LTS
- macOS Monterey and later

</details>

<details>
<summary><strong>💾 Installation</strong></summary>

### Step 1: Clone Repository

```bash
git clone https://github.com/aws-samples/aws-ai-ml-workshop-kr.git
cd aws-ai-ml-workshop-kr/genai/aws-gen-ai-kr/20_applications/08_bedrock_manus/use_cases/06_insight_extractor_strands_sdk_workshop_phase_1
```

### Step 2: Create Environment

**Recommended: UV Environment**

UV provides fast and reliable dependency management:

```bash
# Navigate to setup directory
cd setup/

# Create UV environment with Python 3.12
# This automatically installs all dependencies and creates symlinks
./create-uv-env.sh deep-insight 3.12

# Return to project root
cd ..
```

The script automatically:
- Creates virtual environment with UV
- Installs all dependencies from `pyproject.toml`
- Creates symlinks in root for easy execution
- Configures Python path

### Step 3: Configure AWS Credentials

```bash
# Option 1: AWS CLI configuration
aws configure

# Option 2: Environment variables
export AWS_REGION=us-west-2
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key

# Option 3: Use .env file
cp .env.example .env
# Edit .env with your credentials
```

### Step 4: (Optional) Install Korean Font

For PDF reports with Korean text:

```bash
cd setup/
./install_korean_font.sh
```

### Step 5: Verify Installation

```bash
# Test Python version
python --version  # Should show Python 3.12.x

# Test framework import
python -c "from src.graph.builder import build_graph; print('✓ Installation successful!')"

# Test Korean font (if installed)
python setup/test_korean_font.py
```

</details>

<details>
<summary><strong>🔧 Configuration</strong></summary>

### Environment Variables

Copy and edit the environment template:

```bash
cp .env.example .env
```

Available configuration options:

```bash
# AWS Configuration
AWS_REGION=us-west-2                    # AWS region for Bedrock
AWS_DEFAULT_REGION=us-west-2            # Fallback region

# Model Configuration
BEDROCK_MODEL_ID=claude-sonnet-4        # Default LLM model

# Output Settings
OUTPUT_DIR=./artifacts                   # Directory for generated reports

# Logging
LOG_LEVEL=INFO                          # DEBUG, INFO, WARNING, ERROR
```

### Agent Configuration

Customize agent behavior in `src/config/agents.py`:

```python
# Modify agent system prompts
# Enable/disable prompt caching
# Change model assignments
# Add custom tools
```

### Prompt Customization

Edit agent prompts in `src/prompts/`:
- `coordinator.md` - Entry point agent behavior
- `planner.md` - Planning strategy and reasoning
- `supervisor.md` - Task delegation logic
- `coder.md` - Code execution guidelines
- `reporter.md` - Report writing style

</details>

---

## 📖 Usage

<details>
<summary><strong>🎯 Basic Usage</strong></summary>

### CLI Execution

Run with default predefined query:

```bash
python main.py
```

Run with custom query:

```bash
python main.py --user_query "Analyze customer churn data and create a predictive model"
```

### Jupyter Notebook

```bash
jupyter lab main.ipynb
```

Execute cells to:
1. Initialize the framework
2. Submit your query
3. View streaming results
4. Access generated artifacts

### Streamlit UI

```bash
cd app/
streamlit run app.py
```

Web interface provides:
- Text input for queries
- Real-time streaming output
- Download links for artifacts
- Workflow visualization

</details>

<details>
<summary><strong>🎨 Advanced Usage Examples</strong></summary>

### Example 1: Data Analysis

```python
from src.graph.builder import build_graph

# Initialize graph
graph = build_graph()

# Submit analysis query
query = """
Analyze the sales data in ./data/sales.csv:
1. Calculate summary statistics
2. Identify top products
3. Create trend visualizations
4. Generate PDF report with findings
"""

result = await graph.stream_async(query)
```

### Example 2: Custom Model Selection

```python
# Use specific model for task
import os
os.environ['BEDROCK_MODEL_ID'] = 'claude-opus'

# Run with premium model
python main.py --user_query "Complex reasoning task"
```

### Example 3: Batch Processing

```python
queries = [
    "Analyze Q1 sales data",
    "Analyze Q2 sales data",
    "Analyze Q3 sales data",
    "Analyze Q4 sales data"
]

for query in queries:
    result = await graph.stream_async(query)
    # Results saved to ./artifacts/
```

### Example 4: Custom Tool Integration

Add custom tools to agents in `src/tools/`:

```python
# Create custom_tool.py
TOOL_SPEC = {
    "name": "custom_analyzer",
    "description": "Performs custom analysis",
    "inputSchema": {...}
}

def handle_custom_analyzer(data: str):
    # Your custom logic
    return result
```

Then register in agent configuration.

</details>

<details>
<summary><strong>📂 Output Files</strong></summary>

### Artifacts Directory

All generated outputs are saved to `./artifacts/`:

```
artifacts/
├── analysis_report.pdf       # Final PDF report with visualizations
├── analysis_report.html      # HTML version of report
├── analysis_report.md        # Markdown version
├── data_summary.json         # Structured analysis results
├── execution_log.txt         # Detailed execution log
└── visualizations/           # Generated charts
    ├── trend_chart.png
    ├── correlation_matrix.png
    └── distribution_plot.png
```

### Artifact Cleanup

The `artifacts/` directory is automatically cleaned at the start of each run by `remove_artifact_folder()` in `main.py`. To preserve artifacts:

```python
# Comment out in main.py:
# remove_artifact_folder()
```

Or manually backup before running:

```bash
cp -r artifacts/ artifacts_backup_$(date +%Y%m%d)/
```

</details>

---

## 💼 Business Impact

<details>
<summary><strong>📈 Market Opportunity & Use Cases</strong></summary>

### Transforming Traditional Workflows

Deep Insight transforms the traditional **waterfall sequential work structure** into multi-agent collaboration, dramatically reducing the long time required from topic selection to final report creation. The cycle of data collection, analysis, and insight derivation through AI is automated, enabling rapid business execution and response.

### Enterprise Use Cases

With simple natural language prompts, Deep Insight can handle:

**Sales & Marketing**
- Sales performance and trend analysis
- Target customer strategy establishment
- Campaign performance comparison and cause analysis
- Market penetration analysis across regions

**Business Intelligence**
- Market status and technology trend analysis
- Competitive landscape assessment
- Customer segmentation and behavior patterns
- Product portfolio optimization

**Operations & Strategy**
- Business process automation opportunities
- Resource allocation optimization
- Risk assessment and mitigation planning
- Performance KPI tracking and forecasting

### Real-World Example: Food Company Ad Campaign Analysis

A food company ran advertisements across multiple platforms (Amazon.com, Walmart.com) and provided Deep Insight with campaign results in CSV format.

**Prompt**: "Please analyze the advertising campaign and the sales generated through the advertisements based on the given ad campaign data."

**Results**: Deep Insight automatically:
1. Analyzed campaign performance across platforms
2. Identified top-performing products and channels
3. Calculated ROI and conversion metrics
4. Generated visualizations comparing platform effectiveness
5. Provided actionable recommendations for budget allocation
6. Created a professional PDF report ready for executive presentation

**Time Saved**: What traditionally took 2-3 days was completed in 15 minutes.

### Extensibility Beyond Reporting

The customizable codebase allows transformation into various agent applications:

- **Shopping Assistance Agents**: Product recommendation and comparison
- **Customer Support Agents**: Automated inquiry handling and ticket routing
- **Log Analysis Agents**: System monitoring and anomaly detection
- **Content Generation Agents**: Marketing copy and documentation creation

### Customer Reception

When introduced to enterprise customers through sales presentations:
- **70.5% expressed interest** in implementing the solution
- Customers highlighted the ability to **gain business insights within 15 minutes** using agentic AI
- Primary interest drivers: customization capability, VPC deployment, and extensibility

</details>

<details>
<summary><strong>🎯 Competitive Differentiation</strong></summary>

### Why Choose Deep Insight Over Managed Services?

**When Managed Services are Best:**
- Need immediate deployment with zero setup
- General-purpose analytical tasks
- Limited customization requirements
- Multi-tenant SaaS deployment is acceptable
- Want vendor-managed infrastructure

**When Deep Insight is the Right Choice:**
- Require deep customization of agent behavior and logic
- Have regulatory/compliance requirements for data isolation
- Need domain-specific workflows beyond standard use cases
- Want competitive differentiation through proprietary agent systems
- Prefer to own and evolve infrastructure as strategic asset
- Plan to extend beyond reporting to other agent use cases

### Strategic Benefits

1. **Knowledge Accumulation**: Build internal expertise in Strands Agent and AgentCore
2. **Competitive Edge**: Develop proprietary agent capabilities competitors can't replicate
3. **Cost Predictability**: Control infrastructure costs within your AWS account
4. **Data Sovereignty**: Maintain complete control over sensitive business data
5. **Innovation Speed**: Rapidly prototype new agent use cases on proven foundation

### ROI Considerations

**Initial Investment:**
- Development time to customize and deploy
- AWS infrastructure costs
- Team training on Strands/AgentCore

**Long-term Returns:**
- Reduced manual analysis time (70-90% time savings reported)
- Faster decision-making cycles
- Reusable platform for multiple agent use cases
- Strategic ownership of AI capabilities
- No per-user licensing fees

</details>

---

## 🔍 Troubleshooting

<details>
<summary><strong>❗ Common Issues</strong></summary>

### Installation Problems

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

If using an older version, install Python 3.12 or use conda:
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

---

**Problem**: `Region not supported` error

**Solution**: Ensure you're using a region where Bedrock is available:
```bash
export AWS_REGION=us-west-2  # or us-east-1
```

Supported regions: us-east-1, us-west-2, eu-west-1, ap-northeast-1

### Runtime Issues

**Problem**: PDF report generation fails

**Solution**: Install Korean font (if using Korean text):
```bash
cd setup/
./install_korean_font.sh
python test_korean_font.py
```

---

**Problem**: Out of memory errors

**Solution**: Reduce batch size or use streaming more aggressively:
```python
# In agent configuration
max_tokens = 2048  # Reduce from default 4096
```

---

**Problem**: Workflow hangs or times out

**Solution**: Check event queue and clear if needed:
```python
from src.utils.event_queue import clear_queue
clear_queue()
```

</details>

<details>
<summary><strong>💬 Getting Help</strong></summary>

### Resources

- **Documentation**: See [CLAUDE.md](CLAUDE.md) for developer documentation
- **Issues**: [Report bugs or request features](https://github.com/aws-samples/aws-ai-ml-workshop-kr/issues)
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines

### Debugging Tips

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python main.py
```

Check execution logs:
```bash
# View real-time logs
tail -f artifacts/execution_log.txt

# Search for errors
grep -i error artifacts/execution_log.txt
```

Validate configuration:
```bash
# Test AWS connection
aws bedrock list-foundation-models --region us-west-2

# Test model access
python -c "import boto3; client = boto3.client('bedrock-runtime', region_name='us-west-2'); print('✓ Connection successful')"
```

</details>

---

## 🤝 Contributing

<details>
<summary><strong>📝 How to Contribute</strong></summary>

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

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

</details>

---

## 👥 Contributors

- **Dongjin Jang, Ph.D.** - AWS AI/ML Specialist Solutions Architect
  - [Email](mailto:dongjinj@amazon.com) | [LinkedIn](https://www.linkedin.com/in/dongjin-jang-kr/) | [GitHub](https://github.com/dongjin-ml) | [Hugging Face](https://huggingface.co/Dongjin-kr)

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

### Open Source Foundation

Deep Insight is built on the shoulders of giants:

- **[LangManus](https://github.com/Darwin-lfl/langmanus)** - Original open-source framework that inspired this project
- **[Strands Agent SDK](https://github.com/anthropics/anthropic-sdk-python)** - Agent orchestration and LLM integration
- **[AgentCore](https://aws.amazon.com/agentcore/)** - MCP server integration and tool gateway
- **[LangChain](https://github.com/langchain-ai/langchain)** - LLM application framework
- **[Amazon Bedrock](https://aws.amazon.com/bedrock/)** - Managed LLM service

### Key Libraries

- **Data Analysis**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Document Generation**: WeasyPrint, ReportLab
- **UI**: Streamlit

### Philosophy

> **"Come From Open Source, Back to Open Source"**

We believe in the power of open collaboration. Deep Insight takes the excellent work of the LangManus community and extends it with AWS-native capabilities including Strands Agent SDK and AgentCore integration, then contributes those enhancements back to the community.

Special thanks to all contributors who make this project possible.

---

## 📚 Additional Resources

<details>
<summary><strong>🔗 Related Projects</strong></summary>

- [LangManus](https://github.com/Darwin-lfl/langmanus) - Original open-source framework
- [LangGraph](https://github.com/langchain-ai/langgraph) - Graph-based LLM workflows
- [AutoGen](https://github.com/microsoft/autogen) - Multi-agent conversation framework
- [CrewAI](https://github.com/joaomdmoura/crewAI) - Role-based agent orchestration

</details>

<details>
<summary><strong>📖 Learning Resources</strong></summary>

### Amazon Bedrock Documentation
- [Amazon Bedrock User Guide](https://docs.aws.amazon.com/bedrock/)
- [Model Access and Pricing](https://aws.amazon.com/bedrock/pricing/)
- [Best Practices for Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/best-practices.html)

### Agent Systems
- [Multi-Agent Systems Design Patterns](https://www.anthropic.com/research/building-effective-agents)
- [LangChain Agent Documentation](https://python.langchain.com/docs/modules/agents/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

### Data Science & Visualization
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)
- [Plotly Python Guide](https://plotly.com/python/)

</details>

---

<p align="center">
    <strong>Built with ❤️ by the AWS AI/ML team</strong><br>
    <sub>Empowering enterprises to build customizable agentic AI systems</sub>
</p>

---

<p align="center">
    <sub>Deep Insight: A customized tool designed and assembled to fit your hands.</sub>
</p>
