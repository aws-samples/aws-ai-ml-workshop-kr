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
    ◆ <a href="#demo">Demo</a>
    ◆ <a href="#installation">Installation</a>
    ◆ <a href="#architecture">Architecture</a>
    ◆ <a href="#usage">Usage</a>
  </p>
</div>

## *Latest News* 🔥

- **[2025/01]** Released Deep Insight framework built on Strands SDK and Amazon Bedrock with hierarchical multi-agent architecture
- **[2025/01]** Added support for Claude Sonnet 4.5 with enhanced reasoning capabilities
- **[2025/01]** Integrated AgentCore Gateway for extensible MCP tool integration

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
# 1. Clone and setup (see Installation for details)
git clone https://github.com/aws-samples/aws-ai-ml-workshop-kr.git
cd aws-ai-ml-workshop-kr/genai/aws-gen-ai-kr/20_applications/08_bedrock_manus/use_cases/06_insight_extractor_strands_sdk_workshop_phase_1
cd setup/ && ./create-uv-env.sh deep-insight 3.12 && cd ..

# 2. Run your analysis
uv run python main.py --user_query "너가 작성할 것은 moon market 의 판매 현황 보고서야. 세일즈 및 마케팅 관점으로 분석을 해주고, 차트 생성 및 인사이트도 뽑아서 pdf 파일로 만들어줘. 분석대상은 './data/Dat-fresh-food-claude.csv' 파일 입니다."
```

> **Note**: Requires Python 3.12+ and AWS credentials configured (tested in us-west-2 region).

## Demo

### Amazon Sales Data Analysis

> **Task**: "I would like to analyze Amazon product sales data. The target file is `./data/Amazon_Sale_Report.csv`. Please conduct comprehensive analysis to extract marketing insights—explore data attributes, product trends, variable relationships, and combinations. Include detailed analysis with supporting charts and save the final report as PDF."

[![Demo](./assets/demo.gif)](https://youtu.be/DwWICGLEv14)

[▶️ Watch Full Demo on YouTube](https://youtu.be/DwWICGLEv14)

### Sample Outputs

- 📄 [English Report (6 pages)](./assets/report_en.pdf)
- 📄 [Korean Report (10 pages)](./assets/report.pdf)
- 📊 Dataset: [Amazon Sale Report from Kaggle](https://www.kaggle.com/datasets/thedevastator/unlock-profits-with-e-commerce-sales-data)

### Output Structure

Results are automatically saved to `./artifacts/` directory:

```
artifacts/
├── analysis_report.pdf       # Final PDF report
├── data_summary.json         # Structured results
└── visualizations/           # Generated charts
    ├── trend_chart.png
    └── correlation_matrix.png
```

## Installation

### Environment Setup

```bash
# Navigate to setup directory
cd setup/

# Create UV environment with Python 3.12
./create-uv-env.sh deep-insight 3.12

# Return to project root and run
cd ..
uv run python main.py --user_query "Your analysis request here"
```

### Configure AWS Credentials

**Option 1: AWS CLI (Recommended)**

```bash
aws configure
```

**Option 2: Environment Variables**

```bash
# Direct export
export AWS_REGION=us-west-2
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key

# OR use .env file
cp .env.example .env
# Edit .env with your AWS credentials
```

## Architecture

### System Overview

![Deep Insight Architecture](./assets/architecture.png)

### Three-Tier Agent Hierarchy

```
┌─────────────────────────────────────────────────────────┐
│                     User Input                          │
│              (Natural Language Query)                   │
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
└──────────┬──────────┬──────────┬────────────────────────┘
           │          │          │          │
     ┌─────┘    ┌─────┘    ┌─────┘    ┌─────┘
     ▼          ▼          ▼          ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────┐
│  CODER  │ │REPORTER │ │TRACKER  │ │VALIDATOR │
│         │ │         │ │         │ │          │
│ Python  │ │ Report  │ │Progress │ │ Quality  │
│ Bash    │ │ Format  │ │Monitor  │ │ Validate │
│ Analysis│ │ Generate│ │ State   │ │ Verify   │
└─────────┘ └─────────┘ └─────────┘ └──────────┘
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

| State Field | Purpose |
|-------------|---------|
| `messages` | Current conversation messages for each agent |
| `request` | User's original request |
| `full_plan` | Planner's execution plan |
| `clues` | Accumulated context from agent executions |
| `history` | List of agent interactions with format `{"agent": "name", "message": "text"}` |

This enables stateful communication across the entire workflow.

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

### AWS & Bedrock Issues

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

**Problem**: Workflow hangs or times out

**Solution**: Check event queue and clear if needed:
```python
from src.utils.event_queue import clear_queue
clear_queue()
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for details on:
- Reporting bugs & requesting features
- Development setup and workflow
- Submitting Pull Requests
- Code style guidelines
- Security issue reporting

### Quick Start for Contributors

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/aws-ai-ml-workshop-kr.git
cd aws-ai-ml-workshop-kr/genai/aws-gen-ai-kr/20_applications/08_bedrock_manus/use_cases/06_insight_extractor_strands_sdk_workshop_phase_1

# Follow installation steps above to set up your environment

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes, test, then commit and push
git add .
git commit -m "Add feature: description"
git push origin feature/your-feature-name

# Open a Pull Request on GitHub
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
- **[Amazon Bedrock](https://aws.amazon.com/bedrock/)** - Managed LLM service

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
