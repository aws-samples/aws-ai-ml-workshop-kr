<h1 align="left"><b>Drug Discovery Agent based on Amazon Bedrock</b></h1>

- - -

## Overview

The Drug Discovery Agent is an AI-powered agent designed to assist pharmaceutical researchers in exploring scientific literature, clinical trials, and drug databases. This tool leverages Amazon Bedrock's large language models to provide interactive conversations about drug discovery, target proteins, diseases, and related research.

## Features

- **Interactive Chat Interface**: Engage in natural language conversations about drug discovery topics
- **Multiple Data Sources**: Access information from various scientific databases:
  - arXiv (scientific papers)
  - PubMed (biomedical literature)
  - ChEMBL (bioactive molecules)
  - ClinicalTrials.gov (clinical trials)
  - Web search via Tavily

- **Comprehensive Analysis**: Get detailed information about:
  - Target proteins and their inhibitors
  - Disease mechanisms
  - Drug candidates and their properties
  - Clinical trial results
  - Recent research findings

## Getting Started

### Prerequisites
- Required Python packages (install using `pip install -r requirements.txt`)
- AWS credentials configured
- API keys for external services (Tavily)

### Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your API keys to the `.env` file
   - Download ttf file for a font you will use and move it to `assets/` and change `font_path` in `chat.py`

### Running the Application

1. Start the MCP servers (Model Context Protocol servers that connect to external data sources):
   ```
   python application/launcher.py
   ```
   This will launch all necessary MCP servers for arXiv, PubMed, ChEMBL, ClinicalTrials.gov, and Tavily.

2. Start the Streamlit web interface:
   ```
   streamlit run application/app.py
   ```

3. Open your browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

## Using the Drug Discovery Agent

1. **Select a Model**: Choose from available foundation models (Claude 3.7 Sonnet, Claude 3.5 Sonnet, or Claude 3.5 Haiku)

2. **Ask Questions**: Examples of questions you can ask:
   - "Please generate a report for HER2 including recent news, recent research, related compounds, and ongoing clinical trials."
   - "Find recent research papers about BRCA1 inhibitors"
   - "What are the most promising drug candidates for targeting coronavirus proteins?"
   - "Summarize the mechanism of action for HER2 targeted therapies"
   
3. **Generate Reports**: The agent can compile comprehensive reports about specific targets or diseases

## Architecture

The Drug Discovery Agent is built using:

- **Strands Agent SDK**: For creating AI agents with specific capabilities
- **Streamlit**: For the web interface
- **MCP (Model Context Protocol)**: For connecting to external data sources
- **Amazon Bedrock**: For accessing powerful language models like Claude

Each MCP server provides specialized tools for accessing different scientific databases:
- `mcp_server_arxiv.py`: Search and retrieve scientific papers from arXiv
- `mcp_server_chembl.py`: Access chemical and bioactivity data from ChEMBL
- `mcp_server_clinicaltrial.py`: Search and analyze clinical trials
- `mcp_server_pubmed.py`: Access biomedical literature from PubMed
- `mcp_server_tavily.py`: Perform web searches for recent information

## Limitations
- This repository is intended for Proof of Concept (PoC) and demonstration purposes only. It is NOT intended for commercial or production use.
- The agent relies on external APIs which may have rate limits
- Information is limited to what's available in the connected databases

## Future Enhancements
- Integration with additional drug discovery tools and databases
- Enhanced visualization of molecular structures and interactions
- Support for proprietary research databases

## Contributors
- Hasun Yu, Ph.D. (AWS AI/ML Specialist Solutions Architect) | [Mail](mailto:hasunyu@amazon.com) | [LinkedIn](https://www.linkedin.com/in/hasunyu/)

## Citation
- If you find this repository useful, please consider giving a star ⭐ and citation

## References
- [Strands Agents SDK](https://strandsagents.com/0.1.x/)
- [Strands Agents Samples](https://github.com/strands-agents/samples/tree/main)
- [Strands Agents Samples - Korean](https://github.com/kyopark2014/strands-agent)