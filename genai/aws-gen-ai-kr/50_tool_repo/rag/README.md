# RAG Tool with OpenSearch and Strands Agent SDK

Advanced Retrieval-Augmented Generation (RAG) system using OpenSearch hybrid search, multimodal document processing, and Strands Agent SDK for intelligent question answering.

## Features

- **Hybrid Search**: Combines semantic (vector) and lexical (BM25) search with RRF (Reciprocal Rank Fusion)
- **Multimodal Support**: Processes and retrieves text, tables, and images from complex PDF documents
- **Advanced Retrieval**: Supports RAG Fusion, HyDE, Parent Document retrieval, and Reranker
- **Strands Agent Integration**: Uses Strands Agent SDK for intelligent answer generation with streaming
- **OpenSearch**: Scalable vector database with Korean language support (Nori plugin)

## Architecture

```
PDF Document
    ↓
Upstage Document Parse (SageMaker)
    ↓
Extract: Text | Tables | Images
    ↓
Summarize Tables/Images (Claude 4)
    ↓
Chunking: Parent (1024) → Child (256)
    ↓
Embed (Cohere V4, 1024-dim)
    ↓
Index to OpenSearch
    ↓
OpenSearchHybridSearchRetriever (Advanced Search)
    ↓
Strands Agent (Claude 4.5 Sonnet)
    ↓
Multimodal RAG Response
```

## Prerequisites

- AWS Account with access to:
  - Amazon Bedrock (Claude 4.5 Sonnet, Cohere Embed V4)
  - Amazon OpenSearch Service
  - Amazon SageMaker (for Upstage Document Parse)
- Python 3.12+
- `uv` package manager

## Quick Start

### 1. Setup Environment

```bash
# Navigate to project directory
cd /path/to/rag

# Create and setup virtual environment using uv
cd setup
./create-uv-env.sh tool-rag 3.12
cd ..

# Activate virtual environment (automatically done via symlink)
# The .venv symlink points to setup/.venv
```

### 2. Configure Environment Variables

Copy and edit the `.env` file:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```bash
# AWS Configuration
AWS_REGION=us-west-2
AWS_DEFAULT_REGION=us-west-2
AWS_ACCOUNT_ID=your-account-id

# SageMaker Configuration
SAGEMAKER_EXECUTION_ROLE_ARN=arn:aws:iam::YOUR_ACCOUNT_ID:role/service-role/AmazonSageMaker-ExecutionRole-XXXXXXXX
DOCUMENT_PARSE_ENDPOINT_NAME=Upstage-Document-Parse-Test

# Model Configuration
LLM_MODEL_NAME=Claude-V4-5-Sonnet-CRI
EMBEDDING_MODEL_NAME=Cohere-Embed-V4-CRI
EMBEDDING_DIMENSION=1024

# Chunking Configuration
PARENT_CHUNK_SIZE=1024
PARENT_CHUNK_OVERLAP=0
CHILD_CHUNK_SIZE=256
CHILD_CHUNK_OVERLAP=64

# OpenSearch Configuration (will be auto-populated after OpenSearch setup)
INDEX_NAME=complex-doc-index
OPENSEARCH_VERSION=3.1
OPENSEARCH_DOMAIN_NAME=my-opensearch-domain
OPENSEARCH_DOMAIN_ENDPOINT=https://your-opensearch-domain.us-west-2.es.amazonaws.com
OPENSEARCH_USER_ID=your-opensearch-username
OPENSEARCH_USER_PASSWORD=your-opensearch-password
```

### 3. Setup OpenSearch

Create an OpenSearch domain with Korean language support:

```bash
cd opensearch

# Make script executable
chmod +x create-opensearch.sh

# Run setup (uses .env configuration)
uv run ./create-opensearch.sh
```

**Setup Time**: 30-40 minutes (dev mode), 50-60 minutes (prod mode)

The script will:
- Create OpenSearch domain with Nori plugin (Korean language analyzer)
- Configure security settings
- Auto-update `.env` with `OPENSEARCH_DOMAIN_ENDPOINT`

For detailed OpenSearch setup guide, see [opensearch/opensearch.md](opensearch/opensearch.md).

### 4. Document Indexing

#### 4.1 Create Upstage Document Parse Endpoint

First, create the SageMaker endpoint for document parsing:

📓 **[1.create_endpoint_upstage_document_parse.ipynb](document_parser/1.create_endpoint_upstage_document_parse.ipynb)**

Follow the notebook to:
- Deploy Upstage Document Parse endpoint on SageMaker
- Update `.env` with the endpoint name: `DOCUMENT_PARSE_ENDPOINT_NAME`

#### 4.2 Index Documents

**Option A: Using Python Script (Recommended for Production)**

```bash
cd document_parser

# Index a single document
uv run python 2.script_document_indexing.py --file_path ../data/sample.pdf --output_dir ./output

# The script will:
# 1. Parse PDF with Upstage Document Parse
# 2. Extract text, tables, and images
# 3. Summarize tables/images with Claude
# 4. Create parent-child chunks
# 5. Generate embeddings with Cohere
# 6. Index to OpenSearch
```

**Option B: Using Notebook (Recommended for Learning)**

📓 **[2.notebook_document_indexing_opensearch.ipynb](document_parser/2.notebook_document_indexing_opensearch.ipynb)**

Follow sections 1-5 to:
- Understand the indexing pipeline
- See visualizations of extracted content
- Test each step interactively

**Option C: Using SageMaker Processing Job (Recommended for Batch)**

For large-scale document processing:
1. Open `2.notebook_document_indexing_opensearch.ipynb`
2. Navigate to "SageMaker Processing Job" section
3. Follow steps to create Docker image and run distributed indexing

For detailed indexing guide, see [document_parser/USAGE.md](document_parser/USAGE.md).

#### 4.3 Test Search

📓 **[3.search_test_opensearch.ipynb](document_parser/3.search_test_opensearch.ipynb)**

Test the following:
- Hybrid search (semantic + lexical)
- RAG question answering
- Custom queries

### 5. Using RAG Tool in Your Agent

For detailed usage examples and integration guides, see the interactive tutorial:

📓 **[how_to_use.ipynb](how_to_use.ipynb)** - Complete guide with examples:
- Standalone RAG tool testing
- Integration with Strands Agent
- Tool specification and configuration
- Real-world usage examples with sample queries

## Configuration

### Search Parameters

Edit `src/tools/rag_tool.py` to customize search behavior:

```python
opensearch_hybrid_retriever = OpenSearchHybridSearchRetriever(
    # Fusion algorithm
    fusion_algorithm="RRF",  # or "simple_weighted"
    ensemble_weights=[.51, .49],  # [semantic, lexical] weights

    # Advanced features
    reranker=False,  # Enable if you have reranker endpoint
    parent_document=True,  # Retrieve full parent chunks
    complex_doc=True,  # Support tables and images

    # Optional query augmentation
    # rag_fusion=True,  # Multiple query generation
    # hyde=True,  # Hypothetical document embeddings

    # Search options
    k=2,  # Number of documents to retrieve
    async_mode=True,
    verbose=False,
)
```

### Chunking Parameters

Edit `.env` to adjust chunking:

```bash
# Parent chunks (indexed for metadata)
PARENT_CHUNK_SIZE=1024
PARENT_CHUNK_OVERLAP=0

# Child chunks (indexed for retrieval)
CHILD_CHUNK_SIZE=256
CHILD_CHUNK_OVERLAP=64
```

## Advanced Features

### 1. Multimodal RAG

The RAG tool automatically handles multimodal content:

```python
# Retriever returns: (documents, tables, images)
retrieval, tables, images = retriever.invoke(query)

# Images and tables are automatically:
# 1. Decoded from base64 to bytes
# 2. Wrapped in Strands ContentBlock
# 3. Sent to Claude 4.5 Sonnet with text context
# 4. Agent analyzes visual content for better answers
```

### 2. RAG Fusion

Enable multiple query generation for better recall:

```python
opensearch_hybrid_retriever = OpenSearchHybridSearchRetriever(
    rag_fusion=True,  # Enable RAG Fusion
    query_augmentation_size=3,  # Generate 3 similar queries
    llm_text=llm_text,  # LLM for query generation
    # ...
)
```

### 3. HyDE (Hypothetical Document Embeddings)

Generate hypothetical answers for better semantic search:

```python
opensearch_hybrid_retriever = OpenSearchHybridSearchRetriever(
    hyde=True,  # Enable HyDE
    hyde_query=["web_search"],  # Query type
    llm_text=llm_text,  # LLM for hypothetical doc generation
    # ...
)
```

### 4. Reranker

Re-rank results with a cross-encoder model:

```python
opensearch_hybrid_retriever = OpenSearchHybridSearchRetriever(
    reranker=True,
    reranker_endpoint_name="your-reranker-endpoint",
    # ...
)
```

Note: Requires a SageMaker endpoint with a reranker model (e.g., cross-encoder).

## Project Structure

```
rag/
├── README.md                    # This file
├── .env                         # Environment configuration
├── .env.example                 # Example configuration
├── setup/                       # Dependency management
│   ├── pyproject.toml          # uv project file
│   └── uv.lock                 # Locked dependencies
├── opensearch/                  # OpenSearch setup
│   ├── opensearch.md           # Setup guide
│   ├── create-opensearch.sh    # Setup script
│   └── create-opensearch.py    # Python implementation
├── document_parser/             # Document indexing
│   ├── USAGE.md                # Detailed usage guide
│   ├── 1.create_endpoint_upstage_document_parse.ipynb
│   ├── 2.script_document_indexing.py
│   ├── 2.notebook_document_indexing_opensearch.ipynb
│   └── 3.search_test_opensearch.ipynb
├── src/
│   ├── tools/
│   │   └── rag_tool.py         # RAG tool implementation
│   └── utils/
│       ├── agentic_rag.py      # OpenSearchHybridSearchRetriever
│       ├── bedrock/            # Bedrock utilities
│       ├── opensearch/         # OpenSearch utilities
│       └── strands_sdk_utils/  # Strands SDK helpers
└── data/                        # Sample documents
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**
   ```bash
   cd setup
   ./create-uv-env.sh tool-rag 3.12
   ```

2. **OpenSearch connection error**
   - Check `.env` credentials
   - Verify security group allows your IP
   - Confirm domain is active in AWS Console

3. **Bedrock model access denied**
   - Enable models in AWS Bedrock console
   - Check IAM permissions for `bedrock:InvokeModel`

4. **Document Parse endpoint not found**
   - Verify endpoint name in `.env`
   - Check endpoint status in SageMaker console
   - Ensure endpoint is in same region

5. **Reranker error (404)**
   - Set `reranker=False` if you don't have a reranker endpoint
   - Or deploy a reranker model to SageMaker

6. **Out of memory during indexing**
   - Reduce chunk sizes in `.env`
   - Process fewer documents at once
   - Use SageMaker Processing Job for large batches

## Performance Optimization

### Search Speed

```python
# Faster search with fewer documents
opensearch_hybrid_retriever = OpenSearchHybridSearchRetriever(
    k=2,  # Reduce from default 3
    async_mode=True,  # Enable async
    parent_document=False,  # Disable if not needed
)
```

### Cost Optimization

```python
# Reduce Bedrock API calls
opensearch_hybrid_retriever = OpenSearchHybridSearchRetriever(
    rag_fusion=False,  # Disable query augmentation
    hyde=False,  # Disable hypothetical doc generation
    reranker=False,  # Disable reranker
)
```

### Accuracy Improvement

```python
# Better results with more documents and reranking
opensearch_hybrid_retriever = OpenSearchHybridSearchRetriever(
    k=10,  # Retrieve more candidates
    reranker=True,  # Re-rank with cross-encoder
    rag_fusion=True,  # Multiple query perspectives
    ensemble_weights=[0.7, 0.3],  # Favor semantic search
)
```

## Models Used

- **LLM**: Claude 4.5 Sonnet (16,384 token output)
- **Embeddings**: Cohere Embed V4 (1024 dimensions)
  - `input_type: "search_document"` for indexing
  - `input_type: "search_query"` for searching
- **Document Parser**: Upstage Document Parse (SageMaker)

## Cost Estimation

### OpenSearch
- **Dev mode**: ~$50/month (t3.small.search, 1 AZ)
- **Prod mode**: ~$200/month (t3.small.search, 3 AZ with standby)

### Bedrock (per 1000 documents)
- **Claude 4.5 Sonnet**: ~$15 (for summarization)
- **Cohere Embed V4**: ~$0.10 (for embeddings)
- **Per query**: ~$0.01-0.05 depending on context size

### SageMaker
- **Upstage Endpoint**: ~$100/month (ml.g4dn.xlarge)
- **Processing Job**: Pay per use (~$2/hour)

## References

- [OpenSearch Documentation](https://opensearch.org/docs/latest/)
- [Strands Agent SDK](https://github.com/anthropics/strands)
- [AWS Bedrock](https://aws.amazon.com/bedrock/)
- [Upstage Document Parse](https://www.upstage.ai/feed/product/document-parse-api)

## License

This project is part of the AWS AI/ML Workshop Korea repository.

## Contributing

For issues or questions, please refer to the main workshop repository:
https://github.com/aws-samples/aws-ai-ml-workshop-kr

## Contributors

| Name | Role | Contact |
|------|------|---------|
| **Dongjin Jang, Ph.D.** | AWS Sr. AI/ML Specialist SA | [Email](mailto:dongjinj@amazon.com) · [LinkedIn](https://www.linkedin.com/in/dongjin-jang-kr/) · [GitHub](https://github.com/dongjin-ml) · [Hugging Face](https://huggingface.co/Dongjin-kr) |

---

<div align="center">
  <p>
    <strong>Built with ❤️ by AWS KOREA SA Team</strong><br>
    <sub>Empowering developers with advanced RAG capabilities on AWS</sub>
  </p>
</div>
