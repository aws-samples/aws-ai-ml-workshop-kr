# Document Indexing and Search - Usage Guide

This directory contains tools for indexing complex PDF documents (with text, tables, and images) into OpenSearch and testing search functionality.

## Files Overview

### 1. Configuration
- **`.env`** (in parent directory): Contains all configuration parameters
  - AWS credentials and region
  - Model selection (LLM and embeddings)
  - Chunking parameters
  - OpenSearch connection details
  - SageMaker endpoint names

### 2. Setup
- **`1.create_endpoint_upstage_document_parse.ipynb`**: Guide for creating the Upstage Document Parse SageMaker endpoint (prerequisite)

### 3. Indexing (Choose One)
- **`2.script_document_indexing.py`**: Command-line script for indexing documents
  - Best for: Production use, automation, processing multiple files
  - Usage: `python 2.script_document_indexing.py --file_path ../data/sample.pdf --output_dir ./output`

- **`2.notebook_document_indexing_opensearch.ipynb`**: Interactive notebook for indexing
  - Best for: Learning, debugging, experimenting with parameters
  - Contains detailed explanations and visualizations
  - **Also includes**: SageMaker Processing Job for batch/large-scale indexing

### 4. Search Testing
- **`3.search_test_opensearch.ipynb`**: Test search functionality after indexing
  - Hybrid search testing (semantic + lexical)
  - RAG (Retrieval-Augmented Generation) testing
  - Custom query testing

## Quick Start

### Prerequisites
1. Configure `.env` file in the parent directory with your credentials
2. Create Upstage Document Parse endpoint (see notebook 1)
3. Set up OpenSearch domain
4. Install dependencies: `uv sync` (in setup directory)

### Workflow

#### Option A: Using Python Script (Recommended for Production)
```bash
# Navigate to document_parser directory
cd /path/to/rag/document_parser

# Index a document
python 2.script_document_indexing.py --file_path ../data/sample.pdf --output_dir ./output

# Then open 3.search_test_opensearch.ipynb to test search
```

#### Option B: Using Notebooks (Recommended for Learning)
```bash
# 1. Open and run 2.notebook_document_indexing_opensearch.ipynb
#    - This will index your document into OpenSearch
#    - Follow the cells sequentially (sections 1-5)
#    - Optional: Use SageMaker Processing Job section for batch processing

# 2. Open and run 3.search_test_opensearch.ipynb
#    - Test hybrid search with sample queries
#    - Try RAG question-answering
#    - Experiment with custom queries
```

#### Option C: Using SageMaker Processing Job (Recommended for Batch/Large-scale)
```bash
# 1. Open 2.notebook_document_indexing_opensearch.ipynb
# 2. Follow sections 1-5 to understand the indexing process
# 3. Jump to "SageMaker Processing Job" section
# 4. Run cells to:
#    - Create preprocessing script
#    - Build Docker image for SageMaker
#    - Upload documents to S3
#    - Launch SageMaker Processing Job
```

**When to use SageMaker Processing Job:**
- Processing large volumes of documents
- Need for scalable, distributed processing
- Scheduled batch jobs
- Integration with ML pipelines

## Key Features

### Document Processing
- **PDF Parsing**: Uses Upstage Document Parse for complex layouts
- **Multi-modal**: Extracts text, tables, and images separately
- **Summarization**: Uses Claude 4 to summarize tables/images for better retrieval
- **Chunking**: Parent-child chunking strategy (configurable via .env)

### Search Capabilities
- **Hybrid Search**: Combines semantic (vector) and lexical (keyword) search
- **RRF Fusion**: Reciprocal Rank Fusion algorithm for merging results
- **Parent Document Retrieval**: Returns full parent chunks for better context
- **Complex Document Support**: Retrieves text, tables, and images

### Models Used
- **LLM**: Claude 4.5 Sonnet (16,384 token output)
- **Embeddings**: Cohere Embed V4 (1024 dimensions)
  - Uses `input_type: "search_document"` for indexing
  - Uses `input_type: "search_query"` for searching

## Configuration Parameters

All parameters are in `../.env`:

### Chunking
- `PARENT_CHUNK_SIZE`: 1024 (size of parent chunks)
- `PARENT_CHUNK_OVERLAP`: 0 (overlap for parent chunks)
- `CHILD_CHUNK_SIZE`: 256 (size of child chunks for retrieval)
- `CHILD_CHUNK_OVERLAP`: 64 (overlap for child chunks)

### Search
- Fusion algorithm: RRF
- Ensemble weights: [0.51, 0.49] (semantic vs lexical)
- K: 10 (number of documents to retrieve)

## Output

### 2.script_document_indexing.py
- **Console**: Colored output showing progress through pipeline stages
- **OpenSearch**: Documents indexed in the configured index
- **Files**: Extracted images saved to output directory

### Notebooks
- **Visualizations**: Tables and images displayed inline
- **Search Results**: Formatted results with context
- **Metrics**: Document counts, processing times

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Run `uv sync` in the setup directory
2. **OpenSearch connection error**: Check credentials in .env
3. **Bedrock model access**: Ensure models are enabled in your AWS account
4. **Document Parse endpoint**: Verify endpoint name in .env matches your SageMaker endpoint

### Validation

Test your setup:
```python
# In Python or notebook
from dotenv import load_dotenv
import os

load_dotenv('../.env')
print(f"Region: {os.getenv('AWS_DEFAULT_REGION')}")
print(f"LLM: {os.getenv('LLM_MODEL_NAME')}")
print(f"Embedding: {os.getenv('EMBEDDING_MODEL_NAME')}")
print(f"Index: {os.getenv('INDEX_NAME')}")
```

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
Hybrid Search (Semantic + Lexical + RRF)
    ↓
RAG with Claude 4
```

## Next Steps

1. Index your first document using the script or notebook
2. Test search functionality with various queries
3. Experiment with chunking parameters in .env
4. Try different search configurations (weights, k value)
5. Integrate into your application via the rag_tool.py wrapper

## Indexing Options Comparison

| Feature | Python Script | Notebook (Local) | SageMaker Processing Job |
|---------|--------------|------------------|-------------------------|
| **Best For** | Production/Automation | Learning/Debugging | Batch/Large-scale |
| **Ease of Use** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Scalability** | Single machine | Single machine | Distributed |
| **Cost** | Local compute | Local compute | SageMaker pricing |
| **Monitoring** | Console output | Notebook output | CloudWatch logs |
| **Scheduling** | Cron/scripts | Manual | SageMaker Pipelines |
| **Setup Complexity** | Low | Low | High (Docker, ECR) |

## Notes

- The script (`2.script_document_indexing.py`) and notebook (sections 1-5) are functionally equivalent for indexing
- Notebook 3 (`3.search_test_opensearch.ipynb`) is specifically for testing after indexing is complete
- SageMaker Processing Job section in notebook 2 is for production batch processing
- All configuration is centralized in .env for easy management
- Extracted images are saved locally for inspection (or to S3 in SageMaker mode)
