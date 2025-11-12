import os
import sys
import asyncio
import base64
from pathlib import Path

module_path = "../.."
sys.path.append(os.path.abspath(module_path))

import boto3
import logging
from textwrap import dedent
from typing import Any, Annotated
from dotenv import load_dotenv

from langchain_aws import ChatBedrock
from langchain_aws import BedrockEmbeddings

from src.utils import bedrock
from src.utils.bedrock import bedrock_info
from src.utils.opensearch import opensearch_utils
from src.utils.strands_sdk_utils import strands_utils
from src.utils.agentic_rag import OpenSearchHybridSearchRetriever

from src.tools.decorators import log_io
from strands.types.tools import ToolResult, ToolUse
from strands.types.content import ContentBlock, ImageContent

# Load .env file
env_path = Path(__file__).parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded configuration from {env_path}")
else:
    print(f"Warning: .env file not found at {env_path}")

# Logger setup
logger = logging.getLogger(__name__)
logger.propagate = False
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
handler = logging.StreamHandler()
formatter = logging.Formatter('\n%(levelname)s [%(name)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

TOOL_SPEC = {
    "name": "rag_tool",
    "description": "Use this tool when you need to answer questions about specific documents or knowledge base content. This tool performs Retrieval-Augmented Generation (RAG) by searching through indexed documents in OpenSearch and generating contextual answers. Use when the user asks questions that require information from the knowledge base or when you need to retrieve specific facts from documents.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question or query to search for in the knowledge base and generate an answer."
                }
            },
            "required": ["query"]
        }
    }
}

class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

#######################################################
## OpenSearch Configuration
#######################################################
def opensearch_config():
    """Load OpenSearch configuration from .env"""
    opensearch_domain_endpoint = os.getenv('OPENSEARCH_DOMAIN_ENDPOINT')
    opensearch_user_id = os.getenv('OPENSEARCH_USER_ID')
    opensearch_user_password = os.getenv('OPENSEARCH_USER_PASSWORD')
    index_name = os.getenv('INDEX_NAME', 'complex-doc-index')

    if not all([opensearch_domain_endpoint, opensearch_user_id, opensearch_user_password]):
        raise ValueError(
            "OpenSearch credentials not found in .env file. "
            "Please set OPENSEARCH_DOMAIN_ENDPOINT, OPENSEARCH_USER_ID, and OPENSEARCH_USER_PASSWORD"
        )

    aws_region = os.getenv("AWS_DEFAULT_REGION", "us-west-2")
    http_auth = (opensearch_user_id, opensearch_user_password)
    os_client = opensearch_utils.create_aws_opensearch_client(
        aws_region,
        opensearch_domain_endpoint,
        http_auth
    )

    return os_client, index_name

#######################################################
## LangChain Models (for OpenSearchHybridSearchRetriever)
#######################################################
def get_model():
    """Load LangChain LLM and Embeddings for OpenSearchHybridSearchRetriever"""

    # Get model names from environment variables
    llm_model_name = os.getenv('LLM_MODEL_NAME', 'Claude-V4-5-Sonnet-CRI')
    embedding_model_name = os.getenv('EMBEDDING_MODEL_NAME', 'Cohere-Embed-V4-CRI')
    embedding_dimension = int(os.getenv('EMBEDDING_DIMENSION', '1024'))

    boto3_bedrock = bedrock.get_bedrock_client(
        assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
        endpoint_url=os.environ.get("BEDROCK_ENDPOINT_URL", None),
        region=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"),
    )

    llm_text = ChatBedrock(
        model_id=bedrock_info.get_model_id(model_name=llm_model_name),
        client=boto3_bedrock,
        streaming=True,
        model_kwargs={
            "max_tokens": 16384,
        }
    )

    llm_emb = BedrockEmbeddings(
        client=boto3_bedrock,
        model_id=bedrock_info.get_model_id(model_name=embedding_model_name),
        model_kwargs={
            "output_dimension": embedding_dimension,
            "embedding_types": ["float"],
            "input_type": "search_document"
        }
    )

    logger.info(f"{Colors.GREEN}Bedrock Models Loaded - LLM: {llm_model_name}, Embedding: {embedding_model_name}{Colors.END}")

    return llm_text, llm_emb

#######################################################
## OpenSearch Hybrid Search Retriever Setup
#######################################################
def get_retriever():
    """Create OpenSearchHybridSearchRetriever with advanced search features"""

    os_client, index_name = opensearch_config()
    llm_text, llm_emb = get_model()

    opensearch_hybrid_retriever = OpenSearchHybridSearchRetriever(
        os_client=os_client,
        index_name=index_name,
        llm_text=llm_text,  # For query augmentation (RAG Fusion, HyDE)
        llm_emb=llm_emb,    # For semantic search

        # Fusion algorithm
        fusion_algorithm="RRF",  # ["RRF", "simple_weighted"]
        ensemble_weights=[.51, .49],  # [semantic, lexical] weights

        # Advanced features
        reranker=True,  # Enable reranker if needed
        reranker_endpoint_name="cohere-reranker-3-5",
        parent_document=True,  # Enable parent document retrieval
        complex_doc=True,  # Support tables and images

        # Optional advanced query techniques
        # rag_fusion=True,  # Enable RAG Fusion for query augmentation
        # hyde=True,  # Enable HyDE (Hypothetical Document Embeddings)

        # Search options
        minimum_should_match=0,
        filter=[],
        async_mode=True,
        k=7,  # Number of documents to retrieve
        verbose=False,
    )

    logger.info(f"{Colors.GREEN}OpenSearchHybridSearchRetriever initialized with advanced features{Colors.END}")

    return opensearch_hybrid_retriever

#######################################################
## RAG with Strands Agent
#######################################################
async def perform_rag_with_agent(query: str):
    """Perform RAG using OpenSearchHybridSearchRetriever + Strands Agent SDK"""

    logger.info(f"{Colors.GREEN}===== Executing RAG with Strands Agent ====={Colors.END}")
    logger.info(f"{Colors.BOLD}===== RAG - Query: {query} ====={Colors.END}")

    # Step 1: Get retriever with advanced search features
    retriever = get_retriever()

    # Step 2: Retrieve documents using OpenSearchHybridSearchRetriever
    # Since complex_doc=True, it returns (retrieval, tables, images) tuple
    try:
        retrieval_result = retriever.invoke(query)

        # Handle complex_doc return format
        if isinstance(retrieval_result, tuple) and len(retrieval_result) == 3:
            retrieval, tables, images = retrieval_result
        else:
            # Fallback if not tuple (shouldn't happen with complex_doc=True)
            retrieval = retrieval_result if isinstance(retrieval_result, list) else [retrieval_result]
            tables = []
            images = []

        logger.info(f"{Colors.BLUE}Retrieved {len(retrieval)} documents, {len(tables)} tables, {len(images)} images{Colors.END}")

        if not retrieval:
            return "No relevant contexts found.", []

    except Exception as e:
        logger.error(f"{Colors.RED}Error retrieving documents: {str(e)}{Colors.END}")
        return f"Error retrieving documents: {str(e)}", []

    # Step 3: Extract content from LangChain Document objects
    context_text = "\n\n".join([doc.page_content for doc in retrieval])
    tables_text = "\n\n".join([doc.page_content for doc in tables]) if tables else ""

    # Step 4: Create system prompt
    system_prompt = dedent(
        """
        You are a master answer bot designed to answer user's questions.
        I'm going to give you contexts which consist of texts, tables and images.
        Read the contexts carefully, because I'm going to ask you a question about it.
        """
    )

    # Step 5: Create Strands Agent
    llm_model_name = os.getenv('LLM_MODEL_NAME', 'Claude-V4-5-Sonnet-CRI')
    agent_type = "claude-sonnet-4-5" if "4-5" in llm_model_name else "claude-sonnet-4"

    agent = strands_utils.get_agent(
        agent_name="rag_agent",
        system_prompts=system_prompt,
        agent_type=agent_type,
        enable_reasoning=False,
        prompt_cache_info=(False, None),
        streaming=True,
    )

    # Step 6: Create user message with contexts (including images for multimodal support)
    user_message_content = []

    # Add images first (for better context)
    if images:
        logger.info(f"{Colors.YELLOW}Adding {len(images)} images to agent input{Colors.END}")
        for idx, image_doc in enumerate(images):
            # Extract base64 image from Document.page_content and decode to bytes
            image_base64 = image_doc.page_content
            image_bytes = base64.b64decode(image_base64)
            logger.info(f"{Colors.YELLOW}  - Image {idx+1}: {len(image_bytes)} bytes{Colors.END}")

            # Create Strands ImageContent and ContentBlock with bytes
            image_content = ImageContent(format="png", source={"bytes": image_bytes})
            user_message_content.append(ContentBlock(image=image_content))

    # Add tables as images if they have image_base64 in metadata
    if tables:
        for idx, table_doc in enumerate(tables):
            if "image_base64" in table_doc.metadata and table_doc.metadata["image_base64"]:
                table_image_bytes = base64.b64decode(table_doc.metadata["image_base64"])
                logger.info(f"{Colors.YELLOW}Adding table {idx+1} as image to agent input: {len(table_image_bytes)} bytes{Colors.END}")
                image_content = ImageContent(format="png", source={"bytes": table_image_bytes})
                user_message_content.append(ContentBlock(image=image_content))

    # Build text content
    text_parts = [f"Here is the contexts as texts: <contexts>{context_text}</contexts>"]

    if tables_text: text_parts.append(f"\nHere are some tables: <tables>{tables_text}</tables>")

    text_parts.append(dedent(
        f"""
        First, find a few paragraphs or sentences from the contexts that are most relevant to answering the question.
        Then, answer the question as much as you can.

        Skip the preamble and go straight into the answer.
        Don't insert any XML tag such as <contexts> and </contexts> when answering.
        Answer in Korean.

        Here is the question: <question>{query}</question>

        If the question cannot be answered by the contexts, say "No relevant contexts".
        """
    ))

    # Add text content using Strands ContentBlock
    user_message_content.append(ContentBlock(text="\n".join(text_parts)))

    # If multimodal content exists, use list format; otherwise use string
    if len(user_message_content) > 1:
        user_message = user_message_content
        logger.info(f"{Colors.GREEN}Sending multimodal message with {len(user_message_content)} content blocks{Colors.END}")
    else:
        # Extract text from ContentBlock
        user_message = user_message_content[0]["text"]
        logger.info(f"{Colors.GREEN}Sending text-only message{Colors.END}")

    # Step 7: Process streaming response

    full_text = ""
    async for event in strands_utils.process_streaming_response_yield(
        agent, user_message, agent_name="rag_agent", source="rag_tool"
    ):
        if event.get("event_type") == "text_chunk":
            full_text += event.get("data", "")

    # Return response and original retrieval documents
    return full_text, retrieval

#######################################################
## Tool Handler
#######################################################
@log_io
def handle_rag_tool(query: Annotated[str, "The question or query to search for in the knowledge base."]):
    """Use this tool to perform RAG queries and get contextual answers from documents."""

    try:
        # Run async function in sync context
        response, contexts = asyncio.run(perform_rag_with_agent(query))

        # Return result
        results = "||".join([query, response])
        return results + "\n"

    except Exception as e:
        # Catch any other exceptions
        error_message = f"Error executing RAG query: {str(e)}"
        logger.error(f"{Colors.RED}{error_message}{Colors.END}")
        return error_message

# Function name must match tool name (for Strands Agent SDK integration)
def rag_tool(tool: ToolUse, **kwargs: Any) -> ToolResult:
    tool_use_id = tool["toolUseId"]
    query = tool["input"]["query"]

    # Use the existing handle_rag_tool function
    result = handle_rag_tool(query)

    # Check if execution was successful based on the result string
    if "Error executing RAG query" in result:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": result}]
        }
    else:
        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": result}]
        }

if __name__ == "__main__":
    # Test example using the handle_rag_tool function directly
    # This query is designed to potentially retrieve documents with images/charts
    test_query = "만기상환 여부에 따른 투자 수익률"

    # Uncomment to test with a query that might retrieve more images
    # test_query = "수익률 구조를 보여주는 그래프나 차트가 있나요?"

    print(handle_rag_tool(test_query))
