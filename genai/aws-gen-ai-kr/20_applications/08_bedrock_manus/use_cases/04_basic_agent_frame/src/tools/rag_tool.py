import os
import sys

module_path = "../.."
sys.path.append(os.path.abspath(module_path))

import boto3
import logging
from pprint import pprint
from textwrap import dedent
from termcolor import colored
from typing import Any, Annotated

from langchain_aws import ChatBedrock
from langchain_aws import BedrockEmbeddings
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from src.utils import bedrock
from src.utils.bedrock import bedrock_info

from src.utils.ssm import parameter_store
from src.utils.opensearch import opensearch_utils

from src.utils.agentic_rag import rag_chain
from src.utils.agentic_rag import OpenSearchHybridSearchRetriever

from src.tools.decorators import log_io
from strands.types.tools import ToolResult, ToolUse

# 새 핸들러와 포맷터 설정
logger = logging.getLogger(__name__)
logger.propagate = False  # 상위 로거로 메시지 전파 중지
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
handler = logging.StreamHandler()
formatter = logging.Formatter('\n%(levelname)s [%(name)s] %(message)s')  # 로그 레벨이 동적으로 표시되도록 변경
handler.setFormatter(formatter)
logger.addHandler(handler)
# DEBUG와 INFO 중 원하는 레벨로 설정
logger.setLevel(logging.INFO)  # 기본 레벨은 INFO로 설정

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
## SSM
#######################################################
def ssm_config():
    region=boto3.Session().region_name
    pm = parameter_store(region)

    return pm

#######################################################
## OpenSearch
#######################################################
def opensearch_config():

    pm = ssm_config()
    opensearch_domain_endpoint = pm.get_params(key="opensearch_domain_endpoint", enc=False)
    opensearch_user_id = pm.get_params(key="opensearch_user_id", enc=False)
    opensearch_user_password = pm.get_params(key="opensearch_user_password", enc=True)

    aws_region = os.environ.get("AWS_DEFAULT_REGION", None)
    http_auth = (opensearch_user_id, opensearch_user_password) # Master username, Master password
    os_client = opensearch_utils.create_aws_opensearch_client(aws_region, opensearch_domain_endpoint, http_auth)

    index_name = pm.get_params(key="opensearch_index_name", enc=False)
    
    return os_client, index_name

#######################################################
## LLM
#######################################################
def get_model():

    boto3_bedrock = bedrock.get_bedrock_client(
        assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
        endpoint_url=os.environ.get("BEDROCK_ENDPOINT_URL", None),
        region=os.environ.get("AWS_DEFAULT_REGION", None),
    )

    #print (colored("\n== FM lists ==", "green"))
    #pprint (bedrock_info.get_list_fm_models(verbose=False))

    llm_text = ChatBedrock(
        model_id=bedrock_info.get_model_id(model_name="Claude-V3-7-Sonnet-CRI"),
        client=boto3_bedrock,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        model_kwargs={
            "max_tokens": 8192,
            "stop_sequences": ["\n\nHuman"],
            # "temperature": 0,
            # "top_k": 350,
            # "top_p": 0.999
        }
    )

    llm_emb = BedrockEmbeddings(
        client=boto3_bedrock,
        model_id=bedrock_info.get_model_id(model_name="Titan-Text-Embeddings-V2")
    )
    print("Bedrock Embeddings Model Loaded")

    return llm_text, llm_emb

#######################################################
## RAG Chain
#######################################################
def get_rag_chain():

    os_client, index_name = opensearch_config()
    llm_text, llm_emb = get_model()

    opensearch_hybrid_retriever = OpenSearchHybridSearchRetriever(
        os_client=os_client,
        index_name=index_name,
        llm_text=llm_text, # llm for query augmentation in both rag_fusion and HyDE
        llm_emb=llm_emb, # Used in semantic search based on opensearch 

        # hybird-search debugger
        #hybrid_search_debugger = "semantic", #[semantic, lexical, None]
        
        # option for lexical
        minimum_should_match=0,
        filter=[],

        # option for search
        fusion_algorithm="RRF", # ["RRF", "simple_weighted"], rank fusion 방식 정의
        ensemble_weights=[.51, .49], # [for semantic, for lexical], Semantic, Lexical search 결과에 대한 최종 반영 비율 정의
        reranker=False, # enable reranker with reranker model
        #reranker_endpoint_name=endpoint_name, # endpoint name for reranking model
        parent_document=True, # enable parent document
        
        # option for complex pdf consisting of text, table and image
        complex_doc=True,
        
        # option for async search
        async_mode=True,

        # option for output
        k=2, # 최종 Document 수 정의
        verbose=False,
    )

    system_prompt = dedent(
        """
        You are a master answer bot designed to answer user's questions.
        I'm going to give you contexts which consist of texts, tables and images.
        Read the contexts carefully, because I'm going to ask you a question about it.
        """
    )

    human_prompt = dedent(
        """
        Here is the contexts as texts: <contexts>{contexts}</contexts>

        First, find a few paragraphs or sentences from the contexts that are most relevant to answering the question.
        Then, answer the question as much as you can.

        Skip the preamble and go straight into the answer.
        Don't insert any XML tag such as <contexts> and </contexts> when answering.
        Answer in Korean.

        Here is the question: <question>{question}</question>

        If the question cannot be answered by the contexts, say "No relevant contexts".
        """
    )

    rag_chain_ = rag_chain(
        llm_text=llm_text,
        retriever=opensearch_hybrid_retriever,
        system_prompt=system_prompt,
        human_prompt=human_prompt,
        return_context=True,
        verbose=False,
    )

    return rag_chain_


@log_io
def handle_rag_tool(query: Annotated[str, "The question or query to search for in the knowledge base."]):
    """Use this tool to perform RAG queries and get contextual answers from documents."""

    logger.info(f"{Colors.GREEN}===== Executing RAG ====={Colors.END}")
    logger.info(f"{Colors.BOLD}===== RAG - Query: {query} ====={Colors.END}")

    rag_chain = get_rag_chain()

    try:
        # Execute RAG query
        response, contexts = rag_chain.invoke(query=query)
        
        # Return stdout as the result
        results = "||".join([query, response])
        return results + "\n"
        
    except Exception as e:
        # Catch any other exceptions
        error_message = f"Error executing RAG query: {str(e)}"
        logger.error(f"{Colors.RED}{error_message}{Colors.END}")
        return error_message

# Function name must match tool name
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
    test_query = "만기상환 여부에 따른 투자 수익률"
    print(handle_rag_tool(test_query))

