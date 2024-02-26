from utils.rag import prompt_repo
from utils.opensearch import opensearch_utils
from utils.rag import OpenSearchHybridSearchRetriever
from utils.proc_docs import get_parameter
from langchain.chains import RetrievalQA
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
import boto3
import os
import sys
module_path = "../../.."
sys.path.append(os.path.abspath(module_path))


# 사용 중인 Opensearch index
index_name = "v15-genai-poc-knox-parent-doc-retriever"


# 텍스트 생성 LLM 가져오기, streaming_callback을 인자로 받아옴
def get_llm(streaming_callback):
    model_kwargs = {  # Anthropic 모델
        "max_tokens_to_sample": 4000,
        "temperature": 0.1,
        "top_k": 3,
        "top_p": 0.1,
        "stop_sequences": ["\n\nHuman:"]
    }

    llm = Bedrock(
        model_id="anthropic.claude-v2:1",  # 파운데이션 모델 설정
        model_kwargs=model_kwargs,  # Claud에 대한 속성 구성
        streaming=True,
        callbacks=[streaming_callback],
    )
    return llm

# 임베딩 모델 가져오기


def get_embedding_model():
    llm_emb = BedrockEmbeddings(model_id='amazon.titan-embed-text-v1')

    return llm_emb

# Opensearch vectorDB 가져오기


def get_opensearch_client():
    ssm = boto3.client("ssm", "us-east-1")

    opensearch_domain_endpoint = get_parameter(
        boto3_clinet=ssm,
        parameter_name='knox_opensearch_domain_endpoint',
    )

    opensearch_user_id = get_parameter(
        boto3_clinet=ssm,
        parameter_name='knox_opensearch_userid',
    )

    opensearch_user_password = get_parameter(
        boto3_clinet=ssm,
        parameter_name='knox_opensearch_password',
    )

    opensearch_domain_endpoint = opensearch_domain_endpoint
    rag_user_name = opensearch_user_id
    rag_user_password = opensearch_user_password
    aws_region = os.environ.get("AWS_DEFAULT_REGION", None)
    http_auth = (rag_user_name, rag_user_password)

    os_client = opensearch_utils.create_aws_opensearch_client(
        aws_region,
        opensearch_domain_endpoint,
        http_auth
    )

    return os_client

# hybrid search retriever 만들기


def get_retriever(streaming_callback):
    os_client = get_opensearch_client()
    llm_text = get_llm(streaming_callback)
    llm_emb = get_embedding_model()
    reranker_endpoint_name = "huggingface-pytorch-inference-2023-11-15-07-53-21-605"

    opensearch_hybrid_retriever = OpenSearchHybridSearchRetriever(
        os_client=os_client,
        index_name=index_name,
        llm_text=llm_text,  # llm for query augmentation in both rag_fusion and HyDE
        llm_emb=llm_emb,  # Used in semantic search based on opensearch

        # option for lexical
        minimum_should_match=0,
        filter=[],

        # option for search
        # ["RRF", "simple_weighted"], rank fusion 방식 정의
        fusion_algorithm="RRF",
        # [for lexical, for semantic], Lexical, Semantic search 결과에 대한 최종 반영 비율 정의
        ensemble_weights=[.51, .49],
        reranker=True,  # enable reranker with reranker model
        # endpoint name for reranking model
        reranker_endpoint_name=reranker_endpoint_name,
        parent_document=True,  # enable parent document

        # option for async search
        async_mode=True,

        # option for output
        k=6,  # 최종 Document 수 정의
        verbose=False,
    )
    return opensearch_hybrid_retriever

# 모델에 query하기


def invoke(query, streaming_callback):

    # llm, retriever 가져오기
    llm_text = get_llm(streaming_callback)
    opensearch_hybrid_retriever = get_retriever(streaming_callback)

    # answer only 선택
    PROMPT = prompt_repo.get_qa(prompt_type="answer_only")

    qa = RetrievalQA.from_chain_type(
        llm=llm_text,
        chain_type="stuff",
        retriever=opensearch_hybrid_retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PROMPT,
            "verbose": False,
        },
        verbose=False
    )

    response = qa(query)

    answer = response["result"]
    source_documents = response['source_documents']
    document = source_documents[0]

    if document:
        document_dict = document.__dict__
        metadata = document_dict.get('metadata', {})
        url = metadata.get('url', '')

    return answer, url, source_documents
