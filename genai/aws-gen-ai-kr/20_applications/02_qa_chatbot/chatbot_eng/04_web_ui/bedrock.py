import os, sys, boto3

module_path = "../../../../"
sys.path.append(os.path.abspath(module_path))

from textwrap import dedent
from utils import bedrock
from utils.bedrock import bedrock_info
from utils.rag import rag_chain, OpenSearchHybridSearchRetriever
from utils.opensearch import opensearch_utils
from utils.ssm import parameter_store
from langchain.embeddings import BedrockEmbeddings
from langchain_aws import ChatBedrock


region = boto3.Session().region_name
pm = parameter_store(region)

# 텍스트 생성 LLM 가져오기, streaming_callback을 인자로 받아옴
def get_llm(streaming_callback):
    boto3_bedrock = bedrock.get_bedrock_client(
    assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    endpoint_url=os.environ.get("BEDROCK_ENDPOINT_URL", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None),
    )
    llm = ChatBedrock(
    model_id=bedrock_info.get_model_id(model_name="Claude-V3-5-Sonnet"),
    client=boto3_bedrock,
    model_kwargs={
        "max_tokens": 1024,
        "stop_sequences": ["\n\nHuman"],
    },
    callbacks=[streaming_callback],
    )
    return llm

# 임베딩 모델 가져오기
def get_embedding_model():
    llm_emb = BedrockEmbeddings(model_id='amazon.titan-embed-text-v2:0')
    return llm_emb

# Opensearch vectorDB 가져오기
def get_opensearch_client():
    opensearch_domain_endpoint = pm.get_params(key='opensearch_domain_endpoint', enc=False)
    opensearch_user_id = pm.get_params(key='opensearch_user_id', enc=False)
    opensearch_user_password = pm.get_params(key='opensearch_user_password', enc=True)
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
def get_retriever(streaming_callback, parent, reranker, alpha):
    os_client = get_opensearch_client()
    llm_text = get_llm(streaming_callback)
    llm_emb = get_embedding_model()
    reranker_endpoint_name = pm.get_params(key="reranker_endpoint",enc=False)
    index_name = pm.get_params(key="opensearch_index_name",enc=True)
    # index_name = pm.get_params(key="opensearch-index-name-workshop-app", enc=True)
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
        ensemble_weights=[alpha, 1.0-alpha],
        reranker=reranker,  # enable reranker with reranker model
        # endpoint name for reranking model
        reranker_endpoint_name=reranker_endpoint_name,
        parent_document=parent,  # enable parent document
        # option for async search
        async_mode=True,
        # option for output
        k=5,  # 최종 Document 수 정의
        verbose=True,
    )
    return opensearch_hybrid_retriever

def invoke(query, streaming_callback, parent, reranker, alpha):
    # llm, retriever 가져오기
    llm_text = get_llm(streaming_callback)
    opensearch_hybrid_retriever = get_retriever(streaming_callback, parent, reranker, alpha)
    # context, tables, images = opensearch_hybrid_retriever._get_relevant_documents()
    # answer only 선택
    system_prompt = dedent(
    """
    You are a master answer bot designed to answer user's questions in English.
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
        Answer in English.
    
        Here is the question: <question>{question}</question>
    
        If the question cannot be answered by the contexts, say "No relevant contexts".
        """
    )

    qa = rag_chain(
        llm_text=llm_text,
        retriever=opensearch_hybrid_retriever,
        system_prompt=system_prompt,
        human_prompt=human_prompt,
        return_context=True,
        verbose=False,
    )
    response, similar_docs = qa.invoke(query = query)

    return response, similar_docs
 