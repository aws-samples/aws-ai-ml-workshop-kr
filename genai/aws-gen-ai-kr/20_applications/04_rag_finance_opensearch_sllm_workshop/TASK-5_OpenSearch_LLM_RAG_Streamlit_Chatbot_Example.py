import streamlit as st
import sys
import json
import boto3
import numpy as np
from typing import Any, Dict, List, Optional
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.llms.sagemaker_endpoint import LLMContentHandler, SagemakerEndpoint
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from streamlit_chat import message
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import Chroma, AtlasDB, FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import csv
from langchain.vectorstores import OpenSearchVectorSearch
import os
import copy

import sys
sys.path.append('./utils') # src 폴더 경로 설정
from streamlit_util import KoSimCSERobertaContentHandler, KullmContentHandler, SagemakerEndpointEmbeddingsJumpStart, KoSimCSERobertaContentHandler

##########################################################################################################################################################################
# pip install -r ./requirements.txt in the system terminal
# Studio의 Stramlit URL은 domain의 lab이 proxy/8501/webapp로 대체됩니다.
# ex > https://d-l2kk7xvxmnbl.studio.us-east-1.sagemaker.aws/jupyter/default/proxy/8501/
# 참고 : https://aws.amazon.com/ko/blogs/tech/build-a-powerful-question-answering-bot-with-amazon-sagemaker-amazon-opensearch-service-streamlit-and-langchain/
#########################################################################################################################################################################


######## AWS Setting
aws_region = 'us-east-1'
region ='us-east-1'
service ='es'

######## For SageMaker
# LLM Endpoint Name :
llm_endpoint_name = 'kullm-polyglot-5-8b-v2-2023-08-23-15-47-39-450-endpoint'
# Embedding Vector Model Endpoint Name :
embvec_endpoint_name= 'KoSimCSE-roberta-2023-08-23-14-07-12'

######## For OpenSearch 
# Opensearch index name : 
index_name = 'fsi-sample'
# Opensearch domain_endpoin name :
opensearch_domain_endpoint = "https://search-ragopensearch-2pz3fgitugmvrz7vbngitqljzu.us-east-1.es.amazonaws.com"
# Opensearch master user auth
username = 'raguser'
password = 'MarsEarth1!'

#aws_access_key = os.environ['AWS_ACCESS_KEY']
#aws_secret_key =os.environ['AWS_SECRET_KEY']
##########################################################################################################################################################################
# 검색 rank 개수 
faiss_k =3

# Kullum LLM 파라미터 설정
params = {
      'do_sample': False,
      'max_new_tokens': 512, #128
      'temperature': 1.0,  # 0.5 ~ 1.0 default = 1.0 높으면 랜덤하게 자유도. 다음 생성 문장 토큰의 자유도
      'top_k': 0,
      'top_p': 0.9,
      'return_full_text': False,
      'repetition_penalty': 1.1,
      'presence_penalty': None,
      'eos_token_id': 2
}
##########################################################################################################################################################################


def load_chain(llm_endpoint_name):
    # KULLUM LLM 로드
    LLMTextContentHandler = KullmContentHandler()
    endpoint_name_text = llm_endpoint_name
    seperator = "||SPEPERATOR||"

    llm_text = SagemakerEndpoint(
        endpoint_name=endpoint_name_text,
        region_name=aws_region,
        model_kwargs=params,
        content_handler=LLMTextContentHandler,
    )
    prompt_template = ''.join(["{context}", seperator, "{question}"])

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=llm_text, chain_type="stuff", prompt=PROMPT, verbose=True)
    return chain

##################################################################################################
# FAISS VectorStore - OpenSearch
##################################################################################################
def load_emb_vec(embvec_endpoint_name):
    LLMEmbHandler = KoSimCSERobertaContentHandler()
    emb_vec = SagemakerEndpointEmbeddingsJumpStart(
        endpoint_name=embvec_endpoint_name,
        region_name=aws_region,
        content_handler=LLMEmbHandler,
    )
    return emb_vec

# opensearch score seems like ranking
def filter_and_remove_score_opensearch_vector_score(res, cutoff_score = 0.006, variance=0.95):
    # Get the lowest score
    highest_score = max(score for doc, score in res)
    print('highest_score : ', highest_score)
    # If the lowest score is over 200, return an empty list
    if highest_score < cutoff_score:
        return []
    # Calculate the upper bound for scores
    lower_bound = highest_score * variance
    print('lower_bound : ', lower_bound)
    # Filter the list and remove the score
    res = [doc for doc, score in res if score >= lower_bound]

    return res


def get_similiar_docs(query, k=5, fetch_k=300, score=True, bank="신한은행"):
    print("bank : ", bank)
    #query = f'{bank}, {query}'
    print("query : ",query)

    if score:
        pre_similar_doc = vectro_db.similarity_search_with_score(
            query,
            k=k,
            fetch_k=fetch_k,
            search_type="approximate_search",  # approximate_search, script_scoring, painless_scripting
            space_type="l2",  # "l2", "l1", "linf", "cosinesimil", "innerproduct", "hammingbit";
            pre_filter={"bool": {"filter": {"term": {"text": bank}}}},
            boolean_filter={"bool": {"filter": {"term": {"text": bank}}}}
            # filter=dict(source=bank)
        )
        print('jhs : ', pre_similar_doc)
        pretty_print_documents(pre_similar_doc)
        similar_docs = filter_and_remove_score_opensearch_vector_score(pre_similar_doc)
    else:
        similar_docs = vectro_db.similarity_search(
            query,
            k=k,
            search_type="approximate_search",  # approximate_search, script_scoring, painless_scripting
            space_type="12",  # "l2", "l1", "linf", "cosinesimil", "innerproduct", "hammingbit";
            pre_filter={"bool": {"filter": {"term": {"text": bank}}}},
            boolean_filter={"bool": {"filter": {"term": {"text": bank}}}}

        )
    similar_docs_copy = copy.deepcopy(similar_docs)

    # print('similar_docs_copy : \n', similar_docs_copy)

    return similar_docs_copy

# 임베딩 벡터 로드
emb_vec = load_emb_vec(embvec_endpoint_name)

# LLM 로드
chain = load_chain(llm_endpoint_name)

http_auth = (username, password) # opensearch user

#OpenSearch Vector Indexer

vectro_db = OpenSearchVectorSearch(
    index_name=index_name,
    opensearch_url=opensearch_domain_endpoint,
    embedding_function=emb_vec,
    http_auth=http_auth,
    is_aoss = False,
    engine="faiss",
    space_type="12"
)

##################################################################################################
def pretty_print_documents(response):
    for doc, score in response:
        print(f'\nScore: {score}')
        print(f'Document Number: {doc.metadata["row"]}')
        print(f'Source: {doc.metadata["source"]}')

        # Split the page content into lines
        lines = doc.page_content.split("\n")

        # Extract and print each piece of information if it exists
        for line in lines:
            split_line = line.split(": ")
            if len(split_line) > 1:
                print(f'{split_line[0]}: {split_line[1]}')

        print('-' * 50)


def get_answer(query):
    k = 3
    search_query = query

    similar_docs = get_similiar_docs(search_query, k=k, bank='신한은행')

    llm_query = ''+query+' Category에 대한 Information을 찾아서 설명해주세요.'

    if not similar_docs:
        llm_query = query

    answer = chain.run(input_documents=similar_docs, question=llm_query)

    return answer



##################################################################################################
# Streamlit UI
# From here down is all the StreamLit UI.
##################################################################################################
st.set_page_config(page_title="FSI RAG FAQ Demo vectorstore mode", page_icon="🦜", layout="wide")
st.header("🦜 FSI RAG Demo - Opensearch vectorstore with LLM mode")

def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text

# 사용자로부터 입력을 받습니다.
# user_input = get_text()

# if "generated" not in st.session_state:
#     st.session_state["generated"] = []
#
# if "past" not in st.session_state:
#     st.session_state["past"] = []
#
# # 사용자가 입력을 제공했는지 확인합니다.
# if user_input:
#     output = get_answer(user_input)
#     print("OUTPUT : ", output)
#     st.session_state.past.append(user_input)
#     st.session_state.generated.append(output)
#
#
#
#
# if st.session_state["generated"]:
#
#     for i in range(len(st.session_state["generated"]) - 1, -1, -1):
#         message(st.session_state["generated"][i], key=str(i))
#         message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")



from langchain.callbacks import StreamlitCallbackHandler
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="여기에 금융 FAQ 질문해주세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)


    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = get_answer(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)