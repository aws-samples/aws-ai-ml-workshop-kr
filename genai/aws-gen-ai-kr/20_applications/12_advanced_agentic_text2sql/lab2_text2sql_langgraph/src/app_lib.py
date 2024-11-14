# AWS Bedrock과 OpenSearch를 활용한 데이터베이스 쿼리 및 검색 기능 구현
import boto3
import json
import copy
from botocore.config import Config

# 필요한 라이브러리 임포트
from langchain_aws import BedrockEmbeddings
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from src.opensearch import OpenSearchHybridRetriever, OpenSearchClient

# AWS 리전과 Claude 3 모델 설정
region_name = "us-west-2"
llm_model = "anthropic.claude-3-sonnet-20240229-v1:0"

# SQLite 데이터베이스 연결 설정
engine = create_engine("sqlite:///Chinook.db")
db = SQLDatabase(engine)
DIALECT = "sqlite"

# Bedrock의 Claude 모델과 대화하는 함수
def converse_with_bedrock(sys_prompt, usr_prompt):
    temperature = 0.0
    top_p = 0.1
    top_k = 1
    inference_config = {"temperature": temperature, "topP": top_p}
    additional_model_fields = {"top_k": top_k}

# Bedrock 모델 호출    
    response = boto3_client.converse(
        modelId=llm_model, 
        messages=usr_prompt, 
        system=sys_prompt,
        inferenceConfig=inference_config,
        additionalModelRequestFields=additional_model_fields
    )
    return response['output']['message']['content'][0]['text']

# AWS Bedrock 클라이언트 초기화 함수
def init_boto3_client(region: str):
    retry_config = Config(
        region_name=region,
        retries={"max_attempts": 10, "mode": "standard"}
    )
    return boto3.client("bedrock-runtime", region_name=region, config=retry_config)

# OpenSearch 리소스 초기화 함수
def init_search_resources():  
    embedding_model = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", region_name=region_name, model_kwargs={"dimensions":1024})
    
# SQL 검색과 테이블 검색을 위한 OpenSearch 클라이언트 설정
    sql_search_client = OpenSearchClient(emb=embedding_model, index_name='example_queries', mapping_name='mappings-sql', vector="input_v", text="input", output=["input", "query"])
    table_search_client = OpenSearchClient(emb=embedding_model, index_name='schema_descriptions', mapping_name='mappings-detailed-schema', vector="table_summary_v", text="table_summary", output=["table_name", "table_summary"])

    sql_retriever = OpenSearchHybridRetriever(sql_search_client, k=10)
    table_retriever = OpenSearchHybridRetriever(table_search_client, k=10)
    return sql_search_client, table_search_client, sql_retriever, table_retriever

# 테이블의 컬럼 설명을 검색하는 함수
def get_column_description(table_name):
    query = {
        "query": {
            "match": {
                "table_name": table_name
            }
        }
    }
    response = table_search_client.conn.search(index=table_search_client.index_name, body=query)
    # 검색 결과 처리
    if response['hits']['total']['value'] > 0:
        source = response['hits']['hits'][0]['_source']
        columns = source.get('columns', [])
        if columns:
            return {col['col_name']: col['col_desc'] for col in columns}
        else:
            return {}
    else:
        return {}
    
# 키워드로 테이블과 컬럼을 검색하는 함수
def search_by_keywords(keyword):
    query = {
        "size": 10, 
        "query": {
            "nested": {
                "path": "columns",
                "query": {
                    "match": {
                        "columns.col_desc": f"{keyword}"
                    }
                },
                "inner_hits": {
                    "size": 1, 
                    "_source": ["columns.col_name", "columns.col_desc"]
                }
            }
        },
        "_source": ["table_name"]
    }
    response = table_search_client.conn.search(
        index=table_search_client.index_name,
        body=query
    )
    
    search_result = ""
    try:
        results = []
        table_names = set()  
        if 'hits' in response and 'hits' in response['hits']:
            for hit in response['hits']['hits']:
                table_name = hit['_source']['table_name']
                table_names.add(table_name)  
                for inner_hit in hit['inner_hits']['columns']['hits']['hits']:
                    column_name = inner_hit['_source']['col_name']
                    column_description = inner_hit['_source']['col_desc']
                    results.append({
                        "table_name": table_name,
                        "column_name": column_name,
                        "column_description": column_description
                    })
                    if len(results) >= 5:
                        break
                if len(results) >= 5:
                    break
        search_result += json.dumps(results, ensure_ascii=False)
    except:
        search_result += f"{keyword} not found"
    return search_result    

# 프롬프트 생성 함수
def create_prompt(sys_template, user_template, **kwargs):
    sys_prompt = [{"text": sys_template.format(**kwargs)}]
    usr_prompt = [{"role": "user", "content": [{"text": user_template.format(**kwargs)}]}]
    return sys_prompt, usr_prompt

# 초기화
boto3_client = init_boto3_client(region_name)
sql_search_client, table_search_client, sql_retriever, table_retriever = init_search_resources()





