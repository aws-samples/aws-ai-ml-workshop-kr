import boto3
import json
import copy
from botocore.config import Config

from langchain_aws import BedrockEmbeddings
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from src.opensearch import OpenSearchHybridRetriever, OpenSearchClient

    
def converse_with_bedrock(sys_prompt, usr_prompt):
    temperature = 0.0
    top_p = 0.1
    top_k = 1
    inference_config = {"temperature": temperature, "topP": top_p}
    additional_model_fields = {"top_k": top_k}
    response = boto3_client.converse(
        modelId=llm_model, 
        messages=usr_prompt, 
        system=sys_prompt,
        inferenceConfig=inference_config,
        additionalModelRequestFields=additional_model_fields
    )
    return response['output']['message']['content'][0]['text']

def init_boto3_client(region: str):
    retry_config = Config(
        region_name=region,
        retries={"max_attempts": 10, "mode": "standard"}
    )
    return boto3.client("bedrock-runtime", region_name=region, config=retry_config)

def init_search_resources(region_name, k=10):  
    embedding_model = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", region_name=region_name, model_kwargs={"dimensions":1024})
    sql_search_client = OpenSearchClient(emb=embedding_model, index_name='example_queries', mapping_name='mappings-sql', vector="input_v", text="input", output=["input", "query"])
    table_search_client = OpenSearchClient(emb=embedding_model, index_name='schema_descriptions', mapping_name='mappings-detailed-schema', vector="table_summary_v", text="table_summary", output=["table_name", "table_summary"])

    sql_retriever = OpenSearchHybridRetriever(sql_search_client, k=k)
    table_retriever = OpenSearchHybridRetriever(table_search_client, k=k)
    return sql_search_client, table_search_client, sql_retriever, table_retriever

def get_column_description(table_name):
    query = {
        "query": {
            "match": {
                "table_name": table_name
            }
        }
    }
    response = table_search_client.conn.search(index=table_search_client.index_name, body=query)

    if response['hits']['total']['value'] > 0:
        source = response['hits']['hits'][0]['_source']
        columns = source.get('columns', [])
        if columns:
            return {col['col_name']: col['col_desc'] for col in columns}
        else:
            return {}
    else:
        return {}


def create_prompt(sys_template, user_template, **kwargs):
    sys_prompt = [{"text": sys_template.format(**kwargs)}]
    usr_prompt = [{"role": "user", "content": [{"text": user_template.format(**kwargs)}]}]
    return sys_prompt, usr_prompt


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


session = boto3.session.Session()
region_name = session.region_name
sql_search_client, table_search_client, sql_retriever, table_retriever = init_search_resources(region_name, k=5)



######################################################       
# Grape Node 정의
######################################################       
from textwrap import dedent

csv_list_response_format = "Your response should be a list of comma separated values, eg: `foo, bar, baz` or `foo,bar,baz`"
json_response_format = dedent("""
    The output should be formatted as a JSON instance that conforms to the JSON schema below.

    As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
    the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. 
    The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

    Here is the output schema:
    ```
    {
        "properties": {
            "setup": {
                "title": "Setup", 
                "description": "question to set up a joke", 
                "type": "string"
            }, 
            "punchline": {
                "title": "Punchline", 
                "description": "answer to resolve the joke", 
                "type": "string"
            }
        }, 
        "required": ["setup", "punchline"]
    }
    ```
""")


class NodeTester:
    '''
    test_state = {
     "question": "서울시의 인구 통계를 보여줘",
     "intent": "",
     "sample_queries": [],
     "readiness": "",
     "tables_summaries": [],
     "table_names": ["population", "demographics"],
     "table_details": [],
     "query_state": {},
     "next_action": "",
     "answer": "",
     "dialect": "postgresql"
    }
    '''
    def __init__(self):
        pass

    def test(self, node_function, test_state, verbose=True):
        result = node_function(test_state)
        if verbose:
            print("## Test Result:")
            print(json.dumps(result, indent=2, ensure_ascii=False))

        return result

######################################################       
# Langfuse 정의
######################################################       

from langfuse.decorators import observe, langfuse_context
from botocore.exceptions import ClientError
 
# @observe(as_type="generation", name="Bedrock Converse")
# @observe(as_type="generation")
def wrapped_bedrock_converse(**kwargs):

    function_name = kwargs.pop('function_name', "Bedrock Converse")
    boto3_client = kwargs.pop('boto3_client', [])

    # Langfuse 컨텍스트에 이름 업데이트
    langfuse_context.update_current_observation(name=function_name)    

    # 1. extract model metadata
    kwargs_clone = kwargs.copy()
    messages = kwargs_clone.pop('messages', None)
    system = kwargs_clone.pop('system', None)
    
    # messages와 system을 형식에 맞게 처리
    input_data = {
        "kwargs": {
            "system": system,            
            "messages": messages,
        }
    }    

    modelId = kwargs_clone.pop('modelId', None)
    model_parameters = {
        **kwargs_clone.pop('inferenceConfig', {}),
        **kwargs_clone.pop('additionalModelRequestFields', {})
    }
  # 1. Langfuse 관측 컨텍스트에 입력, 모델 ID, 파라미터, 기타 메타데이터를 업데이트합니다.
    langfuse_context.update_current_observation(
        input=input_data,
        model=modelId,
        model_parameters=model_parameters,
        metadata=kwargs_clone
    )

 
    # 2. model call with error handling
    try:
        response = boto3_client.converse(**kwargs)
    except (ClientError, Exception) as e:
        error_message = f"ERROR: Can't invoke '{modelId}'. Reason: {e}"
        langfuse_context.update_current_observation(level="ERROR", status_message=error_message)
        print(error_message)
        return

    
 
  # 3. extract response metadata
  # Langfuse에 출력 텍스트, 토큰 사용량, 응답 메타데이터를 기록합니다.
    try:
        response_text = response["output"]["message"]["content"][0]["text"]
    
        langfuse_context.update_current_observation(
        output=response_text,
    
        usage_details={
            "input": response["usage"]["inputTokens"],
            "output": response["usage"]["outputTokens"],
            "total": response["usage"]["totalTokens"]
            },
            metadata={
                "ResponseMetadata": response["ResponseMetadata"],
            }
        )
    except (ClientError, Exception) as e:
        print("## response: \n", response) 
        error_message = f"ERROR: Can't parse:  Reason: {e}"
        langfuse_context.update_current_observation(level="ERROR", status_message=error_message)
        print(error_message)
        return error_message

    return response_text

import boto3
def converse_with_bedrock_langfuse(sys_prompt, usr_prompt, model_id, function_name=None):
    
    session = boto3.session.Session()
    region_name = session.region_name

    boto3_client = init_boto3_client(region_name)


    # 기본 함수 이름 설정
    observation_name = function_name or "Bedrock Converse"
    
    # 함수를 직접 호출하는 대신, 먼저 데코레이터를 동적으로 적용
    # 이렇게 하면 observe에서 사용하는
    # 장식된 함수가 매번 새롭게 생성됩니다
    decorated_func = observe(as_type="generation", name=observation_name)(wrapped_bedrock_converse)
    
    
    if "nova" in model_id:
        response_text = decorated_func(
            boto3_client = boto3_client,
            modelId=model_id,
            messages=usr_prompt,
            system=sys_prompt,
            inferenceConfig={"maxTokens":4096,"temperature": 0.0, "topP": 0.1},
            # additionalModelRequestFields={"top_k":1} # <-- Nova 에서는 top_k 지원 안함.
            )   
    else:
        response_text = decorated_func(
            boto3_client = boto3_client,
            modelId=model_id,
            messages=usr_prompt,
            system=sys_prompt,
            inferenceConfig={"maxTokens":4096,"temperature": 0.0, "topP": 0.1},
            additionalModelRequestFields={"top_k":1}
            )   


    return response_text

