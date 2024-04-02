import boto3
import json
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import BedrockChat

BEDROCK_REGION_NAME=os.environ.get('BEDROCK_REGION_NAME', 'us-west-2')

bedrock_runtime = boto3.client(
  service_name='bedrock-runtime',
  region_name=BEDROCK_REGION_NAME
)

MODEL_ID = 'anthropic.claude-3-sonnet-20240229-v1:0'


def ask_with_langching(input_text):
    model_kwargs =  { 
        "max_tokens": 2048,
        "temperature": 0.0,
        "top_k": 250,
        "top_p": 1,
        "stop_sequences": ["\n\nHuman"],
    }

    model = BedrockChat(
        client=bedrock_runtime,
        model_id=MODEL_ID,
        model_kwargs=model_kwargs,
    )

    messages = [
        ("system", "You are a helpful assistant."),
        ("human", "{question}"),
    ]

    prompt = ChatPromptTemplate.from_messages(messages)

    chain = prompt | model | StrOutputParser()

    # Chain Invoke
    question = f'{input_text}\n\n위 리뷰를 읽고 판매자에게 유용한 정보를 색상, 핏, 소재, 세탁, 가격으로 요약해서 한글로 설명해줘.'
    response = chain.invoke({"question": question})
    print(response) 

    return response 


def lambda_handler(event, context):
    print(event)

    prompt = event['body']

    command = """위의 리뷰들은 상품 구매자들이 작성한 것입니다.
위 리뷰들을 읽고 판매자에게 유용한 정보를 색상, 핏, 소재, 세탁, 가격으로 요약합니다. 
"""

    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt

                    },
                    {
                        "type": "text",
                        "text": command
                    }
                ]
            }
        ],
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "top_k": 50,
        "top_p": 0.92,
        "temperature": 0.9
    }

    response = bedrock_runtime.invoke_model(
        modelId=MODEL_ID,
        contentType='application/json',
        accept='application/json',
        body=json.dumps(body)
    )

    print(response)

    response_body = json.loads(response.get('body').read())
    output = response_body['content'][0]['text']
    print(output)  

    return {
        'statusCode': 200,
        'body': output
    }
