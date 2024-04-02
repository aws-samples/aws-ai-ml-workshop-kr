import boto3
import json
import os

BEDROCK_REGION_NAME=os.environ.get('BEDROCK_REGION_NAME', 'us-west-2')

bedrock_runtime = boto3.client(
  service_name='bedrock-runtime',
  region_name=BEDROCK_REGION_NAME
)

MODEL_ID = 'anthropic.claude-3-sonnet-20240229-v1:0'


def lambda_handler(event, context):
    print(event)

    prompt = event['body'] 

    command = """위의 리뷰에는 구매자의 상품에 대한 평가를 담고 있습니다. 
위 리뷰를 분석해서 구매자의 상품에 대한 감정인 Sentiment와 판매자 입장에서 작성한 답변인 Generated를 json 형식으로 만듭니다.
구매자의 Sentiment는 NEGATIVE, POSITIVE, MIXED로 표현합니다.
분석한 Sentiment가 POSITIVE일 경우에는 감사의 마음을 담아 Generated를 작성하고, 
NEGATIVE일 경우에는 구매자에게 정중하게 사과문을 작성합니다."""    

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
                        "text": command,
                    }                    
                ]
            }
        ],
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2000,
        "top_k": 50,
        "top_p": 0.92,
        "temperature": 0.9
    }

    # Run Bedrock API
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
