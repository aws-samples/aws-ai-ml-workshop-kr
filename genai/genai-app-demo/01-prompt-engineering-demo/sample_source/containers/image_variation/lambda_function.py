import boto3
import json
import base64
import os

from io import BytesIO


BEDROCK_REGION_NAME=os.environ.get('BEDROCK_REGION_NAME', 'us-west-2')
BUCKET_NAME = os.environ.get('BUCKET_NAME')

bedrock_runtime = boto3.client(
  service_name='bedrock-runtime',
  region_name=BEDROCK_REGION_NAME
)

s3 = boto3.client('s3')


TITAN_MODEL_ID = 'amazon.titan-image-generator-v1'


def titan_generate_image(base64_data, quality, prompt: str = '', negative_text: str = ''):
    body = {
        'imageVariationParams': {
            'images': [base64_data]
        }, 
        'taskType': 'IMAGE_VARIATION', 
        'imageGenerationConfig': {
            'cfgScale': 8, 
            'seed': 0, 
            'quality': quality, 
            'width': 512, 
            'height': 512, 
            'numberOfImages': 3
        }
    }

    if prompt:
        body['imageVariationParams']['text'] = prompt

    if negative_text:
        body['imageVariationParams']['negativeText'] = negative_text

    response = bedrock_runtime.invoke_model(
        modelId=TITAN_MODEL_ID,
        contentType='application/json',
        accept='application/json',
        body=json.dumps(body)
    )

    print(response)

    response_body = json.loads(response.get('body').read())
    finish_reason = response_body.get("error")

    if finish_reason is not None:
        raise Exception(finish_reason)

    return response_body.get("images")


def lambda_handler(event, context):
    print(event)

    orig_image_name = json.loads(event['body'])['name']
    prompt = json.loads(event['body'])['prompt']
    negative_text = json.loads(event['body'])['negative_text']
    quality = json.loads(event['body'])['quality']

    response = s3.get_object(Bucket=BUCKET_NAME, Key=orig_image_name)
    print(response)
    image_bytes = response['Body'].read()
    base64_data = base64.b64encode(image_bytes).decode('utf-8')

    base64_str_list = titan_generate_image(base64_data, quality, prompt, negative_text)

    i = 0
    image_names = []
    for base64_str in base64_str_list:
        image_name = f'changed_image_{i}.png'
        decoded_bytes = BytesIO(base64.b64decode(base64_str))
        s3.upload_fileobj(decoded_bytes, BUCKET_NAME, f'images/{image_name}')
        image_names.append(image_name)
        i += 1

    print(image_names)

    return {
        'statusCode': 200,
        'body': json.dumps(image_names)
    }
