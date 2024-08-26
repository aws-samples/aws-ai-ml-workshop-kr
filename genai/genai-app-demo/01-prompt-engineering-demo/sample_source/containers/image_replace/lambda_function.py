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


def titan_generate_image(base64_data, prompt, mask_prompt, negative_text: str = 'bad quality, low res'):
    body = {
        'taskType': 'OUTPAINTING',
        'outPaintingParams': {
            'text': prompt,
            'negativeText': negative_text,        
            'image': base64_data,                         
            'maskPrompt': mask_prompt,                      
            'outPaintingMode': 'PRECISE'   # 'PRECISE' or 'DEFAULT'
        },
        'imageGenerationConfig': {
            'cfgScale': 8,
            'seed': 0,
            'quality': 'standard',
            'width': 512,
            'height': 512,
            'numberOfImages': 1
        }
    }

    response = bedrock_runtime.invoke_model(
        modelId=TITAN_MODEL_ID,
        contentType='application/json',
        accept='application/json',
        body=json.dumps(body)
    )

    print(response)

    response_body = json.loads(response.get('body').read())
    finish_reason = response_body.get('error')

    if finish_reason is not None:
        raise Exception(finish_reason)

    base64_str = response_body.get('images')[0]
    decoded_bytes = BytesIO(base64.b64decode(base64_str))

    return decoded_bytes


def lambda_handler(event, context):
    print(event)

    orig_image_name = json.loads(event['body'])['name']
    prompt = json.loads(event['body'])['prompt']
    mask_prompt = json.loads(event['body'])['mask_prompt']

    response = s3.get_object(Bucket=BUCKET_NAME, Key=orig_image_name)
    image_bytes = response['Body'].read()
    base64_data = base64.b64encode(image_bytes).decode('utf-8')

    image_obj = titan_generate_image(base64_data, prompt, mask_prompt)
    image_name = 'replaced_image.png'
    s3.upload_fileobj(image_obj, BUCKET_NAME, f'images/{image_name}')

    return {
        'statusCode': 200,
        'body': image_name
    }
