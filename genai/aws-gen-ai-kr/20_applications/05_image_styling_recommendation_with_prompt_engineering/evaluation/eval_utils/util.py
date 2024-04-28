import os
import base64
import json
import boto3
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def set_body(prompt, image, multimodal):
    multimodal_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "temperature": 0.0,
            "top_k": 5,
            "top_p": 0.3,            
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image,
                            },
                        },
                    ],
                }
            ],
        }

    text_only_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "temperature": 0.0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        }

    if multimodal:
        return multimodal_body
    else:
        return text_only_body

from botocore.exceptions import ClientError

def invoke_claude3_sonnet_model(client, claude3_model_id, prompt, image, multimodal=False):
    """
    Invokes Anthropic Claude 3 Sonnet to run a multimodal inference using the input
    provided in the request body.

    :param prompt:            The prompt that you want Claude 3 to use.
    :param base64_image_data: The base64-encoded image that you want to add to the request.
    :return: Inference response from the model.
    """

    # Invoke the model with the prompt and the encoded image
    model_id = claude3_model_id
    request_body = set_body(prompt, image, multimodal)

    try:
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body),
        )

        # Process and print the response
        result = json.loads(response.get("body").read())
        input_tokens = result["usage"]["input_tokens"]
        output_tokens = result["usage"]["output_tokens"]
        output_list = result.get("content", [])

        #print("Invocation details:")
        #print(f"- The input length is {input_tokens} tokens.")
        #print(f"- The output length is {output_tokens} tokens.")

        print(f"- The model returned {len(output_list)} response(s):")
        for output in output_list:
            print(output["text"])

        return result

    except ClientError as err:
        logger.error(
            "Couldn't invoke Claude 3 Sonnet. Here's why: %s: %s",
            err.response["Error"]["Code"],
            err.response["Error"]["Message"],
        )
        raise

def encoding_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf8")

    return encoded_image

from PIL import Image

def extract_properties_from_image(all_items,client,claude3_model_id,multimodal_text_prompt ):
    target = None
    associates = []
    for k, v in all_items.items():
        # print(k, v)
        if 'target' in k:
            print("## This is target image")
            print(v)
            target = invoke_claude3_sonnet_model(client, claude3_model_id, multimodal_text_prompt,
                                                encoding_image(v), multimodal=True)
            Image.open(v).show()
        else:
            for asso in v:
                print("## This is candidate image")                
                # print(asso)
                image_path = {"image_path": asso}
                response = invoke_claude3_sonnet_model(
                    client, claude3_model_id, multimodal_text_prompt, encoding_image(asso), multimodal=True)
                merged_dict = {**image_path, **response}                    
                associates.append(merged_dict)
                Image.open(asso).show()

    return target, associates



def get_files_in_folder(folder_path):
    # folder_path = f'samples/{sample}/'
    files = {'target': None, 'associates': []}
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            if 'target' in file_path:
                files['target'] = file_path
            else:
                files['associates'].append(file_path)

    return files


from io import StringIO
import sys
import textwrap



import json

def print_ww(*args, width: int = 100, **kwargs):
    """Like print(), but wraps output to `width` characters (default 100)"""
    buffer = StringIO()
    try:
        _stdout = sys.stdout
        sys.stdout = buffer
        print(*args, **kwargs)
        output = buffer.getvalue()
    finally:
        sys.stdout = _stdout
    for line in output.splitlines():
        print("\n".join(textwrap.wrap(line, width=width)))

def pretty_print_json(data, sort_keys=False):
    print(json.dumps(data, indent=4, sort_keys=sort_keys))                                           


def parse_output_select_reason(response):
    select_reason_sonnet = response["content"][0]["text"]
    reason = json.loads(select_reason_sonnet)["reason"]

    return reason


# from langchain.callbacks import StreamlitCallbackHandler
# model_id="anthropic.claude-3-sonnet-20240229-v1:0", # Claude 3 Sonnet 모델 선택
# # 텍스트 생성 LLM 가져오기, streaming_callback을 인자로 받아옴
# def get_llm(boto3_bedrock, model_id):
#     llm = BedrockChat(
#     model_id= model_id,
#     client=boto3_bedrock,
#     model_kwargs={
#         "max_tokens": 1024,
#         "stop_sequences": ["\n\nHuman"],
#     }
#     )
#     return llm
# llm = get_llm(boto3_bedrock=client, model_id = model_id)
# response_text = llm.invoke(prompt) #프롬프트에 응답 반환
# print(response_text.content)
