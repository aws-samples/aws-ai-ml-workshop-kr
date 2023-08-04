import os
import io
import json
import boto3
import base64
import logging
import numpy as np

from chalice import Chalice
from chalice import BadRequestError

app = Chalice(app_name="genai-rag-workshop")
app.debug = True

smr_client = boto3.client("runtime.sagemaker")
logger = logging.getLogger("genai-rag-workshop")
logger.setLevel(logging.DEBUG)

@app.route("/")
def index():
    return {'hello': 'world'}


@app.route("/emb/{variant_name}", methods=["POST"], content_types=["application/json"])
def invoke_emb(variant_name):

    models = ['gptj_6b', 'kosimcse']
    if variant_name not in models:
        raise BadRequestError("[ERROR] Invalid model!")
    
    logger.info(f"embedding model: {variant_name}")

    if variant_name == "gptj_6b":
        endpoint_name = os.environ["ENDPOINT_EMB_GPTJ_6B"]
    elif variant_name == "kosimcse":
        endpoint_name = os.environ["ENDPOINT_EMB_KOSIMCSE"]        

    payload = app.current_request.json_body

    try:
        response = smr_client.invoke_endpoint(
            EndpointName=endpoint_name, 
            ContentType='application/json',                        
            Body=json.dumps(payload).encode("utf-8")
        ) 
        res = response['Body'].read()
        return json.loads(res.decode("utf-8"))

    except Exception as e:
        print(e)
        print(payload)
        
        
@app.route("/llm/{variant_name}", methods=["POST"], content_types=["application/json"])
def invoke_llm(variant_name):
    
    models = ['llama2_7b', 'llama2_13b', 'kkulm_12_8b']
    if variant_name not in models:
        raise BadRequestError("[ERROR] Invalid model!")
        
    logger.info(f"txt2txt model: {variant_name}")

    if variant_name == "llama2_7b":
        endpoint_name = os.environ["ENDPOINT_LLM_LLAMA2_7B"]
    elif variant_name == "llama2_13b":
        endpoint_name = os.environ["ENDPOINT_LLM_LLAMA2_13B"]
    elif variant_name == "kkulm_12_8b":
        endpoint_name = os.environ["ENDPOINT_LLM_KKULM_12_8B"]

    payload = app.current_request.json_body

    try:
        if "llama2" in variant_name:
            response = smr_client.invoke_endpoint(
                EndpointName=endpoint_name, 
                ContentType='application/json',                        
                Body=json.dumps(payload).encode("utf-8"),
                CustomAttributes="accept_eula=true",
            )
        else:
             response = smr_client.invoke_endpoint(
                EndpointName=endpoint_name, 
                ContentType='application/json',                        
                Body=json.dumps(payload).encode("utf-8")
            )           
        res = response['Body'].read()
        return json.loads(res.decode("utf-8"))
        
    except Exception as e:
        print(e)
        print(payload)