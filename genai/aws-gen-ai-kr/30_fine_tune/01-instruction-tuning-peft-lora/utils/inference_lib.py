import boto3
import time
import json
import os.path as osp
from typing import Union
import pprint

def parse_response(query_response):
    
    def traverse(o, tree_types=(list, tuple)):
        if isinstance(o, tree_types):
            for value in o:
                for subvalue in traverse(value, tree_types):
                    yield subvalue
        else:
            yield o

    data = eval(query_response)
    
    listRes = []
    for value in traverse(data):
        listRes.append(value["generated_text"])
        
    if len(listRes) >= 2: return listRes
    else: return listRes[0].strip()

# def invoke_inference(endpoint_name, prompt):
#     '''
#     KoAlpaca 프롬프트를 제공하여 엔드포인트 호출
#     '''
#     client = boto3.client("sagemaker-runtime")
    
#     content_type = "text/plain"
#     response = client.invoke_endpoint(
#         EndpointName=endpoint_name, ContentType=content_type, Body=prompt
#     )
#     #print(response["Body"].read())
#     res = response["Body"].read().decode()
#     print (eval(res)[0]['generated_text'])

# def invoke_inference_DJ(endpoint_name, prompt):

#     client = boto3.client("sagemaker-runtime")

#     content_type = "application/json"
#     response = client.invoke_endpoint(
#         EndpointName=endpoint_name,
#         ContentType=content_type,
#         Body=json.dumps(prompt)
#     )

#     res = response["Body"].read().decode()
#     return res

# def query_endpoint_with_text_payload(plain_text, endpoint_name, content_type="text/plain"):
#     '''
#     content_type 이 text/plain 인 경우 사용
#     '''
#     client = boto3.client("runtime.sagemaker")
#     response = client.invoke_endpoint(
#         EndpointName=endpoint_name, ContentType=content_type, Body=plain_text
#     )
#     return response


# def parse_response_text_model(query_response):
#     '''
#     content_type 이 text/plain 인 경우 사용
#     '''
    
#     model_predictions = json.loads(query_response["Body"].read())
#     # print("model_predictions: \n", model_predictions)
#     generated_text = model_predictions[0]["generated_text"]
#     return generated_text


"""
A dedicated helper to manage templates and prompt building.
"""

class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("./templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


def describe_endpoint(endpoint_name):
    '''
    엔드폰인트 생성 유무를 확인. 생성 중이면 기다림.
    '''
    sm_client = boto3.client("sagemaker")

    while(True):
        response = sm_client.describe_endpoint(
            EndpointName= endpoint_name
        )    
        status = response['EndpointStatus']
        if status == 'Creating':
            print("Endpoint is ", status)
            time.sleep(60)
        else:
            print("Endpoint is ", status)
            break


class KoLLMSageMakerEndpoint(object):
    def __init__(self, endpoint_name):
        self.endpoint_name = endpoint_name
        self.prompter  = Prompter("kullm")
        self.smr_client  = boto3.client('sagemaker-runtime')
        
    def get_payload(self, instruction, input_text, params):
        prompt = self.prompter.generate_prompt(instruction, input_text)
        payload = {
            'inputs': prompt,
            'parameters': params
        }
        payload_str = json.dumps(payload)
        return payload_str.encode("utf-8")

    def infer(self, payload, content_type="application/json", verbose=True):
        response = self.smr_client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType=content_type,
            Body=payload
        )

        res = json.loads(response['Body'].read().decode("utf-8"))
        generated_text = res[0]["generated_text"]
        #generated_text = self.prompter.get_response(generated_text)

        generated_text = generated_text.split('###')[0]
        if verbose:
            pprint.pprint(f'Response: {generated_text}')
        return generated_text

    
################################################
# Embedding Handler
################################################

# from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
# from langchain.embeddings import SagemakerEndpointEmbeddings
# from langchain.llms.sagemaker_endpoint import ContentHandlerBase
# from typing import Any, Dict, List, Optional

# class SagemakerEndpointEmbeddingsJumpStart(SagemakerEndpointEmbeddings):
#     def embed_documents(self, texts: List[str], chunk_size: int = 5) -> List[List[float]]:
#         """Compute doc embeddings using a SageMaker Inference Endpoint.

#         Args:
#             texts: The list of texts to embed.
#             chunk_size: The chunk size defines how many input texts will
#                 be grouped together as request. If None, will use the
#                 chunk size specified by the class.

#         Returns:
#             List of embeddings, one for each text.
#         """
#         results = []
#         _chunk_size = len(texts) if chunk_size > len(texts) else chunk_size

#         # print("text size: ", len(texts))
#         # print("_chunk_size: ", _chunk_size)

#         for i in range(0, len(texts), _chunk_size):
#             response = self._embedding_func(texts[i : i + _chunk_size])
#             print
#             results.extend(response)
#         return results

# import numpy as np
    
# class KoSimCSERobertaContentHandler(EmbeddingsContentHandler):
#     content_type = "application/json"
#     accepts = "application/json"

#     def transform_input(self, prompt: str, model_kwargs={}) -> bytes:
#         input_str = json.dumps({"inputs": prompt, **model_kwargs})
#         return input_str.encode("utf-8")

#     def transform_output(self, output: bytes) -> str:
#         response_json = json.loads(output.read().decode("utf-8"))
#         ndim = np.array(response_json).ndim    
#         # print("response_json ndim: \n", ndim)        
#         # print("response_json shape: \n", np.array(response_json).shape)    
#         if ndim == 4:
#             # Original shape (1, 1, n, 768)
#             emb = response_json[0][0][0]
#             emb = np.expand_dims(emb, axis=0).tolist()
#             # print("emb shape: ", np.array(emb).shape)
#             # print("emb TYPE: ", type(emb))        
#         elif ndim == 2:
#             # Original shape (n, 1)
#             # print(response_json[0])
#             emb = []
#             for ele in response_json:
#                 # print(np.array(response_json[0]).shape)
#                 e = ele[0][0]
#                 #emb = np.expand_dims(emb, axis=0).tolist()
#                 # print("emb shape: ", np.array(emb).shape)
#                 # print("emb TYPE: ", type(emb))        
#                 emb.append(e)
#             # print("emb_list shape: ", np.array(emb).shape)
#             # print("emb_list TYPE: ", type(emb))        
#         else:
#             print(f"Other # of dimension: {ndim}")
#             emb = None
#         return emb
    

# ################################################
# # LLM  Handler
# ################################################
# from langchain.llms.sagemaker_endpoint import LLMContentHandler
# import json

# class KoAlpacaContentHandler(LLMContentHandler):
#     content_type = "application/json"
#     accepts = "application/json"

#     def transform_input(self, prompt: str, model_kwargs={}) -> bytes:
#         input_str = json.dumps({"text_inputs": prompt, **model_kwargs})
#         return input_str.encode("utf-8")

#     def transform_output(self, output: bytes) -> str:
#         print("In KoAlpacaContentHandler")
#         # print("output: ", output)
#         response_json = json.loads(output.read().decode("utf-8"))
#         print("response_json: ", response_json)        
# #        return response_json["generated_texts"][0]
#         doc = response_json[0]['generated_text']
#         doc = json.loads(doc)
#         doc = doc['text_inputs']
#         return doc