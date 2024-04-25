import json
import time
import boto3
import os.path as osp
from typing import Union
import pprint
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
        #file_name = osp.join("templates", f"{template_name}.json")
        file_name = osp.join("./utils", f"{template_name}.json")
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
    
def invoke_inference(endpoint_name, prompt):

    client = boto3.client("sagemaker-runtime")
    content_type = "application/json"
    
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType=content_type,
        Body=json.dumps(prompt)
    )
    res = response["Body"].read().decode()
    
    return res

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
    
    
    
class KoLLMSageMakerEndpoint(object):
    def __init__(self, endpoint_name):
        self.endpoint_name = endpoint_name
        self.prompter  = Prompter("kullm")
        self.smr_client  = boto3.client('sagemaker-runtime')
        
    def get_payload(self, instruction, input_text, params):
        prompter = Prompter("kullm")
        prompt = prompter.generate_prompt(instruction, input_text)
        payload = {
            'inputs': prompt,
            'parameters': params
        }
        return payload

    def infer(self, payload, verbose=True):
        
        content_type = "application/json"
        response = self.smr_client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType=content_type,
            Body=json.dumps(payload)
        )

        #model_predictions = json.loads(response['Body'].read().decode())
        #s = model_predictions[0]['generated_text']
        #generated_text = self.prompter.get_response(s)
        res = response["Body"].read().decode()
        generated_text = parse_response(res)
        generated_text = generated_text.split('###')[0]
        if verbose:
            pprint.pprint(f'Response: {generated_text}')
        return generated_text
