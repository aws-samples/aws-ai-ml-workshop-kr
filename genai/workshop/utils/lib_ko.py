import json
import requests
import numpy as np
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, root_validator
from langchain.embeddings.base import Embeddings
from langchain.llms import AmazonAPIGateway
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
import os.path as osp

class Prompter(object):
    """
    A dedicated helper to manage templates and prompt building.
    """    
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("../templates", f"{template_name}.json")
        
        #file_name = osp.join("../templates", f"{template_name}.json")
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
    
prompter  = Prompter("kullm")

def get_payload(instruction, input_text, params):
    prompt = prompter.generate_prompt(instruction, input_text)
    payload = {
        'inputs': prompt,
        'parameters': params
    }
    return payload


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
        
    
class KoSimCSERobertaContentHandlerAmazonAPIGateway:

    @classmethod
    def transform_input(
        cls, prompt: str, model_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {"inputs": prompt, **model_kwargs}
    
    @classmethod
    def transform_output(cls, response: Any) -> str:
        response_json = response.json()
        ndim = np.array(response_json).ndim    
        if ndim == 4:
            # Original shape (1, 1, n, 768)
            emb = response_json[0][0][0]
            emb = np.expand_dims(emb, axis=0).tolist()
        elif ndim == 2:
            # Original shape (n, 1)
            emb = []
            for ele in response_json:
                e = ele[0][0]
                emb.append(e)
        else:
            print(f"Other # of dimension: {ndim}")
            emb = None
        return emb 
    

class EmbeddingAmazonApiGateway(BaseModel, Embeddings):

    api_url: str
    """API Gateway URL"""

    headers: Optional[Dict] = None
    """API Gateway HTTP Headers to send, e.g. for authentication"""

    model_kwargs: Optional[Dict] = None
    """Key word arguments to pass to the model."""

    content_handler: KoSimCSERobertaContentHandlerAmazonAPIGateway = KoSimCSERobertaContentHandlerAmazonAPIGateway()
    """The content handler class that provides an input and
    output transform functions to handle formats between LLM
    and the endpoint.
    """

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        try:
            if values["headers"] is None:
                values["headers"] = {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
        except Exception as error:
            pass

        return values
    
    class Config:
        """Configuration for this pydantic object."""
        skip_on_failure = True
        arbitrary_types_allowed=True
        # extra = Extra.forbid


    def _embedding_func(self, texts: List[str]) -> List[List[float]]:
        """Call out to SageMaker Inference embedding endpoint."""
        # replace newlines, which can negatively affect performance.
        texts = list(map(lambda x: x.replace("\n", " "), texts))

        _model_kwargs = self.model_kwargs or {}

        payload = self.content_handler.transform_input(texts, _model_kwargs)
        
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
            )

            text = self.content_handler.transform_output(response)

        except Exception as error:
            raise ValueError(f"Error raised by the service: {error}")
        
        return text

    def embed_documents(
        self, texts: List[str], chunk_size: int = 64
    ) -> List[List[float]]:
        """Compute doc embeddings using a SageMaker Inference Endpoint.

        Args:
            texts: The list of texts to embed.
            chunk_size: The chunk size defines how many input texts will
                be grouped together as request. If None, will use the
                chunk size specified by the class.


        Returns:
            List of embeddings, one for each text.
        """
        results = []
        _chunk_size = len(texts) if chunk_size > len(texts) else chunk_size
        #print(f"_chunk size={_chunk_size}")
        
        for i in range(0, len(texts), _chunk_size):    
            #print (i, texts[i : i + _chunk_size])
            response = self._embedding_func(texts[i : i + _chunk_size])
            #print (i, response, len(response[0].shape))
            results.extend(response)
            
        return results

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a SageMaker inference endpoint.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        
        return self._embedding_func([text])[0]
