import json
import boto3
import numpy as np
from inference_utils import Prompter
from typing import Any, Dict, List, Optional
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.llms.sagemaker_endpoint import LLMContentHandler, SagemakerEndpoint
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler

prompter = Prompter("kullm")

class KullmContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs={}) -> bytes:
        '''
        입력 데이터 전처리 후에 리턴
        '''
        context, question = prompt.split("||SPEPERATOR||")
        prompt = prompter.generate_prompt(question, context)

        # print ("prompt", prompt)
        payload = {
            'inputs': [prompt],
            'parameters': model_kwargs
        }

        input_str = json.dumps(payload)

        return input_str.encode('utf-8')

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        generated_text = response_json[0][0]["generated_text"]

        return generated_text


class SagemakerEndpointEmbeddingsJumpStart(SagemakerEndpointEmbeddings):
    def embed_documents(self, texts: List[str], chunk_size: int = 1) -> List[List[float]]:
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

        print("text size: ", len(texts))
        print("_chunk_size: ", _chunk_size)

        for i in range(0, len(texts), _chunk_size):
            # print (i, texts[i : i + _chunk_size])
            response = self._embedding_func(texts[i: i + _chunk_size])
            # print (i, response, len(response[0].shape))

            results.extend(response)
        return results


class KoSimCSERobertaContentHandler(EmbeddingsContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs={}) -> bytes:

        input_str = json.dumps({"inputs": prompt, **model_kwargs})

        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:

        response_json = json.loads(output.read().decode("utf-8"))
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


