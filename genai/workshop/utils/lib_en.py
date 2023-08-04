import json
import requests
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, root_validator
from langchain.embeddings.base import Embeddings
from langchain.llms import AmazonAPIGateway
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType


class Llama2ContentHandlerAmazonAPIGateway:
    """Adapter to prepare the inputs from Langchain to a format
    that LLM model expects.

    It also provides helper function to extract
    the generated text from the model response."""

    @classmethod
    def transform_input(
        cls, prompt: str, model_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {"inputs": prompt, "parameters": model_kwargs}

    @classmethod
    def transform_output(cls, response: Any) -> str:
        return response.json()[0]["generation"]
    
    
class ContentHandlerEmbeddingAmazonAPIGateway:
    """Adapter to prepare the inputs from Langchain to a format
    that LLM model expects.

    It also provides helper function to extract
    the generated text from the model response."""

    @classmethod
    def transform_input(
        cls, prompt: str, model_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {"text_inputs": prompt}

    @classmethod
    def transform_output(cls, response: Any) -> str:
        return response.json()["embedding"]
        
        
class EmbeddingAmazonApiGateway(BaseModel, Embeddings):

    api_url: str
    """API Gateway URL"""

    headers: Optional[Dict] = None
    """API Gateway HTTP Headers to send, e.g. for authentication"""

    model_kwargs: Optional[Dict] = None
    """Key word arguments to pass to the model."""

    content_handler: ContentHandlerEmbeddingAmazonAPIGateway = ContentHandlerEmbeddingAmazonAPIGateway()
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
        
        # content_type = self.content_handler.content_type
        # accepts = self.content_handler.accepts
        
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
        for i in range(0, len(texts), _chunk_size):
            response = self._embedding_func(texts[i : i + _chunk_size])
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
    
    