import json
from typing import List, Dict
import boto3
import botocore

import logging
logger = logging.getLogger(__name__)

class BedrockService:
    def __init__(self, aws_region: str, aws_profile: str, retries: int, embed_model_id: str, model_id: str, max_tokens: int, temperature: float, top_p: float):
        self.embed_model_id = embed_model_id
        self.model_id = model_id
        
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.bedrock_client = self._init_bedrock_client(aws_region, aws_profile, retries)
    
    def embedding(self, text: str):
        response = self.bedrock_client.invoke_model(
            modelId=self.embed_model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"inputText": text})
        )
        return json.loads(response['body'].read())['embedding']
    
    def converse(self, messages: List[Dict], system_prompt: str, model_id: None=None, max_tokens: None=None, temperature: None=None, top_p: None=None) -> Dict:
        try:
            return self.bedrock_client.converse(
                modelId=model_id if model_id else self.model_id,
                messages=messages,
                system=[{'text': system_prompt}],
                inferenceConfig={
                    'maxTokens': max_tokens if max_tokens else self.max_tokens,
                    'temperature': temperature if temperature else self.temperature,
                    'topP': top_p if top_p else self.top_p
                }
            )
        except Exception as e:
            logger.error(f"Error in converse: {e}")
            raise
    
    def converse_with_tools(self, messages: List[Dict], system_prompt: str, tools: List[Dict], model_id: None=None, max_tokens: None=None, temperature: None=None, top_p: None=None, top_k: None = None) -> Dict:
        try:
            return self.bedrock_client.converse(
                modelId=model_id if model_id else self.model_id,
                messages=messages,
                system=[{'text': system_prompt}],
                toolConfig=tools,
                inferenceConfig={
                    'maxTokens': max_tokens if max_tokens else self.max_tokens,
                    'temperature': temperature if temperature else self.temperature,
                    'topP': top_p if top_p else self.top_p
                },
                additionalModelRequestFields={
                    'topK': top_k if top_k else None
                }
            )
        except Exception as e:
            logger.error(f"Error in converse_with_tools: {e}")
            raise
        
    def _init_bedrock_client(self, aws_region: str, aws_profile: str, retries: int):
        retry_config = botocore.config.Config(
            retries={"max_attempts": retries, "mode": "standard"}
        )
        return boto3.Session(
            region_name=aws_region,
            profile_name=aws_profile
        ).client("bedrock-runtime", config=retry_config)
