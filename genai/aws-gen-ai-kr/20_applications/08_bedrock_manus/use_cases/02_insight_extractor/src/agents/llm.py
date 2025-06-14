import os
from textwrap import dedent
from typing import Optional
from src.utils import bedrock
from src.utils.bedrock import bedrock_info, bedrock_model
from src.config.agents import LLMType
from src.utils.bedrock import bedrock_utils, bedrock_chain
from langchain_aws import ChatBedrock
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from strands.models import BedrockModel
from botocore.config import Config

from src.config import (
    REASONING_MODEL,
    REASONING_BASE_URL,
    REASONING_API_KEY,
    BASIC_MODEL,
    BASIC_BASE_URL,
    BASIC_API_KEY,
    VL_MODEL,
    VL_BASE_URL,
    VL_API_KEY,
)
"""
Supported models:
"Claude-Instant-V1": "anthropic.claude-instant-v1",
"Claude-V1": "anthropic.claude-v1",
"Claude-V2": "anthropic.claude-v2",
"Claude-V2-1": "anthropic.claude-v2:1",
"Claude-V3-Sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
"Claude-V3-Haiku": "anthropic.claude-3-haiku-20240307-v1:0",
"Claude-V3-Opus": "anthropic.claude-3-sonnet-20240229-v1:0",
"Claude-V3-5-Sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
"Claude-V3-5-V-2-Sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0",
"Claude-V3-5-V-2-Sonnet-CRI": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
"Claude-V3-7-Sonnet-CRI": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
"Jurassic-2-Mid": "ai21.j2-mid-v1",
"Jurassic-2-Ultra": "ai21.j2-ultra-v1",
"Command": "cohere.command-text-v14",
"Command-Light": "cohere.command-light-text-v14",
"Cohere-Embeddings-En": "cohere.embed-english-v3",
"Cohere-Embeddings-Multilingual": "cohere.embed-multilingual-v3",
"Titan-Embeddings-G1": "amazon.titan-embed-text-v1",
"Titan-Text-Embeddings-V2": "amazon.titan-embed-text-v2:0",
"Titan-Text-G1": "amazon.titan-text-express-v1",
"Titan-Text-G1-Light": "amazon.titan-text-lite-v1",
"Titan-Text-G1-Premier": "amazon.titan-text-premier-v1:0",
"Titan-Text-G1-Express": "amazon.titan-text-express-v1",
"Llama2-13b-Chat": "meta.llama2-13b-chat-v1",
"Nova-Canvas": "amazon.nova-canvas-v1:0",
"Nova-Reel": "amazon.nova-reel-v1:0",
"Nova-Micro": "amazon.nova-micro-v1:0",
"Nova-Lite": "amazon.nova-lite-v1:0",
"Nova-Pro": "amazon.nova-pro-v1:0",
"Nova-Pro-CRI": "us.amazon.nova-pro-v1:0",
"SD-3-5-Large": "stability.sd3-5-large-v1:0",
"SD-Ultra": "stability.stable-image-ultra-v1:1",
"SD-3-Large": "stability.sd3-large-v1:0"
"""

def get_llm_by_type(llm_type, cache_type=None, enable_reasoning=False):
    """
    Get LLM instance by type. Returns cached instance if available.
    """
    if llm_type == "reasoning":
        
        ## BedrockModel params: https://strandsagents.com/latest/api-reference/models/?h=bedrockmodel#strands.models.bedrock.BedrockModel
        llm = BedrockModel(
            model_id=bedrock_info.get_model_id(model_name="Claude-V3-7-Sonnet-CRI"),
            streaming=True,
            max_tokens=8192*5,
            stop_sequencesb=["\n\nHuman"],
            temperature=1 if enable_reasoning else 0.01, 
            additional_request_fields={
                "thinking": {
                    "type": "enabled" if enable_reasoning else "disabled", 
                    **({"budget_tokens": 8192} if enable_reasoning else {}),
                }
            },
            cache_prompt=cache_type, # None/ephemeral/defalut
            #cache_tools: Cache point type for tools
            boto_client_config=Config(
                read_timeout=900,
                connect_timeout=900,
                retries=dict(max_attempts=50, mode="adaptive"),
            )
        )
        
    elif llm_type == "basic":
        ## BedrockModel params: https://strandsagents.com/latest/api-reference/models/?h=bedrockmodel#strands.models.bedrock.BedrockModel
        llm = BedrockModel(
            model_id=bedrock_info.get_model_id(model_name="Claude-V3-5-V-2-Sonnet-CRI"),
            streaming=True,
            max_tokens=8192,
            stop_sequencesb=["\n\nHuman"],
            temperature=0.01,
            cache_prompt=cache_type, # None/ephemeral/defalut
            #cache_tools: Cache point type for tools
            boto_client_config=Config(
                read_timeout=900,
                connect_timeout=900,
                retries=dict(max_attempts=50, mode="standard"),
            )
        )

    elif llm_type == "vision":
         ## BedrockModel params: https://strandsagents.com/latest/api-reference/models/?h=bedrockmodel#strands.models.bedrock.BedrockModel
        llm = BedrockModel(
            model_id=bedrock_info.get_model_id(model_name="Claude-V3-5-V-2-Sonnet-CRI"),
            streaming=True,
            max_tokens=8192,
            stop_sequencesb=["\n\nHuman"],
            temperature=1 if enable_reasoning else 0.01, 
            additional_request_fields={
                "thinking": {
                    "type": "enabled" if enable_reasoning else "disabled", 
                    **({"budget_tokens": 2048} if enable_reasoning else {}),
                }
            },
            cache_prompt=cache_type, # None/ephemeral/defalut
            #cache_tools: Cache point type for tools
            boto_client_config=Config(
                read_timeout=900,
                connect_timeout=900,
                retries=dict(max_attempts=50, mode="adaptive"),
            )
        )
    elif llm_type == "browser":
         ## BedrockModel params: https://strandsagents.com/latest/api-reference/models/?h=bedrockmodel#strands.models.bedrock.BedrockModel
        llm = BedrockModel(
            model_id=bedrock_info.get_model_id(model_name="Claude-V3-5-V-2-Sonnet-CRI"),
            streaming=True,
            max_tokens=8192,
            stop_sequencesb=["\n\nHuman"],
            temperature=1 if enable_reasoning else 0.01, 
            additional_request_fields={
                "thinking": {
                    "type": "enabled" if enable_reasoning else "disabled", 
                    **({"budget_tokens": 2048} if enable_reasoning else {}),
                }
            },
            cache_prompt=cache_type, # None/ephemeral/defalut
            #cache_tools: Cache point type for tools
            boto_client_config=Config(
                read_timeout=900,
                connect_timeout=900,
                retries=dict(max_attempts=50, mode="adaptive"),
            )
        )
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")
        
    return llm

# Initialize LLMs for different purposes - now these will be cached
reasoning_llm = get_llm_by_type("reasoning")
basic_llm = get_llm_by_type("basic")
browser_llm = get_llm_by_type("browser")

if __name__ == "__main__":
    stream = reasoning_llm.stream("what is mcp?")
    full_response = ""
    for chunk in stream:
        full_response += chunk.content
    print(full_response)

    basic_llm.invoke("Hello")
    browser_llm.invoke("Hello")