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

# Removed unused OpenAI-style environment variables - using Bedrock models directly
"""
Supported models:
"Claude-V3-5-V-2-Sonnet-CRI": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
"Claude-V3-7-Sonnet-CRI": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
"
"""

def get_llm_by_type(llm_type, cache_type=None, enable_reasoning=False):
    """
    Get LLM instance by type. Returns cached instance if available.
    """
    if llm_type == "claude-sonnet-3-7":
        
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
        
    elif llm_type == "claude-sonnet-3-5-v-2":
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

    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")
        
    return llm

if __name__ == "__main__":
    print ("hello")