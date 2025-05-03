import os
import json
from textwrap import dedent
from typing import Optional
from src.utils import bedrock
from src.utils.bedrock import bedrock_info, bedrock_model
from src.config.agents import LLMType
from src.utils.bedrock import bedrock_utils, bedrock_chain
from langchain_aws import ChatBedrock
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import streamlit as st

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
    
class llm_call():

    def __init__(self, **kwargs):

        self.llm=kwargs["llm"]
        self.verbose = kwargs.get("verbose", False)
        self.chain = bedrock_chain(bedrock_utils.converse_api) | bedrock_chain(bedrock_utils.outputparser)

        self.origin_max_tokens = self.llm.inference_config["maxTokens"]
        self.origin_temperature = self.llm.inference_config["temperature"]

    def _message_format(self, role, message):
        if role == "user":
             message_format = {
                "role": "user",
                "content": [{"text": dedent(message)}]
            }
        elif role == "assistant":
            message_format = {
                "role": "assistant",
                'content': [{'text': message}]
            }

        return message_format
            
    def invoke(self, **kwargs):

        agent_name = kwargs.get("agent_name", None)
        system_prompts = kwargs.get("system_prompts", None)
        messages = kwargs["messages"]
        enable_reasoning = kwargs.get("enable_reasoning", False)
        reasoning_budget_tokens = kwargs.get("reasoning_budget_tokens", 1024)
        tool_config = kwargs.get("tool_config", None)
        efficient_token = kwargs.get("efficient_token", False)
        #print ("enable_reasoning", enable_reasoning)
        
        if efficient_token:
            additional_model_request_fields = self.llm.additional_model_request_fields
            if self.llm.additional_model_request_fields == None:
                efficient_token_config = {
                    "anthropic_beta": ["token-efficient-tools-2025-02-19"]  # Add this beta flag
                }
            else:
                additional_model_request_fields["anthropic_beta"] = ["token-efficient-tools-2025-02-19"]  # Add this beta flag
                efficient_token_config = additional_model_request_fields
            self.llm.additional_model_request_fields = efficient_token_config

        if enable_reasoning:
            additional_model_request_fields = self.llm.additional_model_request_fields
            if self.llm.additional_model_request_fields == None:
                reasoning_config = {
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": reasoning_budget_tokens
                    }
                }
            else:
                additional_model_request_fields["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": reasoning_budget_tokens
                }
                reasoning_config = additional_model_request_fields

            # Ensure maxTokens is greater than reasoning_budget
            if self.llm.inference_config["maxTokens"] <= reasoning_budget_tokens:
                # Make it just one token more than the reasoning budget
                adjusted_max_tokens = reasoning_budget_tokens + 1
                print(f'Info: Extended Thinking enabled increasing maxTokens from {self.llm.inference_config["maxTokens"]} to {adjusted_max_tokens} to exceed reasoning budget')
                self.llm.inference_config["maxTokens"] = adjusted_max_tokens

            self.llm.additional_model_request_fields = reasoning_config
            self.llm.inference_config["temperature"] = 1.0

        response = bedrock_utils.converse_api(
            llm=self.llm,
            system_prompts=system_prompts,
            messages=messages,
            tool_config=tool_config,
            verbose=self.verbose
        )
        
        tool_use = {}   
        output = {"text": "","reasoning": "", "signature": "", "toolUse": None}
        message = {"content": []}
        
        for event in response["response"]["stream"]:
            if 'messageStart' in event:
                message['role'] = event['messageStart']['role']
            elif 'contentBlockStart' in event:
                if 'toolUse' in event['contentBlockStart']['start']:
                    tool = event['contentBlockStart']['start']['toolUse']
                    tool_use['toolUseId'] = tool['toolUseId']
                    tool_use['name'] = tool['name']
            elif 'contentBlockDelta' in event:
                delta = event['contentBlockDelta']['delta']
                if "reasoningContent" in delta:
                    if "text" in delta["reasoningContent"]:
                        reasoning_text = delta["reasoningContent"]["text"]
                        output["reasoning"] += reasoning_text
                        print("\033[94m" + reasoning_text + "\033[0m", end="")
                        st.session_state["process_containers"][agent_name].markdown(output["reasoning"] + "▌")
                    elif 'signature' in delta["reasoningContent"]:
                        output["signature"] += delta["reasoningContent"]["signature"]
                    else:
                        print("") 
                if 'toolUse' in delta:
                    if 'input' not in tool_use: tool_use['input'] = ''
                    tool_use['input'] += delta['toolUse']['input']
                    print("\033[92m" + delta['toolUse']['input'] + "\033[0m", end="")
                    #st.session_state["tool_containers"][agent_name]["input"].markdown(tool_use["input"] + "▌")
                    
                elif 'text' in delta:
                    delta_text = delta['text']
                    output["text"] += delta_text
                    st.session_state["process_containers"][agent_name].markdown(output["text"] + "▌")
                         
            elif 'contentBlockStop' in event:
                if 'input' in tool_use:
                    tool_use['input'] = json.loads(tool_use['input'])
                    message['content'].append({'toolUse': tool_use})
                    output["toolUse"] = {'toolUse': tool_use}
                    tool_use = {}
                else:
                    if output["text"] != "":
                        message['content'].append({'text': output["text"]})
                    if output["reasoning"] != "":
                        message['content'].append({'reasoningContent': {"reasoningText": {"text": output["reasoning"], "signature": output["signature"]}}})
            elif 'messageStop' in event:
                stop_reason = event['messageStop']['stopReason']
                print ("stop_reason", stop_reason)
                output["stop_reason"] = stop_reason

        #st.session_state["ph1"].markdown(output["text"])
        st.session_state["process_containers"][agent_name].markdown(output["text"])
        st.session_state.messages.append({"role": "assistant", "content": output["text"]})
        
        #ai_message = self._message_format(role="assistant", message=output["text"])

        # Reset
        if enable_reasoning:
            self.llm.additional_model_request_fields = None
            self.llm.inference_config["maxTokens"] = self.origin_max_tokens
            self.llm.inference_config["temperature"] = self.origin_temperature
            
        return output, message

def get_llm_by_type(llm_type: LLMType):
    """
    Get LLM instance by type. Returns cached instance if available.
    """

    boto3_bedrock = bedrock.get_bedrock_client(
        assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
        endpoint_url=os.environ.get("BEDROCK_ENDPOINT_URL", None),
        region=os.environ.get("AWS_DEFAULT_REGION", None),
    )

    if llm_type == "reasoning":
        llm = bedrock_model(
            model_id=bedrock_info.get_model_id(model_name="Claude-V3-7-Sonnet-CRI"),
            bedrock_client=boto3_bedrock,
            stream=True,
            callbacks=[StreamingStdOutCallbackHandler()],
            inference_config={
                'maxTokens': 8192*3,
                #'stopSequences': ["\n\nHuman"],
                'temperature': 0.01,
            }
        )
        
    elif llm_type == "basic":
        llm = bedrock_model(
            #model_id=bedrock_info.get_model_id(model_name="Nova-Pro-CRI"),
            model_id=bedrock_info.get_model_id(model_name="Claude-V3-5-V-2-Sonnet-CRI"),
            bedrock_client=boto3_bedrock,
            stream=True,
            callbacks=[StreamingStdOutCallbackHandler()],
            inference_config={
                'maxTokens': 8192,
                #'stopSequences': ["\n\nHuman"],
                'temperature': 0.01,
            }
        )
    elif llm_type == "vision":
        llm = bedrock_model(
            #model_id=bedrock_info.get_model_id(model_name="Claude-V3-7-Sonnet-CRI"),
            model_id=bedrock_info.get_model_id(model_name="Claude-V3-5-V-2-Sonnet-CRI"),
            bedrock_client=boto3_bedrock,
            stream=True,
            callbacks=[StreamingStdOutCallbackHandler()],
            inference_config={
                'maxTokens': 8192,
                #'stopSequences': ["\n\nHuman"],
                'temperature': 0.01,
            }
        )
    elif llm_type == "browser":
        llm = ChatBedrock(
            #model_id=bedrock_info.get_model_id(model_name="Claude-V3-7-Sonnet-CRI"),
            model_id=bedrock_info.get_model_id(model_name="Claude-V3-5-V-2-Sonnet-CRI"),
            client=boto3_bedrock,
            model_kwargs={
                "max_tokens": 8192,
                "stop_sequences": ["\n\nHuman"],
            },
            #streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
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