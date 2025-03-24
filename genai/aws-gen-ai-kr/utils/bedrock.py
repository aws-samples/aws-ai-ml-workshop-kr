# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Helper utilities for working with Amazon Bedrock from Python notebooks"""
# Python Built-Ins:
import os
from typing import Optional

# External Dependencies:
import json
import boto3
from textwrap import dedent
from botocore.config import Config
from botocore.exceptions import ClientError

# Langchain
from langchain.callbacks.base import BaseCallbackHandler

# LangFuse
from langfuse.decorators import observe, langfuse_context

def get_bedrock_client(
    assumed_role: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    region: Optional[str] = None,
):
    """Create a boto3 client for Amazon Bedrock, with optional configuration overrides

    Parameters
    ----------
    assumed_role :
        Optional ARN of an AWS IAM role to assume for calling the Bedrock service. If not
        specified, the current active credentials will be used.
    endpoint_url :
        Optional override for the Bedrock service API Endpoint. If setting this, it should usually
        include the protocol i.e. "https://..."
    region :
        Optional name of the AWS Region in which the service should be called (e.g. "us-east-1").
        If not specified, AWS_REGION or AWS_DEFAULT_REGION environment variable will be used.
    """
    if region is None:
        target_region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION"))
    else:
        target_region = region

    print(f"Create new client\n  Using region: {target_region}")
    session_kwargs = {"region_name": target_region}
    client_kwargs = {**session_kwargs}

    profile_name = os.environ.get("AWS_PROFILE")
    print(f"  Using profile: {profile_name}")
    if profile_name:
        print(f"  Using profile: {profile_name}")
        session_kwargs["profile_name"] = profile_name

    retry_config = Config(
        region_name=target_region,
        retries={
            "max_attempts": 50,
            "mode": "standard",
        },
    )
    session = boto3.Session(**session_kwargs)

    if assumed_role:
        print(f"  Using role: {assumed_role}", end='')
        sts = session.client("sts")
        response = sts.assume_role(
            RoleArn=str(assumed_role),
            RoleSessionName="langchain-llm-1"
        )
        print(" ... successful!")
        client_kwargs["aws_access_key_id"] = response["Credentials"]["AccessKeyId"]
        client_kwargs["aws_secret_access_key"] = response["Credentials"]["SecretAccessKey"]
        client_kwargs["aws_session_token"] = response["Credentials"]["SessionToken"]

    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url

    bedrock_client = session.client(
        service_name="bedrock-runtime",
        config=retry_config,
        **client_kwargs
    )

    print("boto3 Bedrock client successfully created!")
    print(bedrock_client._endpoint)
    return bedrock_client


class bedrock_info():

    _BEDROCK_MODEL_INFO = {
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
    }

    @classmethod
    def get_list_fm_models(cls, verbose=False):

        if verbose:
            bedrock = boto3.client(service_name='bedrock')
            model_list = bedrock.list_foundation_models()
            return model_list["modelSummaries"]
        else:
            return cls._BEDROCK_MODEL_INFO

    @classmethod
    def get_model_id(cls, model_name):

        assert model_name in cls._BEDROCK_MODEL_INFO.keys(), "Check model name"

        return cls._BEDROCK_MODEL_INFO[model_name]
    
class bedrock_model():

    def __init__(self, **kwargs):

        self.model_id = kwargs["model_id"] 
        self.bedrock_client = kwargs["bedrock_client"]
        self.stream = kwargs.get("stream", False)
        self.callbacks = kwargs.get("callbacks", None)
        self.inference_config = kwargs.get("inference_config", None)
        self.additional_model_request_fields = kwargs.get("additional_model_request_fields", None)

class bedrock_chain:

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __or__(self, other):
        def combined(*args, **kwargs):
            result = self(*args, **kwargs)
            if isinstance(result, dict): return other(**result)
            else: return other(result)
        return bedrock_chain(combined)

class bedrock_utils():

    @staticmethod
    def get_message_from_string(role, string, img=None):
        message = {
            "role": role,
            "content": [{"text": dedent(string)}]
        }
        if img is not None:
            img_message = {
                "image": {
                    "format": 'png',
                    "source": {"bytes": img}
                }
            }
            message["content"].append(img_message)

        return message
        
    @staticmethod
    def get_system_prompt(system_prompts):
        return [{"text": system_prompts}]

    @staticmethod
    @observe(as_type="generation", name="ConverseAPI")
    def converse_api(**kwargs):

        llm = kwargs["llm"]
        model_id = llm.model_id
        bedrock_client = llm.bedrock_client
        inference_config = llm.inference_config
        additional_model_request_fields = llm.additional_model_request_fields
        stream = llm.stream

        messages = kwargs["messages"]
        system_prompts = kwargs.get("system_prompts", None)
        tool_config = kwargs.get("tool_config", None)
        verbose = kwargs.get("verbose", False)
        tracking = kwargs.get("tracking", False)
        args = {}

        if system_prompts != None: args["system"] = system_prompts
        if inference_config != None: args["inferenceConfig"] = inference_config
        if additional_model_request_fields != None: args["additionalModelRequestFields"] = additional_model_request_fields
        else: args["additionalModelRequestFields"] = {}
        if tool_config != None:
            args["toolConfig"] = tool_config
            print ('args["toolConfig"]', args["toolConfig"])
        args["messages"], args["modelId"] = messages, model_id

        if tracking:
            # Langfuse 관측 컨텍스트에 입력, 모델 ID, 파라미터, 기타 메타데이터를 업데이트합니다.
            langfuse_context.update_current_observation(
                input=args["messages"],
                model=args["modelId"],
                model_parameters={**args["inferenceConfig"], **args["additionalModelRequestFields"]},
                metadata=args.copy()
            )
            
        if stream:
            try:
                response = bedrock_client.converse_stream(**args)
            except (ClientError, Exception) as e:
                error_message = f"ERROR: Can't invoke '{model_id}'. Reason: {e}"
                if tracking: langfuse_context.update_current_observation(level="ERROR", status_message=error_message)
                print(error_message)
        else:
            try:
                response = bedrock_client.converse(**args)
            except (ClientError, Exception) as e:
                error_message = f"ERROR: Can't invoke '{model_id}'. Reason: {e}"
                if tracking: langfuse_context.update_current_observation(level="ERROR", status_message=error_message)
                print(error_message)
                

        # if stream:
        #     response = bedrock_client.converse_stream(**args)
        # else:
        #     response = bedrock_client.converse(**args)

            
        return {"response": response, "verbose": verbose, "stream": stream, "callback": llm.callbacks[0]}

    @staticmethod
    def outputparser(**kwargs):

        response = kwargs["response"]
        verbose = kwargs.get("verbose", False)
        stream = kwargs["stream"]
        callback = kwargs["callback"]
        
        output = {"text": "", "toolUse": None}
        if not stream:

            try:
                output_message = response['output']['message']

                for content in output_message['content']:
                    if content.get("text"):
                        output["text"] = content['text']
                if verbose:
                    for content in output_message['content']:
                        if content.get("text"):
                            print(f"Text: {content['text']}")
                        elif content.get("toolUse"):
                            print(f"toolUseId: {content['toolUse']['toolUseId']}")
                            print(f"name: {content['toolUse']['name']}")
                            print(f"input: {content['toolUse']['input']}")
                    token_usage = response['usage']
                    print(f"Input tokens:  {token_usage['inputTokens']}")
                    print(f"Output tokens:  {token_usage['outputTokens']}")
                    print(f"Total tokens:  {token_usage['totalTokens']}")
                    print(f"Stop reason: {response['stopReason']}")
                    output["token_usage"] = {
                        "inputTokens": token_usage['inputTokens'],
                        "outputTokens": token_usage['outputTokens'],
                        "totalTokens": token_usage['totalTokens']
                    }

            except ClientError as err:
                message = err.response['Error']['Message']
                print("A client error occurred: %s", message)
        else:

            try:
                tool_use = {}
                stream_response = response["stream"]

                for event in stream_response:
                                        
                    if 'messageStart' in event:
                        if verbose: print(f"\nRole: {event['messageStart']['role']}")
                        pass

                    elif 'contentBlockStart' in event:
                        tool = event['contentBlockStart']['start']['toolUse']
                        tool_use['toolUseId'] = tool['toolUseId']
                        tool_use['name'] = tool['name']

                    elif 'contentBlockDelta' in event:
                        delta = event['contentBlockDelta']['delta']
                        
                        if "reasoningContent" in delta:
                            if "text" in delta["reasoningContent"]:
                                reasoning_text = delta["reasoningContent"]["text"]
                                print("\033[92m" + reasoning_text + "\033[0m", end="")
                            else:
                                print("") 
                        if 'toolUse' in delta:
                            if 'input' not in tool_use:
                                tool_use['input'] = ''
                            tool_use['input'] += delta['toolUse']['input']
                        elif 'text' in delta:
                            output["text"] += delta['text']
                            callback.on_llm_new_token(delta['text'])
                        
                    elif 'contentBlockStop' in event:
                        if 'input' in tool_use:
                            tool_use['input'] = json.loads(tool_use['input'])
                            output["toolUse"] = {'toolUse': tool_use}
                            #message['content'].append({'toolUse': tool_use})
                            #tool_use = {}
                        #else:
                        #    message['content'].append({'text': text})
                        #    output = ''

                if verbose:
                    if 'messageStop' in event:
                        print(f"\nStop reason: {event['messageStop']['stopReason']}")

                    elif 'metadata' in event:
                        metadata = event['metadata']
                        if 'usage' in metadata:
                            print("\nToken usage")
                            print(f"Input tokens: {metadata['usage']['inputTokens']}")
                            print(
                                f"Output tokens: {metadata['usage']['outputTokens']}")
                            print(f"Total tokens: {metadata['usage']['totalTokens']}")
                            output["token_usage"] = {
                                "inputTokens": metadata['usage']['inputTokens'],
                                "outputTokens": metadata['usage']['outputTokens'],
                                "totalTokens": metadata['usage']['totalTokens']
                            }
                        if 'metrics' in event['metadata']:
                            print(
                                f"Latency: {metadata['metrics']['latencyMs']} milliseconds")
                            output["latency"] = metadata['metrics']['latencyMs']
            except ClientError as err:
                message = err.response['Error']['Message']
                print("A client error occurred: %s", message)

        
        #print (output)
        
        return output