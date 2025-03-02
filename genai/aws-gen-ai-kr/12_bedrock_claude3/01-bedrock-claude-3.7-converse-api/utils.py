# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shows how to send an image with the Converse API to Anthropic Claude 3 Sonnet (on demand).
"""

import logging
import boto3


def generate_reasoning_generation(bedrock_client, modelId, messages, additionalModelRequestFields):
    response = bedrock_client.converse(
        modelId= modelId,
        messages=messages,
        additionalModelRequestFields=additionalModelRequestFields
    )

    return response    


def generate_reasoning_stream_generation(bedrock_client, modelId, messages, additionalModelRequestFields):
    response = bedrock_client.converse_stream(
        modelId= modelId,
        messages=messages,
        additionalModelRequestFields=additionalModelRequestFields
    )


    reasoning = ''
    text = ''

    for chunk in response["stream"]:
        if "contentBlockDelta" in chunk:
            delta = chunk['contentBlockDelta']['delta']

            if "reasoningContent" in delta:
                reasoningContent = delta['reasoningContent']

                if 'text' in reasoningContent:
                    reasoning += reasoningContent['text']
            if 'text' in delta:
                text += delta['text']
                
    print("## reasoning: \n")
    print(reasoning)
    print("## \n\nresponse: \n")
    print(text)
