# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Helper utilities for working with Amazon Bedrock from Python notebooks"""
# Python Built-Ins:
import os
from typing import Optional

# External Dependencies:
import boto3
from botocore.config import Config

class BedrockClient():

    @classmethod
    def get_list_fm_models(verbose=False):
        boto3_bedrock = boto3.client(service_name='bedrock')
        response = boto3_bedrock.list_foundation_models()
        models = response['modelSummaries']

        return [model['modelId'] for model in models]

        
    @classmethod
    def get_bedrock_client(region=None):
        session = boto3.Session()

        config = Config(
            retries={
                'max_attempts': 10,
                'mode': 'standard'
            }
        )

        return session.client(
            service_name='bedrock',
            region_name=region,
            config=config
        )