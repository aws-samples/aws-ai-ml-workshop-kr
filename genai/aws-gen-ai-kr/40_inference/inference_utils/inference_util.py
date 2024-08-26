# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""General helper utilities the workshop notebooks"""
# Python Built-Ins:
from io import StringIO
import sys
import textwrap


def print_ww(*args, width: int = 100, **kwargs):
    """Like print(), but wraps output to `width` characters (default 100)"""
    buffer = StringIO()
    try:
        _stdout = sys.stdout
        sys.stdout = buffer
        print(*args, **kwargs)
        output = buffer.getvalue()
    finally:
        sys.stdout = _stdout
    for line in output.splitlines():
        print("\n".join(textwrap.wrap(line, width=width)))

import json 

def pretty_print_json(data, sort_keys=False):
    print(json.dumps(data, indent=4, sort_keys=sort_keys))

import boto3

def invoke_endpoint_sagemaker(endpoint_name, pay_load):

    # Set up the SageMaker runtime client
    sagemaker_runtime = boto3.client('sagemaker-runtime')
    # Set the endpoint name
    endpoint_name = endpoint_name


    # Invoke the endpoint
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(pay_load)
    )

    # Get the response from the endpoint
    result = response['Body'].read().decode('utf-8')

    return result
