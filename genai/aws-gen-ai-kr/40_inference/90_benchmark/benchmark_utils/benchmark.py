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

# class CustomTokenizer:    
#     """A custom tokenizer class"""
#     TOKENS: int = 1000
#     WORDS: int = 750

#     def __init__(self, local_dir):
#         print(f"CustomTokenizer, based on HF transformers")
#         # Load the tokenizer from the local directory
#         dir_not_empty = any(Path(local_dir).iterdir())
#         if dir_not_empty is True:
#             logger.info("loading the provided tokenizer")
#             self.tokenizer = AutoTokenizer.from_pretrained(local_dir)
#         else:
#             logger.error(f"no tokenizer provided, the {local_dir} is empty, "
#                          f"using default tokenizer i.e. {self.WORDS} words = {self.TOKENS} tokens")
#             self.tokenizer = None

#     def count_tokens(self, text):
#         if self.tokenizer is not None:
#             return len(self.tokenizer.encode(text))
#         else:
#             return int(math.ceil((self.TOKENS/self.WORDS) * len(text.split())))
    
# _tokenizer = CustomTokenizer(globals.TOKENIZER)    

# def count_tokens(text: str) -> int:
#     global _tokenizer
#     return _tokenizer.count_tokens(text)



# class CustomTokenizer:    
#     """A custom tokenizer class"""
#     TOKENS: int = 1000
#     WORDS: int = 750

#     def __init__(self, bucket, prefix, local_dir):
#         print(f"CustomTokenizer, based on HF transformers")
#         # Check if the tokenizer files exist in s3 and if not, use the autotokenizer       
#         _download_from_s3(bucket, prefix, local_dir)
#         # Load the tokenizer from the local directory
#         dir_not_empty = any(Path(local_dir).iterdir())
#         if dir_not_empty is True:
#             logger.info("loading the provided tokenizer")
#             self.tokenizer = AutoTokenizer.from_pretrained(local_dir)
#         else:
#             logger.error(f"no tokenizer provided, the {local_dir} is empty, "
#                          f"using default tokenizer i.e. {self.WORDS} words = {self.TOKENS} tokens")
#             self.tokenizer = None

#     def count_tokens(self, text):
#         if self.tokenizer is not None:
#             return len(self.tokenizer.encode(text))
#         else:
#             return int(math.ceil((self.TOKENS/self.WORDS) * len(text.split())))
    
# _tokenizer = CustomTokenizer(globals.READ_BUCKET_NAME, globals.TOKENIZER_DIR_S3, globals.TOKENIZER)    
