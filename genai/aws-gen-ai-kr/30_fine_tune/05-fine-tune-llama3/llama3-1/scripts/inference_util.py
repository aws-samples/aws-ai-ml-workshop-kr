import boto3
import json
from io import StringIO
import sys
import textwrap

def print_json(data):
    print(json.dumps(data, indent = 4))


def create_messages_parameters(system_prompt, user_prompt, verbose=False):
  # Prompt to generate
  messages=[
      { "role": "system", "content": f"{system_prompt}" },
      { "role": "user", "content":  user_prompt}
    ]

  # Generation arguments
  parameters = {
      "model": "meta-llama-3-fine-tuned", # placeholder, needed
      "top_p": 0.6,
      "temperature": 0.0,
      "max_tokens": 512,
      "stop": ["<|eot_id|>"],
  }

  if verbose:
    print("messages:")
    print_json(messages)
    print("parameters:")
    print_json(parameters)

  return messages, parameters


def create_boto3_request_body(system_prompt, user_prompt):
    request_body = {
        "messages": [
            {"role": "system", "content": f"{system_prompt}"},
            {"role": "user", "content": f"{user_prompt}"},
        ],
        "model": "meta-llama-3-fine-tuned",
        "parameters": {"max_tokens":256,
                    "top_p": 0.9,
                    "temperature": 0.6,
                    "max_tokens": 512,
                    "stop": ["<|eot_id|>"]}
    }

    return request_body    


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


def invoke_endpoint_IC_sagemaker(endpoint_name, pay_load, inference_component):

    # Set up the SageMaker runtime client
    sagemaker_runtime = boto3.client('sagemaker-runtime')
    # Set the endpoint name
    endpoint_name = endpoint_name


    # Invoke the endpoint
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        InferenceComponentName = inference_component,
        ContentType='application/json',
        Body=json.dumps(pay_load)
    )
    # Get the response from the endpoint
    result = response['Body'].read().decode('utf-8')

    return result


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


from datasets import load_dataset 
from random import randint

def get_message_from_dataset(sample_dataset_json_file, verbose=False):
    # Load our test dataset
    full_test_dataset = load_dataset("json", data_files=sample_dataset_json_file, split="train")

    # Test on sample 
    rand_idx = randint(0, len(full_test_dataset)-1)
    rand_idx = 75
    
    messages = full_test_dataset[rand_idx]["messages"][:2]
    # messages = test_dataset[rand_idx]["text"][:2]
    if verbose:
        print("rand_idx: ", rand_idx)
        print("messages: \n", messages)

    return messages, full_test_dataset, rand_idx

def extract_system_user_prompt(messages, verbose=False):
    system_prompt = messages[0]
    user_prompt = messages[1]

    if verbose:
        print("system_prompt: \n", system_prompt['content'])
        print("user_prompt: \n", user_prompt['content'])

    return system_prompt['content'], user_prompt['content']

# system_prompt, user_prompt = extract_system_user_prompt(messages)    

import time
def run_inference(sm_endpoint_name, system_prompt,user_prompt, verbose=False ):
    request_body = create_boto3_request_body(system_prompt=system_prompt, user_prompt=user_prompt)
        
    s = time.perf_counter()

    # sm_endpoint_name = "llama3-endpoint-mnist-1719625657"
    response = invoke_endpoint_sagemaker(endpoint_name = sm_endpoint_name, 
                            pay_load = request_body)    

    elapsed_async = time.perf_counter() - s

    print(f"elapsed time: {round(elapsed_async,3)} second")
    parsed_data = json.loads(response)
    answer = parsed_data["choices"][0]["message"]["content"].strip()

    if verbose:
        print("request_body: \n", request_body)
        print("response body: \n", json.dumps(parsed_data, indent=4, ensure_ascii=False))

    return answer, request_body

def run_inference_IC(sm_endpoint_name, system_prompt,user_prompt, inference_component, verbose=False ):
    request_body = create_boto3_request_body(system_prompt=system_prompt, user_prompt=user_prompt)
        
    s = time.perf_counter()

    # sm_endpoint_name = "llama3-endpoint-mnist-1719625657"
    response = invoke_endpoint_IC_sagemaker(endpoint_name = sm_endpoint_name, 
                            pay_load = request_body, inference_component = inference_component)    

    elapsed_async = time.perf_counter() - s

    print(f"elapsed time: {round(elapsed_async,3)} second")
    parsed_data = json.loads(response)
    answer = parsed_data["choices"][0]["message"]["content"].strip()

    if verbose:
        print("request_body: \n", request_body)
        print("response body: \n", json.dumps(parsed_data, indent=4, ensure_ascii=False))

    return answer, request_body    

def generate_response(messages,sm_endpoint_name, full_test_dataset, rand_idx):
    system_prompt, user_prompt = extract_system_user_prompt(messages, verbose=False)    
    answer, request_body = run_inference(sm_endpoint_name, system_prompt,user_prompt, verbose=False )
    print(f"**Query:**\n{request_body}")
    # print(f"**Query:**\n{test_dataset[rand_idx]['text'][1]['content']}\n")
    # print(f"**Original Answer:**\n{test_dataset[rand_idx]['text'][2]['content']}\n")
    print(f"**Original Answer:**\n{full_test_dataset[rand_idx]['messages'][2]['content']}\n")

    print(f"**Generated Answer:**\n{answer}")
    
def generate_response_IC(messages,sm_endpoint_name, full_test_dataset, rand_idx, inference_component):
    system_prompt, user_prompt = extract_system_user_prompt(messages, verbose=False)    
    answer, request_body = run_inference_IC(sm_endpoint_name, system_prompt,user_prompt, inference_component, verbose=False )
    print(f"**Query:**\n{request_body}")
    # print(f"**Query:**\n{test_dataset[rand_idx]['text'][1]['content']}\n")
    # print(f"**Original Answer:**\n{test_dataset[rand_idx]['text'][2]['content']}\n")
    print(f"**Original Answer:**\n{full_test_dataset[rand_idx]['messages'][2]['content']}\n")

    print(f"**Generated Answer:**\n{answer}")

# answer = run_inference(sm_endpoint_name, system_prompt,user_prompt, verbose=True )
# answer = run_inference(sm_endpoint_name, system_prompt,user_prompt, verbose=False )        
