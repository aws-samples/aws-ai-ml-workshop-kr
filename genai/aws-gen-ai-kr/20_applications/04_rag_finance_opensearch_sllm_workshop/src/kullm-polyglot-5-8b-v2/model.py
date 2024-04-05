from djl_python import Input, Output
import os
import deepspeed
import torch
import logging
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from transformers import GPTNeoXLayer

predictor = None

def get_model(properties):
    
    tp_degree = properties["tensor_parallel_degree"]
    model_location = properties["model_dir"]
    if "model_id" in properties:
        model_location = properties["model_id"]
    task = properties["task"]
    
    logging.info(f"Loading model in {model_location}")    
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    tokenizer = AutoTokenizer.from_pretrained(model_location)

    model = AutoModelForCausalLM.from_pretrained(
        model_location,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    
    model.requires_grad_(False)
    model.eval()
    
    ds_config = {
        "tensor_parallel": {"tp_size": tp_degree},
        "dtype": model.dtype,
        "injection_policy": {
            GPTNeoXLayer:('attention.dense', 'mlp.dense_4h_to_h')
        }
    }
    logging.info(f"Starting DeepSpeed init with TP={tp_degree}")        
    model = deepspeed.init_inference(model, ds_config)  
    
    generator = pipeline(
        task=task, model=model, tokenizer=tokenizer, device=local_rank
    )
    # https://huggingface.co/docs/hub/models-tasks
    return generator
    
def handle(inputs: Input) -> None:
    """
    inputs: Contains the configurations from serving.properties
    """    
    global predictor
    if not predictor:
        predictor = get_model(inputs.get_properties())

    if inputs.is_empty():
        # Model server makes an empty call to warmup the model on startup
        logging.info("is_empty")
        return None

    data = inputs.get_as_json() #inputs.get_as_string()
    logging.info("data:", data)
    
    input_prompt, params = data["inputs"], data["parameters"]
    result = predictor(input_prompt, **params)
    logging.info("result:", result)

    return Output().add_as_json(result) #Output().add(result)
