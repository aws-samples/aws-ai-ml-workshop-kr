from djl_python import Input, Output
import os
import deepspeed
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from transformers import GPTNeoXLayer

predictor = None


def get_model(properties):
    
    model_name = "EleutherAI/polyglot-ko-5.8b"
    tensor_parallel = properties["tensor_parallel_degree"]
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model.eval()
    model = deepspeed.init_inference(
        model,
        mp_size=tensor_parallel,
        dtype=model.dtype,
        replace_method="auto",
        #replace_with_kernel_inject=True,
        injection_policy={
            GPTNeoXLayer:('attention.dense', 'mlp.dense_4h_to_h')
        }
    )
    generator = pipeline(
        task="text-generation", model=model, tokenizer=tokenizer, device=local_rank
    )
    # https://huggingface.co/docs/hub/models-tasks
    return generator
    
def handle(inputs: Input) -> None:
    
    global predictor
    if not predictor:
        predictor = get_model(inputs.get_properties())

    if inputs.is_empty():
        # Model server makes an empty call to warmup the model on startup
        print ("is_empty")
        return None

    data = inputs.get_as_json() #inputs.get_as_string()
    
    print ("data:",  data)
    
    input_prompt, params = data["prompt"], data["params"]
    
    print ("input_prompt", input_prompt)
    print ("params", params)
    
    result = predictor(
        input_prompt,
        **params
    )
    
    print ("result:", result)

    return Output().add_as_json(result) #Output().add(result)
