from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftConfig, PeftModel
import torch


def model_fn(model_dir):
    print("************************** model_fn Start **************************")
    # load model and processor from model_dir
    peft_config = PeftConfig.from_pretrained(model_dir)
    print("************************** model_fn Start 1 **************************")
    model = AutoModelForSeq2SeqLM.from_pretrained(peft_config.base_model_name_or_path, device_map="auto", load_in_8bit=True)
    print("************************** model_fn Start 2 **************************")
    model = PeftModel.from_pretrained(model, 
                                      model_dir,         
                                      device_map={'': 0})
    print("************************** model_fn Start 4 **************************")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    print("************************** model_fn End **************************")
    return model, tokenizer

def input_fn(request_body, request_content_type):
    import numpy as np
    from io import BytesIO

    print("Content type: ", request_content_type)
    if request_content_type == "application/x-npy":        
        stream = BytesIO(request_body)
        data = np.load(stream, allow_pickle=True).tolist()
        return data
    else:
        raise ValueError(
            "Content type {} is not supported.".format(request_content_type)
        )


def predict_fn(data, model_and_tokenizer):
    print("************************** predict_fn Start **************************")
    # unpack model and tokenizer
    model, tokenizer = model_and_tokenizer
    
    print("************************** predict_fn model tokenizer **************************")
    print(f"data : {data}")
    # process input
    inputs = data.pop("inputs", data)
    parameters = data.pop("parameters", None)
    print(f"inputs : {inputs}")
    print(f"parameters : {parameters}")
    # preprocess
    input_ids = tokenizer(inputs, return_tensors="pt").input_ids.to(model.device)

    # pass inputs with all kwargs in data
    if parameters is not None:
        outputs = model.generate(input_ids=input_ids, **parameters)
    else:
        outputs = model.generate(input_ids=input_ids)

    # postprocess the prediction
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return [{"generated_text": prediction}]

