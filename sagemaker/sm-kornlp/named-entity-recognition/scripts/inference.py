import os
import sys
import json
import torch
import logging
import numpy as np
from transformers import BertTokenizerFast, BertConfig, BertForTokenClassification, pipeline
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    level=logging.INFO, 
    format='[{%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def model_fn(model_dir):
    tokenizer = BertTokenizerFast.from_pretrained(f'{model_dir}')

    with open(os.path.join(model_dir, 'tag2id.json'), 'r') as f:
        tag2id = json.loads(f.read())

    with open(os.path.join(model_dir, 'id2tag.json'), 'r') as f:
        id2tag = json.loads(f.read())    

    with open(os.path.join(model_dir, 'tag2entity.json'), 'r') as f:
        tag2entity = json.loads(f.read())

    model_file = 'pytorch_model.bin'
    model_id = 'bert-base-multilingual-cased'
    model = BertForTokenClassification.from_pretrained(model_id, num_labels=len(id2tag))
    
    tag2id = {k:int(v) for k,v in tag2id.items()}     
    id2tag = {int(k):v for k,v in id2tag.items()}  
    
    model.config.id2label = id2tag
    model.config.label2id = tag2id
    model.load_state_dict(torch.load(f'{model_dir}/{model_file}', map_location=torch.device(device)))
    model = model.eval()
    return (model, tokenizer)


def input_fn(input_data, content_type="application/jsonlines"): 
    
    data_str = input_data.decode("utf-8")
    jsonlines = data_str.split("\n")
    inputs = []

    for jsonline in jsonlines:
        text = json.loads(jsonline)["text"][0]
        logger.info("input text: {}".format(text)) 
        inputs.append(text)
        
    return inputs


def predict_fn(inputs, model_tuple): 
    model, tokenizer = model_tuple
    device_id = -1 if device.type == "cpu" else 0
    outputs = []
    
    for example in inputs:
        nlp = pipeline("ner", model=model.to(device), device=device_id, 
                       tokenizer=tokenizer, aggregation_strategy='average')
        output = nlp(example)
        logger.info("predicted_results: {}".format(output))
        print("predicted_results: {}".format(output))
        
        prediction_dict = {}
        prediction_dict["output"] = output        

        outputs.append(output)
        
    output = outputs[0]
    jsonlines = []

    for entity in output:
        for k, v in entity.items():
            if type(v) == np.float32:
                entity[k] = v.item()

        jsonline = json.dumps(entity)
        jsonlines.append(jsonline)

    jsonlines_output = '\n'.join(jsonlines)

    return jsonlines_output


def output_fn(outputs, accept="application/jsonlines"):
    return outputs, accept
