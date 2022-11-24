import json
import sys
import logging
import torch
from torch import nn
from transformers import ElectraConfig
from transformers import ElectraModel, AutoTokenizer, ElectraTokenizer, ElectraForSequenceClassification

logging.basicConfig(
    level=logging.INFO, 
    format='[{%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(filename='tmp.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

max_seq_length = 128
classes = ['Neg', 'Pos']

tokenizer = AutoTokenizer.from_pretrained("daekeun-ml/koelectra-small-v3-nsmc")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_fn(model_path=None):
    ####
    # If you have your own trained model
    # Huggingface pre-trained model: 'monologg/koelectra-small-v3-discriminator'
    ####
    #config = ElectraConfig.from_json_file(f'{model_path}/config.json')
    #model = ElectraForSequenceClassification.from_pretrained(f'{model_path}/model.pth', config=config)
    
    # Download model from the Huggingface hub
    model = ElectraForSequenceClassification.from_pretrained('daekeun-ml/koelectra-small-v3-nsmc')   
    model.to(device)
    return model


def input_fn(input_data, content_type="application/jsonlines"): 
    data_str = input_data.decode("utf-8")
    jsonlines = data_str.split("\n")
    transformed_inputs = []

    for jsonline in jsonlines:
        text = json.loads(jsonline)["text"][0]
        logger.info("input text: {}".format(text))          
        encode_plus_token = tokenizer.encode_plus(
            text,
            max_length=max_seq_length,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )
        transformed_inputs.append(encode_plus_token)
        
    return transformed_inputs


def predict_fn(transformed_inputs, model):
    predicted_classes = []
    
    for data in transformed_inputs:
        data = data.to(device)
        output = model(**data)

        softmax_fn = nn.Softmax(dim=1)
        softmax_output = softmax_fn(output[0])
        _, prediction = torch.max(softmax_output, dim=1)

        predicted_class_idx = prediction.item()
        predicted_class = classes[predicted_class_idx]
        score = softmax_output[0][predicted_class_idx]
        logger.info("predicted_class: {}".format(predicted_class))

        prediction_dict = {}
        prediction_dict["predicted_label"] = predicted_class
        prediction_dict['score'] = score.cpu().detach().numpy().tolist()

        jsonline = json.dumps(prediction_dict)
        logger.info("jsonline: {}".format(jsonline))        
        predicted_classes.append(jsonline)

    predicted_classes_jsonlines = "\n".join(predicted_classes)
    return predicted_classes_jsonlines


def output_fn(outputs, accept="application/jsonlines"):
    return outputs, accept