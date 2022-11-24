import os
import sys
import json
import logging
import argparse
import torch
from torch import nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer, BertTokenizerFast, BertConfig, BertForTokenClassification, 
    Trainer, TrainingArguments, set_seed
)
from transformers.trainer_utils import get_last_checkpoint


def parser_args(train_notebook=False):
    parser = argparse.ArgumentParser()

    # Default Setting
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--disable_tqdm", type=bool, default=False)
    parser.add_argument("--debug", type=bool, default=False)
    #parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--tokenizer_id", type=str, default='bert-base-multilingual-cased')
    #parser.add_argument("--model_id", type=str, default='distilbert-base-multilingual-cased')    
    parser.add_argument("--model_id", type=str, default='bert-base-multilingual-cased')
    
    # SageMaker Container environment
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--train_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--valid_dir", type=str, default=os.environ["SM_CHANNEL_VALID"])
    parser.add_argument('--chkpt_dir', type=str, default='/opt/ml/checkpoints')     

    if train_notebook:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    return args


def compute_metrics(p):
    logits = p.predictions
    labels = p.label_ids.ravel()
    preds = logits.argmax(-1).ravel()
    
    preds = preds[labels != -100]
    labels = labels[labels != -100]

    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)

    metrics = {
        'precision': prec,
        'recall': rec,
        'f1': f1,        
        'accuracy': acc
    }
    
    return metrics


class NERDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_masks, labels=None, max_len=128):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
        self.max_len = max_len
        
    def __getitem__(self, idx):
        item = {}
        item['input_ids'] = self.input_ids[idx]
        item['attention_mask'] = self.attention_masks[idx]
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

def main():

    is_sm_container = True    
    
    if os.environ.get('SM_CURRENT_HOST') is None:
        is_sm_container = False        
        train_dir = 'ner_train'
        valid_dir = 'ner_valid'
        model_dir = 'model'
        output_data_dir = 'data'
        src_dir = '/'.join(os.getcwd().split('/')[:-1])
        #src_dir = os.getcwd()
        os.environ['SM_MODEL_DIR'] = f'{src_dir}/{model_dir}'
        os.environ['SM_OUTPUT_DATA_DIR'] = f'{src_dir}/{output_data_dir}'
        os.environ['SM_NUM_GPUS'] = str(1)
        os.environ['SM_CHANNEL_TRAIN'] = f'{src_dir}/{train_dir}'
        os.environ['SM_CHANNEL_VALID'] = f'{src_dir}/{valid_dir}'
        
    # Set up logging 
    logging.basicConfig(
        level=logging.INFO, 
        format='[{%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    args = parser_args()    
    
    if os.environ.get('SM_CURRENT_HOST') is None:
        args.chkpt_dir = 'chkpt'
        
    n_gpus = torch.cuda.device_count()

    if os.getenv("SM_NUM_GPUS")==None:
        print("Explicitly specifying the number of GPUs.")
        os.environ["GPU_NUM_DEVICES"] = n_gpus
    else:
        os.environ["GPU_NUM_DEVICES"] = os.environ["SM_NUM_GPUS"]
    
    logger.info("***** Arguments *****")    
    logger.info(''.join(f'{k}={v}\n' for k, v in vars(args).items()))
    
    os.makedirs(args.chkpt_dir, exist_ok=True) 
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_data_dir, exist_ok=True)    
    
    # Load datasets    
    train_dict = torch.load(os.path.join(args.train_dir, 'train_features.pt'))
    valid_dict = torch.load(os.path.join(args.valid_dir, 'valid_features.pt'))
    
    with open(os.path.join(args.train_dir, 'tag2id.json'), 'r') as f:
        tag2id = json.loads(f.read())
        
    with open(os.path.join(args.train_dir, 'id2tag.json'), 'r') as f:
        id2tag = json.loads(f.read())    
        
    with open(os.path.join(args.train_dir, 'tag2entity.json'), 'r') as f:
        tag2entity = json.loads(f.read())     

    tag2id = {k:int(v) for k,v in tag2id.items()}     
    id2tag = {int(k):v for k,v in id2tag.items()}          

    train_ids, train_attention_masks, train_labels = train_dict['input_ids'], train_dict['attention_mask'], train_dict['labels']   
    valid_ids, valid_attention_masks, valid_labels = valid_dict['input_ids'], valid_dict['attention_mask'], valid_dict['labels']     
    
    # Debug
    if args.debug:
        num_debug_samples = 500
        train_ids = train_ids[:num_debug_samples, :]
        train_attention_masks = train_attention_masks[:num_debug_samples, :]
        train_labels = train_labels[:num_debug_samples, :]

        valid_ids = valid_ids[:num_debug_samples, :]
        valid_attention_masks = valid_attention_masks[:num_debug_samples, :]
        valid_labels = valid_labels[:num_debug_samples, :]

    logger.info(f"Loaded train_dataset length is: {len(train_dict['input_ids'])}")
    logger.info(f"Loaded test_dataset length is: {len(valid_dict['input_ids'])}")

    # download tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_id)

    # Set seed before initializing model
    set_seed(args.seed)
                
    train_dataset = NERDataset(train_ids, train_attention_masks, train_labels)
    valid_dataset = NERDataset(valid_ids, valid_attention_masks, valid_labels)
    logger.info(f'num_train samples={len(train_dataset)}, num_valid samples={len(valid_dataset)}')
                
    # Load pre-trained model
    model = BertForTokenClassification.from_pretrained(
        args.model_id, num_labels=len(tag2id), label2id=tag2id, id2label=id2tag
    )              
    #model.config.id2label = id2tag
    #model.config.label2id = tag2id
                
#     # Download pytorch model
#     model = ElectraForSequenceClassification.from_pretrained(
#         args.model_id, num_labels=num_labels, label2id=label2id, id2label=id2label
#     )

    # define training args
    training_args = TrainingArguments(
        output_dir=args.chkpt_dir,          # output directory
        overwrite_output_dir=True if get_last_checkpoint(args.chkpt_dir) is not None else False,
        num_train_epochs=args.epochs,              # total number of training epochs
        per_device_train_batch_size=args.train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.eval_batch_size,   # batch size for evaluation
        warmup_steps=args.warmup_steps,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir=f"{args.output_data_dir}/logs",            # directory for storing logs
        logging_steps=50,
        #eval_steps=100,
        learning_rate=float(args.learning_rate),
        #load_best_model_at_end=True,
        save_strategy="epoch",
        #evaluation_strategy="steps",
        metric_for_best_model="f1",
    )
                
    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset, 
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # train model
    if get_last_checkpoint(args.chkpt_dir) is not None:
        logger.info("***** Continue Training *****")
        last_checkpoint = get_last_checkpoint(args.chkpt_dir)
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()
                
    # evaluate model
    outputs = trainer.predict(valid_dataset)
    eval_results = outputs.metrics

    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        print(f"***** Evaluation results at {args.output_data_dir} *****")
        for key, value in sorted(eval_results.items()):
            writer.write(f"{key} = {value}\n")
            logger.info(f"{key} = {value}\n")

    # Saves the model to s3 uses os.environ["SM_MODEL_DIR"] to make sure checkpointing works               
    tokenizer.save_pretrained(args.model_dir)                
    trainer.save_model(args.model_dir)

    with open(os.path.join(args.model_dir, 'tag2id.json'), 'w') as f:
        json.dump(tag2id, f)    

    with open(os.path.join(args.model_dir, 'id2tag.json'), 'w') as f:
        json.dump(id2tag, f)

    with open(os.path.join(args.model_dir, 'tag2entity.json'), 'w') as f:
        json.dump(tag2entity, f)             

    
def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()    