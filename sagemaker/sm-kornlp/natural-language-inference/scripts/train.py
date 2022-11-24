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
from IPython.display import display, HTML

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, set_seed
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset, load_metric, ClassLabel, Sequence

logging.basicConfig(
    level=logging.INFO, 
    format='[{%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
metric = load_metric("glue", "qnli")


def parser_args(train_notebook=False):
    parser = argparse.ArgumentParser()

    # Default Setting
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=str, default=3e-5)
    parser.add_argument("--disable_tqdm", type=bool, default=False)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--tokenizer_id", type=str, default='klue/roberta-base')
    parser.add_argument("--model_id", type=str, default='klue/roberta-base')
    
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


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


def main():

    is_sm_container = True    
    
    if os.environ.get('SM_CURRENT_HOST') is None:
        is_sm_container = False        
        train_dir = 'nli_train'
        valid_dir = 'nli_valid'
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
    #train_dict = torch.load(os.path.join(args.train_dir, 'train_features.pt'))
    #valid_dict = torch.load(os.path.join(args.valid_dir, 'valid_features.pt'))
    
    from datasets import load_from_disk
    train_dataset = load_from_disk(args.train_dir)
    valid_dataset = load_from_disk(args.valid_dir)

    # download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id, use_fast=True)

    # Set seed before initializing model
    set_seed(args.seed)
                
    #train_dataset = NERDataset(train_ids, train_attention_masks, train_labels)
    #valid_dataset = NERDataset(valid_ids, valid_attention_masks, valid_labels)
    logger.info(f'num_train samples={len(train_dataset)}, num_valid samples={len(valid_dataset)}')
               
    # Load pre-trained model
    model = AutoModelForSequenceClassification.from_pretrained(args.model_id, num_labels=3)    

    # define training args
    training_args = TrainingArguments(
        output_dir=args.chkpt_dir,
        overwrite_output_dir=True if get_last_checkpoint(args.chkpt_dir) is not None else False,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        learning_rate=float(args.learning_rate),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="accuracy",
    )
       
    # create Trainer instance
    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
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

    
def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()    