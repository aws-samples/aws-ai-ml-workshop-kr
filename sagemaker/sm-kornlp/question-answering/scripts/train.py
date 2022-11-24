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
from pathlib import Path

from transformers import (
    BertForQuestionAnswering,
    Trainer, TrainingArguments, set_seed
)
from transformers import AutoModel, AutoTokenizer, BertTokenizer
from transformers.trainer_utils import get_last_checkpoint

def parser_args(train_notebook=False):
    parser = argparse.ArgumentParser()

    # Default Setting
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--disable_tqdm", type=bool, default=False)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--tokenizer_id", type=str, default='salti/bert-base-multilingual-cased-finetuned-squad')
    parser.add_argument("--model_id", type=str, default='salti/bert-base-multilingual-cased-finetuned-squad')
    
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


class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)
    

def read_squad(path):
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)

    return contexts, questions, answers


def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx-1:end_idx-1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
        elif context[start_idx-2:end_idx-2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters
    return answers


def add_token_positions(encodings, answers, tokenizer):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))
        # if None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length
            
        # If the start and end positions are greater than max_length, both must be changed to max_length.
        if start_positions[-1] is None or start_positions[-1] > tokenizer.model_max_length:
            start_positions[-1] = tokenizer.model_max_length
        
        if end_positions[-1] is None or end_positions[-1] > tokenizer.model_max_length:
            end_positions[-1] = tokenizer.model_max_length

    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})   
    return encodings


def main():

    is_sm_container = True    
    
    if os.environ.get('SM_CURRENT_HOST') is None:
        is_sm_container = False        
        train_dir = 'qna_train'
        valid_dir = 'qna_valid'
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
    train_contexts, train_questions, train_answers = read_squad(f'{args.train_dir}/KorQuAD_v1.0_train.json')
    val_contexts, val_questions, val_answers = read_squad(f'{args.valid_dir}/KorQuAD_v1.0_dev.json')
    
    # Add end position index
    train_answers = add_end_idx(train_answers, train_contexts)
    val_answers = add_end_idx(val_answers, val_contexts)
    
    # Download tokenizer    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)

    # Encoding
    train_encodings = tokenizer(
        train_contexts, 
        train_questions, 
        truncation=True, 
        max_length=args.max_length,
        stride=args.stride, 
        padding="max_length"
    )
    val_encodings = tokenizer(
        val_contexts, 
        val_questions, 
        truncation=True, 
        max_length=args.max_length,
        stride=args.stride, 
        padding="max_length"
    )

    # Add token position (Start potision, end position)
    train_encodings = add_token_positions(train_encodings, train_answers, tokenizer)
    val_encodings = add_token_positions(val_encodings, val_answers, tokenizer)

    # Set seed before initializing model
    set_seed(args.seed)
    
    # Initializat Dataset
    train_dataset = SquadDataset(train_encodings)
    val_dataset = SquadDataset(val_encodings)
    logger.info(f'num_train samples={len(train_dataset)}, num_valid samples={len(val_dataset)}')
                
    # Load pre-trained model
    model = BertForQuestionAnswering.from_pretrained(args.model_id)         

    # define training args
    training_args = TrainingArguments(
        output_dir=args.chkpt_dir,
        overwrite_output_dir=True if get_last_checkpoint(args.chkpt_dir) is not None else False,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size, 
        per_device_eval_batch_size=args.eval_batch_size, 
        warmup_steps=args.warmup_steps, 
        weight_decay=0.01,    
        logging_dir=f"{args.output_data_dir}/logs",
        logging_steps=args.logging_steps,
        learning_rate=float(args.learning_rate),
        save_total_limit=5,
        save_strategy="epoch",
        fp16=args.fp16,
        gradient_accumulation_steps=4,
        #evaluation_strategy="steps",
    )
    
    # create Trainer instance
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset            # evaluation dataset
    )

    # train model
    if get_last_checkpoint(args.chkpt_dir) is not None:
        logger.info("***** Continue Training *****")
        last_checkpoint = get_last_checkpoint(args.chkpt_dir)
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()

    # compute metrics (EM and F1)
    from evaluate import get_metrics_korquadv1, get_prediction
    em, f1 = get_metrics_korquadv1(args.valid_dir, tokenizer, model)
    logger.info(f"EM = {em}")
    logger.info(f"F1 = {f1}")
    
    # evaluate model
    outputs = trainer.predict(val_dataset)
    eval_results = outputs.metrics
    eval_results['f1'] = f1
    eval_results['em'] = em

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