import os
import sys
import json
import logging
import argparse
import torch
import gzip
import csv
import math
import urllib
from torch import nn
import numpy as np
import pandas as pd
from tqdm import tqdm

from datetime import datetime
from datasets import load_dataset
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
from transformers.trainer_utils import get_last_checkpoint


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[LoggingHandler()]
)
logger = logging.getLogger(__name__)

def parser_args(train_notebook=False):
    parser = argparse.ArgumentParser()

    # Default Setting
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--disable_tqdm", type=bool, default=False)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--tokenizer_id", type=str, default='sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
    parser.add_argument("--model_id", type=str, default='sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
    
    # SageMaker Container environment
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--train_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--valid_dir", type=str, default=os.environ["SM_CHANNEL_VALID"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])    
    parser.add_argument('--chkpt_dir', type=str, default='/opt/ml/checkpoints')     

    if train_notebook:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    return args
            

def main():

    is_sm_container = True    
    
    if os.environ.get('SM_CURRENT_HOST') is None:
        is_sm_container = False        
        train_dir = 'train'
        valid_dir = 'valid'
        test_dir = 'test'
        model_dir = 'model'
        output_data_dir = 'data'
        src_dir = '/'.join(os.getcwd().split('/')[:-1])
        #src_dir = os.getcwd()
        os.environ['SM_MODEL_DIR'] = f'{src_dir}/{model_dir}'
        os.environ['SM_OUTPUT_DATA_DIR'] = f'{src_dir}/{output_data_dir}'
        os.environ['SM_NUM_GPUS'] = str(1)
        os.environ['SM_CHANNEL_TRAIN'] = f'{src_dir}/{train_dir}'
        os.environ['SM_CHANNEL_VALID'] = f'{src_dir}/{valid_dir}'
        os.environ['SM_CHANNEL_TEST'] = f'{src_dir}/{test_dir}'
        
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
       
    ################################################
    # Load KLUS-STS Datasets
    ################################################    
    logger.info("Read KLUE-STS train/dev dataset")
    datasets = load_dataset("klue", "sts")

    train_samples = []
    dev_samples = []

    for phase in ["train", "validation"]:
        examples = datasets[phase]

        for example in examples:
            score = float(example["labels"]["label"]) / 5.0  # 0.0 ~ 1.0 스케일로 유사도 정규화
            inp_example = InputExample(texts=[example["sentence1"], example["sentence2"]], label=score)

            if phase == "validation":
                dev_samples.append(inp_example)
            else:
                train_samples.append(inp_example)    

    ################################################
    # Load KorSTS Datasets
    ################################################       
    logger.info("Read KorSTS train dataset")

    with open(f'{args.train_dir}/sts-train.tsv', 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if row["sentence1"] and row["sentence2"]:          
                score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
                inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)
                train_samples.append(inp_example)

    logging.info("Read KorSTS dev dataset")            
    with open(f'{args.valid_dir}/sts-dev.tsv', 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if row["sentence1"] and row["sentence2"]:        
                score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
                inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)
                dev_samples.append(inp_example)
           
    ################################################
    # Training preparation
    ################################################     
    model_name = 'sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens'

    train_batch_size = args.train_batch_size
    num_epochs = args.epochs
    model_save_path = f'{args.model_dir}/training_sts_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger.info(model_save_path)

    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    word_embedding_model = models.Transformer(model_name)
    
    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    train_dataset = SentencesDataset(train_samples, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)

    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')

    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) # 10% of train data for warm-up
    logger.info("Warmup-steps: {}".format(warmup_steps))
    
    train_dataset = SentencesDataset(train_samples, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)

    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')

    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) # 10% of train data for warm-up
    logger.info("Warmup-steps: {}".format(warmup_steps))
    
    ################################################
    # Training
    ################################################     
    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=num_epochs,
        evaluation_steps=int(len(train_dataloader)*0.5),
        warmup_steps=warmup_steps,
        output_path=model_save_path,
        use_amp=True,
        show_progress_bar=False
    )    
    
    ################################################
    # Evaluation
    ################################################      
    test_samples = []
    logger.info("Read KorSTS test dataset")            
    with open(f'{args.test_dir}/sts-test.tsv', 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if row["sentence1"] and row["sentence2"]:        
                score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
                inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)
                test_samples.append(inp_example)            

    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
    test_evaluator(model, output_path=model_save_path)
    
    
    ################################################
    # Chatbot data embedding
    ################################################  
    chatbot_df = pd.read_csv(f'{args.train_dir}/chatbot-train.csv')
    chatbot_data = chatbot_df['A'].tolist()
    
    chatbot_emb = model.encode(chatbot_data, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    logger.info(f"Chatbot Embeddings computed: shape={chatbot_emb.shape}")
    np.save(f'{args.model_dir}/chatbot_emb.npy', chatbot_emb)
    
    ################################################
    # News data embedding
    ################################################  
    news_data = []
    f = open(f'{args.train_dir}/KCCq28_Korean_sentences_EUCKR_v2.txt', 'rt', encoding='cp949')
    lines = f.readlines()
    for line in lines:
        line = line.strip()  # 줄 끝의 줄 바꿈 문자를 제거한다.
        news_data.append(line)
    f.close()
    #news_data = news_data[:10000]
    
    if n_gpus == 1:
        news_emb = model.encode(news_data, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    else:  
        logger.info("Start the multi-process pool on all available CUDA devices.")
        pool = model.start_multi_process_pool()
        news_emb = model.encode_multi_process(news_data, pool, batch_size=64)
        model.stop_multi_process_pool(pool)        
    
    logger.info(f"News Embeddings computed: shape={news_emb.shape}")
    np.save(f'{args.model_dir}/news_emb.npy', news_emb)    
    
 
    
    # if n_gpus == 1:
    #     chatbot_emb = model.encode(chatbot_data, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    # else:  
    #     # Start the multi-process pool on all available CUDA devices
    #     pool = model.start_multi_process_pool()
    #     chatbot_emb = model.encode_multi_process(data, pool, batch_size=64)
    #     model.stop_multi_process_pool(pool)        

        
        
    # Saves the model to s3 uses os.environ["SM_MODEL_DIR"] to make sure checkpointing works               
    #tokenizer.save_pretrained(args.model_dir)                
    #trainer.save_model(args.model_dir)



if __name__ == "__main__":
    main()    