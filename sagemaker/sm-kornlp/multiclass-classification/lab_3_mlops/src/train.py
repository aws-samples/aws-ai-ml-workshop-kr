import os
import json
import argparse
import logging
import sys
import numpy as np
import torch
from datasets import load_from_disk, load_metric
from transformers import (
    ElectraModel, ElectraTokenizer, ElectraForSequenceClassification, Trainer, TrainingArguments, set_seed
)
from transformers.trainer_utils import get_last_checkpoint
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def parser_args(train_notebook=False):
    parser = argparse.ArgumentParser()

    # Default Setting
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--disable_tqdm", type=bool, default=True)
    parser.add_argument("--fp16", type=bool, default=True)    
    parser.add_argument("--tokenizer_id", type=str, default='monologg/koelectra-small-v3-discriminator')
    parser.add_argument("--model_id", type=str, default='monologg/koelectra-small-v3-discriminator')
    
    # SageMaker Container environment
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--train_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument('--chkpt_dir', type=str, default='/opt/ml/checkpoints')     

    if train_notebook:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    return args


# compute metrics function for binary classification
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def main():

    is_sm_container = True    
    if os.environ.get('SM_CURRENT_HOST') is None:
        is_sm_container = False        
        train_dir = 'datasets/train'
        test_dir = 'datasets/test'
        model_dir = 'model'
        output_data_dir = 'data'
        src_dir = '/'.join(os.getcwd().split('/')[:-1])
        #src_dir = os.getcwd()
        os.environ['SM_MODEL_DIR'] = f'{src_dir}/{model_dir}'
        os.environ['SM_OUTPUT_DATA_DIR'] = f'{src_dir}/{output_data_dir}'
        os.environ['SM_NUM_GPUS'] = str(1)
        os.environ['SM_CHANNEL_TRAIN'] = f'{src_dir}/{train_dir}'
        os.environ['SM_CHANNEL_TEST'] = f'{src_dir}/{test_dir}'
        
    # Set up logging 
    logging.basicConfig(
        level=logging.INFO, 
        format='[{%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    args = parser_args()    
    n_gpus = torch.cuda.device_count()

    if os.getenv("SM_NUM_GPUS")==None:
        print("Explicitly specifying the number of GPUs.")
        os.environ["GPU_NUM_DEVICES"] = n_gpus
    else:
        os.environ["GPU_NUM_DEVICES"] = os.environ["SM_NUM_GPUS"]
    
    logger.info("***** Arguments *****")    
    logger.info(''.join(f'{k}={v}\n' for k, v in vars(args).items()))
    
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_data_dir, exist_ok=True)    

    # load datasets
    train_dataset = load_from_disk(args.train_dir)
    test_dataset = load_from_disk(args.test_dir)

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(test_dataset)}")
    logger.info(train_dataset[0])    

    # download tokenizer
    tokenizer = ElectraTokenizer.from_pretrained(args.tokenizer_id)

    # tokenizer helper function
    def tokenize(batch):
        return tokenizer(batch['document'], padding='max_length', truncation=True)

    # tokenize dataset
    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    # set format for pytorch
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # Prepare model labels - useful in inference API
    labels = train_dataset.features["label"].names
    num_labels = len(labels)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Set seed before initializing model
    set_seed(args.seed)
    
    # Download pytorch model
    model = ElectraForSequenceClassification.from_pretrained(
        args.model_id, num_labels=num_labels, label2id=label2id, id2label=id2label
    )

    # define training args
    training_args = TrainingArguments(
        output_dir=args.chkpt_dir,
        overwrite_output_dir=True if get_last_checkpoint(args.chkpt_dir) is not None else False,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        disable_tqdm=args.disable_tqdm,
        logging_dir=f"{args.output_data_dir}/logs",
        learning_rate=float(args.learning_rate),
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
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
    eval_result = trainer.evaluate(eval_dataset=test_dataset)

    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(args.model_dir, "evaluation.json"), "w") as writer:
        logger.info(f"***** Evaluation results *****")
        logger.info(eval_result)
        writer.write(json.dumps(eval_result))

#     with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
#         print("***** Evaluation results *****")
#         for key, value in sorted(eval_result.items()):
#             writer.write(f"{key} = {value}\n")
#             logger.info(f"{key} = {value}\n")

    # Saves the model to s3 uses os.environ["SM_MODEL_DIR"] to make sure checkpointing works
    trainer.save_model(args.model_dir)
    
    
def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()    