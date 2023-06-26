import os
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    default_data_collator,
)
from datasets import load_from_disk
import torch
from transformers import Trainer, TrainingArguments, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from peft import PeftConfig, PeftModel
import shutil


def parse_arge():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    # add model id and dataset path argument
    parser.add_argument(
        "--model_id",
        type=str,
        default="google/flan-t5-xl",
        help="Model id to use for training.",
    )
    parser.add_argument("--tokenized_train_path", type=str, default="", help="Path to tokenized train dataset.")
    parser.add_argument("--tokenized_test_path", type=str, default="", help="Path to tokenized test dataset.")
    parser.add_argument("--test_dataset_path", type=str, default="", help="Path to original testdataset.")
    # add training hyperparameters for epochs, batch size, learning rate, and seed
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for.")
    parser.add_argument("--eval_sample", type=int, default=0, help="Number of samples for rogue-metric")
    
    
    parser.add_argument("--output_dir", type=str,default="/tmp", help="Path to output.")
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size to use for training.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size to use for evaluation.",
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate to use for training.")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for training.")
    parser.add_argument(
        "--gradient_checkpointing",
        type=bool,
        default=True,
        help="Path to deepspeed config file.",
    )
    parser.add_argument(
        "--bf16",
        type=bool,
        default=True if torch.cuda.get_device_capability()[0] == 8 else False,
        help="Whether to use bf16.",
    )
    args = parser.parse_known_args()
    return args


def data_collator(tokenizer, model):
    from transformers import DataCollatorForSeq2Seq

    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )
    
    return data_collator


def create_peft_config(model):
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        prepare_model_for_int8_training,
    )

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q", "v"],
    )

    # prepare int-8 model for training
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def training_function(args):
    # set seed
    set_seed(args.seed)

    tokenized_train = load_from_disk(args.tokenized_train_path)
    tokenized_test = load_from_disk(args.tokenized_test_path)
    test_dataset = load_from_disk(args.test_dataset_path)
    
    # load model from the hub
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_id,
        use_cache=False if args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
        device_map="auto",
        load_in_8bit=True,
    )
    # create peft config
    model = create_peft_config(model)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Define training args
    # output_dir = "/tmp"
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        bf16=args.bf16,  # Use BF16 if available
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_accumulation_steps=2,
        # logging strategies
        logging_dir=f"{args.output_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="no",
        optim="adafactor",
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=data_collator(tokenizer, model),
    )

    # Start training
    trainer.train()

    # merge adapter weights with base model and save
    # save int 8 model
    trainer.model.save_pretrained(args.output_dir)
    
    # clear memory
    del model 
    del trainer
    # load PEFT model in fp16
    peft_config = PeftConfig.from_pretrained(args.output_dir)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        peft_config.base_model_name_or_path,
        load_in_8bit=True, 
        device_map={"":0}
    )
    model = PeftModel.from_pretrained(model, 
                                      args.output_dir,         
                                      device_map={'': 0})
    model.eval()

    print("#################### model ####################")
    evaluate(args, tokenized_test, test_dataset, model, tokenizer)
    
    
#     # Merge LoRA and base model and save
#     merged_model = model.merge_and_unload()
#     merged_model.save_pretrained(args.output_dir)
    
#     print("#################### merged_model ####################")
#     evaluate(args, tokenized_test, merged_model, tokenizer)

    # save tokenizer for easy inference
    tokenizer.save_pretrained(args.output_dir)
    
    # # copy inference script
    # os.makedirs("/opt/ml/model/code", exist_ok=True)
    # shutil.copyfile(
    #     os.path.join(os.path.dirname(__file__), "inference.py"),
    #     "/opt/ml/model/code/inference.py",
    # )
    # shutil.copyfile(
    #     os.path.join(os.path.dirname(__file__), "requirements.txt"),
    #     "/opt/ml/model/code/requirements.txt",
    # )


def evaluate(args, tokenized_test, test_dataset, model, tokenizer):
    from datasets import load_dataset
    from random import randrange

    # Load dataset from the hub and get a sample
    sample = test_dataset[randrange(len(test_dataset))]
    
    input_ids = tokenizer(sample["dialogue"], return_tensors="pt", truncation=True).input_ids.to("cuda")
    
    # with torch.inference_mode():
    outputs = model.generate(input_ids=input_ids, max_new_tokens=10, do_sample=True, top_p=0.9)
    print(f"input sentence: {sample['dialogue']}\n{'---'* 20}")
    print(f"summary:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]}")
    
    get_rogue_metric(args, tokenized_test, model, tokenizer)


def get_rogue_metric(args, tokenized_test, model, tokenizer):

    import evaluate
    import numpy as np
    from datasets import load_from_disk
    from random import randrange
    from tqdm import tqdm
    import pickle
    import gzip

    # Metric
    metric = evaluate.load("rouge")
    tokenized_test = tokenized_test.with_format("torch")

    # run predictions
    # this can take ~45 minutes
    if args.eval_sample > 0:
        cnt = args.eval_sample
    else:
        cnt = len(tokenized_test)
        
    predictions, references = [], []
    
    for sample in tqdm(tokenized_test):
        if cnt > 0:
            p,l = evaluate_peft_model(sample, model, tokenizer)
            predictions.append(p)
            references.append(l)
            cnt -= 1
    
    # compute metric
    rogue = metric.compute(predictions=predictions, references=references, use_stemmer=True)
    
    with gzip.open(args.output_dir + "/rogue.pickle", 'wb') as f:
        pickle.dump(rogue, f) 

    # print results
    print(f"Rogue1: {rogue['rouge1']* 100:2f}%")
    print(f"rouge2: {rogue['rouge2']* 100:2f}%")
    print(f"rougeL: {rogue['rougeL']* 100:2f}%")
    print(f"rougeLsum: {rogue['rougeLsum']* 100:2f}%")


def evaluate_peft_model(sample, model, tokenizer, max_target_length=50):
    import numpy as np
    
    # generate summary
    outputs = model.generate(input_ids=sample["input_ids"].unsqueeze(0).cuda(), do_sample=True, top_p=0.9, max_new_tokens=max_target_length)
    prediction = tokenizer.decode(outputs[0].detach().cpu().numpy(), skip_special_tokens=True)
    # decode eval sample
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(sample['labels'] != -100, sample['labels'], tokenizer.pad_token_id)
    labels = tokenizer.decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    return prediction, labels

def train_sagemaker(args):
    print(f"Train_sagemaker")
    if os.environ.get('SM_MODEL_DIR') is not None:
        args.tokenized_train_path = os.environ.get('SM_CHANNEL_TOKENIZED_TRAIN')
        args.tokenized_test_path = os.environ.get('SM_CHANNEL_TOKENIZED_TEST')
        args.test_dataset_path = os.environ.get('SM_CHANNEL_TEST')
        args.output_dir = os.environ.get('SM_MODEL_DIR')
    return args

def main():
    args, _ = parse_arge()
    args = train_sagemaker(args)
    training_function(args)


if __name__ == "__main__":
    main()
