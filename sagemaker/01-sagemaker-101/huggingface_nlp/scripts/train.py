"""Transformer-based text classification on SageMaker with Hugging Face"""

# Python Built-Ins:
import argparse
import logging
import os
import sys
from typing import List, Optional

# External Dependencies:
import datasets
#from datasets import disable_progress_bar as disable_datasets_progress_bar
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Set up logging:
logging.basicConfig(
    level=logging.getLevelName("INFO"),
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
datasets.disable_progress_bar()  # Too noisy on conventional log streams

# Factoring your code out into smaller helper functions can help with debugging:


def parse_args():
    """Parse hyperparameters and data args from CLI arguments and environment variables"""
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--class_names", type=lambda s: s.split(","), required=True)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_max_steps", type=int, default=-1)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--fp16", type=int, default=1)

    # Data, model, and output folders are set by combination of CLI args and env vars:
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    # parser.add_argument("--n_gpus", type=int, default=os.environ.get("SM_NUM_GPUS"))

    args, _ = parser.parse_known_args()
    return args


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="micro")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def get_model(model_id: str, class_names: List[str]) -> (
    AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
):
    """Set up tokenizer, model, data_collator from job parameters"""
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=len(class_names)
    )
    model.config.label2id = {name: ix for ix, name in enumerate(class_names)}
    model.config.id2label = {ix: name for ix, name in enumerate(class_names)}

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    return tokenizer, model, data_collator


def load_datasets(tokenizer: AutoTokenizer, train_dir: str, test_dir: Optional[str] = None) -> (
    datasets.Dataset, Optional[datasets.Dataset]
):
    """Load and pre-process training (+ validation?) dataset(s)"""

    def preprocess(batch):
        """Tokenize and pre-process raw examples for training/validation"""
        result = tokenizer(batch["title"], truncation=True)
        result["label"] = batch["category"]
        return result


    raw_train_dataset = datasets.load_dataset(
        "csv",
        data_files=[os.path.join(train_dir, f) for f in os.listdir(train_dir)],
        column_names=["category", "title", "content"],
        split=datasets.Split.ALL,
    )
    train_dataset = raw_train_dataset.map(
        preprocess, batched=True, batch_size=1000, remove_columns=raw_train_dataset.column_names
    )
    logger.info(f"Loaded train_dataset length is: {len(train_dataset)}")
    if test_dir:
        # test channel is optional:
        raw_test_dataset = datasets.load_dataset(
            "csv",
            data_files=[os.path.join(test_dir, f) for f in os.listdir(test_dir)],
            column_names=["category", "title", "content"],
            split=datasets.Split.ALL,
        )
        test_dataset = raw_test_dataset.map(
            preprocess, batched=True, batch_size=1000, remove_columns=raw_test_dataset.column_names
        )
        logger.info(f"Loaded test_dataset length is: {len(test_dataset)}")
    else:
        test_dataset = None
        logger.info("No test_dataset provided")
    return train_dataset, test_dataset


# Only run this main block if running as a script (e.g. in training), not when imported as a module
# (which would be the case if used at inference):
if __name__ == "__main__":
    # Load job parameters:
    args = parse_args()
    training_args = TrainingArguments(
        max_steps=args.train_max_steps,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        fp16=bool(args.fp16),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        disable_tqdm=True,  # Interactive progress bars too noisy on conventional log streams
        # You could save checkpoints & logs under args.output_data_dir to upload them, but it
        # increases job run time by a few minutes:
        output_dir="/tmp/transformers/checkpoints",
        logging_dir="/tmp/transformers/logs",
    )

    # Load tokenizer/model/collator:
    tokenizer, model, collator = get_model(model_id=args.model_id, class_names=args.class_names)

    # Load and pre-process the dataset:
    train_dataset, test_dataset = load_datasets(
        tokenizer=tokenizer,
        train_dir=args.train,
        test_dir=args.test,
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # Train the model
    trainer.train()

    # Save the model output
    trainer.save_model(args.model_dir)

    # Evaluate the final model and save a report, if test dataset provided:
    if test_dataset:
        eval_result = trainer.evaluate(eval_dataset=test_dataset)
        # The 'output' folder will also (separately from model) get uploaded to S3 by SageMaker:
        if args.output_data_dir:
            os.makedirs(args.output_data_dir, exist_ok=True)
            with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
                print("***** Eval results *****")
                for key, value in sorted(eval_result.items()):
                    writer.write(f"{key} = {value}\n")
