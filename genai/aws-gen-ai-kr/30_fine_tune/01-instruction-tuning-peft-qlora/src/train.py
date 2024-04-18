import os
import sys
from typing import List
import argparse
import torch
import transformers
from datasets import load_dataset, load_from_disk
from transformers import BitsAndBytesConfig
from pathlib import Path
from huggingface_hub import snapshot_download

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast

def parse_args():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()

    # path
    parser.add_argument("--base_model", type=str, default="nlpai-lab/kullm-polyglot-12.8b-v2", help="Model id to use for training.")
    parser.add_argument("--cache_dir", type=str, default=os.environ["HF_DATASETS_CACHE"])
    parser.add_argument("--pretrained_model_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, default="../data/kkulm-v2", help="Path to dataset.")
    parser.add_argument("--output_dir", type=str, default="./lora-alpaca")
    parser.add_argument("--save_path", type=str, default="./model")
    parser.add_argument("--save_merged_model", type=bool, default=False)

    # add training hyperparameters
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size to use for training.")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate to use for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")

    # quantization parameters
    parser.add_argument("--quant_8bit", type=bool, default=False)
    parser.add_argument("--quant_4bit", type=bool, default=True)

    # lora hyperparams
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # llm hyperparams
    parser.add_argument("--group_by_length", type=bool, default=False)

    # wandb params
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--wandb_watch", type=str, default="") # options: false | gradients | all
    parser.add_argument("--wandb_log_model", type=str, default="") # options: false | true

    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)

    parser.add_argument(
        "--bf16",
        type=bool,
        default=True if torch.cuda.get_device_capability()[0] == 8 else False,
        help="Whether to use bf16.",
    )
    args = parser.parse_known_args()
    return args


def train(args):
    base_model = args.base_model
    cache_dir = args.cache_dir
    pretrained_model_path = args.pretrained_model_path
    data_path = args.data_path
    output_dir = args.output_dir
    save_path = args.save_path
    save_merged_model = args.save_merged_model
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    gradient_accumulation_steps = args.gradient_accumulation_steps
    lr_scheduler_type = args.lr_scheduler_type
    quant_8bit = args.quant_8bit
    quant_4bit = args.quant_4bit
    lora_r = args.lora_r
    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout
    group_by_length = args.group_by_length
    wandb_project = args.wandb_project
    wandb_run_name = args.wandb_run_name
    wandb_watch = args.wandb_watch
    wandb_log_model = args.wandb_log_model
    save_steps = args.save_steps
    eval_steps = args.eval_steps
    bf16 = args.bf16

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"cache_dir: {cache_dir}\n"
            f"pretrained_model_path: {pretrained_model_path}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"save_path: {save_path}\n"
            f"save_merged_model: {save_merged_model}\n"
            f"batch_size: {batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"gradient_accumulation_steps: {gradient_accumulation_steps}\n"
            f"lr_scheduler_type: {lr_scheduler_type}\n"
            f"quant_8bit: {quant_8bit}\n"
            f"quant_4bit: {quant_4bit}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"save_steps: {save_steps}\n"
            f"eval_steps: {eval_steps}\n"
        )
    assert base_model, "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    os.makedirs(output_dir, exist_ok=True)
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(f"world_size: {world_size}")

    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        print("Activated Distributed Data Parallel.")
        #gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or ("WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0)

    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    if quant_4bit:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        nf4_config = None

    tokenizer = GPTNeoXTokenizerFast.from_pretrained(pretrained_model_path)
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    model = GPTNeoXForCausalLM.from_pretrained(
        #base_model,
        pretrained_model_path,
        load_in_8bit=True if quant_8bit else False,
        torch_dtype=torch.float16,
        device_map=device_map,
        cache_dir=cache_dir,
        quantization_config=nf4_config,
    )

    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["query_key_value", "xxx"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    data = load_from_disk(data_path)
    # if data_path.endswith(".json") or data_path.endswith(".jsonl"):
    #     data = load_dataset("json", data_files=data_path)
    # else:
    #     data = load_dataset(data_path)

    # Check if checkpoints exists
    if len(os.listdir(output_dir)) > 0:
        last_checkpoint = transformers.trainer_utils.get_last_checkpoint(output_dir)

        # Check the available weights and load them
        checkpoint_name = os.path.join(last_checkpoint, "pytorch_model.bin")  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                last_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit

        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    # if val_set_size > 0:
    #     train_val = data["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=42)
    #     train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    #     val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    # else:
    #     train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
    #     val_data = None
    val_set_size = 0
    train_data = data
    val_data = None

    # if not ddp and torch.cuda.device_count() > 1:
    #     # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    #     model.is_parallelizable = True
    #     model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=bf16,  # Use BF16 if available
            logging_steps=1,
            optim="paged_adamw_8bit",
            lr_scheduler_type=lr_scheduler_type,
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            save_steps=save_steps,
            eval_steps=eval_steps if val_set_size > 0 else None,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else "none",
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    ## PEFT 구 버전에서 443byte로 저장되는 문제가 있음. 최신 버전에서는 해결되었지만, 일부 훈련 코드에서 아래 코드가 들어간 경우 여전히 443bytes로 저장되므로 주의 필요
    ## Reference: https://github.com/huggingface/peft/issues/286
    # old_state_dict = model.state_dict
    # model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
    #     model, type(model)
    # )

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()

    # Save Model
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        if save_merged_model:
            print(f'Save LoRA model: {output_dir}')
            trainer.model.save_pretrained(output_dir)

            # Saving merged model
            print(f'Save merged model: {save_path}')
            from peft import AutoPeftModelForCausalLM
            os.makedirs(save_path, exist_ok=True)
            model = AutoPeftModelForCausalLM.from_pretrained(output_dir, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(save_path, safe_serialization=True)
        else:
            print(f'Save LoRA model: {save_path}')
            trainer.model.save_pretrained(save_path)

        tokenizer.save_pretrained(save_path)

        # clear memory
    del model
    del trainer

if __name__ == "__main__":
    args, _ = parse_args()
    print(args)
    train(args)