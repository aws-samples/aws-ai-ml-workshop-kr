##############################################
# 1. Loading python libs
##############################################
import logging
from dataclasses import dataclass, field
import os
import random
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments
from trl.commands.cli_utils import  TrlParser
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    # BitsAndBytesConfig,
    set_seed,

)
from trl import setup_chat_format
from peft import LoraConfig

# FSDP를 위한 import 추가
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch,
)


from trl import (
   SFTTrainer)

# Comment in if you want to use the Llama 3 instruct template but make sure to add modules_to_save
# LLAMA_3_CHAT_TEMPLATE="{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

# Anthropic/Vicuna like template without the need for special tokens
LLAMA_3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
            "{{ message['content'] }}"
        "{% elif message['role'] == 'user' %}"
            "{{ '\n\nHuman: ' + message['content'] +  eos_token }}"
        "{% elif message['role'] == 'assistant' %}"
            "{{ '\n\nAssistant: '  + message['content'] +  eos_token  }}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '\n\nAssistant: ' }}"
    "{% endif %}"
)


# ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=4 ./scripts/run_fsdp_qlora.py --config llama_3_70b_fsdp_qlora.yaml

@dataclass
class ScriptArguments:
    train_dataset_path: str = field(
        default=None,
        metadata={"help": "Path to the train dataset "},
    )
    validation_dataset_path: str = field(
        default=None,
        metadata={"help": "Path to the validation dataset"},
    )    

    model_id: str = field(
        default=None, metadata={"help": "Model ID to use for SFT training"}
    )
    max_seq_length: int = field(
        default=512, metadata={"help": "The maximum sequence length for SFT Trainer"}
    )


def training_function(script_args, training_args):
    ################
    # Dataset
    ################
    if training_args.process_index == 0:
        print("## Starting loading datasets")
    
    train_dataset = load_dataset(
        "json",
        data_files=os.path.join(script_args.train_dataset_path, "train_dataset.json"),
        split="train",
    )
    validation_dataset = load_dataset(
        "json",
        data_files=os.path.join(script_args.validation_dataset_path, "validation_dataset.json"),
        split="train",
    )    

    ################
    # Model & Tokenizer
    ################

    # Tokenizer        
    print("## Starting loading tokineizer")
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE
    
    # template dataset
    def template_dataset(examples):
        return{"text":  tokenizer.apply_chat_template(examples["messages"], tokenize=False)}
    
    if training_args.process_index == 0:
        print("## Starting transforming datasets")
        
    train_dataset = train_dataset.map(template_dataset, remove_columns=["messages"])
    validation_dataset = validation_dataset.map(template_dataset, remove_columns=["messages"])
    
    # print random sample
    with training_args.main_process_first(
        desc="Log a few random samples from the processed training set"
    ):
        for index in random.sample(range(len(train_dataset)), 2):
            print(train_dataset[index]["text"])

    # Model    
    torch_dtype = torch.bfloat16

    if training_args.process_index == 0:
        print("## Starting loading llama3 model")
    
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_id,
        attn_implementation="sdpa", # use sdpa, alternatively use "flash_attention_2"
        torch_dtype=torch_dtype,
        use_cache=False, 
        device_map=None,  # 추가
    )
    
    # FSDP를 위한 mixed precision 정책
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch_dtype,
        reduce_dtype=torch_dtype,
        buffer_dtype=torch_dtype,
    )

    # 모든 파라미터를 동일한 dtype으로 변환
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch_dtype)

    # FSDP 설정
    model.config.use_cache = False
    model = model.to(torch_dtype)  # 모델 전체를 bfloat16으로 변환            
    
    # FSDP 관련 설정 추가
    model.config.use_cache = False  # FSDP에서는 cache 사용하지 않음
    
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # 추가

    ################
    # PEFT
    ################

    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    peft_config = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        # modules_to_save = ["lm_head", "embed_tokens"] # add if you want to use the Llama 3 instruct template
    )


    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,        
        dataset_text_field="text",        
        peft_config=peft_config,
        max_seq_length=script_args.max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False,  # No need to add additional separator token
        },
        # optimizers=(None, None),  # optimizer를 None으로 설정하여 기본값 사용
    )
    if trainer.accelerator.is_main_process:
        trainer.model.print_trainable_parameters()



    ##########################
    # Train model
    ##########################
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        
    if training_args.process_index == 0:        
        print("## Starting training")        
        
        
    trainer.train(resume_from_checkpoint=checkpoint)

    ##########################
    # SAVE MODEL FOR SAGEMAKER
    ##########################
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    
    if training_args.process_index == 0:
        print("## Starting saving model")                
        
    trainer.save_model()
    
    if training_args.process_index == 0:
        print("## Training is finished")                    
    
if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_and_config()    

    if training_args.process_index == 0:
        print("## script_args: \n", script_args)
        print("## training_args: \n", training_args)    
    
    # set use reentrant to False
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    # set seed
    set_seed(training_args.seed)
    
    if training_args.process_index == 0:
        print("## Starting train function")
    # launch training
    training_function(script_args, training_args)
