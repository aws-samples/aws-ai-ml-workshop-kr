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
    BitsAndBytesConfig,
        set_seed,

)
from trl import setup_chat_format
from peft import LoraConfig


from trl import (
   SFTTrainer)
import mlflow


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

# Command Argument 로 받을 변수를 기술함.
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
    MLFLOW_TRACKING_ARN: str = field(
        default=None, metadata={"help": "ML Flow tracking servier ARN"}
    )
    mlflow_experiment_name: str = field(
        default=None, metadata={"help": "ML Flow experiment name"}
    )

def merge_and_save_model(model_id, adapter_dir, output_dir):
    '''
    output_dir: /opt/ml/model
    '''
    from peft import PeftModel

    print("## Trying to load a Peft model. It might take a while without feedback")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
    )
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    model = peft_model.merge_and_unload()

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    base_model.config.save_pretrained(output_dir)
    print(f"\n## Saving the newly created merged model to {output_dir}")
    os.system(f"find {output_dir}")


def training_function(script_args, training_args):
    ################
    # Dataset
    ################
        
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
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE
    
    # template dataset
    def template_dataset(examples):
        return{"text":  tokenizer.apply_chat_template(examples["messages"], tokenize=False)}
    
    train_dataset = train_dataset.map(template_dataset, remove_columns=["messages"])
    validation_dataset = validation_dataset.map(template_dataset, remove_columns=["messages"])    
    # test_dataset = test_dataset.map(template_dataset, remove_columns=["messages"])
    
    # print random sample
    with training_args.main_process_first(
        desc="Log a few random samples from the processed training set"
    ):
        for index in random.sample(range(len(train_dataset)), 1):
            print(train_dataset[index]["text"])

    # Model    
    torch_dtype = torch.bfloat16
    quant_storage_dtype = torch.bfloat16

    quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_id,
        quantization_config=quantization_config,
        attn_implementation="flash_attention_2",
        torch_dtype=quant_storage_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
    )
    
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

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
    )


    if trainer.accelerator.is_main_process:
        trainer.model.print_trainable_parameters()
        MLFLOW_TRACKING_ARN = script_args.MLFLOW_TRACKING_ARN
        print("## MLFLOW_TRACKING_ARN: ", MLFLOW_TRACKING_ARN)
        mlflow.set_tracking_uri(MLFLOW_TRACKING_ARN)
        mlflow_experiment_name = script_args.mlflow_experiment_name
        mlflow.set_experiment(mlflow_experiment_name)


    # MLflow 실험 시작
    with mlflow.start_run():
        # 하이퍼파라미터 로깅
        mlflow.log_params({
            "model_id": script_args.model_id,
            "learning_rate": training_args.learning_rate,
            "num_train_epochs": training_args.num_train_epochs,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            # 추가 하이퍼파라미터들...
        })

        ##########################
        # Train model
        ##########################
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint

        # 훈련 시작
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        # 훈련 결과 로깅
        log_extended_metrics(train_result, trainer)

        ## if running type is local mode, we intentionally do not save model
        is_local_mode = os.getenv('SM_CURRENT_INSTANCE_TYPE')
        if is_local_mode != 'local': # cloud mode
            ##########################
            # SAVE MODEL FOR SAGEMAKER
            ##########################
            if trainer.is_fsdp_enabled:
                trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
            trainer.save_model()

            # load and merge
            if training_args.distributed_state.is_main_process:
                merge_and_save_model(
                    script_args.model_id, training_args.output_dir, "/opt/ml/model"
                )
                tokenizer.save_pretrained("/opt/ml/model")

        else: # local mode
            print("## Because of local mode, we do not save model")

        # 선택적: 추가 아티팩트 로깅
        mlflow.log_artifact(training_args.output_dir, artifact_path="training_output")

    training_args.distributed_state.wait_for_everyone()  # wait for all processes to print


def log_extended_metrics(train_result, trainer):
    # 기본 메트릭 로깅
    metrics = {
        "train_loss": train_result.training_loss,
        "train_runtime": train_result.metrics["train_runtime"],
        "total_steps": trainer.state.global_step,
        "epoch": trainer.state.epoch,
    }

    # 추가 시간 관련 메트릭
    metrics.update({
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
        "train_steps_per_second": train_result.metrics.get("train_steps_per_second"),
    })

    # 평가 메트릭 (만약 평가를 수행했다면)
    if "eval_loss" in train_result.metrics:
        metrics.update({
            "eval_loss": train_result.metrics["eval_loss"],
            "perplexity": math.exp(eval_loss) if eval_loss > 0 else float('inf'),
        })

    # 학습률 관련 메트릭
    if hasattr(trainer, "lr_scheduler"):
        metrics["final_learning_rate"] = trainer.lr_scheduler.get_last_lr()[0]

    # GPU 메모리 사용량 (만약 가능하다면)
    if torch.cuda.is_available():
        metrics["max_gpu_memory_used"] = torch.cuda.max_memory_allocated() / 1024**3  # GB 단위

    # 모델 크기 관련 메트릭
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    metrics.update({
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_parameters_percent": trainable_params / total_params * 100,
    })

    # MLflow에 메트릭 로깅
    mlflow.log_metrics(metrics)

    # 선택적: 학습 곡선을 위한 히스토리 로깅
    if hasattr(trainer, "state") and hasattr(trainer.state, "log_history"):
        for log in trainer.state.log_history:
            step = log.get("step", 0)
            for key, value in log.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"history_{key}", value, step=step)
    
if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_and_config()    

    if training_args.local_rank == 0:
        import os
        print("## storage info: \n")
        os.system("df -h")    
        print("## SM_CURRENT_INSTANCE_TYPE: ", os.getenv('SM_CURRENT_INSTANCE_TYPE'))
        print("## script_args: \n", script_args)
        print("## training_args: \n", training_args)    
    
    # set use reentrant to False
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    # set seed
    set_seed(training_args.seed)
  
    # launch training
    training_function(script_args, training_args)


    if training_args.local_rank == 0:
        print("## storage info: \n")
        os.system("df -h")    

