import os
import sys
import torch
import transformers
import logging
import argparse
import time, datetime
from types import SimpleNamespace
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from transformers import get_scheduler
from transformers import BertForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk, load_dataset
import evaluate
from pprint import pformat
from tqdm.auto import tqdm, trange
from helper import TqdmLoggingHandler

# Distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
os.environ['TOKENIZERS_PARALLELISM'] = "True"
        
import deepspeed
    
def setup():
    '''
    SageMaker에서 DeepSpeed를 사용하려면, `deepspeed` 커맨드가 아닌 `mpirun` 커맨드를 사용해야 합니다. `mpirun`에 대한 세부 파라미터는 SageMaker Estimator 호출 시 `distribution = {"mpi": mpi_options}`로 설정하시면 되며, `deepspeed.init_distributed(...)`는 호출할 필요가 없습니다.
    '''


    if 'WORLD_SIZE' in os.environ:
        # Environment variables set by torch.distributed.launch or torchrun
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        print("## os.environ has WORLD_SIZE")
    elif 'OMPI_COMM_WORLD_SIZE' in os.environ:
        # Environment variables set by mpirun 
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        print("## os.environ has OMPI_COMM_WORLD_SIZE")
    else:
        print("## Can't find the evironment variables for local rank")            
        sys.exit("Can't find the evironment variables for local rank")
    
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)    

    # SageMaker Training does not need to call deepspeed.init_distributed()
    if not 'SM_CHANNEL' in os.environ:
        deepspeed.init_distributed("nccl")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if rank == 0 else logging.WARNING,
        handlers=[TqdmLoggingHandler()])
    logging.info(f"Initialized the distributed environment. world_size={world_size}, rank={rank}, local_rank={local_rank}")
            
    config = SimpleNamespace()
    config.world_size = world_size
    config.rank = rank
    config.local_rank = local_rank
    
    deepspeed_config = {
        "train_micro_batch_size_per_gpu": 32,
        "gradient_accumulation": 1,
        "steps_per_print": 10,        
        "bf16": {
            "enabled": True,
        },
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True,
                "fast_init": True
            },
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 3e-05,
                "betas": [
                    0.9,
                    0.999
                ],
                "eps": 1e-8
            }
        },
    }    
    return config, deepspeed_config

def cleanup():
    dist.destroy_process_group()

def parser_args(train_notebook=False):
    parser = argparse.ArgumentParser()

    # Default Setting
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--disable_tqdm", type=bool, default=True)
    parser.add_argument("--log_interval", type=int, default=10)    
    parser.add_argument("--model_id", type=str, default='bert-base-multilingual-cased')
    
    # SageMaker Container environment
    parser.add_argument("--train_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--eval_dir", type=str, default=os.environ["SM_CHANNEL_EVAL"])
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument('--chkpt_dir', type=str, default='/opt/ml/checkpoints') 
    
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    parser = deepspeed.add_config_arguments(parser)    
    
    if train_notebook:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    return args

def main(args, deepspeed_config):
    
    torch.manual_seed(args.seed)
    #device = torch.device("cuda", args.local_rank)

    # load datasets
    train_dataset = load_from_disk(args.train_dir)
    eval_dataset = load_from_disk(args.eval_dir)
    
#     train_num_samples = 5000
#     eval_num_samples = 1000
#     train_dataset = train_dataset.shuffle(seed=42).select(range(train_num_samples))
#     eval_dataset = eval_dataset.shuffle(seed=42).select(range(eval_num_samples))
    
    logging.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logging.info(f" loaded test_dataset length is: {len(eval_dataset)}")
    
    # 미니배치가 겹치지 않게 함
    train_sampler = DistributedSampler(train_dataset)
    eval_sampler = DistributedSampler(eval_dataset)
     
    train_loader = DataLoader(
        dataset=train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, 
        num_workers=0, shuffle=False
    )    
    eval_loader = DataLoader(
        dataset=eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, 
        num_workers=0, shuffle=False
    )

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_data_dir, exist_ok=True)    
    
    model = BertForSequenceClassification.from_pretrained(args.model_id, num_labels=2)
    model, optimizer, _, _ = deepspeed.initialize(
        args=args, 
        model=model,
        config_params=deepspeed_config,
        model_parameters=model.parameters()
    )

    num_training_steps = args.num_epochs * len(train_loader)
    args.num_training_steps = num_training_steps
    
    logging.info(f"num_training_steps: {num_training_steps}")
    # lr_scheduler = get_scheduler(
    #     name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    # )
    for epoch in range(1, args.num_epochs+1):
        if args.rank == 0:
            logging.info(f"==== Epoch {epoch} start ====")
        train_sampler.set_epoch(epoch)
        eval_sampler.set_epoch(epoch)
        
        train_model(args, model, train_loader, eval_loader, optimizer, epoch)
        eval_model(model, eval_loader)

    if args.model_dir and args.rank == 0:
        logging.info('==== Save Model ====')
        torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pt"))
            
            
def train_model(args, model, train_loader, eval_loader, optimizer, epoch):
    model.train()

    if args.rank == 0:
        epoch_pbar = tqdm(total=len(train_loader), colour="blue", leave=True, desc=f"Training epoch {epoch}")    
        
    for batch_idx, batch in enumerate(train_loader):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        optimizer.zero_grad()
        
        outputs = model(**batch)
        loss = outputs.loss        
        model.backward(loss)
        model.step()
        
        if args.rank == 0:
            epoch_pbar.update(1)
        
        if batch_idx % args.log_interval == 0 and args.rank == 0:
            logging.info(f"Train loss: {loss.item()}")
            
    if args.rank == 0:
        epoch_pbar.close()
        

def eval_model(model, eval_loader):
    model.eval()
    metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            labels = batch['labels'].to(model.device)
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            metrics.add_batch(predictions=preds, references=batch["labels"])

    logging.info(f"Eval. loss: {loss.item()}")          
    logging.info(pformat(metrics.compute()))

if __name__ == "__main__":
    
    is_sm_container = True    
    if os.environ.get('SM_CURRENT_HOST') is None:
        is_sm_container = False        
        train_dir = 'train'
        eval_dir = 'eval'
        model_dir = 'model'
        output_data_dir = 'output_data'
        src_dir = '/'.join(os.getcwd().split('/')[:-1])
        #src_dir = os.getcwd()
        os.environ['SM_MODEL_DIR'] = f'{src_dir}/{model_dir}'
        os.environ['SM_OUTPUT_DATA_DIR'] = f'{src_dir}/{output_data_dir}'
        os.environ['SM_CHANNEL_TRAIN'] = f'{src_dir}/{train_dir}'
        os.environ['SM_CHANNEL_EVAL'] = f'{src_dir}/{eval_dir}'

    args = parser_args()
    config, deepspeed_config = setup() 
    
    args.world_size = config.world_size
    args.rank = config.rank
    #args.local_rank = local_rank

    start = time.time()
    main(args, deepspeed_config)     
    secs = time.time() - start
    result = datetime.timedelta(seconds=secs)
    if config.rank == 0:
        logging.info(f"Elapsed time: {result}")
    cleanup()