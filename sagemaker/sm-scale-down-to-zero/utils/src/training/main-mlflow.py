import os
import torch
import mlflow
import pickle
import logging
import argparse
import numpy as np
import torch.nn as nn
from torchinfo import summary
from dataset import CustomDataset
from autoencoder import AutoEncoder, get_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Trainer():

    def __init__(self, args, model, optimizer, train_loader, val_loader, scheduler, device, epoch):

        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.device = device
        self.epoch = epoch

        self.mlflow_tracking_arn = self.args.mlflow_tracking_arn
        self.experiment_name = self.args.experiment_name
        self.mlflow_run_name = self.args.mlflow_run_name
        print ("MLFLOW_TRACKING_ARN", self.mlflow_tracking_arn)
        print ("experiment_name", self.experiment_name)
        print ("run_name", self.mlflow_run_name)

        # Loss Function
        self.criterion = nn.L1Loss().to(self.device)
        self.anomaly_calculator = nn.L1Loss(reduction="none").to(self.device)

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def fit(self, ):

        mlflow.set_tracking_uri(self.mlflow_tracking_arn)
        mlflow.set_experiment(self.experiment_name)

        filter_string = f"run_name='{self.mlflow_run_name}'"
        run_id = mlflow.search_runs(filter_string=filter_string)["run_id"][0]
        print ("filter_string", filter_string)
        print ("mlflow.search_runs(filter_string=filter_string)", mlflow.search_runs(filter_string=filter_string))
        print ("run_id", run_id)
        params = {k: o for k, o in vars(self.args).items()}

        with mlflow.start_run(run_id=run_id, log_system_metrics=True):
            with mlflow.start_run(run_name="Training", log_system_metrics=True, nested=True) as training_run:

                #mlflow.pytorch.autolog()
                 # Enable autologging in MLflow
                mlflow.log_params({**params})
                mlflow.autolog()

                self.model.to(self.device)
                best_score = 0
                for epoch in range(self.epoch):
                    self.model.train()
                    train_loss = []
                    for time, x, y in self.train_loader:
                        time, x = time.to(self.device), x.to(self.device)

                        self.optimizer.zero_grad()

                        _x = self.model(time, x)
                        t_emb, _x = self.model(time, x)
                        x = torch.cat([t_emb, x], dim=1)

                        loss = self.criterion(x, _x)
                        loss.backward()
                        self.optimizer.step()

                        train_loss.append(loss.item())

                    if epoch % 10 == 0 :
                        score = self.validation(self.model, 0.95)
                        diff = self.cos(x, _x).cpu().tolist()
                        print(f'Epoch : [{epoch}] Train loss : [{np.mean(train_loss)}], Train cos : [{np.mean(diff)}] Val cos : [{score}])')

                    if self.scheduler is not None:
                        self.scheduler.step(score)

                    if best_score < score:
                        best_score = score
                        #torch.save(model.module.state_dict(), './best_model.pth', _use_new_zipfile_serialization=False)
                        torch.save(self.model.module.state_dict(), os.path.join(self.args.model_dir, "./best_model.pth"), _use_new_zipfile_serialization=False)
                        
                        # Log model summary.
                        with open('model_summary.txt', 'w') as f:
                            f.write(str(summary(self.model)))
                        mlflow.log_artifact('model_summary.txt')
                    
                    mlflow.log_metric(
                        key='train_loss',
                        value=np.mean(train_loss),
                        step=(epoch//10)
                    )

                    mlflow.log_metric(
                        key='train_cosim',
                        value=np.mean(diff),
                        step=(epoch//10)
                    )

                    mlflow.log_metric(
                        key='validation_cosim',
                        value=score,
                        step=(epoch//10)
                    )
                
        return self.model
    
    def validation(self, eval_model, thr):
        
        eval_model.eval()
        with torch.no_grad():
            for time, x, y in self.val_loader:
                time, x, y= time.to(self.device), x.to(self.device), y.to(self.device)
                _x = self.model(time, x)

                t_emb, _x = self.model(time, x)
                x = torch.cat([t_emb, x], dim=1)

                anomal_score = self.anomaly_calculator(x, _x)
                diff = self.cos(x, _x).cpu().tolist()

        return np.mean(diff)

def check_gpu():

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"# DEVICE {i}: {torch.cuda.get_device_name(i)}")
            print("- Memory Usage:")
            print(f"  Allocated: {round(torch.cuda.memory_allocated(i)/1024**3,1)} GB")
            print(f"  Cached:    {round(torch.cuda.memory_reserved(i)/1024**3,1)} GB\n")

    else:
        print("# GPU is not available")

    # GPU 할당 변경하기
    #GPU_NUM = 0 # 원하는 GPU 번호 입력
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #torch.cuda.set_device(device) # change allocation of current GPU
    print ('# Current cuda device: ', torch.cuda.current_device()) # check
    
    return device

def from_pickle(obj_path):

    with open(file=obj_path, mode="rb") as f:
        obj=pickle.load(f)

    return obj

def get_and_define_dataset(args):
    
    train_x_scaled_shingle = from_pickle(
        obj_path=os.path.join(
            args.train_data_dir,
            "data_x_scaled_shingle.pkl"
        )
    )
    
    train_y_shingle = from_pickle(
        obj_path=os.path.join(
            args.train_data_dir,
            "data_y_shingle.pkl"
        )
    )

    train_ds = CustomDataset(
        x=train_x_scaled_shingle,
        y=train_y_shingle
    )

    test_ds = CustomDataset(
        x=train_x_scaled_shingle,
        y=train_y_shingle
    )
    
    return train_ds, test_ds

def get_dataloader(args, train_ds, test_ds):
    
    train_loader = torch.utils.data.DataLoader(
        dataset = train_ds,
        batch_size = args.batch_size,
        shuffle = True,
        pin_memory=True,
        num_workers=args.workers,
        prefetch_factor=3
    )

    val_loader = torch.utils.data.DataLoader(
        dataset = test_ds,
        batch_size = args.batch_size,
        shuffle = False,
        pin_memory=True,
        num_workers=args.workers,
        prefetch_factor=3
    )
    
    return train_loader, val_loader

def train(args):
    
    logger.info("Check gpu..")
    device = check_gpu()
    logger.info("Device Type: {}".format(device))
    
    logger.info("Load and define dataset..")
    train_ds, test_ds = get_and_define_dataset(args)
    
    logger.info("Define dataloader..")
    train_loader, val_loader = get_dataloader(args, train_ds, test_ds)
    
    logger.info("Set components..")

    model = nn.DataParallel(
        get_model(
            input_dim=args.num_features*args.shingle_size + args.emb_size,
            hidden_sizes=[64, 48],
            btl_size=32,
            emb_size=args.emb_size
        )
    )
    
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=args.lr
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='max',
        factor=0.5,
        patience=10,
        threshold_mode='abs',
        min_lr=1e-8,
        verbose=True
    )
    
    logger.info("Define trainer..")
    trainer = Trainer(
        args=args,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        device=device,
        epoch=args.epochs
    )
    
    logger.info("Start training..")
    model = trainer.fit()
    
    # Register the model with MLflow
    run_id = mlflow.last_active_run().info.run_id
    artifact_path = "model"
    model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=artifact_path)
    model_details = mlflow.register_model(model_uri=model_uri, name="sm-job-experiment-model")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--workers", type=int, default=os.environ["SM_HP_WORKERS"], metavar="W", help="number of data loading workers (default: 2)")
    parser.add_argument("--epochs", type=int, default=os.environ["SM_HP_EPOCHS"], metavar="E", help="number of total epochs to run (default: 150)")
    parser.add_argument("--batch_size", type=int, default=os.environ["SM_HP_BATCH_SIZE"], metavar="BS", help="batch size (default: 512)")
    parser.add_argument("--lr", type=float, default=os.environ["SM_HP_LR"], metavar="LR", help="initial learning rate (default: 0.001)")
    #parser.add_argument("--momentum", type=float, default=0.9, metavar="M", help="momentum (default: 0.9)")
    #parser.add_argument("--dist_backend", type=str, default="gloo", help="distributed backend (default: gloo)")
    #parser.add_argument("--hosts", type=json.loads, default=os.environ["SM_HOSTS"]) #["algo-1"]
    #parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"]) #"algo-1"
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"]) #"/opt/ml/model"
    parser.add_argument("--train_data_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"]) # /opt/ml/input/data/train
    parser.add_argument("--val_data_dir", type=str, default=os.environ["SM_CHANNEL_VALIDATION"]) # /opt/ml/input/data/valication
    parser.add_argument("--num_gpus", type=int, default=os.environ["SM_NUM_GPUS"]) #1

    parser.add_argument("--shingle_size", type=int, default=os.environ["SM_HP_SHINGLE_SIZE"])
    parser.add_argument("--num_features", type=int, default=os.environ["SM_HP_NUM_FEATURES"])
    parser.add_argument("--emb_size", type=int, default=os.environ["SM_HP_EMB_SIZE"])

    parser.add_argument("--mlflow_tracking_arn", type=str, default=os.environ['MLFLOW_TRACKING_ARN'])
    parser.add_argument("--experiment_name", type=str, default=os.environ['EXPERIMENT_NAME'])
    parser.add_argument("--mlflow_run_name", type=str, default=os.environ['MLFLOW_RUN_NAME'])

    print (parser.parse_args())

    train(parser.parse_args())