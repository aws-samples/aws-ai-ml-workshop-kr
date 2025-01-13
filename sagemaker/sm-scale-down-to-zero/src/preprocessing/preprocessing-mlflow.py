import os
import mlflow
import pickle
import argparse
import numpy as np
import pandas as pd
from distutils.dir_util import copy_tree
from sklearn.preprocessing import StandardScaler

class preprocess():
    
    def __init__(self, args):
        
        self.args = args
        self.proc_prefix = self.args.proc_prefix #'/opt/ml/processing'
        
        self.input_dir = os.path.join(self.proc_prefix, "input")
        self.output_dir = os.path.join(self.proc_prefix, "output")
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        self.mlflow_tracking_arn = self.args.mlflow_tracking_arn
        self.experiment_name = self.args.experiment_name
        self.mlflow_run_name = self.args.mlflow_run_name
        print ("MLFLOW_TRACKING_ARN", self.mlflow_tracking_arn)
        print ("experiment_name", self.experiment_name)
        print ("run_name", self.mlflow_run_name)
    
    def _data_split(self, ):
            
        #train_data_ratio = 0.99
        clicks_1T = pd.read_csv(os.path.join(self.input_dir, self.args.train_data_name))
    
        pdData = clicks_1T[["time", "page", "user", "click", "residual", "fault"]]
        print (f'Data shape: {pdData.shape}')
        
        mlflow.log_input(
            mlflow.data.from_pandas(
                df=pdData,
                source=self.input_dir,
                targets="fault"
            ),
            context="Preprocessing-input",
        )

        #train_x, train_y = pdTrain[[strCol for strCol in pdTrain.columns if strCol != "fault"]].values, pdTrain["fault"].values.reshape(-1, 1)
        #test_x, test_y = pdTest[[strCol for strCol in pdTest.columns if strCol != "fault"]].values, pdTest["fault"].values.reshape(-1, 1)
        #print (f'train_x: {train_x.shape}, train_y: {train_y.shape}, test_x: {test_x.shape}, test_y: {test_y.shape}')
        
        data_x, data_y = pdData[[strCol for strCol in pdData.columns if strCol != "fault"]].values, pdData["fault"].values.reshape(-1, 1)
        data_x, data_time = data_x[:, 1:], data_x[:, 0].reshape(-1, 1)
        print (f'data_x: {data_x.shape}, data_y: {data_y.shape}, data_time: {data_time.shape}')
        
        return data_x, data_y, data_time
    
    def _normalization(self, data_x):
        
        scaler = StandardScaler()
        scaler.fit(data_x)
        
        data_x_scaled = scaler.transform(data_x)
        
        dump_path = os.path.join(self.output_dir, "StandardScaler", "scaler.pkl")
        os.makedirs(os.path.join(self.output_dir, "StandardScaler"), exist_ok=True)   
        self._to_pickle(scaler, dump_path)
        
        return data_x_scaled
    
    def _shingle(self, data):
        
        shingle_size = self.args.shingle_size
        
        num_data, num_features = data.shape[0], data.shape[1]
        shingled_data = np.zeros((num_data-shingle_size+1, shingle_size*num_features))

        print (num_data, shingled_data.shape)

        for idx_feature in range(num_features):

            if idx_feature == 0:
                start, end = 0, shingle_size
            else:
                start = end
                end = start + shingle_size

            for n in range(num_data - shingle_size + 1):
                if n+shingle_size == num_data: shingled_data[n, start:end] = data[n:, idx_feature]    
                else: shingled_data[n, start:end] = data[n:(n+shingle_size), idx_feature]
                
        return shingled_data
    
    def _to_pickle(self, obj, save_path):
        
        with open(file=save_path, mode="wb") as f:
            pickle.dump(obj, f)
    
    def _from_pickle(self, obj_path):
        
        with open(file=obj_path, mode="rb") as f:
            obj=pickle.load(f)
        
        return obj
        

    def execution(self, ):

        mlflow.set_tracking_uri(self.mlflow_tracking_arn)
        mlflow.set_experiment(self.experiment_name)
        
        with mlflow.start_run(
            run_name=self.mlflow_run_name,
            log_system_metrics=True) as run:

            run_id = run.info.run_id
            print ("run_id", run_id)
            filter_string = f"run_name='{self.mlflow_run_name}'"
            run_id = mlflow.search_runs(filter_string=filter_string)["run_id"]
            print ("run_id 2", run_id)
 
            with mlflow.start_run(run_name="Preprocessing", log_system_metrics=True, nested=True):
                
                data_x, data_y, data_time = self._data_split()
                data_x_scaled = self._normalization(data_x)
                
                data_x_scaled_shingle = self._shingle(data_x_scaled)
                data_time_shingle = self._shingle(data_time)[:, -1].reshape(-1, 1)
                data_y_shingle = self._shingle(data_y)[:, -1].reshape(-1, 1)
                data_x_scaled_shingle = np.concatenate([data_time_shingle, data_x_scaled_shingle], axis=1)
                
                print (f'data_x_scaled_shingle: {data_x_scaled_shingle.shape}')
                print (f'data_y_shingle: {data_y_shingle.shape}')
                print (f'check label: {sum(data_y_shingle == data_y[self.args.shingle_size-1:])}')
                print (f'fault cnt, train_y_shingle: {sum(data_y_shingle)}, train_y: {sum(data_y[self.args.shingle_size-1:])}')
                
                self._to_pickle(data_x_scaled_shingle, os.path.join(self.output_dir, "data_x_scaled_shingle.pkl"))
                self._to_pickle(data_y_shingle, os.path.join(self.output_dir, "data_y_shingle.pkl"))
    
                
                
                print (self.args.shingle_size, type(self.args.shingle_size))
                print ("data_dir", os.listdir(self.input_dir))
                print ("self.output_dir", os.listdir(self.output_dir))
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc_prefix", type=str, default="/opt/ml/processing")
    parser.add_argument("--shingle_size", type=int, default=4)
    parser.add_argument("--train_data_name", type=str, default="merged_clicks_1T.csv")
    parser.add_argument("--mlflow_tracking_arn", type=str, default="mlflow_tracking_arn")
    parser.add_argument("--experiment_name", type=str, default="experiment_name")
    parser.add_argument("--mlflow_run_name", type=str, default="mlflow_run_name")

    args, _ = parser.parse_known_args()

    print("Received arguments {}".format(args))
        
    prep = preprocess(args)
    prep.execution()