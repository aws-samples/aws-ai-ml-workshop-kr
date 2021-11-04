import argparse
import logging
import os
import json
import boto3

import subprocess
import sys

from urllib.parse import urlparse

os.system('pip install autogluon')
# from autogluon import TabularPrediction as task
from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd # this should come after the pip install. 

logging.basicConfig(level=logging.DEBUG)
logging.info(subprocess.call('ls -lR /opt/ml/input'.split()))

# ------------------------------------------------------------ #
# Training methods                                             #
# ------------------------------------------------------------ #


def train(args):
  # SageMaker passes num_cpus, num_gpus and other args we can use to tailor training to
  # the current container environment, but here we just use simple cpu context.

  num_gpus = int(os.environ['SM_NUM_GPUS'])
  current_host = args.current_host
  hosts = args.hosts
  model_dir = args.model_dir
  target = args.target

  # load training and validation data

  training_dir = args.train
  filename = args.filename
  logging.info(training_dir)
#   train_data = task.Dataset(file_path=training_dir + '/' + filename)
  train_data = TabularDataset(data=training_dir + '/' + filename)

#   predictor = task.fit(train_data = train_data, label=target, output_directory=model_dir)
  predictor = TabularPredictor(label=target, path=model_dir).fit(train_data)
          
  return predictor
    

# ------------------------------------------------------------ #
# Hosting methods                                              #
# ------------------------------------------------------------ #

# def model_fn(model_dir):
#   """
#   Load the gluon model. Called once when hosting service starts.
#   :param: model_dir The directory where model files are stored.
#   :return: a model (in this case an AutoGluon network)
#   """
#   net = TabularPredictor.load(model_dir)
#   return net


def transform_fn(net, data, input_content_type, output_content_type):
  """
  Transform a request using the Gluon model. Called once per request.
  :param net: The AutoGluon model.
  :param data: The request payload.
  :param input_content_type: The request content type.
  :param output_content_type: The (desired) response content type.
  :return: response payload and content type.
  """
  # we can use content types to vary input/output handling, but
  # here we just assume json for both
  data = json.loads(data)
  # the input request payload has to be deserialized twice since it has a discrete header
  data = json.loads(data)
  df_parsed = pd.DataFrame(data)

  prediction = net.predict(df_parsed)

  response_body = json.dumps(prediction.tolist())

  return response_body, output_content_type


# ------------------------------------------------------------ #
# Training execution                                           #
# ------------------------------------------------------------ #

def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
  parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
  parser.add_argument('--filename', type=str, default='train.csv')

  parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
  parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
  
  parser.add_argument('--target', type=str, default='target')
  parser.add_argument('--s3-output', type=str, default='s3://autogluon-test/results')
  parser.add_argument('--training-job-name', type=str, default=json.loads(os.environ['SM_TRAINING_ENV'])['job_name'])

  return parser.parse_args()


if __name__ == '__main__':
  args = parse_args()
  predictor = train(args)
  
  training_dir = args.train
  train_file = args.filename
  test_file = train_file.replace('train', 'test', 1)
  dataset_name = train_file.split('_')[0]
  print(dataset_name)
  
#   test_data = task.Dataset(file_path=os.path.join(training_dir, test_file))
  test_data = TabularDataset(data=os.path.join(training_dir, test_file))
  u = urlparse(args.s3_output, allow_fragments=False)
  bucket = u.netloc
  print(bucket)
  prefix = u.path.strip('/')
  print(prefix)

  s3 = boto3.client('s3')
  
  try:
    y_test = test_data[args.target]  # values to predict
    test_data_nolab = test_data.drop(labels=[args.target], axis=1) # delete label column to prove we're not cheating

    y_pred = predictor.predict(test_data_nolab)
    y_pred_df = pd.DataFrame.from_dict({'True': y_test, 'Predicted': y_pred})
    pred_file = f'{dataset_name}_test_predictions.csv'
    y_pred_df.to_csv(pred_file, index=False, header=True)

    leaderboard = predictor.leaderboard()
    lead_file = f'{dataset_name}_leaderboard.csv'
    leaderboard.to_csv(lead_file)

    perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
    perf_file = f'{dataset_name}_model_performance.txt'
    with open(perf_file, 'w') as f:
      print(json.dumps(perf, indent=4), file=f)

    summary = predictor.fit_summary()
    summ_file = f'{dataset_name}_fit_summary.txt'
    with open(summ_file, 'w') as f:
      print(summary, file=f)

    files_to_upload = [pred_file, lead_file, perf_file, summ_file]

  except:
    y_pred = predictor.predict(test_data)
    y_pred_df = pd.DataFrame.from_dict({'Predicted': y_pred})
    pred_file = f'{dataset_name}_test_predictions.csv'
    y_pred_df.to_csv(pred_file, index=False, header=True)
    
    leaderboard = predictor.leaderboard()
    lead_file = f'{dataset_name}_leaderboard.csv'
    leaderboard.to_csv(lead_file)
    
    files_to_upload = [pred_file, lead_file]
      
  for file in files_to_upload:
    s3.upload_file(file, bucket, os.path.join(prefix, args.training_job_name.replace('mxnet-training', 'autogluon', 1), file))
