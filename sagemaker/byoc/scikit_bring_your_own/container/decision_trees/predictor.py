# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import io
import json
import os
import pickle
import signal
import sys
import traceback

import flask
import pandas as pd

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

import logging
import time
from datetime import datetime

# 로깅 설정
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 기존 핸들러 제거 (중복 방지)
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# 커스텀 로그 포맷 생성
class GunicornStyleFormatter(logging.Formatter):
    def format(self, record):
        # 현재 UTC 시간을 Gunicorn 형식으로 포맷팅 (+0000은 UTC 타임존)
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S +0000')
        # 현재 프로세스 ID
        process_id = os.getpid()
        # 로그 레벨
        level = record.levelname
        # 로그 메시지
        message = record.getMessage()
        
        # Gunicorn 스타일 포맷 적용
        return f"[{timestamp}] [{process_id}] [{level}] {message}"

# 로그 핸들러 추가
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(GunicornStyleFormatter())
logger.addHandler(handler)


class ScoringService(object):
    model = None  # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        logger.info("get_model")
        if cls.model == None:
            try:
                with open(os.path.join(model_path, "decision-tree-model.pkl"), "rb") as inp:
                    cls.model = pickle.load(inp)
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise
        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.
        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        logger.info("predict start")
        try:
            clf = cls.get_model()
            predictions = clf.predict(input)
            logger.info(f"Prediction complete, generated {len(predictions)} predictions")
            return predictions
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    logger.info("ping start")    
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == "text/csv":
        data = flask.request.data.decode("utf-8")
        s = io.StringIO(data)
        data = pd.read_csv(s, header=None)
    else:
        return flask.Response(
            response="This predictor only supports CSV data", status=415, mimetype="text/plain"
        )

    print("Invoked with {} records".format(data.shape[0]))

    # Do the prediction
    predictions = ScoringService.predict(data)

    # Convert from numpy back to CSV
    out = io.StringIO()
    pd.DataFrame({"results": predictions}).to_csv(out, header=False, index=False)
    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype="text/csv")