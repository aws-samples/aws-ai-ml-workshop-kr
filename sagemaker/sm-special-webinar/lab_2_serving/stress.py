import os
import json
import time
import boto3
import io
from io import StringIO
import pandas as pd
from locust import HttpUser, task, events, between


class SageMakerConfig:

    def __init__(self):
        self.__config__ = None

    @property
    def data_file(self):
        return self.config["dataFile"]

    @property
    def content_type(self):
        return self.config["contentType"]

    @property
    def show_endpoint_response(self):
        return self.config["showEndpointResponse"]
    
    @property
    def num_test_samples(self):
        return self.config["numTestSamples"]

    @property
    def config(self):
        self.__config__ = self.__config__ or self.load_config()
        return self.__config__

    def load_config(self):
        config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        with open(config_file, "r") as c:
            return json.loads(c.read())


    
class SageMakerEndpointTestSet(HttpUser):
    wait_time = between(5, 15)
    
    def __init__(self, parent):
        super().__init__(parent)
        self.config = SageMakerConfig()
        
        
    def on_start(self):
        data_file_full_path = os.path.join(os.path.dirname(__file__), self.config.data_file)
        
        test_df = pd.read_csv(data_file_full_path)
        y_test = test_df.iloc[:, 0].astype('int')
        test_df = test_df.drop('fraud', axis=1)

        csv_file = io.StringIO()
        test_df[0:self.config.num_test_samples].to_csv(csv_file, sep=",", header=False, index=False)
        self.payload = csv_file.getvalue()
        

    @task
    def test_invoke(self):
        response = self._locust_wrapper(self._invoke_endpoint, self.payload)
        if self.config.show_endpoint_response:
            print(response["Body"].read())

    
    def _invoke_endpoint(self, payload):
        region = self.client.base_url.split("://")[1].split(".")[2]
        endpoint_name = self.client.base_url.split("/")[-2]
        runtime_client = boto3.client('sagemaker-runtime', region_name=region)

        response = runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            Body=payload,
            ContentType=self.config.content_type
        )

        return response
    

    def _locust_wrapper(self, func, *args, **kwargs):
        """
        Locust wrapper so that the func fires the sucess and failure events for custom boto3 client
        :param func: The function to invoke
        :param args: args to use
        :param kwargs:
        :return:
        """
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            total_time = int((time.time() - start_time) * 1000)
            events.request_success.fire(request_type="boto3", name="invoke_endpoint", response_time=total_time,
                                        response_length=0)

            return result
        except Exception as e:
            total_time = int((time.time() - start_time) * 1000)
            events.request_failure.fire(request_type="boto3", name="invoke_endpoint", response_time=total_time,
                                        response_length=0,
                                        exception=e)

            raise e
