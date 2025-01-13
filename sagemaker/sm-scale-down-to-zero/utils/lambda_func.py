import ast
import boto3
from pprint import pprint

class lambda_handler():
    
    def __init__(self, region_name="ap-northeast-2"):
        
        self.client = boto3.client('lambda', region_name=region_name)
        
    def _has_function(self, strFuncName):
        
        try:
            response = self.client.get_function(
                FunctionName=strFuncName,
            )
            return True
        except Exception as e:
            if "Function not found" in str(e):
                return False
            else:
                return e
                
    def create_function(self, **kwargs):
        
        print ("== CREATE LAMBDA FUNCTION ==")
        
        strFuncName = kwargs["FunctionName"]
        if self._has_function(strFuncName):
            print (f"  lambda function: [{strFuncName}] is already exist!!, so, this will be deleted and re-created.")
            self.delete_function(strFuncName)
        
        response = self.client.create_function(**kwargs)
        
        print("Argments for lambda below:\n")
        pprint (response)
        print (f"\n  lambda function: [{strFuncName}] is created successfully")
        print ("== COMPLETED ==")
        
        return response["FunctionArn"]
   
    def delete_function(self, strFuncName):
        
        response = self.client.delete_function(
            FunctionName=strFuncName,
        )
        print (f"  lambda function: [{strFuncName}] is deleted successfully")

if __name__ == "__main__":
    #https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda.html#Lambda.Client.create_function
    pass