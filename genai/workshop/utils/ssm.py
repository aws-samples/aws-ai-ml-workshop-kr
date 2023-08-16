import os
import boto3

class parameter_store():
    
    def __init__(self, region_name="ap-northeast-2"):
        
        self.ssm = boto3.client('ssm', region_name=region_name)
        
    def put_params(self, key, value, dtype="String", overwrite=False, enc=False) -> str:
        
        #aws ssm put-parameter --name "RDS-MASTER-PASSWORD" --value 'PASSWORD' --type "SecureString"
        if enc: dtype="SecureString"
        if overwrite:
            strQuery = ''.join(['aws ssm put-parameter', ' --name ', '"', str(key), '"', ' --value ', '"', str(value), '"', ' --type ', '"', str(dtype), '"', ' --overwrite'])
        strResponse = os.popen(strQuery).read()
        
        if strResponse != '': return 'Store suceess'
        else: return 'Error'
    
    def get_params(self, key, enc=False):
        
        if enc: WithDecryption = True
        else: WithDecryption = False
        response = self.ssm.get_parameters(
            Names=[key,],
            WithDecryption=WithDecryption
        )
        
        return response['Parameters'][0]['Value']

    def get_all_params(self, ):

        response = self.ssm.describe_parameters(MaxResults=50)

        return [dicParam["Name"] for dicParam in response["Parameters"]]

    def delete_param(self, listParams):

        response = self.ssm.delete_parameters(
            Names=listParams
        )
        print (f"  parameters: {listParams} is deleted successfully")
        

    
