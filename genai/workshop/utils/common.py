import boto3

def get_apigateway_url():

    aws_region = boto3.Session().region_name
    
    key_dic = {
        "us-east-1": "6bk4r5mo4f", 
        "us-west-2": "eilrpc1i0l"
    }
    restapi_id = key_dic[aws_region]
    url = f'https://{restapi_id}.execute-api.{aws_region}.amazonaws.com/api/'.replace('"','')
    
    return restapi_id, url