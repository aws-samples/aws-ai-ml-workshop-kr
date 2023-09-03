import boto3

def get_apigateway_url():

    aws_region = boto3.Session().region_name
    
    key_dic = {
        "us-east-1": "gefy6h11cg",
        "us-west-2": "a6uibyuabj"
    }
    restapi_id = key_dic[aws_region]
    url = f'https://{restapi_id}.execute-api.{aws_region}.amazonaws.com/api/'.replace('"','')
    
    return restapi_id, url