

#########################
# 배포 관련 함수 
#########################

import boto3
sagemaker_boto_client = boto3.client('sagemaker')

from IPython.display import display as dp

def clean_pipeline(pipeline_name, isDeletePipeline=False, verbose=False):
    '''
    파이프라인 삭제
    pipeline_name = 'sagemaker-pipeline-step-by-step-phase01'
    clean_pipeline(pipeline_name = pipeline_name, isDeletePipeline=False, verbose=False)   
    '''
    # project_prefix 의 prefix 를 가진 파이프라인을 모두 가져오기
    response = sagemaker_boto_client.list_pipelines(
        PipelineNamePrefix= pipeline_name,
        SortBy= 'Name',    
        SortOrder='Descending',
        #NextToken='string',
        MaxResults=100
    )

    if verbose:
        print(f"\n### Display pipelines with this prefix {pipeline_name} \n")        
        dp(response)

    
    # pipeline_name 보여주기
    if any(pipeline["PipelineDisplayName"] == pipeline_name for pipeline in response["PipelineSummaries"]):
        print(f"pipeline {pipeline_name} exists")
        response = sagemaker_boto_client.describe_pipeline(
            PipelineName= pipeline_name
        )    
    
        if verbose:
            print(f"\n### pipeline {pipeline_name} definiton is \n")
            dp(response)
            
        if isDeletePipeline:
            sagemaker_boto_client.delete_pipeline(PipelineName= pipeline_name)            
            print(f"pipeline {pipeline_name} is deleted")            

    else:
        print(f"pipeline {pipeline_name} doesn't exists")



def is_available_endpoint(endpoint_name, verbose=False):
    '''
    Return True if endpoint is in service, otherise do False
    '''
    try:
        response = sagemaker_boto_client.list_endpoints(NameContains=endpoint_name)
        #existing_endpoints = sagemaker_boto_client.list_endpoints(NameContains=endpoint_name)['Endpoints']

        if verbose:
            print("Response: \n", response)


        EndpointStatus = response['Endpoints'][0]['EndpointStatus']
        if verbose:
            print("EndpointStatus: ", EndpointStatus)
    
        
        if EndpointStatus == 'InService':
            return True
        else:
            return False
    except BaseException as error:
        error_msg = f"An exception occurred: {error}"
        print(error_msg)  
        return False

    
    
def delete_endpoint(client, endpoint_name ,is_del_model=True):
    '''
    model, EndpointConfig, Endpoint 삭제
    '''
    response = client.describe_endpoint(EndpointName=endpoint_name)
    EndpointConfigName = response['EndpointConfigName']
    
    response = client.describe_endpoint_config(EndpointConfigName=EndpointConfigName)
    model_name = response['ProductionVariants'][0]['ModelName']    

#     print("EndpointConfigName: \n", EndpointConfigName)
#     print("model_name: \n", model_name)    

    if is_del_model: # 모델도 삭제 여부 임.
        client.delete_model(ModelName=model_name)    
        
    client.delete_endpoint(EndpointName=endpoint_name)
    client.delete_endpoint_config(EndpointConfigName=EndpointConfigName)    
    
    print(f'--- Deleted model: {model_name}')
    print(f'--- Deleted endpoint: {endpoint_name}')
    print(f'--- Deleted endpoint_config: {EndpointConfigName}')    


import sagemaker

def get_predictor(endpoint_name, session, csv_deserializer):
    '''
    predictor = get_predictor(endpoint_name, session, csv_deserializer)    
    '''
    predictor = sagemaker.predictor.Predictor(
        endpoint_name=endpoint_name,
        sagemaker_session= session,
        deserializer = csv_deserializer, # byte stream을 csv 형태로 변환하여 제공        
    )
    return predictor

    
    
from IPython.display import display as dp
 
def get_payload(sample, label_col = 'fraud', verbose=False):    
    '''
    아래왁 같이 ',' 형태의 문자형을 리턴함.
    0,0,750,2,0,2,0,2,1,16596.0,1,18,0,59,1,0,0,4500.0,0,0,1,1,0,52,2020,3,0,0,0,2,1,0,0,0,0,0,0,10,12096.0,1,3000,1,0,1,0    
    '''

    sample = sample.drop(columns=[label_col]) # 레이블 제거

    payload = sample.to_csv(header=None, index=None).splitlines() # 
    payload = payload[0]

    if verbose:
        #dp(sample)
        # print("payload length: \n", len(payload))    
        print("pay load type: ", type(payload))
        print("payload: \n", payload)
    
    return payload

def predict(predictor, payload):
    '''
    프리딕터에 콤마 분리된 문자형과 ContentType을 'text/csv' 로 제공
    참고:
        CSVDeserializer 를 사용하지 않으면 byte stream 으로 제공되기에, 아래와 같이 디코딩 하여 사용함.
        result = float(result.decode())
    '''

    result = predictor.predict(payload, initial_args = {"ContentType": "text/csv"})
    result = result[0]
    
    return result

    
def get_pos_scale_weight(df, label):
    '''
    1, 0 의 레이블 분포를 계산하여 클래스 가중치 리턴
    예: 1: 10, 0: 90 이면 90/10 = 9 를 제공함. 
    호출:
        class_weight = get_pos_scale_weight(train_prep_df, label='fraud')
    '''
    fraud_sum = df[df[label] == 1].shape[0]
    non_fraud_sum = df[df[label] == 0].shape[0]
    class_weight = int(non_fraud_sum / fraud_sum)
    print(f"fraud_sum: {fraud_sum} , non_fraud_sum: {non_fraud_sum}, class_weight: {class_weight}")
    return class_weight
    

