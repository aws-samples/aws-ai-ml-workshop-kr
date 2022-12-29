def invoke_endpoint(runtime_client, endpoint_name, payload, content_type):
    '''
    로컬 엔드 포인트 호출
    '''


    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name, 
        ContentType=content_type, 
        # Accept='application/json',
        Body=payload,
        )

    result = response['Body'].read().decode().splitlines()    
    
    return result

def delete_endpoint(client, endpoint_name):
    print("#### Start")
    response = client.describe_endpoint(EndpointName=endpoint_name)
    EndpointConfigName = response['EndpointConfigName']    
    
    response = client.describe_endpoint_config(EndpointConfigName=EndpointConfigName)
    
    model_name = response['ProductionVariants'][0]['ModelName']


    print(f'--- Deleted model: {model_name}')
    print(f'--- Deleted endpoint: {endpoint_name}')
    print(f'--- Deleted endpoint_config: {EndpointConfigName}')    
    
    client.delete_model(ModelName=model_name)    
    client.delete_endpoint_config(EndpointConfigName=EndpointConfigName)        
    client.delete_endpoint(EndpointName=endpoint_name)

    
    
    return None


def delete_endpoint_detail(client, endpoint_name ,is_delete=False, is_del_model=True, is_del_endconfig=True,is_del_endpoint=True):
    '''
    model, EndpointConfig, Endpoint 삭제
    '''
    try:
        response = client.describe_endpoint(EndpointName=endpoint_name)
        EndpointConfigName = response['EndpointConfigName']

        response = client.describe_endpoint_config(EndpointConfigName=EndpointConfigName)
        model_name = response['ProductionVariants'][0]['ModelName']    

        print("model_name: \n", model_name)        
        print("EndpointConfigName: \n", EndpointConfigName)
        print("endpoint_name: \n", endpoint_name)    

        if is_delete: # is_delete가 True 이면 삭제
            if is_del_endconfig:
                client.delete_endpoint_config(EndpointConfigName=EndpointConfigName)    
                print(f'--- Deleted endpoint: {endpoint_name}')                


            if is_del_model: # 모델도 삭제 여부 임.
                client.delete_model(ModelName=model_name)    
                print(f'--- Deleted model: {model_name}')                

            if is_del_endpoint:
                client.delete_endpoint(EndpointName=endpoint_name)
                print(f'--- Deleted endpoint_config: {EndpointConfigName}')             
    except:
        pass


    
def create_sm_model(client, sm_model_name, ecr_image, model_artifact, role):
    # create sagemaker model
    response = client.list_models(
        NameContains=sm_model_name, 
    )

    existing_models = response['Models']    
    
    if not existing_models:     
        create_model_api_response = client.create_model(
                                            ModelName=sm_model_name, 
                                            PrimaryContainer={
                                                'Image': ecr_image,
                                                'ModelDataUrl': model_artifact,
                                                'Environment': {}
                                            },
                                            ExecutionRoleArn= role
                                    )

        print ("create_model API response: \n", create_model_api_response)
    else:
        print ("existing_models: ", existing_models)

def create_endpoint_config(sm_client, instance_type, sm_model_name, verbose=False ):
    endpoint_config_name=f'{sm_model_name}-config'    
    existing_configs = sm_client.list_endpoint_configs(NameContains=endpoint_config_name)['EndpointConfigs']
    
    if verbose:
        print("existing_configs: \n", existing_configs)

    #########################################
    ## endpoint_config 생성
    #########################################
        
    if not existing_configs:
        create_ep_config_response = sm_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[{
                'InstanceType': instance_type,
                'InitialVariantWeight': 1,
                'InitialInstanceCount': 1,
                'ModelName': sm_model_name,
                'VariantName': 'AllTraffic'
            }]
        )
        print(f"{endpoint_config_name} is created")
        existing_configs = sm_client.list_endpoint_configs(NameContains=endpoint_config_name)['EndpointConfigs']        

    else:
        print("existing_configs exists")
        
    return existing_configs[0]['EndpointConfigName']

import time
def create_sm_endpoint(sm_client, instance_type, endpoint_config_name, endpoint_name, verbose=False ):
    existing_configs = sm_client.list_endpoint_configs(NameContains=endpoint_config_name)['EndpointConfigs']
    
    if verbose:
        print("existing_configs: \n", existing_configs)


    existing_endpoints = sm_client.list_endpoints(NameContains=endpoint_name)['Endpoints']    
    if verbose:
        print("existing_endpoints: \n", existing_endpoints)

    if not existing_endpoints:
        print(f"Creating endpoint")        
        create_endpoint_response = sm_client.create_endpoint(
            EndpointName=endpoint_name, 
            EndpointConfigName=endpoint_config_name)
    else:
        print(f"Endpoint exists")       
        
    endpoint_info = sm_client.describe_endpoint(EndpointName=endpoint_name)
    endpoint_status = endpoint_info['EndpointStatus']


    print(f'Endpoint status is creating')    
    while endpoint_status == 'Creating':
        endpoint_info = sm_client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_status = endpoint_info['EndpointStatus']
        print(f'Endpoint status: {endpoint_status}')
        if endpoint_status == 'Creating':
            time.sleep(30)        

def show_inference_objects(client, endpoint_name):
    '''
    엔드포인트 관련 컨피그, 모델 정보 보여 줌
    '''
    print("endpoint_name: \n", endpoint_name)    
    endpoint_config = client.describe_endpoint(EndpointName=endpoint_name)['EndpointConfigName']
    print("endpoint_config: \n", endpoint_config)
    response = client.describe_endpoint_config(EndpointConfigName= endpoint_config)
    model_name = response['ProductionVariants'][0]['ModelName']
    print("model_name: \n", model_name)

def update_sm_endpoint(client, endpoint_name, new_endpoint_config_name):
    '''
    엔드포인트 업데이트 
    '''
    print("endpoint_name: \n", endpoint_name)    
    print("new_endpoint_config_name: ", new_endpoint_config_name)        
    
    current_endpoint_config_name = client.describe_endpoint(EndpointName=endpoint_name)['EndpointConfigName']
    print("current_endpoint_config_name: ", current_endpoint_config_name)    

    
    response = client.update_endpoint(
        EndpointName= endpoint_name,
        EndpointConfigName= new_endpoint_config_name,
    )
    
    print("Swapping is done")
    changed_endpoint_config_name = client.describe_endpoint(EndpointName=endpoint_name)['EndpointConfigName']
    print("changed_endpoint_config_name: \n", changed_endpoint_config_name)        

    
    return response

