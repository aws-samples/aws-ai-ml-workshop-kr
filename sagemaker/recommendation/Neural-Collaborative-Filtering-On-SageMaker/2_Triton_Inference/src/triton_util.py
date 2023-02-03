import os

account_id_map = {
    "us-east-1": "785573368785",
    "us-east-2": "007439368137",
    "us-west-1": "710691900526",
    "us-west-2": "301217895009",
    "eu-west-1": "802834080501",
    "eu-west-2": "205493899709",
    "eu-west-3": "254080097072",
    "eu-north-1": "601324751636",
    "eu-south-1": "966458181534",
    "eu-central-1": "746233611703",
    "ap-east-1": "110948597952",
    "ap-south-1": "763008648453",
    "ap-northeast-1": "941853720454",
    "ap-northeast-2": "151534178276",
    "ap-southeast-1": "324986816169",
    "ap-southeast-2": "355873309152",
    "cn-northwest-1": "474822919863",
    "cn-north-1": "472730292857",
    "sa-east-1": "756306329178",
    "ca-central-1": "464438896020",
    "me-south-1": "836785723513",
    "af-south-1": "774647643957",
}

def setup_triton_client():
    import numpy as np
    import sys

    import tritonclient.grpc as grpcclient

    try:
        keepalive_options = grpcclient.KeepAliveOptions(
            keepalive_time_ms=2**31 - 1,
            keepalive_timeout_ms=20000,
            keepalive_permit_without_calls=False,
            http2_max_pings_without_data=2
        )
        triton_client = grpcclient.InferenceServerClient(
            url='localhost:8001',
            verbose=False,
            keepalive_options=keepalive_options)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit()
    
    return triton_client, grpcclient

def infer_triton_client(triton_client, model_name, inputs, outputs):
    '''
    Triton 추론 요청
    '''
    # Test with outputs
    results = triton_client.infer(model_name=model_name,
                                    inputs=inputs,
                                    outputs=outputs,
                                    headers={'test': '1'})

    # Get the output arrays from the results
    output0_data = results.as_numpy('OUTPUT__0')
    print("#### output #####")
    print(output0_data.shape)
    print("#### output values #####")
    print(output0_data)    
    
    return None


def make_folder_structure(model_serving_folder, model_name):
    '''
    폴더 구조 생성
    '''
    os.makedirs(model_serving_folder, exist_ok=True)
    os.makedirs(f"{model_serving_folder}/{model_name}", exist_ok=True)
    os.makedirs(f"{model_serving_folder}/{model_name}/1", exist_ok=True)    
    
    return None

def copy_artifact(model_serving_folder, model_name, model_artifact, config):
    '''
    model.pt, config.pbtxt 파일을 지정된 위치에 복사
    '''
    os.system(f"cp {model_artifact} {model_serving_folder}/{model_name}/1/model.pt")    
    os.system(f"cp {config} {model_serving_folder}/{model_name}/config.pbtxt")        
    os.system(f"ls -R {model_serving_folder}")
    
    return None

def remove_folder(model_serving_folder):
    '''
    해당 폴더 전체를 삭제
    '''
    os.system(f"rm -rf  {model_serving_folder}")   
    print(f"{model_serving_folder} is removed") 
    
    return None

def tar_artifact(model_serving_folder, model_name):
    '''
    해당 폴더를 tar 로 압축
    '''
    model_tar_file = f"{model_name}.model.tar.gz"
    os.system(f"tar -C {model_serving_folder}/ -czf {model_tar_file} {model_name}")
    os.system(f"tar tvf {model_tar_file}")
    
    return model_tar_file

def upload_tar_s3(sagemaker_session, tar_file_path, prefix):
    '''
    해당 파일을 S3에 업로딩
    '''
    model_uri_pt = sagemaker_session.upload_data(path=tar_file_path, key_prefix=prefix)
    
    return model_uri_pt
