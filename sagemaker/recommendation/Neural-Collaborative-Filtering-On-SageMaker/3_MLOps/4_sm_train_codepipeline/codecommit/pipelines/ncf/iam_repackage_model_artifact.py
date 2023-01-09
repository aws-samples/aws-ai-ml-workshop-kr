import json
import time
import boto3
import os
import tarfile


def lambda_handler(event, context):
    """ 
    입력으로 세이지 메이커 모델, 앤드 포인트 컨피그 및 앤드 포인트 이름을 받아서, 앤드포인트를 생성 함.
    """
    import boto3
    
    sm_client = boto3.client("sagemaker")
    

    ###################################
    # 입력 변수 저장
    ###################################
    
    source_path = event["source_path"]
    model_path = event["model_path"]
    bucket = event["bucket"]
    prefix = event["prefix"]        

    print("Display Input Arguments:\n")
    print("source_path: \n", source_path)
    print("model_path: \n", model_path)        


    ####################################
    ## 기본 폴더 이하의 파일 폴더 구조 확인 하기
    ####################################    
    base_dir = '/tmp'

    show_files_folder("\n###### Display Current Folder: ", base_dir)


    ####################################
    ## 다운로드  모델 아티펙트
    ####################################    
    model_dir = f"{base_dir}/model"
    os.makedirs(model_dir, exist_ok=True)

    model_download_path = download_s3_object(model_dir, model_path)

    logTitle = "\n### Downloading model.tar.gz"
    show_files_folder(logTitle, model_dir)    

    ####################################
    ## 다운로드  소스 아티펙트
    ####################################    
    source_dir = f"{base_dir}/source"
    os.makedirs(source_dir, exist_ok=True)

    source_download_path = download_s3_object(source_dir, source_path)

    logTitle = "\n### Downloading source.tar.gz"
    show_files_folder(logTitle, source_dir)    


    ####################################
    ## 모델 아티펙트의 압축 해제
    ####################################    
    temp_dir = f"{base_dir}/temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    with tarfile.open(model_download_path) as tar:
        tar.extractall(path= temp_dir)
    
    logTitle = "\n### Folder Status After untaring model artifact "
    show_files_folder(logTitle, temp_dir)    



    ####################################
    ## 소스 아티펙트의 압축 해제
    ####################################    
    code_dir = f"{temp_dir}/code"
    os.makedirs(code_dir, exist_ok=True)

    with tarfile.open(source_download_path) as tar:
        tar.extractall(path= code_dir)
    
    logTitle = "\n### Folder Status After untaring source artifact "
    show_files_folder(logTitle, temp_dir)    


    ####################################
    ## temp 의 파일을 압축하여 repackage 에 저장
    ####################################    
    repackage_dir = f"{base_dir}/repackage"
    os.makedirs(repackage_dir, exist_ok=True)
    
    target_file_name = f"{repackage_dir}/model.tar.gz"
    create_tar_dir(temp_dir, target_file_name)

    
    logTitle = "\n########## After taring temp folder, final repackage_dir is: "
    show_files_folder(logTitle, repackage_dir)    
    
    ####################################
    ## Uplaod final model.tar.gz
    ####################################    
    
    s3_model_url = upload_s3_object(bucket, prefix, target_file_name)
    
    
    return_msg = f"Repackaging is Done"        

    return {
        "statusCode": 200,
        "body": json.dumps(return_msg),
        "S3_Model_URI": s3_model_url,
    }

# coding: utf-8
import os
import tarfile

def create_tar_dir(path, target_file_name):
    '''
    path = 'code_pkg'
    target_file_name = 'model.tar.gz'
    create_tar_dir(path, target_file_name)
    '''
    tar = tarfile.open(target_file_name, 'w:gz')
   
    for i, (root, dirs, files) in enumerate(os.walk(path)):
        # 현재 폴더에 code 폴더, XXX.pth 일 경우
        if (len(files) > 0) & (len(dirs) > 0) :
            print("i:", i, root)
            print("dirs: ", dirs)
            print("files: ", files)            
            print("root: ", root)
            
            for file_name in files:
                tar.add(os.path.join(root, file_name), file_name)
                print("i: ",i,  " Adding file_name : ",file_name)                
        # code 폴더 처리
        else:
            print("i:", i)
            print("dirs: ", dirs)
            print("files: ", files)            
            print("root: ", root)
            subfolder = root.split("/")[-1]
            print("subfolder: ", subfolder)
            for file_name in files:
                tar.add(os.path.join(root, file_name), f"{subfolder}/{file_name}")
                print("i: ",i,  "Adding file_name : ",file_name)                

    tar.close()
    
    return None


def upload_s3_object(bucket, prefix, file_path):
    '''
    upload model file
    '''
    import boto3
    import botocore
    
    file_name = file_path.split("/")[-1]
    Target = f"s3://{bucket}/{prefix}/{file_name}"
    key = f"{prefix}/{file_name}"
    
    s3 = boto3.resource('s3')

    # print("bucket: ", bucket)
    # print("prefix: ", prefix)    
    # print("file_path: ", file_path)
    # print("file_name: ", file_name)
    print("##### Target: ", Target)        
    

    try:
        s3.Bucket(bucket).upload_file(
            file_path,
            key)
            
        return Target
        
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise
    

def download_s3_object(destination, s3_url):
    '''
    Parsing s3_url and download_file
    '''
    import boto3
    import botocore
    
    # s3_Uri parsing to bucket , key, file name
    tokens = s3_url.split('/')
    
    # print("####### tokens: ", tokens)
    
    BUCKET_NAME = tokens[2]
    file_name = tokens[-1]
    
    new_delimiter = BUCKET_NAME + '/'
    tokens = s3_url.split(new_delimiter)    
    # print("####### tokens: ", tokens)    
    
    KEY = tokens[-1]
    
    # print("BUCKET_NAME: ", BUCKET_NAME)
    # print("KEY: ", KEY)    
    # print("file_name: ", file_name)
    

    Target = f"{destination}/{file_name}"
    
    s3 = boto3.resource('s3')
    
    try:
        s3.Bucket(BUCKET_NAME).download_file(KEY, Target)
        return Target
        
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise


def show_files_folder(logTitle, folder):
    # Traverse all files
    print(logTitle)
    for file in os.walk(folder):
        # logger.info(f"{file}")
        print(f"{file}")        

    return None





        
