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

