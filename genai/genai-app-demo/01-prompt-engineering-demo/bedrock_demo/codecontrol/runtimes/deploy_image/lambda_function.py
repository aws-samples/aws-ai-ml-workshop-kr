import os
import json
import boto3

FUNC_ROLE = os.environ['FUNC_ROLE']
FUNC_SG_ID = os.environ['FUNC_SG_ID']
FUNC_SUBNET_ID = os.environ['FUNC_SUBNET_ID']
REPO_NAME = os.environ['REPO_NAME']

lambda_ = boto3.client('lambda')
codecommit = boto3.client('codecommit')


def lambda_handler(event, context):
    print(event)

    account = event['account']
    region = event['region']
    repository_name = event['detail']['repository-name']
    image_digest = event['detail']['image-digest']

    # 'xxxxxxxxx.dkr.ecr.ap-northeast-2.amazonaws.com/text@sha256:673368794197a1b022064aa9e50534e6b9240db9af3cbfeecde29912e8e05fc6'
    image_uri = f'{account}.dkr.ecr.{region}.amazonaws.com/{repository_name}@{image_digest}'

    # Get Func Name
    response = codecommit.get_file(
        repositoryName=REPO_NAME,
        commitSpecifier='main',
        filePath='lambda_func.json'
    )

    # image repo name : lambda func name
    LAMBDA_FUNC = json.loads(response['fileContent'])
    func_name = LAMBDA_FUNC[repository_name]
    
    try:
        response = lambda_.update_function_code(
            FunctionName=func_name,
            ImageUri=image_uri,
            Architectures=['x86_64']
        )
        print('Function updated')
    except lambda_.exceptions.ResourceNotFoundException as e:
        if 'Function not found' in e.response['Error']['Message']:
            print('Function not found. It will be created')
            response = lambda_.create_function(
                FunctionName=func_name,
                Role=FUNC_ROLE,
                Code={
                    'ImageUri': image_uri,
                },
                Architectures=['x86_64'],
                Timeout=300,
                VpcConfig={
                    'SubnetIds': FUNC_SUBNET_ID.split(','),
                    'SecurityGroupIds': [
                        FUNC_SG_ID,
                    ],
                    'Ipv6AllowedForDualStack': False
                },
                PackageType='Image',
                Environment={
                    'Variables': {
                        'BEDROCK_REGION_NAME': 'us-west-2'
                    }
                }
            )
        else:
            raise e
    
    print(response)
