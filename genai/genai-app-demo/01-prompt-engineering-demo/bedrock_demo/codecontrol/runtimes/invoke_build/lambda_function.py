import os
import boto3
import json

REPO_NAME = os.environ['REPO_NAME']
LAST_COMMIT_ID = os.environ['LAST_COMMIT_ID']
BUILD_PROJECT_NAME = os.environ['BUILD_PROJECT_NAME']
DIR_FILTER = os.environ['DIR_FILTER']

codecommit = boto3.client('codecommit')
codebuild = boto3.client('codebuild')
ssm = boto3.client('ssm')
secrets = boto3.client('secretsmanager')


def detect_file_change(current_commit_id: str) -> set:
    params = {
        'repositoryName': REPO_NAME,
        'MaxResults': 100,
        'afterCommitSpecifier': current_commit_id
    }
    
    try:
        response = ssm.get_parameter(Name=LAST_COMMIT_ID)
        previous_commit_id = response['Parameter']['Value']
        params['beforeCommitSpecifier'] = previous_commit_id
        print('Previous Commit ID Found. Changes will be those between previous and lastest.')
    except ssm.exceptions.ParameterNotFound:
        print('Previous Commit ID not Found. All changes will be detected.')

    response = codecommit.get_differences(**params)
    detected = get_changed_file(response, DIR_FILTER)
    
    while 'NextToken' in response:
        response = codecommit.get_differences(NextToken=response['NextToken'], **params)
        detected.extend(get_changed_file(response, DIR_FILTER))

    print(detected)
    
    # Set Collection. For duplicated items.
    container_identifier = {'/'.join(d.split('/')[:2]) for d in detected}

    return container_identifier


def get_changed_file(data: dict, path_filter: str) -> list:
    """
    changeType : A(Addition), D(Deletion), M(Modification)
    Take Only A, M
    """
    changed = []

    items = data['differences']
    if not items:
       return changed 
    
    for item in items:        
        if item['changeType'] == 'A' or item['changeType'] == 'M':            
            if item['afterBlob']['path'].startswith(path_filter):                
                changed.append(item['afterBlob']['path'])

    return changed


def lambda_handler(event, context):
    print(event)
    
    current_commit_id = event['detail']['commitId']
    container_identifier = detect_file_change(current_commit_id)
    
    # Invoke Container Build
    response = codecommit.get_file(
        repositoryName=REPO_NAME,
        commitSpecifier='main',
        filePath='image_repo.json'
    )

    # source path : image repo name
    IMAGE_REPOS = json.loads(response['fileContent'])

    for identifier in container_identifier:
        print(identifier, IMAGE_REPOS[identifier])
        codebuild.start_build(
            projectName=BUILD_PROJECT_NAME,
            environmentVariablesOverride=[
                {
                    'name': 'IMAGE_REPO_NAME',
                    'value': IMAGE_REPOS[identifier],
                    'type': 'PLAINTEXT'
                },
                {
                    'name': 'CONTAINER_FOLDER',
                    'value': identifier,
                    'type': 'PLAINTEXT'
                }
            ]
        )
    
    # Store Commit ID
    ssm.put_parameter(Name=LAST_COMMIT_ID, Value=current_commit_id, Type='String', Overwrite=True)
