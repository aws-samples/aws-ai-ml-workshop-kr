import boto3
import sys
from typing import Optional

def find_s3_bucket_name_by_suffix(name_suffix: str) -> Optional[str]:
    """Find S3 bucket name by name suffix"""
    client = boto3.client('s3')
    
    response = client.list_buckets()
    for bucket in response['Buckets']:
        if bucket['Name'].endswith(name_suffix):
            return bucket['Name']
    return None

def find_state_machine_arn_by_prefix(name_prefix: str) -> Optional[str]:
    """Find state machine ARN by name prefix"""
    client = boto3.client('stepfunctions')
    
    paginator = client.get_paginator('list_state_machines')
    for page in paginator.paginate():
        for sm in page['stateMachines']:
            if sm['name'].startswith(name_prefix):
                return sm['stateMachineArn']
    return None

def get_role_arn(role_name_part):
    """
    Retrieve IAM role ARN based on partial role name match.
    
    Args:
        role_name_part: Part of the role name to search for
        
    Returns:
        Role ARN if found, None otherwise
    """
    iam = boto3.client('iam')
    
    try:
        response = iam.list_roles()
        for role in response['Roles']:
            if role_name_part in role['RoleName']:
                return role['Arn']
        print("[ERROR] Role not found!")
        return None
    except Exception as e:
        print(f"Error retrieving role: {e}")
        return None