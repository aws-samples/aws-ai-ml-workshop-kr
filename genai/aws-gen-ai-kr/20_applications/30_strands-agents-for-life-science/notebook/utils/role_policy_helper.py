import boto3
import json
from botocore.exceptions import ClientError, NoCredentialsError
from typing import List

class SageMakerRolePolicyChecker:
    def __init__(self):
        """Initialize the checker with AWS clients."""
        try:
            self.iam_client = boto3.client('iam')
            self.sagemaker_client = boto3.client('sagemaker')
        except NoCredentialsError:
            raise Exception("AWS credentials not found. Please configure your credentials.")
    
    def get_required_policies(self) -> List[str]:
        """Return the list of required managed policies."""
        return [
            'AmazonBedrockFullAccess',
            'AmazonRedshiftQueryEditor', 
            'AmazonS3FullAccess',
            'AmazonSageMakerFullAccess',
            'AWSLambda_FullAccess',
            'AWSStepFunctionsFullAccess',
            'IAMFullAccess',
            'AWSCodeBuildAdminAccess'
        ]
    
    def get_notebook_instance_role(self, notebook_instance_name: str) -> str:
        """Get the IAM role ARN for a SageMaker notebook instance."""
        try:
            response = self.sagemaker_client.describe_notebook_instance(
                NotebookInstanceName=notebook_instance_name
            )
            return response['RoleArn']
        except ClientError as e:
            if e.response['Error']['Code'] == 'ValidationException':
                raise Exception(f"Notebook instance '{notebook_instance_name}' not found")
            else:
                raise Exception(f"Error retrieving notebook instance: {e}")
    
    def extract_role_name_from_arn(self, role_arn: str) -> str:
        """Extract role name from ARN."""
        return role_arn.split('/')[-1]
    
    def get_attached_managed_policies(self, role_name: str) -> List[str]:
        """Get all managed policies attached to a role."""
        try:
            paginator = self.iam_client.get_paginator('list_attached_role_policies')
            attached_policies = []
            
            for page in paginator.paginate(RoleName=role_name):
                for policy in page['AttachedPolicies']:
                    attached_policies.append(policy['PolicyName'])
            
            return attached_policies
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchEntity':
                raise Exception(f"Role '{role_name}' not found")
            else:
                raise Exception(f"Error retrieving attached policies: {e}")
    
    def check_single_role_policies(self, role_name: str):
        """Check if a single role has all required managed policies. Throws exception if missing."""
        attached_policies = self.get_attached_managed_policies(role_name)

        # Check if role has AdministratorAccess (which grants all permissions)
        if 'AdministratorAccess' in attached_policies:
            return
        
        required_policies = self.get_required_policies()
        missing_policies = [policy for policy in required_policies if policy not in attached_policies]
        
        if missing_policies:
            raise Exception(f"Role '{role_name}' is missing required policies: {', '.join(missing_policies)}")
    
    def attach_healthomics_policy(self, role_name: str):
        """Attach a custom policy for AWS HealthOmics permissions to the role."""
        policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "omics:*"
                    ],
                    "Resource": "*"
                }
            ]
        }
        
        policy_name = "HealthOmicsFullAccess"
        
        try:
            # Create the policy
            self.iam_client.create_policy(
                PolicyName=policy_name,
                PolicyDocument=json.dumps(policy_document),
                Description="Full access to AWS HealthOmics services"
            )
        except ClientError as e:
            if e.response['Error']['Code'] != 'EntityAlreadyExists':
                raise Exception(f"Error creating policy: {e}")
        
        # Get account ID for policy ARN
        sts_client = boto3.client('sts')
        account_id = sts_client.get_caller_identity()['Account']
        policy_arn = f"arn:aws:iam::{account_id}:policy/{policy_name}"
        
        try:
            # Attach the policy to the role
            self.iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn=policy_arn
            )
        except ClientError as e:
            if e.response['Error']['Code'] != 'EntityAlreadyExists':
                raise Exception(f"Error attaching policy: {e}")
    
    def check_policies(self, role_arn: str = None):
        """
        Check if role(s) have all required managed policies. Throws exception if any are missing.
        
        Args:
            role_arn: Direct IAM role ARN (str)
            
        Raises:
            Exception: If any required policies are missing from the role
        """
        # Handle single role ARN
        if isinstance(role_arn, str):
            role_name = self.extract_role_name_from_arn(role_arn)
            self.check_single_role_policies(role_name)
            return
                
        raise ValueError("role_arn must be a string or list of strings")