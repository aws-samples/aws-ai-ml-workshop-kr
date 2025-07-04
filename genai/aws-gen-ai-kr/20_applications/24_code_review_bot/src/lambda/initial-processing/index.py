import json
import os
import hmac
import hashlib
import boto3
from typing import Dict, Any, Optional
from enum import Enum
import base64

class RepositoryType(Enum):
    GITHUB = "github"
    GITLAB = "gitlab"
    BITBUCKET = "bitbucket"

class WebhookProcessor:
    def __init__(self):
        self.ssm = boto3.client('ssm')
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Parameter Store에서 설정 로드"""
        config = {}
        try:
            # 기본 설정 로드
            response = self.ssm.get_parameters_by_path(
                Path='/pr-reviewer/config/',
                Recursive=True,
                WithDecryption=True
            )
            
            for param in response['Parameters']:
                # 파라미터 이름에서 마지막 부분만 추출
                name = param['Name'].split('/')[-1]
                config[name] = param['Value']
                   
        except Exception as e:
            print(f"Error loading config: {e}")
            raise

        return config

    def process_github_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """GitHub webhook 페이로드 처리"""
        return {
            'repository_type': RepositoryType.GITHUB.value,
            'repository': payload['repository']['full_name'],
            'pr_url': payload['pull_request']['html_url'],
            'pr_id': str(payload['pull_request']['number']),
            'title': payload['pull_request']['title'],
            'author': payload['pull_request']['user']['login'],
            'base_branch': payload['pull_request']['base']['ref'],
            'head_branch': payload['pull_request']['head']['ref']
        }

    def process_gitlab_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """GitLab webhook 페이로드 처리"""
        return {
            'repository_type': RepositoryType.GITLAB.value,
            'repository': payload['project']['path_with_namespace'],
            'pr_url': payload['object_attributes']['url'],
            'pr_id': str(payload['object_attributes']['iid']),
            'title': payload['object_attributes']['title'],
            'author': payload['user']['username'],
            'base_branch': payload['object_attributes']['target_branch'],
            'head_branch': payload['object_attributes']['source_branch']
        }

    def process_bitbucket_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Bitbucket webhook 페이로드 처리"""
        return {
            'repository_type': RepositoryType.BITBUCKET.value,
            'repository': payload['repository']['full_name'],
            'pr_url': payload['pullrequest']['links']['html']['href'],
            'pr_id': str(payload['pullrequest']['id']),
            'title': payload['pullrequest']['title'],
            'author': payload['pullrequest']['author']['display_name'],
            'base_branch': payload['pullrequest']['destination']['branch']['name'],
            'head_branch': payload['pullrequest']['source']['branch']['name']
        }

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda 핸들러"""
    processor = WebhookProcessor()
    
    try:
        # API Gateway에서 전달된 데이터 파싱
        body = event.get('body', {})
        
        # 저장소 타입 감지
        repo_type = processor.config['repo_type']

        if not repo_type:
            raise ValueError("Unsupported repository type")
            
        # 저장소 타입별 페이로드 처리
        if repo_type == RepositoryType.GITHUB.value:
            result = processor.process_github_payload(body)
        elif repo_type == RepositoryType.GITLAB.value:
            result = processor.process_gitlab_payload(body)
        elif repo_type == RepositoryType.BITBUCKET.value:
            result = processor.process_bitbucket_payload(body)

        # 설정 정보 추가
        result.update({
            'config': {
                'aws_region': processor.config['aws_region'],
                'model': processor.config['model'],
                'max_tokens': int(processor.config['max_tokens']),
                'temperature': float(processor.config['temperature']),
                'slack_notification': processor.config['slack_notification'],
                'slack_channel': processor.config['slack_channel']
            }
        })
        
        return {
            'statusCode': 200,
            'body': {
                'message': 'Successfully processed webhook',
                'data': result
            }
        }
        
    except ValueError as e:
        return {
            'statusCode': 400,
            'body': {
                'error': str(e)
            }
        }
        
    except Exception as e:
        print(f"Error processing webhook: {e}")
        return {
            'statusCode': 500,
            'body': {
                'error': 'Internal server error'
            }
        }