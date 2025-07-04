import json
import boto3
import requests
from typing import Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class PRDetails:
    repository_type: str
    repository: str
    pr_id: str
    comment: str

class CommentPoster(ABC):
    def __init__(self, credentials: Dict[str, str]):
        self.credentials = credentials

    @abstractmethod
    def post_comment(self, pr_details: PRDetails) -> bool:
        pass

class GitHubCommentPoster(CommentPoster):
    def post_comment(self, pr_details: PRDetails) -> bool:
        headers = {
            "Authorization": f"Bearer {self.credentials['access_token']}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        
        url = f"https://api.github.com/repos/{pr_details.repository}/issues/{pr_details.pr_id}/comments"
        
        try:
            response = requests.post(
                url,
                headers=headers,
                json={"body": pr_details.comment}
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error posting GitHub comment: {e}")
            if response := getattr(e, 'response', None):
                print(f"Response: {response.text}")
            return False

class GitLabCommentPoster(CommentPoster):
    def post_comment(self, pr_details: PRDetails) -> bool:
        headers = {
            "PRIVATE-TOKEN": self.credentials['access_token']
        }
        
        encoded_repo = requests.utils.quote(pr_details.repository, safe='')
        url = f"{self.credentials['gitlab_url']}/api/v4/projects/{encoded_repo}/merge_requests/{pr_details.pr_id}/notes"
        
        try:
            response = requests.post(
                url,
                headers=headers,
                json={"body": pr_details.comment}
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error posting GitLab comment: {e}")
            if response := getattr(e, 'response', None):
                print(f"Response: {response.text}")
            return False

class BitbucketCommentPoster(CommentPoster):
    def post_comment(self, pr_details: PRDetails) -> bool:
        headers = {
            "Authorization": f"Bearer {self.credentials['access_token']}",
            "Accept": "application/vnd.github+json"
        }
        
        workspace, repo_slug = pr_details.repository.split('/')
        url = f"https://api.bitbucket.org/2.0/repositories/{workspace}/{repo_slug}/pullrequests/{pr_details.pr_id}/comments"
        
        try:
            response = requests.post(
                url,
                headers=headers,
                json={"content": {"raw": pr_details.comment}}
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error posting Bitbucket comment: {e}")
            if response := getattr(e, 'response', None):
                print(f"Response: {response.text}")
            return False

class CommentPosterFactory:
    @staticmethod
    def create_poster(repository_type: str, credentials: Dict[str, str]) -> CommentPoster:
        poster_map = {
            'github': GitHubCommentPoster,
            'gitlab': GitLabCommentPoster,
            'bitbucket': BitbucketCommentPoster
        }
        
        poster_class = poster_map.get(repository_type.lower())
        if not poster_class:
            raise ValueError(f"Unsupported repository type: {repository_type}")
            
        return poster_class(credentials)

def normalize_body(body: Dict[str, Any]) -> Dict[str, Any]:
    """이벤트 body 정규화"""
    if isinstance(body, str):
        return json.loads(body)
    return body

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda 핸들러"""
    try:
        # 이벤트 body 정규화
        body = normalize_body(event.get('body', {}))
        
        # PR 정보 추출
        pr_info = body.get('pr_details', {})
        repository_type = pr_info.get('repository_type')
        repository = pr_info.get('repository')
        pr_id = pr_info.get('pr_id')
        
        # 전체 리포트를 코멘트로 사용
        full_comment = body.get('markdown_report', '')
        
        if not all([repository_type, repository, pr_id, full_comment]):
            raise ValueError("Missing required parameters")
        
        # Secrets Manager에서 인증 정보 로드
        secrets = boto3.client('secretsmanager')
        try:
            response = secrets.get_secret_value(
                SecretId=f'/pr-reviewer/tokens/{repository_type.lower()}'
            )
            credentials = json.loads(response['SecretString'])
        except Exception as e:
            print(f"Error loading credentials: {e}")
            raise
        
        # PR 상세 정보 생성
        pr_details = PRDetails(
            repository_type=repository_type,
            repository=repository,
            pr_id=pr_id,
            comment=full_comment
        )
        
        # 코멘트 포스터 생성 및 코멘트 작성
        poster = CommentPosterFactory.create_poster(repository_type, credentials)
        success = poster.post_comment(pr_details)
        
        if not success:
            raise Exception("Failed to post comment")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Successfully posted PR comment',
                'repository': repository,
                'pr_id': pr_id,
                'stats': {
                    'primary_files': body.get('summary', {}).get('total_primary_files', 0),
                    'reference_files': body.get('summary', {}).get('total_reference_files', 0),
                    'total_issues': body.get('summary', {}).get('total_issues', 0)
                }
            })
        }
        
    except ValueError as e:
        return {
            'statusCode': 400,
            'body': json.dumps({
                'error': str(e)
            })
        }
        
    except Exception as e:
        print(f"Error posting PR comment: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Internal server error'
            })
        }