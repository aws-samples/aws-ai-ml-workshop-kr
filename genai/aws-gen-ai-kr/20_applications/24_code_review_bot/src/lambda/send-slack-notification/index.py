import json
import boto3
import requests
from typing import Dict, Any, Optional
from dataclasses import dataclass
import time
from botocore.exceptions import ClientError

@dataclass
class SlackConfig:
    token: str
    channel: str
    notification: str
    max_retries: int = 3
    retry_delay: float = 1.0

@dataclass
class ReviewStats:
    primary_files: int
    reference_files: int
    total_issues: int
    duration: float

class SlackNotifier:
    def __init__(self, config: SlackConfig):
        self.config = config
        self.base_url = "https://slack.com/api"
        self.headers = {
            "Authorization": f"Bearer {self.config.token}",
            "Content-Type": "application/json"
        }

    def send_message(self, message: Dict[str, Any]) -> bool:
        """메시지를 Slack으로 전송"""
        url = f"{self.base_url}/chat.postMessage"
        
        # 채널 정보 추가
        message["channel"] = self.config.channel
        
        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(
                    url,
                    headers=self.headers,
                    json=message
                )
                response.raise_for_status()
                
                response_data = response.json()
                if not response_data.get("ok"):
                    raise Exception(f"Slack API error: {response_data.get('error')}")
                
                return True
                
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))  # 지수 백오프
                else:
                    raise
        
        return False

class MessageFormatter:
    @staticmethod
    def format_error_message(error: str) -> Dict[str, Any]:
        """에러 메시지 포맷팅"""
        return {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "❌ Code Review Error"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Error:*\n{error}"
                    }
                }
            ]
        }

    @staticmethod
    def add_review_stats(message: Dict[str, Any], stats: ReviewStats) -> Dict[str, Any]:
        """리뷰 통계 정보 추가"""
        if "blocks" not in message:
            message["blocks"] = []
            
        stats_block = {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        f"*Files:* {stats.primary_files} primary + {stats.reference_files} reference | "
                        f"*Issues:* {stats.total_issues} | "
                        f"*Duration:* {stats.duration:.1f}s"
                    )
                }
            ]
        }
        
        # 통계 블록을 PR 링크 버튼 앞에 삽입
        button_block_index = next(
            (i for i, block in enumerate(message['blocks']) 
             if block.get('type') == 'actions'),
            len(message['blocks'])
        )
        
        message['blocks'].insert(button_block_index, stats_block)
        return message

def get_slack_config(body: Dict[str, Any]) -> Optional[SlackConfig]:
    """Secrets Manager에서 Slack 설정 로드"""
    secrets = boto3.client('secretsmanager')
    try:
        # Slack 토큰 가져오기
        secret_response = secrets.get_secret_value(
            SecretId='/pr-reviewer/tokens/slack'
        )
        secret_data = json.loads(secret_response['SecretString'])
        
        # PR 상세 정보에서 채널 정보 가져오기
        pr_details = body.get('pr_details', {})
        config = pr_details.get('config', {})
        channel = config.get('slack_channel')
        notification = config.get('slack_notification')
        
        if not channel:
            raise ValueError("Slack channel not found in PR details")
        if not secret_data.get('token'):
            raise ValueError("Slack token not found in secrets")
        
        return SlackConfig(
            token=secret_data['token'],
            channel=channel,
            notification=notification
        )
        
    except Exception as e:
        print(f"Error loading Slack configuration: {e}")
        return None

def extract_review_stats(body: Dict[str, Any], event: Dict[str, Any]) -> ReviewStats:
    """리뷰 통계 추출"""
    summary = body.get('summary', {})
    return ReviewStats(
        primary_files=summary.get('total_primary_files', 0),
        reference_files=summary.get('total_reference_files', 0),
        total_issues=summary.get('total_issues', 0),
        duration=float(event.get('duration', 0))
    )

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda 핸들러"""
    config = None
    
    try:
        # 이벤트 body 정규화
        if isinstance(event.get('body'), str):
            body = json.loads(event.get('body', '{}'))
        else:
            body = event.get('body', {})
            
        slack_message = body.get('slack_message')
        
        if not slack_message:
            raise ValueError("Missing Slack message in the input")
        
        # Slack 설정 로드
        config = get_slack_config(body)
        if not config:
            raise ValueError("Failed to load Slack configuration")

        if config.notification == "disable":
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Slack notification is disabled'
                })
            }
        
        # 리뷰 통계 추출
        stats = extract_review_stats(body, event)
        
        # 메시지에 통계 추가
        formatted_message = MessageFormatter.add_review_stats(
            slack_message,
            stats
        )
        
        # Slack 알림 전송
        notifier = SlackNotifier(config)
        success = notifier.send_message(formatted_message)
        
        if not success:
            raise Exception("Failed to send Slack notification")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Successfully sent Slack notification',
                'stats': {
                    'primary_files': stats.primary_files,
                    'reference_files': stats.reference_files,
                    'total_issues': stats.total_issues,
                    'duration': stats.duration
                }
            })
        }
        
    except ValueError as e:
        error_message = MessageFormatter.format_error_message(str(e))
        
        if config:  # 설정이 있는 경우에만 에러 메시지 전송
            try:
                notifier = SlackNotifier(config)
                notifier.send_message(error_message)
            except Exception as slack_error:
                print(f"Error sending error message to Slack: {slack_error}")
        
        return {
            'statusCode': 400,
            'body': json.dumps({
                'error': str(e)
            })
        }
        
    except Exception as e:
        print(f"Error sending Slack notification: {e}")
        
        if config:  # 설정이 있는 경우에만 에러 메시지 전송
            try:
                error_message = MessageFormatter.format_error_message(
                    "Internal error occurred during code review notification"
                )
                notifier = SlackNotifier(config)
                notifier.send_message(error_message)
            except Exception as slack_error:
                print(f"Error sending error message to Slack: {slack_error}")
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }