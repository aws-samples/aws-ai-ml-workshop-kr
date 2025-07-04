import json
import boto3
import requests
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import traceback

@dataclass
class ErrorContext:
    error_type: str
    error_message: str
    state_name: str
    execution_id: str
    repository: str
    pr_id: str
    stack_trace: str
    timestamp: str

class ErrorAnalyzer:
    ERROR_CATEGORIES = {
        'API_ERROR': [
            'RequestException',
            'HTTPError',
            'ConnectionError',
            'Timeout'
        ],
        'AUTHENTICATION_ERROR': [
            'AuthenticationError',
            'TokenExpired',
            'Unauthorized',
            '401',
            '403'
        ],
        'RATE_LIMIT_ERROR': [
            'RateLimitExceeded',
            'TooManyRequests',
            '429'
        ],
        'VALIDATION_ERROR': [
            'ValueError',
            'ValidationError',
            'InvalidParameter'
        ],
        'RESOURCE_ERROR': [
            'ResourceNotFound',
            'NotFound',
            '404'
        ]
    }

    @classmethod
    def categorize_error(cls, error_message: str) -> str:
        """에러 메시지를 카테고리로 분류"""
        error_message_lower = error_message.lower()
        
        for category, patterns in cls.ERROR_CATEGORIES.items():
            if any(pattern.lower() in error_message_lower for pattern in patterns):
                return category
                
        return 'UNKNOWN_ERROR'

    @classmethod
    def is_retriable(cls, error_category: str) -> bool:
        """에러가 재시도 가능한지 판단"""
        retriable_categories = {
            'API_ERROR',
            'RATE_LIMIT_ERROR',
            'RESOURCE_ERROR'
        }
        return error_category in retriable_categories

class ErrorNotifier:
    def __init__(self):
        self.secrets = boto3.client('secretsmanager')
        self.ssm = boto3.client('ssm')
        self._load_config()

    def _load_config(self):
        """Slack 설정 로드"""
        try:
            secret_response = self.secrets.get_secret_value(
                SecretId='/pr-reviewer/tokens/slack'
            )
            secret_data = json.loads(secret_response['SecretString'])
            
            param_response = self.ssm.get_parameter(
                Name='/pr-reviewer/config/slack_channel',
                WithDecryption=True
            )
            
            self.slack_token = secret_data['token']
            self.slack_channel = param_response['Parameter']['Value']
            
        except Exception as e:
            print(f"Error loading configuration: {e}")
            raise

    def send_error_notification(self, error_context: ErrorContext) -> bool:
        """Slack으로 에러 알림 전송"""
        error_category = ErrorAnalyzer.categorize_error(error_context.error_message)
        is_retriable = ErrorAnalyzer.is_retriable(error_category)
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"❌ Code Review Error: {error_category}"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Repository:*\n{error_context.repository}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*PR ID:*\n{error_context.pr_id}"
                    }
                ]
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*State:*\n{error_context.state_name}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Execution ID:*\n{error_context.execution_id}"
                    }
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Error Message:*\n```{error_context.error_message}```"
                }
            }
        ]
        
        # 스택 트레이스가 있는 경우 추가
        if error_context.stack_trace:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Stack Trace:*\n```{error_context.stack_trace[:1000]}```"  # Slack 제한
                }
            })
        
        # 재시도 가능 여부 표시
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"Retriable: {'✅' if is_retriable else '❌'} | Timestamp: {error_context.timestamp}"
                }
            ]
        })
        
        try:
            response = requests.post(
                "https://slack.com/api/chat.postMessage",
                headers={
                    "Authorization": f"Bearer {self.slack_token}",
                    "Content-Type": "application/json"
                },
                json={
                    "channel": self.slack_channel,
                    "blocks": blocks,
                    "text": f"Error in Code Review: {error_category}"  # 폴백 텍스트
                }
            )
            response.raise_for_status()
            return True
            
        except Exception as e:
            print(f"Error sending Slack notification: {e}")
            return False

class ErrorLogger:
    def __init__(self):
        self.cloudwatch = boto3.client('cloudwatch')
        self.logs = boto3.client('logs')
        self.log_group = '/aws/pr-reviewer/errors'

    def log_error(self, error_context: ErrorContext):
        """에러 정보를 CloudWatch에 기록"""
        try:
            # 로그 스트림 생성 또는 존재 확인
            timestamp = int(datetime.now().timestamp() * 1000)
            stream_name = f"{error_context.execution_id}-{timestamp}"
            
            try:
                self.logs.create_log_stream(
                    logGroupName=self.log_group,
                    logStreamName=stream_name
                )
            except self.logs.exceptions.ResourceAlreadyExistsException:
                pass
            
            # 로그 이벤트 작성
            self.logs.put_log_events(
                logGroupName=self.log_group,
                logStreamName=stream_name,
                logEvents=[
                    {
                        'timestamp': timestamp,
                        'message': json.dumps({
                            'error_type': error_context.error_type,
                            'error_message': error_context.error_message,
                            'state_name': error_context.state_name,
                            'execution_id': error_context.execution_id,
                            'repository': error_context.repository,
                            'pr_id': error_context.pr_id,
                            'stack_trace': error_context.stack_trace
                        })
                    }
                ]
            )
            
            # CloudWatch 메트릭 업데이트
            error_category = ErrorAnalyzer.categorize_error(error_context.error_message)
            self.cloudwatch.put_metric_data(
                Namespace='PRReviewer',
                MetricData=[
                    {
                        'MetricName': 'Errors',
                        'Value': 1,
                        'Unit': 'Count',
                        'Dimensions': [
                            {
                                'Name': 'ErrorCategory',
                                'Value': error_category
                            },
                            {
                                'Name': 'StateName',
                                'Value': error_context.state_name
                            }
                        ]
                    }
                ]
            )
            
        except Exception as e:
            print(f"Error logging to CloudWatch: {e}")

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda 핸들러"""
    try:
        # 에러 컨텍스트 생성
        error_context = ErrorContext(
            error_type=event.get('error', {}).get('Error', 'Unknown'),
            error_message=event.get('error', {}).get('Cause', 'Unknown error occurred'),
            state_name=event.get('state_name', 'Unknown'),
            execution_id=event.get('execution_id', 'Unknown'),
            repository=event.get('repository', 'Unknown'),
            pr_id=event.get('pr_id', 'Unknown'),
            stack_trace=event.get('error', {}).get('Stack', ''),
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        )
        
        # 에러 로깅
        logger = ErrorLogger()
        logger.log_error(error_context)
        
        # Slack 알림 전송
        notifier = ErrorNotifier()
        notification_sent = notifier.send_error_notification(error_context)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Error handled successfully',
                'notification_sent': notification_sent,
                'error_category': ErrorAnalyzer.categorize_error(error_context.error_message),
                'is_retriable': ErrorAnalyzer.is_retriable(
                    ErrorAnalyzer.categorize_error(error_context.error_message)
                )
            })
        }
        
    except Exception as e:
        print(f"Error in error handler: {e}")
        traceback.print_exc()
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Error handling failed',
                'details': str(e)
            })
        }