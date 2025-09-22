#!/usr/bin/env python3
"""
AWS Lambda Session Manager
Fargate 태스크 생성 및 관리를 위한 Lambda 함수
"""

import json
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

import boto3
from botocore.exceptions import ClientError

# AWS 클라이언트 초기화
ecs = boto3.client('ecs')
dynamodb = boto3.resource('dynamodb')
elbv2 = boto3.client('elbv2')

# 환경 변수
CLUSTER_NAME = os.environ['CLUSTER_NAME']
TASK_DEFINITION = os.environ['TASK_DEFINITION']
SUBNET_IDS = os.environ['SUBNET_IDS'].split(',')
SECURITY_GROUP_ID = os.environ['SECURITY_GROUP_ID']
TARGET_GROUP_ARN = os.environ['TARGET_GROUP_ARN']
SESSION_TABLE_NAME = os.environ['SESSION_TABLE_NAME']

# DynamoDB 테이블
table = dynamodb.Table(SESSION_TABLE_NAME)

def lambda_handler(event, context):
    """Lambda 진입점"""
    try:
        print(f"Received event: {json.dumps(event)}")
        
        # HTTP 요청 처리
        if 'httpMethod' in event:
            return handle_http_request(event, context)
        
        # 직접 함수 호출 처리
        action = event.get('action', 'get_or_create')
        session_id = event.get('session_id')
        
        if not session_id:
            return error_response("session_id is required", 400)
        
        if action == 'get_or_create':
            return get_or_create_session(session_id)
        elif action == 'cleanup':
            return cleanup_inactive_sessions()
        elif action == 'stop_session':
            return stop_session(session_id)
        else:
            return error_response(f"Unknown action: {action}", 400)
            
    except Exception as e:
        print(f"Error in lambda_handler: {str(e)}")
        return error_response(f"Internal server error: {str(e)}", 500)

def handle_http_request(event, context):
    """HTTP 요청 처리"""
    method = event['httpMethod']
    path = event.get('path', '')
    
    if method == 'POST' and path == '/session':
        body = json.loads(event.get('body', '{}'))
        session_id = body.get('session_id')
        action = body.get('action', 'get_or_create')
        
        if not session_id:
            return http_response({"error": "session_id is required"}, 400)
        
        if action == 'get_or_create':
            result = get_or_create_session(session_id)
            return http_response(result, 200)
        elif action == 'stop':
            result = stop_session(session_id)
            return http_response(result, 200)
        else:
            return http_response({"error": f"Unknown action: {action}"}, 400)
    
    elif method == 'OPTIONS':
        return http_response({}, 200)
    
    else:
        return http_response({"error": "Method not allowed"}, 405)

def get_or_create_session(session_id: str) -> Dict[str, Any]:
    """세션 조회 또는 생성"""
    try:
        print(f"Getting or creating session: {session_id}")
        
        # 기존 세션 조회
        response = table.get_item(Key={'session_id': session_id})
        
        if 'Item' in response:
            session_data = response['Item']
            task_arn = session_data.get('task_arn')
            
            # 태스크가 여전히 실행 중인지 확인
            if task_arn and is_task_running(task_arn):
                # 마지막 액세스 시간 업데이트
                update_last_access(session_id)
                
                return {
                    'status': 'existing',
                    'session_id': session_id,
                    'endpoint': session_data.get('endpoint'),
                    'task_arn': task_arn,
                    'created_at': session_data.get('created_at')
                }
        
        # 새 세션 생성
        print(f"Creating new session for: {session_id}")
        session_data = create_new_session(session_id)
        
        return {
            'status': 'created',
            'session_id': session_id,
            'endpoint': session_data['endpoint'],
            'task_arn': session_data['task_arn'],
            'created_at': session_data['created_at']
        }
        
    except Exception as e:
        print(f"Error in get_or_create_session: {str(e)}")
        raise

def create_new_session(session_id: str) -> Dict[str, Any]:
    """새로운 Fargate 세션 생성"""
    try:
        # Fargate 태스크 시작
        task_arn = start_fargate_task(session_id)
        
        # 태스크 IP 주소 대기 및 획득
        task_ip = wait_for_task_ip(task_arn)
        endpoint = f"http://{task_ip}:8080"
        
        # ALB에 타겟 등록
        register_target(task_ip)
        
        # DynamoDB에 세션 정보 저장
        current_time = int(time.time())
        ttl_time = current_time + (24 * 3600)  # 24시간 TTL
        
        session_data = {
            'session_id': session_id,
            'task_arn': task_arn,
            'task_ip': task_ip,
            'endpoint': endpoint,
            'status': 'running',
            'created_at': current_time,
            'last_access': current_time,
            'ttl': ttl_time
        }
        
        table.put_item(Item=session_data)
        
        print(f"Created new session: {session_id}, endpoint: {endpoint}")
        
        return session_data
        
    except Exception as e:
        print(f"Error creating new session: {str(e)}")
        raise

def start_fargate_task(session_id: str) -> str:
    """Fargate 태스크 시작"""
    try:
        response = ecs.run_task(
            cluster=CLUSTER_NAME,
            taskDefinition=TASK_DEFINITION,
            launchType='FARGATE',
            networkConfiguration={
                'awsvpcConfiguration': {
                    'subnets': SUBNET_IDS,
                    'securityGroups': [SECURITY_GROUP_ID],
                    'assignPublicIp': 'ENABLED'
                }
            },
            tags=[
                {'key': 'SessionId', 'value': session_id},
                {'key': 'Purpose', 'value': 'mcp-sandbox'},
                {'key': 'CreatedBy', 'value': 'session-manager'}
            ],
            overrides={
                'containerOverrides': [
                    {
                        'name': 'sandbox-container',
                        'environment': [
                            {'name': 'SESSION_ID', 'value': session_id},
                            {'name': 'CREATED_AT', 'value': str(int(time.time()))}
                        ]
                    }
                ]
            }
        )
        
        if response['failures']:
            raise Exception(f"Failed to start task: {response['failures']}")
        
        task_arn = response['tasks'][0]['taskArn']
        print(f"Started Fargate task: {task_arn}")
        
        return task_arn
        
    except Exception as e:
        print(f"Error starting Fargate task: {str(e)}")
        raise

def wait_for_task_ip(task_arn: str, max_wait_time: int = 120) -> str:
    """태스크 IP 주소 대기"""
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        try:
            response = ecs.describe_tasks(
                cluster=CLUSTER_NAME,
                tasks=[task_arn]
            )
            
            if response['tasks']:
                task = response['tasks'][0]
                
                # RUNNING 상태 확인
                if task['lastStatus'] == 'RUNNING':
                    # 네트워크 인터페이스에서 IP 주소 추출
                    attachments = task.get('attachments', [])
                    for attachment in attachments:
                        if attachment['type'] == 'ElasticNetworkInterface':
                            for detail in attachment['details']:
                                if detail['name'] == 'privateIPv4Address':
                                    ip_address = detail['value']
                                    print(f"Task IP address: {ip_address}")
                                    return ip_address
                
                print(f"Task status: {task['lastStatus']}, waiting...")
                
        except Exception as e:
            print(f"Error checking task status: {str(e)}")
        
        time.sleep(5)
    
    raise Exception(f"Task IP not available within {max_wait_time} seconds")

def register_target(task_ip: str):
    """ALB에 타겟 등록"""
    try:
        elbv2.register_targets(
            TargetGroupArn=TARGET_GROUP_ARN,
            Targets=[
                {
                    'Id': task_ip,
                    'Port': 8080
                }
            ]
        )
        print(f"Registered target: {task_ip}")
        
    except Exception as e:
        print(f"Error registering target: {str(e)}")
        # ALB 등록 실패해도 직접 IP 접근은 가능하므로 에러 무시

def is_task_running(task_arn: str) -> bool:
    """태스크 실행 상태 확인"""
    try:
        response = ecs.describe_tasks(
            cluster=CLUSTER_NAME,
            tasks=[task_arn]
        )
        
        if response['tasks']:
            task_status = response['tasks'][0]['lastStatus']
            return task_status in ['PENDING', 'RUNNING']
        
        return False
        
    except Exception as e:
        print(f"Error checking task status: {str(e)}")
        return False

def stop_session(session_id: str) -> Dict[str, Any]:
    """세션 종료"""
    try:
        # DynamoDB에서 세션 정보 조회
        response = table.get_item(Key={'session_id': session_id})
        
        if 'Item' not in response:
            return {'status': 'not_found', 'session_id': session_id}
        
        session_data = response['Item']
        task_arn = session_data.get('task_arn')
        task_ip = session_data.get('task_ip')
        
        # 태스크 중지
        if task_arn:
            try:
                ecs.stop_task(
                    cluster=CLUSTER_NAME,
                    task=task_arn,
                    reason='Session stopped by user'
                )
                print(f"Stopped task: {task_arn}")
            except Exception as e:
                print(f"Error stopping task: {str(e)}")
        
        # ALB에서 타겟 제거
        if task_ip:
            try:
                elbv2.deregister_targets(
                    TargetGroupArn=TARGET_GROUP_ARN,
                    Targets=[{'Id': task_ip, 'Port': 8080}]
                )
                print(f"Deregistered target: {task_ip}")
            except Exception as e:
                print(f"Error deregistering target: {str(e)}")
        
        # DynamoDB에서 세션 삭제
        table.delete_item(Key={'session_id': session_id})
        
        return {
            'status': 'stopped',
            'session_id': session_id,
            'task_arn': task_arn
        }
        
    except Exception as e:
        print(f"Error stopping session: {str(e)}")
        raise

def cleanup_inactive_sessions() -> Dict[str, Any]:
    """비활성 세션 정리"""
    try:
        current_time = int(time.time())
        inactive_threshold = current_time - (30 * 60)  # 30분
        
        # 모든 세션 스캔
        response = table.scan()
        sessions = response.get('Items', [])
        
        cleaned_sessions = []
        
        for session in sessions:
            session_id = session['session_id']
            last_access = session.get('last_access', 0)
            task_arn = session.get('task_arn')
            
            # 30분 이상 비활성이거나 태스크가 중지된 경우
            if last_access < inactive_threshold or not is_task_running(task_arn):
                try:
                    stop_session(session_id)
                    cleaned_sessions.append(session_id)
                    print(f"Cleaned up session: {session_id}")
                except Exception as e:
                    print(f"Error cleaning up session {session_id}: {str(e)}")
        
        return {
            'status': 'completed',
            'cleaned_sessions': cleaned_sessions,
            'total_cleaned': len(cleaned_sessions)
        }
        
    except Exception as e:
        print(f"Error in cleanup: {str(e)}")
        raise

def update_last_access(session_id: str):
    """마지막 액세스 시간 업데이트"""
    try:
        table.update_item(
            Key={'session_id': session_id},
            UpdateExpression='SET last_access = :timestamp',
            ExpressionAttributeValues={
                ':timestamp': int(time.time())
            }
        )
    except Exception as e:
        print(f"Error updating last access: {str(e)}")

def error_response(message: str, status_code: int = 500) -> Dict[str, Any]:
    """에러 응답 생성"""
    return {
        'error': message,
        'status_code': status_code
    }

def http_response(body: Dict[str, Any], status_code: int = 200) -> Dict[str, Any]:
    """HTTP 응답 생성"""
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
            'Access-Control-Allow-Methods': 'GET,POST,PUT,DELETE,OPTIONS'
        },
        'body': json.dumps(body)
    }