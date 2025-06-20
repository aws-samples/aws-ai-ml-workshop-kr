#!/bin/bash

# OpenSearch 환경변수 설정 및 MCP 서버 실행 스크립트

echo "OpenSearch 환경변수를 설정하고 있습니다..."

# Python 스크립트를 통해 환경변수 값 가져오기
eval $(python3 -c "
import os
import sys
import boto3

# 모듈 경로 추가
sys.path.append('..')
from utils.ssm import parameter_store

region = boto3.Session().region_name
pm = parameter_store(region)

opensearch_url = pm.get_params(key='opensearch_domain_endpoint', enc=False)
opensearch_username = pm.get_params(key='opensearch_user_id', enc=False)
opensearch_password = pm.get_params(key='opensearch_user_password', enc=True)

print(f'export OPENSEARCH_URL=\"{opensearch_url}\"')
print(f'export OPENSEARCH_USERNAME=\"{opensearch_username}\"')
print(f'export OPENSEARCH_PASSWORD=\"{opensearch_password}\"')
")

# 환경변수가 제대로 설정되었는지 확인
if [ -z "$OPENSEARCH_URL" ] || [ -z "$OPENSEARCH_USERNAME" ] || [ -z "$OPENSEARCH_PASSWORD" ]; then
    echo "❌ 환경변수 설정에 실패했습니다."
    echo "OPENSEARCH_URL: $OPENSEARCH_URL"
    echo "OPENSEARCH_USERNAME: $OPENSEARCH_USERNAME"
    echo "OPENSEARCH_PASSWORD: [설정됨]"
    exit 1
fi

echo "✅ 환경변수가 성공적으로 설정되었습니다."
echo "OPENSEARCH_URL: $OPENSEARCH_URL"
echo "OPENSEARCH_USERNAME: $OPENSEARCH_USERNAME"
echo "OPENSEARCH_PASSWORD: [설정됨]"

echo ""
echo "MCP OpenSearch 서버를 실행합니다..."

# MCP 서버 실행
python -m mcp_server_opensearch