# AWS MCP Sandbox

AWS 기반 서버리스 코드 실행 샌드박스를 위한 MCP (Model Context Protocol) 서버

## 아키텍처

```
Claude Code → MCP Server → API Gateway → Lambda → Fargate Tasks
                                          ↓
                                   DynamoDB (세션 관리)
                                   EFS (영구 저장)
```

## 주요 기능

- **세션별 격리**: 각 세션마다 독립된 Fargate 태스크 실행
- **상태 유지**: 동일 세션에서 Python 변수, 설치된 패키지 유지
- **서버리스**: 사용하지 않을 때는 과금 없음
- **15분+ 실행**: Lambda 제한 없이 긴 작업 실행 가능
- **동적 패키지 설치**: 런타임에 pip install, apt install 지원

## 구성 요소

### 1. MCP Server (`mcp-server/`)
- Claude Code와 연동되는 MCP 프로토콜 서버
- python_execute, bash_execute 도구 제공

### 2. Fargate Runtime (`fargate-runtime/`)
- Docker 컨테이너로 실행되는 코드 실행 환경
- HTTP API로 코드 실행 요청 처리

### 3. AWS Infrastructure (`aws-infrastructure/`)
- CloudFormation 템플릿
- Lambda 세션 관리자
- DynamoDB, ECS, API Gateway 설정

## 설치 및 실행

```bash
# 1. AWS 인프라 배포
cd aws-infrastructure
./deploy.sh

# 2. Docker 이미지 빌드 및 푸시
cd ../docker
./build-and-push.sh

# 3. MCP 서버 실행
cd ../mcp-server
pip install -r requirements.txt
python server.py
```

## MCP 클라이언트 설정

Claude Desktop 설정에 추가:

```json
{
  "mcpServers": {
    "aws-code-sandbox": {
      "command": "python",
      "args": ["/path/to/setup/aws-mcp-sandbox/mcp-server/server.py"],
      "env": {
        "AWS_REGION": "us-east-1",
        "SESSION_MANAGER_URL": "https://your-api-gateway-url"
      }
    }
  }
}
```