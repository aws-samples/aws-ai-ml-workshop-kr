# AWS MCP Sandbox 빠른 시작 가이드

AWS 기반 서버리스 MCP 샌드박스를 설정하고 테스트하는 단계별 가이드입니다.

## 📋 사전 요구사항

### 필수 도구
- [Docker](https://www.docker.com/) - 컨테이너 빌드 및 테스트
- [AWS CLI](https://aws.amazon.com/cli/) - AWS 리소스 관리
- [Python 3.9+](https://www.python.org/) - MCP 서버 실행
- [curl](https://curl.se/) - API 테스트

### AWS 자격증명 설정
```bash
aws configure
# AWS Access Key ID, Secret Access Key, Region 입력
```

## 🚀 1단계: 로컬 테스트

### Docker 이미지 빌드
```bash
cd setup/aws-mcp-sandbox/docker
./build-and-push.sh
```

### 로컬 컨테이너 테스트
```bash
cd ..
python test_local.py
```

예상 출력:
```
🚀 AWS MCP Sandbox 로컬 테스트 시작
🐳 로컬 Docker 컨테이너 시작 중...
✅ Docker 컨테이너 시작됨
🔍 헬스체크 대기 중...
✅ 헬스체크 성공
...
🎉 모든 테스트 완료!
```

## 🏗️ 2단계: AWS 인프라 배포

### CloudFormation 스택 배포
```bash
cd aws-infrastructure/scripts
./deploy.sh
```

배포 과정에서 다음이 생성됩니다:
- VPC 및 네트워킹 리소스
- ECS Fargate 클러스터
- DynamoDB 세션 테이블
- API Gateway 및 Lambda 함수
- Application Load Balancer
- ECR 리포지토리

### 환경 변수 설정
배포 완료 후 자동으로 생성된 `.env` 파일 확인:
```bash
cat ../../mcp-server/.env
```

## 🐳 3단계: Docker 이미지 푸시

### ECR에 이미지 업로드
```bash
cd ../../docker
./build-and-push.sh
```

성공 시 출력:
```
✅ ECR 로그인 성공
✅ Docker 이미지 빌드 완료
✅ 이미지 푸시 완료
```

## 🖥️ 4단계: MCP 서버 테스트

### 의존성 설치 및 서버 실행
```bash
cd ../mcp-server
pip install -r requirements.txt
python server.py
```

### 별도 터미널에서 간단한 테스트
```bash
# 세션 생성 테스트
curl -X POST "$API_GATEWAY_URL/session" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test-session-1", "action": "get_or_create"}'
```

## 🔧 5단계: Claude Desktop 연동

### MCP 설정 파일 업데이트
Claude Desktop 설정 파일에 추가:

```json
{
  "mcpServers": {
    "aws-code-sandbox": {
      "command": "python",
      "args": ["/absolute/path/to/setup/aws-mcp-sandbox/mcp-server/server.py"],
      "env": {
        "AWS_REGION": "us-west-2",
        "SESSION_MANAGER_URL": "https://your-api-gateway-url.execute-api.us-west-2.amazonaws.com/prod"
      }
    }
  }
}
```

## 📖 사용 예제

### 기본 사용법
```python
# Claude Code에서 사용
python_execute(session_id="my-session", code="print('Hello World!')")
bash_execute(session_id="my-session", command="ls -la")
```

### 패키지 설치 및 사용
```python
# 패키지 설치
python_execute(session_id="my-session", code="!pip install pandas numpy")

# 설치된 패키지 사용
python_execute(session_id="my-session", code="""
import pandas as pd
import numpy as np

data = np.random.randn(100, 3)
df = pd.DataFrame(data, columns=['A', 'B', 'C'])
print(df.head())
""")
```

### 세션 상태 관리
```python
# 세션 상태 확인
session_status(session_id="my-session")

# 세션 리셋
reset_session(session_id="my-session")
```

## 🔍 문제 해결

### 일반적인 문제들

#### 1. Docker 이미지 빌드 실패
```bash
# Docker 데몬 상태 확인
sudo systemctl status docker

# Docker 로그 확인
docker logs mcp-sandbox-test
```

#### 2. AWS 권한 오류
```bash
# IAM 정책 확인
aws sts get-caller-identity
aws iam get-user
```

#### 3. Fargate 태스크 시작 실패
```bash
# ECS 클러스터 상태 확인
aws ecs describe-clusters --clusters mcp-sandbox-cluster

# 태스크 로그 확인
aws logs get-log-events --log-group-name /aws/ecs/mcp-sandbox --log-stream-name [stream-name]
```

#### 4. API Gateway 연결 문제
```bash
# API Gateway 엔드포인트 테스트
curl -X POST "$API_GATEWAY_URL/session" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test", "action": "get_or_create"}'
```

### 로그 확인
- **Fargate 로그**: CloudWatch Logs `/aws/ecs/mcp-sandbox`
- **Lambda 로그**: CloudWatch Logs `/aws/lambda/mcp-sandbox-session-manager`
- **API Gateway 로그**: CloudWatch Logs (활성화 필요)

## 📊 모니터링

### CloudWatch 대시보드
AWS Console에서 다음 메트릭 모니터링:
- ECS 태스크 수 및 상태
- Lambda 실행 시간 및 오류율
- DynamoDB 읽기/쓰기 단위
- API Gateway 요청 수 및 지연시간

### 비용 모니터링
주요 비용 요소:
- **Fargate**: vCPU/메모리 사용 시간
- **Lambda**: 실행 횟수 및 시간
- **DynamoDB**: 요청 단위 및 저장소
- **API Gateway**: API 호출 수
- **ALB**: 시간당 요금 및 LCU

## 🧹 정리

### 리소스 삭제
```bash
# CloudFormation 스택 삭제
aws cloudformation delete-stack --stack-name mcp-sandbox-infrastructure

# ECR 이미지 삭제
aws ecr batch-delete-image --repository-name mcp-sandbox --image-ids imageTag=latest
```

### 로컬 정리
```bash
# Docker 이미지 정리
docker rmi mcp-sandbox:latest
docker system prune -f
```

## 📚 추가 정보

- [MCP Protocol Specification](https://spec.modelcontextprotocol.io/)
- [AWS Fargate 문서](https://docs.aws.amazon.com/fargate/)
- [Claude Code 문서](https://docs.anthropic.com/en/docs/claude-code)

## 🤝 기여하기

버그 리포트나 기능 요청은 GitHub Issues를 통해 제출해 주세요.

---

**참고**: 이 시스템은 코드 실행을 위한 격리된 환경을 제공하지만, 프로덕션 환경에서는 추가적인 보안 검토가 필요할 수 있습니다.