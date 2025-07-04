# Amazon Bedrock PR Review Bot

Amazon Bedrock을 활용해 생성된 Pull request에 대해 자동으로 Code review를 제공하는 애플리케이션입니다.
AWS CDK로 구축되었으며, GitHub, GitLab, Bitbucket과 통합되어 Slack 알림과 함께 상세한 코드 리뷰를 제공합니다.

PR에 여러 파일이 포함될 때 Amazon Bedrock PR Review Bot은 파일 간의 관계를 분석하여 연관된 파일들을 함께 검토함으로써 더 정확하고 의미 있는 리뷰를 제공합니다. 

또한 이전 리뷰와 현재 리뷰를 비교하여 해결된 이슈, 새로운 이슈, 지속되는 이슈를 추적하고 품질 개선의 진행 상황을 시각화합니다.


```mermaid
graph TD
    root[Project Root] --> auth[auth/]
    root --> models[models/]
    root --> utils[utils/]
    
    subgraph "Chunk 1: Auth Module"
        auth --> user[user.py]
        auth --> user_test[user_test.py]
        auth --> auth_py[auth.py]
        
        classDef authFiles fill:#d0e0ff,stroke:#333,stroke-width:2px
        class user,user_test,auth_py authFiles
    end
    
    subgraph "Chunk 2: Data Layer"
        models --> models_py[models.py]
        models --> db[database.py]
        models --> migrations[migrations.py]
        
        classDef modelFiles fill:#ffe0d0,stroke:#333,stroke-width:2px
        class models_py,db,migrations modelFiles
    end
    
    subgraph "Chunk 3: Utilities"
        utils --> config[config.py]
        utils --> utils_py[utils.py]
        utils --> constants[constants.py]
        
        classDef utilFiles fill:#d0ffd0,stroke:#333,stroke-width:2px
        class config,utils_py,constants utilFiles
    end

    classDef folder fill:#f9f9f9,stroke:#333,stroke-width:2px
    class root,auth,models,utils folder

    %% 관계 표시
    user -.->|imports| auth_py
    user_test -.->|imports| user
    user_test -.->|imports| auth_py
    models_py -.->|imports| db
    migrations -.->|imports| db
    utils_py -.->|imports| constants
    config -.->|imports| utils_py
```

## 아키텍처

![아키텍처 다이어그램](docs/img/architecture.png)

사용된 AWS 서비스:
- Amazon API Gateway: 웹훅 엔드포인트
- AWS Step Functions: 워크플로우 오케스트레이션
- AWS Lambda: 서버리스 컴퓨팅
- Amazon Bedrock: AI 기반 코드 리뷰
- Amazon DynamoDB: 리뷰 결과 저장 및 이력 관리
- AWS Secrets Manager와 Parameter Store: 설정 관리
- Amazon CloudWatch: 모니터링 및 로깅

## 사전 요구사항

- AWS 계정 (적절한 권한 필요) / 현재 ap-northeast-2 리전에서만 정상적으로 동작합니다.
- Node.js 18.x 이상
- Python 3.12
- AWS CDK CLI v2.x
- Git
- Amazon Bedrock 접근 권한
- 액세스 토큰:
  - GitHub/GitLab/Bitbucket (사용하는 저장소에 따라)
  - Slack (알림용)

## 설치 방법

1. 저장소 클론:
```bash
git clone https://github.com/muylucir/amazon-bedrock-pr-review-bot.git
cd amazon-bedrock-pr-review-bot
```

2. 의존성 설치:
```bash
npm install
```

3. Lambda Layer 생성를 위한 준비
```bash
chmod +x setup-layer.sh
./setup-layer.sh
```

setup-layer.sh는 Python 3.12버전을 사용하는 것을 명시하도록 pip3.12 명령어가 지정되어 있습니다. 이미 Python 3.12을 사용하고 있다면 아래와 같이 수정합니다. 

```bash
vi setup-layer.sh

#!/bin/bash
# setup-layers.sh

# Requests Layer
echo "Setting up requests layer..."
mkdir -p layer/requests/python
cd layer/requests/python
pip3 install requests -t .
cd ../../..

# Networkx Layer
echo "Setting up networkx layer..."
mkdir -p layer/networkx/python
cd layer/networkx/python
pip3 install networkx numpy -t .
cd ../../..

# Clean up unnecessary files
find layer -type d -name "__pycache__" -exec rm -rf {} +
find layer -type d -name "*.dist-info" -exec rm -rf {} +
find layer -type d -name "*.egg-info" -exec rm -rf {} +
find layer -type f -name "*.pyc" -delete

echo "Layer setup complete!"
```


## 배포

1. AWS 환경 부트스트랩 (처음 한 번만):
```bash
cdk bootstrap
```

2. 스택 배포:
```bash
cdk deploy
```

3. Secret Manager에 Token을 실제 값으로 Update (해당하는 Repository만 실행하면 됩니다.)
```bash
# GitHub : Personal access tokens (classic)만 지원합니다.
aws secretsmanager update-secret \
  --secret-id /pr-reviewer/tokens/github \
  --secret-string '{"access_token":"input-your-actual-token"}'

# GitLab
aws secretsmanager update-secret \
  --secret-id /pr-reviewer/tokens/gitlab \
  --secret-string '{"access_token":"input-your-actual-token"}'

# Bitbucket
aws secretsmanager update-secret \
  --secret-id /pr-reviewer/tokens/bitbucket \
  --secret-string '{"access_token":"input-your-actual-token"}'

# Slack : Slack 알림이 필요한 경우에 설정 합니다.
aws secretsmanager update-secret \
  --secret-id /pr-reviewer/tokens/slack \
  --secret-string '{"access_token":"xoxb-your-actual-token"}'
```

4. 사용하는 저장소의 종류에 따라 Parameter Store에 설정 값을 입력합니다.
```bash
aws ssm put-parameter --name /pr-reviewer/config/repo_type --value <github | gitlab | bitbucket> --overwrite
```

## 저장소 설정

### GitHub
1. 저장소 설정으로 이동
2. Webhooks 메뉴로 이동
3. 새 웹훅 추가:
   - Payload URL: 배포 시 받은 WebhookUrl
   - Content type: application/json
   - Events: Pull requests
   - Active: Yes

### GitLab
1. 저장소 설정으로 이동
2. Webhooks 메뉴로 이동
3. 새 웹훅 추가:
   - URL: 배포 시 받은 WebhookUrl
   - Trigger: Merge request events
   - SSL 검증 활성화

### Bitbucket
1. 저장소 설정으로 이동
2. Webhooks 메뉴로 이동
3. 새 웹훅 추가:
   - URL: 배포 시 받은 WebhookUrl
   - Triggers: Pull Request: Created, Updated

## 사용 방법

봇은 자동으로 다음 작업을 수행합니다:
1. 새로운 PR 또는 기존 PR 업데이트 검토
2. 파일 간 관계 분석 및 관련 파일 그룹화
3. Amazon Bedrock을 사용한 코드 변경 분석
4. 이전 리뷰와 비교하여 개선 사항 추적
5. PR에 상세한 리뷰 코멘트 작성
6. (Option) 설정된 Slack 채널로 요약 전송

```bash
# Slack 알림 활성화
aws ssm put-parameter --name /pr-reviewer/config/slack_notification --value "enable" --overwrite

# 알림을 받을 채널 설정
aws ssm put-parameter --name /pr-reviewer/config/slack_channel --value "your-channel-name" --overwrite
```

[샘플 Report 보기](https://github.com/muylucir/amazon-bedrock-pr-review-bot/blob/main/docs/sample_report.md)

## 리뷰 데이터 관리
리뷰 데이터는 DynamoDB에 자동으로 저장되어 다음과 같은 이점을 제공합니다:

- 리뷰 이력 추적: 동일한 PR에 대한 여러 리뷰 결과를 시간순으로 저장
- 이슈 개선 분석: 해결된 이슈, 새로운 이슈, 지속되는 이슈를 비교 분석
- 대용량 데이터 처리: Lambda 함수 간 페이로드 크기 제한 우회

저장된 데이터는 30일 후 TTL(Time-To-Live)에 의해 자동으로 삭제됩니다.

## 마크다운 리포트 저장
리뷰 마크다운 리포트는 별도의 DynamoDB 테이블(PRReviewerReports)에 저장됩니다:

- 리포지토리와 PR ID로 쉽게 조회 가능
- 보고서 내용, 제목, 작성자 등 PR 관련 메타데이터 함께 저장
- 타임스탬프 기반 인덱스로 최신 리포트 조회 지원
- 실행 ID 기반 인덱스로 특정 실행의 리포트 검색 가능


이를 통해 PR 코드 리뷰 보고서의 이력을 쉽게 검색하고 관리할 수 있습니다.

## 모니터링

배포에 포함된 모니터링 도구:
- Lambda 함수용 CloudWatch 대시보드
- Step Functions 실행 이력
- API Gateway 로그
- 오류 추적용 사용자 정의 메트릭

AWS 콘솔에서 접근하거나 필요에 따라 추가 알림을 구성할 수 있습니다.

## 커스터마이제이션

### 리뷰 동작 수정

Bedrock 모델 매개변수를 `lib/constructs/secrets-and-parameters.ts`에서 조정:
```typescript
maxTokens: new ssm.StringParameter(this, 'MaxTokens', {
  parameterName: '/pr-reviewer/config/max_tokens',
  stringValue: String(props.maxTokens || 4096),
  description: 'Maximum tokens for model response'
}),
temperature: new ssm.StringParameter(this, 'Temperature', {
  parameterName: '/pr-reviewer/config/temperature',
  stringValue: String(props.temperature || 0.7),
  description: 'Temperature for model response'
})
```

### 리뷰 형식 수정

리뷰 형식은 ProcessChunk Lambda 함수(`src/lambda/process-chunk/index.py`)에서 사용자 정의할 수 있습니다.

### 새로운 기능 추가

모듈식 아키텍처로 새로운 기능을 쉽게 추가할 수 있습니다:
1. `lib/constructs/lambda.ts`에 새 Lambda 함수 추가
2. `lib/constructs/step-functions.ts`에서 Step Functions 워크플로우 수정
3. `lib/constructs/review-bot-role.ts`에서 관련 IAM 역할 업데이트

## 문제 해결

### 일반적인 문제

1. API Gateway 통합 문제:
   - API Gateway 로그 확인
   - IAM 역할과 권한 확인
   - 웹훅 엔드포인트 수동 테스트

2. Step Functions 오류:
   - 실행 이력 확인
   - 상태 머신 입력 형식 확인
   - Lambda 함수 로그 확인

3. Bedrock 통합 문제:
   - Bedrock 접근 권한 확인
   - 리전에서의 모델 사용 가능 여부 확인
   - 모델 매개변수 검토

### 로깅

`lib/constructs/lambda.ts`에서 LOG_LEVEL 환경 변수를 수정하여 상세 로깅 활성화:
```typescript
environment: {
  POWERTOOLS_SERVICE_NAME: 'pr-reviewer',
  LOG_LEVEL: 'DEBUG'  // 더 자세한 로그를 위해 DEBUG로 변경
}
```

## 보안

이 애플리케이션은 다음과 같은 AWS 보안 모범 사례를 따릅니다:
- 최소 권한 IAM 역할
- 시크릿 관리
- 리전별 API 엔드포인트
- CloudWatch 로깅
- 요청 인증

필요에 따라 보안 설정을 검토하고 조정하세요.


## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다 - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.


---
