#!/bin/bash

# AWS MCP Sandbox 배포 스크립트

set -e

# 기본 설정
PROJECT_NAME="mcp-sandbox"
AWS_REGION="${AWS_REGION:-us-west-2}"
STACK_NAME="${PROJECT_NAME}-infrastructure"

# 색상 출력
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# AWS CLI 확인
check_aws_cli() {
    if ! command -v aws &> /dev/null; then
        echo_error "AWS CLI가 설치되지 않았습니다."
        exit 1
    fi
    
    if ! aws sts get-caller-identity &> /dev/null; then
        echo_error "AWS 자격증명이 구성되지 않았습니다."
        exit 1
    fi
    
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    echo_info "AWS Account ID: $ACCOUNT_ID"
    echo_info "AWS Region: $AWS_REGION"
}

# ECR 리포지토리 생성 (존재하지 않는 경우)
create_ecr_repository() {
    echo_info "ECR 리포지토리 확인 중..."
    
    if ! aws ecr describe-repositories --repository-names $PROJECT_NAME --region $AWS_REGION &> /dev/null; then
        echo_info "ECR 리포지토리 생성 중..."
        aws ecr create-repository \
            --repository-name $PROJECT_NAME \
            --region $AWS_REGION \
            --image-scanning-configuration scanOnPush=true
    else
        echo_info "ECR 리포지토리가 이미 존재합니다."
    fi
}

# CloudFormation 스택 배포
deploy_infrastructure() {
    echo_info "CloudFormation 스택 배포 중..."
    
    aws cloudformation deploy \
        --template-file ../cloudformation.yaml \
        --stack-name $STACK_NAME \
        --parameter-overrides \
            ProjectName=$PROJECT_NAME \
        --capabilities CAPABILITY_NAMED_IAM \
        --region $AWS_REGION
    
    if [ $? -eq 0 ]; then
        echo_info "CloudFormation 스택 배포 완료"
    else
        echo_error "CloudFormation 스택 배포 실패"
        exit 1
    fi
}

# 스택 출력값 가져오기
get_stack_outputs() {
    echo_info "스택 출력값 조회 중..."
    
    API_URL=$(aws cloudformation describe-stacks \
        --stack-name $STACK_NAME \
        --query 'Stacks[0].Outputs[?OutputKey==`APIGatewayURL`].OutputValue' \
        --output text \
        --region $AWS_REGION)
    
    ECR_URI=$(aws cloudformation describe-stacks \
        --stack-name $STACK_NAME \
        --query 'Stacks[0].Outputs[?OutputKey==`ECRRepositoryURI`].OutputValue' \
        --output text \
        --region $AWS_REGION)
    
    ALB_URL=$(aws cloudformation describe-stacks \
        --stack-name $STACK_NAME \
        --query 'Stacks[0].Outputs[?OutputKey==`LoadBalancerURL`].OutputValue' \
        --output text \
        --region $AWS_REGION)
    
    echo_info "API Gateway URL: $API_URL"
    echo_info "ECR Repository URI: $ECR_URI"
    echo_info "Load Balancer URL: $ALB_URL"
}

# Lambda 함수 코드 업데이트
update_lambda_code() {
    echo_info "Lambda 함수 코드 업데이트 중..."
    
    # 임시 디렉토리 생성
    TEMP_DIR=$(mktemp -d)
    cp ../lambda/session_manager.py $TEMP_DIR/
    
    # ZIP 파일 생성
    cd $TEMP_DIR
    zip -r session_manager.zip session_manager.py
    
    # Lambda 함수 업데이트
    aws lambda update-function-code \
        --function-name "${PROJECT_NAME}-session-manager" \
        --zip-file fileb://session_manager.zip \
        --region $AWS_REGION
    
    # 임시 파일 정리
    cd - > /dev/null
    rm -rf $TEMP_DIR
    
    echo_info "Lambda 함수 업데이트 완료"
}

# 환경 설정 파일 생성
create_env_file() {
    echo_info ".env 파일 생성 중..."
    
    ENV_FILE="../../mcp-server/.env"
    
    cat > $ENV_FILE << EOF
# AWS MCP Sandbox 환경 설정
AWS_REGION=$AWS_REGION
SESSION_MANAGER_URL=$API_URL
DEFAULT_TIMEOUT=300
LOG_LEVEL=INFO

# AWS 자격증명 (필요시 설정)
# AWS_ACCESS_KEY_ID=your-access-key
# AWS_SECRET_ACCESS_KEY=your-secret-key
EOF
    
    echo_info ".env 파일이 생성되었습니다: $ENV_FILE"
}

# MCP 설정 예제 출력
show_mcp_config() {
    echo_info "MCP 클라이언트 설정 예제:"
    
    cat << EOF

Claude Desktop 설정에 추가할 내용:

{
  "mcpServers": {
    "aws-code-sandbox": {
      "command": "python",
      "args": ["$(pwd)/../../mcp-server/server.py"],
      "env": {
        "AWS_REGION": "$AWS_REGION",
        "SESSION_MANAGER_URL": "$API_URL"
      }
    }
  }
}

EOF
}

# 배포 상태 확인
check_deployment() {
    echo_info "배포 상태 확인 중..."
    
    # API Gateway 헬스체크 (Lambda가 제대로 응답하는지 확인)
    echo_info "API Gateway 테스트 중..."
    
    TEST_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST "$API_URL/session" \
        -H "Content-Type: application/json" \
        -d '{"session_id": "test-session", "action": "get_or_create"}')
    
    if [ "$TEST_RESPONSE" = "200" ]; then
        echo_info "API Gateway 테스트 성공"
    else
        echo_warn "API Gateway 테스트 실패 (HTTP $TEST_RESPONSE)"
        echo_warn "이는 Docker 이미지가 아직 빌드되지 않았기 때문일 수 있습니다."
    fi
}

# 메인 실행
main() {
    echo_info "AWS MCP Sandbox 인프라 배포 시작"
    
    check_aws_cli
    create_ecr_repository
    deploy_infrastructure
    get_stack_outputs
    update_lambda_code
    create_env_file
    show_mcp_config
    check_deployment
    
    echo_info "배포 완료!"
    echo_info "다음 단계:"
    echo_info "1. Docker 이미지 빌드: cd ../../docker && ./build-and-push.sh"
    echo_info "2. MCP 서버 테스트: cd ../../mcp-server && python server.py"
    echo_info "3. Claude Desktop에 MCP 서버 설정 추가"
}

# 도움말
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "AWS MCP Sandbox 배포 스크립트"
    echo ""
    echo "사용법: $0 [OPTIONS]"
    echo ""
    echo "환경 변수:"
    echo "  AWS_REGION    AWS 리전 (기본값: us-west-2)"
    echo "  PROJECT_NAME  프로젝트 이름 (기본값: mcp-sandbox)"
    echo ""
    echo "예제:"
    echo "  $0                    # 기본 설정으로 배포"
    echo "  AWS_REGION=us-west-2 $0  # 다른 리전에 배포"
    echo ""
    exit 0
fi

main