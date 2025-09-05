#!/bin/bash

# Docker 이미지 빌드 및 ECR 푸시 스크립트

set -e

# 기본 설정
PROJECT_NAME="mcp-sandbox"
AWS_REGION="${AWS_REGION:-us-west-2}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

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

# 필수 도구 확인
check_prerequisites() {
    echo_info "필수 도구 확인 중..."
    
    if ! command -v docker &> /dev/null; then
        echo_error "Docker가 설치되지 않았습니다."
        exit 1
    fi
    
    if ! command -v aws &> /dev/null; then
        echo_error "AWS CLI가 설치되지 않았습니다."
        exit 1
    fi
    
    if ! aws sts get-caller-identity &> /dev/null; then
        echo_error "AWS 자격증명이 구성되지 않았습니다."
        exit 1
    fi
    
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${PROJECT_NAME}"
    
    echo_info "AWS Account ID: $ACCOUNT_ID"
    echo_info "ECR URI: $ECR_URI"
}

# ECR 로그인
ecr_login() {
    echo_info "ECR 로그인 중..."
    
    aws ecr get-login-password --region $AWS_REGION | \
        docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
    
    if [ $? -eq 0 ]; then
        echo_info "ECR 로그인 성공"
    else
        echo_error "ECR 로그인 실패"
        exit 1
    fi
}

# ECR 리포지토리 확인
check_ecr_repository() {
    echo_info "ECR 리포지토리 확인 중..."
    
    if ! aws ecr describe-repositories --repository-names $PROJECT_NAME --region $AWS_REGION &> /dev/null; then
        echo_warn "ECR 리포지토리가 존재하지 않습니다. 생성 중..."
        aws ecr create-repository \
            --repository-name $PROJECT_NAME \
            --region $AWS_REGION \
            --image-scanning-configuration scanOnPush=true
        echo_info "ECR 리포지토리 생성 완료"
    else
        echo_info "ECR 리포지토리 존재 확인"
    fi
}

# Docker 이미지 빌드
build_image() {
    echo_info "Docker 이미지 빌드 시작..."
    
    # fargate-runtime 디렉토리로 이동
    cd ../fargate-runtime
    
    # 현재 시간 태그 추가
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    
    echo_info "빌드 컨텍스트: $(pwd)"
    echo_info "이미지 태그: $IMAGE_TAG"
    
    # Docker 이미지 빌드 (네이티브 ARM64 빌드)
    echo_info "네이티브 ARM64 이미지 빌드 중..."
    docker build \
        --tag ${PROJECT_NAME}:${IMAGE_TAG} \
        --tag ${PROJECT_NAME}:${TIMESTAMP} \
        --tag ${ECR_URI}:${IMAGE_TAG} \
        --tag ${ECR_URI}:${TIMESTAMP} \
        .
    
    if [ $? -eq 0 ]; then
        echo_info "Docker 이미지 빌드 완료"
        echo_info "로컬 태그: ${PROJECT_NAME}:${IMAGE_TAG}"
        echo_info "ECR 태그: ${ECR_URI}:${IMAGE_TAG}"
    else
        echo_error "Docker 이미지 빌드 실패"
        exit 1
    fi
    
    # 원래 디렉토리로 복귀
    cd - > /dev/null
}

# Docker 이미지 푸시
push_image() {
    echo_info "ECR에 이미지 푸시 중..."
    
    # latest 태그 푸시
    docker push ${ECR_URI}:${IMAGE_TAG}
    
    if [ $? -eq 0 ]; then
        echo_info "이미지 푸시 완료: ${ECR_URI}:${IMAGE_TAG}"
    else
        echo_error "이미지 푸시 실패"
        exit 1
    fi
}

# 이미지 테스트
test_image() {
    echo_info "로컬 이미지 테스트 중..."
    
    # 이미 실행 중인 컨테이너가 있으면 중지
    if docker ps -q --filter "name=${PROJECT_NAME}-test" | grep -q .; then
        echo_info "기존 테스트 컨테이너 중지 중..."
        docker stop ${PROJECT_NAME}-test
        docker rm ${PROJECT_NAME}-test
    fi
    
    # 테스트 컨테이너 실행
    echo_info "테스트 컨테이너 시작 중..."
    docker run -d \
        --name ${PROJECT_NAME}-test \
        -p 8080:8080 \
        ${PROJECT_NAME}:${IMAGE_TAG}
    
    # 컨테이너가 시작될 때까지 대기
    echo_info "컨테이너 시작 대기 중..."
    sleep 10
    
    # 헬스체크
    echo_info "헬스체크 수행 중..."
    HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health)
    
    if [ "$HEALTH_STATUS" = "200" ]; then
        echo_info "헬스체크 성공 (HTTP 200)"
        
        # 간단한 Python 코드 테스트
        echo_info "Python 실행 테스트 중..."
        TEST_RESPONSE=$(curl -s -X POST http://localhost:8080/execute \
            -H "Content-Type: application/json" \
            -d '{"code": "print(\"Hello from Docker!\")", "type": "python"}')
        
        if echo "$TEST_RESPONSE" | grep -q "Hello from Docker"; then
            echo_info "Python 실행 테스트 성공"
        else
            echo_warn "Python 실행 테스트 실패"
            echo "응답: $TEST_RESPONSE"
        fi
    else
        echo_error "헬스체크 실패 (HTTP $HEALTH_STATUS)"
    fi
    
    # 테스트 컨테이너 중지 및 제거
    echo_info "테스트 컨테이너 정리 중..."
    docker stop ${PROJECT_NAME}-test
    docker rm ${PROJECT_NAME}-test
}

# 이미지 정보 출력
show_image_info() {
    echo_info "빌드된 이미지 정보:"
    docker images | grep $PROJECT_NAME
    
    echo ""
    echo_info "ECR 리포지토리 정보:"
    aws ecr describe-images \
        --repository-name $PROJECT_NAME \
        --region $AWS_REGION \
        --query 'imageDetails[*].[imageTags[0],imagePushedAt,imageSizeInBytes]' \
        --output table
}

# 메인 실행
main() {
    echo_info "Docker 이미지 빌드 및 푸시 시작"
    echo_info "프로젝트: $PROJECT_NAME"
    echo_info "리전: $AWS_REGION"
    echo_info "태그: $IMAGE_TAG"
    
    check_prerequisites
    check_ecr_repository
    ecr_login
    build_image
    
    # 로컬 테스트 (선택사항)
    if [ "$SKIP_TEST" != "true" ]; then
        test_image
    fi
    
    push_image
    show_image_info
    
    echo_info "완료!"
    echo_info "이미지가 성공적으로 ECR에 푸시되었습니다: ${ECR_URI}:${IMAGE_TAG}"
    echo ""
    echo_info "다음 단계:"
    echo_info "1. ECS 서비스 업데이트 (필요시)"
    echo_info "2. MCP 서버 테스트: cd ../mcp-server && python server.py"
}

# 도움말
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Docker 이미지 빌드 및 ECR 푸시 스크립트"
    echo ""
    echo "사용법: $0 [OPTIONS]"
    echo ""
    echo "환경 변수:"
    echo "  AWS_REGION    AWS 리전 (기본값: us-west-2)"
    echo "  PROJECT_NAME  프로젝트 이름 (기본값: mcp-sandbox)"
    echo "  IMAGE_TAG     이미지 태그 (기본값: latest)"
    echo "  SKIP_TEST     로컬 테스트 건너뛰기 (true/false)"
    echo ""
    echo "예제:"
    echo "  $0                    # 기본 설정으로 빌드 및 푸시"
    echo "  IMAGE_TAG=v1.0 $0    # 특정 태그로 빌드"
    echo "  SKIP_TEST=true $0    # 테스트 없이 빌드만"
    echo ""
    exit 0
fi

main