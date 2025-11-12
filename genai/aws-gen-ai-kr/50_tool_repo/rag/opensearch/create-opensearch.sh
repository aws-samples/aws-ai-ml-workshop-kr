#!/bin/bash

# OpenSearch 클러스터 설정 Shell 스크립트
# 사용법: ./create-opensearch.sh [옵션]

set -e  # 오류 발생 시 스크립트 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# .env 파일 로드
ENV_FILE="../.env"
if [[ -f "$ENV_FILE" ]]; then
    echo -e "${GREEN}Loading configuration from $ENV_FILE${NC}"
    # .env에서 값 추출 (주석과 빈 줄 제외)
    export $(grep -v '^#' "$ENV_FILE" | grep -v '^$' | xargs)
fi

# 기본값 설정 (.env에서 로드하거나 하드코딩된 기본값 사용)
DEFAULT_VERSION="${OPENSEARCH_VERSION:-3.1}"
DEFAULT_USER_ID="${OPENSEARCH_USER_ID:-raguser}"
DEFAULT_PASSWORD="${OPENSEARCH_USER_PASSWORD:-MarsEarth1!}"
DEFAULT_DOMAIN_NAME="${OPENSEARCH_DOMAIN_NAME:-}"
DEFAULT_MODE="dev"

# 도움말 함수
show_help() {
    echo -e "${BLUE}OpenSearch 클러스터 설정 스크립트${NC}"
    echo ""
    echo "사용법: $0 [옵션]"
    echo ""
    echo "옵션:"
    echo "  -v, --version VERSION     OpenSearch 버전 (기본값: $DEFAULT_VERSION)"
    echo "                           지원 버전: 1.3, 2.3, 2.5, 2.7, 2.9, 2.11, 2.13, 2.15, 2.17, 2.19, 3.1"
    echo "  -u, --user-id USER_ID     OpenSearch 사용자 ID (기본값: $DEFAULT_USER_ID)"
    echo "  -p, --password PASSWORD   OpenSearch 사용자 비밀번호 (기본값: $DEFAULT_PASSWORD)"
    echo "  -d, --domain-name NAME    OpenSearch 도메인 이름 (기본값: 자동 생성)"
    echo "                           3-28자, 소문자로 시작, 소문자/숫자/하이픈만 사용"
    echo "  -m, --mode MODE          실행 모드 (기본값: $DEFAULT_MODE)"
    echo "                           dev: 개발 모드 (1-AZ without standby)"
    echo "                           prod: 프로덕션 모드 (3-AZ with standby)"
    echo "  -i, --interactive        대화형 모드로 실행"
    echo "  -h, --help               이 도움말 표시"
    echo ""
    echo "지원 리전: us-east-1, us-west-2, ap-northeast-2"
    echo ""
    echo "예시:"
    echo "  $0                                          # 기본값으로 실행"
    echo "  $0 -v 3.1 -u myuser -p mypass123           # 최신 버전으로 실행"
    echo "  $0 -d my-search-cluster                     # 도메인 이름 지정"
    echo "  $0 --interactive                           # 대화형 모드"
    echo "  $0 --mode prod                             # 프로덕션 모드"
    echo "  AWS_DEFAULT_REGION=ap-northeast-2 $0       # 서울 리전에서 실행"
    echo ""
}

# 입력 유효성 검사 함수
validate_version() {
    local version=$1
    case $version in
        1.3|2.3|2.5|2.7|2.9|2.11|2.13|2.15|2.17|2.19|3.1)
            return 0
            ;;
        *)
            echo -e "${RED}오류: 지원되지 않는 OpenSearch 버전입니다: $version${NC}"
            echo -e "${YELLOW}지원 버전: 1.3, 2.3, 2.5, 2.7, 2.9, 2.11, 2.13, 2.15, 2.17, 2.19, 3.1${NC}"
            return 1
            ;;
    esac
}

validate_password() {
    local password=$1
    if [[ ${#password} -lt 8 ]]; then
        echo -e "${RED}오류: 비밀번호는 최소 8자 이상이어야 합니다.${NC}"
        return 1
    fi
    
    # 비밀번호 복잡성 검사 (대문자, 소문자, 숫자, 특수문자 포함)
    if [[ ! $password =~ [A-Z] ]] || [[ ! $password =~ [a-z] ]] || [[ ! $password =~ [0-9] ]] || [[ ! $password =~ [^a-zA-Z0-9] ]]; then
        echo -e "${YELLOW}경고: 비밀번호는 대문자, 소문자, 숫자, 특수문자를 포함하는 것이 권장됩니다.${NC}"
    fi
    
    return 0
}

validate_domain_name() {
    local domain_name=$1
    
    # 빈 문자열은 허용 (자동 생성)
    if [[ -z "$domain_name" ]]; then
        return 0
    fi
    
    # 길이 검사 (3-28자)
    if [[ ${#domain_name} -lt 3 || ${#domain_name} -gt 28 ]]; then
        echo -e "${RED}오류: 도메인 이름은 3-28자여야 합니다.${NC}"
        return 1
    fi
    
    # 형식 검사: 소문자로 시작, 소문자/숫자/하이픈만 포함, 하이픈으로 끝나면 안됨
    if [[ ! $domain_name =~ ^[a-z][a-z0-9\-]*[a-z0-9]$ ]]; then
        echo -e "${RED}오류: 도메인 이름은 소문자로 시작하고, 소문자/숫자/하이픈만 포함할 수 있으며, 하이픈으로 끝날 수 없습니다.${NC}"
        echo -e "${YELLOW}예시: my-opensearch, search-cluster-01${NC}"
        return 1
    fi
    
    return 0
}

validate_mode() {
    local mode=$1
    case $mode in
        dev|prod)
            return 0
            ;;
        *)
            echo -e "${RED}오류: 지원되지 않는 모드입니다: $mode${NC}"
            echo -e "${YELLOW}지원 모드: dev, prod${NC}"
            return 1
            ;;
    esac
}

# 대화형 입력 함수
interactive_input() {
    echo -e "${BLUE}=== 대화형 모드 ===${NC}"
    echo ""
    
    # OpenSearch 버전 입력
    while true; do
        read -p "OpenSearch 버전을 입력하세요 (1.3/2.3/2.5/2.7/2.9/2.11/2.13/2.15/2.17/2.19/3.1) [기본값: $DEFAULT_VERSION]: " input_version
        version=${input_version:-$DEFAULT_VERSION}
        if validate_version "$version"; then
            break
        fi
    done
    
    # 사용자 ID 입력
    read -p "OpenSearch 사용자 ID를 입력하세요 [기본값: $DEFAULT_USER_ID]: " input_user_id
    user_id=${input_user_id:-$DEFAULT_USER_ID}
    
    # 비밀번호 입력
    while true; do
        read -s -p "OpenSearch 사용자 비밀번호를 입력하세요 [기본값: $DEFAULT_PASSWORD]: " input_password
        echo ""
        password=${input_password:-$DEFAULT_PASSWORD}
        if validate_password "$password"; then
            break
        fi
    done
    
    # 도메인 이름 입력
    while true; do
        read -p "도메인 이름을 입력하세요 (3-28자, 소문자로 시작) [기본값: 자동 생성]: " input_domain_name
        domain_name=${input_domain_name:-$DEFAULT_DOMAIN_NAME}
        if validate_domain_name "$domain_name"; then
            break
        fi
    done
    
    # 실행 모드 입력
    while true; do
        read -p "실행 모드를 선택하세요 (dev/prod) [기본값: $DEFAULT_MODE]: " input_mode
        mode=${input_mode:-$DEFAULT_MODE}
        if validate_mode "$mode"; then
            break
        fi
    done
}

# 설정 확인 함수
confirm_settings() {
    echo ""
    echo -e "${BLUE}=== 설정 확인 ===${NC}"
    echo "OpenSearch 버전: $version"
    echo "사용자 ID: $user_id"
    echo "비밀번호: $(echo $password | sed 's/./*/g')"
    echo "도메인 이름: ${domain_name:-'자동 생성'}"
    echo "실행 모드: $mode"
    echo ""
    
    read -p "위 설정으로 진행하시겠습니까? (y/N): " confirm
    case $confirm in
        [Yy]|[Yy][Ee][Ss])
            return 0
            ;;
        *)
            echo -e "${YELLOW}설정이 취소되었습니다.${NC}"
            return 1
            ;;
    esac
}

# 필수 조건 확인 함수
check_prerequisites() {
    echo -e "${BLUE}=== 필수 조건 확인 ===${NC}"
    
    # 현재 AWS 리전 확인
    current_region=$(aws configure get region 2>/dev/null || echo "미설정")
    if [[ -n "$AWS_DEFAULT_REGION" ]]; then
        current_region="$AWS_DEFAULT_REGION"
    fi
    echo "현재 AWS 리전: $current_region"
    
    # Python 확인
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}오류: Python3이 설치되어 있지 않습니다.${NC}"
        return 1
    fi
    
    # AWS CLI 확인
    if ! command -v aws &> /dev/null; then
        echo -e "${RED}오류: AWS CLI가 설치되어 있지 않습니다.${NC}"
        return 1
    fi
    
    # AWS 자격 증명 확인
    if ! aws sts get-caller-identity &> /dev/null; then
        echo -e "${RED}오류: AWS 자격 증명이 설정되어 있지 않습니다.${NC}"
        return 1
    fi
    
    # Python 스크립트 파일 확인
    if [[ ! -f "create-opensearch.py" ]]; then
        echo -e "${RED}오류: create-opensearch.py 파일을 찾을 수 없습니다.${NC}"
        return 1
    fi
    
    echo -e "${GREEN}모든 필수 조건이 충족되었습니다.${NC}"
    return 0
}

# 메인 실행 함수
run_opensearch_setup() {
    echo -e "${BLUE}=== OpenSearch 클러스터 설정 시작 ===${NC}"
    echo -e "${YELLOW}예상 소요 시간: 약 50분${NC}"
    echo ""
    
    # Python 스크립트 실행
    local python_args=""
    
    # 항상 버전을 명시적으로 전달
    python_args="$python_args --version $version"
    
    # 항상 사용자 ID를 명시적으로 전달
    python_args="$python_args --user-id $user_id"
    
    # 항상 비밀번호를 명시적으로 전달
    python_args="$python_args --password $password"
    
    # 도메인 이름이 설정된 경우 전달
    if [[ -n "$domain_name" ]]; then
        python_args="$python_args --domain-name $domain_name"
    fi
    
    # 모드 설정
    if [[ "$mode" == "prod" ]]; then
        python_args="$python_args --prod"
    fi
    
    echo -e "${GREEN}Python 스크립트를 실행합니다...${NC}"
    echo "python create-opensearch.py$python_args"
    echo ""
    
    if python create-opensearch.py$python_args; then
        echo ""
        echo -e "${GREEN}=== OpenSearch 클러스터 설정이 완료되었습니다! ===${NC}"
        return 0
    else
        echo ""
        echo -e "${RED}=== OpenSearch 클러스터 설정 중 오류가 발생했습니다! ===${NC}"
        return 1
    fi
}

# 메인 스크립트 시작
main() {
    # 기본값 설정
    version="$DEFAULT_VERSION"
    user_id="$DEFAULT_USER_ID"
    password="$DEFAULT_PASSWORD"
    domain_name="$DEFAULT_DOMAIN_NAME"
    mode="$DEFAULT_MODE"
    interactive=false
    
    # 명령행 인수 파싱
    while [[ $# -gt 0 ]]; do
        case $1 in
            -v|--version)
                version="$2"
                shift 2
                ;;
            -u|--user-id)
                user_id="$2"
                shift 2
                ;;
            -p|--password)
                password="$2"
                shift 2
                ;;
            -d|--domain-name)
                domain_name="$2"
                shift 2
                ;;
            -m|--mode)
                mode="$2"
                shift 2
                ;;
            -i|--interactive)
                interactive=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                echo -e "${RED}알 수 없는 옵션: $1${NC}"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 대화형 모드 실행
    if [[ "$interactive" == true ]]; then
        interactive_input
    fi
    
    # 입력값 유효성 검사
    if ! validate_version "$version" || ! validate_password "$password" || ! validate_domain_name "$domain_name" || ! validate_mode "$mode"; then
        exit 1
    fi
    
    # 설정 확인
    if ! confirm_settings; then
        exit 1
    fi
    
    # 필수 조건 확인
    if ! check_prerequisites; then
        exit 1
    fi
    
    # OpenSearch 설정 실행
    if run_opensearch_setup; then
        exit 0
    else
        exit 1
    fi
}

# 스크립트 실행
main "$@"