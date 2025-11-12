# OpenSearch 클러스터 설정

## Prerequisites

### 1. uv 환경 설정
먼저 프로젝트 루트의 `setup` 디렉토리에서 uv 환경을 설정해야 합니다:
자세한 내용은 [setup/README.md](../setup/README.md)를 참고하세요.

### 2. 필수 권한
실습을 위해 실행하는 IAM Role에 다음 권한이 필요합니다:
- `AmazonOpenSearchServiceFullAccess`
- `AmazonSSMFullAccess`

### 3. 환경 설정
프로젝트 루트의 `.env` 파일에서 OpenSearch 설정을 구성할 수 있습니다:

```bash
# OpenSearch Configuration
OPENSEARCH_VERSION=3.1
OPENSEARCH_DOMAIN_NAME=my-opensearch-domain
OPENSEARCH_USER_ID=your-username
OPENSEARCH_USER_PASSWORD=your-password
```

설정 우선순위: **Command-line arguments > .env > Default values**

## 설정 방법

### Option 1: .env 파일 사용 (권장)

1. `.env` 파일에 OpenSearch 설정 추가:
```bash
OPENSEARCH_VERSION=3.1
OPENSEARCH_DOMAIN_NAME=my-opensearch-domain
OPENSEARCH_USER_ID=raguser
OPENSEARCH_USER_PASSWORD=MarsEarth1!
```

2. 스크립트 실행 권한 부여:
```bash
chmod +x create-opensearch.sh
```

3. uv run으로 스크립트 실행:
```bash
uv run ./create-opensearch.sh
```

4. 필요시 특정 옵션만 override:
```bash
uv run ./create-opensearch.sh -d my-domain-name -m prod
```

### Option 2: Command-line arguments 사용

모든 설정을 명령행으로 지정:
```bash
chmod +x create-opensearch.sh
uv run ./create-opensearch.sh -v 3.1 -d my-domain -u myuser -p MyPassword1! -m prod
```

### Option 3: Interactive 모드

대화형으로 설정 입력:
```bash
uv run ./create-opensearch.sh --interactive
```

## Parameters

| Parameter | Short | Description | Default (.env) | Default (hardcoded) |
|-----------|-------|-------------|----------------|---------------------|
| `--version` | `-v` | OpenSearch version (1.3, 2.3, 2.5, 2.7, 2.9, 2.11, 2.13, 2.15, 2.17, 2.19, 3.1) | `OPENSEARCH_VERSION` | 3.1 |
| `--user-id` | `-u` | OpenSearch master user ID | `OPENSEARCH_USER_ID` | raguser |
| `--password` | `-p` | OpenSearch master password (8+ chars, mixed case, numbers, special chars) | `OPENSEARCH_USER_PASSWORD` | MarsEarth1! |
| `--domain-name` | `-d` | Domain name (3-28 chars, lowercase, starts with letter) | `OPENSEARCH_DOMAIN_NAME` | auto-generated |
| `--mode` | `-m` | Deployment mode: `dev` (1-AZ) or `prod` (3-AZ with standby) | - | dev |
| `--interactive` | `-i` | Run in interactive mode | - | - |
| `--help` | `-h` | Show help message | - | - |

## 실행 예시

### 기본 설정으로 개발 환경 구축
```bash
uv run ./create-opensearch.sh
```

### 프로덕션 환경 구축
```bash
uv run ./create-opensearch.sh -m prod -d production-search
```

### 특정 버전과 커스텀 자격증명
```bash
uv run ./create-opensearch.sh -v 3.1 -u admin -p SecurePass123!
```

### Python 스크립트 직접 실행
```bash
uv run python create-opensearch.py --version 3.1 --domain-name my-cluster --prod
```

## 실행 시간

- 개발 모드 (dev): 약 30-40분
- 프로덕션 모드 (prod): 약 50-60분

## 생성 결과

스크립트 완료 후:
- `.env` 파일이 자동으로 업데이트됩니다:
  - `OPENSEARCH_DOMAIN_NAME`: 생성된 도메인 이름
  - `OPENSEARCH_DOMAIN_ENDPOINT`: OpenSearch 도메인 엔드포인트
- (선택사항) `src.utils.ssm` 모듈이 있는 경우 AWS Systems Manager Parameter Store에도 저장됩니다

## 주의사항

1. 도메인 이름은 소문자로 시작하고, 소문자/숫자/하이픈만 포함 가능
2. 비밀번호는 대문자, 소문자, 숫자, 특수문자를 포함해야 함
3. 프로덕션 모드는 비용이 더 높으니 개발 시에는 dev 모드 사용 권장
4. 한국어 분석을 위한 Nori 플러그인이 자동으로 설치됨
5. 지원 리전: us-east-1, us-west-2, ap-northeast-2
