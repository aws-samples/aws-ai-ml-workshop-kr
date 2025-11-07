# YouTube 콘텐츠 보고서 통합 가이드

## 개요

긴 검토 과정을 거쳐 **agentcore_gateway.py + main.py 패턴**으로 최종 결정되었습니다.
이 가이드는 YouTube 콘텐츠를 보고서에 통합하기 위한 완전한 워크플로우를 제공합니다.

## 사전 준비: YouTube API 키 설정

### 1. Google Cloud Console에서 API 키 생성
1. [Google Cloud Console](https://console.cloud.google.com/) 접속
2. **API 및 서비스** → **사용자 인증 정보** 이동
3. **+ 사용자 인증 정보 만들기** → **API 키** 선택
4. **YouTube Data API v3** 활성화 확인

### 2. 환경 변수 설정
`.env` 파일에 YouTube API 키 추가:
```bash
# YouTube API Key
YOUTUBE_API_KEY=AIzaSyC-your-actual-api-key-here
```

### 3. 오류 해결
`python agentcore_gateway.py` 실행 시 "youtube key없다" 오류가 발생하면:
- `.env` 파일에서 `YOUTUBE_API_KEY` 값이 올바르게 설정되었는지 확인
- API 키에 YouTube Data API v3 권한이 있는지 확인

#### Secrets Manager 권한 오류 해결
`AccessDeniedException: secretsmanager:PutSecretValue` 오류 발생 시:

**방법 1: IAM 정책 추가 (권장)**
1. AWS Console → IAM → 역할 → `AmazonSageMaker-ExecutionRole-*` 검색
2. 권한 추가 → 정책 연결
3. 다음 정책들을 생성 후 연결:

**Bedrock AgentCore 정책 (`bedrock-agentcore-policy.json`):**
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock-agentcore:*"
            ],
            "Resource": "*"
        }
    ]
}
```

**Secrets Manager 정책 (`secrets-manager-policy.json`):**
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "secretsmanager:CreateSecret",
                "secretsmanager:GetSecretValue",
                "secretsmanager:UpdateSecret",
                "secretsmanager:DeleteSecret",
                "secretsmanager:DescribeSecret",
                "secretsmanager:PutSecretValue"
            ],
            "Resource": "*"
        }
    ]
}
```

**Cognito 정책 (`cognito-policy.json`):**
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "cognito-idp:DescribeResourceServer",
                "cognito-idp:DescribeUserPool",
                "cognito-idp:DescribeUserPoolClient",
                "cognito-idp:ListUserPools",
                "cognito-idp:CreateResourceServer",
                "cognito-idp:UpdateResourceServer",
                "cognito-idp:DeleteResourceServer"
            ],
            "Resource": "*"
        }
    ]
}
```

**방법 2: 로컬 환경 변수만 사용**
```bash
# .env 파일에 API 키만 설정하고 Secrets Manager 업데이트 무시
YOUTUBE_API_KEY=your_actual_api_key_here
```

## 최종 아키텍처

```
1. agentcore_gateway.py (YouTube 데이터 수집)
   ↓
2. /data/youtube_raw_data.json (수집된 데이터)
   ↓  
3. main.py (CSV + YouTube 통합 분석)
   ↓
4. PDF 보고서 생성
```

## 주요 수정 사항

### 1. OpenAPI 스펙 개선 - getVideoDetails 엔드포인트 추가

#### **Before (searchVideos만 존재)**
```json
{
  "paths": {
    "/search": {
      "get": {
        "operationId": "searchVideos",
        "summary": "Search for YouTube videos"
      }
    }
  }
}
```

#### **After (getVideoDetails 추가)**
```json
{
  "paths": {
    "/search": {
      "get": {
        "operationId": "searchVideos",
        "summary": "Search for YouTube videos"
      }
    },
    "/videos": {
      "get": {
        "operationId": "getVideoDetails",
        "summary": "Get detailed information about specific videos",
        "parameters": [
          {"name": "part", "default": "snippet,statistics,contentDetails"},
          {"name": "id", "required": true}
          // "key" 파라미터는 제거 - Gateway가 자동 주입
        ]
      }
    }
  }
}
```

**추가된 기능:**
- 비디오 상세 정보 조회: 조회수, 좋아요, 댓글, 재생시간
- `searchVideos`로 비디오 ID 수집 → `getVideoDetails`로 메타데이터 보강
- **주의**: `key` 파라미터는 OpenAPI 스펙에서 제거 (Gateway 자격 증명 공급자와 충돌 방지)

### 2. agentcore_gateway.py 개선

#### **키워드 생성 로직 개선**
```python
# Before (농산물 도매시장 중심)
"가격", "동향", "시세", "전망", "경매", "시장"

# After (이커머스 중심)
"온라인 판매", "이커머스", "배송", "소비 트렌드", "구매 패턴"
```

#### **동음이의어 방지**
```python
# Before
"사과 온라인 판매" → Apple 회사 영상 포함 ❌

# After
"사과 과일 온라인 판매" → 실제 과일 사과 영상만 ✅
```

**수정된 검색어 생성 원칙:**
```
6. 중요 : 동음이의어 방지를 위해 명확한 한정어 사용
   (예: "사과" → "사과 과일", "배" → "배 과일", "브로콜리" → "브로콜리 채소")
```

### 2. main.py 쿼리 수정

#### **Before (도구 기반 접근 - 실패)**
```python
user_query = """
2. youtube_data_collection_tool을 사용해서 관련 시장 동향 영상 검색
3. 생성된 키워드로 youtube_data_collection_tool을 사용해서...
"""
```

#### **After (파일 기반 접근 - 성공)**
```python
user_query = """
1. './data/Dat-fresh-food-claude.csv' 파일의 세일즈 및 마케팅 관점 분석
2. './data/youtube_raw_data.json' 파일의 YouTube 시장 동향 데이터 활용
   (이 파일은 CSV 데이터를 기반으로 추출된 키워드로 수집된 YouTube 트렌드 정보입니다)
3. CSV 판매 데이터와 YouTube 트렌드 정보를 종합한 인사이트 도출
4. 차트 생성 및 PDF 파일로 최종 보고서 작성
"""
```

### 3. Gateway Target 관리 개선

#### **문제점**
- 실행할 때마다 새로운 Target 생성
- Target 누적으로 인한 관리 복잡성 (10개 이상 중복)
- OpenAPI 스펙 변경이 기존 Target에 반영 안 됨
- 비동기 삭제로 인한 불완전한 정리

#### **해결책: agentcore_gateway.py에 자동 정리 통합**

이제 **별도의 정리 스크립트가 불필요**합니다. `agentcore_gateway.py`가 실행 시 자동으로:

```python
# agentcore_gateway.py 내부 동작
1. Gateway 확인/생성
2. 기존 타겟 완전 삭제 (반복 확인으로 비동기 처리)  ← 자동 수행
3. 새 타겟 생성 (getVideoDetails 포함)
4. YouTube 데이터 수집
```

**자동 정리 특징:**
- ✅ 반복 삭제 로직 (최대 5회, 비동기 처리 대응)
- ✅ OpenAPI 스펙 자동 업로드 (최신 버전 보장)
- ✅ 단일 Target 생성 (searchVideos + getVideoDetails)
- ✅ AWS 이름 규칙 준수 (하이픈만 사용)
- ✅ 에러 복구 및 상태 확인

**정리 결과:**
```
Before: 10개 이상의 중복 Target (searchVideos만 존재)
After: 1개의 깨끗한 Target (searchVideos + getVideoDetails)
```

**긴급 복구용 스크립트 (문제 발생 시만):**
```bash
# agentcore_gateway.py 자체가 실패하는 경우에만 사용
python cleanup_and_recreate_targets.py
python check_target_status.py
```

## 실행 워크플로우

### 1단계: 환경 준비
```bash
cd /path/to/project
```

### 2단계: YouTube 데이터 수집 (자동 정리 포함)
```bash
# agentcore_gateway.py가 자동으로 다음을 수행:
# 1. Gateway 확인/생성
# 2. 기존 타겟 완전 삭제 (중복 제거)
# 3. 최신 OpenAPI 스펙 업로드
# 4. 새 타겟 생성 (getVideoDetails 포함)
# 5. YouTube 데이터 수집
python agentcore_gateway.py
```

**자동으로 수행되는 작업:**
- ✅ 기존 중복 Target 완전 제거 (10개 이상 → 1개)
- ✅ 최신 OpenAPI 스펙 적용 (getVideoDetails 포함)
- ✅ 새 Target 생성 및 상태 확인
- ✅ YouTube 데이터 수집

**실행 결과 예시:**
```
=== 게이트웨이 확인/생성 ===
✅ 기존 게이트웨이 'DemoGWOpenAPIAPIKeyYouTube' 사용!
   Gateway ID: demogwopenapiapikeyyoutube-rq7bi6cizg

=== 기존 타겟 완전 정리 ===
반복 1/5: 발견된 타겟 수 = 10
  삭제 요청 완료: 10/10개
✅ 모든 타겟이 삭제되었습니다!

=== 게이트웨이 타겟 생성 ===
새 게이트웨이 타겟 생성: YouTubeCompleteAPI-1761414570
✅ 새 게이트웨이 타겟 'YouTubeCompleteAPI-1761414570' 생성 완료!
   포함된 도구: searchVideos, getVideoDetails

=== YouTube 원시 데이터 수집 ===
에이전트에 로드된 도구들: ['YouTubeCompleteAPI-1761414570___searchVideos',
                        'YouTubeCompleteAPI-1761414570___getVideoDetails']
✅ getVideoDetails 도구 사용 가능!
```

### 3단계: Target 상태 확인 (선택, 문제 발생 시만)
```bash
python check_target_status.py
```

### 4단계: 통합 분석 및 보고서 생성
```bash
python main.py
```

**수행 작업:**
- CSV 판매 데이터 분석
- YouTube 트렌드 데이터 분석 (메타데이터 포함)
- 두 데이터 소스 통합 인사이트 도출
- 차트 및 시각화 생성
- PDF 보고서 작성

## 데이터 흐름

### 입력 데이터
```
./data/Dat-fresh-food-claude.csv (Moon Market 판매 데이터)
```

### 중간 데이터
```
./data/youtube_raw_data.json (YouTube 시장 동향 데이터)
./data/csv_keywords.json (추출된 키워드)
./data/data_collection_summary.txt (수집 요약)
```

### 출력 데이터
```
PDF 보고서 (CSV + YouTube 통합 분석)
차트 및 시각화 파일
```

## 키워드 매핑 예시

### CSV 데이터 → YouTube 키워드
```
CSV 상품명: "Apple" 
→ YouTube 키워드: "사과 과일 온라인 판매 트렌드"

CSV 카테고리: "Fruits"
→ YouTube 키워드: "과일 이커머스 소비 패턴"

CSV 상품명: "Broccoli"
→ YouTube 키워드: "브로콜리 채소 배송 서비스"
```

## 장점

### ✅ **데이터 기반 접근**
- 하드코딩된 키워드 대신 실제 CSV 데이터 기반
- Moon Market 판매 현황과 직접적 연관성

### ✅ **정확성 향상**
- 동음이의어 방지로 관련성 높은 콘텐츠만 수집
- 이커머스 중심 키워드로 비즈니스 맥락 일치

### ✅ **안정성 확보**
- 검증된 파일 기반 접근 방식
- Strands 프레임워크 도구 인식 문제 회피

### ✅ **관리 효율성**
- Target 정리 스크립트로 깔끔한 리소스 관리
- 재사용 가능한 워크플로우

## 트러블슈팅

### 1. Gateway Target 캐싱 및 중복 문제

#### **문제 증상**
```
에이전트에 로드된 도구들: ['DemoOpenAPITargetS3YouTube-1759246733___searchVideos',
'DemoOpenAPITargetS3YouTube-1759309315___searchVideos', ...] (10개 이상 중복)
❌ getVideoDetails 도구 없음 - searchVideos만 사용
```

#### **근본 원인**
1. **Target 캐싱**: 이전에 생성된 Target들이 삭제되지 않고 누적
2. **OpenAPI 스펙 미반영**: 새로운 OpenAPI 스펙 업로드가 기존 Target에 적용 안 됨
3. **비동기 삭제**: Target 삭제 API가 비동기로 동작하여 완전 삭제 전 새 Target 생성

#### **해결 방법**

**Step 1: agentcore_gateway.py 재실행 (권장)**

이제 **별도 스크립트가 불필요**합니다:

```bash
# 자동으로 모든 문제 해결
python agentcore_gateway.py
```

**자동으로 수행되는 작업:**
```python
1. Gateway 확인/생성
2. 기존 타겟 완전 삭제 (반복 확인, 비동기 처리)  ← 자동
3. 새 타겟 생성 (getVideoDetails 포함)
4. YouTube 데이터 수집
```

**Step 2: 문제 지속 시 상태 확인 (선택)**
```bash
python check_target_status.py
```

**Step 3: 긴급 복구 (Step 1 실패 시만)**
```bash
# agentcore_gateway.py가 계속 실패하는 경우에만
python cleanup_and_recreate_targets.py
python agentcore_gateway.py
```

**기대 결과:**
```
타겟: YouTubeCompleteAPI-1761414570
  상태: READY
  기대 도구: searchVideos, getVideoDetails
```

### 2. OpenAPI 스펙 충돌 문제

#### **문제 증상**
```json
{
  "status": "FAILED",
  "statusReasons": [
    "The tool parameter key conflicts with api key credential provider configuration."
  ]
}
```

#### **근본 원인**
OpenAPI 스펙의 `/videos` 엔드포인트에 명시적으로 `key` 파라미터가 정의되어 있어서, Gateway의 API Key 자격 증명 공급자 설정(`credentialParameterName: "key"`)과 충돌

#### **해결 방법**

**Before (충돌 발생):**
```json
{
  "/videos": {
    "get": {
      "parameters": [
        {"name": "part", "required": true},
        {"name": "id", "required": true},
        {"name": "key", "required": true}  ❌ Gateway와 충돌
      ]
    }
  }
}
```

**After (충돌 해결):**
```json
{
  "/videos": {
    "get": {
      "parameters": [
        {"name": "part", "required": true},
        {"name": "id", "required": true}
        // "key" 파라미터 제거 - Gateway가 자동 주입
      ]
    }
  }
}
```

**수정된 파일:** `assets/youtube_api_openapi.json:238-246`

### 3. YouTube 데이터 수집 실패시

```bash
# 1. 자동 정리 포함하여 재시도 (권장)
python agentcore_gateway.py

# 2. 계속 실패 시 긴급 복구
python cleanup_and_recreate_targets.py
python agentcore_gateway.py

# 3. 권한 문제 확인
# SageMaker 실행 역할에 필요한 권한:
# - bedrock-agentcore:*
# - secretsmanager:PutSecretValue
# - s3:GetObject, s3:PutObject
```

### 4. JSON 파싱 오류 발생시

```bash
# 일부 동영상에서 오류가 발생해도 계속 진행
# 최종적으로 youtube_raw_data.json 파일 생성 확인
ls -la ./data/youtube_raw_data.json
```

### 5. 보고서 생성 실패시

```bash
# YouTube 데이터 파일 존재 확인
cat ./data/youtube_raw_data.json | head -10

# main.py 재실행
python main.py
```

## 핵심 성과 및 개선 사항 요약

### 🎯 완료된 주요 작업

1. **OpenAPI 스펙 확장**
   - `/videos` 엔드포인트 추가 (`getVideoDetails`)
   - 비디오 상세 메타데이터 수집 가능 (조회수, 좋아요, 댓글, 재생시간)
   - Gateway 자격 증명 충돌 해결 (`key` 파라미터 제거)

2. **Gateway Target 관리 자동화**
   - 통합 정리/재생성 스크립트 (`cleanup_and_recreate_targets.py`)
   - 비동기 삭제 처리 (반복 확인 로직)
   - 중복 Target 완전 제거 (10개+ → 1개)
   - Target 상태 자동 확인 (READY)

3. **문제 해결 체계화**
   - 실시간 Target 상태 확인 스크립트 (`check_target_status.py`)
   - OpenAPI 스펙 충돌 진단 및 해결
   - 트러블슈팅 가이드 문서화

### ✨ 주요 장점

1. **완전한 API 활용**: `searchVideos` + `getVideoDetails` 조합으로 풍부한 데이터 수집
2. **검증된 안정성**: 파일 기반 접근으로 도구 인식 문제 없음
3. **데이터 정확성**: CSV 기반 키워드 생성으로 관련성 극대화
4. **비즈니스 적합성**: 이커머스 중심 키워드로 실제 비즈니스 맥락 반영
5. **관리 편의성**: 자동화된 Target 정리 및 재사용 가능한 워크플로우
6. **문제 해결 용이성**: 명확한 오류 진단 및 해결 방법 제공

### 🚀 권장 실행 순서

```bash
# 1. YouTube 데이터 수집 (자동 정리 포함)
python agentcore_gateway.py
# → Gateway 확인, 타겟 정리, 새 타겟 생성, 데이터 수집 모두 자동 수행

# 2. 통합 분석 및 보고서 생성
python main.py

# 문제 발생 시에만:
python check_target_status.py  # 상태 확인
python cleanup_and_recreate_targets.py  # 강제 정리 (별도 스크립트)
```

## 결론

**agentcore_gateway.py + main.py 패턴**은 다음과 같은 이유로 최적의 선택입니다:

1. **완전한 YouTube API 통합**: searchVideos + getVideoDetails로 메타데이터 완전 수집
2. **검증된 안정성**: 파일 기반 접근으로 도구 인식 문제 없음
3. **데이터 정확성**: CSV 기반 키워드 생성으로 관련성 극대화
4. **비즈니스 적합성**: 이커머스 중심 키워드로 실제 비즈니스 맥락 반영
5. **관리 자동화**: Target 정리 및 재사용 가능한 워크플로우
6. **문제 해결 체계**: 명확한 진단 도구 및 해결 방법

이 가이드를 따라 실행하면 Moon Market 판매 데이터와 YouTube 시장 동향을 완벽하게 통합한 인사이트 보고서를 생성할 수 있습니다.

---

**마지막 업데이트**: 2025-10-25
**주요 개선**:
- Gateway Target 캐싱 문제 해결 (자동 정리 통합)
- getVideoDetails 엔드포인트 추가
- `agentcore_gateway.py`에 자동 정리 로직 통합 (단일 스크립트 실행으로 완료)
- `cleanup_and_recreate_targets.py`는 긴급 복구용으로 유지
