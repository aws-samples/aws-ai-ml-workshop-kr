# AI 메뉴보드 생성기 (AI Menu Board Generator)

Amazon Bedrock의 Nova Canvas와 Nova Pro 모델을 활용한 지능형 메뉴보드 생성 애플리케이션입니다.

## 개요 (Overview)

이 애플리케이션은 Amazon Bedrock의 최신 Nova 모델들을 사용하여 레스토랑, 카페, 패스트푸드점 등을 위한 전문적인 메뉴보드를 자동으로 생성합니다. 한국어 프롬프트를 입력하면 AI가 자동으로 번역하고 맞춤형 메뉴보드 이미지를 생성합니다.

## 아키텍처 (Architecture)

```
사용자 입력 (한국어) → Amazon Nova Pro (번역) → Amazon Nova Canvas (이미지 생성) → Streamlit UI → 최종 메뉴보드
```

## 주요 기능 (Key Features)

- **🎨 AI 기반 이미지 생성**: Amazon Nova Canvas를 사용한 고품질 메뉴보드 배경 생성
- **🌐 자동 번역**: Amazon Nova Pro를 통한 한국어-영어 실시간 번역
- **📝 텍스트 오버레이**: 생성된 이미지 위에 메뉴명과 가격 정보 추가
- **🖱️ 인터랙티브 UI**: Streamlit 기반의 직관적인 웹 인터페이스
- **🎯 정밀 좌표 설정**: 드래그 앤 드롭 캔버스를 통한 정확한 텍스트 위치 지정
- **💾 다운로드 기능**: 완성된 메뉴보드를 PNG 형식으로 저장


- - -

## 📋 사전 요구사항 (Prerequisites)

### 시스템 요구사항
- Python 3.8 이상
- macOS, Linux, 또는 Windows
- 최소 4GB RAM 권장

### AWS 요구사항
1. **AWS 계정**: 활성화된 AWS 계정이 필요합니다
2. **Amazon Bedrock 액세스**: Nova Canvas와 Nova Pro 모델에 대한 액세스 권한
3. **AWS 자격 증명**: 프로그래밍 방식 액세스를 위한 자격 증명 설정

### Amazon Bedrock 모델 액세스 설정
1. AWS 콘솔에서 Amazon Bedrock 서비스로 이동
2. 왼쪽 메뉴에서 "Model access" 선택
3. "Manage model access" 클릭
4. 다음 모델들에 대한 액세스 요청:
   - `Amazon Nova Canvas`
   - `Amazon Nova Pro`
5. 액세스 승인까지 몇 분 정도 소요될 수 있습니다

### AWS 자격 증명 설정
다음 중 하나의 방법으로 AWS 자격 증명을 설정하세요:

#### 방법 1: AWS CLI 설정 (권장)
```bash
# AWS CLI 설치
pip install awscli

# 자격 증명 설정
aws configure
```

#### 방법 2: 환경 변수 설정
```bash
export AWS_ACCESS_KEY_ID=your_access_key_here
export AWS_SECRET_ACCESS_KEY=your_secret_key_here
export AWS_DEFAULT_REGION=us-east-1
```

#### 방법 3: IAM 역할 (EC2/ECS에서 실행하는 경우)
EC2 인스턴스나 ECS 태스크에서 실행하는 경우 IAM 역할을 사용할 수 있습니다.

### 필요한 IAM 권한
사용자 또는 역할에 다음 권한이 필요합니다:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel"
            ],
            "Resource": [
                "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-canvas-v1:0",
                "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-pro-v1:0"
            ]
        }
    ]
}
```


- - -

## 🛠️ 설치 및 실행 (Installation & Setup)

### 1. 저장소 클론
```bash
git clone <repository-url>
cd menuboard-generator
```

### 2. 가상 환경 생성 (권장)
```bash
# Python 가상 환경 생성
python -m venv venv

# 가상 환경 활성화
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정 (선택사항)
```bash
# .env.example을 복사하여 .env 파일 생성
cp .env.example .env

# .env 파일을 편집하여 AWS 자격 증명 설정 (선택사항)
```

### 5. 애플리케이션 실행
```bash
streamlit run app.py
```

### 6. 브라우저에서 접속
애플리케이션이 실행되면 자동으로 브라우저가 열리거나, 다음 주소로 접속하세요:
```
http://localhost:8501
```


- - -

## 📦 의존성 (Dependencies)

이 프로젝트는 다음 주요 라이브러리들을 사용합니다:

- **streamlit** (1.28.1): 웹 애플리케이션 프레임워크
- **boto3** (1.34.0): AWS SDK for Python
- **Pillow** (10.1.0): 이미지 처리 라이브러리
- **numpy** (1.24.3): 수치 계산 라이브러리
- **opencv-python** (4.8.1.78): 컴퓨터 비전 라이브러리
- **streamlit-drawable-canvas** (0.9.3): 인터랙티브 캔버스 컴포넌트


- - -

## 📖 사용 방법 (Usage Guide)

### 1. 메뉴보드 이미지 생성
- 사이드바에서 메뉴보드 설명을 한국어로 입력
- "메뉴보드 이미지 생성" 버튼 클릭
- AI가 자동으로 한국어를 영어로 번역하고 이미지 생성

### 2. 메뉴 정보 추가
- 메뉴명과 가격 입력
- X, Y 좌표와 너비, 높이 설정
- "메뉴 추가" 버튼으로 텍스트 오버레이 추가

### 3. 영역 선택 도구 활용
- 캔버스에서 직접 사각형을 그려 정확한 좌표 확인
- 확인된 좌표를 메뉴 정보 입력에 활용

### 4. 완성된 메뉴보드 다운로드
- 모든 메뉴 추가 완료 후 PNG 파일로 다운로드

## 🎨 프롬프트 예시 (Prompt Examples)

### 카페 메뉴보드
```
깔끔하고 모던한 카페 메뉴보드, 여러 메뉴 항목을 위한 빈 공간이 있는 디자인, 따뜻한 색감
```

### 레스토랑 메뉴보드
```
고급스러운 레스토랑 메뉴보드, 우아한 디자인, 메뉴 항목을 위한 구분된 섹션들
```

### 패스트푸드 메뉴보드
```
밝고 활기찬 패스트푸드 메뉴보드, 컬러풀한 디자인, 큰 글씨를 위한 공간
```

## 🔧 커스터마이징 (Customization)

### 폰트 변경
`app.py` 파일의 `get_korean_font` 함수에서 폰트 경로를 수정하세요:
```python
korean_fonts = [
    "/path/to/your/font.ttf",
    # 기존 폰트 목록...
]
```

### 이미지 크기 조정
`generate_menuboard_image` 함수에서 이미지 크기를 변경할 수 있습니다:
```python
"height": 768,  # 원하는 높이
"width": 1024,  # 원하는 너비
```

### 지원 언어 추가
번역 기능을 확장하여 다른 언어도 지원할 수 있습니다.

## 🚨 문제 해결 (Troubleshooting)

### 일반적인 오류

#### 1. AWS 자격 증명 오류
```
NoCredentialsError: Unable to locate credentials
```
**해결 방법:**
- AWS 자격 증명이 올바르게 설정되었는지 확인
- `aws configure` 명령어로 자격 증명 재설정
- 환경 변수가 올바르게 설정되었는지 확인

#### 2. 모델 액세스 오류
```
AccessDeniedException: You don't have access to the model
```
**해결 방법:**
- Amazon Bedrock 콘솔에서 Nova 모델에 대한 액세스 권한 요청
- IAM 권한이 올바르게 설정되었는지 확인

#### 3. 폰트 관련 오류
```
OSError: cannot open resource
```
**해결 방법:**
- 시스템에 맞는 폰트 경로로 수정
- 기본 폰트 사용으로 대체
- macOS의 경우 시스템 폰트 경로 확인

#### 4. 이미지 생성 실패
**해결 방법:**
- 프롬프트가 너무 복잡하지 않은지 확인
- 네트워크 연결 상태 확인
- AWS 서비스 상태 확인

#### 5. Streamlit 실행 오류
```
ModuleNotFoundError: No module named 'streamlit'
```
**해결 방법:**
- 가상 환경이 활성화되었는지 확인
- `pip install -r requirements.txt` 재실행
- Python 버전 호환성 확인

### 성능 최적화

- **이미지 캐싱**: 생성된 이미지는 세션 상태에 저장되어 재사용됩니다
- **AWS 클라이언트 캐싱**: `@st.cache_resource`를 사용하여 AWS 클라이언트를 캐시합니다
- **메모리 관리**: 큰 이미지 파일 처리 시 메모리 사용량에 주의하세요


- - -

## 🎥 Demo

[Video](https://youtu.be/gN67DHNn4kc)


- - -


## 📊 프로젝트 구조 (Project Structure)

```
menuboard-generator/
├── app.py                 # 메인 애플리케이션 파일
├── requirements.txt       # Python 의존성 목록
├── README.md             # 프로젝트 문서
├── .env.example          # 환경 변수 예시 파일
├── .gitignore           # Git 무시 파일 목록
└── venv/                # 가상 환경 (생성 후)
```


- - -

## 🔒 보안 고려사항 (Security Considerations)

- AWS 자격 증명을 코드에 하드코딩하지 마세요
- 환경 변수나 AWS IAM 역할을 사용하세요
- 프로덕션 환경에서는 최소 권한 원칙을 적용하세요
- API 키나 민감한 정보는 `.env` 파일에 저장하고 `.gitignore`에 추가하세요


- - -

## 🚀 배포 (Deployment)

### Streamlit Cloud
1. GitHub 저장소에 코드 푸시
2. [Streamlit Cloud](https://streamlit.io/cloud)에서 앱 배포
3. 환경 변수에 AWS 자격 증명 설정

### AWS EC2
1. EC2 인스턴스 생성 및 설정
2. IAM 역할을 통한 권한 부여
3. 애플리케이션 배포 및 실행

### Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

- - -


## 👥 Contributors
- Kyutae Park, Ph.D (AWS Solutions Architect) | [Mail](mailto:kyutae@amazon.com) | [Linkedin](https://www.linkedin.com/in/ren-ai-ssance/) | [Git](https://github.com/ren-ai-ssance) |

- - -

## 🔑 License
- This is licensed under the [MIT License](https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/LICENSE).