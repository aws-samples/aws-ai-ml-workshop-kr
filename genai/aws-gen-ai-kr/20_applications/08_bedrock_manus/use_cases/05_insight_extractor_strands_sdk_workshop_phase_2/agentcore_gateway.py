#!/usr/bin/env python3
"""
Bedrock AgentCore Gateway를 사용하여 YouTube API를 MCP 도구로 변환하기

개요:
고객은 JSON 또는 YAML 형식의 OpenAPI 스펙을 가져와서 Bedrock AgentCore Gateway를 사용하여
API를 MCP 도구로 변환할 수 있습니다. 이 스크립트는 API 키를 사용하여 YouTube Data API를
호출하는 YouTube 검색 에이전트를 구축합니다.

워크플로우:
1. Gateway용 도구 생성 - REST API용 OpenAPI 사양을 사용하여 도구를 정의
2. Gateway 엔드포인트 생성 - 인바운드 인증과 함께 MCP 진입점 역할을 할 게이트웨이를 생성
3. Gateway에 타겟 추가 - 게이트웨이가 특정 도구로 요청을 라우팅하는 방법을 정의
4. 에이전트 코드 업데이트 - MCP 인터페이스를 통해 모든 구성된 도구에 액세스
"""

import os
import json
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv

import boto3
import requests
import pandas as pd
from botocore.exceptions import ClientError

from strands.models import BedrockModel
from mcp.client.streamable_http import streamablehttp_client
from strands.tools.mcp.mcp_client import MCPClient
from strands import Agent

import utils


# 1. Config 클래스
class Config:
    """설정 상수"""
    DEFAULT_REGION = 'us-east-1'
    DEFAULT_MODEL_ID = 'us.anthropic.claude-3-7-sonnet-20250219-v1:0'
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_QUERY = "2025년 국내 신선식품 동향"
    
    # 재시도 설정
    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 10
    
    # Cognito 설정
    USER_POOL_NAME = "sample-agentcore-gateway-pool"
    RESOURCE_SERVER_ID = "sample-agentcore-gateway-id"
    RESOURCE_SERVER_NAME = "sample-agentcore-gateway-name"
    CLIENT_NAME = "sample-agentcore-gateway-client"
    
    # Gateway 설정
    GATEWAY_NAME = 'DemoGWOpenAPIAPIKeyYouTube'
    
    # 스코프 설정
    SCOPES = [
        {"ScopeName": "gateway:read", "ScopeDescription": "읽기 액세스"},
        {"ScopeName": "gateway:write", "ScopeDescription": "쓰기 액세스"}
    ]

# 2. AgentCoreGatewayManager 클래스
class AgentCoreGatewayManager:
    """Bedrock AgentCore Gateway 관리 클래스"""

    def __init__(self):
        """초기화 및 환경 설정"""
        # AWS 자격 증명 설정
        self._setup_aws_credentials()

        # AWS 설정
        self.region = os.getenv('AWS_DEFAULT_REGION', Config.DEFAULT_REGION)
        os.environ['AWS_DEFAULT_REGION'] = self.region

        # YouTube API 키 검증
        self._validate_environment()
        self.youtube_api_key = os.getenv('YOUTUBE_API_KEY')

        # AWS 클라이언트 초기화 (중복 제거)
        self.gateway_client = boto3.client('bedrock-agentcore-control', region_name=self.region)
        self.cognito = boto3.client("cognito-idp", region_name=self.region)
        self.s3_client = boto3.client('s3')
        self.sts_client = boto3.client('sts')

        # Cognito 설정
        self.user_pool_name = Config.USER_POOL_NAME
        self.resource_server_id = Config.RESOURCE_SERVER_ID
        self.resource_server_name = Config.RESOURCE_SERVER_NAME
        self.client_name = Config.CLIENT_NAME
        self.scopes = Config.SCOPES
        self.scope_string = f"{self.resource_server_id}/gateway:read {self.resource_server_id}/gateway:write"

        # 상태 변수
        self.gateway_id = None
        self.gateway_url = None
        self.gateway_name = Config.GATEWAY_NAME
        self.user_pool_id = None
        self.client_id = None
        self.client_secret = None
        self.access_token = None
        self.credential_provider_arn = None

        # 로깅 설정
        logging.getLogger("strands").setLevel(logging.INFO)
        logging.basicConfig(format="%(levelname)s | %(name)s | %(message)s", handlers=[logging.StreamHandler()])

        print("✅ AgentCore Gateway Manager 초기화 완료")

    def _setup_aws_credentials(self):
        """AWS 자격 증명 설정 (SageMaker 및 로컬 환경 지원)"""
        # SageMaker 환경 확인
        is_sagemaker = os.path.exists('/opt/ml') or 'SM_' in os.environ

        if not is_sagemaker:
            # 로컬 환경: .env 파일 로드
            try:
                load_dotenv()
                
                # 필수 환경 변수 확인
                required_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']
                for var in required_vars:
                    if not os.getenv(var):
                        raise ValueError(f"환경 변수 {var}가 설정되지 않았습니다.")
            except Exception as e:
                print(f"dotenv 로드 중 오류: {e}")

        # AWS_DEFAULT_REGION 설정
        os.environ['AWS_DEFAULT_REGION'] = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')

        # AWS 자격 증명 확인
        try:
            session = boto3.Session()
            credentials = session.get_credentials()
            if credentials:
                env_type = "SageMaker" if is_sagemaker else "로컬"
                print(f"✅ AWS 자격 증명이 성공적으로 로드되었습니다. ({env_type} 환경, Region: {os.environ['AWS_DEFAULT_REGION']})")
            else:
                raise ValueError("AWS 자격 증명을 찾을 수 없습니다.")
        except Exception as e:
            print(f"AWS 자격 증명 확인 중 오류: {e}")

    def _validate_environment(self):
        """환경 변수 유효성 검사 (YouTube API 키 선택적 확인)"""
        youtube_api_key = os.getenv('YOUTUBE_API_KEY')
        if not youtube_api_key:
            print("⚠️ YOUTUBE_API_KEY가 설정되지 않았습니다. 데모용 키를 사용합니다.")
            # 데모용 키 설정 (실제 환경에서는 실제 키 필요)
            os.environ['YOUTUBE_API_KEY'] = 'demo_key_for_testing'
        else:
            print("✅ YouTube API 키가 성공적으로 로드되었습니다.")

    def create_iam_role(self) -> Dict[str, Any]:
        """게이트웨이용 IAM 역할 생성"""
        print("=== IAM 역할 생성 ===")
        agentcore_gateway_iam_role = utils.create_agentcore_gateway_role("sample-lambdagateway")
        print("Agentcore gateway role ARN:", agentcore_gateway_iam_role['Role']['Arn'])
        return agentcore_gateway_iam_role

    def setup_cognito(self):
        """Amazon Cognito 풀 생성 및 설정"""
        print("=== Cognito 리소스 생성 또는 검색 ===")

        self.user_pool_id = utils.get_or_create_user_pool(self.cognito, self.user_pool_name)
        print(f"User Pool ID: {self.user_pool_id}")

        utils.get_or_create_resource_server(
            self.cognito, self.user_pool_id, self.resource_server_id,
            self.resource_server_name, self.scopes
        )
        print("리소스 서버 확인됨.")

        self.client_id, self.client_secret = utils.get_or_create_m2m_client(
            self.cognito, self.user_pool_id, self.client_name, self.resource_server_id
        )
        print(f"Client ID: {self.client_id}")

        # Discovery URL 생성
        self.cognito_discovery_url = f'https://cognito-idp.{self.region}.amazonaws.com/{self.user_pool_id}/.well-known/openid-configuration'
        print(f"Discovery URL: {self.cognito_discovery_url}")

    def create_gateway(self, agentcore_gateway_iam_role: Dict[str, Any]):
        """Bedrock AgentCore Gateway 생성"""
        print("=== 게이트웨이 생성 ===")

        # Cognito 인증 설정
        auth_config = {
            "customJWTAuthorizer": {
                "allowedClients": [self.client_id],
                "discoveryUrl": self.cognito_discovery_url
            }
        }

        try:
            # 기존 게이트웨이 확인
            list_response = self.gateway_client.list_gateways()
            gateways = list_response.get('items', [])

            create_response = None
            for gateway in gateways:
                if gateway['name'] == self.gateway_name:
                    print(f"✅ 기존 게이트웨이 '{self.gateway_name}' 사용!")
                    create_response = gateway
                    break

            if not create_response:
                print(f"새 게이트웨이 '{self.gateway_name}' 생성...")
                create_response = self.gateway_client.create_gateway(
                    name=self.gateway_name,
                    roleArn=agentcore_gateway_iam_role['Role']['Arn'],
                    protocolType='MCP',
                    authorizerType='CUSTOM_JWT',
                    authorizerConfiguration=auth_config,
                    description='AgentCore Gateway with OpenAPI target'
                )
                print(f"✅ 새 게이트웨이 '{self.gateway_name}' 생성 완료!")

        except ClientError as e:
            if e.response['Error']['Code'] == 'ConflictException':
                print(f"⚠️ ConflictException 발생: {e}")
                print("기존 게이트웨이를 다시 조회합니다...")

                list_response = self.gateway_client.list_gateways()
                gateways = list_response.get('items', [])

                for gateway in gateways:
                    if gateway['name'] == self.gateway_name:
                        print(f"✅ 기존 게이트웨이 '{self.gateway_name}' 찾음!")
                        create_response = gateway
                        break

                if not create_response:
                    raise Exception(f"게이트웨이 '{self.gateway_name}'를 찾을 수 없습니다.")
            else:
                print(f"❌ 게이트웨이 생성 실패: {e}")
                raise e

        # 결과 처리
        if create_response:
            self.gateway_id = create_response["gatewayId"]

            if "gatewayUrl" in create_response:
                self.gateway_url = create_response["gatewayUrl"]
            else:
                self.gateway_url = f"https://{self.gateway_id}.gateway.bedrock-agentcore.{self.region}.amazonaws.com/mcp"

            print(f"\n🎉 게이트웨이 준비 완료!")
            print(f"Gateway ID: {self.gateway_id}")
            print(f"Gateway URL: {self.gateway_url}")
            print(f"Gateway Name: {self.gateway_name}")
            print(f"Status: {create_response.get('status', 'Unknown')}")
        else:
            raise Exception("게이트웨이 생성 또는 조회에 실패했습니다.")

    def create_api_key_credential_provider(self):
        """API KEY 자격 증명 공급자 생성 또는 재사용"""
        print("=== API KEY 자격 증명 공급자 생성 ===")

        try:
            # 기존 provider 목록 확인
            response = self.gateway_client.list_api_key_credential_providers()
            providers = response.get('credentialProviders', [])

            # YouTubeAPIKey로 시작하는 provider 찾기
            youtube_providers = [p for p in providers if p['name'].startswith('YouTubeAPIKey')]

            if youtube_providers:
                # 기존 provider 재사용
                existing_provider = youtube_providers[-1]  # 최신 것 사용
                self.credential_provider_arn = existing_provider['credentialProviderArn']
                provider_name = existing_provider['name']
                print(f"✅ 기존 자격 증명 공급자 재사용: {provider_name}")
                print(f"ARN: {self.credential_provider_arn}")

                # API 키 업데이트 (새 API 키 반영)
                try:
                    self.gateway_client.update_api_key_credential_provider(
                        name=provider_name,
                        apiKey=self.youtube_api_key
                    )
                    print("✅ API 키 업데이트 완료!")
                except Exception as update_error:
                    print(f"⚠️ API 키 업데이트 실패 (무시): {update_error}")

            else:
                # provider가 없으면 새로 생성
                timestamp = int(time.time())
                credential_provider_name = f"YouTubeAPIKey_{timestamp}"

                print(f"새 자격 증명 공급자 생성: {credential_provider_name}")
                response = self.gateway_client.create_api_key_credential_provider(
                    name=credential_provider_name,
                    apiKey=self.youtube_api_key,
                )

                self.credential_provider_arn = response['credentialProviderArn']
                print("✅ 새 자격 증명 공급자 생성 완료!")
                print(f"ARN: {self.credential_provider_arn}")

        except Exception as e:
            print(f"❌ 실패: {e}")
            raise e

    def upload_openapi_spec_to_s3(self) -> str:
        """OpenAPI 스펙을 S3에 업로드"""
        print("=== OpenAPI 스펙 S3 업로드 ===")

        # AWS 계정 ID 검색
        account_id = self.sts_client.get_caller_identity()["Account"]

        # S3 버킷 및 파일 설정
        bucket_name = f'agentcore-gateway-{account_id}-{self.region}'
        file_path = 'assets/youtube_api_openapi.json'
        object_key = 'youtube_api_openapi.json'

        try:
            # S3 버킷 생성
            if self.region == "us-east-1":
                self.s3_client.create_bucket(Bucket=bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )

            # 파일 업로드
            with open(file_path, 'rb') as file_data:
                self.s3_client.put_object(
                    Bucket=bucket_name,
                    Key=object_key,
                    Body=file_data
                )

            openapi_s3_uri = f's3://{bucket_name}/{object_key}'
            print(f'업로드된 객체 S3 URI: {openapi_s3_uri}')
            return openapi_s3_uri

        except Exception as e:
            print(f'파일 업로드 오류: {e}')
            raise e

    def create_gateway_target(self, openapi_s3_uri: str):
        """아웃바운드 인증 구성 및 게이트웨이 타겟 생성"""
        print("=== 게이트웨이 타겟 생성 ===")

        # 설정
        youtube_openapi_s3_target_config = {
            "mcp": {
                "openApiSchema": {
                    "s3": {
                        "uri": openapi_s3_uri
                    }
                }
            }
        }

        api_key_credential_config = [
            {
                "credentialProviderType": "API_KEY",
                "credentialProvider": {
                    "apiKeyCredentialProvider": {
                        "credentialParameterName": "key",
                        "providerArn": self.credential_provider_arn,
                        "credentialLocation": "QUERY_PARAMETER",
                    }
                }
            }
        ]

        # 고유한 타겟 이름 생성 (AWS 이름 규칙: 하이픈만 사용)
        timestamp = int(time.time())
        target_name = f'YouTubeCompleteAPI-{timestamp}'

        try:
            print(f"새 게이트웨이 타겟 생성: {target_name}")
            response = self.gateway_client.create_gateway_target(
                gatewayIdentifier=self.gateway_id,
                name=target_name,
                description='Complete YouTube API with searchVideos and getVideoDetails',
                targetConfiguration=youtube_openapi_s3_target_config,
                credentialProviderConfigurations=api_key_credential_config
            )
            print(f"✅ 새 게이트웨이 타겟 '{target_name}' 생성 완료!")
            print(f"   포함된 도구: searchVideos, getVideoDetails")

        except Exception as e:
            print(f"❌ 게이트웨이 타겟 생성 실패: {e}")
            raise e

        print(f"게이트웨이 타겟 '{target_name}' 준비 완료!")

    def get_access_token(self):
        """Amazon Cognito에서 액세스 토큰 요청"""
        print("=== 액세스 토큰 요청 ===")
        print("Amazon Cognito 인증자에서 액세스 토큰 요청 중...도메인 이름 전파가 완료될 때까지 일정 시간 실패할 수 있습니다")

        token_response = utils.get_token(
            self.user_pool_id, self.client_id, self.client_secret,
            self.scope_string, self.region
        )
        self.access_token = token_response["access_token"]
        print("토큰 응답:", self.access_token[:50] + "...")

    def create_streamable_http_transport(self):
        """Streamable HTTP 전송 생성"""
        return streamablehttp_client(
            self.gateway_url,
            headers={"Authorization": f"Bearer {self.access_token}"}
        )

    def _create_search_prompt(self, query: str) -> str:
        """YouTube 검색용 프롬프트 생성"""
        return f"""'{query}' 키워드를 분석해서 관련된 여러 검색어로 YouTube를 검색해주세요.

예를 들어 "2025년 한국의 신선식품"이면:
- '2025년 신선식품 트렌드'
- '한국 농산물 동향 2025' 
- '신선식품 시장 전망'
등으로 검색하세요.

**중요: 검색 결과를 반드시 다음 JSON 형식으로만 출력하세요:**

```json
{{
  "videos": [
    {{
      "title": "비디오 제목",
      "channel": "채널명",
      "url": "https://youtube.com/watch?v=비디오ID",
      "video_id": "비디오ID",
      "view_count": 1234567,
      "like_count": 5678,
      "comment_count": 890,
      "published_date": "2024-01-15",
      "duration": "PT10M30S",
      "description": "비디오 설명 요약"
    }}
  ]
}}
```

각 비디오마다 다음 메타데이터를 반드시 포함하세요:
- 조회수 (view_count): 숫자로 표시
- 좋아요 수 (like_count): 숫자로 표시  
- 댓글 수 (comment_count): 숫자로 표시
- 업로드 날짜 (published_date): YYYY-MM-DD 형식
- 영상 길이 (duration): YouTube 표준 형식
- 설명 요약 (description): 핵심 내용 1-2줄

각 검색어별로 최소 3-5개의 관련 비디오를 찾아서 위 JSON 형식으로 정리해주세요.

총 15-20개 정도의 영상을 찾아주세요."""

    def _create_bedrock_model(self) -> BedrockModel:
        """Bedrock 모델 생성"""
        model_id = os.getenv('BEDROCK_MODEL_ID', Config.DEFAULT_MODEL_ID)
        temperature = float(os.getenv('BEDROCK_TEMPERATURE', Config.DEFAULT_TEMPERATURE))
        
        return BedrockModel(
            model_id=model_id,
            temperature=temperature,
        )

    def analyze_csv_and_extract_keywords(self, csv_path: str = "./data/Dat-fresh-food-claude.csv") -> List[str]:
        """CSV 파일을 분석하여 YouTube 검색용 키워드 추출"""
        import pandas as pd
        from collections import Counter
        
        try:
            print(f"=== CSV 파일 분석: {csv_path} ===")
            df = pd.read_csv(csv_path)
            print(f"총 {len(df)}개 레코드 발견")
            
            keywords = []
            
            # 1. 상품명 (Style) 분석
            if 'Style' in df.columns:
                products = df['Style'].value_counts().head(10)
                print(f"주요 상품: {list(products.index)}")
                keywords.extend([f"{product} 시장 동향" for product in products.index])
                keywords.extend([f"{product} 트렌드" for product in products.index[:5]])
            
            # 2. 카테고리 분석
            if 'Category' in df.columns:
                categories = df['Category'].value_counts()
                print(f"주요 카테고리: {list(categories.index)}")
                keywords.extend([f"{cat} 시장 전망 2025" for cat in categories.index])
                keywords.extend([f"신선 {cat} 동향" for cat in categories.index])
            
            # 3. 지역별 분석
            if 'ship-state' in df.columns:
                regions = df['ship-state'].value_counts().head(5)
                print(f"주요 지역: {list(regions.index)}")
                keywords.extend([f"{region} 신선식품 시장" for region in regions.index])
            
            # 4. 연령대별 선호도 분석
            if 'Age Group' in df.columns:
                age_groups = df['Age Group'].value_counts()
                print(f"주요 연령대: {list(age_groups.index)}")
                keywords.extend([f"{age} 식품 트렌드" for age in age_groups.index[:3]])
            
            # 5. 프로모션 분석
            if 'promotion-ids' in df.columns:
                promos = df[df['promotion-ids'].notna()]['promotion-ids'].value_counts()
                if len(promos) > 0:
                    print(f"활성 프로모션: {list(promos.index)}")
                    keywords.extend(["신선식품 할인 트렌드", "온라인 식품 마케팅"])
            
            # 6. 일반적인 신선식품 키워드 추가
            general_keywords = [
                "2025년 신선식품 시장 전망",
                "온라인 신선식품 배송",
                "유기농 식품 트렌드",
                "신선식품 이커머스",
                "건강식품 소비 패턴"
            ]
            keywords.extend(general_keywords)
            
            # 중복 제거 및 상위 15개 선택
            unique_keywords = list(dict.fromkeys(keywords))[:15]
            
            print(f"추출된 키워드 ({len(unique_keywords)}개):")
            for i, keyword in enumerate(unique_keywords, 1):
                print(f"  {i}. {keyword}")
            
            return unique_keywords
            
        except Exception as e:
            print(f"CSV 분석 중 오류: {e}")
            # 기본 키워드 반환
            return [Config.DEFAULT_QUERY]

    def _generate_optimized_queries_with_llm(self, keyword: str, csv_context: dict) -> List[str]:
        """LLM을 활용해 YouTube 친화적 검색어 생성"""

        import json
        import boto3

        # Bedrock Runtime 클라이언트 직접 사용
        bedrock_runtime = boto3.client('bedrock-runtime', region_name=self.region)

        prompt = f"""당신은 YouTube 검색 전문가입니다.

**목표**: "{keyword}" 키워드로 한국 YouTube에서 실제 관련 영상을 찾을 수 있는 검색어 3개 생성

**CSV 데이터 컨텍스트**:
- 주요 상품: {csv_context.get('products', [])[:5]}
- 카테고리: {csv_context.get('categories', [])}
- 주요 지역: {csv_context.get('regions', [])[:3]}

**검색어 생성 원칙**:
1. 한국 YouTube에서 실제 콘텐츠가 많은 키워드 사용
2. 온라인 쇼핑몰/이커머스 시장 분석 콘텐츠를 찾을 수 있는 용어
3. "온라인 판매", "이커머스", "배송", "소비 트렌드", "구매 패턴" 같은 실제 사용 용어 활용
4. 너무 구체적이지 않고, 너무 일반적이지 않은 균형
5. CSV 컨텍스트를 참고하되 YouTube에서 검색 가능한 수준으로 조정
6. **중요**: 동음이의어 방지를 위해 명확한 한정어 사용 (예: "사과" → "사과 과일", "배" → "배 과일")

**중요**: 검색어는 실제 YouTube 크리에이터들이 사용하는 용어여야 합니다.

검색어를 JSON 배열로만 반환하세요 (다른 설명 없이):
["검색어1", "검색어2", "검색어3"]"""

        try:
            # Bedrock API 직접 호출
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 500,
                "temperature": 0.7,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }

            response = bedrock_runtime.invoke_model(
                modelId=Config.DEFAULT_MODEL_ID,
                body=json.dumps(request_body)
            )

            response_body = json.loads(response['body'].read())
            response_text = response_body['content'][0]['text']

            # JSON 배열 추출
            import re
            json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            if json_match:
                queries = json.loads(json_match.group(0))
                print(f"✅ LLM 생성 검색어: {queries}")
                return queries[:3]
            else:
                print(f"⚠️ LLM 응답 파싱 실패, 기본 검색어 사용")
                return [keyword]

        except Exception as e:
            print(f"⚠️ LLM 검색어 생성 실패: {e}")
            # Fallback: 규칙 기반 검색어
            base_word = keyword.split()[0]
            fallback_queries = [
                f"{base_word} 가격 동향",
                f"{base_word} 시장 전망 2025",
                f"{base_word} 재배 현황"
            ]
            print(f"📋 Fallback 검색어 사용: {fallback_queries}")
            return fallback_queries

    def _extract_video_ids_from_response(self, response_text: str) -> List[str]:
        """에이전트 응답에서 video_id 추출"""
        import re
        import json

        video_ids = []

        # 패턴 1: {"video_ids": ["id1", "id2", ...]}
        json_match = re.search(r'\{"video_ids":\s*\[(.*?)\]\}', response_text, re.DOTALL)
        if json_match:
            ids_str = json_match.group(1)
            # 따옴표로 감싸진 ID 추출
            video_ids = re.findall(r'"([a-zA-Z0-9_-]{11})"', ids_str)

        # 패턴 2: "video_id": "xxxxx" 형식
        if not video_ids:
            video_ids = re.findall(r'"video[_-]?[iI][dD]"\s*:\s*"([a-zA-Z0-9_-]{11})"', response_text)

        # 패턴 3: YouTube ID만 (11자리)
        if not video_ids:
            video_ids = re.findall(r'\b([a-zA-Z0-9_-]{11})\b', response_text)

        # 중복 제거
        unique_ids = list(dict.fromkeys(video_ids))

        print(f"추출된 video_id ({len(unique_ids)}개): {unique_ids[:5]}...")
        return unique_ids

    def _calculate_quality_score(self, videos: List[dict], keyword: str) -> float:
        """검색 결과 품질 점수 계산"""
        if not videos:
            return 0.0

        total_score = 0
        keyword_parts = keyword.split()

        for video in videos:
            score = 0

            # 1. 키워드 관련성 (제목/설명)
            title = video.get('title', '').lower()
            description = video.get('description', '').lower()

            for part in keyword_parts:
                if part.lower() in title:
                    score += 15
                if part.lower() in description:
                    score += 5

            # 2. 조회수 기준 (신뢰성)
            view_count = video.get('view_count', 0)
            if isinstance(view_count, str):
                view_count = int(view_count.replace(',', ''))

            if view_count > 1000:
                score += 10
            if view_count > 10000:
                score += 15
            if view_count > 50000:
                score += 20

            # 3. 업로드 날짜 (최신성)
            pub_date = str(video.get('published_date', ''))
            if '2024' in pub_date or '2025' in pub_date:
                score += 20
            elif '2023' in pub_date:
                score += 10

            # 4. 채널 신뢰도
            trusted_keywords = ['KREI', 'KBS', 'YTN', '한국농수산', '가락시장', '농업', '농촌', '시장']
            channel = video.get('channel', '')
            if any(kw in channel for kw in trusted_keywords):
                score += 15

            total_score += score

        avg_score = total_score / len(videos)
        print(f"📊 품질 점수: {avg_score:.1f}/100")

        return avg_score

    def run_smart_youtube_search(self, csv_path: str = "./data/Dat-fresh-food-claude.csv") -> Dict[str, Any]:
        """LLM 기반 검색어 최적화 + 2단계 검색 시스템"""
        print("=== 🚀 LLM 기반 스마트 YouTube 검색 시작 ===")

        # 1. CSV에서 키워드 및 컨텍스트 추출
        keywords = self.analyze_csv_and_extract_keywords(csv_path)

        # CSV 컨텍스트 구성
        csv_context = {}
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            csv_context = {
                'products': df['Style'].value_counts().head(10).index.tolist() if 'Style' in df.columns else [],
                'categories': df['Category'].value_counts().index.tolist() if 'Category' in df.columns else [],
                'regions': df['ship-state'].value_counts().head(5).index.tolist() if 'ship-state' in df.columns else []
            }
        except Exception as e:
            print(f"⚠️ CSV 컨텍스트 추출 실패: {e}")

        # 2. MCP 클라이언트 및 에이전트 초기화
        client = MCPClient(self.create_streamable_http_transport)
        model = self._create_bedrock_model()

        all_results = {}

        with client:
            tools = client.list_tools_sync()
            agent = Agent(model=model, tools=tools)

            print(f"에이전트에 로드된 도구: {agent.tool_names}")

            # 상위 5개 키워드만 검색 (할당량 최적화)
            search_limit = min(5, len(keywords))
            top_keywords = keywords[:search_limit]

            for i, keyword in enumerate(top_keywords, 1):
                print(f"\n{'='*60}")
                print(f"🔍 검색 {i}/{len(top_keywords)}: {keyword}")
                print(f"{'='*60}")

                # 3. LLM으로 최적화된 검색어 생성
                optimized_queries = self._generate_optimized_queries_with_llm(keyword, csv_context)

                keyword_result = {
                    'original_keyword': keyword,
                    'optimized_queries': optimized_queries,
                    'videos': [],
                    'quality_score': 0,
                    'attempts': 0
                }

                # 4. 최적화된 검색어로 순차 검색 (품질 기준 통과시 중단)
                for attempt, query in enumerate(optimized_queries, 1):
                    print(f"\n--- 시도 {attempt}/{len(optimized_queries)}: '{query}' ---")

                    try:
                        # 4-1. searchVideos로 검색 (1회만)
                        search_prompt = f"""YouTube에서 '{query}'를 검색하세요.

**중요 지시사항**:
1. searchVideos 도구를 정확히 1번만 호출하세요
2. 파라미터:
   - q: "{query}"
   - part: "snippet"
   - maxResults: 10
   - order: "relevance"
3. 검색 결과의 videoId만 추출해서 리스트로 반환하세요

응답 형식 (JSON):
{{"video_ids": ["videoId1", "videoId2", ...]}}

절대 여러 번 검색하지 마세요!"""

                        search_response = agent(search_prompt)
                        video_ids = self._extract_video_ids_from_response(str(search_response))

                        if not video_ids:
                            print(f"⚠️ video_id 추출 실패, 다음 검색어 시도")
                            continue

                        # 4-2. getVideoDetails로 메타데이터 수집
                        print(f"📥 {len(video_ids)}개 영상의 상세 정보 수집 중...")

                        details_prompt = f"""다음 video ID들의 상세 정보를 getVideoDetails로 조회하세요.

**video IDs**: {','.join(video_ids[:10])}

**파라미터**:
- id: "{','.join(video_ids[:10])}"
- part: "snippet,statistics,contentDetails"

**응답 형식 (JSON)**:
```json
{{
  "videos": [
    {{
      "title": "제목",
      "channel": "채널명",
      "url": "https://youtube.com/watch?v=VIDEO_ID",
      "video_id": "VIDEO_ID",
      "view_count": 숫자,
      "like_count": 숫자,
      "comment_count": 숫자,
      "published_date": "YYYY-MM-DD",
      "duration": "PTXXMXXS",
      "description": "설명 요약"
    }}
  ]
}}
```

정확히 1회만 호출하세요!"""

                        details_response = agent(details_prompt)

                        # 4-3. 응답 파싱
                        import re
                        import json

                        response_text = str(details_response)
                        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                        if not json_match:
                            json_match = re.search(r'(\{.*?"videos".*?\})', response_text, re.DOTALL)

                        if json_match:
                            data = json.loads(json_match.group(1))
                            videos = data.get('videos', [])

                            # 4-4. 품질 검증
                            if len(videos) >= 3:
                                quality_score = self._calculate_quality_score(videos, keyword)

                                if quality_score >= 30:  # 품질 기준 통과
                                    print(f"✅ 품질 기준 통과! (점수: {quality_score:.1f})")
                                    keyword_result['videos'] = videos
                                    keyword_result['quality_score'] = quality_score
                                    keyword_result['attempts'] = attempt
                                    keyword_result['successful_query'] = query
                                    break
                                else:
                                    print(f"⚠️ 품질 부족 (점수: {quality_score:.1f} < 30), 다음 검색어 시도")
                            else:
                                print(f"⚠️ 결과 부족 ({len(videos)}개 < 3개), 다음 검색어 시도")
                        else:
                            print(f"⚠️ JSON 파싱 실패, 다음 검색어 시도")

                    except Exception as e:
                        print(f"❌ 검색 오류: {e}")
                        continue

                # 5. 결과 저장
                if keyword_result['videos']:
                    print(f"\n🎉 '{keyword}' 검색 성공! (최종 점수: {keyword_result['quality_score']:.1f})")
                else:
                    print(f"\n⚠️ '{keyword}' 검색 실패 - 모든 시도 소진")

                all_results[keyword] = keyword_result

        return {
            'csv_analysis': f"분석된 CSV: {csv_path}",
            'extracted_keywords': keywords,
            'search_results': all_results,
            'total_keywords': len(keywords),
            'searched_keywords': len(top_keywords),
            'csv_context': csv_context
        }


    def save_raw_youtube_data(self, results: Dict[str, Any]):
        """main.py 분석용 원시 YouTube 데이터 저장 (LLM 기반 검색 구조)"""
        import json

        # data 디렉토리 생성
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)

        # 검색 결과 재구성 (main.py 호환 형식)
        search_results_formatted = {}
        total_videos = 0
        successful_searches = 0

        for keyword, keyword_result in results.get('search_results', {}).items():
            if isinstance(keyword_result, dict) and keyword_result.get('videos'):
                # 성공한 검색
                videos = keyword_result['videos']
                search_results_formatted[keyword] = {
                    'keyword': keyword,
                    'successful_query': keyword_result.get('successful_query', keyword),
                    'optimized_queries': keyword_result.get('optimized_queries', []),
                    'quality_score': keyword_result.get('quality_score', 0),
                    'attempts': keyword_result.get('attempts', 1),
                    'videos': videos
                }
                total_videos += len(videos)
                successful_searches += 1
            else:
                # 실패한 검색
                search_results_formatted[keyword] = {
                    'keyword': keyword,
                    'error': '검색 실패 - 품질 기준 미달',
                    'videos': []
                }

        # 1. YouTube 원시 데이터 저장 (JSON)
        youtube_data_path = os.path.join(data_dir, "youtube_raw_data.json")
        youtube_data = {
            "collection_time": time.strftime('%Y-%m-%d %H:%M:%S'),
            "csv_source": results.get('csv_analysis', ''),
            "csv_context": results.get('csv_context', {}),
            "total_keywords": results.get('total_keywords', 0),
            "searched_keywords": results.get('searched_keywords', 0),
            "successful_searches": successful_searches,
            "total_videos_collected": total_videos,
            "search_results": search_results_formatted,
            "metadata": {
                "collection_method": "LLM-optimized search + YouTube Data API v3",
                "data_format": "JSON with quality scores and metadata",
                "purpose": "Raw data for main.py analysis",
                "features": [
                    "LLM-generated search queries",
                    "2-stage search (searchVideos + getVideoDetails)",
                    "Quality validation with scoring",
                    "Automatic retry with alternative queries"
                ]
            }
        }

        with open(youtube_data_path, 'w', encoding='utf-8') as f:
            json.dump(youtube_data, f, ensure_ascii=False, indent=2)

        # 2. 추출된 키워드 저장 (별도 파일)
        keywords_path = os.path.join(data_dir, "csv_keywords.json")

        extracted_keywords = results.get('extracted_keywords', [])
        csv_context = results.get('csv_context', {})

        keywords_data = {
            "extraction_time": time.strftime('%Y-%m-%d %H:%M:%S'),
            "source_csv": results.get('csv_analysis', ''),
            "csv_context": csv_context,
            "keywords": extracted_keywords,
            "keyword_count": len(extracted_keywords),
            "categories": {
                "product_trends": [k for k in extracted_keywords if '트렌드' in k],
                "market_outlook": [k for k in extracted_keywords if '전망' in k or '동향' in k],
                "regional_analysis": [k for k in extracted_keywords if any(region in k for region in csv_context.get('regions', []))],
                "general_keywords": [k for k in extracted_keywords if '신선식품' in k or '이커머스' in k]
            }
        }

        with open(keywords_path, 'w', encoding='utf-8') as f:
            json.dump(keywords_data, f, ensure_ascii=False, indent=2)

        # 3. 상세 요약 텍스트 파일
        summary_path = os.path.join(data_dir, "data_collection_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("YouTube 데이터 수집 요약 (LLM 기반 검색)\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"수집 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"CSV 소스: {results.get('csv_analysis', 'N/A')}\n\n")

            f.write("--- 검색 통계 ---\n")
            f.write(f"추출된 키워드 수: {results.get('total_keywords', 0)}\n")
            f.write(f"검색된 키워드 수: {results.get('searched_keywords', 0)}\n")
            f.write(f"성공한 검색: {successful_searches}\n")
            f.write(f"실패한 검색: {results.get('searched_keywords', 0) - successful_searches}\n")
            f.write(f"총 수집 영상 수: {total_videos}\n")
            f.write(f"평균 영상 수/키워드: {total_videos/successful_searches if successful_searches > 0 else 0:.1f}\n\n")

            f.write("--- CSV 컨텍스트 ---\n")
            f.write(f"주요 상품: {csv_context.get('products', [])[:5]}\n")
            f.write(f"카테고리: {csv_context.get('categories', [])}\n")
            f.write(f"주요 지역: {csv_context.get('regions', [])}\n\n")

            f.write("--- 키워드별 검색 결과 ---\n")
            for keyword, result in search_results_formatted.items():
                if result.get('videos'):
                    f.write(f"\n✅ {keyword}\n")
                    f.write(f"   최종 검색어: {result.get('successful_query', 'N/A')}\n")
                    f.write(f"   품질 점수: {result.get('quality_score', 0):.1f}/100\n")
                    f.write(f"   시도 횟수: {result.get('attempts', 0)}\n")
                    f.write(f"   수집 영상: {len(result['videos'])}개\n")
                else:
                    f.write(f"\n❌ {keyword}\n")
                    f.write(f"   상태: 검색 실패\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("수집된 데이터 파일\n")
            f.write("=" * 60 + "\n")
            f.write(f"  📄 youtube_raw_data.json (main.py 분석용)\n")
            f.write(f"  📄 csv_keywords.json (키워드 분석용)\n")
            f.write(f"  📄 data_collection_summary.txt (이 파일)\n\n")

            f.write("다음 단계:\n")
            f.write("  python main.py 실행하여 분석 시작\n")

        print(f"\n{'='*60}")
        print("✅ 원시 데이터 저장 완료")
        print(f"{'='*60}")
        print(f"📄 YouTube 데이터: {youtube_data_path}")
        print(f"📄 키워드 데이터: {keywords_path}")
        print(f"📄 수집 요약: {summary_path}")
        print(f"\n📊 수집 통계:")
        print(f"   - 성공: {successful_searches}/{results.get('searched_keywords', 0)}")
        print(f"   - 총 영상: {total_videos}개")
        print(f"   - 평균 품질: {sum(r.get('quality_score', 0) for r in search_results_formatted.values() if r.get('videos'))/successful_searches if successful_searches > 0 else 0:.1f}/100")

        return {
            "youtube_data_file": youtube_data_path,
            "keywords_file": keywords_path,
            "summary_file": summary_path,
            "statistics": {
                "successful_searches": successful_searches,
                "total_videos": total_videos,
                "searched_keywords": results.get('searched_keywords', 0)
            }
        }

    def _analyze_youtube_metadata(self, response_text: str) -> Dict[str, Any]:
        """YouTube 응답에서 메타데이터 분석"""
        import json
        import re
        
        try:
            # JSON 블록 찾기
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if not json_match:
                json_match = re.search(r'(\{.*?"videos".*?\})', response_text, re.DOTALL)
            
            if not json_match:
                return {}
                
            data = json.loads(json_match.group(1))
            videos = data.get('videos', [])
            
            if not videos:
                return {}
            
            # 메타데이터 분석
            total_views = 0
            total_likes = 0
            total_comments = 0
            dates = []
            channels = {}
            
            for video in videos:
                # 조회수 분석
                views = video.get('view_count', 0)
                if isinstance(views, (int, str)):
                    try:
                        total_views += int(str(views).replace(',', ''))
                    except:
                        pass
                
                # 좋아요 분석
                likes = video.get('like_count', 0)
                if isinstance(likes, (int, str)):
                    try:
                        total_likes += int(str(likes).replace(',', ''))
                    except:
                        pass
                
                # 댓글 분석
                comments = video.get('comment_count', 0)
                if isinstance(comments, (int, str)):
                    try:
                        total_comments += int(str(comments).replace(',', ''))
                    except:
                        pass
                
                # 날짜 분석
                pub_date = video.get('published_date')
                if pub_date:
                    dates.append(pub_date)
                
                # 채널 분석
                channel = video.get('channel', '')
                if channel:
                    channels[channel] = channels.get(channel, 0) + 1
            
            # 분석 결과
            analysis = {
                'total_videos': len(videos),
                'total_views': total_views,
                'avg_views': total_views // len(videos) if videos else 0,
                'total_likes': total_likes,
                'avg_likes': total_likes // len(videos) if videos else 0,
                'total_comments': total_comments,
                'avg_comments': total_comments // len(videos) if videos else 0,
                'latest_date': max(dates) if dates else 'N/A',
                'top_channel': max(channels.items(), key=lambda x: x[1])[0] if channels else 'N/A'
            }
            
            return analysis
            
        except Exception as e:
            print(f"메타데이터 분석 중 오류: {e}")
            return {}

    def delete_gateway_targets(self):
        """Gateway의 모든 타겟 완전 삭제 (반복 확인으로 비동기 삭제 처리)"""
        if not self.gateway_id:
            print("⚠️ Gateway ID가 없습니다.")
            return True

        print(f"\n=== Gateway 타겟 완전 삭제: {self.gateway_id} ===")

        max_iterations = 5  # 최대 5번 반복
        for iteration in range(max_iterations):
            try:
                response = self.gateway_client.list_gateway_targets(
                    gatewayIdentifier=self.gateway_id
                )

                targets = response.get('items', [])

                if not targets:
                    print("✅ 모든 타겟이 삭제되었습니다!")
                    return True

                print(f"\n반복 {iteration + 1}/{max_iterations}: 발견된 타겟 수 = {len(targets)}")

                # 각 타겟 삭제
                deleted_count = 0
                for target in targets:
                    target_id = target.get('targetId')
                    target_name = target.get('name', 'Unknown')

                    try:
                        print(f"  타겟 삭제 중: {target_name} (ID: {target_id})")
                        self.gateway_client.delete_gateway_target(
                            gatewayIdentifier=self.gateway_id,
                            targetId=target_id
                        )
                        print(f"    ✅ 삭제 요청 완료")
                        deleted_count += 1

                    except ClientError as e:
                        error_code = e.response.get('Error', {}).get('Code', '')
                        if 'NotFound' in error_code or 'ResourceNotFound' in error_code:
                            print(f"    ⚠️ 이미 삭제됨")
                        else:
                            print(f"    ❌ 삭제 실패: {e}")

                print(f"  삭제 요청 완료: {deleted_count}/{len(targets)}개")

                # 삭제 완료 대기 (점진적으로 증가)
                wait_time = 5 + (iteration * 2)
                print(f"  삭제 완료 대기 중... ({wait_time}초)")
                time.sleep(wait_time)

            except Exception as e:
                print(f"❌ 타겟 조회 중 오류: {e}")
                return False

        # 최종 확인
        try:
            response = self.gateway_client.list_gateway_targets(
                gatewayIdentifier=self.gateway_id
            )
            remaining = response.get('items', [])

            if remaining:
                print(f"\n⚠️ 최대 반복 횟수 초과. 아직 {len(remaining)}개 타겟이 남아있습니다:")
                for t in remaining:
                    print(f"  - {t.get('name', 'Unknown')}")
                print("\n계속 진행합니다. (남은 타겟은 무시됨)")
                return True
            else:
                print("\n✅ 모든 타겟이 성공적으로 삭제되었습니다!")
                return True

        except Exception as e:
            print(f"❌ 최종 확인 중 오류: {e}")
            return False

    def delete_gateway(self):
        """생성된 Gateway 삭제 (선택사항)"""
        if not self.gateway_id:
            print("⚠️ 삭제할 Gateway ID가 없습니다.")
            return

        print(f"=== Gateway 삭제: {self.gateway_id} ===")

        # 1. 먼저 연결된 타겟들 삭제
        self.delete_gateway_targets()

        # 2. Gateway 상태 확인 및 삭제 재시도
        for attempt in range(Config.MAX_RETRIES):
            try:
                print(f"Gateway 삭제 시도 {attempt + 1}/{Config.MAX_RETRIES}")
                response = self.gateway_client.delete_gateway(
                    gatewayIdentifier=self.gateway_id
                )
                print(f"✅ Gateway '{self.gateway_name}' (ID: {self.gateway_id}) 삭제 완료!")
                return
                
            except ClientError as e:
                error_message = str(e)
                if "has targets associated" in error_message:
                    print(f"❌ 시도 {attempt + 1}: 타겟이 여전히 연결되어 있습니다. {Config.RETRY_DELAY_SECONDS}초 대기 후 재시도...")
                    time.sleep(Config.RETRY_DELAY_SECONDS)
                elif "ResourceNotFoundException" in error_message:
                    print("✅ Gateway가 이미 삭제되었습니다.")
                    return
                else:
                    print(f"❌ Gateway 삭제 실패: {e}")
                    if attempt == Config.MAX_RETRIES - 1:
                        print("❌ 최대 재시도 횟수 초과. 수동으로 삭제하세요.")
                        
            except Exception as e:
                print(f"❌ 예상치 못한 오류로 Gateway 삭제 실패: {e}")
                if attempt == Config.MAX_RETRIES - 1:
                    print("❌ 최대 재시도 횟수 초과. 수동으로 삭제하세요.")

    def delete_credential_provider(self):
        """생성된 자격 증명 공급자 삭제"""
        if not self.credential_provider_arn:
            print("⚠️ 삭제할 자격 증명 공급자 ARN이 없습니다.")
            return

        print(f"=== 자격 증명 공급자 삭제 ===")
        try:
            # ARN에서 provider name 추출
            provider_name = self.credential_provider_arn.split('/')[-1]

            response = self.gateway_client.delete_api_key_credential_provider(
                name=provider_name
            )
            print(f"✅ 자격 증명 공급자 '{provider_name}' 삭제 완료!")
        except ClientError as e:
            print(f"❌ 자격 증명 공급자 삭제 실패: {e}")
        except Exception as e:
            print(f"❌ 예상치 못한 오류로 자격 증명 공급자 삭제 실패: {e}")

    def delete_gateway_completely(self):
        """게이트웨이를 완전히 삭제"""
        print("=== 게이트웨이 완전 삭제 ===")
        try:
            # 1. 모든 타겟 삭제
            targets = self.gateway_client.list_gateway_targets(gatewayIdentifier=self.gateway_id)
            for target in targets.get('gatewayTargets', []):
                target_name = target['name']
                print(f"타겟 삭제: {target_name}")
                self.gateway_client.delete_gateway_target(
                    gatewayIdentifier=self.gateway_id,
                    targetName=target_name
                )
            
            import time
            time.sleep(5)
            
            # 2. 게이트웨이 삭제
            print(f"게이트웨이 삭제: {self.gateway_id}")
            self.gateway_client.delete_gateway(gatewayIdentifier=self.gateway_id)
            
            time.sleep(5)
            print("✅ 게이트웨이 완전 삭제 완료")
            
        except Exception as e:
            print(f"게이트웨이 삭제 중 오류: {e}")

    def cleanup_all_targets(self):
        """모든 기존 타겟 정리"""
        print("=== 모든 기존 타겟 정리 ===")
        try:
            targets = self.gateway_client.list_gateway_targets(gatewayIdentifier=self.gateway_id)
            target_list = targets.get('gatewayTargets', [])
            
            if not target_list:
                print("정리할 타겟이 없습니다.")
                return
                
            print(f"총 {len(target_list)}개 타겟 발견")
            
            for target in target_list:
                target_name = target['name']
                print(f"타겟 삭제 중: {target_name}")
                try:
                    self.gateway_client.delete_gateway_target(
                        gatewayIdentifier=self.gateway_id,
                        targetName=target_name
                    )
                    print(f"✅ 삭제 완료: {target_name}")
                except Exception as e:
                    print(f"❌ 삭제 실패: {target_name} - {e}")
            
            # 삭제 완료 확인
            import time
            for i in range(10):  # 최대 10초 대기
                time.sleep(1)
                remaining = self.gateway_client.list_gateway_targets(gatewayIdentifier=self.gateway_id)
                if not remaining.get('gatewayTargets', []):
                    print("✅ 모든 타겟 정리 완료")
                    return
                print(f"대기 중... ({i+1}/10)")
            
            print("⚠️ 일부 타겟이 아직 남아있을 수 있습니다.")
            
        except Exception as e:
            print(f"타겟 정리 중 오류: {e}")

    def force_recreate_gateway(self, iam_role: Dict[str, Any]):
        """게이트웨이를 완전히 삭제하고 videos API 포함해서 새로 생성"""
        print("=== 게이트웨이 완전 재생성 ===")
        
        try:
            # 1. 기존 게이트웨이 타겟들 먼저 삭제
            if hasattr(self, 'gateway_id') and self.gateway_id:
                try:
                    print("기존 타겟들 삭제 중...")
                    targets = self.gateway_client.list_gateway_targets(gatewayIdentifier=self.gateway_id)
                    for target in targets.get('gatewayTargets', []):
                        target_name = target['name']
                        print(f"타겟 삭제: {target_name}")
                        try:
                            self.gateway_client.delete_gateway_target(
                                gatewayIdentifier=self.gateway_id,
                                targetName=target_name
                            )
                            print(f"✅ 타겟 삭제 완료: {target_name}")
                        except Exception as target_error:
                            print(f"타겟 삭제 실패: {target_name} - {target_error}")
                    
                    import time
                    time.sleep(5)  # 타겟 삭제 완료 대기
                    
                    # 2. 게이트웨이 삭제
                    print(f"게이트웨이 삭제: {self.gateway_id}")
                    self.gateway_client.delete_gateway(gatewayIdentifier=self.gateway_id)
                    print("✅ 기존 게이트웨이 삭제 완료")
                    time.sleep(5)  # 삭제 완료 대기
                    
                except Exception as e:
                    print(f"게이트웨이 삭제 실패 (무시): {e}")
            
            # 3. 새 게이트웨이 생성
            self.create_gateway(iam_role)
            
            # 4. 새 OpenAPI 스펙 업로드
            openapi_s3_uri = self.upload_openapi_spec_to_s3()
            
            # 5. 새 게이트웨이 타겟 생성
            self.create_gateway_target(openapi_s3_uri)
            
            print("✅ 게이트웨이 완전 재생성 완료! searchVideos + getVideoDetails 사용 가능")
            
        except Exception as e:
            print(f"❌ 게이트웨이 완전 재생성 실패: {e}")
            raise e
        """기존 게이트웨이를 정리하고 videos API가 포함된 새 게이트웨이 생성"""
        print("=== 게이트웨이 재생성 (videos API 포함) ===")
        
        try:
            # 1. 기존 게이트웨이 타겟 삭제
            try:
                targets = self.gateway_client.list_gateway_targets(gatewayIdentifier=self.gateway_id)
                for target in targets.get('gatewayTargets', []):
                    target_name = target['name']
                    print(f"기존 타겟 삭제 중: {target_name}")
                    try:
                        self.gateway_client.delete_gateway_target(
                            gatewayIdentifier=self.gateway_id,
                            targetName=target_name
                        )
                        print(f"✅ 타겟 삭제 완료: {target_name}")
                        # 삭제 후 잠시 대기
                        import time
                        time.sleep(2)
                    except Exception as delete_error:
                        print(f"타겟 삭제 실패: {target_name} - {delete_error}")
            except Exception as list_error:
                print(f"타겟 목록 조회 실패: {list_error}")
            
            # 2. 새 OpenAPI 스펙 업로드 (강제 덮어쓰기)
            openapi_s3_uri = self.upload_openapi_spec_to_s3()
            
            # 3. 새 타겟명 생성 (충돌 방지)
            import time
            new_target_name = f"DemoOpenAPITargetS3YouTube-{int(time.time())}"
            
            # 4. 새 게이트웨이 타겟 생성
            print(f"새 타겟 생성: {new_target_name}")
            
            youtube_openapi_s3_target_config = {
                "mcp": {
                    "openApiSchema": {
                        "s3": {
                            "uri": openapi_s3_uri
                        }
                    }
                }
            }

            api_key_credential_config = [
                {
                    "credentialProviderType": "API_KEY",
                    "credentialProvider": {
                        "apiKeyCredentialProvider": {
                            "credentialParameterName": "key",
                            "providerArn": self.credential_provider_arn,
                            "credentialLocation": "QUERY_PARAMETER",
                        }
                    }
                }
            ]
            
            response = self.gateway_client.create_gateway_target(
                gatewayIdentifier=self.gateway_id,
                name=new_target_name,
                description='OpenAPI Target with videos API support',
                targetConfiguration=youtube_openapi_s3_target_config,
                credentialProviderConfigurations=api_key_credential_config
            )
            
            print("✅ 게이트웨이 재생성 완료! searchVideos + getVideoDetails 사용 가능")
            
        except Exception as e:
            print(f"❌ 게이트웨이 재생성 실패: {e}")
            raise e

    def cleanup_all_resources(self):
        """모든 생성된 리소스 정리"""
        print("\n=== 전체 리소스 정리 시작 ===")

        # 1. Gateway 삭제 (타겟 포함)
        self.delete_gateway()

        # 2. 자격 증명 공급자 삭제
        self.delete_credential_provider()

        # 참고: IAM 역할, Cognito 풀, S3 버킷은 다른 용도로 재사용 가능하므로 유지
        print("\n💡 참고사항:")
        print("- IAM 역할: 재사용 가능하므로 유지됩니다")
        print("- Cognito 풀: 재사용 가능하므로 유지됩니다")
        print("- S3 버킷: 재사용 가능하므로 유지됩니다")
        print("- 필요시 AWS 콘솔에서 수동으로 정리하세요")

        print("\n✅ 주요 리소스 정리 완료!")


def main():
    """메인 실행 함수"""
    try:
        # AgentCore Gateway Manager 초기화
        manager = AgentCoreGatewayManager()

        # 1. IAM 역할 생성
        iam_role = manager.create_iam_role()

        # 2. Cognito 설정
        manager.setup_cognito()

        # 3. Gateway 생성 또는 기존 것 사용
        print("\n=== 게이트웨이 확인/생성 ===")
        manager.create_gateway(iam_role)

        # 4. 기존 타겟 완전 삭제 (중복 방지 및 최신 OpenAPI 적용)
        print("\n=== 기존 타겟 완전 정리 ===")
        manager.delete_gateway_targets()

        # 5. API Key 자격 증명 공급자 생성
        manager.create_api_key_credential_provider()

        # 6. OpenAPI 스펙 S3 업로드 (getVideoDetails 포함)
        openapi_s3_uri = manager.upload_openapi_spec_to_s3()

        # 7. 새 Gateway 타겟 생성 (searchVideos + getVideoDetails)
        manager.create_gateway_target(openapi_s3_uri)

        # 8. 액세스 토큰 획득
        manager.get_access_token()

        # 9. 스마트 YouTube 검색으로 원시 데이터 수집
        print("\n=== YouTube 원시 데이터 수집 ===")
        results = manager.run_smart_youtube_search()

        # 10. 원시 데이터 저장 (분석용)
        print("=== 분석용 원시 데이터 저장 ===")
        manager.save_raw_youtube_data(results)
        
        print("\n✅ YouTube 원시 데이터 수집 완료!")
        print("📁 다음 파일들이 생성되었습니다:")
        print("   - ./data/youtube_raw_data.json (main.py 분석용)")
        print("   - ./data/csv_keywords.json (추출된 키워드)")
        print("\n🚀 이제 main.py를 실행하여 분석을 시작하세요!")

        # 10. 리소스 정리 (자동 스킵)
        print("\n=== 리소스 정리 ===")
        print("ℹ️ 데모 완료! 리소스는 유지됩니다. 필요시 수동으로 정리하세요.")
        print(f"Gateway ID: {manager.gateway_id}")
        print(f"Gateway Name: {manager.gateway_name}")
        if manager.credential_provider_arn:
            provider_name = manager.credential_provider_arn.split('/')[-1]
            print(f"자격 증명 공급자: {provider_name}")
        
        print("\n💡 리소스를 정리하려면 다음 메서드를 호출하세요:")
        print("manager.cleanup_all_resources()")
        print(f"자격 증명 공급자: {provider_name}")

        print("\n🎉 모든 프로세스가 성공적으로 완료되었습니다!")

        return results

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        raise e


if __name__ == "__main__":
    main()