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
import logging
import re
from typing import Dict, List, Any
from dotenv import load_dotenv

import boto3
import requests
from botocore.exceptions import ClientError

from strands.models import BedrockModel
from mcp.client.streamable_http import streamablehttp_client
from strands.tools.mcp.mcp_client import MCPClient
from strands import Agent

import utils


class AgentCoreGatewayManager:
    """Bedrock AgentCore Gateway 관리 클래스"""

    def __init__(self):
        """초기화 및 환경 설정"""
        # 환경 변수 로드
        load_dotenv()

        # 필수 환경 변수 확인
        self._validate_environment()

        # AWS 설정
        self.region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        os.environ['AWS_DEFAULT_REGION'] = self.region

        # YouTube API 키
        self.youtube_api_key = os.getenv('YOUTUBE_API_KEY')

        # AWS 클라이언트 초기화
        self.gateway_client = boto3.client('bedrock-agentcore-control', region_name=self.region)
        self.cognito = boto3.client("cognito-idp", region_name=self.region)
        self.s3_client = boto3.client('s3')
        self.sts_client = boto3.client('sts')
        self.acps = boto3.client(service_name="bedrock-agentcore-control")

        # Cognito 설정
        self.user_pool_name = "sample-agentcore-gateway-pool"
        self.resource_server_id = "sample-agentcore-gateway-id"
        self.resource_server_name = "sample-agentcore-gateway-name"
        self.client_name = "sample-agentcore-gateway-client"
        self.scopes = [
            {"ScopeName": "gateway:read", "ScopeDescription": "읽기 액세스"},
            {"ScopeName": "gateway:write", "ScopeDescription": "쓰기 액세스"}
        ]
        self.scope_string = f"{self.resource_server_id}/gateway:read {self.resource_server_id}/gateway:write"

        # 상태 변수
        self.gateway_id = None
        self.gateway_url = None
        self.gateway_name = 'DemoGWOpenAPIAPIKeyYouTube'
        self.user_pool_id = None
        self.client_id = None
        self.client_secret = None
        self.access_token = None
        self.credential_provider_arn = None

        # 로깅 설정
        logging.getLogger("strands").setLevel(logging.INFO)
        logging.basicConfig(format="%(levelname)s | %(name)s | %(message)s", handlers=[logging.StreamHandler()])

        print("✅ AgentCore Gateway Manager 초기화 완료")

    def _validate_environment(self):
        """환경 변수 유효성 검사"""
        required_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'YOUTUBE_API_KEY']
        for var in required_vars:
            if not os.getenv(var):
                raise ValueError(f"환경 변수 {var}가 설정되지 않았습니다.")

        print("✅ AWS 자격 증명 및 YouTube API 키가 성공적으로 로드되었습니다.")

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
        """API KEY 자격 증명 공급자 생성"""
        print("=== API KEY 자격 증명 공급자 생성 ===")

        # 고유한 이름으로 생성
        timestamp = int(time.time())
        credential_provider_name = f"YouTubeAPIKey_{timestamp}"

        try:
            print(f"새 이름으로 자격 증명 공급자 생성: {credential_provider_name}")
            response = self.acps.create_api_key_credential_provider(
                name=credential_provider_name,
                apiKey=self.youtube_api_key,
            )

            self.credential_provider_arn = response['credentialProviderArn']
            print("✅ 새 자격 증명 공급자 생성 완료!")
            print(f"ARN: {self.credential_provider_arn}")

        except Exception as e:
            print(f"❌ 생성 실패: {e}")
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

        # 고유한 타겟 이름 생성
        timestamp = int(time.time())
        target_name = f'DemoOpenAPITargetS3YouTube-{timestamp}'

        try:
            print(f"새 게이트웨이 타겟 생성: {target_name}")
            response = self.gateway_client.create_gateway_target(
                gatewayIdentifier=self.gateway_id,
                name=target_name,
                description='OpenAPI Target with S3Uri using SDK - Fresh',
                targetConfiguration=youtube_openapi_s3_target_config,
                credentialProviderConfigurations=api_key_credential_config
            )
            print(f"✅ 새 게이트웨이 타겟 '{target_name}' 생성 완료!")

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

    def extract_video_info(self, agent_response_text: str) -> List[Dict[str, str]]:
        """YouTube 검색 결과에서 필요한 정보만 추출"""
        text = str(agent_response_text)
        videos = []

        print("=== 텍스트 분석 ===")
        print(text[:1000] + "..." if len(text) > 1000 else text)
        print("==================")

        # 각 비디오 블록을 개별적으로 파싱하여 제목과 채널을 매칭
        video_blocks = re.split(r'\n\n\d+\.', text)  # 각 번호별 블록으로 분리
        if len(video_blocks) > 1:
            video_blocks = [video_blocks[0]] + [f"\n{i+1}.{block}" for i, block in enumerate(video_blocks[1:])]

        print(f"=== 비디오 블록 수: {len(video_blocks)} ===")
        for i, block in enumerate(video_blocks):
            print(f"--- 블록 {i} ---")
            print(block[:300] + "..." if len(block) > 300 else block)
            print("---")

        titles = []
        channels = []
        video_ids = []

        for i, block in enumerate(video_blocks):
            print(f"\n=== 블록 {i} 파싱 중 ===")
            print(f"블록 내용: {block[:200]}...")
            # 제목 추출 (여러 패턴 시도)
            title_patterns = [
                r'\*\*제목\*\*:\s*([^\n]+)',      # **제목**: 내용
                r'\d+\.\s*\*\*제목\*\*:\s*([^\n]+)',  # 1. **제목**: 내용
                r'제목":\s*"([^"]+)"',            # 제목": "내용"
                r'\d+\.\s*\*\*([^*\n]+)\*\*'     # 1. **제목**
            ]

            title = None
            for pattern in title_patterns:
                match = re.search(pattern, block)
                if match:
                    title = match.group(1).strip()
                    print(f"제목 찾음 (패턴: {pattern}): '{title}'")
                    break

            if not title:
                print("제목을 찾지 못했습니다.")

            # 채널 추출 (해당 블록 내에서만)
            channel_patterns = [
                r'-\s*채널:\s*([^\n]+)',          # - 채널: 내용
                r'\*\*채널\*\*:\s*([^\n]+)',     # **채널**: 내용
                r'채널":\s*"([^"]+)"',           # 채널": "내용"
                r'-\s*\*\*채널\*\*:\s*([^\n]+)', # - **채널**: 내용
                r'채널:\s*([^\n]+)',             # 채널: 내용
                r'-\s*([^-\n]+)',                # - 채널명 (단순 형태)
            ]

            channel = None
            for pattern in channel_patterns:
                match = re.search(pattern, block)
                if match:
                    channel = match.group(1).strip()
                    print(f"채널 찾음 (패턴: {pattern}): '{channel}'")
                    break

            if not channel:
                print("채널을 찾지 못했습니다.")

            # 비디오 ID 추출
            video_id_patterns = [
                r'\*\*비디오 ID\*\*:\s*([^\n]+)',    # **비디오 ID**: 내용
                r'비디오 ID":\s*"([^"]+)"',         # 비디오 ID": "내용"
                r'watch\?v=([a-zA-Z0-9_-]+)'        # YouTube URL에서 추출
            ]

            video_id = None
            for pattern in video_id_patterns:
                match = re.search(pattern, block)
                if match:
                    video_id = match.group(1).strip()
                    break

            # 유효한 제목이 있는 경우만 추가
            if title:
                titles.append(title)
                channels.append(channel if channel else "알 수 없음")
                video_ids.append(video_id if video_id else None)

        # 기존 방식으로 URL 추출 (fallback용)
        existing_urls = re.findall(r'https://www\.youtube\.com/watch\?v=[^\s\n]+', text)

        print(f"찾은 제목들: {titles}")
        print(f"찾은 채널들: {channels}")
        print(f"찾은 비디오 IDs: {video_ids}")
        print(f"기존 URLs: {existing_urls}")

        # 비디오 정보 구성 - 블록별로 매칭된 정보 사용
        for i, title in enumerate(titles):
            video = {"title": title}

            # 채널 정보 추가
            if i < len(channels):
                video["channel"] = channels[i]

            # YouTube URL 생성
            if i < len(video_ids) and video_ids[i]:
                # 비디오 ID가 있는 경우 직접 URL 생성
                video["url"] = f"https://www.youtube.com/watch?v={video_ids[i]}"
            elif i < len(existing_urls):
                # 기존 방식으로 추출된 URL 사용
                video["url"] = existing_urls[i]
            else:
                # 제목으로 검색 URL 생성
                search_query = title.replace(' ', '+').replace('"', '').replace(':', '')
                video["search_url"] = f"https://www.youtube.com/results?search_query={search_query}"

            videos.append(video)

        print(f"총 추출된 영상 수: {len(videos)}")
        return videos

    def run_youtube_search_agent(self, query: str = "2025년 국내 신선식품 동향") -> Dict[str, Any]:
        """YouTube 검색 에이전트 실행"""
        print("=== YouTube 검색 에이전트 실행 ===")

        # MCP 클라이언트 생성
        client = MCPClient(self.create_streamable_http_transport)

        # Bedrock 모델 설정
        model = BedrockModel(
            model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            temperature=0.7,
        )

        results = {}

        with client:
            tools = client.list_tools_sync()
            agent = Agent(model=model, tools=tools)

            print(f"에이전트에 로드된 도구들: {agent.tool_names}")

            # 도구 목록 확인
            print("\n=== 도구 목록 확인 ===")
            tool_list_response = agent("안녕하세요, 사용 가능한 모든 도구를 나열해 주실 수 있나요?")

            # YouTube 검색 실행
            print(f"\n=== YouTube 검색: {query} ===")
            search_query = f"""'{query}' 키워드를 분석해서 관련된 여러 검색어로 YouTube를 검색해주세요.

예를 들어 "2025년 한국의 신선식품"이면:
- '2025년 신선식품 트렌드'
- '한국 농산물 동향 2025' 
- '신선식품 시장 전망'
등으로 검색하세요.

각 검색 결과마다 다음 정보를 포함해주세요:
1. 제목
2. 채널명
3. YouTube URL (https://www.youtube.com/watch?v=비디오ID)
4. 업로드 날짜

총 15-20개 정도의 영상을 찾아주세요."""
            
            search_response = agent(search_query)

            # 결과 저장 (파싱 없이)
            results['tool_list'] = str(tool_list_response)
            results['search_response'] = str(search_response)

        return results

    def save_results(self, results: Dict[str, Any], filename: str = "youtube_search_results.txt"):
        """결과를 TXT 파일로 저장 (파싱 없이 원본 그대로)"""
        # data 디렉토리 생성
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)

        # 파일 경로를 data 디렉토리 아래로 설정
        filepath = os.path.join(data_dir, filename)
        print(f"=== 결과 저장: {filepath} ===")

        # 원본 응답 그대로 저장
        search_response = results.get('search_response', '')
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=== YouTube 검색 결과 ===\n")
            f.write(f"검색 쿼리: 2025년 국내 신선식품 동향\n")
            f.write(f"검색 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            f.write(search_response)

        print(f"✅ 결과가 {filepath}에 저장되었습니다.")
        return {"saved_file": filepath, "content_length": len(search_response)}

    def delete_gateway_targets(self):
        """Gateway의 모든 타겟 삭제"""
        if not self.gateway_id:
            print("⚠️ Gateway ID가 없습니다.")
            return

        print(f"=== Gateway 타겟들 삭제 중: {self.gateway_id} ===")
        try:
            # Gateway의 모든 타겟 조회
            response = self.gateway_client.list_gateway_targets(
                gatewayIdentifier=self.gateway_id
            )

            targets = response.get('items', [])
            print(f"찾은 타겟 수: {len(targets)}")

            # 각 타겟 삭제
            for target in targets:
                target_id = target.get('targetId')
                target_name = target.get('name', 'Unknown')

                if target_id:
                    try:
                        print(f"타겟 삭제 중: {target_name} (ID: {target_id})")
                        self.gateway_client.delete_gateway_target(
                            gatewayIdentifier=self.gateway_id,
                            targetId=target_id
                        )
                        print(f"✅ 타겟 '{target_name}' 삭제 완료")
                        
                        # 타겟 삭제 후 잠시 대기
                        time.sleep(2)
                        
                    except ClientError as target_error:
                        print(f"❌ 타겟 '{target_name}' 삭제 실패: {target_error}")

            if targets:
                print("✅ 모든 Gateway 타겟 삭제 완료")
                # 모든 타겟 삭제 후 추가 대기
                time.sleep(5)
            else:
                print("ℹ️ 삭제할 타겟이 없습니다.")

        except ClientError as e:
            print(f"❌ Gateway 타겟 조회 실패: {e}")
        except Exception as e:
            print(f"❌ 예상치 못한 오류: {e}")

    def delete_gateway(self):
        """생성된 Gateway 삭제 (선택사항)"""
        if not self.gateway_id:
            print("⚠️ 삭제할 Gateway ID가 없습니다.")
            return

        print(f"=== Gateway 삭제: {self.gateway_id} ===")

        # 1. 먼저 연결된 타겟들 삭제
        self.delete_gateway_targets()

        # 2. Gateway 상태 확인 및 삭제 재시도
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"Gateway 삭제 시도 {attempt + 1}/{max_retries}")
                response = self.gateway_client.delete_gateway(
                    gatewayIdentifier=self.gateway_id
                )
                print(f"✅ Gateway '{self.gateway_name}' (ID: {self.gateway_id}) 삭제 완료!")
                return
                
            except ClientError as e:
                error_message = str(e)
                if "has targets associated" in error_message:
                    print(f"❌ 시도 {attempt + 1}: 타겟이 여전히 연결되어 있습니다. 10초 대기 후 재시도...")
                    time.sleep(10)
                elif "ResourceNotFoundException" in error_message:
                    print("✅ Gateway가 이미 삭제되었습니다.")
                    return
                else:
                    print(f"❌ Gateway 삭제 실패: {e}")
                    if attempt == max_retries - 1:
                        print("❌ 최대 재시도 횟수 초과. 수동으로 삭제하세요.")
                        
            except Exception as e:
                print(f"❌ 예상치 못한 오류로 Gateway 삭제 실패: {e}")
                if attempt == max_retries - 1:
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

            response = self.acps.delete_api_key_credential_provider(
                name=provider_name
            )
            print(f"✅ 자격 증명 공급자 '{provider_name}' 삭제 완료!")
        except ClientError as e:
            print(f"❌ 자격 증명 공급자 삭제 실패: {e}")
        except Exception as e:
            print(f"❌ 예상치 못한 오류로 자격 증명 공급자 삭제 실패: {e}")

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

        # 3. Gateway 생성
        manager.create_gateway(iam_role)

        # 4. API Key 자격 증명 공급자 생성
        manager.create_api_key_credential_provider()

        # 5. OpenAPI 스펙 S3 업로드
        openapi_s3_uri = manager.upload_openapi_spec_to_s3()

        # 6. Gateway 타겟 생성
        manager.create_gateway_target(openapi_s3_uri)

        # 7. 액세스 토큰 획득
        manager.get_access_token()

        # 8. YouTube 검색 에이전트 실행
        results = manager.run_youtube_search_agent("2025년 국내 신선식품 동향")

        # 9. 결과 저장
        manager.save_results(results)

        # 10. 리소스 정리 (선택사항)
        print("\n=== 리소스 정리 ===")
        cleanup_input = input("생성된 리소스를 정리하시겠습니까? (y/N): ").strip().lower()
        print(f"입력받은 값: '{cleanup_input}'")

        if cleanup_input in ['y', 'yes', 'Y', 'YES']:
            print("리소스 정리를 진행합니다...")
            manager.cleanup_all_resources()
        else:
            print("ℹ️ 리소스가 유지됩니다. 필요시 수동으로 정리하세요.")
            print(f"Gateway ID: {manager.gateway_id}")
            print(f"Gateway Name: {manager.gateway_name}")
            if manager.credential_provider_arn:
                provider_name = manager.credential_provider_arn.split('/')[-1]
                print(f"자격 증명 공급자: {provider_name}")

        print("\n🎉 모든 프로세스가 성공적으로 완료되었습니다!")

        return results

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        raise e


if __name__ == "__main__":
    main()