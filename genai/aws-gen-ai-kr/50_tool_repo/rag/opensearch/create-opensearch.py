#!/usr/bin/env python3
"""
필수 패키지 설정, OpenSearch 클러스터 생성, Nori 플러그인 설치 (약 50분 소요)
이 스크립트는 SageMaker Studio Data Science 3.0 kernel 및 ml.t3.medium 인스턴스에서 테스트되었습니다.

필수 사항:
- 실습을 위해서 스크립트를 실행하는 역할(Role)에 아래 권한이 추가되어 있어야 합니다.
    - AmazonOpenSearchServiceFullAccess
    - AmazonSSMFullAccess
"""

import os
import sys
import boto3
import uuid
import botocore
import time
import argparse
import re
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded configuration from {env_path}")

# 모듈 경로 추가
module_path = ".."
sys.path.append(os.path.abspath(module_path))

def parse_arguments():
    """명령행 인수 파싱 (.env 파일에서 기본값 로드)"""
    # .env에서 기본값 가져오기 (없으면 하드코딩된 기본값 사용)
    default_user_id = os.getenv('OPENSEARCH_USER_ID', 'raguser')
    default_password = os.getenv('OPENSEARCH_USER_PASSWORD', 'MarsEarth1!')
    default_version = os.getenv('OPENSEARCH_VERSION', '3.1')
    default_domain_name = os.getenv('OPENSEARCH_DOMAIN_NAME', '')

    parser = argparse.ArgumentParser(description='OpenSearch 클러스터 설정 스크립트')
    parser.add_argument('--version', '-v', default=default_version,
                       help=f'OpenSearch 버전 (예: 1.3, 2.3, 2.5, 2.7, 2.9, 2.11, 2.13, 2.15, 2.17, 2.19, 3.1). 기본값: {default_version}')
    parser.add_argument('--user-id', '-u', default=default_user_id,
                       help=f'OpenSearch 사용자 ID. 기본값: {default_user_id}')
    parser.add_argument('--password', '-p', default=default_password,
                       help=f'OpenSearch 사용자 비밀번호. 기본값: {"*" * len(default_password)}')
    parser.add_argument('--domain-name', '-d', default=default_domain_name,
                       help=f'OpenSearch 도메인 이름. 지정하지 않으면 자동 생성됩니다. (예: my-opensearch-cluster). 기본값: {default_domain_name if default_domain_name else "자동 생성"}')
    parser.add_argument('--dev', action='store_true', default=True,
                       help='개발 모드 (1-AZ without standby). 기본값: True')
    parser.add_argument('--prod', action='store_true',
                       help='프로덕션 모드 (3-AZ with standby)')

    args = parser.parse_args()

    # --prod 플래그가 설정되면 DEV를 False로 변경
    if args.prod:
        args.dev = False

    return args

def validate_domain_name(domain_name):
    """도메인 이름 유효성 검사"""
    if not domain_name:
        return True  # 빈 이름은 자동 생성하므로 허용
    
    # OpenSearch 도메인 이름 규칙:
    # - 3-28자 길이
    # - 소문자로 시작
    # - 소문자, 숫자, 하이픈만 포함
    # - 하이픈으로 끝나면 안됨
    if len(domain_name) < 3 or len(domain_name) > 28:
        raise ValueError("도메인 이름은 3-28자여야 합니다.")
    
    if not re.match(r'^[a-z][a-z0-9\-]*[a-z0-9]$', domain_name):
        raise ValueError("도메인 이름은 소문자로 시작하고, 소문자, 숫자, 하이픈만 포함할 수 있으며, 하이픈으로 끝날 수 없습니다.")
    
    return True

def create_opensearch_domain(version, user_id, password, domain_name, dev_mode):
    """OpenSearch 도메인 생성"""
    region = boto3.Session().region_name
    account_id = boto3.client("sts").get_caller_identity()["Account"]
    opensearch = boto3.client('opensearch', region)
    
    # 도메인 이름 설정
    if domain_name:
        validate_domain_name(domain_name)
        final_domain_name = domain_name
    else:
        rand_str = uuid.uuid4().hex[:8]
        final_domain_name = f'rag-hol-{rand_str}'

    # 클러스터 설정
    cluster_config_prod = {
        'InstanceCount': 3,
        'InstanceType': 'r6g.large.search',
        'ZoneAwarenessEnabled': True,
        'DedicatedMasterEnabled': True,
        'MultiAZWithStandbyEnabled': True,
        'DedicatedMasterType': 'r6g.large.search',
        'DedicatedMasterCount': 3
    }

    cluster_config_dev = {
        'InstanceCount': 1,
        'InstanceType': 'r6g.large.search',
        'ZoneAwarenessEnabled': False,
        'DedicatedMasterEnabled': False,
    }

    ebs_options = {
        'EBSEnabled': True,
        'VolumeType': 'gp3',
        'VolumeSize': 100,
    }

    advanced_security_options = {
        'Enabled': True,
        'InternalUserDatabaseEnabled': True,
        'MasterUserOptions': {
            'MasterUserName': user_id,
            'MasterUserPassword': password
        }
    }

    ap = f'{{"Version":"2012-10-17","Statement":[{{"Effect":"Allow","Principal":{{"AWS":"*"}},"Action":"es:*","Resource":"arn:aws:es:{region}:{account_id}:domain/{final_domain_name}/*"}}]}}'

    cluster_config = cluster_config_dev if dev_mode else cluster_config_prod

    print(f"OpenSearch 도메인 '{final_domain_name}' 생성 중...")
    response = opensearch.create_domain(
        DomainName=final_domain_name,
        EngineVersion=f'OpenSearch_{version}',
        ClusterConfig=cluster_config,
        AccessPolicies=ap,
        EBSOptions=ebs_options,
        AdvancedSecurityOptions=advanced_security_options,
        NodeToNodeEncryptionOptions={'Enabled': True},
        EncryptionAtRestOptions={'Enabled': True},
        DomainEndpointOptions={'EnforceHTTPS': True}
    )

    return final_domain_name, opensearch, region

def wait_for_domain_creation(opensearch, domain_name):
    """도메인 생성 완료 대기"""
    try:
        response = opensearch.describe_domain(DomainName=domain_name)
        
        # 엔드포인트가 생성될 때까지 60초마다 확인
        while 'Endpoint' not in response['DomainStatus']:
            print('OpenSearch 도메인 생성 중...')
            time.sleep(60)
            response = opensearch.describe_domain(DomainName=domain_name)

        # 루프를 벗어나면 도메인이 데이터 수집 준비 완료
        endpoint = response['DomainStatus']['Endpoint']
        print('도메인 엔드포인트 준비 완료: ' + endpoint)
        return endpoint
        
    except botocore.exceptions.ClientError as error:
        if error.response['Error']['Code'] == 'ResourceNotFoundException':
            print('도메인을 찾을 수 없습니다.')
        else:
            raise error

def store_credentials_in_ssm(opensearch_domain_endpoint, user_id, password, region):
    """OpenSearch 인증정보를 SSM에 저장 (선택적)"""
    try:
        from src.utils.ssm import parameter_store
        print("SSM에 인증정보 저장 중...")
        pm = parameter_store(region)

        pm.put_params(
            key="opensearch_domain_endpoint",
            value=f'{opensearch_domain_endpoint}',
            overwrite=True,
            enc=False
        )

        pm.put_params(
            key="opensearch_user_id",
            value=f'{user_id}',
            overwrite=True,
            enc=False
        )

        pm.put_params(
            key="opensearch_user_password",
            value=f'{password}',
            overwrite=True,
            enc=True
        )

        return pm
    except ImportError:
        print("SSM 모듈을 찾을 수 없습니다. .env 파일을 사용하여 설정을 관리하세요.")
        return None

def install_nori_plugin(opensearch, domain_name, region, version):
    """한국어 분석을 위한 노리(Nori) 플러그인 설치"""
    print("Nori 플러그인 설치 중...")
    
    # 각 리전별 Nori 플러그인 패키지 ID (최신 정보 반영)
    nori_pkg_id = {
        'us-east-1': {
            '1.3': 'G39874436',
            '2.3': 'G196105221',
            '2.5': 'G240285063',
            '2.7': 'G16029449',
            '2.9': 'G60209291',
            '2.11': 'G181660338',
            '2.13': 'G225840180',
            '2.15': 'G1584566',
            '2.17': 'G45764408',
            '2.19': 'G89944250',
            '3.1': 'G39720708'
        },
        'us-west-2': {
            '1.3': 'G206252145',
            '2.3': 'G94047474',
            '2.5': 'G138227316',
            '2.7': 'G182407158',
            '2.9': 'G226587000',
            '2.11': 'G79602591',
            '2.13': 'G123782433',
            '2.15': 'G167962275',
            '2.17': 'G212142117',
            '2.19': 'G256321959',
            '3.1': 'G206098417'
        },
        'ap-northeast-2': {
            '1.3': 'G81033971',
            '2.3': 'G32784146',
            '2.5': 'G39108304',
            '2.7': 'G45432462',
            '2.9': 'G51756620',
            '2.11': 'G248827013',
            '2.13': 'G255151171',
            '2.15': 'G261475329',
            '2.17': 'G267799487',
            '2.19': 'G5688189',
            '3.1': 'G246645619'
        }
    }

    if region not in nori_pkg_id:
        available_regions = list(nori_pkg_id.keys())
        raise ValueError(f"지원되지 않는 리전입니다: {region}. 지원되는 리전: {available_regions}")
    
    if version not in nori_pkg_id[region]:
        available_versions = list(nori_pkg_id[region].keys())
        raise ValueError(f"리전 {region}에서 지원되지 않는 OpenSearch 버전입니다: {version}. 지원되는 버전: {available_versions}")

    print(f"리전 {region}에서 OpenSearch {version} 버전용 Nori 플러그인 (패키지 ID: {nori_pkg_id[region][version]})을 설치합니다...")
    
    pkg_response = opensearch.associate_package(
        PackageID=nori_pkg_id[region][version],  # nori plugin
        DomainName=domain_name
    )

def wait_for_associate_package(opensearch, domain_name, max_results=1):
    """패키지 연결 완료 대기"""
    response = opensearch.list_packages_for_domain(
        DomainName=domain_name,
        MaxResults=1
    )

    # 패키지 연결이 완료될 때까지 60초마다 확인
    while response['DomainPackageDetailsList'][0]['DomainPackageStatus'] == "ASSOCIATING":
        print('패키지 연결 중...')
        time.sleep(60)
        response = opensearch.list_packages_for_domain(
            DomainName=domain_name,
            MaxResults=1
        )

    print('Nori 플러그인 연결 완료!')

def update_env_file(domain_name, opensearch_domain_endpoint):
    """OpenSearch 설정 정보를 .env 파일에 업데이트"""
    env_path = Path(__file__).parent.parent / '.env'

    if not env_path.exists():
        print(f"경고: .env 파일을 찾을 수 없습니다: {env_path}")
        return False

    # .env 파일 읽기
    with open(env_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 업데이트할 항목
    updated_domain_name = False
    updated_endpoint = False

    # 기존 라인 업데이트
    new_lines = []
    for line in lines:
        if line.strip().startswith('OPENSEARCH_DOMAIN_NAME='):
            new_lines.append(f'OPENSEARCH_DOMAIN_NAME={domain_name}\n')
            updated_domain_name = True
        elif line.strip().startswith('OPENSEARCH_DOMAIN_ENDPOINT='):
            new_lines.append(f'OPENSEARCH_DOMAIN_ENDPOINT={opensearch_domain_endpoint}\n')
            updated_endpoint = True
        else:
            new_lines.append(line)

    # 없으면 추가 (OpenSearch Configuration 섹션에)
    if not updated_domain_name or not updated_endpoint:
        final_lines = []
        opensearch_section_found = False

        for i, line in enumerate(new_lines):
            final_lines.append(line)

            # OpenSearch Configuration 섹션 찾기
            if '# OpenSearch Configuration' in line:
                opensearch_section_found = True

            # OpenSearch 섹션 내에서 적절한 위치에 추가
            if opensearch_section_found and line.strip().startswith('OPENSEARCH_VERSION='):
                if not updated_domain_name:
                    final_lines.append(f'OPENSEARCH_DOMAIN_NAME={domain_name}\n')
                    updated_domain_name = True
                if not updated_endpoint:
                    # DOMAIN_ENDPOINT는 다음 줄에 추가
                    if i + 1 < len(new_lines) and not new_lines[i + 1].strip().startswith('OPENSEARCH_DOMAIN_NAME='):
                        pass  # 다음 반복에서 처리

        new_lines = final_lines

    # .env 파일 쓰기
    with open(env_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print(f"\n✓ .env 파일 업데이트 완료: {env_path}")
    print(f"  - OPENSEARCH_DOMAIN_NAME={domain_name}")
    print(f"  - OPENSEARCH_DOMAIN_ENDPOINT={opensearch_domain_endpoint}")

    return True

def main():
    """메인 실행 함수"""
    # 명령행 인수 파싱
    args = parse_arguments()
    
    start_time = time.time()
    
    print("=== OpenSearch 클러스터 설정 시작 ===")
    print(f"개발 모드: {args.dev}")
    print(f"OpenSearch 버전: {args.version}")
    print(f"사용자 ID: {args.user_id}")
    print(f"비밀번호: {'*' * len(args.password)}")
    print(f"도메인 이름: {args.domain_name if args.domain_name else '자동 생성'}")
    
    try:
        # 1. OpenSearch 도메인 생성
        domain_name, opensearch, region = create_opensearch_domain(
            args.version, args.user_id, args.password, args.domain_name, args.dev
        )
        
        # 2. 도메인 생성 완료 대기
        endpoint = wait_for_domain_creation(opensearch, domain_name)
        opensearch_domain_endpoint = f"https://{endpoint}"
        
        # 3. SSM에 인증정보 저장
        pm = store_credentials_in_ssm(
            opensearch_domain_endpoint, args.user_id, args.password, region
        )
        
        # 4. Nori 플러그인 설치
        install_nori_plugin(opensearch, domain_name, region, args.version)
        
        # 5. 플러그인 설치 완료 대기
        wait_for_associate_package(opensearch, domain_name)

        # 6. .env 파일 업데이트
        update_env_file(domain_name, opensearch_domain_endpoint)

        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60  # 분 단위로 변환

        print(f"\n=== 설정 완료 ===")
        print(f"총 소요 시간: {elapsed_time:.1f}분")
        print(f"도메인 이름: {domain_name}")
        print(f"엔드포인트: {opensearch_domain_endpoint}")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main()