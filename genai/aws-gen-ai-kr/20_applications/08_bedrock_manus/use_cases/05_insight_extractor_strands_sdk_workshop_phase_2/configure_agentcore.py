#!/usr/bin/env python
"""
AgentCore Runtime 설정 스크립트 (터미널용)
노트북에서 interactive 프롬프트 문제를 피하기 위해 터미널에서 실행
"""
import sys
from bedrock_agentcore_starter_toolkit import Runtime
from boto3.session import Session

def main():
    # 명령줄 인자로 agent_name과 execution_role을 받음
    if len(sys.argv) < 3:
        print("Usage: python configure_agentcore.py <agent_name> <execution_role_arn>")
        sys.exit(1)

    agent_name = sys.argv[1]
    execution_role = sys.argv[2]

    # Region 설정
    boto_session = Session()
    region = boto_session.region_name

    print(f"Configuring AgentCore Runtime...")
    print(f"  Agent Name: {agent_name}")
    print(f"  Region: {region}")
    print(f"  Execution Role: {execution_role}")

    # Runtime 설정
    agentcore_runtime = Runtime()

    response = agentcore_runtime.configure(
        agent_name=agent_name,
        entrypoint="agentcore_runtime.py",
        execution_role=execution_role,
        auto_create_ecr=True,
        requirements_file="requirements.txt",
        region=region
    )

    print("\n✅ Configuration completed successfully!")
    print(f"Agent ID: {response.agent_id}")
    print(f"Config Path: {response.config_path}")

    return response

if __name__ == "__main__":
    main()
