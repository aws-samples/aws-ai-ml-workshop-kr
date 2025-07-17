#!/usr/bin/env python3
"""
Setup script for AI Menu Board Generator
AWS GenAI Workshop - Application Example
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ 모든 의존성이 성공적으로 설치되었습니다.")
    except subprocess.CalledProcessError as e:
        print(f"❌ 의존성 설치 중 오류가 발생했습니다: {e}")
        sys.exit(1)

def check_aws_credentials():
    """Check if AWS credentials are configured"""
    try:
        import boto3
        session = boto3.Session()
        credentials = session.get_credentials()
        if credentials is None:
            print("⚠️  AWS 자격 증명이 설정되지 않았습니다.")
            print("다음 중 하나의 방법으로 설정하세요:")
            print("1. aws configure")
            print("2. 환경 변수 설정")
            print("3. IAM 역할 사용")
            return False
        else:
            print("✅ AWS 자격 증명이 확인되었습니다.")
            return True
    except ImportError:
        print("❌ boto3가 설치되지 않았습니다.")
        return False

def main():
    print("🚀 AI Menu Board Generator 설정을 시작합니다...")
    
    # Install requirements
    print("\n📦 의존성 설치 중...")
    install_requirements()
    
    # Check AWS credentials
    print("\n🔐 AWS 자격 증명 확인 중...")
    check_aws_credentials()
    
    print("\n✨ 설정이 완료되었습니다!")
    print("다음 명령어로 애플리케이션을 실행하세요:")
    print("streamlit run app.py")

if __name__ == "__main__":
    main()
