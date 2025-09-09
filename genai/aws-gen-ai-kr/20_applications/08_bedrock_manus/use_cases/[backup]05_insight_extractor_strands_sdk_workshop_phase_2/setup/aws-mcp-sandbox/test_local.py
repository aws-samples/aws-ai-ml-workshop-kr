#!/usr/bin/env python3
"""
AWS MCP Sandbox 로컬 테스트 스크립트
Fargate 런타임을 로컬에서 테스트
"""

import json
import time
import requests
import subprocess
import signal
import sys
from threading import Thread

class LocalTester:
    def __init__(self):
        self.container_name = "mcp-sandbox-test"
        self.base_url = "http://localhost:8080"
        self.container_process = None
        
    def start_local_container(self):
        """로컬 Docker 컨테이너 시작"""
        print("🐳 로컬 Docker 컨테이너 시작 중...")
        
        # 기존 컨테이너 정리
        try:
            subprocess.run(["docker", "stop", self.container_name], 
                          capture_output=True, check=False)
            subprocess.run(["docker", "rm", self.container_name], 
                          capture_output=True, check=False)
        except:
            pass
        
        # 새 컨테이너 시작
        try:
            subprocess.run([
                "docker", "run", "-d",
                "--name", self.container_name,
                "-p", "8080:8080",
                "mcp-sandbox:latest"
            ], check=True)
            
            print("✅ Docker 컨테이너 시작됨")
            time.sleep(5)  # 컨테이너 초기화 대기
            return True
            
        except subprocess.CalledProcessError:
            print("❌ Docker 컨테이너 시작 실패")
            print("💡 먼저 Docker 이미지를 빌드하세요: cd docker && ./build-and-push.sh")
            return False
    
    def wait_for_health(self, timeout=30):
        """헬스체크 대기"""
        print("🔍 헬스체크 대기 중...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                if response.status_code == 200:
                    print("✅ 헬스체크 성공")
                    return True
            except requests.RequestException:
                pass
            
            time.sleep(2)
        
        print("❌ 헬스체크 타임아웃")
        return False
    
    def test_python_execution(self):
        """Python 코드 실행 테스트"""
        print("\n🐍 Python 실행 테스트")
        
        test_cases = [
            {
                "name": "기본 출력",
                "code": "print('Hello, MCP Sandbox!')",
                "expected": "Hello, MCP Sandbox!"
            },
            {
                "name": "수학 계산",
                "code": "result = 2 + 3 * 4\nprint(f'Result: {result}')",
                "expected": "Result: 14"
            },
            {
                "name": "변수 유지 테스트",
                "code": "x = 42\nprint(f'x = {x}')",
                "expected": "x = 42"
            },
            {
                "name": "이전 변수 접근",
                "code": "print(f'Previous x: {x}')",
                "expected": "Previous x: 42"
            },
            {
                "name": "패키지 테스트",
                "code": "import datetime\nprint(f'Current time: {datetime.datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')",
                "expected": "Current time:"
            }
        ]
        
        for test in test_cases:
            print(f"  📝 {test['name']}")
            
            try:
                response = requests.post(
                    f"{self.base_url}/execute",
                    json={"code": test["code"], "type": "python"},
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    output = result.get("output", "").strip()
                    
                    if test["expected"] in output:
                        print(f"     ✅ 성공: {output}")
                    else:
                        print(f"     ❌ 실패: {output}")
                else:
                    print(f"     ❌ HTTP 오류: {response.status_code}")
                    
            except requests.RequestException as e:
                print(f"     ❌ 요청 오류: {str(e)}")
    
    def test_bash_execution(self):
        """Bash 명령 실행 테스트"""
        print("\n🔧 Bash 실행 테스트")
        
        test_cases = [
            {
                "name": "기본 명령어",
                "command": "echo 'Hello from bash!'",
                "expected": "Hello from bash!"
            },
            {
                "name": "시스템 정보",
                "command": "uname -a",
                "expected": "Linux"
            },
            {
                "name": "Python 버전",
                "command": "python --version",
                "expected": "Python 3"
            },
            {
                "name": "디렉토리 목록",
                "command": "ls -la",
                "expected": "total"
            }
        ]
        
        for test in test_cases:
            print(f"  📝 {test['name']}")
            
            try:
                response = requests.post(
                    f"{self.base_url}/execute",
                    json={"code": test["command"], "type": "bash"},
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    output = result.get("output", "").strip()
                    
                    if test["expected"] in output:
                        print(f"     ✅ 성공")
                    else:
                        print(f"     ❌ 실패: {output}")
                else:
                    print(f"     ❌ HTTP 오류: {response.status_code}")
                    
            except requests.RequestException as e:
                print(f"     ❌ 요청 오류: {str(e)}")
    
    def test_package_installation(self):
        """패키지 설치 테스트"""
        print("\n📦 패키지 설치 테스트")
        
        # 패키지 설치
        print("  📥 requests 패키지 설치 중...")
        try:
            response = requests.post(
                f"{self.base_url}/execute",
                json={"code": "pip install requests", "type": "python"},
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if "Successfully installed" in result.get("output", ""):
                    print("     ✅ 설치 성공")
                else:
                    print(f"     ❌ 설치 실패: {result.get('output', '')}")
            else:
                print(f"     ❌ HTTP 오류: {response.status_code}")
                
        except requests.RequestException as e:
            print(f"     ❌ 요청 오류: {str(e)}")
        
        # 설치된 패키지 사용
        print("  🔍 설치된 패키지 사용 테스트...")
        try:
            response = requests.post(
                f"{self.base_url}/execute",
                json={"code": "import requests\nprint(f'Requests version: {requests.__version__}')", "type": "python"},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                output = result.get("output", "").strip()
                if "Requests version:" in output:
                    print(f"     ✅ 사용 성공: {output}")
                else:
                    print(f"     ❌ 사용 실패: {output}")
            else:
                print(f"     ❌ HTTP 오류: {response.status_code}")
                
        except requests.RequestException as e:
            print(f"     ❌ 요청 오류: {str(e)}")
    
    def test_status_endpoint(self):
        """상태 조회 테스트"""
        print("\n📊 상태 조회 테스트")
        
        try:
            response = requests.get(f"{self.base_url}/status", timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                print("     ✅ 상태 조회 성공:")
                print(f"        - Global variables: {len(result.get('python_globals_keys', []))}")
                print(f"        - Local variables: {len(result.get('python_locals_keys', []))}")
                print(f"        - Installed packages: {len(result.get('installed_packages', []))}")
                print(f"        - Working directory: {result.get('working_directory', 'Unknown')}")
            else:
                print(f"     ❌ HTTP 오류: {response.status_code}")
                
        except requests.RequestException as e:
            print(f"     ❌ 요청 오류: {str(e)}")
    
    def test_session_reset(self):
        """세션 리셋 테스트"""
        print("\n🔄 세션 리셋 테스트")
        
        try:
            response = requests.post(f"{self.base_url}/reset", timeout=10)
            
            if response.status_code == 200:
                print("     ✅ 세션 리셋 성공")
                
                # 리셋 후 변수 확인
                response = requests.post(
                    f"{self.base_url}/execute",
                    json={"code": "print(f'x exists: {\"x\" in globals()}')", "type": "python"},
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "x exists: False" in result.get("output", ""):
                        print("     ✅ 변수가 정상적으로 리셋됨")
                    else:
                        print(f"     ❌ 변수 리셋 확인 실패: {result.get('output', '')}")
            else:
                print(f"     ❌ HTTP 오류: {response.status_code}")
                
        except requests.RequestException as e:
            print(f"     ❌ 요청 오류: {str(e)}")
    
    def cleanup(self):
        """테스트 후 정리"""
        print("\n🧹 테스트 환경 정리 중...")
        
        try:
            subprocess.run(["docker", "stop", self.container_name], 
                          capture_output=True, check=False)
            subprocess.run(["docker", "rm", self.container_name], 
                          capture_output=True, check=False)
            print("✅ Docker 컨테이너 정리 완료")
        except:
            pass
    
    def run_all_tests(self):
        """모든 테스트 실행"""
        print("🚀 AWS MCP Sandbox 로컬 테스트 시작")
        print("=" * 50)
        
        try:
            # Docker 컨테이너 시작
            if not self.start_local_container():
                return False
            
            # 헬스체크 대기
            if not self.wait_for_health():
                return False
            
            # 테스트 실행
            self.test_python_execution()
            self.test_bash_execution()
            self.test_package_installation()
            self.test_status_endpoint()
            self.test_session_reset()
            
            print("\n" + "=" * 50)
            print("🎉 모든 테스트 완료!")
            return True
            
        except KeyboardInterrupt:
            print("\n\n⚠️ 테스트가 중단되었습니다.")
            return False
        
        finally:
            self.cleanup()

def signal_handler(sig, frame):
    """시그널 핸들러"""
    print("\n\n⚠️ 테스트 중단 중...")
    sys.exit(1)

def main():
    """메인 함수"""
    signal.signal(signal.SIGINT, signal_handler)
    
    tester = LocalTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ 로컬 테스트가 성공적으로 완료되었습니다!")
        print("💡 다음 단계: AWS 배포 및 MCP 서버 테스트")
    else:
        print("\n❌ 테스트 중 오류가 발생했습니다.")
        sys.exit(1)

if __name__ == "__main__":
    main()