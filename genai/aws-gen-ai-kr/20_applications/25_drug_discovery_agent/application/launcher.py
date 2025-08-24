import subprocess
import sys
import time
import signal
import os
import shlex

# 실행할 MCP 서버 목록
mcp_servers = [
    "application/mcp_server_tavily.py",
    "application/mcp_server_arxiv.py",
    "application/mcp_server_pubmed.py",
    "application/mcp_server_chembl.py",
    "application/mcp_server_clinicaltrial.py",
]

processes = []

def signal_handler(sig, frame):
    print("\nCtrl+C 감지됨. 모든 서버를 종료합니다...")
    for process in processes:
        if process.poll() is None:  # 프로세스가 아직 실행 중인 경우
            process.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def main():
    print("모든 MCP 서버를 시작합니다...")
    
    for server in mcp_servers:
        print(f"{server} 시작 중...")

        normalized_path = os.path.realpath(server)
        
        # 서버 시작
        process = subprocess.Popen(
            [sys.executable,shlex.quote(normalized_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            shell=False,
        )
        processes.append(process)
        
        # 초기 로그 출력 확인
        if process.poll() is not None:
            # 프로세스가 이미 종료된 경우
            stdout, stderr = process.communicate()
            print(f"오류: {server} 시작 실패")
            print(f"STDERR: {stderr}")
            print(f"STDOUT: {stdout}")
            # 다른 모든 프로세스 종료
            for p in processes:
                if p != process and p.poll() is None:
                    p.terminate()
            sys.exit(1)
    
    print("\n모든 MCP 서버가 성공적으로 시작되었습니다.")
    print("서버 로그:")
    
    # 모든 서버의 로그를 실시간으로 모니터링
    try:
        while True:
            for i, process in enumerate(processes):
                if process.poll() is not None:
                    # 프로세스가 예기치 않게 종료된 경우
                    stdout, stderr = process.communicate()
                    print(f"\n오류: {mcp_servers[i]} 서버가 예기치 않게 종료되었습니다.")
                    print(f"STDERR: {stderr}")
                    # 다른 모든 프로세스 종료
                    for p in processes:
                        if p != process and p.poll() is None:
                            p.terminate()
                    sys.exit(1)
                
                # 표준 출력 및 오류 읽기
                for line in iter(process.stdout.readline, ""):
                    if not line:
                        break
                    print(f"[{mcp_servers[i]}] {line.strip()}")
                
                for line in iter(process.stderr.readline, ""):
                    if not line:
                        break
                    print(f"[{mcp_servers[i]} ERROR] {line.strip()}")
                    
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    main()