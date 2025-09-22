import subprocess  # nosec B404
import sys
import time
import signal
import os
import shlex

# List of allowed MCP servers only
ALLOWED_MCP_SERVERS = [
    "application/mcp_server_tavily.py",
    "application/mcp_server_arxiv.py",
    "application/mcp_server_pubmed.py",
    "application/mcp_server_chembl.py",
    "application/mcp_server_clinicaltrial.py",
]

processes = []

def validate_server_path(server_path):
    """Server path validation - prevents CWE-78"""
    # 1. Check if in allowed server list
    if server_path not in ALLOWED_MCP_SERVERS:
        raise ValueError(f"Unauthorized server: {server_path}")
    
    # 2. Check if file actually exists
    normalized_path = os.path.realpath(server_path)
    if not os.path.exists(normalized_path):
        raise ValueError(f"Server file does not exist: {normalized_path}")
    
    # 3. Check if it's a Python file
    if not normalized_path.endswith('.py'):
        raise ValueError(f"Not a Python file: {normalized_path}")
    
    # 4. Check if it's under current directory (prevent directory traversal)
    current_dir = os.path.realpath(os.getcwd())
    if not normalized_path.startswith(current_dir):
        raise ValueError(f"Unauthorized path: {normalized_path}")
    
    return normalized_path

def signal_handler(sig, frame):
    print("\nCtrl+C detected. Terminating all servers...")
    for process in processes:
        if process.poll() is None:  # If process is still running
            process.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def main():
    print("Starting all MCP servers...")
    
    for server in ALLOWED_MCP_SERVERS:
        try:
            # Path validation (prevents CWE-78)
            normalized_path = validate_server_path(server)
            print(f"Starting {server}...")
            
            # Use validated path with shlex.quote() - recommended by scanner
            process = subprocess.Popen(  # nosec B603
                [sys.executable, shlex.quote(server)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                shell=False,
            )
            processes.append(process)
            
        except ValueError as e:
            print(f"Server validation failed: {e}")
            # Clean up already started processes
            for p in processes:
                if p.poll() is None:
                    p.terminate()
            sys.exit(1)
        
        # Check initial log output
        if process.poll() is not None:
            # Process already terminated
            stdout, stderr = process.communicate()
            print(f"Error: Failed to start {server}")
            print(f"STDERR: {stderr}")
            print(f"STDOUT: {stdout}")
            # Terminate all other processes
            for p in processes:
                if p != process and p.poll() is None:
                    p.terminate()
            sys.exit(1)
    
    print("\nAll MCP servers started successfully.")
    print("Server logs:")
    
    # Monitor all server logs in real-time
    try:
        while True:
            for i, process in enumerate(processes):
                if process.poll() is not None:
                    # Process terminated unexpectedly
                    stdout, stderr = process.communicate()
                    print(f"\nError: {ALLOWED_MCP_SERVERS[i]} server terminated unexpectedly.")
                    print(f"STDERR: {stderr}")
                    # Terminate all other processes
                    for p in processes:
                        if p != process and p.poll() is None:
                            p.terminate()
                    sys.exit(1)
                
                # Read stdout and stderr
                for line in iter(process.stdout.readline, ""):
                    if not line:
                        break
                    print(f"[{ALLOWED_MCP_SERVERS[i]}] {line.strip()}")
                
                for line in iter(process.stderr.readline, ""):
                    if not line:
                        break
                    print(f"[{ALLOWED_MCP_SERVERS[i]} ERROR] {line.strip()}")
                    
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    main()
