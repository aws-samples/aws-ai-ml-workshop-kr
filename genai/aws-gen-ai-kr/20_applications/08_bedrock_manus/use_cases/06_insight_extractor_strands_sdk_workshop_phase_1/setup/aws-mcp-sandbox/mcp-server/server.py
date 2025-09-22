#!/usr/bin/env python3
"""
AWS Code Sandbox MCP Server
Claude Code와 연동되는 MCP (Model Context Protocol) 서버
"""

import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, Optional
import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
    ListToolsResult,
)

# 환경 설정
AWS_REGION = os.getenv('AWS_REGION', 'us-west-2')
SESSION_MANAGER_URL = os.getenv('SESSION_MANAGER_URL', 'http://localhost:3000')
DEFAULT_TIMEOUT = int(os.getenv('DEFAULT_TIMEOUT', '300'))

class AWSCodeSandboxServer:
    def __init__(self):
        self.server = Server("aws-code-sandbox")
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(DEFAULT_TIMEOUT))
        self.sessions = {}  # session_id -> endpoint 매핑
        
        # 도구 등록
        self._register_tools()
    
    def _register_tools(self):
        """MCP 도구 등록"""
        
        @self.server.list_tools()
        async def list_tools() -> ListToolsResult:
            """사용 가능한 도구 목록 반환"""
            return ListToolsResult(tools=[
                Tool(
                    name="python_execute",
                    description="Execute Python code in AWS Fargate sandbox. Maintains session state including variables and installed packages.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "session_id": {
                                "type": "string",
                                "description": "Session identifier to maintain state across executions"
                            },
                            "code": {
                                "type": "string", 
                                "description": "Python code to execute"
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "Execution timeout in seconds (default: 300)",
                                "default": 300
                            }
                        },
                        "required": ["session_id", "code"]
                    }
                ),
                Tool(
                    name="bash_execute", 
                    description="Execute bash commands in AWS Fargate sandbox. Useful for system commands, file operations, and package installation.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "session_id": {
                                "type": "string",
                                "description": "Session identifier to maintain state across executions"
                            },
                            "command": {
                                "type": "string",
                                "description": "Bash command to execute"
                            },
                            "timeout": {
                                "type": "integer", 
                                "description": "Execution timeout in seconds (default: 300)",
                                "default": 300
                            }
                        },
                        "required": ["session_id", "command"]
                    }
                ),
                Tool(
                    name="session_status",
                    description="Get current session status including installed packages and environment state.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "session_id": {
                                "type": "string",
                                "description": "Session identifier"
                            }
                        },
                        "required": ["session_id"]
                    }
                ),
                Tool(
                    name="reset_session",
                    description="Reset session state, clearing all variables and installed packages.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "session_id": {
                                "type": "string",
                                "description": "Session identifier to reset"
                            }
                        },
                        "required": ["session_id"]
                    }
                )
            ])

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            """도구 호출 처리"""
            try:
                if name == "python_execute":
                    return await self._python_execute(arguments)
                elif name == "bash_execute":
                    return await self._bash_execute(arguments)
                elif name == "session_status":
                    return await self._session_status(arguments)
                elif name == "reset_session":
                    return await self._reset_session(arguments)
                else:
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text=f"Unknown tool: {name}"
                        )],
                        isError=True
                    )
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(
                        type="text", 
                        text=f"Error executing {name}: {str(e)}"
                    )],
                    isError=True
                )

    async def _get_session_endpoint(self, session_id: str) -> str:
        """세션별 Fargate 엔드포인트 획득 또는 생성"""
        
        # 기존 세션 엔드포인트가 있으면 재사용
        if session_id in self.sessions:
            endpoint = self.sessions[session_id]
            # 헬스체크로 세션 유효성 확인
            try:
                response = await self.client.get(f"{endpoint}/health")
                if response.status_code == 200:
                    return endpoint
            except:
                # 세션이 유효하지 않으면 제거
                del self.sessions[session_id]
        
        # 새로운 세션 생성 요청
        # 실제 구현에서는 Lambda 함수 또는 API Gateway 호출
        # 지금은 로컬 테스트를 위한 임시 구현
        
        # TODO: AWS API Gateway + Lambda 호출로 변경
        endpoint = await self._create_new_session(session_id)
        self.sessions[session_id] = endpoint
        
        return endpoint
    
    async def _create_new_session(self, session_id: str) -> str:
        """새로운 Fargate 세션 생성 (임시 구현)"""
        # 실제로는 AWS API를 호출하여 Fargate 태스크 생성
        # 지금은 로컬 테스트용 고정 엔드포인트 반환
        
        print(f"Creating new session: {session_id}")
        
        # 로컬 테스트용 - 실제로는 Fargate 태스크 생성 후 엔드포인트 반환
        return "http://localhost:8080"
    
    async def _python_execute(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Python 코드 실행"""
        session_id = arguments["session_id"]
        code = arguments["code"]
        timeout = arguments.get("timeout", DEFAULT_TIMEOUT)
        
        try:
            endpoint = await self._get_session_endpoint(session_id)
            
            response = await self.client.post(
                f"{endpoint}/execute",
                json={
                    "code": code,
                    "type": "python", 
                    "timeout": timeout
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                status = result.get("status", "unknown")
                output = result.get("output", "")
                execution_time = result.get("execution_time", 0)
                
                # 출력 포맷팅
                formatted_output = f"```\n{output}\n```"
                if execution_time > 0:
                    formatted_output += f"\n\n*Execution time: {execution_time:.2f}s*"
                
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=formatted_output
                    )],
                    isError=(status == "error")
                )
            else:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"HTTP Error {response.status_code}: {response.text}"
                    )],
                    isError=True
                )
                
        except Exception as e:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Execution error: {str(e)}"
                )],
                isError=True
            )
    
    async def _bash_execute(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Bash 명령어 실행"""
        session_id = arguments["session_id"]
        command = arguments["command"]
        timeout = arguments.get("timeout", DEFAULT_TIMEOUT)
        
        try:
            endpoint = await self._get_session_endpoint(session_id)
            
            response = await self.client.post(
                f"{endpoint}/execute",
                json={
                    "code": command,
                    "type": "bash",
                    "timeout": timeout
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                status = result.get("status", "unknown")
                output = result.get("output", "")
                return_code = result.get("return_code", 0)
                execution_time = result.get("execution_time", 0)
                
                # 출력 포맷팅
                formatted_output = f"```bash\n{output}\n```"
                formatted_output += f"\n\n*Return code: {return_code}, Execution time: {execution_time:.2f}s*"
                
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=formatted_output
                    )],
                    isError=(status == "error" or return_code != 0)
                )
            else:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"HTTP Error {response.status_code}: {response.text}"
                    )],
                    isError=True
                )
                
        except Exception as e:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Execution error: {str(e)}"
                )],
                isError=True
            )
    
    async def _session_status(self, arguments: Dict[str, Any]) -> CallToolResult:
        """세션 상태 조회"""
        session_id = arguments["session_id"]
        
        try:
            endpoint = await self._get_session_endpoint(session_id)
            
            response = await self.client.get(f"{endpoint}/status")
            
            if response.status_code == 200:
                result = response.json()
                
                status_text = f"""Session Status for: {session_id}

**Python Environment:**
- Global variables: {len(result.get('python_globals_keys', []))} defined
- Local variables: {len(result.get('python_locals_keys', []))} defined  
- Installed packages: {', '.join(result.get('installed_packages', []))}

**System:**
- Working directory: {result.get('working_directory', 'Unknown')}
- Endpoint: {endpoint}
"""
                
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=status_text
                    )],
                    isError=False
                )
            else:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Failed to get session status: HTTP {response.status_code}"
                    )],
                    isError=True
                )
                
        except Exception as e:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error getting session status: {str(e)}"
                )],
                isError=True
            )
    
    async def _reset_session(self, arguments: Dict[str, Any]) -> CallToolResult:
        """세션 상태 초기화"""
        session_id = arguments["session_id"]
        
        try:
            endpoint = await self._get_session_endpoint(session_id)
            
            response = await self.client.post(f"{endpoint}/reset")
            
            if response.status_code == 200:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Session {session_id} has been reset successfully."
                    )],
                    isError=False
                )
            else:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Failed to reset session: HTTP {response.status_code}"
                    )],
                    isError=True
                )
                
        except Exception as e:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error resetting session: {str(e)}"
                )],
                isError=True
            )

async def main():
    """MCP 서버 실행"""
    print("Starting AWS Code Sandbox MCP Server...")
    print(f"AWS Region: {AWS_REGION}")
    print(f"Session Manager URL: {SESSION_MANAGER_URL}")
    print(f"Default Timeout: {DEFAULT_TIMEOUT}s")
    
    sandbox_server = AWSCodeSandboxServer()
    
    async with stdio_server() as (read_stream, write_stream):
        await sandbox_server.server.run(
            read_stream,
            write_stream,
            sandbox_server.server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())