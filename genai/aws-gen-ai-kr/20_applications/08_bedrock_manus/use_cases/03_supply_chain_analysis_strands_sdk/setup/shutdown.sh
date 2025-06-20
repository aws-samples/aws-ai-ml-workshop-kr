#!/bin/bash

echo "=== MCP OpenSearch 서버 프로세스 확인 ==="

# 1. mcp_server_opensearch 프로세스 찾기 (grep 자체 제외)
echo "1. mcp_server_opensearch 프로세스:"
ps aux | grep mcp_server_opensearch | grep -v grep
if [ $? -ne 0 ]; then
    echo "   → 실행 중인 mcp_server_opensearch 프로세스가 없습니다."
fi

echo ""

# 2. 모든 Python MCP 관련 프로세스 찾기
echo "2. Python MCP 관련 프로세스:"
ps aux | grep python | grep mcp | grep -v grep
if [ $? -ne 0 ]; then
    echo "   → 실행 중인 Python MCP 프로세스가 없습니다."
fi

echo ""

# 3. 포트 사용 확인 (MCP는 보통 stdio를 사용하므로 특정 포트는 없음)
echo "3. Python 프로세스들:"
ps aux | grep python | grep -v grep | head -5
if [ $? -ne 0 ]; then
    echo "   → 실행 중인 Python 프로세스가 없습니다."
fi

echo ""

# 4. pgrep 사용해서 더 정확하게 찾기
echo "4. pgrep으로 MCP 프로세스 검색:"
pgrep -f "mcp_server_opensearch"
if [ $? -ne 0 ]; then
    echo "   → pgrep에서도 MCP 프로세스를 찾을 수 없습니다."
fi

echo ""
echo "=== 검사 완료 ==="