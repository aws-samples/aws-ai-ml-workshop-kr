
1. UV 환경셋팅 (sh | 환경이름 | 파이선 버젼)
    chmod +x create-uv-env.sh
    ./create-uv-env.sh agent_frame 3.12
    setup dir에서 source .venv/bin/activate


2. opensearch 셋텡
    SSM, OpenSearch 권한 필요
    chmod +x create-opensearch.sh
    ./create-opensearch.sh -v 2.19 -d dongjin-os -u dongjin -p MarsEarth1! -m prod
    (버젼, 도메인네임, 유저네임, 패스워드, 모드 (dev or prod))

3. os indexing
    python os_indexing.py 

4. 오픈서치 mcp 서버 실행
    chmod +x execution-os-mcp-server.sh
    ./execution-os-mcp-server.sh
     



2. FileScopeMCP
    chmod +x install-filescopemcp.sh
    ./install-filescopemcp.sh

    (호스팅) FileScopeMCP 디렉토리에서 ./run.sh



jupyter kernelspec list
jupyter kernelspec remove <커널이름>

