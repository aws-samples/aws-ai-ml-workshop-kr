"""
SCM file utility functions for managing artifacts and previous results.
Provides standardized methods for reading and writing analysis results.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any


def ensure_artifacts_folder() -> str:
    """
    Ensure artifacts folder exists and return its path.
    
    Returns:
        str: Path to artifacts folder
    """
    artifacts_path = "./artifacts/"
    if not os.path.exists(artifacts_path):
        os.makedirs(artifacts_path)
    return artifacts_path


def save_analysis_result(
    content: str, 
    step_number: int, 
    agent_name: str, 
    filename_override: Optional[str] = None
) -> str:
    """
    Save analysis result to artifacts folder with standardized naming.
    
    Args:
        content: Content to save
        step_number: Step number in workflow (01, 02, etc.)
        agent_name: Name of the agent
        filename_override: Optional custom filename
        
    Returns:
        str: Path to saved file
    """
    artifacts_path = ensure_artifacts_folder()
    
    if filename_override:
        filename = filename_override
    else:
        filename = f"{step_number:02d}_{agent_name}_results.txt"
    
    filepath = os.path.join(artifacts_path, filename)
    
    # Add metadata header
    timestamp = datetime.now().isoformat()
    formatted_content = f"""=== {agent_name.upper()} ANALYSIS RESULTS ===
Generated: {timestamp}
Step: {step_number:02d}
Agent: {agent_name}

{content}
"""
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(formatted_content)
    
    return filepath


def read_previous_results(max_step: Optional[int] = None) -> Dict[str, str]:
    """
    Read all previous analysis results from artifacts folder.
    
    Args:
        max_step: Maximum step number to read (None for all)
        
    Returns:
        Dict[str, str]: Dictionary mapping filename to content
    """
    artifacts_path = ensure_artifacts_folder()
    results = {}
    
    # Standard filenames to look for
    standard_files = [
        "01_research_results.txt",
        "02_business_insights.txt", 
        "03_analysis_plan.txt",
        "04_impact_analysis.txt",
        "05_correlation_analysis.txt",
        "06_mitigation_plan.txt",
        "07_final_report.txt"
    ]
    
    for filename in standard_files:
        if max_step:
            step_num = int(filename[:2])
            if step_num > max_step:
                continue
                
        filepath = os.path.join(artifacts_path, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    results[filename] = f.read()
            except Exception as e:
                results[filename] = f"Error reading file: {str(e)}"
    
    return results


def generate_file_reading_code(
    files_to_read: List[str],
    print_separator: bool = True,
    include_error_handling: bool = True
) -> str:
    """
    Generate Python code for reading previous result files with print statements.
    
    Args:
        files_to_read: List of filenames to read
        print_separator: Whether to include separator lines
        include_error_handling: Whether to include try/catch blocks
        
    Returns:
        str: Python code for reading files
    """
    code_lines = [
        "# Read previous analysis results",
        "import os",
        "print('=== 이전 결과 파일들 확인 ===')",
        ""
    ]
    
    for filename in files_to_read:
        file_label = {
            "01_research_results.txt": "📋 연구 결과",
            "02_business_insights.txt": "💡 비즈니스 인사이트", 
            "03_analysis_plan.txt": "📊 분석 계획",
            "04_impact_analysis.txt": "📈 영향 분석",
            "05_correlation_analysis.txt": "🔗 상관관계 분석",
            "06_mitigation_plan.txt": "🛡️ 대응 방안",
            "07_final_report.txt": "📄 최종 보고서"
        }.get(filename, f"📁 {filename}")
        
        if include_error_handling:
            code_lines.extend([
                "try:",
                f"    with open('./artifacts/{filename}', 'r', encoding='utf-8') as f:",
                f"        {filename.replace('.txt', '_data').replace('-', '_')} = f.read()",
                f"    print('{file_label}:')",
                f"    print({filename.replace('.txt', '_data').replace('-', '_')})",
                f"except FileNotFoundError:",
                f"    print('⚠️ {filename} 파일이 없습니다.')"
            ])
        else:
            code_lines.extend([
                f"with open('./artifacts/{filename}', 'r', encoding='utf-8') as f:",
                f"    {filename.replace('.txt', '_data').replace('-', '_')} = f.read()",
                f"print('{file_label}:')",
                f"print({filename.replace('.txt', '_data').replace('-', '_')})"
            ])
        
        if print_separator:
            code_lines.append("print('\\n' + '='*50 + '\\n')")
        
        code_lines.append("")
    
    return "\n".join(code_lines)


def get_scm_workflow_files() -> List[str]:
    """
    Get the standard list of SCM workflow files in order.
    
    Returns:
        List[str]: Ordered list of workflow filenames
    """
    return [
        "01_research_results.txt",
        "02_business_insights.txt", 
        "03_analysis_plan.txt",
        "04_impact_analysis.txt",
        "05_correlation_analysis.txt",
        "06_mitigation_plan.txt",
        "07_final_report.txt"
    ]


def create_opensearch_mcp_code() -> str:
    """
    Generate standard OpenSearch MCP connection code.
    
    Returns:
        str: Python code for OpenSearch MCP setup
    """
    return """
# OpenSearch MCP 연결 설정
import os
import boto3
from strands import Agent
from strands.tools.mcp import MCPClient
from mcp import stdio_client, StdioServerParameters
from utils.ssm import parameter_store

print("=== OpenSearch MCP 연결 시작 ===")

# 환경변수 설정
region = boto3.Session().region_name
pm = parameter_store(region)

os.environ["OPENSEARCH_URL"] = pm.get_params(key="opensearch_domain_endpoint", enc=False)
os.environ["OPENSEARCH_USERNAME"] = pm.get_params(key="opensearch_user_id", enc=False)
os.environ["OPENSEARCH_PASSWORD"] = pm.get_params(key="opensearch_user_password", enc=True)

# MCP 클라이언트 설정
env_vars = os.environ.copy()
opensearch_mcp_client = MCPClient(
    lambda: stdio_client(StdioServerParameters(
        command="python",
        args=["-m", "mcp_server_opensearch"],
        env=env_vars
    ))
)

print("✅ OpenSearch MCP 클라이언트 설정 완료")
"""


def create_analysis_template(
    agent_name: str,
    step_number: int,
    previous_files: List[str],
    analysis_instructions: str,
    save_filename: str
) -> str:
    """
    Create a standardized analysis code template.
    
    Args:
        agent_name: Name of the agent
        step_number: Step number in workflow
        previous_files: List of previous files to read
        analysis_instructions: Specific analysis instructions
        save_filename: Filename to save results to
        
    Returns:
        str: Complete analysis code template
    """
    file_reading_code = generate_file_reading_code(previous_files)
    opensearch_code = create_opensearch_mcp_code()
    
    template = f"""
당신은 {agent_name} 전문가입니다. 이전 분석 결과들을 바탕으로 작업을 수행해야 합니다.

먼저 이전 결과들을 모두 확인하세요:

```python
{file_reading_code}
```

{analysis_instructions}

{opensearch_code if 'mcp' in analysis_instructions.lower() or 'opensearch' in analysis_instructions.lower() else ''}

분석 결과를 ./artifacts/{save_filename}로 저장하세요:

```python
# 분석 결과 저장
from datetime import datetime

analysis_result = '''
=== {agent_name.upper()} 분석 결과 ===
생성 시간: {{datetime.now().isoformat()}}
단계: {step_number:02d}

[여기에 분석 결과 작성]
'''

with open('./artifacts/{save_filename}', 'w', encoding='utf-8') as f:
    f.write(analysis_result)

print(f"✅ 분석 결과가 ./artifacts/{save_filename}에 저장되었습니다.")
```
"""
    
    return template


def cleanup_artifacts_folder() -> None:
    """Remove all files from artifacts folder."""
    artifacts_path = ensure_artifacts_folder()
    
    for filename in os.listdir(artifacts_path):
        filepath = os.path.join(artifacts_path, filename)
        if os.path.isfile(filepath):
            os.remove(filepath)


def get_artifacts_summary() -> Dict[str, Any]:
    """
    Get summary of all artifacts in the folder.
    
    Returns:
        Dict containing file info and statistics
    """
    artifacts_path = ensure_artifacts_folder()
    
    summary = {
        "folder_path": artifacts_path,
        "files": [],
        "total_files": 0,
        "total_size_bytes": 0,
        "last_modified": None
    }
    
    if not os.path.exists(artifacts_path):
        return summary
    
    latest_time = 0
    
    for filename in os.listdir(artifacts_path):
        filepath = os.path.join(artifacts_path, filename)
        if os.path.isfile(filepath):
            stat = os.stat(filepath)
            file_info = {
                "filename": filename,
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "step_number": None
            }
            
            # Extract step number if follows naming convention
            if filename[:2].isdigit():
                file_info["step_number"] = int(filename[:2])
            
            summary["files"].append(file_info)
            summary["total_size_bytes"] += stat.st_size
            
            if stat.st_mtime > latest_time:
                latest_time = stat.st_mtime
                summary["last_modified"] = file_info["modified"]
    
    summary["total_files"] = len(summary["files"])
    summary["files"].sort(key=lambda x: x.get("step_number", 999))
    
    return summary