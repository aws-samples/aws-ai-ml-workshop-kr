import json
import logging
import os
from typing import Any, Annotated
from strands.types.tools import ToolResult, ToolUse
from langchain_community.tools.tavily_search import TavilySearchResults
from src.tools.decorators import log_io, create_logged_tool

# Load TAVILY_MAX_RESULTS from environment variable, default to 5
TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "5"))

# 새 핸들러와 포맷터 설정
logger = logging.getLogger(__name__)
logger.propagate = False  # 상위 로거로 메시지 전파 중지
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
handler = logging.StreamHandler()
formatter = logging.Formatter('\n%(levelname)s [%(name)s] %(message)s')  # 로그 레벨이 동적으로 표시되도록 변경
handler.setFormatter(formatter)
logger.addHandler(handler)
# DEBUG와 INFO 중 원하는 레벨로 설정
logger.setLevel(logging.INFO)  # 기본 레벨은 INFO로 설정

TOOL_SPEC = {
    "name": "tavily_tool",
    "description": "Use this tool to search the internet for real-time information, current events, or specific data. Provides relevant search results from Tavily's search engine API.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on the internet."
                }
            },
            "required": ["query"]
        }
    }
}

class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# Initialize Tavily search tool with logging
LoggedTavilySearch = create_logged_tool(TavilySearchResults)
tavily_search_instance = LoggedTavilySearch(
    name="tavily_search",
    max_results=TAVILY_MAX_RESULTS,
    include_answer=False,
    include_raw_content=False,
    include_images=False,
    include_image_descriptions=False,
    search_depth="advanced",  # "basic" 또는 "advanced" 설정 가능
)

@log_io
def handle_tavily_tool(query: Annotated[str, "The search query to look up on the internet."]) -> str:
    """
    Use this tool to search the internet for real-time information, current events, or specific data. 
    Provides relevant search results from Tavily's search engine API.
    """
    logger.info(f"{Colors.BLUE}===== Searching for: {query} ====={Colors.END}")
    try:
        searched_content = tavily_search_instance.invoke({"query": query})
        
        results = f"\n\n# Relative Search Results\n\n{json.dumps([{'title': elem['title'], 'url': elem['url'], **(({'raw_content': elem['raw_content']} if 'raw_content' in elem and elem['raw_content'] is not None else {'content': elem['content']}))} for elem in searched_content], ensure_ascii=False)}"
        
        logger.info(f"{Colors.GREEN}===== Search successful ====={Colors.END}")
        logger.debug(f'Search Results: {results}')
        
        return results
    except Exception as e:
        error_msg = f"Failed to search. Error: {repr(e)}"
        logger.debug(f"{Colors.RED}Failed to search. Error: {repr(e)}{Colors.END}")
        return error_msg

# Function name must match tool name
def tavily_tool(tool: ToolUse, **kwargs: Any) -> ToolResult:
    tool_use_id = tool["toolUseId"]
    query = tool["input"]["query"]
    
    # Use the existing handle_tavily_tool function
    result = handle_tavily_tool(query)
    
    # Check if search was successful based on the result string
    if "Failed to search" in result:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": result}]
        }
    else:
        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": result}]
        }