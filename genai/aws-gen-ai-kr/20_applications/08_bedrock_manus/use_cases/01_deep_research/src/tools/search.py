import json
import logging
from langchain_community.tools.tavily_search import TavilySearchResults
from src.config import TAVILY_MAX_RESULTS
from .decorators import create_logged_tool

logger = logging.getLogger(__name__)

# Initialize Tavily search tool with logging
LoggedTavilySearch = create_logged_tool(TavilySearchResults)
#tavily_tool = LoggedTavilySearch(name="tavily_search", max_results=TAVILY_MAX_RESULTS)
tavily_tool = LoggedTavilySearch(
    name="tavily_search",
    max_results=TAVILY_MAX_RESULTS,
    include_answer=False,
    include_raw_content=True,
    include_images=False,
    include_image_descriptions=False,
    search_depth="advanced",  # "basic" 또는 "advanced" 설정 가능
)

def handle_tavily_tool(query):
    '''
    Use this tool to search the internet for real-time information, current events, or specific data. Provides relevant search results from Tavily's search engine API.
    '''
    searched_content = tavily_tool.invoke({"query": query})
    #results = f"\n\n# Relative Search Results\n\n{json.dumps([{**{'title': elem['title'], 'content': elem['content']}, **({'raw_content': elem['raw_content']} if 'raw_content' in elem and elem['raw_content'] is not None else {})} for elem in searched_content], ensure_ascii=False)}"
    results = f"\n\n# Relative Search Results\n\n{json.dumps([{'title': elem['title'], 'url': elem['url'], **(({'raw_content': elem['raw_content']} if 'raw_content' in elem and elem['raw_content'] is not None else {'content': elem['content']}))} for elem in searched_content], ensure_ascii=False)}"
    #results = f"\n\n# Relative Search Results\n\n{json.dumps([{'titile': elem['title'], 'url': elem['url'], 'content': elem['content']} for elem in searched_content], ensure_ascii=False)}"
    print (f'Search Results: "\n\n# Relative Search Results\n\n{json.dumps([{'title': elem['title'], 'url': elem['url'], **(({'raw_content': elem['raw_content']} if 'raw_content' in elem and elem['raw_content'] is not None else {'content': elem['content']}))} for elem in searched_content], ensure_ascii=False)}')
    return results
    
    
    
