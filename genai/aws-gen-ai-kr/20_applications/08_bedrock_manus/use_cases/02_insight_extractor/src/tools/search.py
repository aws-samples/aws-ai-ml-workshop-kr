import json
import logging
from langchain_community.tools.tavily_search import TavilySearchResults
from src.config import TAVILY_MAX_RESULTS
from .decorators import create_logged_tool

logger = logging.getLogger(__name__)

# Initialize Tavily search tool with logging
LoggedTavilySearch = create_logged_tool(TavilySearchResults)
tavily_tool = LoggedTavilySearch(name="tavily_search", max_results=TAVILY_MAX_RESULTS)


def handle_tavily_tool(query):
    '''
    Use this tool to search the internet for real-time information, current events, or specific data. Provides relevant search results from Tavily's search engine API.
    '''
    searched_content = tavily_tool.invoke({"query": query})
    results = f"\n\n# Relative Search Results\n\n{json.dumps([{'titile': elem['title'], 'url': elem['url'], 'content': elem['content']} for elem in searched_content], ensure_ascii=False)}"
    print (f'Search Results: \n\n {[{'titile': elem['title'], 'url': elem['url'], 'content': elem['content']} for elem in searched_content]}')
    return results
    
    
    
