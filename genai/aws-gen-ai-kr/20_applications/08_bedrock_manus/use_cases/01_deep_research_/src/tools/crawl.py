import logging
from typing import Annotated

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from .decorators import log_io

from src.crawler import Crawler

logger = logging.getLogger(__name__)

@log_io
def handle_crawl_tool(url: Annotated[str, "The url to crawl."]) -> str:
    """Handler for crawl_tool invocations
    
    Args:
        input_data: Dictionary containing the 'url' parameter
        
    Returns:
        String containing the crawled content in markdown format
    """
    #url = input_data['url']
    url = url["url"]
    logger.info(f"Crawling URL: {url}")
    try:
        # Initialize the crawler
        crawler = Crawler()
        
        # Crawl the URL
        article = crawler.crawl(url)
        
        # Get the message content
        content = article.to_message()[-1]["text"]
        
        return content
    except Exception as e:
        error_msg = f"Failed to crawl. Error: {repr(e)}"
        logger.error(error_msg)
        return error_msg
