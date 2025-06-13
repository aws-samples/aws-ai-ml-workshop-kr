import logging
from typing import Annotated

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from .decorators import log_io

from src.crawler import Crawler

logger = logging.getLogger(__name__)


# @tool
# @log_io
# def crawl_tool(
#     url: Annotated[str, "The url to crawl."],
# ) -> HumanMessage:
#     """Use this to crawl a url and get a readable content in markdown format."""
#     try:
#         crawler = Crawler()
#         article = crawler.crawl(url)
#         return {"role": "user", "content": article.to_message()}
#     except BaseException as e:
#         error_msg = f"Failed to crawl. Error: {repr(e)}"
#         logger.error(error_msg)
#         return error_msg
    
#def handle_crawl_tool(input_data: Dict[str, Any]) -> str:
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
        
        print ("article", article.to_message()[-1]["text"])
        
        # Get the message content
        content = article.to_message()[-1]["text"]
        
        return content
    except Exception as e:
        error_msg = f"Failed to crawl. Error: {repr(e)}"
        logger.error(error_msg)
        return error_msg
