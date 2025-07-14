from mcp.server.fastmcp import FastMCP
import arxiv
import logging
import sys
from typing import Dict, Any, List
from datetime import datetime, timezone
from dateutil import parser

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("arxiv_mcp")

# Maximum number of results setting
MAX_RESULTS = 10

# FastMCP initialization
try:
    mcp = FastMCP(
        name="arxiv_tools",
    )
    logger.info("arXiv MCP server initialized successfully")
except Exception as e:
    err_msg = f"Error: {str(e)}"
    logger.error(f"{err_msg}")

def _is_within_date_range(
    date: datetime, start: datetime | None, end: datetime | None
) -> bool:
    """Check if a date falls within the specified range."""
    if start and not start.tzinfo:
        start = start.replace(tzinfo=timezone.utc)
    if end and not end.tzinfo:
        end = end.replace(tzinfo=timezone.utc)

    if start and date < start:
        return False
    if end and date > end:
        return False
    return True

def _process_paper(paper: arxiv.Result) -> Dict[str, Any]:
    """Process paper information with resource URI."""
    return {
        "id": paper.get_short_id(),
        "title": paper.title,
        "authors": [author.name for author in paper.authors],
        "abstract": paper.summary,
        "categories": paper.categories,
        "published": paper.published.isoformat(),
        "url": paper.pdf_url,
        "resource_uri": f"arxiv://{paper.get_short_id()}",
    }

@mcp.tool()
async def search_papers(
    query: str, 
    max_results: int = 10, 
    date_from: str = None, 
    date_to: str = None, 
    categories: List[str] = None
) -> List[Dict[str, Any]]:
    """Search for papers on arXiv with advanced filtering.
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        date_from: Search start date (YYYY-MM-DD)
        date_to: Search end date (YYYY-MM-DD)
        categories: List of category filters
        
    Returns:
        List of searched papers
    """
    try:
        client = arxiv.Client()
        max_results = min(int(max_results), MAX_RESULTS)

        # Build search query with category filtering
        if categories:
            category_filter = " OR ".join(f"cat:{cat}" for cat in categories)
            query = f"({query}) AND ({category_filter})"

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        # Process results with date filtering
        results = []
        try:
            date_from_obj = parser.parse(date_from).replace(tzinfo=timezone.utc) if date_from else None
            date_to_obj = parser.parse(date_to).replace(tzinfo=timezone.utc) if date_to else None
        except (ValueError, TypeError) as e:
            return [{"error": f"Invalid date format - {str(e)}"}]

        for paper in client.results(search):
            if _is_within_date_range(paper.published, date_from_obj, date_to_obj):
                results.append(_process_paper(paper))

            if len(results) >= max_results:
                break

        return results

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return [{"error": f"Search failed: {str(e)}"}]

@mcp.tool()
async def download_paper(paper_id: str) -> Dict[str, Any]:
    """Download a paper from arXiv.
    
    Args:
        paper_id: arXiv paper ID
        
    Returns:
        Download result information
    """
    try:
        client = arxiv.Client()
        search = arxiv.Search(id_list=[paper_id])
        
        for paper in client.results(search):
            return {
                "id": paper.get_short_id(),
                "title": paper.title,
                "url": paper.pdf_url,
                "download_status": "success",
                "resource_uri": f"arxiv://{paper.get_short_id()}"
            }
        
        return {"error": f"Paper with ID {paper_id} not found"}
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return {"error": f"Download failed: {str(e)}"}

@mcp.tool()
async def read_paper(paper_id: str) -> Dict[str, Any]:
    """Read the content of an arXiv paper.
    
    Args:
        paper_id: arXiv paper ID
        
    Returns:
        Paper content and metadata
    """
    try:
        client = arxiv.Client()
        search = arxiv.Search(id_list=[paper_id])
        
        for paper in client.results(search):
            return {
                "id": paper.get_short_id(),
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "abstract": paper.summary,
                "categories": paper.categories,
                "published": paper.published.isoformat(),
                "content_type": "text",
                "content": paper.summary  # Only providing the summary here, additional work needed to get full content
            }
        
        return {"error": f"Paper with ID {paper_id} not found"}
    except Exception as e:
        logger.error(f"Read error: {str(e)}")
        return {"error": f"Read failed: {str(e)}"}

@mcp.tool()
async def list_papers(category: str = None, max_results: int = 10) -> List[Dict[str, Any]]:
    """Get a list of the latest papers in a specific category.
    
    Args:
        category: arXiv category code
        max_results: Maximum number of results to return
        
    Returns:
        List of papers
    """
    try:
        client = arxiv.Client()
        max_results = min(int(max_results), MAX_RESULTS)
        
        query = f"cat:{category}" if category else ""
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        
        results = []
        for paper in client.results(search):
            results.append(_process_paper(paper))
            if len(results) >= max_results:
                break
                
        return results
    except Exception as e:
        logger.error(f"List error: {str(e)}")
        return [{"error": f"List failed: {str(e)}"}]

if __name__ == "__main__":
    mcp.run()