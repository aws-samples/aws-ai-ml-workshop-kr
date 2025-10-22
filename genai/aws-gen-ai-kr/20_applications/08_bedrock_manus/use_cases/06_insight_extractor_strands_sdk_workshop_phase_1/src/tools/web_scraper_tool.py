import logging
from typing import Any, Annotated
from strands.types.tools import ToolResult, ToolUse
from src.tools.decorators import log_io


# Simple logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TOOL_SPEC = {
    "name": "web_scraper_tool",
    "description": "Fetches and extracts text content from a given URL. Returns clean text content from the webpage, removing HTML tags and scripts.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the webpage to scrape. Must be a valid HTTP or HTTPS URL."
                },
                "timeout": {
                    "type": "integer",
                    "description": "Request timeout in seconds. Default is 30 seconds.",
                    "default": 30
                }
            },
            "required": ["url"]
        }
    }
}

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    END = '\033[0m'

@log_io
def handle_web_scraper_tool(
    url: Annotated[str, "The URL of the webpage to scrape"],
    timeout: Annotated[int, "Request timeout in seconds"] = 30
):
    """Fetches and extracts text content from a given URL."""

    print()  # Add newline before log
    logger.info(f"\n{Colors.GREEN}Scraping URL: {url}{Colors.END}")

    try:
        # Import required libraries
        import requests
        from bs4 import BeautifulSoup

        # Validate URL format
        if not url.startswith(('http://', 'https://')):
            raise ValueError("URL must start with http:// or https://")

        # Set headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Fetch the webpage
        logger.info(f"{Colors.YELLOW}Fetching webpage...{Colors.END}")
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()  # Raise exception for bad status codes

        # Parse HTML content
        logger.info(f"{Colors.YELLOW}Parsing HTML content...{Colors.END}")
        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style", "meta", "link"]):
            script.decompose()

        # Extract text content
        text = soup.get_text()

        # Clean up text: remove extra whitespace and empty lines
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)

        # Log success with content preview
        preview = text[:200] + "..." if len(text) > 200 else text
        logger.info(f"{Colors.GREEN}Successfully scraped {len(text)} characters{Colors.END}")
        logger.info(f"{Colors.YELLOW}Preview: {preview}{Colors.END}")

        # Format result with metadata
        result = f"URL: {url}\n"
        result += f"Status Code: {response.status_code}\n"
        result += f"Content Length: {len(text)} characters\n"
        result += f"Encoding: {response.encoding}\n"
        result += f"\n{'='*80}\n"
        result += f"TEXT CONTENT:\n"
        result += f"{'='*80}\n\n"
        result += text

        logger.info(f"{Colors.GREEN}web_scraper_tool completed successfully{Colors.END}")
        return result

    except requests.exceptions.Timeout:
        error_message = f"Error: Request timeout after {timeout} seconds for URL: {url}"
        logger.error(f"{Colors.RED}{error_message}{Colors.END}")
        return error_message

    except requests.exceptions.ConnectionError:
        error_message = f"Error: Failed to connect to URL: {url}. Please check the URL and your internet connection."
        logger.error(f"{Colors.RED}{error_message}{Colors.END}")
        return error_message

    except requests.exceptions.HTTPError as e:
        error_message = f"Error: HTTP error occurred: {str(e)}"
        logger.error(f"{Colors.RED}{error_message}{Colors.END}")
        return error_message

    except ValueError as e:
        error_message = f"Error: Invalid URL format - {str(e)}"
        logger.error(f"{Colors.RED}{error_message}{Colors.END}")
        return error_message

    except ImportError as e:
        error_message = f"Error: Missing required library. Please install: pip install requests beautifulsoup4"
        logger.error(f"{Colors.RED}{error_message}{Colors.END}")
        return error_message

    except Exception as e:
        error_message = f"Error in web_scraper_tool: {str(e)}"
        logger.error(f"{Colors.RED}Error: {str(e)}{Colors.END}")
        return error_message

# Function name must match tool name
def web_scraper_tool(tool: ToolUse, **_kwargs: Any) -> ToolResult:
    tool_use_id = tool["toolUseId"]
    url = tool["input"]["url"]
    timeout = tool["input"].get("timeout", 30)

    # Use the existing handle function
    result = handle_web_scraper_tool(url, timeout)

    # Check if execution was successful
    if result.startswith("Error"):
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

if __name__ == "__main__":
    # Test example
    test_url = "https://www.example.com"
    print(f"Testing web scraper with URL: {test_url}")
    print(handle_web_scraper_tool(test_url))
