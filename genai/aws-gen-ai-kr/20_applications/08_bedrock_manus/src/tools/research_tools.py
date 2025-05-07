from typing import Dict, Any, List
from src.tools.crawl import handle_crawl_tool
from src.tools.search import handle_tavily_tool
from src.tools.bash_tool import handle_bash_tool
from src.tools.python_repl import handle_python_repl_tool

tool_list = [
    {
        "toolSpec": {
            "name": "crawl_tool",
            "description": "Use this to crawl a url and get a readable content in markdown format.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The url to crawl."
                        }
                    },
                    "required": ["url"]
                }
            }
        }
    },
    {
        "toolSpec": {
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
    },
     {
        "toolSpec": {
            "name": "python_repl_tool",
            "description": "Use this to execute python code and do data analysis or calculation. If you want to see the output of a value, you should print it out with `print(...)`. This is visible to the user.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The python code to execute to do further analysis or calculation."
                        }
                    },
                    "required": ["code"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "bash_tool",
            "description": "Use this to execute bash command and do necessary operations.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "cmd": {
                            "type": "string",
                            "description": "The bash command to be executed."
                        }
                    },
                    "required": ["cmd"]
                }
            }
        }
    }
]

research_tool_config = {
    "tools": tool_list,
    # "toolChoice": {
    #    "tool": {
    #        "name": "summarize_email"
    #    }
    # }
}

def process_search_tool(tool) -> str:
    """Process a tool invocation
    
    Args:
        tool_name: Name of the tool to invoke
        tool_input: Input parameters for the tool
        
    Returns:
        Result of the tool invocation as a string
    """
    
    tool_name, tool_input = tool['name'], tool['input']
    
    if tool_name == "tavily_tool":
        # Create a new instance of the Tavily search tool
        results = handle_tavily_tool(query=tool_input["query"])
        tool_result = {
            "toolUseId": tool['toolUseId'],
            "content": [{"json": {"text": results}}]
        }
        #return response
    elif tool_name == "crawl_tool":
        results = handle_crawl_tool(tool_input)
        tool_result = {
            "toolUseId": tool['toolUseId'],
            "content": [{"json": {"text": results}}]
        }
    elif tool_name == "python_repl_tool":
        # Create a new instance of the Tavily search tool
        results = handle_python_repl_tool(code=tool_input["code"])
        tool_result = {
            "toolUseId": tool['toolUseId'],
            "content": [{"json": {"text": results}}]
        }
        #return response
    elif tool_name == "bash_tool":
        results = handle_bash_tool(cmd=tool_input["cmd"])
        tool_result = {
            "toolUseId": tool['toolUseId'],
            "content": [{"json": {"text": results}}]
        }
    else:
        print (f"Unknown tool: {tool_name}")
        
    resutls = {"role": "user","content": [{"toolResult": tool_result}]}
    
    return resutls