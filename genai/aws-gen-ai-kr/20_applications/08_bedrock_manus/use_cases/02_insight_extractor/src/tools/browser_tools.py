from typing import Dict, Any, List
from src.tools.browser import handle_browser_tool

tool_list = [
    {
        "toolSpec": {
            "name": "browser_tool",
            "description": "Use this tool to interact with web browsers. Input should be a natural language description of what you want to do with the browser, such as 'Go to google.com and search for browser-use', or 'Navigate to Reddit and find the top post about AI'.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "instruction": {
                            "type": "string",
                            "description": "The instruction to use browser"
                        }
                    },
                    "required": ["instruction"]
                }
            }
        }
    }
]

browser_tool_config = {
    "tools": tool_list,
    # "toolChoice": {
    #    "tool": {
    #        "name": "summarize_email"
    #    }
    # }
}

def process_browser_tool(tool) -> str:
    """Process a tool invocation
    
    Args:
        tool_name: Name of the tool to invoke
        tool_input: Input parameters for the tool
        
    Returns:
        Result of the tool invocation as a string
    """
    
    tool_name, tool_input = tool["name"], tool["input"]

    print ("process_browser_tool", "tool_input", tool_input)
    
    if tool_name == "browser_tool":
        # Create a new instance of the Tavily search tool
        results = handle_browser_tool(instruction=tool_input["instruction"])

        print ("browser_results", results)

        tool_result = {
            "toolUseId": tool['toolUseId'],
            "content": [{"json": {"text": results}}]
        }
    else:
        print (f"Unknown tool: {tool_name}")
        
    resutls = {"role": "user","content": [{"toolResult": tool_result}]}
    
    return resutls