from typing import Literal
from typing_extensions import TypedDict
#from langgraph.graph import MessagesState

from src.config import TEAM_MEMBERS

# Define routing options
OPTIONS = TEAM_MEMBERS + ["FINISH"]

class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next: Literal[*OPTIONS]
    
class State(TypedDict):
    """State for the agent system, extends MessagesState with next field."""

    # Constants
    TEAM_MEMBERS: list[str]

    # Runtime Variables
    next: str
    full_plan: str
    
    # Messages
    messages: list[dict]
    messages_name: str
    history: list[dict]

    # Results
    artifacts: list[list]
    clues: str
    
    # Request
    request: str
    request_prompt: str