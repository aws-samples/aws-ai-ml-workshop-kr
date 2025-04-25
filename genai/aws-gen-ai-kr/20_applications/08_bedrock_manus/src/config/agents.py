from typing import Literal

# Define available LLM types
LLMType = Literal["basic", "reasoning", "vision"]

# Define agent-LLM mapping
AGENT_LLM_MAP: dict[str, LLMType] = {
    "coordinator": "basic",
    "planner": "reasoning",
    "supervisor": "reasoning",
    "researcher": "basic",
    "coder": "basic",
    "browser": "vision",
    "reporter": "reasoning"
}

