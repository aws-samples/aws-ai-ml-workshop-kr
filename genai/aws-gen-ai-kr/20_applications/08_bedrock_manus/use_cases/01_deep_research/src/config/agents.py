from typing import Literal, Tuple

# Define available LLM types
LLMType = Literal["basic", "reasoning", "vision"]
CACHEType = Tuple[bool, Literal["default", "ephemeral"]]

# Define agent-LLM mapping
AGENT_LLM_MAP: dict[str, LLMType] = {
    "clarifier": "basic",
    #"coordinator": "basic",
    "human_feedback": "reasoning",
    "planner": "reasoning",
    "supervisor": "reasoning",
    "researcher": "basic",
    "coder": "basic",
    "browser": "vision",
    "reporter": "reasoning"
}
AGENT_PROMPT_CACHE_MAP: dict[bool, CACHEType] = {
    "clarifier": (False, "default"),
    "planner": (True, "default"),
    "human_feedback": (False, "default"),
    "supervisor": (True, "default"),
    "researcher": (False, "default"),
    "coder": (False, "default"),
    "browser": (False, "default"),
    "reporter": (True, "default")
    # "coordinator": (False, "default"),
}

