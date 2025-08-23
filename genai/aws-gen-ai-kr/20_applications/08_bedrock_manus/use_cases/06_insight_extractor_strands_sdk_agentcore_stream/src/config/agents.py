from typing import Literal, Tuple

# Define available LLM types
LLMType = Literal["basic", "reasoning", "vision"]
CACHEType = Tuple[bool, Literal["default", "ephemeral"]]

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
AGENT_PROMPT_CACHE_MAP: dict[bool, CACHEType] = {
    "coordinator": (False, None),
    "planner": (True, "default"),
    "supervisor": (True, "default"),
    "researcher": (False, None),
    "coder": (False, None),
    "browser": (False, None),
    "reporter": (True, "default")
}

