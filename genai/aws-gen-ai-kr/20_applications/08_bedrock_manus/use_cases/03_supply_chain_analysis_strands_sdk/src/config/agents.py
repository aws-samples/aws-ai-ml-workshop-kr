from typing import Literal, Tuple

# Define available LLM types
LLMType = Literal["basic", "reasoning", "vision"]
CACHEType = Tuple[bool, Literal["default", "ephemeral"]]

# Define agent-LLM mapping
AGENT_LLM_MAP: dict[str, LLMType] = {
    "clarifier": "basic",
    "human_feedback": "reasoning",
    "planner": "reasoning",
    "supervisor": "reasoning",
    "researcher": "basic",
    "coder": "basic",
    #"browser": "vision",
    "reporter": "reasoning",
    # SCM specialized agents
    "scm_researcher": "basic",
    "scm_insight_analyzer": "reasoning",
    "scm_impact_analyzer": "basic",
    "scm_correlation_analyzer": "reasoning",
    "scm_mitigation_planner": "reasoning"
}

AGENT_PROMPT_CACHE_MAP: dict[bool, CACHEType] = {
    "clarifier": (False, None),
    "planner": (True, "default"),
    "human_feedback": (False, None),
    "supervisor": (False, None),
    "researcher": (False, None),
    "coder": (False, None),
    #"browser": (False, None),
    "reporter": (True, "default"),
    # SCM specialized agents
    "scm_researcher": (False, None),
    "scm_insight_analyzer": (False, None),
    "scm_impact_analyzer": (False, None),
    "scm_correlation_analyzer": (True, "default"),
    "scm_mitigation_planner": (True, "default")
}