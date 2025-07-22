from .env import (
    # Reasoning LLM
    REASONING_MODEL,
    REASONING_BASE_URL,
    REASONING_API_KEY,
    # Basic LLM
    BASIC_MODEL,
    BASIC_BASE_URL,
    BASIC_API_KEY,
    # Vision-language LLM
    VL_MODEL,
    VL_BASE_URL,
    VL_API_KEY,
    # Other configurations
    CHROME_INSTANCE_PATH,
    BROWSER_HEADLESS
)
from .tools import TAVILY_MAX_RESULTS

# Team configuration
#TEAM_MEMBERS = ["researcher", "coder", "reporter"]

# SCM specialized team configuration
SCM_TEAM_MEMBERS = ["scm_impact_analyzer", "scm_correlation_analyzer", "scm_mitigation_planner", "planner", "reporter"]

__all__ = [
    # Reasoning LLM
    "REASONING_MODEL",
    "REASONING_BASE_URL",
    "REASONING_API_KEY",
    # Basic LLM
    "BASIC_MODEL",
    "BASIC_BASE_URL",
    "BASIC_API_KEY",
    # Vision-language LLM
    "VL_MODEL",
    "VL_BASE_URL",
    "VL_API_KEY",
    # Other configurations
    "SCM_TEAM_MEMBERS",
    "TAVILY_MAX_RESULTS",
    "CHROME_INSTANCE_PATH",
    "BROWSER_HEADLESS"
]
