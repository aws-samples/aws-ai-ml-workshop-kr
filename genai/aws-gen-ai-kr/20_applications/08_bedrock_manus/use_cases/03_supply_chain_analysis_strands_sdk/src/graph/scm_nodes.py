"""
Simplified SCM (Supply Chain Management) specialized agent nodes with file-based workflow.
Each agent uses template-based prompts and saves output to artifacts folder.
"""

import os
import json
import logging
import asyncio
from typing import Literal
from langgraph.types import Command

from src.config import TEAM_MEMBERS
from .types import State
from src.utils.common_utils import get_message_from_string
from src.utils.strands_sdk_utils import strands_utils
from src.tools import python_repl_tool, bash_tool, tavily_tool, crawl_tool

# 로거 설정
logger = logging.getLogger(__name__)
logger.propagate = False
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
handler = logging.StreamHandler()
formatter = logging.Formatter('\n%(levelname)s [%(name)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

RESPONSE_FORMAT = "Response from {}:\n\n<response>\n{}\n</response>\n\n*Please execute the next step.*"
CLUES_FORMAT = "Here is clues from {}:\n\n<clues>\n{}\n</clues>\n\n"

# SCM 에이전트 간 전달 메시지 포맷
SCM_NEXT_STEP_MESSAGE = {
    "scm_researcher": "SCM research completed. Please proceed with business insight analysis.",
    "scm_insight_analyzer": "Business insights completed. Please proceed with analysis planning.", 
    "scm_impact_analyzer": "Impact analysis completed. Please proceed with correlation analysis.",
    "scm_correlation_analyzer": "Correlation analysis completed. Please proceed with mitigation planning.",
    "scm_mitigation_planner": "Mitigation planning completed. All SCM analysis phases finished."
}

class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def ensure_artifacts_folder():
    """Ensure artifacts folder exists"""
    artifacts_path = "./artifacts/"
    if not os.path.exists(artifacts_path):
        os.makedirs(artifacts_path)
    return artifacts_path


def scm_researcher_node(state: State) -> Command[Literal["scm_insight_analyzer"]]:
    """SCM-specialized researcher agent for supply chain disruption analysis."""
    logger.info(f"{Colors.GREEN}===== SCM Researcher starting task ====={Colors.END}")
    
    agent = strands_utils.get_agent(
        agent_name="scm_researcher",
        state=state,
        tools=[crawl_tool, tavily_tool, python_repl_tool, bash_tool],
        streaming=True
    )

    message = state["request_prompt"]
    agent, response = asyncio.run(strands_utils.process_streaming_response(agent, message, agent_name="scm_researcher"))
    
    clues = state.get("clues", "")
    clues = '\n\n'.join([clues, CLUES_FORMAT.format("scm_researcher", response["text"])])
    
    history = state.get("history", [])
    history.append({"agent": "scm_researcher", "message": response["text"]})
    
    logger.info(f"{Colors.GREEN}SCM Researcher completed task{Colors.END}")
    
    return Command(
        update={
            "messages": [get_message_from_string(role="user", string=SCM_NEXT_STEP_MESSAGE["scm_researcher"], imgs=[])],
            "messages_name": "scm_researcher",
            "clues": clues,
            "history": history
        },
        goto="scm_insight_analyzer",
    )


def scm_insight_analyzer_node(state: State) -> Command[Literal["planner"]]:
    """Analyze research results and extract business insights for SCM impact."""
    logger.info(f"{Colors.GREEN}===== SCM Insight Analyzer starting task ====={Colors.END}")
    
    agent = strands_utils.get_agent(
        agent_name="scm_insight_analyzer",
        state=state,
        tools=[python_repl_tool, bash_tool],
        streaming=True
    )
    
    message = state["messages"][-1]["content"][-1]["text"]
    agent, response = asyncio.run(strands_utils.process_streaming_response(agent, message, agent_name="scm_insight_analyzer"))
    
    clues = state.get("clues", "")
    clues = '\n\n'.join([clues, CLUES_FORMAT.format("scm_insight_analyzer", response["text"])])
    
    history = state.get("history", [])
    history.append({"agent": "scm_insight_analyzer", "message": response["text"]})
    
    logger.info(f"{Colors.GREEN}SCM Insight Analyzer completed task{Colors.END}")
    
    return Command(
        update={
            "messages": [get_message_from_string(role="user", string=SCM_NEXT_STEP_MESSAGE["scm_insight_analyzer"], imgs=[])],
            "messages_name": "scm_insight_analyzer", 
            "clues": clues,
            "history": history
        },
        goto="planner",
    )


def scm_impact_analyzer_node(state: State) -> Command[Literal["supervisor"]]:
    """Analyze quantitative impact on SCM KPIs using OpenSearch data."""
    logger.info(f"{Colors.GREEN}===== SCM Impact Analyzer starting task ====={Colors.END}")
    
    agent = strands_utils.get_agent(
        agent_name="scm_impact_analyzer",
        state=state,
        tools=[python_repl_tool, bash_tool],
        streaming=True
    )

    clues, messages = state.get("clues", ""), state["messages"]
    message = '\n\n'.join([messages[-1]["content"][-1]["text"], clues])
    agent, response = asyncio.run(strands_utils.process_streaming_response(agent, message, agent_name="scm_impact_analyzer"))
    
    clues = '\n\n'.join([clues, CLUES_FORMAT.format("scm_impact_analyzer", response["text"])])
    
    history = state.get("history", [])
    history.append({"agent": "scm_impact_analyzer", "message": response["text"]})

    logger.info(f"{Colors.GREEN}SCM Impact Analyzer completed task{Colors.END}")
    
    return Command(
        update={
            "messages": [get_message_from_string(role="user", string=SCM_NEXT_STEP_MESSAGE["scm_impact_analyzer"], imgs=[])],
            "messages_name": "scm_impact_analyzer",
            "clues": clues,
            "history": history
        },
        goto="supervisor",
    )


def scm_correlation_analyzer_node(state: State) -> Command[Literal["supervisor"]]:
    """Analyze correlations and chain effects between SCM KPIs."""
    logger.info(f"{Colors.GREEN}===== SCM Correlation Analyzer starting task ====={Colors.END}")
    
    agent = strands_utils.get_agent(
        agent_name="scm_correlation_analyzer",
        state=state,
        tools=[python_repl_tool, bash_tool],
        streaming=True
    )

    clues, messages = state.get("clues", ""), state["messages"]
    message = '\n\n'.join([messages[-1]["content"][-1]["text"], clues])
    agent, response = asyncio.run(strands_utils.process_streaming_response(agent, message, agent_name="scm_correlation_analyzer"))
    
    clues = '\n\n'.join([clues, CLUES_FORMAT.format("scm_correlation_analyzer", response["text"])])
    
    history = state.get("history", [])
    history.append({"agent": "scm_correlation_analyzer", "message": response["text"]})

    logger.info(f"{Colors.GREEN}SCM Correlation Analyzer completed task{Colors.END}")
    
    return Command(
        update={
            "messages": [get_message_from_string(role="user", string=SCM_NEXT_STEP_MESSAGE["scm_correlation_analyzer"], imgs=[])],
            "messages_name": "scm_correlation_analyzer",
            "clues": clues,
            "history": history
        },
        goto="supervisor",
    )


def scm_mitigation_planner_node(state: State) -> Command[Literal["supervisor"]]:
    """Plan mitigation strategies based on impact and correlation analysis."""
    logger.info(f"{Colors.GREEN}===== SCM Mitigation Planner starting task ====={Colors.END}")
    
    agent = strands_utils.get_agent(
        agent_name="scm_mitigation_planner",
        state=state,
        tools=[python_repl_tool, bash_tool, tavily_tool],
        streaming=True
    )

    clues, messages = state.get("clues", ""), state["messages"]
    message = '\n\n'.join([messages[-1]["content"][-1]["text"], clues])
    agent, response = asyncio.run(strands_utils.process_streaming_response(agent, message, agent_name="scm_mitigation_planner"))
    
    clues = '\n\n'.join([clues, CLUES_FORMAT.format("scm_mitigation_planner", response["text"])])
    
    history = state.get("history", [])
    history.append({"agent": "scm_mitigation_planner", "message": response["text"]})

    logger.info(f"{Colors.GREEN}SCM Mitigation Planner completed task{Colors.END}")
    
    return Command(
        update={
            "messages": [get_message_from_string(role="user", string=SCM_NEXT_STEP_MESSAGE["scm_mitigation_planner"], imgs=[])],
            "messages_name": "scm_mitigation_planner",
            "clues": clues,
            "history": history
        },
        goto="supervisor",
    )