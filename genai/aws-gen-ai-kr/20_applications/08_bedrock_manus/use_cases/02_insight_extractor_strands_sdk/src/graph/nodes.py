import os
import json
import pprint
import logging
import asyncio
import traceback
import streamlit as st
from typing import Literal
from langgraph.types import Command

from src.config import TEAM_MEMBERS
from .types import State

from textwrap import dedent
from src.utils.common_utils import get_message_from_string

from src.utils.strands_sdk_utils import strands_utils
from src.tools import python_repl_tool, bash_tool

application = os.environ.get('APP', 'False')
if application == 'True': from src.utils.strands_sdk_utils_st import strands_utils
else: from src.utils.strands_sdk_utils import strands_utils

# 새 핸들러와 포맷터 설정
logger = logging.getLogger(__name__)
logger.propagate = False  # 상위 로거로 메시지 전파 중지
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
handler = logging.StreamHandler()
formatter = logging.Formatter('\n%(levelname)s [%(name)s] %(message)s')  # 로그 레벨이 동적으로 표시되도록 변경
handler.setFormatter(formatter)
logger.addHandler(handler)
# DEBUG와 INFO 중 원하는 레벨로 설정
logger.setLevel(logging.INFO)  # 기본 레벨은 INFO로 설정

RESPONSE_FORMAT = "Response from {}:\n\n<response>\n{}\n</response>\n\n*Please execute the next step.*"
FULL_PLAN_FORMAT = "Here is full plan :\n\n<full_plan>\n{}\n</full_plan>\n\n*Please consider this to select the next step.*"
CLUES_FORMAT = "Here is clues form {}:\n\n<clues>\n{}\n</clues>\n\n"

class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def code_node(state: State) -> Command[Literal["supervisor"]]:
    """Node for the coder agent that executes Python code."""
    logger.info(f"{Colors.GREEN}===== Code agent starting task ====={Colors.END}")
        
    agent = strands_utils.get_agent(
        agent_name="coder",
        state=state,
        tools=[python_repl_tool, bash_tool],
        streaming=True
    )

    clues, messages = state.get("clues", ""), state["messages"] 
    message = '\n\n'.join([messages[-1]["content"][-1]["text"], clues])
    agent, response = asyncio.run(strands_utils.process_streaming_response(agent, message, agent_name="coder"))

    clues = state.get("clues", "")
    clues = '\n\n'.join([clues, CLUES_FORMAT.format("coder", response["text"])])

    logger.debug(f"\n{Colors.RED}Coder - current state messages:\n{pprint.pformat(state['messages'], indent=2, width=100)}{Colors.END}")
    logger.debug(f"\n{Colors.RED}Coder response:\n{pprint.pformat(response["text"], indent=2, width=100)}{Colors.END}")

    history = state.get("history", [])
    history.append({"agent":"coder", "message": response["text"]})

    logger.info(f"{Colors.GREEN}===== Coder completed task ====={Colors.END}")
    return Command(
        update={
            "messages": [get_message_from_string(role="user", string=RESPONSE_FORMAT.format("coder", response["text"]), imgs=[])],
            "messages_name": "coder",
            "clues": clues,
            "history": history
        },
        goto="supervisor",
    )

def supervisor_node(state: State) -> Command[Literal[*TEAM_MEMBERS, "__end__"]]:
    """Supervisor node that decides which agent should act next."""
    logger.info(f"{Colors.GREEN}===== Supervisor evaluating next action ====={Colors.END}")
        
    agent = strands_utils.get_agent(
        agent_name="supervisor",
        state=state,
        streaming=True
    )

    clues, full_plan, messages = state.get("clues", ""), state.get("full_plan", ""), state["messages"]
    message = '\n\n'.join([messages[-1]["content"][-1]["text"], FULL_PLAN_FORMAT.format(full_plan), clues])
    agent, response = asyncio.run(strands_utils.process_streaming_response(agent, message, agent_name="supervisor"))

    #full_response = response["text"]
    if response["text"].startswith("```json"): response["text"] = response["text"].removeprefix("```json")
    if response["text"].endswith("```"): response["text"] = response["text"].removesuffix("```")
    
    response["text"] = json.loads(response["text"])   
    goto = response["text"]["next"]

    logger.debug(f"\n{Colors.RED}Supervisor - current state messages:\n{pprint.pformat(state['messages'], indent=2, width=100)}{Colors.END}")
    logger.debug(f"\n{Colors.RED}Supervisor response:{response["text"]}{Colors.END}")

    if goto == "FINISH":
        goto = "__end__"
        logger.info(f"\n{Colors.GREEN}===== Workflow completed ====={Colors.END}")
    else:
        logger.info(f"{Colors.GREEN}Supervisor delegating to: {goto}{Colors.END}")

    history = state.get("history", [])
    history.append({"agent":"supervisor", "message": response["text"]})
    logger.info(f"{Colors.GREEN}===== Supervisor completed task ====={Colors.END}")
    return Command(
        goto=goto,
        update={
            "next": goto,
            "history": history
        }
    )

def planner_node(state: State) -> Command[Literal["supervisor", "__end__"]]:
    """Planner node that generate the full plan."""
    logger.info(f"{Colors.GREEN}===== Planner generating full plan ====={Colors.END}")
    logger.debug(f"\n{Colors.RED}Planner - current state messages:\n{pprint.pformat(state['messages'], indent=2, width=100)}{Colors.END}")
    
    agent = strands_utils.get_agent(
        agent_name="planner",
        state=state,
        streaming=True
    )

    full_plan, messages = state.get("full_plan", ""), state["messages"]
    message = '\n\n'.join([messages[-1]["content"][-1]["text"], FULL_PLAN_FORMAT.format(full_plan)])
    agent, response = asyncio.run(strands_utils.process_streaming_response(agent, message, agent_name="planner"))
    logger.debug(f"\n{Colors.RED}Planner response:\n{pprint.pformat(response["text"], indent=2, width=100)}{Colors.END}")

    goto = "supervisor"
        
    history = state.get("history", [])
    history.append({"agent":"planner", "message": response["text"]})
    logger.info(f"{Colors.GREEN}===== Planner completed task ====={Colors.END}")
    return Command(
        update={
            "messages": [get_message_from_string(role="user", string=response["text"], imgs=[])],
            "messages_name": "planner",
            "full_plan": response["text"],
            "history": history
        },
        goto=goto,
    )

def coordinator_node(state: State) -> Command[Literal["planner", "__end__"]]:
    """Coordinator node that communicate with customers."""
    logger.info(f"{Colors.GREEN}===== Coordinator talking...... ====={Colors.END}")

    agent = strands_utils.get_agent(
        agent_name="coordinator",
        state=state,
        streaming=True
    )

    message = state["request_prompt"]
    agent, response = asyncio.run(strands_utils.process_streaming_response(agent, message, agent_name="coordinator"))
    
    state["messages"] = agent.messages
    logger.debug(f"\n{Colors.RED}Current state messages:\n{pprint.pformat(state['messages'][:-1], indent=2, width=100)}{Colors.END}")
    logger.debug(f"\n{Colors.RED}Coordinator response:\n{pprint.pformat(response["text"], indent=2, width=100)}{Colors.END}")

    goto = "__end__"
    if "handoff_to_planner" in response["text"]: goto = "planner"

    history = state.get("history", [])
    history.append({"agent":"coordinator", "message": response["text"]})

    logger.info(f"{Colors.GREEN}===== Coordinator completed task ====={Colors.END}")

    return Command(
        update={"history": history},
        goto=goto,
    )

def reporter_node(state: State) -> Command[Literal["supervisor"]]:
    """Reporter node that write a final report."""
    logger.info(f"{Colors.GREEN}===== Reporter write final report ====={Colors.END}")
    logger.debug(f"\n{Colors.RED}Reporter11 - current state messages:\n{pprint.pformat(state['messages'], indent=2, width=100)}{Colors.END}")
    
    agent = strands_utils.get_agent(
        agent_name="reporter",
        state=state,
        tools=[python_repl_tool, bash_tool],
        streaming=True
    )

    clues, messages = state.get("clues", ""), state["messages"] 
    message = '\n\n'.join([messages[-1]["content"][-1]["text"], clues])
    agent, response = asyncio.run(strands_utils.process_streaming_response(agent, message, agent_name="reporter"))

    clues = '\n\n'.join([clues, CLUES_FORMAT.format("reporter", response["text"])])
    logger.debug(f"\n{Colors.RED}Reporter - current state messages:\n{pprint.pformat(state['messages'], indent=2, width=100)}{Colors.END}")
    logger.debug(f"\n{Colors.RED}Reporter response:\n{pprint.pformat(response["text"], indent=2, width=100)}{Colors.END}")

    history = state.get("history", [])
    history.append({"agent":"reporter", "message": response["text"]})
    logger.info(f"{Colors.GREEN}===== Reporter completed task ====={Colors.END}")
    return Command(
        update={
            "messages": [get_message_from_string(role="user", string=RESPONSE_FORMAT.format("reporter", response["text"]), imgs=[])],
            "messages_name": "reporter",
            "history": history,
            "clues": clues
        },
        goto="supervisor"
    )