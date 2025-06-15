# import os
# import json
# import pprint
# import logging

# from typing import Literal
# from langchain_core.messages import HumanMessage
# from langgraph.types import Command, interrupt
# from langgraph.graph import END
# from src.agents.agents import create_react_agent
# from src.config import TEAM_MEMBERS
# from src.config.agents import AGENT_LLM_MAP, AGENT_PROMPT_CACHE_MAP
# from src.prompts.template import apply_prompt_template
# from src.tools.search import tavily_tool
# from .types import State, Router

# from textwrap import dedent
# from src.utils.common_utils import get_message_from_string

import os
import json
import pprint
import logging
import asyncio
import traceback
import streamlit as st
from copy import deepcopy
from typing import Literal
from langgraph.types import Command
from langgraph.graph import END

from src.config import TEAM_MEMBERS
from .types import State

from textwrap import dedent
from src.utils.common_utils import get_message_from_string

from src.utils.strands_sdk_utils import strands_utils
from src.tools import python_repl_tool, bash_tool, tavily_tool, crawl_tool
from src.tools.tavily_tool import tavily_search_instance


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
FEEDBACK_FORMAT = "Feedback from {}:\n\n<user_feedback>\n{}\n</user_feedback>\n\n"
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


def clarification_node(state: State):
    """Node for the clarifier agent that improve understanding of user's intent."""
    logger.info(f"{Colors.GREEN}===== Clarification agent starting task ====={Colors.END}")
    
    agent = strands_utils.get_agent(
        agent_name="clarifier",
        state=state,
        tools=[crawl_tool, tavily_tool, python_repl_tool, bash_tool],
        streaming=True
    )

    message = state["request_prompt"]
    agent, response = asyncio.run(strands_utils.process_streaming_response(agent, message, agent_name="clarifier"))
    follow_up_questions = json.loads(response["text"])
    
    clues = state.get("clues", "")
    clues = '\n\n'.join([clues, CLUES_FORMAT.format("clarifier", response["text"])])
    logger.info("Clarification agent completed task")
    logger.debug(f"Clarification agent response: {response["text"]}")

    history = state.get("history", [])
    history.append({"agent":"clarifier", "message": response["text"]})
    return Command(
        update={
            "messages": [get_message_from_string(role="user", string=RESPONSE_FORMAT.format("clarifier", response["text"]), imgs=[])],
            "messages_name": "clarifier",
            "clues":clues,
            "history": history,
            "follow_up_questions": follow_up_questions["questions"]
        },
        goto="human_feedback",
    )

def human_feedback_node(state: State):
    """Node for the human_feedback agent that improve understanding of user's intent."""
    logger.info(f"{Colors.GREEN}===== Human feedback agent starting task ====={Colors.END}")
    follow_up_questions = state.get("follow_up_questions", "")  
    follow_up_questions_str = "\n".join(follow_up_questions)

    # Get feedback on the report plan from interrupt
    interrupt_message = f"""Please provide addtional information on your topics. 
                        \n\n{follow_up_questions_str}\n
                        \nprovide answers on follow-up questions:"""
    
    # 인터럽트 발생시키기
    logger.info(f"{Colors.GREEN}{interrupt_message}{Colors.END}")
    feedback = input()

    history = state.get("history", [])
    history.append({"agent":"human_feedback", "message": feedback})

    return Command(
        update={
            "messages": [get_message_from_string(role="user", string=FEEDBACK_FORMAT.format("human_feedback", feedback), imgs=[])],
            "messages_name": "human_feedback",
            "history": history,
            "user_feedback": feedback
        },
        goto="planner",
    )

def research_node(state: State) -> Command[Literal["supervisor"]]:
    """Node for the researcher agent that performs research tasks."""
    logger.info(f"{Colors.GREEN}===== Research agent starting task ====={Colors.END}")
    
    agent = strands_utils.get_agent(
        agent_name="researcher",
        state=state,
        tools=[crawl_tool, tavily_tool, python_repl_tool, bash_tool],
        streaming=True
    )
    
    clues, messages = state.get("clues", ""), state["messages"]
    message = '\n\n'.join([messages[-1]["content"][-1]["text"], clues])
    agent, response = asyncio.run(strands_utils.process_streaming_response(agent, message, agent_name="researcher"))

    clues = '\n\n'.join([clues, CLUES_FORMAT.format("researcher", response["text"])])
    logger.info("Research agent completed task")
    logger.debug(f"Research agent response: {response["text"]}")

    history = state.get("history", [])
    history.append({"agent":"researcher", "message": response["text"]})
    return Command(
        update={
            "messages": [get_message_from_string(role="user", string=RESPONSE_FORMAT.format("researcher", response["text"]), imgs=[])],
            "messages_name": "researcher",
            "clues": clues,
            "history": history
        },
        goto="supervisor",
    )

def code_node(state: State) -> Command[Literal["supervisor"]]:
    """Node for the coder agent that executes Python code."""
    logger.info(f"{Colors.GREEN}===== Code agent starting task ====={Colors.END}")
    coder_agent = create_react_agent(agent_name="coder")
    result = coder_agent.invoke(state=state)

    clues = state.get("clues", "")
    clues = '\n\n'.join([clues, CLUES_FORMAT.format("coder", result["content"][-1]["text"])])

    logger.debug(f"\n{Colors.RED}Coder - current state messages:\n{pprint.pformat(state['messages'], indent=2, width=100)}{Colors.END}")
    logger.debug(f"\n{Colors.RED}Coder response:\n{pprint.pformat(result["content"][-1]["text"], indent=2, width=100)}{Colors.END}")

    history = state.get("history", [])
    history.append({"agent":"coder", "message": result["content"][-1]["text"]})

    logger.info(f"{Colors.GREEN}===== Coder completed task ====={Colors.END}")
    return Command(
        update={
            "messages": [get_message_from_string(role="user", string=RESPONSE_FORMAT.format("coder", result["content"][-1]["text"]), imgs=[])],
            "messages_name": "coder",
            "clues": clues,
            "history": history
        },
        goto="supervisor",
    )

def browser_node(state: State) -> Command[Literal["supervisor"]]:
    """Node for the browser agent that performs web browsing tasks."""
    logger.info("Browser agent starting task")
    browser_agent = create_react_agent(agent_name="browser")
    result = browser_agent.invoke(state=state)
    
    clues = state.get("clues", "")
    clues = '\n\n'.join([clues, CLUES_FORMAT.format("browser", result["content"][-1]["text"])])
    logger.info("Browser agent completed task")
    logger.debug(f"Browser agent response: {result['content'][-1]["text"]}")

    history = state.get("history", [])
    history.append({"agent":"browser", "message": result["content"][-1]["text"]})
    return Command(
        update={
            "messages": [get_message_from_string(role="user", string=RESPONSE_FORMAT.format("browser", result["content"][-1]["text"]), imgs=[])],
            "messages_name": "browser",
            "clues": clues,
            "history": history
        },
        goto="supervisor"
    )

def supervisor_node(state: State):# -> Command[Literal[*TEAM_MEMBERS, "__end__"]]:
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
    logger.info(f"{Colors.BLUE}===== Planner - Search before planning: {state.get("search_before_planning")} ====={Colors.END}")
    logger.debug(f"\n{Colors.RED}Planner - current state messages:\n{pprint.pformat(state['messages'], indent=2, width=100)}{Colors.END}")
    
    agent = strands_utils.get_agent(
        agent_name="planner",
        state=state,
        streaming=True
    )

    full_plan, messages = state.get("full_plan", ""), state["messages"]
    message = '\n\n'.join([messages[-1]["content"][-1]["text"], FULL_PLAN_FORMAT.format(full_plan)])
     
    if state.get("search_before_planning"):
        searched_content = tavily_search_instance.invoke({"query": state["request"]})
        messages = deepcopy(messages)
        message += f"\n\n# Relative Search Results\n\n{json.dumps([{'title': elem['title'], 'content': elem['content']} for elem in searched_content], ensure_ascii=False)}"
    
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

def reporter_node(state: State) -> Command[Literal["supervisor"]]:
    """Reporter node that write a final report."""
    logger.info(f"{Colors.GREEN}===== Reporter write final report ====={Colors.END}")
    logger.debug(f"\n{Colors.RED}Reporter11 - current state messages:\n{pprint.pformat(state['messages'], indent=2, width=100)}{Colors.END}")

    reporter_agent = create_react_agent(agent_name="reporter")
    result = reporter_agent.invoke(state=state)
    full_response = result["content"][-1]["text"]

    clues = state.get("clues", "")
    clues = '\n\n'.join([clues, CLUES_FORMAT.format("reporter", result["content"][-1]["text"])])

    logger.debug(f"\n{Colors.RED}Reporter - current state messages:\n{pprint.pformat(state['messages'], indent=2, width=100)}{Colors.END}")
    logger.debug(f"\n{Colors.RED}Reporter response:\n{pprint.pformat(full_response, indent=2, width=100)}{Colors.END}")

    history = state.get("history", [])
    history.append({"agent":"reporter", "message": full_response})
    logger.info(f"{Colors.GREEN}===== Reporter completed task ====={Colors.END}")
    return Command(
        update={
            "messages": [get_message_from_string(role="user", string=full_response, imgs=[])],
            "messages_name": "reporter",
            "history": history,
            "clues": clues
        },
        goto="supervisor"
    )