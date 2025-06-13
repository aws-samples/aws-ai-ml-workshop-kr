import os
import json
import pprint
import logging
from typing import Literal
from langgraph.types import Command
from langgraph.graph import END

from src.config import TEAM_MEMBERS
from src.config.agents import AGENT_LLM_MAP, AGENT_PROMPT_CACHE_MAP
from src.prompts.template import apply_prompt_template
from .types import State

from textwrap import dedent
from src.utils.common_utils import get_message_from_string

from strands import Agent, tool
from src.utils.strands_sdk_utils import strands_utils
from src.tools import python_repl_tool, bash_tool

llm_module = os.environ.get('LLM_MODULE', 'src.agents.llm')
if llm_module == 'src.agents.llm_st': from src.agents.llm_st import get_llm_by_type
else: from src.agents.llm import get_llm_by_type

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
    
    #coder_agent = create_react_agent(agent_name="coder")
    #result = coder_agent.invoke(state=state)

    prompt_cache, cache_type = AGENT_PROMPT_CACHE_MAP["coder"]
    if prompt_cache: logger.debug(f"{Colors.GREEN}Coder - Prompt Cache Enabled{Colors.END}")
    else: logger.debug(f"{Colors.GREEN}Coder - Prompt Cache Disabled{Colors.END}")

    if "reasoning" in AGENT_LLM_MAP["coder"]: enable_reasoning = True
    else: enable_reasoning = False

    system_prompts = apply_prompt_template("coder", state)

    llm = get_llm_by_type(AGENT_LLM_MAP["coder"], cache_type, enable_reasoning)    
    llm.config["streaming"] = True

    agent = Agent(
        model=llm,
        system_prompt=system_prompts,
        tools=[python_repl_tool, bash_tool]
    )

    clues, messages = state.get("clues", ""), state["messages"] 
    message = '\n\n'.join([messages[-1]["content"][-1]["text"], clues])
    response = agent(message)
    response = strands_utils.parsing_text_from_response(response)

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
    
    prompt_cache, cache_type = AGENT_PROMPT_CACHE_MAP["supervisor"]
    if prompt_cache: logger.debug(f"{Colors.GREEN}Supervisor - Prompt Cache Enabled{Colors.END}")
    else: logger.debug(f"{Colors.GREEN}Supervisor - Prompt Cache Disabled{Colors.END}")

    system_prompts = apply_prompt_template("supervisor", state)
    enable_reasoning = False

    llm = get_llm_by_type(AGENT_LLM_MAP["supervisor"], cache_type, enable_reasoning)    
    llm.config["streaming"] = True

    agent = Agent(
        model=llm,
        system_prompt=system_prompts
    )

    clues, full_plan, messages = state.get("clues", ""), state.get("full_plan", ""), state["messages"]    
    message = '\n\n'.join([messages[-1]["content"][-1]["text"], FULL_PLAN_FORMAT.format(full_plan), clues])
    response = agent(message)
    response = strands_utils.parsing_text_from_response(response)

    full_response = response["text"]

    if full_response.startswith("```json"): full_response = full_response.removeprefix("```json")
    if full_response.endswith("```"): full_response = full_response.removesuffix("```")
    
    full_response = json.loads(full_response)   
    goto = full_response["next"]

    logger.debug(f"\n{Colors.RED}Supervisor - current state messages:\n{pprint.pformat(state['messages'], indent=2, width=100)}{Colors.END}")
    logger.debug(f"\n{Colors.RED}Supervisor response:{full_response}{Colors.END}")

    if goto == "FINISH":
        goto = "__end__"
        logger.info(f"\n{Colors.GREEN}===== Workflow completed ====={Colors.END}")
    else:
        logger.info(f"{Colors.GREEN}Supervisor delegating to: {goto}{Colors.END}")

    history = state.get("history", [])
    history.append({"agent":"supervisor", "message": full_response})
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
    
    if "reasoning" in AGENT_LLM_MAP["planner"]: enable_reasoning = True
    else: enable_reasoning = False

    prompt_cache, cache_type = AGENT_PROMPT_CACHE_MAP["planner"]
    if prompt_cache: logger.debug(f"{Colors.GREEN}Planner - Prompt Cache Enabled{Colors.END}")
    else: logger.debug(f"{Colors.GREEN}Planner - Prompt Cache Disabled{Colors.END}")
    system_prompts = apply_prompt_template("planner", state)   
    
    full_plan, messages = state.get("full_plan", ""), state["messages"]
    
    llm = get_llm_by_type(AGENT_LLM_MAP["planner"], cache_type, enable_reasoning)    
    llm.config["streaming"] = True

    agent = Agent(
        model=llm,
        system_prompt=system_prompts
    )

    message = '\n\n'.join([messages[-1]["content"][-1]["text"], FULL_PLAN_FORMAT.format(full_plan)])
    response = agent(message)
    response = strands_utils.parsing_text_from_response(response)

    full_response = response["text"]
    logger.debug(f"\n{Colors.RED}Planner response:\n{pprint.pformat(full_response, indent=2, width=100)}{Colors.END}")

    goto = "supervisor"
        
    history = state.get("history", [])
    history.append({"agent":"planner", "message": full_response})
    logger.info(f"{Colors.GREEN}===== Planner completed task ====={Colors.END}")
    return Command(
        update={
            "messages": [get_message_from_string(role="user", string=full_response, imgs=[])],
            "messages_name": "planner",
            "full_plan": full_response,
            "history": history
        },
        goto=goto,
    )

def coordinator_node(state: State) -> Command[Literal["planner", "__end__"]]:
    """Coordinator node that communicate with customers."""
    logger.info(f"{Colors.GREEN}===== Coordinator talking...... ====={Colors.END}")
    
    if "reasoning" in AGENT_LLM_MAP["coordinator"]: enable_reasoning = True
    else: enable_reasoning = False

    prompt_cache, cache_type = AGENT_PROMPT_CACHE_MAP["coordinator"]
    if prompt_cache: logger.debug(f"{Colors.GREEN}Coordinator - Prompt Cache Enabled{Colors.END}")
    else: logger.debug(f"{Colors.GREEN}Coordinator - Prompt Cache Disabled{Colors.END}")

    system_prompts = apply_prompt_template("coordinator", state)
    llm = get_llm_by_type(AGENT_LLM_MAP["coordinator"], cache_type, enable_reasoning)    
    llm.config["streaming"] = True

    agent = Agent(
        model=llm,
        system_prompt=system_prompts
    )

    message = state["request_prompt"]
    response = agent(message)
    response = strands_utils.parsing_text_from_response(response)

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

    
    
    #reporter_agent = create_react_agent(agent_name="reporter")
    #result = reporter_agent.invoke(state=state)
    #full_response = result["content"][-1]["text"]

    prompt_cache, cache_type = AGENT_PROMPT_CACHE_MAP["reporter"]
    if prompt_cache: logger.debug(f"{Colors.GREEN}Reporter - Prompt Cache Enabled{Colors.END}")
    else: logger.debug(f"{Colors.GREEN}Reporter - Prompt Cache Disabled{Colors.END}")

    if "reasoning" in AGENT_LLM_MAP["reporter"]: enable_reasoning = True
    else: enable_reasoning = False

    system_prompts = apply_prompt_template("reporter", state)

    llm = get_llm_by_type(AGENT_LLM_MAP["reporter"], cache_type, enable_reasoning)    
    llm.config["streaming"] = True

    agent = Agent(
        model=llm,
        system_prompt=system_prompts,
        tools=[python_repl_tool, bash_tool]
    )

    clues, messages = state.get("clues", ""), state["messages"] 
    message = '\n\n'.join([messages[-1]["content"][-1]["text"], clues])
    response = agent(message)
    response = strands_utils.parsing_text_from_response(response)

    clues = '\n\n'.join([clues, CLUES_FORMAT.format("reporter", response["text"])])
    logger.debug(f"\n{Colors.RED}Reporter - current state messages:\n{pprint.pformat(state['messages'], indent=2, width=100)}{Colors.END}")
    logger.debug(f"\n{Colors.RED}Reporter response:\n{pprint.pformat(response["text"], indent=2, width=100)}{Colors.END}")

    history = state.get("history", [])
    history.append({"agent":"reporter", "message": response["text"]})
    logger.info(f"{Colors.GREEN}===== Reporter completed task ====={Colors.END}")
    return Command(
        update={
            "messages": [get_message_from_string(role="user", string=response["text"], imgs=[])],
            "messages_name": "reporter",
            "history": history,
            "clues": clues
        },
        goto="supervisor"
    )