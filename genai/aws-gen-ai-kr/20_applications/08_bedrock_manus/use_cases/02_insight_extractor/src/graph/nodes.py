import os
import json
import pprint
import logging
import asyncio
import traceback
from typing import Literal
from langgraph.types import Command
from langgraph.graph import END

from src.config import TEAM_MEMBERS


from src.config.agents import AGENT_LLM_MAP, AGENT_PROMPT_CACHE_MAP #이동
from src.prompts.template import apply_prompt_template #이동


from .types import State

from textwrap import dedent
from src.utils.common_utils import get_message_from_string

from strands import Agent, tool #이동


from src.utils.strands_sdk_utils import strands_utils
from src.tools import python_repl_tool, bash_tool
from src.utils.common_utils import ColoredStreamingCallback

llm_module = os.environ.get('LLM_MODULE', 'src.agents.llm')
if llm_module == 'src.agents.llm_st': from src.agents.llm_st import get_llm_by_type
else: from src.agents.llm import get_llm_by_type #이동

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
    
    async def process_streaming_response(agent, message):
        callback_reasoning, callback_answer = ColoredStreamingCallback('blue'), ColoredStreamingCallback('white')
        response = {"text": "","reasoning": "", "signature": "", "tool_use": None, "cycle": 0}
        try:
            agent_stream = agent.stream_async(message)
            async for event in agent_stream:
                if "reasoningText" in event:
                    response["reasoning"] += event["reasoningText"]
                    callback_reasoning.on_llm_new_token(event["reasoningText"])
                elif "reasoning_signature" in event:
                    response["signature"] += event["reasoning_signature"]
                elif "data" in event:
                    response["text"] += event["data"]
                    callback_answer.on_llm_new_token(event["data"])
                elif "current_tool_use" in event and event["current_tool_use"].get("name"):
                    response["tool_use"] = event["current_tool_use"]["name"]
                    if "event_loop_metrics" in event:
                        if response["cycle"] != event["event_loop_metrics"].cycle_count:
                            response["cycle"] = event["event_loop_metrics"].cycle_count
                            callback_answer.on_llm_new_token(f' \n## Calling tool: {event["current_tool_use"]["name"]} - # Cycle: {event["event_loop_metrics"].cycle_count}')
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            #message_placeholder.markdown("Sorry, an error occurred while generating the response.")
            logger.error(traceback.format_exc())  # Detailed error logging
        
        return response  # output 리턴 추가

    agent = strands_utils.get_agent(
        agent_name="coder",
        state=state,
        tools=[python_repl_tool, bash_tool],
        streaming=True
    )

    clues, messages = state.get("clues", ""), state["messages"] 
    message = '\n\n'.join([messages[-1]["content"][-1]["text"], clues])
    response = asyncio.run(process_streaming_response(agent, message))

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
    
    async def process_streaming_response(agent, message):
        callback_reasoning, callback_answer = ColoredStreamingCallback('blue'), ColoredStreamingCallback('white')
        response = {"text": "","reasoning": "", "signature": "", "tool_use": None, "cycle": 0}
        try:
            agent_stream = agent.stream_async(message)
            async for event in agent_stream:
                if "reasoningText" in event:
                    response["reasoning"] += event["reasoningText"]
                    callback_reasoning.on_llm_new_token(event["reasoningText"])
                elif "reasoning_signature" in event:
                    response["signature"] += event["reasoning_signature"]
                elif "data" in event:
                    response["text"] += event["data"]
                    callback_answer.on_llm_new_token(event["data"])
                elif "current_tool_use" in event and event["current_tool_use"].get("name"):
                    response["tool_use"] = event["current_tool_use"]["name"]
                    if "event_loop_metrics" in event:
                        if response["cycle"] != event["event_loop_metrics"].cycle_count:
                            response["cycle"] = event["event_loop_metrics"].cycle_count
                            callback_answer.on_llm_new_token(f' \n## Calling tool: {event["current_tool_use"]["name"]} - # Cycle: {event["event_loop_metrics"].cycle_count}')
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            #message_placeholder.markdown("Sorry, an error occurred while generating the response.")
            logger.error(traceback.format_exc())  # Detailed error logging
        
        return response  # output 리턴 추가

    agent = strands_utils.get_agent(
        agent_name="planner",
        state=state,
        streaming=True
    )

    message = state["request_prompt"]    
    response = asyncio.run(process_streaming_response(agent, message))

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
    
    async def process_streaming_response(agent, message):
        callback_reasoning, callback_answer = ColoredStreamingCallback('blue'), ColoredStreamingCallback('white')
        response = {"text": "","reasoning": "", "signature": "", "tool_use": None, "cycle": 0}
        try:
            agent_stream = agent.stream_async(message)
            async for event in agent_stream:
                if "reasoningText" in event:
                    response["reasoning"] += event["reasoningText"]
                    callback_reasoning.on_llm_new_token(event["reasoningText"])
                elif "reasoning_signature" in event:
                    response["signature"] += event["reasoning_signature"]
                elif "data" in event:
                    response["text"] += event["data"]
                    callback_answer.on_llm_new_token(event["data"])
                elif "current_tool_use" in event and event["current_tool_use"].get("name"):
                    response["tool_use"] = event["current_tool_use"]["name"]
                    if "event_loop_metrics" in event:
                        if response["cycle"] != event["event_loop_metrics"].cycle_count:
                            response["cycle"] = event["event_loop_metrics"].cycle_count
                            callback_answer.on_llm_new_token(f' \n## Calling tool: {event["current_tool_use"]["name"]} - # Cycle: {event["event_loop_metrics"].cycle_count}')
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            #message_placeholder.markdown("Sorry, an error occurred while generating the response.")
            logger.error(traceback.format_exc())  # Detailed error logging
        
        return response  # output 리턴 추가

    agent = strands_utils.get_agent(
        agent_name="coordinator",
        state=state,
        streaming=True
    )

    message = state["request_prompt"]    
    response = asyncio.run(process_streaming_response(agent, message))
    
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