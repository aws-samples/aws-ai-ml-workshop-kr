import os
import json
import pprint
import logging
from copy import deepcopy
from typing import Literal
from langchain_core.messages import HumanMessage
from langgraph.types import Command, interrupt
from langgraph.graph import END
from src.agents.agents import create_react_agent
from src.config import TEAM_MEMBERS
from src.config.agents import AGENT_LLM_MAP, AGENT_PROMPT_CACHE_MAP
from src.prompts.template import apply_prompt_template
from src.tools.search import tavily_tool
from .types import State, Router

from textwrap import dedent
from src.utils.common_utils import get_message_from_string

llm_module = os.environ.get('LLM_MODULE', 'src.agents.llm')
if llm_module == 'src.agents.llm_st': from src.agents.llm_st import get_llm_by_type, llm_call
else: from src.agents.llm import get_llm_by_type, llm_call

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
    research_agent = create_react_agent(agent_name="clarifier")
    result = research_agent.invoke(state=state)
    follow_up_questions = json.loads(result["content"][-1]["text"])

    clues = state.get("clues", "")
    clues = '\n\n'.join([clues, CLUES_FORMAT.format("clarifier", result["content"][-1]["text"])])
    logger.info("Clarification agent completed task")
    logger.debug(f"Clarification agent response: {result["content"][-1]["text"]}")

    history = state.get("history", [])
    history.append({"agent":"clarifier", "message": result["content"][-1]["text"]})
    return Command(
        update={
            "messages": [get_message_from_string(role="user", string=RESPONSE_FORMAT.format("clarifier", result["content"][-1]["text"]), imgs=[])],
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
    research_agent = create_react_agent(agent_name="researcher")
    result = research_agent.invoke(state=state)
    
    clues = state.get("clues", "")
    clues = '\n\n'.join([clues, CLUES_FORMAT.format("researcher", result["content"][-1]["text"])])
    logger.info("Research agent completed task")
    logger.debug(f"Research agent response: {result["content"][-1]["text"]}")

    history = state.get("history", [])
    history.append({"agent":"researcher", "message": result["content"][-1]["text"]})
    return Command(
        update={
            "messages": [get_message_from_string(role="user", string=RESPONSE_FORMAT.format("researcher", result["content"][-1]["text"]), imgs=[])],
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

def supervisor_node(state: State) -> Command[Literal[*TEAM_MEMBERS, "__end__"]]:
    """Supervisor node that decides which agent should act next."""
    logger.info(f"{Colors.GREEN}===== Supervisor evaluating next action ====={Colors.END}")
    
    prompt_cache, cache_type = AGENT_PROMPT_CACHE_MAP["supervisor"]
    if prompt_cache: logger.debug(f"{Colors.GREEN}Supervisor - Prompt Cache Enabled{Colors.END}")
    else: logger.debug(f"{Colors.GREEN}Supervisor - Prompt Cache Disabled{Colors.END}")
    system_prompts, messages = apply_prompt_template("supervisor", state, prompt_cache=prompt_cache, cache_type=cache_type)    
    llm = get_llm_by_type(AGENT_LLM_MAP["supervisor"])
    llm.stream = True
    llm_caller = llm_call(llm=llm, verbose=False, tracking=False)
    
    clues, full_plan = state.get("clues", ""), state.get("full_plan", "")  
    messages[-1]["content"][-1]["text"] = '\n\n'.join([messages[-1]["content"][-1]["text"], FULL_PLAN_FORMAT.format(full_plan), clues])
    
    response, ai_message = llm_caller.invoke(
        agent_name="supervisor",
        messages=messages,
        system_prompts=system_prompts,
        enable_reasoning=False,
        reasoning_budget_tokens=8192
    )
    full_response = response["text"]

    if full_response.startswith("```json"): full_response = full_response.removeprefix("```json")
    if full_response.endswith("```"): full_response = full_response.removesuffix("```")
    
    full_response = json.loads(full_response)   
    goto = full_response["next"]

    logger.debug(f"\n{Colors.RED}Supervisor - current state messages:\n{pprint.pformat(state['messages'], indent=2, width=100)}{Colors.END}")
    logger.debug(f"\n{Colors.RED}Supervisor response:{full_response}{Colors.END}")

    if goto == "FINISH":
        goto = "__end__"
        #logger.info("Workflow completed")
        logger.info(f"\n{Colors.GREEN}===== Workflow completed ====={Colors.END}")
    else:
        #logger.info(f"Supervisor delegating to: {goto}")
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
    logger.info(f"{Colors.BLUE}===== Planner - Deep thinking mode: {state.get("deep_thinking_mode")} ====={Colors.END}")
    logger.info(f"{Colors.BLUE}===== Planner - Search before planning: {state.get("search_before_planning")} ====={Colors.END}")
    logger.debug(f"\n{Colors.RED}Planner - current state messages:\n{pprint.pformat(state['messages'], indent=2, width=100)}{Colors.END}")
    
    prompt_cache, cache_type = AGENT_PROMPT_CACHE_MAP["planner"]
    if prompt_cache: logger.debug(f"{Colors.GREEN}Planner - Prompt Cache Enabled{Colors.END}")
    else: logger.debug(f"{Colors.GREEN}Planner - Prompt Cache Disabled{Colors.END}")
    system_prompts, messages = apply_prompt_template("planner", state, prompt_cache=prompt_cache, cache_type=cache_type)
    # whether to enable deep thinking mode

    full_plan = state.get("full_plan", "")
    messages[-1]["content"][-1]["text"] = '\n\n'.join([messages[-1]["content"][-1]["text"], FULL_PLAN_FORMAT.format(full_plan)])
   
    llm = get_llm_by_type(AGENT_LLM_MAP["planner"])    
    llm.stream = True
    llm_caller = llm_call(llm=llm, verbose=False, tracking=False)
        
    if state.get("deep_thinking_mode"): llm = get_llm_by_type("reasoning")
    if state.get("search_before_planning"):
        searched_content = tavily_tool.invoke({"query": state["request"]})
        messages = deepcopy(messages)
        messages[-1]["content"][-1]["text"] += f"\n\n# Relative Search Results\n\n{json.dumps([{'title': elem['title'], 'content': elem['content']} for elem in searched_content], ensure_ascii=False)}"
    
    if AGENT_LLM_MAP["planner"] in ["reasoning"]: enable_reasoning = True
    else: enable_reasoning = False

    response, ai_message = llm_caller.invoke(
        agent_name="planner",
        messages=messages,
        system_prompts=system_prompts,
        enable_reasoning=enable_reasoning,
        reasoning_budget_tokens=8192
    )
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
    
    prompt_cache, cache_type = AGENT_PROMPT_CACHE_MAP["coordinator"]
    if prompt_cache: logger.debug(f"{Colors.GREEN}Coordinator - Prompt Cache Enabled{Colors.END}")
    else: logger.debug(f"{Colors.GREEN}Coordinator - Prompt Cache Disabled{Colors.END}")
    system_prompts, messages = apply_prompt_template("coordinator", state, prompt_cache=prompt_cache, cache_type=cache_type)
    llm = get_llm_by_type(AGENT_LLM_MAP["coordinator"])    
    llm.stream = True
    llm_caller = llm_call(llm=llm, verbose=False, tracking=False)
    if AGENT_LLM_MAP["coordinator"] in ["reasoning"]: enable_reasoning = True
    clarifier
    
    response, ai_message = llm_caller.invoke(
        agent_name="coordinator",
        messages=messages,
        system_prompts=system_prompts,
        enable_reasoning=False,
        reasoning_budget_tokens=8192
    )
    
    logger.debug(f"\n{Colors.RED}Current state messages:\n{pprint.pformat(state['messages'], indent=2, width=100)}{Colors.END}")
    logger.debug(f"\n{Colors.RED}Coordinator response:\n{pprint.pformat(response, indent=2, width=100)}{Colors.END}")

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
            "messages": [get_message_from_string(role="user", string=RESPONSE_FORMAT.format("reporter", result["content"][-1]["text"]), imgs=[])],
            "messages_name": "reporter",
            "history": history,
            "clues": clues
        },
        goto="supervisor"
    )