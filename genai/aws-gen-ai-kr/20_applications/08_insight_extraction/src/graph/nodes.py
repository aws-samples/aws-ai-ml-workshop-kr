import logging
import json
from copy import deepcopy
from typing import Literal
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langgraph.graph import END

#from src.agents import browser_agent # research_agent, coder_agent
from src.agents.agents import create_react_agent
from src.agents.llm import get_llm_by_type, llm_call
from src.config import TEAM_MEMBERS
from src.config.agents import AGENT_LLM_MAP
from src.prompts.template import apply_prompt_template
from src.tools.search import tavily_tool
from .types import State, Router

from textwrap import dedent
from src.utils.common_utils import get_message_from_string

logger = logging.getLogger(__name__)

RESPONSE_FORMAT = "Response from {}:\n\n<response>\n{}\n</response>\n\n*Please execute the next step.*"
FULL_PLAN_FORMAT = "Here is full plan :\n\n<full_plan>\n{}\n</full_plan>\n\n*Please consider this to select the next step.*"
CLUES_FORMAT = "Here is clues form {}:\n\n<clues>\n{}\n</clues>\n\n"

def task_tracker_node(state: State) -> Command[Literal["supervisor"]]:
    """Node for the task_tracker agent that tracks tasks."""
    logger.info("Task tracker agent starting task")
    task_tracker_agent = create_react_agent(agent_name="task_tracker")
    result = task_tracker_agent.invoke(state=state)

    logger.info("Task tracker agent completed task")
    logger.debug(f"Task tracker agent response: {result["content"][-1]["text"]}")

    history = state.get("history", [])
    history.append({"agent":"task_tracker", "message": result["content"][-1]["text"]})
    return Command(
        update={
            "messages": [get_message_from_string(role="user", string=RESPONSE_FORMAT.format("task_tracker", result["content"][-1]["text"]), imgs=[])],
            "messages_name": "task_tracker",
            "history": history
        },
        goto="supervisor",
    )

def research_node(state: State) -> Command[Literal["supervisor"]]:
    """Node for the researcher agent that performs research tasks."""
    logger.info("Research agent starting task")
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
    logger.info("Code agent starting task")
    coder_agent = create_react_agent(agent_name="coder")
    result = coder_agent.invoke(state=state)

    clues = state.get("clues", "")
    clues = '\n\n'.join([clues, CLUES_FORMAT.format("coder", result["content"][-1]["text"])])
    logger.info("Code agent completed task")
    logger.debug(f"Code agent response: {result["content"][-1]["text"]}")

    history = state.get("history", [])
    history.append({"agent":"coder", "message": result["content"][-1]["text"]})
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
    logger.info("Supervisor evaluating next action")
    
    system_prompts, messages = apply_prompt_template("supervisor", state)    
    llm = get_llm_by_type(AGENT_LLM_MAP["supervisor"])    
    llm.stream = True
    llm_caller = llm_call(llm=llm, verbose=False, tracking=False)
    
    clues, full_plan = state.get("clues", ""), state.get("full_plan", "")       
    messages[-1]["content"][-1]["text"] = '\n\n'.join([messages[-1]["content"][-1]["text"], FULL_PLAN_FORMAT.format(full_plan), clues])
    
    response, ai_message = llm_caller.invoke(
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
    logger.debug(f"Current state messages: {state['messages']}")
    logger.debug(f"Supervisor response: {full_response}")

    if goto == "FINISH":
        goto = "__end__"
        logger.info("Workflow completed")
    else:
        logger.info(f"Supervisor delegating to: {goto}")

    history = state.get("history", [])
    history.append({"agent":"supervisor", "message": full_response})
    return Command(
        goto=goto,
        update={
            "next": goto,
            "history": history
        }
    )

def planner_node(state: State) -> Command[Literal["supervisor", "__end__"]]:
    """Planner node that generate the full plan."""
    logger.info("Planner generating full plan")
    system_prompts, messages = apply_prompt_template("planner", state)
    # whether to enable deep thinking mode
       
    #llm = get_llm_by_type("basic")
    llm = get_llm_by_type(AGENT_LLM_MAP["planner"])    
    llm.stream = True
    llm_caller = llm_call(llm=llm, verbose=False, tracking=False)
        
    if state.get("deep_thinking_mode"): llm = get_llm_by_type("reasoning")
        
    if state.get("search_before_planning"):
        searched_content = tavily_tool.invoke({"query": state["request"]})
        messages = deepcopy(messages)
        messages[-1]["content"][-1]["text"] += f"\n\n# Relative Search Results\n\n{json.dumps([{'titile': elem['title'], 'content': elem['content']} for elem in searched_content], ensure_ascii=False)}"
        
    response, ai_message = llm_caller.invoke(
        messages=messages,
        system_prompts=system_prompts,
        enable_reasoning=True,
        reasoning_budget_tokens=8192
    )
    full_response = response["text"]
            
    logger.debug(f"Current state messages: {state['messages']}")
    logger.debug(f"Planner response: {full_response}")

    if full_response.startswith("```json"):
        full_response = full_response.removeprefix("```json")

    if full_response.endswith("```"):
        full_response = full_response.removesuffix("```")

    goto = "supervisor"
    try:
        json.loads(full_response)
    except json.JSONDecodeError:
        logger.warning("Planner response is not a valid JSON")
        goto = "__end__"
        
    history = state.get("history", [])
    history.append({"agent":"planner", "message": full_response})
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
    logger.info("Coordinator talking.")
    
    system_prompts, messages = apply_prompt_template("coordinator", state)
    
    llm = get_llm_by_type(AGENT_LLM_MAP["coordinator"])    
    llm.stream = True
    llm_caller = llm_call(llm=llm, verbose=False, tracking=False)
    
    response, ai_message = llm_caller.invoke(
        messages=messages,
        system_prompts=system_prompts,
        #tool_config=tool_config,
        enable_reasoning=False,
        reasoning_budget_tokens=8192
    )
    
    logger.debug(f"Current state messages: {state['messages']}")
    logger.debug(f"reporter response: {response}")

    goto = "__end__"
    if "handoff_to_planner" in response["text"]: goto = "planner"

    history = state.get("history", [])
    history.append({"agent":"coordinator", "message": response["text"]})
    return Command(
        update={"history": history},
        goto=goto,
    )

def reporter_node(state: State) -> Command[Literal["supervisor"]]:
    """Reporter node that write a final report."""
    logger.info("Reporter write final report")

    #system_prompts, messages = apply_prompt_template("reporter", state)
    #llm = get_llm_by_type(AGENT_LLM_MAP["reporter"])
    #llm.stream = True
    #llm_caller = llm_call(llm=llm, verbose=False, tracking=False)

    # clues = state.get("clues", "")
    # messages[-1]["content"][-1]["text"] = '\n\n'.join([messages[-1]["content"][-1]["text"], clues])

    # response, ai_message = llm_caller.invoke(
    #     messages=messages,
    #     system_prompts=system_prompts,
    #     #tool_config=tool_config,
    #     enable_reasoning=False,
    #     reasoning_budget_tokens=8192
    # )

    #full_response = response["text"]

    clues = state.get("clues", "")
    _, messages = apply_prompt_template("reporter", state)
    messages[-1]["content"][-1]["text"] = '\n\n'.join([messages[-1]["content"][-1]["text"], clues])
    state["messages"] = messages

    print ('state["messages"]', state["messages"])

    reporter_agent = create_react_agent(agent_name="reporter")
    result = reporter_agent.invoke(state=state)

    full_response = result["content"][-1]["text"]

    logger.debug(f"Current state messages: {state['messages']}")
    logger.debug(f"reporter response: {full_response}")

    history = state.get("history", [])
    history.append({"agent":"reporter", "message": full_response})
    return Command(
        update={
            "messages": [get_message_from_string(role="user", string=full_response, imgs=[])],
            "messages_name": "reporter",
            "history": history
        },
        goto="supervisor"
    )