import os
from datetime import datetime
from langchain_core.prompts import PromptTemplate
from langgraph.prebuilt.chat_agent_executor import AgentState

def get_prompt_template(prompt_name: str) -> str:
    template = open(os.path.join(os.path.dirname(__file__), f"{prompt_name}.md")).read()
    return template

def apply_prompt_template(prompt_name: str, state: AgentState) -> list:

    system_prompts = get_prompt_template(prompt_name)

    if prompt_name in ["planner"]:
        context = {
            "CURRENT_TIME": datetime.now().strftime("%a %b %d %Y %H:%M:%S %z"),
            "ORIGINAL_USER_REQUEST": state["request"],
            #"FOLLOW_UP_QUESTIONS": state.get("follow_up_questions", ""),
            "FULL_PLAN": state.get("full_plan", ""),
            #"USER_FEEDBACK": state.get("user_feedback", "")
        }
    elif prompt_name in ["researcher", "coder", "reporter"]:
        context = {
            "CURRENT_TIME": datetime.now().strftime("%a %b %d %Y %H:%M:%S %z"),
            "USER_REQUEST": state["request"],
            "FULL_PLAN": state.get("full_plan", "")
        }
    elif prompt_name in ["scm_researcher", "scm_data_analyzer"]:
        context = {
            "CURRENT_TIME": datetime.now().strftime("%a %b %d %Y %H:%M:%S %z"),
            "ORIGINAL_USER_REQUEST": state.get("request", ""),
        }
    elif prompt_name in ["scm_correlation_analyzer", "scm_mitigation_planner"]:
        context = {
            "CURRENT_TIME": datetime.now().strftime("%a %b %d %Y %H:%M:%S %z"),
            "ORIGINAL_USER_REQUEST": state.get("request", ""),
            "FULL_PLAN": state.get("full_plan", ""),
            "CLUES": state.get("clues", "")
        }
    else: 
        context = {"CURRENT_TIME": datetime.now().strftime("%a %b %d %Y %H:%M:%S %z")}

    system_prompts = system_prompts.format(**context)
    return system_prompts