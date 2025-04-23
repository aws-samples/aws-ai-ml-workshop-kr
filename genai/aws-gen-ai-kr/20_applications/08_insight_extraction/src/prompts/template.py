import os
from datetime import datetime
from langchain_core.prompts import PromptTemplate
from langgraph.prebuilt.chat_agent_executor import AgentState

from src.utils.bedrock import bedrock_utils

def get_prompt_template(prompt_name: str) -> str:
    template = open(os.path.join(os.path.dirname(__file__), f"{prompt_name}.md")).read()
    return template

def apply_prompt_template(prompt_name: str, state: AgentState) -> list:
    
    system_prompts = get_prompt_template(prompt_name)   
    context = {"CURRENT_TIME": datetime.now().strftime("%a %b %d %Y %H:%M:%S %z")}
    system_prompts = system_prompts.format(**context)    
    system_prompts = bedrock_utils.get_system_prompt(system_prompts=system_prompts)
        
    return system_prompts, state["messages"]
    
# def apply_prompt_template_langchain(prompt_name: str, state: AgentState) -> list:
#     system_prompt = PromptTemplate(
#         input_variables=["CURRENT_TIME"],
#         template=get_prompt_template(prompt_name),
#     ).format(CURRENT_TIME=datetime.now().strftime("%a %b %d %Y %H:%M:%S %z"), **state)
#     return [{"role": "system", "content": system_prompt}] + state["messages"]
