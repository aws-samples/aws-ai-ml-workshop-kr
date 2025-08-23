# import os
# from datetime import datetime
# from langchain_core.prompts import PromptTemplate
# from langgraph.prebuilt.chat_agent_executor import AgentState
# #from src.utils.bedrock import bedrock_utils

# def get_prompt_template(prompt_name: str) -> str:
#     template = open(os.path.join(os.path.dirname(__file__), f"{prompt_name}.md")).read()
#     return template

# def apply_prompt_template(prompt_name: str, state: AgentState) -> list:
    
#     system_prompts = get_prompt_template(prompt_name)
#     if prompt_name in ["coder", "reporter"]:
#         context = {
#             "CURRENT_TIME": datetime.now().strftime("%a %b %d %Y %H:%M:%S %z"),
#             "USER_REQUEST": state["request"],
#             "FULL_PLAN": state["full_plan"]
#         }
#     else: context = {"CURRENT_TIME": datetime.now().strftime("%a %b %d %Y %H:%M:%S %z")}
#     system_prompts = system_prompts.format(**context)
#     #system_prompts = bedrock_utils.get_system_prompt(system_prompts=system_prompts, prompt_cache=prompt_cache, cache_type=cache_type) 
#     return system_prompts

import os
from datetime import datetime

def apply_prompt_template(prompt_name: str, prompt_context={}) -> str:
    
    system_prompts = open(os.path.join(os.path.dirname(__file__), f"{prompt_name}.md")).read()
    context = {"CURRENT_TIME": datetime.now().strftime("%a %b %d %Y %H:%M:%S %z")}
    context.update(prompt_context)
    system_prompts = system_prompts.format(**context)
        
    return system_prompts