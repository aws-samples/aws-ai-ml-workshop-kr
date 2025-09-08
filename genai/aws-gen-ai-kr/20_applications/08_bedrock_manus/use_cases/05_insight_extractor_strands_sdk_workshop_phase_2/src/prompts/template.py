import os
from datetime import datetime

def apply_prompt_template(prompt_name: str, prompt_context={}) -> str:
    
    system_prompts = open(os.path.join(os.path.dirname(__file__), f"{prompt_name}.md")).read() ## Template.py가 있는 dir이 기준
    context = {"CURRENT_TIME": datetime.now().strftime("%a %b %d %Y %H:%M:%S %z")}
    context.update(prompt_context)
    system_prompts = system_prompts.format(**context)
        
    return system_prompts