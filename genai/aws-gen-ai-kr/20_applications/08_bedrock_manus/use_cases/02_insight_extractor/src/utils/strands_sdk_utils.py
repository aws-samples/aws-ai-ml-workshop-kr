import logging
from strands import Agent, tool
from src.agents.llm import get_llm_by_type
from src.config.agents import AGENT_LLM_MAP, AGENT_PROMPT_CACHE_MAP
from src.prompts.template import apply_prompt_template

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

class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class strands_utils():

    @staticmethod
    def get_agent(**kwargs):

        agent_name, state = kwargs["agent_name"], kwargs["state"]
        tools = kwargs.get("tools", None)
        streaming = kwargs.get("streaming", True)

        if "reasoning" in AGENT_LLM_MAP[agent_name]: enable_reasoning = True
        else: enable_reasoning = False

        prompt_cache, cache_type = AGENT_PROMPT_CACHE_MAP[agent_name]
        if prompt_cache: logger.info(f"{Colors.GREEN}{agent_name.upper()} - Prompt Cache Enabled{Colors.END}")
        else: logger.info(f"{Colors.GREEN}{agent_name.upper()} - Prompt Cache Disabled{Colors.END}")

        system_prompts = apply_prompt_template(agent_name, state)
        llm = get_llm_by_type(AGENT_LLM_MAP[agent_name], cache_type, enable_reasoning)    
        llm.config["streaming"] = streaming

        agent = Agent(
            model=llm,
            system_prompt=system_prompts,
            tools=tools,
            callback_handler=None # async iterator로 대체 하기 때문에 None 설정
        )

        return agent

    @staticmethod
    def parsing_text_from_response(response):

        """
        Usage (async iterator x): 
        agent = Agent()
        response = agent(query)
        response = strands_utils.parsing_text_from_response(response)
        """

        output = {}
        if len(response.message["content"]) == 2: ## reasoning
            output["reasoning"] = response.message["content"][0]["reasoningContent"]["reasoningText"]["text"]
            output["signature"] = response.message["content"][0]["reasoningContent"]["reasoningText"]["signature"]
        
        output["text"] = response.message["content"][-1]["text"]
    
        return output  