#from langgraph.prebuilt import create_react_agent

#from src.prompts import apply_prompt_template_langchain
#from src.tools import browser_tool

from .llm import get_llm_by_type
from src.config.agents import AGENT_LLM_MAP
from src.prompts.template import apply_prompt_template
from src.agents.llm import get_llm_by_type, llm_call
from src.tools.research_tools import research_tool_config, process_search_tool
from src.tools.coder_tools import coder_tool_config, process_coder_tool
from src.tools.browser_tools import browser_tool_config, process_browser_tool
from src.tools.task_tracker_tool import task_tracker_tool_config, process_task_tracker_tool
from src.tools.reporter_tools import reporter_tool_config, process_reporter_tool

class create_react_agent():

    def __init__(self, **kwargs):

        self.agent_name = kwargs["agent_name"]
        self.llm = get_llm_by_type(AGENT_LLM_MAP[self.agent_name])
        self.llm.stream = True
        self.llm_caller = llm_call(llm=self.llm, verbose=False, tracking=False)
        self.enable_reasoning = False
        
        if self.agent_name == "researcher": self.tool_config = research_tool_config
        elif self.agent_name == "coder": self.tool_config = coder_tool_config
        elif self.agent_name == "browser": self.tool_config = browser_tool_config
        elif self.agent_name == "task_tracker": self.tool_config = task_tracker_tool_config
        elif self.agent_name == "reporter":
            self.tool_config = reporter_tool_config
            self.enable_reasoning = False
        
        # 반복 대화 처리를 위한 설정
        self.MAX_TURNS = 30  # 무한 루프 방지용 최대 턴 수
        self.turn = 0
        self.final_response = False
        
    def invoke(self, **kwargs):

        state = kwargs.get("state", None)
        system_prompts, messages = apply_prompt_template(self.agent_name, state)
        
        # 도구 사용이 종료될 때까지 반복
        while not self.final_response and self.turn < self.MAX_TURNS:
            self.turn += 1
            print(f"\n--- 대화 턴 {self.turn} ---")

            response, ai_message = self.llm_caller.invoke(
                messages=messages,
                system_prompts=system_prompts,
                tool_config=self.tool_config,
                enable_reasoning=self.enable_reasoning,
                reasoning_budget_tokens=8192
            )
            messages.append(ai_message)    

            print ("======")
            #print (messages)
            print(f"응답 상태: {response['stop_reason']}")

            # 도구 사용 요청 확인
            if response["stop_reason"] == "tool_use":
                print("모델이 도구 사용을 요청했습니다.")
                tool_requests_found = False

                # 응답에서 모든 도구 사용 요청 처리
                for content in ai_message['content']:
                    if 'toolUse' in content:
                        tool = content['toolUse']
                        tool_requests_found = True

                        print(f"요청된 도구:\n {tool['name']}")
                        print(f"입력 데이터:\n {tool['input']}")

                        if self.agent_name == "researcher": tool_result_message = process_search_tool(tool)
                        elif self.agent_name == "coder": tool_result_message = process_coder_tool(tool)
                        elif self.agent_name == "browser": tool_result_message = process_browser_tool(tool)
                        elif self.agent_name == "task_tracker": tool_result_message = process_task_tracker_tool(tool)
                        elif self.agent_name == "reporter": tool_result_message = process_reporter_tool(tool)

                        messages.append(tool_result_message)
                        print(f"도구 실행 결과를 대화에 추가했습니다.")

                # 도구 요청이 없으면 루프 종료
                if not tool_requests_found:
                    print("도구 요청을 찾을 수 없습니다.")
                    self.final_response = True
            else:
                # 도구 사용이 요청되지 않았으면 최종 응답으로 간주
                self.final_response = True
                print("최종 응답을 받았습니다.")

        print("\n=== 대화 완료 ===")
        print("최종 응답:\n", response)
        print("메시지:\n", ai_message)
        
        return ai_message

        
research_agent = None
coder_agent = None
browser_agent = None
