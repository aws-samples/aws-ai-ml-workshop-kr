import asyncio
import nest_asyncio
from pydantic import BaseModel, Field
from typing import ClassVar, Type
from langchain.tools import BaseTool
from browser_use import AgentHistoryList, Browser, BrowserConfig
from browser_use import Agent as BrowserAgent
from src.agents.llm import browser_llm
from src.tools.decorators import create_logged_tool
from src.config import CHROME_INSTANCE_PATH, BROWSER_HEADLESS

nest_asyncio.apply()

# 브라우저 설정 (Chrome 인스턴스 경로가 있는 경우에만)
expected_browser = None
if CHROME_INSTANCE_PATH:
    expected_browser = Browser(
        config=BrowserConfig(
            chrome_instance_path=CHROME_INSTANCE_PATH,
            headless=False
        )
    )

class BrowserUseInput(BaseModel):
    """BrowserTool 입력 스키마"""
    instruction: str = Field(..., description="브라우저 사용 지시사항")


class BrowserTool(BaseTool):
    name: ClassVar[str] = "browser"
    args_schema: Type[BaseModel] = BrowserUseInput
    description: ClassVar[str] = (
        "웹 브라우저와 상호작용하는 도구입니다. 입력은 'google.com에 접속해서 browser-use 검색' 또는 "
        "'Reddit에 방문해서 AI에 관한 인기 게시물 찾기'와 같은 자연어 설명이어야 합니다."
    )

    def _run(self, instruction: str) -> str:
        """브라우저 작업을 동기적으로 실행합니다."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        print ("BROWSER_HEADLESS", BROWSER_HEADLESS)
        
        async def run_agent():
            agent = BrowserAgent(
                task=instruction,
                llm=browser_llm,
                #browser=expected_browser
            )
            return await agent.run()
        
        try:
            result = loop.run_until_complete(run_agent())
            return (
                str(result)
                if not isinstance(result, AgentHistoryList)
                else result.final_result()
            )
        finally:
            loop.close()
        
    async def _arun(self, instruction: str) -> str:
        """브라우저 작업을 비동기적으로 실행합니다."""
        agent = BrowserAgent(
            task=instruction,
            llm=browser_llm,
            #browser=expected_browser
        )
        
        try:
            result = await agent.run()
            return (
                str(result)
                if not isinstance(result, AgentHistoryList)
                else result.final_result()
            )
        except Exception as e:
            return f"브라우저 작업 실행 오류: {str(e)}"
        

def handle_browser_tool(instruction: str) -> str:
    """
    주어진 지시사항에 따라 브라우저 도구 실행을 처리합니다.
    """
    print("브라우저 도구 실행:", instruction)
    
    # 로깅된 브라우저 도구 인스턴스 생성
    LoggedBrowserTool = create_logged_tool(BrowserTool)
    browser_tool = LoggedBrowserTool()
    
    # 동기적 _run 메서드 사용
    return browser_tool._run(instruction)


# 사용 예시:
if __name__ == "__main__":
    # 동기적 사용 예시
    result = handle_browser_tool("google.com에 접속해서 '장동진' 검색")
    print("결과:", result)
