import os
import base64
import json
import boto3
import sys
import textwrap
from io import StringIO
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


from langchain_aws import ChatBedrock

class BedrockLangChain:

    def __init__(self, bedrock_runtime):
        self.bedrock_runtime = bedrock_runtime

    def invoke_rewrite_langchain(self, model_id, model_kwargs, system_prompt, user_prompt, coordination_review, verbose):

        model = ChatBedrock(
            client=self.bedrock_runtime,
            model_id= model_id,
            model_kwargs=model_kwargs,
        )


        messages = [
            ("system", system_prompt),
            ("human", user_prompt)
        ]

        prompt = ChatPromptTemplate.from_messages(messages)
        if verbose:
            print("messages: \n", messages)        
            print("prompt: \n")
            self.print_ww(prompt)

        chain = prompt | model | StrOutputParser()

        print("## Created Prompt:\n")
        response = chain.invoke(
            {
                "coordination_review": coordination_review
            }
        )

        return response



    def invoke_creating_criteria_langchain(self, model_id, model_kwargs, system_prompt, user_prompt, guide, verbose):

        model = ChatBedrock(
            client=self.bedrock_runtime,
            model_id= model_id,
            model_kwargs=model_kwargs,
        )


        messages = [
            ("system", system_prompt),
            ("human", user_prompt)
        ]

        prompt = ChatPromptTemplate.from_messages(messages)
        if verbose:
            print("messages: \n", messages)        
            print("prompt: \n")
            self.print_ww(prompt)

        chain = prompt | model | StrOutputParser()

        print("## Created Prompt:\n")

        for chunk in chain.stream(
            {
                "guide": guide
            }
        ):
            print(chunk, end="", flush=True)


    def invoke_evaluating_fashion_review_langchain(self, model_id, model_kwargs, system_prompt, user_prompt, human_message, AI_message, verbose):

        model = ChatBedrock(
            client=self.bedrock_runtime,
            model_id= model_id,
            model_kwargs=model_kwargs,
        )


        
        messages = [
            ("system", system_prompt),
            ("human", user_prompt)
        ]

        prompt = ChatPromptTemplate.from_messages(messages)
        if verbose:
            print("messages: \n", messages)        
            print("prompt: \n")
            self.print_ww(prompt)

        chain = prompt | model | StrOutputParser()


        for chunk in chain.stream(
            {
                "human_text": human_message,
                "AI_text": AI_message,                
            }
        ):
            print(chunk, end="", flush=True)


    def set_text_langchain_body(self, prompt):
        text_only_body = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
            }
        return text_only_body
    def print_ww(self, *args, width: int = 100, **kwargs):
        """Like print(), but wraps output to `width` characters (default 100)"""
        buffer = StringIO()
        try:
            _stdout = sys.stdout
            sys.stdout = buffer
            print(*args, **kwargs)
            output = buffer.getvalue()
        finally:
            sys.stdout = _stdout
        for line in output.splitlines():
            print("\n".join(textwrap.wrap(line, width=width)))




# from langchain.callbacks import StreamlitCallbackHandler
# model_id="anthropic.claude-3-sonnet-20240229-v1:0", # Claude 3 Sonnet 모델 선택
# # 텍스트 생성 LLM 가져오기, streaming_callback을 인자로 받아옴
# def get_llm(boto3_bedrock, model_id):
#     llm = BedrockChat(
#     model_id= model_id,
#     client=boto3_bedrock,
#     model_kwargs={
#         "max_tokens": 1024,
#         "stop_sequences": ["\n\nHuman"],
#     }
#     )
#     return llm
# llm = get_llm(boto3_bedrock=client, model_id = model_id)
# response_text = llm.invoke(prompt) #프롬프트에 응답 반환
# print(response_text.content)
