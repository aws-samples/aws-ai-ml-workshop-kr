from textwrap import dedent
from utils.bedrock import bedrock_utils, bedrock_chain


class llm_call():

    def __init__(self, **kwargs):

        self.llm = kwargs["llm"]
        self.verbose = kwargs.get("verbose", False)
        self.chain = bedrock_chain(bedrock_utils.converse_api) | bedrock_chain(bedrock_utils.outputparser)

    def _message_format(self, role, message):

        if role == "user":
             message_format = {
                "role": "user",
                "content": [{"text": dedent(message)}]
            }
        elif role == "assistant":
            
            message_format = {
                "role": "assistant",
                'content': [{'text': message}]
            }

        return message_format
            
    def invoke(self, **kwargs):

        system_prompts = kwargs.get("system_prompts", None)
        messages = kwargs["messages"]
    
        response = self.chain( ## pipeline의 제일 처음 func의 argument를 입력으로 한다. 여기서는 converse_api의 arg를 쓴다.
            llm=self.llm,
            system_prompts=system_prompts,
            messages=messages,
            verbose=self.verbose
        )
        #print (messages)
        ai_message = self._message_format(role="assistant", message=response["text"])
        messages.append(ai_message)

        return response, messages

# class llm_execution():
    
#     def __init__(self, **kwargs):

#         self.llm = kwargs["llm"]
#         self.system_prompts = kwargs["system_prompts"]
#         self.user_prompts = kwargs["user_prompts"]
#         self.messages = []
#         self.multi_turn = kwargs.get("multi_turn", False)
#         self.verbose = kwargs.get("verbose", False)
#         self.tool_config = kwargs.get("tool_config", False)

#         self.request_prompt = dedent(
#             '''
#             메시지 작성 시 요청사항 입니다.:
#             <request>{request}</request>
#             '''
#         )

#     def _excution(self, **kwargs):

#         get_analysis_point_pipeline = bedrock_chain(bedrock_utils.converse_api) | bedrock_chain(bedrock_utils.outputparser)

#         context = kwargs["context"]
#         query = self.user_prompts.format(**context)

#         print (query)

#         # if "request" not in context:
#         #     query = self.user_prompts.format(**context)
#         # elif len(context) == 1 and "request" in context:
#         #     query = self.request_prompt.format(**context)
#         # else:
#         #     prompt = self.user_prompts + self.request_prompt
#         #     query = prompt.format(**context)
        
#         user_message = {
#             "role": "user",
#             "content": [{"text": dedent(query)}]
#         }
#         self.messages.append(user_message)
        
#         response = get_analysis_point_pipeline( ## pipeline의 제일 처음 func의 argument를 입력으로 한다. 여기서는 converse_api의 arg를 쓴다.
#             llm=self.llm,
#             system_prompts=self.system_prompts,
#             messages=self.messages,
#             verbose=self.verbose
#         )
        
#         if self.verbose:
#             print (f'Messages: {self.messages}')

#         if self.multi_turn:
#             v
#             self.messages.append(ai_message)
#         else:
#             self.messages = []
            
#         return response
            
            
  

#     def get_messages(self, ):

#         return self.messages
        
        
#     def invoke(self, **kwargs):
        
#         response = self._excution(**kwargs)
        
#         #_ = self._parse_output(response=response)

#         return response

# # class ask_anaysis():
    
# #     def __init__(self, **kwargs):

# #         self.llm = kwargs["llm"]
# #         self.system_prompts = kwargs["system_prompts"]
# #         self.user_prompts = kwargs["user_prompts"]
# #         self.messages = []
# #         self.multi_turn = kwargs.get("multi_turn", False)
# #         self.verbose = kwargs.get("verbose", False)
# #         self.tool_config = kwargs.get("tool_config", False)

# #         self.request_prompt = dedent(
# #             '''
# #             메시지 작성 시 요청사항 입니다.:
# #             <request>{request}</request>
# #             '''
# #         )

# #     def _ask_anaysis(self, **kwargs):

# #         get_analysis_point_pipeline = bedrock_chain(bedrock_utils.converse_api) | bedrock_chain(bedrock_utils.outputparser)

# #         context = kwargs["context"]
# #         query = self.user_prompts.format(**context)

# #         print (query)

# #         # if "request" not in context:
# #         #     query = self.user_prompts.format(**context)
# #         # elif len(context) == 1 and "request" in context:
# #         #     query = self.request_prompt.format(**context)
# #         # else:
# #         #     prompt = self.user_prompts + self.request_prompt
# #         #     query = prompt.format(**context)
        
# #         user_message = {
# #             "role": "user",
# #             "content": [{"text": dedent(query)}]
# #         }
# #         self.messages.append(user_message)
        
# #         response = get_analysis_point_pipeline( ## pipeline의 제일 처음 func의 argument를 입력으로 한다. 여기서는 converse_api의 arg를 쓴다.
# #             llm=self.llm,
# #             system_prompts=self.system_prompts,
# #             messages=self.messages,
# #             verbose=self.verbose
# #         )
        
# #         if self.verbose:
# #             print (f'Messages: {self.messages}')

# #         if self.multi_turn:
# #             if response["toolUse"] == None:
# #                 ai_message = {
# #                     "role": "assistant",
# #                     'content': [{'text': response["text"]}]
# #                 }
# #             else:
# #                 ai_message = {
# #                     "role": "assistant",
# #                     'content': [{'toolUse': response["toolUse"]}]
# #                 }
# #             self.messages.append(ai_message)
# #         else:
# #             self.messages = []
            
# #         return response
            
            
  

# #     def get_messages(self, ):

# #         return self.messages
        
        
# #     def invoke(self, **kwargs):
        
# #         response = self._ask_anaysis(**kwargs)
        
# #         #_ = self._parse_output(response=response)

# #         return response