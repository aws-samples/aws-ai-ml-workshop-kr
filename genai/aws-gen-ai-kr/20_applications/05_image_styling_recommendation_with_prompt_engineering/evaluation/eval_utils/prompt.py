
class FashionPrompt():
    def __init__(self):
        # self.system_prompt = system_prompt
        pass
    pass

    def get_rewrite_system_prompt(self):
        '''
        주어진 문장을 Re-Write 하는 시스템 프롬프트를 제공 함.
        '''
        
        system_prompt = '''The task is to rewrite a given sentence in a different way while preserving its original meaning.\
Your role is to take a sentence provided by the user and rephrase it using different words or sentence structures, \
without altering the core meaning or message conveyed in the original sentence.

Instructions:
1. Read the sentence carefully and ensure you understand its intended meaning.
2. Identify the key components of the sentence, such as the subject, verb, object, and any modifiers or additional information.
3. Think of alternative ways to express the same idea using different vocabulary, sentence structures, or phrasing.
4. Ensure that your rewritten sentence maintains the same essential meaning as the original, without introducing any new information or altering the original intent.
5. Pay attention to grammar, punctuation, and overall coherence to ensure your rewritten sentence is well-formed and easy to understand.
6. If the original sentence contains idioms, metaphors, or cultural references, try to find equivalent expressions or explanations in your rewritten version.
7. Avoid oversimplifying or overly complicating the sentence; aim for a natural and clear rephrasing that maintains the original tone and complexity.

Remember, the goal is to provide a fresh perspective on the sentence while preserving its core meaning and ensuring clarity and coherence in your rewritten version.
'''

        return system_prompt

    def get_rewrite_user_prompt(self):
        '''
        주어진 문장을 Re-Write 하는 유저 프롬프트를 제공 함.
        '''
        
        user_prompt = '''Given <coordination_review> based on the guide on system prompt         
Please write in Korean. Output in JSON format following the <output_example> format, excluding <output_example>        

<coordination_review>{coordination_review}</coordination_review>
<output_example>
"original_coordination_review" : 
"rewrite_original_coordination_review" : 
</output_example>
'''

        return user_prompt


    def get_create_criteria_system_prompt(self):
        '''
        주어진 문장을 Re-Write 하는 유저 프롬프트를 제공 함.
        '''        
        system_prompt = '''You are a prompt engineering expert.'''

        return system_prompt

    def get_create_criteria_user_prompt(self):
        
        user_prompt = '''먼저 당신의 역할과 작업을 XML Tag 없이 기술하세요, \
이후에 아래의 <guide> 에 맟주어서 프롬프트를 영어로 작성해주세요. 
<guide>{guide}</guide>'''

        return user_prompt

    def get_fashion_evaluation_system_prompt(self):
        '''
        의상 코디에 대한 관련성 여부를 평가 하기 위한 시스템 프롬프트를 제공
        '''

        
        system_prompt = '''
You will be provided with two opinions: one from a fashion expert regarding clothing choices, and \
another from an AI system offering recommendations on clothing choices. \
Your task is to evaluate the relevance and coherence between these two opinions \
by assigning a score from 1 to 5, where 1 indicates low relevance and 5 indicates high relevance.\ 
You will need to define the criteria for scoring in the <criteria></criteria> section, and \
outline the steps for evaluating the two opinions in the <steps></steps> section.

<criteria>
1 - The two opinions are completely unrelated and contradict each other.
2 - The opinions share some minor similarities, but the overall themes and recommendations are largely different.
3 - The opinions have moderate overlap in their themes and recommendations, but there are still notable differences.
4 - The opinions are mostly aligned, with only minor differences in their specific recommendations or perspectives.
5 - The two opinions are highly coherent, complementary, and provide consistent recommendations or perspectives on clothing choices.
</criteria>

<steps>
1. Read and understand the opinion provided by the fashion expert.
2. Read and understand the opinion provided by the AI system.
3. Identify the main themes, recommendations, and perspectives presented in each opinion.
4. Compare the two opinions and assess the degree of alignment or contradiction between them.
5. Based on the criteria defined above, assign a score from 1 to 5 to reflect the relevance and coherence between the two opinions.
6. Provide a brief explanation justifying the assigned score.
</steps>
'''
        return system_prompt

    def get_fashion_evaluation_user_prompt(self):
        '''
        의상 코디에 대한 관련성 여부를 평가 하기 위한 유저 프롬프트를 제공
        '''
        
        user_prompt = '''
Given <human_view> and <AI_view>, based on the guide on system prompt         
Write in the form of <evaluation> in korean with JSON format 

<human_view>{human_text}</human_view>
<AI_view>{AI_text}</AI_view>

<evaluation> 
'human_view': 
'AI_view' :  
'score': 4,
'reason': 'AI view is similar to human view'
</evaluation> 
'''
        return user_prompt        

