import re
from textwrap import dedent
from langchain_core.tracers import ConsoleCallbackHandler
from langchain.schema.output_parser import StrOutputParser
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder

class prompt_repo():
     
    @classmethod
    def get_system_prompt(cls, role=None):

        if role == "text2chart":

            system_prompt = dedent(
                '''
                You are a pandas master bot designed to generate Python code for plotting a chart based on the given dataset and user question.
                I'm going to give you dataset.
                Read the given dataset carefully, because I'm going to ask you a question about it.
                '''
            )
        else:
            system_prompt = ""

        return system_prompt

    @classmethod
    def get_human_prompt(cls, role=None):
        
        if role == "text2chart":
        
            human_prompt = dedent(
                 '''
                 This is the result of `print(df.head())`: <dataset>{dataset}</dataset>
                 You should execute code as commanded to either provide information to answer the question or to do the transformations required.
                 You should not assign any variables; you should return a one-liner in Pandas.

                 Update this initial code:

                 ```python
                 # TODO: import the required dependencies
                 import pandas as pd

                 # Write code here

                 ```

                 Here is the question: <question>{question}</question>

                 Variable `df: pd.DataFrame` is already declared.
                 At the end, declare "result" variable as a dictionary of type and value.
                 If you are asked to plot a chart, use "matplotlib" for charts, save as "results.png".
                 Expaination with Koren.
                 Do not use legend and title in plot in Korean.

                 Generate python code and return full updated code within <update_code></update_code>:


                 '''
            )

        else:
            human_prompt = ""
            
        return human_prompt

class text2chart_chain():
    
    def __init__(self, **kwargs):

        system_prompt = kwargs["system_prompt"]
        self.llm_text = kwargs["llm_text"]
        self.system_message_template = SystemMessagePromptTemplate.from_template(system_prompt)
        self.num_rows = kwargs["num_rows"]
        #self.return_context = kwargs.get("return_context", False)
        self.verbose = kwargs.get("verbose", False)
        self.parsing_pattern = kwargs["parsing_pattern"]
        self.show_chart = kwargs.get("verbose", False)
        
    def query(self, **kwargs):
        
        df, query, verbose = kwargs["df"], kwargs["query"], kwargs.get("verbose", self.verbose)
        show_chart = kwargs.get("show_chart", self.show_chart)
        
        if len(df) < self.num_rows: dataset = str(df.to_csv())
        else: dataset = str(df.sample(self.num_rows, random_state=0).to_csv())
        
        invoke_args = {
            "dataset": dataset,
            "question": query
        }
        
        human_prompt = prompt_repo.get_human_prompt(
            role="text2chart"
        )
        human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)
        prompt = ChatPromptTemplate.from_messages(
            [self.system_message_template, human_message_template]
        )
        
        code_generation_chain = prompt | self.llm_text | StrOutputParser()

        self.verbose = verbose
        response = code_generation_chain.invoke(
            invoke_args,
            config={'callbacks': [ConsoleCallbackHandler()]} if self.verbose else {}
        )

        if show_chart:
            results = self.code_execution(
                df=df,
                response=response
            )
            return results
        else:
            return response
    
    def code_execution(self, **kwargs):
        
        df, code = kwargs["df"], self._code_parser(response=kwargs["response"])
        tool = PythonAstREPLTool(locals={"df": df})
        
        results = tool.invoke(code)
        
        return results
        
    def _code_parser(self, **kwargs):
        
        parsed_code, response = "", kwargs["response"]
        match = re.search(self.parsing_pattern, response, re.DOTALL)

        if match: parsed_code = match.group(1)
        else: print("No match found.")
        
        return parsed_code
        