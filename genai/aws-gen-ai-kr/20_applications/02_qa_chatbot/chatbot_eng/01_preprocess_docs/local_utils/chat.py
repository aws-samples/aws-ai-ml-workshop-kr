import ipywidgets as ipw
from utils import print_ww
from langchain import PromptTemplate
from IPython.display import display, clear_output
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory, ConversationBufferMemory

class chat_utils():
    
    MEMORY_TYPE = [
        "ConversationBufferMemory",
        "ConversationBufferWindowMemory",
        "ConversationSummaryBufferMemory"
    ]

    PROMPT_SUMMARY = PromptTemplate(
        input_variables=['summary', 'new_lines'],
        template="""
        \n\nHuman: Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary.

        EXAMPLE
        Current summary:
        The user asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good.

        New lines of conversation:
        User: Why do you think artificial intelligence is a force for good?
        AI: Because artificial intelligence will help users reach their full potential.

        New summary:
        The user asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good because it will help users reach their full potential.
        END OF EXAMPLE

        Current summary:
        {summary}

        New lines of conversation:
        {new_lines}

        \n\nAssistant:"""
    )

    @classmethod
    def get_memory(cls, **kwargs):
        """
        create memory for this chat session
        """
        memory_type = kwargs["memory_type"]

        assert memory_type in cls.MEMORY_TYPE, f"Check Buffer Name, Lists: {cls.MEMORY_TYPE}"
        if memory_type == "ConversationBufferMemory":

            # This memory allows for storing messages and then extracts the messages in a variable.
            memory = ConversationBufferMemory(
                memory_key=kwargs.get("memory_key", "chat_history"),
                human_prefix=kwargs.get("human_prefix", "Human"),
                ai_prefix=kwargs.get("ai_prefix", "AI"),
                return_messages=kwargs.get("return_messages", True)
            )
        elif memory_type == "ConversationBufferWindowMemory":

            # Maintains a history of previous messages
            # ConversationBufferWindowMemory keeps a list of the interactions of the conversation over time.
            # It only uses the last K interactions.
            # This can be useful for keeping a sliding window of the most recent interactions,
            # so the buffer does not get too large.
            # https://python.langchain.com/docs/modules/memory/types/buffer_window
            memory = ConversationBufferWindowMemory(
                k=kwargs.get("k", 5),
                memory_key=kwargs.get("memory_key", "chat_history"),
                human_prefix=kwargs.get("human_prefix", "Human"),
                ai_prefix=kwargs.get("ai_prefix", "AI"),
                return_messages=kwargs.get("return_messages", True),
            )
        elif memory_type == "ConversationSummaryBufferMemory":

            # Maintains a summary of previous messages
            # ConversationSummaryBufferMemory combines the two ideas.
            # It keeps a buffer of recent interactions in memory, but rather than just completely flushing old interactions it compiles them into a summary and uses both. 
            # It uses token length rather than number of interactions to determine when to flush interactions.

            assert kwargs.get("llm", None) != None, "Give your LLM"
            memory = ConversationSummaryBufferMemory(
                llm=kwargs.get("llm", None),
                memory_key=kwargs.get("memory_key", "chat_history"),
                human_prefix=kwargs.get("human_prefix", "User"),
                ai_prefix=kwargs.get("ai_prefix", "AI"),
                return_messages=kwargs.get("return_messages", True),
                max_token_limit=kwargs.get("max_token_limit", 1024),
                prompt=cls.PROMPT_SUMMARY
            )

        return memory
    
    @classmethod
    def get_tokens(cls, chain, prompt):
        token = chain.llm.get_num_tokens(prompt)
        print(f'# tokens: {token}')
        return token

    @classmethod
    def clear_memory(cls, chain):
        return chain.memory.clear()

class ChatUX:
    """ A chat UX using IPWidgets
    """
    def __init__(self, qa, retrievalChain = False):
        self.qa = qa
        self.name = None
        self.b=None
        self.retrievalChain = retrievalChain
        self.out = ipw.Output()

        if "ConversationChain" in str(type(self.qa)):
            self.streaming = self.qa.llm.streaming
        elif "ConversationalRetrievalChain" in str(type(self.qa)):
            self.streaming = self.qa.combine_docs_chain.llm_chain.llm.streaming

    def start_chat(self):
        print("Starting chat bot")
        display(self.out)
        self.chat(None)

    def chat(self, _):
        if self.name is None:
            prompt = ""
        else: 
            prompt = self.name.value
        if 'q' == prompt or 'quit' == prompt or 'Q' == prompt:
            print("Thank you , that was a nice chat !!")
            return
        elif len(prompt) > 0:
            with self.out:
                thinking = ipw.Label(value="Thinking...")
                display(thinking)
                try:
                    if self.retrievalChain:
                        result = self.qa.run({'question': prompt })
                    else:
                        result = self.qa.run({'input': prompt }) #, 'history':chat_history})
                except:
                    result = "No answer"
                thinking.value=""
                if self.streaming:
                    response = f"AI:{result}"
                else:
                    print_ww(f"AI:{result}")
                self.name.disabled = True
                self.b.disabled = True
                self.name = None

        if self.name is None:
            with self.out:
                self.name = ipw.Text(description="You:", placeholder='q to quit')
                self.b = ipw.Button(description="Send")
                self.b.on_click(self.chat)
                display(ipw.Box(children=(self.name, self.b)))