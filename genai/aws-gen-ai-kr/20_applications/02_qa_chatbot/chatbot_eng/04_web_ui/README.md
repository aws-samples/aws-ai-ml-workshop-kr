# How to run this application

## Structure

1. `bedrock.py`

- This file implements RAG techniques such as Amazon Bedrock, Reranker, Hybrid search, and parent-document.

2. `streamlit.py`

- This is the front-end file of the application. When running, it imports `bedrock.py` and uses it.

## Start

1. Access the web_ui folder

```
cd ~/SageMaker/aws-ai-ml-workshop-kr/genai/aws-gen-ai-kr/20_applications/02_qa_chatbot/chatbot_eng/04_web_ui/
```

2. Install Python dependency libraries

```
pip install -r requirements.txt
```

3. Run the Streamlit application

```
streamlit run streamlit.py --server.baseUrlPath="/proxy/absolute/8501"
```

3. Access the application

- To access the application using the SageMaker Notebook domain, append /proxy/absolute/8501 to the domain.
