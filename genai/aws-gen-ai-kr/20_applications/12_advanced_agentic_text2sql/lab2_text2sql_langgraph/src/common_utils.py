import json
from typing import List, Union
import streamlit as st
import os
import yaml
import re
from PIL import Image, UnidentifiedImageError
from langchain.callbacks.base import BaseCallbackHandler

class ToolStreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
        self.placeholder = self.container.empty()

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.placeholder.markdown(self.text)
        
    def on_llm_new_result(self, token: str, **kwargs) -> None:
        try:
            parsed_token = json.loads(token)
            formatted_token = json.dumps(parsed_token, indent=2, ensure_ascii=False)
            self.text += "\n\n```json\n" + formatted_token + "\n```\n\n"
        except json.JSONDecodeError:
            if token.strip().upper().startswith("SELECT"):
                self.text += "\n\n```sql\n" + token + "\n```\n\n"
            elif token.strip().upper().startswith("COUNTRY,TOTALREVENUE"):
                self.text += "\n\n```\n" + token + "\n```\n\n"
            else:
                self.text += "\n\n" + token + "\n\n"
        
        self.placeholder.markdown(self.text)


def display_user_message(message_content: Union[str, List[dict]]) -> None:
    if isinstance(message_content, str):
        message_text = message_content
    elif isinstance(message_content, dict):
        message_text = message_content["input"][0]["content"][0]["text"]
    else:
        message_text = message_content[0]["text"]

    message_content_markdown = message_text.split('</context>\n\n', 1)[-1]
    st.markdown(message_content_markdown)


def display_assistant_message(message_content: Union[str, dict]) -> None:
    if isinstance(message_content, str):
        st.markdown(message_content)
    elif "response" in message_content:
        st.markdown(message_content["response"])

def display_chat_messages(
    uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]
) -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if uploaded_files and "images" in message and message["images"]:
                display_images(message["images"], uploaded_files)
            if message["role"] == "user":
                display_user_message(message["content"])
            if message["role"] == "assistant":
                display_assistant_message(message["content"])

def stream_converse_messages(client, model, tool_config, messages, system, callback, tokens):
    response = client.converse_stream(
        modelId=model,
        messages=messages,
        system=system,
        toolConfig=tool_config
    )
    
    stop_reason = ""
    message = {"content": []}
    text = ''
    tool_use = {}

    for chunk in response['stream']:
        if 'messageStart' in chunk:
            message['role'] = chunk['messageStart']['role']
        elif 'contentBlockStart' in chunk:
            tool = chunk['contentBlockStart']['start']['toolUse']
            tool_use['toolUseId'] = tool['toolUseId']
            tool_use['name'] = tool['name']
        elif 'contentBlockDelta' in chunk:
            delta = chunk['contentBlockDelta']['delta']
            if 'toolUse' in delta:
                if 'input' not in tool_use:
                    tool_use['input'] = ''
                tool_use['input'] += delta['toolUse']['input']
            elif 'text' in delta:
                text += delta['text']
                callback.on_llm_new_token(delta['text'])
        elif 'contentBlockStop' in chunk:
            if 'input' in tool_use:
                tool_use['input'] = json.loads(tool_use['input'])
                message['content'].append({'toolUse': tool_use})
                tool_use = {}
            else:
                message['content'].append({'text': text})
                text = ''
        elif 'messageStop' in chunk:
            stop_reason = chunk['messageStop']['stopReason']
        elif 'metadata' in chunk:
            tokens['total_input_tokens'] += chunk['metadata']['usage']['inputTokens']
            tokens['total_output_tokens'] += chunk['metadata']['usage']['outputTokens']
    return stop_reason, message


def parse_json_format(json_string):
    json_string = re.sub(r'"""\s*(.*?)\s*"""', r'"\1"', json_string, flags=re.DOTALL)
    json_string = re.sub(r'```json|```|</?response_format>|\n\s*', ' ', json_string)
    json_string = json_string.strip()
    match = re.search(r'({.*})', json_string)
    if match:
        json_string = match.group(1)
    else:
        return "No JSON object found in the string."

    try:
        parsed_json = json.loads(json_string)
    except json.JSONDecodeError as e:
        print("Original output: ", json_string)
        return f"JSON Parsing Error: {e}"
    return parsed_json

def update_tokens_and_costs(tokens):
    st.session_state.tokens['delta_input_tokens'] = tokens['total_input_tokens']
    st.session_state.tokens['delta_output_tokens'] = tokens['total_output_tokens']
    st.session_state.tokens['total_input_tokens'] += tokens['total_input_tokens']
    st.session_state.tokens['total_output_tokens'] += tokens['total_output_tokens']
    st.session_state.tokens['delta_total_tokens'] = tokens['total_tokens']
    st.session_state.tokens['total_tokens'] += tokens['total_tokens']

def calculate_and_display_costs(input_cost, output_cost, total_cost):
    with st.sidebar:
        st.header("Token Usage and Cost")
        st.markdown(f"**Input Tokens:** <span style='color:#555555;'>{st.session_state.tokens['total_input_tokens']}</span> <span style='color:green;'>(+{st.session_state.tokens['delta_input_tokens']})</span> (${input_cost:.2f})", unsafe_allow_html=True)
        st.markdown(f"**Output Tokens:** <span style='color:#555555;'>{st.session_state.tokens['total_output_tokens']}</span> <span style='color:green;'>(+{st.session_state.tokens['delta_output_tokens']})</span> (${output_cost:.2f})", unsafe_allow_html=True)
        st.markdown(f"**Total Tokens:** <span style='color:#555555;'>{st.session_state.tokens['total_tokens']}</span> <span style='color:green;'>(+{st.session_state.tokens['delta_total_tokens']})</span> (${total_cost:.2f})", unsafe_allow_html=True)
    st.sidebar.button("Init Tokens", on_click=init_tokens_and_costs, type="primary")

def init_tokens_and_costs() -> None:
    st.session_state.tokens['delta_input_tokens'] = 0
    st.session_state.tokens['delta_output_tokens'] = 0
    st.session_state.tokens['total_input_tokens'] = 0
    st.session_state.tokens['total_output_tokens'] = 0
    st.session_state.tokens['delta_total_tokens'] = 0
    st.session_state.tokens['total_tokens'] = 0


class CustomUploadedFile:
    def __init__(self, name, type, data):
        self.name = name
        self.type = type
        self.data = data
        self.file_id = name 

    def read(self, *args):
        return self.data.read(*args)

    def seek(self, *args):
        return self.data.seek(*args)

    def readlines(self):
        return self.data.readlines()
    
    def readline(self, *args):
        return self.data.readline(*args)
    
    def tell(self):
        return self.data.tell()

def process_uploaded_files(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile], message_images_list: List[str], uploaded_file_ids: List[str]) -> List[Union[dict, str]]:
    num_cols = 10
    cols = st.columns(num_cols)
    i = 0
    content_files = []

    for uploaded_file in uploaded_files:
        if uploaded_file.file_id not in message_images_list:
            uploaded_file_ids.append(uploaded_file.file_id)
            try:
                # Try to open as an image
                img = Image.open(uploaded_file)
                with BytesIO() as output_buffer:
                    img.save(output_buffer, format=img.format)
                    content_image = base64.b64encode(output_buffer.getvalue()).decode("utf8")
                content_files.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": content_image,
                    },
                })
                with cols[i]:
                    st.image(img, caption="", width=75)
                    i += 1
                if i >= num_cols:
                    i = 0
            except UnidentifiedImageError:
                # If not an image, try to read as a text or pdf file
                if uploaded_file.type in ['text/plain', 'text/csv', 'text/x-python-script']:
                    # Ensure we're at the start of the file
                    uploaded_file.seek(0)
                    # Read file line by line
                    lines = uploaded_file.readlines()
                    text = ''.join(line.decode() for line in lines)
                    content_files.append({
                        "type": "text",
                        "text": text
                    })
                    if uploaded_file.type == 'text/x-python-script':
                        st.write(f"🐍 Uploaded Python file: {uploaded_file.name}")
                    else:
                        st.write(f"📄 Uploaded text file: {uploaded_file.name}")
                elif uploaded_file.type == 'application/pdf':
                    # Read pdf file
                    pdf_file = pdfplumber.open(uploaded_file)
                    page_text = ""
                    for page in pdf_file.pages:
                        page_text += page.extract_text()
                    content_files.append({
                        "type": "text",
                        "text": page_text
                    })
                    st.write(f"📑 Uploaded PDF file: {uploaded_file.name}")
                    pdf_file.close()

    return content_files

def load_model_config():
    file_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(file_dir, "config.yml")

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config['models']

def load_language_config(language):
    file_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(file_dir, "config.yml")

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config['languages'][language]



def sample_query_indexing(os_client, lang_config):
    rag_query_file = st.text_input(lang_config['rag_query_file'], value="./db_metadata/chinook_example_queries.jsonl")
    if not os.path.exists(rag_query_file):
        st.warning(lang_config['file_not_found'])
        return

    if st.sidebar.button(lang_config['process_file'], key='query_file_process'):
        with st.spinner("Now processing..."):
            os_client.delete_index()
            os_client.create_index() 

            with open(rag_query_file, 'r') as file:
                bulk_data = file.read()

            response = os_client.conn.bulk(body=bulk_data)
            if response["errors"]:
                st.error("Failed")
            else:
                st.success("Success")


def schema_desc_indexing(os_client, lang_config):
    schema_file = st.text_input(lang_config['schema_file'], value="./db_metadata/chinook_detailed_schema.json")
    if not os.path.exists(schema_file):
        st.warning(lang_config['file_not_found'])
        return

    if st.sidebar.button(lang_config['process_file'], key='schema_file_process'):
        with st.spinner("Now processing..."):
            os_client.delete_index()
            os_client.create_index() 

            with open(schema_file, 'r', encoding='utf-8') as file:
                schema_data = json.load(file)

            bulk_data = []
            for table in schema_data:
                for table_name, table_info in table.items():
                    table_doc = {
                        "table_name": table_name,
                        "table_desc": table_info["table_desc"],
                        "columns": [{"col_name": col["col"], "col_desc": col["col_desc"]} for col in table_info["cols"]],
                        "table_summary": table_info["table_summary"],
                        "table_summary_v": table_info["table_summary_v"]
                    }
                    bulk_data.append({"index": {"_index": os_client.index_name, "_id": table_name}})
                    bulk_data.append(table_doc)
            
            bulk_data_str = '\n'.join(json.dumps(item) for item in bulk_data) + '\n'

            response = os_client.conn.bulk(body=bulk_data_str)
            if response["errors"]:
                st.error("Failed")
            else:
                st.success("Success")
    