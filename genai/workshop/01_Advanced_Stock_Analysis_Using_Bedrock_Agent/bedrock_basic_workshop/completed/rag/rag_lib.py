import boto3
import json
import PyPDF2
import numpy as np
import faiss

def create_embeddings(text):
    session = boto3.Session()
    bedrock = session.client(service_name='bedrock-runtime')

    response = bedrock.invoke_model(
        body=json.dumps({"inputText": text}),
        modelId="amazon.titan-embed-text-v2:0",
        accept="application/json",
        contentType="application/json"
    )

    response_body = json.loads(response['body'].read())
    return np.array(response_body['embedding'])

def load_pdf(file_path):
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def create_text_splitter(text, chunk_size=1000, chunk_overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def create_vector_index(embeddings):
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

def generate_rag_response(index, question, documents):
    session = boto3.Session()
    llm = session.client(service_name='bedrock-runtime')

    question_embedding = create_embeddings(question)
    _, top_indices = index.search(np.array([question_embedding]), 4)
    top_texts = [documents[i] for i in top_indices[0]]
    rag_content = "\n\n".join(top_texts)

    message = {
        "role": "user",
        "content": [
            {"text": rag_content},
            {"text": "Please answer the following question based on the above content. If the question is unrelated to the content, please inform that it's difficult to answer."},
            {"text": question}
        ]
    }

    response = llm.converse(
        modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
        messages=[message],
        inferenceConfig={
            "maxTokens": 2000,
            "temperature": 0,
            "topP": 0.9,
            "stopSequences": []
        },
    )

    return response['output']['message']['content'][0]['text'], top_texts
