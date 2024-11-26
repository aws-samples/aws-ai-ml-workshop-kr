import boto3
import json

# Load language data from JSON file
with open('./localization.json', 'r', encoding='utf-8') as f:
    LANG = json.load(f)

def get_streaming_response(prompt_input, transcription_text, response_placeholder, language='ko'):
    session = boto3.Session()
    bedrock = session.client(service_name='bedrock-runtime')

    message = {
        "role": "user",
        "content": [
            {"text": f"<transcript>{transcription_text}</transcript>"},
            {"text": LANG[language]['transcript_instruction']},
            {"text": prompt_input}
        ]
    }

    response = bedrock.converse_stream(
        modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
        messages=[message],
        inferenceConfig={
            "maxTokens": 2000,
            "temperature": 0.0
        }
    )

    stream = response.get('stream')
    response_text = ""
    for event in stream:
        if "contentBlockDelta" in event:
            response_text += event['contentBlockDelta']['delta']['text']
            response_placeholder.write(response_text)
