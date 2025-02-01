import boto3

def context_retrieval_from_kb(prompt, top_k, region, kb_id, search_type):

    bedrock_agent_client = boto3.client('bedrock-agent-runtime', region_name=region)
    response = bedrock_agent_client.retrieve(
        knowledgeBaseId=kb_id,
        retrievalConfiguration={
            'vectorSearchConfiguration': {
                'numberOfResults': top_k,
                'overrideSearchType': search_type
            }
        },
        retrievalQuery={
            'text': prompt
        }
    )        
    raw_result = response.get('retrievalResults', [])
    
    search_result = []
    context = ""

    if not raw_result:
        context = "No Relevant Context"
    else:
        for idx, result in enumerate(raw_result):
            content = result.get('content', {}).get('text', 'No content available')
            score = result.get('score', 'N/A')
            source = result.get('location', {})

            search_result.append({
                "index": idx + 1,
                "content": content,
                "source": source,
                "score": score
            })

            context += f"Result {idx + 1}:\nContent: {content}\nSource: {source}\nScore: {score}\n\n"

    prompt_new = f"Here is some context for you: \n<context>\n{context}</context>\n\n{prompt}"

    return search_result

