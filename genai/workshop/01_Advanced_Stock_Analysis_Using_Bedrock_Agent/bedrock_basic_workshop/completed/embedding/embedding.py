import json
import boto3
import sys
from numpy import dot
from numpy.linalg import norm

def get_embedding(text):
    session = boto3.Session()
    bedrock = session.client(service_name='bedrock-runtime')

    response = bedrock.invoke_model(
        body=json.dumps({"inputText": text}),
        modelId="amazon.titan-embed-text-v2:0",
        accept="application/json",
        contentType="application/json"
    )

    response_body = json.loads(response['body'].read())
    return response_body['embedding']

class EmbedItem:
    def __init__(self, text):
        self.text = text
        self.embedding = get_embedding(text)

class ComparisonResult:
    def __init__(self, text, similarity):
        self.text = text
        self.similarity = similarity

def calculate_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

with open("items.txt", "r") as f:
    text_items = f.read().splitlines()

items = []
for text in text_items:
    items.append(EmbedItem(text))

input_item = EmbedItem(sys.argv[1])

print(f"Similarity Sorting : '{input_item.text}'")
print("----------------")
cosine_comparisons = []

for item in items:
    similarity_score = calculate_similarity(input_item.embedding, item.embedding)

    cosine_comparisons.append(ComparisonResult(item.text, similarity_score))  # save the comparisons to a list

cosine_comparisons.sort(key=lambda x: x.similarity, reverse=True)  # list the closest matches first

for c in cosine_comparisons:
    print("%.6f" % c.similarity, "\t", c.text)
