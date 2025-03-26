# Contextual RAG with Amazon Bedrock and OpenSearch

This project implements [Contextual Retrieval Augmented Generation (RAG)](https://www.anthropic.com/news/contextual-retrieval) system as introduced by Anthropic, using Amazon Bedrock, Amazon OpenSearch Service and Amazon Bedrock Reranker Model.

## Key Features
- Generate text using Foundation Model on Amazon Bedrock
- Hybrid search using Embedding-based similarity search (KNN) and keyword-based search (BM25)
- Optimize search result through Rank Fusion and Cross-encoder Reranking
- Store and mangage retrieval result using DynamoDB

## Installation
1. Clone this repository
2. Deploy CDK stack
```bash
npm i --save-dev @types/node
cdk bootstrap
cdk synth
cdk deploy --all --require-approval never
```

## Request Access to Foundation Model on Amazon Bedrock
1. Go to Amazon Bedrock Console
2. Open Model Access under Bedrock configurations
3. Request Access to LLM model and embedding model

## Usage
1. Go to Amazon SageMakerAI on your AWS Management Console
2. Open Notebook named `contextual-rag-workshop-instance`
3. Move on to path named `notebook`
```bash
cd contextual-rag-workshop-instance/notebook
```
4. Configure with `0_configuration.ipynb` first
5. Play around with following notebooks
