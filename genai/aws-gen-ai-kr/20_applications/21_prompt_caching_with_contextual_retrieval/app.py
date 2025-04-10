#!/usr/bin/env python3
import os
import aws_cdk as cdk
from document_chatbot_cdk.knowledge_base_stack import KnowledgebaseStack
from document_chatbot_cdk.bedrock_chatbot_cdk_stack import BedrockChatbotStack

app = cdk.App()

# First deploy the Knowledge Base stack
kb_stack = KnowledgebaseStack(app, "DocumentKnowledgeBaseStack")

# Then deploy the Chatbot Stack with the Knowledge Base ID and document URL
BedrockChatbotStack(
    app,
    "BedrockChatbotStack",
    kb_id=kb_stack.outputs["knowledgebase_id"],
    kb_document_url=kb_stack.outputs["document_cloudfront_url"],
    kb_outputs=kb_stack.outputs
)

app.synth()