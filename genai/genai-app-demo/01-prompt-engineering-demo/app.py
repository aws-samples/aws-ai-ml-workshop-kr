#!/usr/bin/env python3
import os
import aws_cdk as cdk

from bedrock_demo.config import app_config
from bedrock_demo.deployment import BedrockDemo
from cdk_nag import AwsSolutionsChecks, NagSuppressions


app = cdk.App()
conf = app_config['config']
stack = BedrockDemo(
    app,
    'BedrockDemo',
    conf=conf,
    env=cdk.Environment(account=os.getenv('CDK_DEFAULT_ACCOUNT'), region=os.getenv('CDK_DEFAULT_REGION')),
    tags={'Project': 'bedrock-demo'}
)
cdk.Aspects.of(app).add(AwsSolutionsChecks())
NagSuppressions.add_stack_suppressions(stack, [
    {'id': 'AwsSolutions-CB3', 'reason': 'allowed in this stack'},
    {'id': 'AwsSolutions-CB4', 'reason': 'allowed in this stack'},
    {'id': 'AwsSolutions-IAM4', 'reason': 'allowed in this stack'},
    {'id': 'AwsSolutions-IAM5', 'reason': 'allowed in this stack'},
    {'id': 'AwsSolutions-VPC7', 'reason': 'allowed in this stack'},
    {'id': 'AwsSolutions-ELB2', 'reason': 'allowed in this stack'},
    {'id': 'AwsSolutions-EC23', 'reason': 'allowed in this stack'},
    {'id': 'AwsSolutions-S1', 'reason': 'allowed in this stack'},
    {'id': 'AwsSolutions-S10', 'reason': 'allowed in this stack'},
])
app.synth()
