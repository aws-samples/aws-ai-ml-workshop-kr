#!/usr/bin/env python3
import os

import aws_cdk as cdk

from mcp_eks.mcp_eks_stack import EksClusterStack


app = cdk.App()
EksClusterStack(app, "McpEksStack")

app.synth()
