import aws_cdk as core
import aws_cdk.assertions as assertions

from mcp_eks.mcp_eks_stack import McpEksStack

# example tests. To run these tests, uncomment this file along with the example
# resource in mcp_eks/mcp_eks_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = McpEksStack(app, "mcp-eks")
    template = assertions.Template.from_stack(stack)

#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })
