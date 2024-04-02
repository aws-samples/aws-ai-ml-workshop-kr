from aws_cdk import (
    aws_iam as iam
)

"""
    "containers/prompt": "prompt-func",
    "containers/summarization": "summary-func",
    "containers/identification": "identify-func",
    "containers/comment": "comment-func",
    "containers/shortform": "shortform-func"  
"""

ECR_LAMBDA_POLICY = iam.PolicyStatement(
    actions=[
        'ecr:BatchGetImage',
        'ecr:DeleteRepositoryPolicy',
        'ecr:GetDownloadUrlForLayer',
        'ecr:GetRepositoryPolicy',
        'ecr:SetRepositoryPolicy'
    ],
    principals=[
        iam.ServicePrincipal("lambda.amazonaws.com")
    ],
    effect=iam.Effect.ALLOW
)