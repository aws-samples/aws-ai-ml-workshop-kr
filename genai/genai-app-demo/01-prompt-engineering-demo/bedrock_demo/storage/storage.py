from aws_cdk import (
    RemovalPolicy, CfnOutput, Aws,
    aws_s3 as s3,
    aws_iam as iam,
)
from constructs import Construct


class Storage(Construct):

    def __init__(self, scope: Construct, id_: str) -> None:
        super().__init__(scope, id_)

        self.bucket = s3.Bucket(
            self,
            'BedrockDemoBucket',
        )

        self.bucket.add_to_resource_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                principals=[iam.AnyPrincipal()],
                actions=[
                    's3:GetObject',
                    's3:GetBucketLocation',
                    's3:ListBucket',
                    's3:PutObject',
                    's3:GetBucketAcl'
                ],
                resources=[
                    self.bucket.bucket_arn,
                    self.bucket.arn_for_objects('*')
                ],
                conditions={
                    'StringEquals': {
                        'aws:PrincipalAccount': [
                            Aws.ACCOUNT_ID
                        ]
                    }
                }
            )
        )

        CfnOutput(
            self,
            'BucketName',
            value=self.bucket.bucket_name
        )
