from aws_cdk import (
    aws_events as events,
    aws_events_targets as targets,
    aws_lambda as lambda_,
)
from constructs import Construct


class DeployImageTrigger(Construct):

    def __init__(self, scope: Construct, id_: str, deploy_image_func: lambda_.Function, ecr_repo_names: list) -> None:
        super().__init__(scope, id_)

        # Eventbridge, event-bus is default
        """
        {
            "source": ["aws.ecr"],
            "detail-type": ["ECR Image Action"],
            "detail": {
                "action-type": ["PUSH"],
                "result": ["SUCCESS"],
                "repository-name": ["json", "text", "recommandations", "test"],
                "image-tag": ["latest"]
            }
        }
        """
        events.Rule(
            self, 
            'DeployImageTrigger',
            description='Deploy Image',
            rule_name='DeployImageTrigger',
            event_pattern=events.EventPattern(                
                source=['aws.ecr'],
                detail_type=['ECR Image Action'],
                detail={
                    'action-type': ['PUSH'],
                    'result': ['SUCCESS'],
                    'repository-name': ecr_repo_names,
                    # 'repository-name': [                        
                    #     prompt_repo.repository_name,
                    #     summary_repo.repository_name,
                    #     identify_repo.repository_name,
                    #     comment_repo.repository_name,
                    #     shortform_repo.repository_name
                    # ],
                    'image-tag': ['latest']
                }
            ),
            targets=[
                targets.LambdaFunction(deploy_image_func)
            ]
        )




