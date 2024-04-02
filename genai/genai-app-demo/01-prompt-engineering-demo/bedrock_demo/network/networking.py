from aws_cdk import (
    CfnOutput,
    aws_ec2 as ec2,
    aws_elasticloadbalancingv2 as elbv2,
)
from constructs import Construct


class Networking(Construct):

    def __init__(self, scope: Construct, id_: str, cidr: str) -> None:
        super().__init__(scope, id_)

        # VPC
        self.vpc = ec2.Vpc(
            self,
            'BedrockDemoVPC',
            ip_addresses=ec2.IpAddresses.cidr(cidr),
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name='Public',
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name='Private',
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24
                )
            ],
            nat_gateways=2,
            enable_dns_hostnames=True,
            enable_dns_support=True,
            max_azs=2,
            restrict_default_security_group=False
        )

        # Security Group for Lambda
        self.sg_lambda = ec2.SecurityGroup(
            self,
            'LAMBDA_SG',
            vpc=self.vpc,
            allow_all_outbound=True,
            description='Security Group for Lambda',
            security_group_name='lambda-sg'
        )

        # Security Group for ALB
        sg_alb = ec2.SecurityGroup(
            self,
            'ALB_SG',
            vpc=self.vpc,
            allow_all_outbound=True,
            description='Security Group for ALB',
            security_group_name='alb-sg'
        )
        sg_alb.add_ingress_rule(
            peer=ec2.Peer.any_ipv4(),
            connection=ec2.Port.tcp(80),
            description="allow HTTP access from Internet"
        )

        alb = elbv2.ApplicationLoadBalancer(
            self,
            "BedrockALB",
            load_balancer_name='bedrock-alb',
            internet_facing=True,
            vpc=self.vpc,
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PUBLIC
            ),
            security_group=sg_alb
        )

        self.listener = alb.add_listener(
            "Listener",
            port=80,
            default_action=elbv2.ListenerAction.fixed_response(
                status_code=200,
                message_body="OK",
                content_type="text/plain"
            )
        )

        CfnOutput(
            self,
            'LoadBalancerDNS',
            value=alb.load_balancer_dns_name
        )
