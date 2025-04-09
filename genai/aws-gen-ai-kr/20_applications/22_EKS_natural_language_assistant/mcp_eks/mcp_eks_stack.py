#!/usr/bin/env python3
import os
import boto3
from aws_cdk import (
    Stack,
    CfnOutput,
    aws_ec2 as ec2,
    aws_eks as eks,
    aws_iam as iam,
    aws_lambda as lambda_,
    RemovalPolicy,
    Duration
)
from aws_cdk.lambda_layer_kubectl_v32 import KubectlV32Layer
from aws_cdk.lambda_layer_awscli import AwsCliLayer
from constructs import Construct
import requests

sts_client = boto3.client('sts')
current_identity = sts_client.get_caller_identity()
builder_user_arn = current_identity['Arn']


def get_public_ip():
    """Retrieve the current public IP address."""
    try:
        response = requests.get('https://api.ipify.org')
        return response.text
    except requests.RequestException:
        raise ValueError("Could not retrieve public IP address")


class EksClusterStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Create a VPC
        vpc = ec2.Vpc(
            self,
            "EksVpc",
            max_azs=2,  # Use 2 Availability Zones for cost-effectiveness
            nat_gateways=1  # Use 1 NAT Gateway to reduce costs
        )

        # Create EC2 Key Pair programmatically
        key_pair = ec2.CfnKeyPair(
            self,
            "EksBastion-KeyPair",
            key_name="eks-bastion-key",
            tags=[{"key": "Name", "value": "eks-bastion-key"}]
        )

        # Set removal policy to delete the key pair when stack is deleted
        key_pair.apply_removal_policy(RemovalPolicy.DESTROY)

        # Create an IAM role for the EKS cluster
        cluster_user = iam.Role.from_role_arn(
            self,
            "ClusterRole",
            role_arn=builder_user_arn
        )

        # Create IAM role for the bastion EC2 instance
        bastion_role = iam.Role(
            self,
            "BastionRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com")
        )

        kubectl_lambda_layer = KubectlV32Layer(self, "kubectl")
        awscli_lambda_layer = AwsCliLayer(self, "AwsCliLayer")

        kubectl_lambda_role = iam.Role(
            self, "KubectlLambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com")
        )

        kubectl_lambda = lambda_.Function(
            self,
            "KubectlExecutionFunction",
            runtime=lambda_.Runtime.PYTHON_3_12,
            code=lambda_.Code.from_asset("lambda/kubectl"),
            handler="index.handler",
            role=kubectl_lambda_role,
            timeout=Duration.seconds(300),
            memory_size=512,
            layers=[kubectl_lambda_layer, awscli_lambda_layer]
        )

        # Add necessary permissions to execute kubectl commands against the EKS cluster
        kubectl_lambda_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEKSClusterPolicy")
        )
        kubectl_lambda_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
        )
        kubectl_lambda_role.add_to_policy(
            iam.PolicyStatement(
                actions=["eks:DescribeCluster"],
                resources=["*"]
            )
        )

        # Create the EKS cluster
        cluster = eks.Cluster(
            self,
            "MCP-Demo-Cluster",
            version=eks.KubernetesVersion.V1_32,
            vpc=vpc,
            vpc_subnets=[ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS)],
            default_capacity=2,  # Start with 2 nodes
            default_capacity_instance=ec2.InstanceType("t3.medium"),  # Use moderate sized instances
            authentication_mode=eks.AuthenticationMode.API_AND_CONFIG_MAP,
            kubectl_layer=kubectl_lambda_layer,
            kubectl_lambda_role=kubectl_lambda_role,
            endpoint_access=eks.EndpointAccess.PUBLIC,
            masters_role=bastion_role
        )

        # Add EKS admin permissions to the bastion role
        bastion_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEKSClusterPolicy")
        )

        # Add permissions to describe EKS clusters and to use kubectl
        bastion_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "eks:*"
                ],
                resources=["*"]
            )
        )

        # Add the bastion role to the EKS cluster's auth config
        cluster.aws_auth.add_masters_role(
            role=bastion_role
        )

        cluster.aws_auth.add_masters_role(
            role=kubectl_lambda_role
        )

        cluster.aws_auth.add_user_mapping(
            user=cluster_user,
            groups=["system:masters"]
        )

        # Create a security group for the bastion
        bastion_sg = ec2.SecurityGroup(
            self,
            "BastionSecurityGroup",
            vpc=vpc,
            description="Security group for the EKS bastion host",
            allow_all_outbound=True
        )

        # Allow SSH inbound from anywhere (you might want to restrict this in production)
        bastion_sg.add_ingress_rule(
            ec2.Peer.ipv4(f"{get_public_ip()}/32"),
            ec2.Port.tcp(8000),
        )

        # The user data script to bootstrap the EC2 instance
        user_data = ec2.UserData.for_linux()
        kubernetes_version = "1.32.0"  # Match this with your EKS cluster version
        cluster_name = cluster.cluster_name
        region = self.region

        bootstrap_script = get_bootstrap_script().format(
            kubernetes_version=kubernetes_version,
            cluster_name=cluster_name,
            region=region
        )
        user_data.add_commands(bootstrap_script)

        # Create the EC2 bastion host
        bastion = ec2.Instance(
            self,
            "EKSBastion",
            instance_type=ec2.InstanceType("t3.medium"),
            machine_image=ec2.AmazonLinuxImage(
                generation=ec2.AmazonLinuxGeneration.AMAZON_LINUX_2023
            ),
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            role=bastion_role,
            security_group=bastion_sg,
            user_data=user_data,
            key_name=key_pair.key_name
        )

        # Add dependency on the key pair
        bastion.node.add_dependency(key_pair)
        bastion.node.add_dependency(cluster)

        env_content = (
            f"BASTION_HOST={bastion.instance.attr_public_dns_name}\n"
            "MCP_PORT=8000\n"
            f"AWS_REGION={self.region}\n"
            f"EKS_CLUSTER={cluster.cluster_name}\n"
            f"LAMBDA_ARN={kubectl_lambda.function_arn}"
        )


        # Create outputs to help user access the cluster
        CfnOutput(
            self,
            "ClusterName",
            value=cluster.cluster_name
        )

        CfnOutput(
            self,
            "BastionInstancePublicDnsName",
            value=bastion.instance_public_dns_name
        )

        CfnOutput(
            self,
            "BastionSSHCommand",
            value=f"ssh -i /path/to/eks-bastion-key.pem ec2-user@{bastion.instance_public_dns_name}"
        )

        CfnOutput(
            self,
            "UpdateKubeConfigCommand",
            value=f"aws eks update-kubeconfig --name {cluster.cluster_name} --region {self.region}"
        )

        # Create outputs to help retrieve the private key
        CfnOutput(
            self,
            "GetSSHKeyCommand",
            value=f"aws ssm get-parameter --name /ec2/keypair/{key_pair.attr_key_pair_id} --region {self.region} --with-decryption --query Parameter.Value --output text > {key_pair.key_name}.pem && chmod 400 {key_pair.key_name}.pem"
        )

        CfnOutput(
            self,
            "DotEnvFileContent",
            value=env_content
        )

def get_bootstrap_script():
    return """#!/bin/bash
# Update and install required packages
dnf update -y

# Install kubectl matching the cluster version
curl -O https://s3.us-west-2.amazonaws.com/amazon-eks/{kubernetes_version}/2024-12-20/bin/linux/amd64/kubectl
chmod +x ./kubectl
mv ./kubectl /usr/local/bin

# Configure kubectl to access the EKS cluster
aws eks update-kubeconfig --name {cluster_name} --region {region}

# Add KUBECONFIG to bashrc
echo 'export KUBECONFIG=$HOME/.kube/config' >> /home/ec2-user/.bashrc
echo 'export KUBECONFIG=$HOME/.kube/config' >> /root/.bashrc

# Copy the kubeconfig to ec2-user
mkdir -p /home/ec2-user/.kube
cp /root/.kube/config /home/ec2-user/.kube/
chown -R ec2-user:ec2-user /home/ec2-user/.kube/

# Download kubernetes-mcp-server binary
mkdir -p /home/ec2-user/k8s-mcp-server
cd /home/ec2-user/k8s-mcp-server
curl -L -o kubernetes-mcp-server-linux-amd64 https://github.com/manusa/kubernetes-mcp-server/releases/download/v0.0.21/kubernetes-mcp-server-linux-amd64
chmod +x kubernetes-mcp-server-linux-amd64

# Log file for tracking startup
LOGFILE="/var/log/kubernetes-mcp-server.log"

# Create a systemd service file for reliable process management
cat << EOF > /etc/systemd/system/kubernetes-mcp-server.service
[Unit]
Description=Kubernetes MCP Server
After=network.target

[Service]
ExecStart=/home/ec2-user/k8s-mcp-server/kubernetes-mcp-server-linux-amd64 --sse-port 8000
Restart=always
User=ec2-user
StandardOutput=append:$LOGFILE
StandardError=append:$LOGFILE

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd to recognize new service
systemctl daemon-reload

# Enable the service to start on boot
systemctl enable kubernetes-mcp-server

# Start the service
systemctl start kubernetes-mcp-server

# Optional: Log the startup attempt
echo "Kubernetes MCP Server startup initiated at $(date)" >> $LOGFILE
"""
