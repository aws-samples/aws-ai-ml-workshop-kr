import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import { Construct } from 'constructs';

interface ReviewBotRoleProps {
  secrets: { [key: string]: cdk.aws_secretsmanager.ISecret };
  region: string;
  account: string;
  dynamodbTableArn: string;
  reportsTableArn: string; // New reports table ARN
}

export class ReviewBotRole extends Construct {
  public readonly role: iam.Role;

  constructor(scope: Construct, id: string, props: ReviewBotRoleProps) {
    super(scope, id);

    this.role = new iam.Role(this, 'ReviewBotLambdaRole', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
      description: 'Shared role for ReviewBot Lambda functions',
      managedPolicies: [
        // Basic Lambda execution permissions
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSLambdaBasicExecutionRole'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonBedrockFullAccess'),
      ]
    });

    // Secrets Manager permissions
    this.role.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: ['secretsmanager:GetSecretValue'],
      resources: Object.values(props.secrets).map(secret => secret.secretArn)
    }));

    // SSM Parameter Store permissions
    this.role.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'ssm:GetParameter',
        'ssm:GetParameters',
        'ssm:GetParametersByPath'
      ],
      resources: [
        `arn:aws:ssm:${props.region}:${props.account}:parameter/pr-reviewer/*`
      ]
    }));

    // CloudWatch Logs permissions
    this.role.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'logs:CreateLogStream',
        'logs:PutLogEvents'
      ],
      resources: [
        `arn:aws:logs:${props.region}:${props.account}:log-group:/aws/lambda/*`
      ]
    }));

    // CloudWatch Metrics permissions
    this.role.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: ['cloudwatch:PutMetricData'],
      resources: ['*'],
      conditions: {
        'StringEquals': {
          'cloudwatch:namespace': 'PRReviewer'
        }
      }
    }));

    // DynamoDB 권한 추가 - 결과 테이블
    this.role.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'dynamodb:PutItem',
        'dynamodb:GetItem',
        'dynamodb:Query',
        'dynamodb:Scan',
        'dynamodb:BatchGetItem',
        'dynamodb:UpdateItem',
        'dynamodb:DeleteItem'
      ],
      resources: [
        props.dynamodbTableArn,
        `${props.dynamodbTableArn}/index/*`
      ]
    }));

    // DynamoDB 권한 추가 - 리포트 테이블 (새로 추가)
    this.role.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'dynamodb:PutItem',
        'dynamodb:GetItem',
        'dynamodb:Query',
        'dynamodb:Scan',
        'dynamodb:BatchGetItem',
        'dynamodb:UpdateItem',
        'dynamodb:DeleteItem'
      ],
      resources: [
        props.reportsTableArn,
        `${props.reportsTableArn}/index/*`
      ]
    }));

    // CloudFormation 출력
    new cdk.CfnOutput(this, 'RoleArn', {
      value: this.role.roleArn,
      description: 'ARN of the ReviewBot Lambda role'
    });
  }
}