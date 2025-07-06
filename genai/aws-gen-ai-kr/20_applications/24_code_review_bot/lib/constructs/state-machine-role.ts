import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import { Construct } from 'constructs';

interface StateMachineRoleProps {
  lambdaFunctions: { [key: string]: lambda.Function };
}

export class StateMachineRole extends Construct {
  public readonly role: iam.Role;

  constructor(scope: Construct, id: string, props: StateMachineRoleProps) {
    super(scope, id);

    this.role = new iam.Role(this, 'ReviewBotStateMachineRole', {
      assumedBy: new iam.ServicePrincipal('states.amazonaws.com'),
      description: 'Role for PR ReviewBot State Machine',
    });

    // Lambda invocation permissions
    this.role.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: ['lambda:InvokeFunction'],
      resources: Object.values(props.lambdaFunctions).map(fn => fn.functionArn)
    }));

    // X-Ray permissions
    this.role.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'xray:PutTraceSegments',
        'xray:PutTelemetryRecords',
        'xray:GetSamplingRules',
        'xray:GetSamplingTargets'
      ],
      resources: ['*']
    }));

    // CloudWatch Logs permissions
    this.role.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'logs:CreateLogDelivery',
        'logs:GetLogDelivery',
        'logs:UpdateLogDelivery',
        'logs:DeleteLogDelivery',
        'logs:ListLogDeliveries',
        'logs:PutResourcePolicy',
        'logs:DescribeResourcePolicies',
        'logs:DescribeLogGroups'
      ],
      resources: ['*']
    }));

    // CloudWatch Metrics permissions
    this.role.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: ['cloudwatch:PutMetricData'],
      resources: ['*'],
      conditions: {
        'StringEquals': {
          'cloudwatch:namespace': 'PRReviewer/StateMachine'
        }
      }
    }));
  }
}