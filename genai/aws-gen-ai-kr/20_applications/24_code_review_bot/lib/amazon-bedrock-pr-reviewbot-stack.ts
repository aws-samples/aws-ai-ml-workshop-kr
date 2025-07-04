import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import { ReviewBotProps } from './interfaces';
import { SecretsAndParameters } from './constructs/secrets-and-parameters';
import { ReviewBotRole } from './constructs/review-bot-role';
import { StateMachineRole } from './constructs/state-machine-role';
import { LambdaLayers } from './constructs/lambda-layer';
import { ReviewBotLambda } from './constructs/lambda';
import { ReviewBotApi } from './constructs/api-gateway';
import { ReviewBotStepFunctions } from './constructs/step-functions';
import { ReviewBotDynamoDB } from './constructs/dynamodb';

export class AmazonBedrockPrReviewbotStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: cdk.StackProps & ReviewBotProps) {
    super(scope, id, props);

    // Create Secrets and Parameters
    const secretsAndParams = new SecretsAndParameters(this, 'SecretsAndParams', props);

    // Create DynamoDB Table
    const dynamodb = new ReviewBotDynamoDB(this, 'ReviewBotDynamoDB');

    // Create Lambda Role with DynamoDB access
    const lambdaRole = new ReviewBotRole(this, 'ReviewBotRole', {
      secrets: secretsAndParams.secrets,
      region: this.region,
      account: this.account,
      dynamodbTableArn: dynamodb.resultsTable.tableArn,
      reportsTableArn: dynamodb.reportsTable.tableArn // Added reports table ARN
    }).role;

    // Create Lambda Layers
    const layers = new LambdaLayers(this, 'ReviewBotLayers');

    // Create Lambda Functions
    const lambdas = new ReviewBotLambda(this, 'ReviewBotLambda', {
      role: lambdaRole,
      requestsLayer: layers.requestsLayer,
      networkxLayer: layers.networkxLayer
    });

    // Create State Machine Role
    const stateMachineRole = new StateMachineRole(this, 'StateMachineRole', {
      lambdaFunctions: lambdas.functions
    }).role;

    // Create Step Functions
    const stepFunctions = new ReviewBotStepFunctions(this, 'ReviewBotStepFunctions', {
      functions: lambdas.functions,
      role: stateMachineRole
    });

    // Create API Gateway
    new ReviewBotApi(this, 'ReviewBotApi', {
      stateMachine: stepFunctions.stateMachine
    });

    // Add stack outputs
    this.addOutputs(stepFunctions, dynamodb);
  }

  private addOutputs(stepFunctions: ReviewBotStepFunctions, dynamodb: ReviewBotDynamoDB) {
    new cdk.CfnOutput(this, 'StateMachineArn', {
      value: stepFunctions.stateMachine.stateMachineArn,
      description: 'State Machine ARN'
    });

    new cdk.CfnOutput(this, 'DynamoDBTableName', {
      value: dynamodb.resultsTable.tableName,
      description: 'DynamoDB Table Name for PR Review results'
    });

    // Add output for reports table
    new cdk.CfnOutput(this, 'ReportsTableName', {
      value: dynamodb.reportsTable.tableName,
      description: 'DynamoDB Table Name for PR Review reports'
    });
  }
}