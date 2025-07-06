#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { AmazonBedrockPrReviewbotStack } from '../lib/amazon-bedrock-pr-reviewbot-stack';

class ReviewBotApp {
  private readonly app: cdk.App;
  private readonly defaultRegion: string = 'ap-northeast-2';
  private readonly requiredEnvVars = [
    'CDK_DEFAULT_ACCOUNT'
  ];

  constructor() {
    this.app = new cdk.App();
    this.validateEnvironment();
    this.createStack();
  }

  private validateEnvironment(): void {
    // 필수 환경 변수 검증
    for (const envVar of this.requiredEnvVars) {
      if (!process.env[envVar]) {
        throw new Error(`Missing required environment variable: ${envVar}`);
      }
    }

    // AWS 리전 검증
    const region = process.env.CDK_DEFAULT_REGION || this.defaultRegion;
    const supportedRegions = [
      'ap-northeast-2',  // Seoul
      'us-east-1',      // N. Virginia
      'us-west-2',      // Oregon
      'eu-west-1'       // Ireland
    ];

    if (!supportedRegions.includes(region)) {
      throw new Error(`Unsupported region: ${region}. Must be one of: ${supportedRegions.join(', ')}`);
    }
  }

  private getEnvironment(): cdk.Environment {
    return {
      account: process.env.CDK_DEFAULT_ACCOUNT,
      region: process.env.CDK_DEFAULT_REGION || this.defaultRegion
    };
  }

  private getStackConfig() {
    // Repository type validation
    const validRepoTypes = ['github', 'gitlab', 'bitbucket'];
    const repoType = process.env.REPO_TYPE || 'github';
    
    if (!validRepoTypes.includes(repoType)) {
      throw new Error(`Invalid repository type: ${repoType}. Must be one of: ${validRepoTypes.join(', ')}`);
    }

    // Bedrock model validation
    const validModels = [
      'apac.anthropic.claude-3-5-sonnet-20241022-v2:0'
    ];
    
    const model = process.env.BEDROCK_MODEL || validModels[0];
    
    if (!validModels.includes(model)) {
      throw new Error(`Invalid Bedrock model: ${model}. Must be one of: ${validModels.join(', ')}`);
    }

    // Configuration object
    return {
      repoType: repoType,
      awsRegion: this.getEnvironment().region!,
      bedrockModel: model,
      maxTokens: parseInt(process.env.MAX_TOKENS || '8192'),
      temperature: parseFloat(process.env.TEMPERATURE || '0.7'),
      slackChannel: process.env.SLACK_CHANNEL || 'pr-reviews',
      slackNotification: process.env.SLACK_NOTIFICATION || 'disable'
    };
  }

  private createStack(): void {
    const stackName = 'AmazonBedrockPrReviewbotStack';
    const env = this.getEnvironment();
    const config = this.getStackConfig();

    // Stack 생성
    const stack = new AmazonBedrockPrReviewbotStack(this.app, stackName, {
      env,
      ...config,
      // Stack 속성
      description: 'Serverless PR Review Bot using Amazon Bedrock',
      terminationProtection: true,
      stackName: stackName,
      // Tags
      tags: {
        Project: 'PR-ReviewBot',
        ManagedBy: 'CDK',
        LastUpdated: new Date().toISOString()
      },
      // Cross-stack references
      crossRegionReferences: true,
      // Analytics reporting
      analyticsReporting: true
    });

    // Context 값 검증
    this.validateContext();
  }

  private validateContext(): void {
    const requiredContext = [
      '@aws-cdk/aws-lambda:recognizeLayerVersion',
      '@aws-cdk/core:checkSecretUsage',
      '@aws-cdk/aws-apigateway:disableCloudWatchRole',
      '@aws-cdk/aws-iam:minimizePolicies'
    ];

    for (const context of requiredContext) {
      if (!this.app.node.tryGetContext(context)) {
        throw new Error(`Missing required context value: ${context}`);
      }
    }
  }

  public synth(): void {
    this.app.synth();
  }
}

// 애플리케이션 실행
try {
  const app = new ReviewBotApp();
  app.synth();
} catch (error) {
  if (error instanceof Error) {
    console.error('Error during stack synthesis:', error.message);
    if (error.stack) {
      console.error('Stack trace:', error.stack);
    }
  }
  process.exit(1);
}
