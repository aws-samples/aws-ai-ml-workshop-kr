import * as cdk from 'aws-cdk-lib';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import * as ssm from 'aws-cdk-lib/aws-ssm';
import { Construct } from 'constructs';
import { ReviewBotProps } from '../interfaces';

export class SecretsAndParameters extends Construct {
  public readonly secrets: { [key: string]: secretsmanager.Secret };
  public readonly parameters: { [key: string]: ssm.StringParameter };

  constructor(scope: Construct, id: string, props: ReviewBotProps) {
    super(scope, id);

    // Create Secrets
    this.secrets = this.createSecrets();
    
    // Create Parameters
    this.parameters = this.createParameters(props);

    // Add CloudFormation outputs
    this.createOutputs();
  }

  private createSecrets(): { [key: string]: secretsmanager.Secret } {
    const secretValues = {
      github: { access_token: 'input your github token' },
      gitlab: { access_token: 'input your gitlab token' },
      bitbucket: { access_token: 'input your bitbucket token' },
      slack: { token: 'input your slack token' }
    };

    const secrets: { [key: string]: secretsmanager.Secret } = {};

    // Create each secret with default value
    Object.entries(secretValues).forEach(([key, value]) => {
      secrets[`${key}Token`] = new secretsmanager.Secret(this, `${key}Token`, {
        secretName: `/pr-reviewer/tokens/${key}`,
        description: `${key.charAt(0).toUpperCase() + key.slice(1)} access token for PR ReviewBot`,
        generateSecretString: {
          secretStringTemplate: JSON.stringify(value),
          generateStringKey: 'dummy' // This key won't be used but is required
        }
      });
    });

    return secrets;
  }

  private createParameters(props: ReviewBotProps): { [key: string]: ssm.StringParameter } {
    const parameters: { [key: string]: ssm.StringParameter } = {
      repoType: new ssm.StringParameter(this, 'RepoType', {
        parameterName: '/pr-reviewer/config/repo_type',
        stringValue: props.repoType,
        description: 'Repository type (github/gitlab/bitbucket)'
      }),
      awsRegion: new ssm.StringParameter(this, 'AwsRegion', {
        parameterName: '/pr-reviewer/config/aws_region',
        stringValue: props.awsRegion,
        description: 'AWS Region for the service'
      }),
      model: new ssm.StringParameter(this, 'Model', {
        parameterName: '/pr-reviewer/config/model',
        stringValue: props.bedrockModel,
        description: 'Amazon Bedrock model ID'
      }),
      maxTokens: new ssm.StringParameter(this, 'MaxTokens', {
        parameterName: '/pr-reviewer/config/max_tokens',
        stringValue: String(props.maxTokens || 4096),
        description: 'Maximum tokens for model response'
      }),
      temperature: new ssm.StringParameter(this, 'Temperature', {
        parameterName: '/pr-reviewer/config/temperature',
        stringValue: String(props.temperature || 0.7),
        description: 'Temperature for model response'
      }),
      slackChannel: new ssm.StringParameter(this, 'SlackChannel', {
        parameterName: '/pr-reviewer/config/slack_channel',
        stringValue: props.slackChannel,
        description: 'Slack channel for notifications'
      }),
      slackNotification: new ssm.StringParameter(this, 'SlackNotification', {
        parameterName: '/pr-reviewer/config/slack_notification',
        stringValue: props.slackNotification,
        description: 'Slack notifications Enable/Disable'
      })
    };

    return parameters;
  }

  private createOutputs() {
    // Secrets outputs
    Object.entries(this.secrets).forEach(([key, secret]) => {
      new cdk.CfnOutput(this, `Secret${key}`, {
        value: secret.secretName,
        description: `Secret name for ${key}`
      });
    });

    // Parameters outputs
    Object.entries(this.parameters).forEach(([key, parameter]) => {
      new cdk.CfnOutput(this, `Parameter${key}`, {
        value: parameter.parameterName,
        description: `Parameter name for ${key}`
      });
    });
  }
}
