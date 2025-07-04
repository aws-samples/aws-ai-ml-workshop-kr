import * as cdk from 'aws-cdk-lib';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as iam from 'aws-cdk-lib/aws-iam';
import { Construct } from 'constructs';

interface ReviewBotLambdaProps {
  role: iam.IRole;
  requestsLayer: lambda.ILayerVersion;
  networkxLayer: lambda.ILayerVersion;
}

export class ReviewBotLambda extends Construct {
  public readonly functions: { [key: string]: lambda.Function } = {};

  constructor(scope: Construct, id: string, props: ReviewBotLambdaProps) {
    super(scope, id);

    // Common Lambda configuration
    const commonProps = {
      runtime: lambda.Runtime.PYTHON_3_12,
      role: props.role,
      tracing: lambda.Tracing.DISABLED,
      logRetention: cdk.aws_logs.RetentionDays.ONE_MONTH,
      environment: {
        POWERTOOLS_SERVICE_NAME: 'pr-reviewer',
        LOG_LEVEL: 'INFO'
      }
    };

    // Initial Processing Function
    this.functions.initialProcessing = new lambda.Function(this, 'InitialProcessing', {
      ...commonProps,
      functionName: 'PRR-InitialProcessingFunction',
      description: 'Process initial webhook payload',
      code: lambda.Code.fromAsset('src/lambda/initial-processing'),
      handler: 'index.lambda_handler',
      timeout: cdk.Duration.seconds(30),
      memorySize: 256,
      layers: [props.requestsLayer]
    });

    // Split PR Function
    this.functions.splitPr = new lambda.Function(this, 'SplitPR', {
      ...commonProps,
      functionName: 'PRR-SplitPRIntoChunksFunction',
      description: 'Split PR into analyzable chunks',
      code: lambda.Code.fromAsset('src/lambda/split-pr'),
      handler: 'index.lambda_handler',
      timeout: cdk.Duration.minutes(5),
      memorySize: 512,
      layers: [props.requestsLayer, props.networkxLayer]
    });

    // Process Chunk Function
    this.functions.processChunk = new lambda.Function(this, 'ProcessChunk', {
      ...commonProps,
      functionName: 'PRR-ProcessChunkFunction',
      description: 'Process individual PR chunk',
      code: lambda.Code.fromAsset('src/lambda/process-chunk'),
      handler: 'index.lambda_handler',
      timeout: cdk.Duration.minutes(10),
      memorySize: 1024,
      layers: [props.requestsLayer]
    });

    // Aggregate Results Function
    this.functions.aggregateResults = new lambda.Function(this, 'AggregateResults', {
      ...commonProps,
      functionName: 'PRR-AggregateResultsFunction',
      description: 'Aggregate review results',
      code: lambda.Code.fromAsset('src/lambda/aggregate-results'),
      handler: 'index.lambda_handler',
      timeout: cdk.Duration.minutes(5),
      memorySize: 512
    });

    // Post PR Comment Function
    this.functions.postPrComment = new lambda.Function(this, 'PostPRComment', {
      ...commonProps,
      functionName: 'PRR-PostPRCommentFunction',
      description: 'Post review comments to PR',
      code: lambda.Code.fromAsset('src/lambda/post-pr-comment'),
      handler: 'index.lambda_handler',
      timeout: cdk.Duration.minutes(2),
      memorySize: 256,
      layers: [props.requestsLayer]
    });

    // Send Slack Notification Function
    this.functions.sendSlackNotification = new lambda.Function(this, 'SendSlackNotification', {
      ...commonProps,
      functionName: 'PRR-SendSlackNotificationFunction',
      description: 'Send Slack notifications',
      code: lambda.Code.fromAsset('src/lambda/send-slack-notification'),
      handler: 'index.lambda_handler',
      timeout: cdk.Duration.minutes(1),
      memorySize: 256,
      layers: [props.requestsLayer]
    });

    // Handle Error Function
    this.functions.handleError = new lambda.Function(this, 'HandleError', {
      ...commonProps,
      functionName: 'PRR-HandleErrorFunction',
      description: 'Handle workflow errors',
      code: lambda.Code.fromAsset('src/lambda/handle-error'),
      handler: 'index.lambda_handler',
      timeout: cdk.Duration.minutes(1),
      memorySize: 256,
      layers: [props.requestsLayer]
    });

    // Add CloudWatch Dashboard
    this.createCloudWatchDashboard();
  }

  private createCloudWatchDashboard() {
    const dashboard = new cdk.aws_cloudwatch.Dashboard(this, 'ReviewBotDashboard', {
      dashboardName: 'PR-ReviewBot-Metrics'
    });

    const widgets: cdk.aws_cloudwatch.IWidget[] = [];

    // Add metrics for each function
    Object.entries(this.functions).forEach(([name, fn]) => {
      widgets.push(
        new cdk.aws_cloudwatch.GraphWidget({
          title: `${name} Metrics`,
          left: [
            fn.metricInvocations(),
            fn.metricErrors(),
            fn.metricThrottles(),
            fn.metricDuration()
          ],
          width: 12
        })
      );
    });

    // Add widgets to dashboard in rows of 2
    for (let i = 0; i < widgets.length; i += 2) {
      dashboard.addWidgets(...widgets.slice(i, i + 2));
    }
  }
}