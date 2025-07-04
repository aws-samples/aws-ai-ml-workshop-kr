import * as cdk from 'aws-cdk-lib';
import * as stepfunctions from 'aws-cdk-lib/aws-stepfunctions';
import * as tasks from 'aws-cdk-lib/aws-stepfunctions-tasks';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as iam from 'aws-cdk-lib/aws-iam';
import { Construct } from 'constructs';

interface ReviewBotStepFunctionsProps {
  functions: { [key: string]: lambda.Function };
  role: iam.IRole;
}

export class ReviewBotStepFunctions extends Construct {
  public readonly stateMachine: stepfunctions.StateMachine;

  constructor(scope: Construct, id: string, props: ReviewBotStepFunctionsProps) {
    super(scope, id);

    // Initial Processing Task
    const initialProcessing = new tasks.LambdaInvoke(this, 'InitialProcessing', {
      lambdaFunction: props.functions.initialProcessing,
      inputPath: '$',
      resultPath: '$.processingResult',
      payloadResponseOnly: true,
      retryOnServiceExceptions: true,
    });

    // Split PR into Chunks Task
    const splitPr = new tasks.LambdaInvoke(this, 'SplitPRIntoChunks', {
      lambdaFunction: props.functions.splitPr,
      inputPath: '$.processingResult.body.data',
      payloadResponseOnly: true,
      retryOnServiceExceptions: true,
    });

    // Process Chunks - 단순화된 단일 Map 상태 사용 (청크 수에 관계없이)
    const processChunks = new stepfunctions.Map(this, 'ProcessChunks', {
      inputPath: '$.body',
      itemsPath: '$.chunks',
      maxConcurrency: 3, // 동시 처리 수 설정 
      resultPath: '$.processResults',
      // 결과 선택기를 통한 결과 구조화
      resultSelector: {
        'results.$': '$[*]'
      }
    }).itemProcessor(new tasks.LambdaInvoke(this, 'ProcessChunk', {
      lambdaFunction: props.functions.processChunk,
      payloadResponseOnly: true,
      retryOnServiceExceptions: true,
    }));

    // 성공 및 실패한 청크 분류
    const classifyResults = new stepfunctions.Pass(this, 'ClassifyResults', {
      parameters: {
        'succeeded.$': "$.processResults.results[?(@.statusCode == 200)]",
        'failed.$': "$.processResults.results[?(@.statusCode == 500 || @.statusCode == 429 || @.body.error && @.body.error.includes('RATE_LIMIT_ERROR'))]"
      },
      resultPath: '$.classifiedResults'
    });

    // 대기 상태
    const waitBeforeRetry = new stepfunctions.Wait(this, 'WaitBeforeRetry', {
      time: stepfunctions.WaitTime.duration(cdk.Duration.seconds(5))
    });

    // 실패한 청크가 있는 경우 재시도할 Map 상태
    const retryFailedChunks = new stepfunctions.Map(this, 'RetryFailedChunks', {
      inputPath: '$.classifiedResults',
      itemsPath: '$.failed',
      maxConcurrency: 1,
      resultPath: '$.retryResults'
    }).itemProcessor(new tasks.LambdaInvoke(this, 'RetryChunk', {
      lambdaFunction: props.functions.processChunk,
      payloadResponseOnly: true,
      retryOnServiceExceptions: true
    }));

    // 간단한 상태 전환만 수행하는 mergeResults
    const mergeResults = new stepfunctions.Pass(this, 'MergeResults', {
      // 아무 작업도 수행하지 않고 원래 데이터를 그대로 전달
      resultPath: '$.mergePhase'
    });
    
    // 이후 실행할 Lambda 함수에 전체 상태를 전달
    const aggregateResults = new tasks.LambdaInvoke(this, 'AggregateResults', {
      lambdaFunction: props.functions.aggregateResults,
      // 이렇게 하면 Lambda 함수가 $.classifiedResults.succeeded와 $.retryResults 모두 접근 가능
      inputPath: '$',
      payloadResponseOnly: true,
      retryOnServiceExceptions: true,
    });

    // Post PR Comment Task
    const postPrComment = new tasks.LambdaInvoke(this, 'PostPRComment', {
      lambdaFunction: props.functions.postPrComment,
      payloadResponseOnly: true,
      retryOnServiceExceptions: true,
    });

    // Send Slack Notification Task
    const sendSlackNotification = new tasks.LambdaInvoke(this, 'SendSlackNotification', {
      lambdaFunction: props.functions.sendSlackNotification,
      payloadResponseOnly: true,
      retryOnServiceExceptions: true,
    });

    // Handle Error Task
    const handleError = new tasks.LambdaInvoke(this, 'HandleError', {
      lambdaFunction: props.functions.handleError,
      payloadResponseOnly: true,
      retryOnServiceExceptions: true,
    });

    // Success State
    const success = new stepfunctions.Succeed(this, 'Success');

    // Failed State
    const failed = new stepfunctions.Fail(this, 'Failed', {
      error: 'WorkflowFailed',
      cause: 'Workflow execution failed'
    });

    // Parallel Post Results
    const postResults = new stepfunctions.Parallel(this, 'PostResults', {
      resultPath: '$.postResults'
    });
    postResults.branch(postPrComment);
    postResults.branch(sendSlackNotification);

    // Add error handlers
    initialProcessing.addCatch(handleError, {
      resultPath: '$.error',
    });
    splitPr.addCatch(handleError, {
      resultPath: '$.error',
    });
    processChunks.addCatch(handleError, {
      resultPath: '$.error',
    });
    aggregateResults.addCatch(handleError, {
      resultPath: '$.error',
    });
    postResults.addCatch(handleError, {
      resultPath: '$.error',
    });

    // Define retry policies
    const defaultRetry = {
      errors: ['States.TaskFailed'],
      interval: cdk.Duration.seconds(3),
      maxAttempts: 2,
      backoffRate: 1.5,
    };

    // Add retry policies to all Lambda tasks
    [initialProcessing, splitPr, aggregateResults, 
     postPrComment, sendSlackNotification, handleError].forEach(task => {
      task.addRetry(defaultRetry);
    });

    // 실패한 청크 재처리를 위한 Choice 상태 - 수정된 부분
    const checkFailedChunks = new stepfunctions.Choice(this, 'CheckFailedChunks');
    
    // Choice 이후 경로 정의
    // 실패한 청크가 있으면 대기 후 재시도
    checkFailedChunks
      .when(
        stepfunctions.Condition.isPresent('$.classifiedResults.failed[0]'),
        waitBeforeRetry.next(retryFailedChunks).next(mergeResults)
      )
      .otherwise(mergeResults);

    // 이후 워크플로우 단계들을 설정
    const finalChain = mergeResults.next(aggregateResults).next(postResults).next(success);

    // Create State Machine - 수정된 워크플로우 정의
    this.stateMachine = new stepfunctions.StateMachine(this, 'PRReviewStateMachine', {
      stateMachineName: 'PR-REVIEWER',
      definitionBody: stepfunctions.DefinitionBody.fromChainable(
        initialProcessing
          .next(splitPr)
          .next(processChunks)
          .next(classifyResults)
          .next(checkFailedChunks) // 여기서 체인이 연결될 수 있도록 수정
      ),
      role: props.role,
      timeout: cdk.Duration.minutes(30),
      tracingEnabled: true,
      logs: {
        destination: new cdk.aws_logs.LogGroup(this, 'ReviewBotStateMachineLogs', {
          logGroupName: '/aws/vendedlogs/states/pr-reviewer',
          retention: cdk.aws_logs.RetentionDays.ONE_MONTH,
          removalPolicy: cdk.RemovalPolicy.DESTROY
        }),
        level: stepfunctions.LogLevel.ALL,
        includeExecutionData: true
      },
      comment: 'State machine for processing PR reviews using Amazon Bedrock'
    });

    // Add CloudFormation outputs
    this.createOutputs();
  }

  private createOutputs() {
    new cdk.CfnOutput(this, 'StateMachineArn', {
      value: this.stateMachine.stateMachineArn,
      description: 'State Machine ARN'
    });

    new cdk.CfnOutput(this, 'StateMachineUrl', {
      value: `https://console.aws.amazon.com/states/home?region=${cdk.Stack.of(this).region}#/statemachines/view/${this.stateMachine.stateMachineArn}`,
      description: 'State Machine Console URL'
    });
  }
}