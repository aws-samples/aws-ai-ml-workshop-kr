import * as cdk from 'aws-cdk-lib';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import { Construct } from 'constructs';

export class ReviewBotDynamoDB extends Construct {
  public readonly resultsTable: dynamodb.Table;
  public readonly reportsTable: dynamodb.Table; // New table for markdown reports

  constructor(scope: Construct, id: string) {
    super(scope, id);

    // 리뷰 결과 저장 테이블 (기존 테이블)
    this.resultsTable = new dynamodb.Table(this, 'ReviewResultsTable', {
      tableName: 'PRReviewerResults',
      partitionKey: { name: 'execution_id', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'chunk_id', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      timeToLiveAttribute: 'ttl'
    });

    // 비용 최적화를 위한 TTL(Time-To-Live) 인덱스 활성화
    this.resultsTable.addLocalSecondaryIndex({
      indexName: 'ttl-index',
      sortKey: { name: 'ttl', type: dynamodb.AttributeType.NUMBER }
    });

    // 시간 기반 쿼리를 위한 글로벌 보조 인덱스 추가
    this.resultsTable.addGlobalSecondaryIndex({
      indexName: 'timestamp-index',
      partitionKey: { name: 'execution_id', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'timestamp', type: dynamodb.AttributeType.STRING },
      projectionType: dynamodb.ProjectionType.ALL
    });

    // PR 관련 정보 쿼리용 글로벌 보조 인덱스 - 개선된 버전
    this.resultsTable.addGlobalSecondaryIndex({
      indexName: 'repository-pr-index',
      partitionKey: { name: 'repository', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'pr_id', type: dynamodb.AttributeType.STRING },
      projectionType: dynamodb.ProjectionType.INCLUDE,
      nonKeyAttributes: [
        'execution_id', 
        'timestamp', 
        'severity', 
        'review_time',
        'results'
      ]
    });

    //PR별 시간순 리뷰 조회용 인덱스 추가 - 새로 추가된 인덱스
    this.resultsTable.addGlobalSecondaryIndex({
      indexName: 'pr-timeline-index',
      partitionKey: { name: 'pr_id', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'review_time', type: dynamodb.AttributeType.STRING },
      projectionType: dynamodb.ProjectionType.INCLUDE,
      nonKeyAttributes: [
        'execution_id',
        'repository',
        'severity',
        'pr_details'
      ]
    });

    // 새로운 마크다운 리포트 저장 테이블
    this.reportsTable = new dynamodb.Table(this, 'ReviewReportsTable', {
      tableName: 'PRReviewerReports',
      partitionKey: { name: 'repository', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'pr_id', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: cdk.RemovalPolicy.DESTROY
    });

    // 시간순 조회를 위한 GSI
    this.reportsTable.addGlobalSecondaryIndex({
      indexName: 'timestamp-index',
      partitionKey: { name: 'repository', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'timestamp', type: dynamodb.AttributeType.STRING },
      projectionType: dynamodb.ProjectionType.ALL
    });

    // 실행 ID로 조회하기 위한 GSI
    this.reportsTable.addGlobalSecondaryIndex({
      indexName: 'execution-id-index',
      partitionKey: { name: 'execution_id', type: dynamodb.AttributeType.STRING },
      projectionType: dynamodb.ProjectionType.ALL
    });

    // CloudFormation 출력 (기존 출력)
    new cdk.CfnOutput(this, 'ResultsTableName', {
      value: this.resultsTable.tableName,
      description: 'DynamoDB table for PR review results'
    });

    new cdk.CfnOutput(this, 'ResultsTableArn', {
      value: this.resultsTable.tableArn,
      description: 'ARN of the PR review results DynamoDB table'
    });

    // 새로운 리포트 테이블 CloudFormation 출력
    new cdk.CfnOutput(this, 'ReportsTableName', {
      value: this.reportsTable.tableName,
      description: 'DynamoDB table for PR review reports'
    });

    new cdk.CfnOutput(this, 'ReportsTableArn', {
      value: this.reportsTable.tableArn,
      description: 'ARN of the PR review reports DynamoDB table'
    });
  }
}