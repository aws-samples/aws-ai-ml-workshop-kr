#!/usr/bin/env node
import { App } from 'aws-cdk-lib';
import { SagemakerNotebookStack } from "../lib/sagemakerNotebookStack/sagemakerNotebookStack";
import { OpensearchStack } from "../lib/openSearchStack";

const STACK_PREFIX = "ContextualRetrieval";
const DEFAULT_REGION = "us-west-2";
const envSetting = {
  env: {
    account: process.env.CDK_DEPLOY_ACCOUNT || process.env.CDK_DEFAULT_ACCOUNT,
    region: DEFAULT_REGION,
  },
};

const app = new App();

// Deploy Sagemaker stack
const sagemakerNotebookStack = new SagemakerNotebookStack(app, `${STACK_PREFIX}-SagemakerNotebookStack`, envSetting);

// Deploy OpenSearch stack
const opensearchStack = new OpensearchStack(app, `${STACK_PREFIX}-OpensearchStack`, envSetting);
opensearchStack.addDependency(sagemakerNotebookStack);

app.synth();


// new ContextualRetrievalStack(app, 'ContextualRetrievalStack', {
//   /* If you don't specify 'env', this stack will be environment-agnostic.
//    * Account/Region-dependent features and context lookups will not work,
//    * but a single synthesized template can be deployed anywhere. */

//   /* Uncomment the next line to specialize this stack for the AWS Account
//    * and Region that are implied by the current CLI configuration. */
//   // env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION },

//   /* Uncomment the next line if you know exactly what Account and Region you
//    * want to deploy the stack to. */
//   // env: { account: '123456789012', region: 'us-east-1' },

//   /* For more information, see https://docs.aws.amazon.com/cdk/latest/guide/environments.html */
// });