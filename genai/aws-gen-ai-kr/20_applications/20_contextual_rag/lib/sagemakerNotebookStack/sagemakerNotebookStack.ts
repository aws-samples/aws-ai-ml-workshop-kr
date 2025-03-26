import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as sagemaker from 'aws-cdk-lib/aws-sagemaker';
import * as fs from 'fs';
import * as path from 'path';


export class SagemakerNotebookStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // The code that defines your stack goes here

    // IAM Role
    const SageMakerNotebookinstanceRole = new iam.Role(this, 'SageMakerNotebookInstanceRole', {
      assumedBy: new iam.ServicePrincipal('sagemaker.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonBedrockFullAccess'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonOpenSearchServiceFullAccess'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSageMakerFullAccess'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSSMFullAccess'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonS3FullAccess'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('SecretsManagerReadWrite')
      ],
    });


    // SageMaker Notebook Instance Lifecycle Configuration

    const onCreateScriptPath1 = path.join(__dirname, 'install_packages.sh')
    const onCreateScriptContent = fs.readFileSync(onCreateScriptPath1, 'utf-8')

    const cfnNotebookInstanceLifecycleConfig = new sagemaker.CfnNotebookInstanceLifecycleConfig(this, 'MyCfnNotebookInstanceLifecycleConfig', /* all optional props */ {
      notebookInstanceLifecycleConfigName: 'notebookInstanceLifecycleConfig',
      onCreate: [{
        content: cdk.Fn.base64(onCreateScriptContent),
      }],
      onStart: [],
    });


    // SageMaker Notebook Instance

    const cfnNotebookInstance = new sagemaker.CfnNotebookInstance(this, 'MyCfnNotebookInstance', {
      instanceType: 'ml.m5.xlarge',
      roleArn: SageMakerNotebookinstanceRole.roleArn,

      // the properties below are optional
      //acceleratorTypes: ['acceleratorTypes'],
      //additionalCodeRepositories: ['additionalCodeRepositories'],
      defaultCodeRepository: 'https://github.com/bae12-jo/contextual-rag-with-amazon-bedrock.git',
      directInternetAccess: 'Enabled',
      //instanceMetadataServiceConfiguration: {
      //  minimumInstanceMetadataServiceVersion: 'minimumInstanceMetadataServiceVersion',
      //},
      //kmsKeyId: 'kmsKeyId',
      lifecycleConfigName: 'notebookInstanceLifecycleConfig',
      notebookInstanceName: 'contextual-rag-workshop-instance',
      //platformIdentifier: 'platformIdentifier',
      //rootAccess: 'rootAccess',
      //securityGroupIds: ['securityGroupIds'],
      //subnetId: 'subnetId',
      //tags: [{
      //  key: 'key',
      //  value: 'value',
      //}],
      volumeSizeInGb: 10,
    });


  }
}