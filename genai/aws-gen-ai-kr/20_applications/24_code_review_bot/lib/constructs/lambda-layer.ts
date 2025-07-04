import * as cdk from 'aws-cdk-lib';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import { Construct } from 'constructs';

export class LambdaLayers extends Construct {
  public readonly requestsLayer: lambda.LayerVersion;
  public readonly networkxLayer: lambda.LayerVersion;

  constructor(scope: Construct, id: string) {
    super(scope, id);

    // Create requests layer
    this.requestsLayer = new lambda.LayerVersion(this, 'RequestsLayer', {
      code: lambda.Code.fromAsset('layer/requests'),
      compatibleRuntimes: [lambda.Runtime.PYTHON_3_12],
      description: 'Layer containing requests library for PR ReviewBot',
      layerVersionName: 'pr-reviewer-requests',
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      license: 'Apache-2.0'
    });

    // Create networkx layer
    this.networkxLayer = new lambda.LayerVersion(this, 'NetworkxLayer', {
      code: lambda.Code.fromAsset('layer/networkx'),
      compatibleRuntimes: [lambda.Runtime.PYTHON_3_12],
      description: 'Layer containing networkx and numpy libraries for PR ReviewBot',
      layerVersionName: 'pr-reviewer-networkx',
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      license: 'BSD-3-Clause'
    });

    // Add CloudFormation outputs
    this.addOutputs();
  }

  private addOutputs() {
    // Requests layer outputs
    new cdk.CfnOutput(this, 'RequestsLayerArn', {
      value: this.requestsLayer.layerVersionArn,
      description: 'ARN of the Requests Lambda Layer'
    });

    // Networkx layer outputs
    new cdk.CfnOutput(this, 'NetworkxLayerArn', {
      value: this.networkxLayer.layerVersionArn,
      description: 'ARN of the Networkx Lambda Layer'
    });
  }
}
