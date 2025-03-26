import { Duration, Stack, StackProps, SecretValue } from "aws-cdk-lib";
import { Construct } from "constructs";

import * as fs from "fs";
import * as cdk from "aws-cdk-lib";
import * as opensearch from "aws-cdk-lib/aws-opensearchservice";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as ssm from "aws-cdk-lib/aws-ssm";
import { PolicyStatement, AnyPrincipal } from "aws-cdk-lib/aws-iam";
import * as secretsmanager from "aws-cdk-lib/aws-secretsmanager";

export class OpensearchStack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    const domainName = `contextual-rag-domain`;

    const opensearch_user_id = "raguser";

    const user_id_pm = new ssm.StringParameter(this, "opensearch_user_id", {
      parameterName: "opensearch_user_id",
      stringValue: "raguser",
    });

    const opensearch_user_password = "pwkey";

    const secret = new secretsmanager.Secret(this, "domain-creds", {
      generateSecretString: {
        secretStringTemplate: JSON.stringify({
          "es.net.http.auth.user": opensearch_user_id,
        }),
        generateStringKey: opensearch_user_password,
        excludeCharacters: '"\'',
      },
      secretName: "opensearch_user_password",
    });

    const domain = new opensearch.Domain(this, "Domain", {
      version: opensearch.EngineVersion.OPENSEARCH_2_11,
      domainName: domainName,
      capacity: {
        masterNodes: 2,
        multiAzWithStandbyEnabled: false,
      },
      ebs: {
        volumeSize: 100,
        volumeType: ec2.EbsDeviceVolumeType.GP3,
        enabled: true,
      },
      enforceHttps: true,
      nodeToNodeEncryption: true,
      encryptionAtRest: { enabled: true },
      fineGrainedAccessControl: {
        masterUserName: opensearch_user_id,
        masterUserPassword: secret.secretValueFromJson(
          opensearch_user_password
        ),
      },
    });

    domain.addAccessPolicies(
      new PolicyStatement({
        actions: ["es:*"],
        principals: [new AnyPrincipal()],
        resources: [domain.domainArn + "/*"],
      })
    );

    const domain_endpoint_pm = new ssm.StringParameter(
      this,
      "opensearch_domain_endpoint",
      {
        parameterName: "opensearch_domain_endpoint",
        stringValue: domain.domainEndpoint,
      }
    );

    new cdk.CfnOutput(this, "OpensearchDomainEndpoint", {
      value: domain.domainEndpoint,
      description: "OpenSearch Domain Endpoint",
    });

    new cdk.CfnOutput(this, "parameter store user id", {
      value: user_id_pm.parameterArn,
      description: "parameter store user id",
    });

    new cdk.CfnOutput(this, "secrets manager user pw", {
      value: secret.secretName,
      description: "secrets manager user pw",
    });
    
    new cdk.CfnOutput(this, 'DomainArn', {
      value: domain.domainArn,
      exportName: 'DomainArn'
    });
    
  }
}