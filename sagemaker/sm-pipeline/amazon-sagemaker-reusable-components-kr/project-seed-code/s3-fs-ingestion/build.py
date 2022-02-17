# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import argparse
import json
import logging
import os
from pipeline.pipeline import create_pipeline
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", type=str, default=os.environ.get("LOGLEVEL", "INFO").upper())
    parser.add_argument("--sagemaker-project-id", type=str, required=True)
    parser.add_argument("--sagemaker-project-name", type=str, required=True)
    parser.add_argument("--pipeline-description", type=str, default="automated ingestion from s3 to feature store")
    parser.add_argument("--pipeline-name-prefix", type=str, default="s3-fs-ingest-pipeline")
    parser.add_argument("--dw-flow-url", type=str, required=True)
    parser.add_argument("--dw-flow-output-name", type=str, required=True)
    parser.add_argument("--s3-data-prefix", type=str, required=True)
    parser.add_argument("--feature-group-name", type=str, required=True)
    parser.add_argument("--execution-role", type=str, default="")

    args, _ = parser.parse_known_args()

    # Configure logging to output the line number and message
    log_format = "%(levelname)s: [%(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(format=log_format, level=args.log_level)

    pipeline = create_pipeline(
            pipeline_name=f"{args.pipeline_name_prefix}-{args.sagemaker_project_id}",
            pipeline_description=args.pipeline_description,
            project_id=args.sagemaker_project_id,
            project_name=args.sagemaker_project_name,
            data_wrangler_flow_s3_url=args.dw_flow_url,
            flow_output_name=args.dw_flow_output_name,
            input_data_s3_url=f"s3://{args.s3_data_prefix}",
            feature_group_name=args.feature_group_name,
            execution_role=args.execution_role,
    )

    logger.info(f"pipeline created:")
    logger.info(f"{json.dumps(json.loads(pipeline.definition()), indent=2, sort_keys=True)}")