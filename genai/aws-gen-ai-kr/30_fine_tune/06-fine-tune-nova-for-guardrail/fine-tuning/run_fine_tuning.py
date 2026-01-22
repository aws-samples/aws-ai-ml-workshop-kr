"""
Fine-tuning pipeline for Amazon Nova 2 Lite model.
"""
import json
import os
import time
from datetime import datetime
from typing import Optional

import boto3
from botocore.exceptions import ClientError


class S3Manager:
    """Manager for S3 operations."""

    def __init__(self, bucket_name: str, region: str = "us-east-1"):
        self.bucket_name = bucket_name
        self.region = region
        self.s3 = boto3.client("s3", region_name=self.region)

    def create_bucket_if_not_exists(self) -> bool:
        """Create S3 bucket if it doesn't exist."""
        try:
            self.s3.head_bucket(Bucket=self.bucket_name)
            print(f"✓ Bucket already exists: {self.bucket_name}")
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code == "404":
                try:
                    if self.region == "us-east-1":
                        self.s3.create_bucket(Bucket=self.bucket_name)
                    else:
                        self.s3.create_bucket(
                            Bucket=self.bucket_name,
                            CreateBucketConfiguration={"LocationConstraint": self.region}
                        )
                    print(f"✓ Created bucket: {self.bucket_name}")
                    return True
                except ClientError as create_error:
                    print(f"✗ Failed to create bucket: {create_error}")
                    return False
            else:
                print(f"✗ Error checking bucket: {e}")
                return False

    def upload_file(self, local_path: str, s3_key: str) -> str:
        """Upload a file to S3."""
        self.create_bucket_if_not_exists()
        try:
            self.s3.upload_file(local_path, self.bucket_name, s3_key)
            s3_uri = f"s3://{self.bucket_name}/{s3_key}"
            print(f"✓ Uploaded: {local_path} -> {s3_uri}")
            return s3_uri
        except ClientError as e:
            print(f"✗ Failed to upload file: {e}")
            raise

    def set_bucket_policy_for_bedrock(self) -> bool:
        """Set bucket policy to allow Bedrock access."""
        self.create_bucket_if_not_exists()

        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "bedrock.amazonaws.com"
                    },
                    "Action": [
                        "s3:GetObject",
                        "s3:PutObject",
                        "s3:ListBucket"
                    ],
                    "Resource": [
                        f"arn:aws:s3:::{self.bucket_name}",
                        f"arn:aws:s3:::{self.bucket_name}/*"
                    ]
                }
            ]
        }

        try:
            self.s3.put_bucket_policy(
                Bucket=self.bucket_name,
                Policy=json.dumps(policy)
            )
            print(f"✓ Set bucket policy for Bedrock access")
            return True
        except ClientError as e:
            print(f"✗ Failed to set bucket policy: {e}")
            return False


class BedrockManagerNova2Lite:
    """Manager for Amazon Bedrock fine-tuning operations for Nova 2 Lite."""

    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.bedrock = boto3.client("bedrock", region_name=self.region)
        # Nova 2 Lite with fine-tuning support (256k context)
        self.base_model_id = "amazon.nova-2-lite-v1:0:256k"
        self.custom_model_name = "guard-rail-nova-2-lite"

    def get_or_create_role_arn(self) -> str:
        """Get or create IAM role for Bedrock fine-tuning."""
        iam = boto3.client("iam", region_name=self.region)
        role_name = "BedrockFineTuningRole"

        try:
            response = iam.get_role(RoleName=role_name)
            role_arn = response["Role"]["Arn"]
            print(f"✓ Using existing role: {role_arn}")
            return role_arn
        except iam.exceptions.NoSuchEntityException:
            pass

        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "bedrock.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }

        print(f"Creating IAM role {role_name}...")

        response = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="IAM role for Amazon Bedrock fine-tuning jobs"
        )
        role_arn = response["Role"]["Arn"]

        # Wait for role to propagate
        print("Waiting for IAM role to propagate...")
        time.sleep(10)

        print(f"✓ Created IAM role: {role_arn}")
        return role_arn

    def create_fine_tuning_job(
        self,
        job_name: str,
        training_s3_uri: str,
        output_s3_uri: str,
        role_arn: str,
        hyperparameters: Optional[dict] = None
    ) -> dict:
        """Create a fine-tuning job for Nova 2 Lite model.

        Note: Nova 2.0 doesn't support validation sets.
        """
        default_hyperparameters = {
            "epochCount": "3",
            "batchSize": "1",
            "learningRate": "1e-5",
            "learningRateWarmupSteps": "0",
        }

        if hyperparameters:
            default_hyperparameters.update({k: str(v) for k, v in hyperparameters.items()})

        custom_model_name = f"{self.custom_model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        print(f"\n{'='*60}")
        print("Creating Fine-Tuning Job for Nova 2 Lite")
        print(f"{'='*60}")
        print(f"Job Name: {job_name}")
        print(f"Base Model: {self.base_model_id}")
        print(f"Custom Model Name: {custom_model_name}")
        print(f"Training Data: {training_s3_uri}")
        print(f"Output: {output_s3_uri}")
        print(f"Hyperparameters: {default_hyperparameters}")
        print("Note: Nova 2.0 doesn't support validation sets")
        print(f"{'='*60}\n")

        try:
            # Nova 2.0 doesn't support validationDataConfig
            response = self.bedrock.create_model_customization_job(
                jobName=job_name,
                customModelName=custom_model_name,
                roleArn=role_arn,
                baseModelIdentifier=self.base_model_id,
                customizationType="FINE_TUNING",
                trainingDataConfig={
                    "s3Uri": training_s3_uri
                },
                outputDataConfig={
                    "s3Uri": output_s3_uri
                },
                hyperParameters=default_hyperparameters
            )

            print(f"✓ Fine-tuning job created!")
            print(f"  Job ARN: {response.get('jobArn')}")

            return {
                "job_name": job_name,
                "job_arn": response.get("jobArn"),
                "custom_model_name": custom_model_name,
                "status": "SUBMITTED",
                "base_model_id": self.base_model_id,
                "hyperparameters": default_hyperparameters,
            }
        except ClientError as e:
            print(f"✗ Error creating fine-tuning job: {e}")
            raise

    def get_fine_tuning_job_status(self, job_name: str) -> dict:
        """Get status of a fine-tuning job."""
        try:
            response = self.bedrock.get_model_customization_job(
                jobIdentifier=job_name
            )
            return {
                "job_name": job_name,
                "status": response.get("status"),
                "creation_time": str(response.get("creationTime")),
                "end_time": str(response.get("endTime")) if response.get("endTime") else None,
                "base_model_arn": response.get("baseModelArn"),
                "output_model_arn": response.get("outputModelArn"),
                "custom_model_name": response.get("outputModelName"),
                "failure_message": response.get("failureMessage"),
                "training_metrics": response.get("trainingMetrics"),
                "validation_metrics": response.get("validationMetrics"),
            }
        except ClientError as e:
            print(f"✗ Error getting job status: {e}")
            raise

    def wait_for_job_completion(
        self,
        job_name: str,
        polling_interval: int = 60,
        timeout: int = 14400  # 4 hours
    ) -> dict:
        """Wait for fine-tuning job to complete."""
        start_time = time.time()
        print(f"\nMonitoring job: {job_name}")
        print(f"Polling every {polling_interval} seconds...")
        print("-" * 40)

        while True:
            status = self.get_fine_tuning_job_status(job_name)
            current_status = status.get("status")
            elapsed = int(time.time() - start_time)

            print(f"[{elapsed//60}m {elapsed%60}s] Status: {current_status}")

            if current_status in ["Completed", "COMPLETED"]:
                print("\n✓ Fine-tuning job completed successfully!")
                print(f"  Custom Model ARN: {status.get('output_model_arn')}")
                if status.get('training_metrics'):
                    print(f"  Training Metrics: {status.get('training_metrics')}")
                if status.get('validation_metrics'):
                    print(f"  Validation Metrics: {status.get('validation_metrics')}")
                return status

            elif current_status in ["Failed", "FAILED", "Stopped", "STOPPED"]:
                print(f"\n✗ Fine-tuning job failed!")
                print(f"  Reason: {status.get('failure_message')}")
                return status

            if elapsed > timeout:
                print(f"\n⚠ Timeout waiting for job completion")
                return status

            time.sleep(polling_interval)


def main():
    """Main fine-tuning pipeline for Nova 2 Lite."""
    print("\n" + "="*60)
    print("Amazon Nova 2 Lite Fine-Tuning Pipeline")
    print("Guard Rail Model Training")
    print("="*60 + "\n")

    # Configuration
    base_dir = os.path.dirname(os.path.dirname(__file__))
    training_file = os.path.join(base_dir, "data", "nova", "training_data.jsonl")
    s3_bucket_name = "guard-rail-fine-tuning-data"

    if not os.path.exists(training_file):
        print("✗ Training data not found.")
        print(f"  Expected: {training_file}")
        return

    # Initialize managers
    s3_manager = S3Manager(bucket_name=s3_bucket_name)
    bedrock_manager = BedrockManagerNova2Lite()

    # Step 1: Set up S3 bucket policy
    print("\n[Step 1/3] Setting up S3 bucket...")
    s3_manager.set_bucket_policy_for_bedrock()

    # Step 2: Upload training data with nova-2-lite prefix
    # Note: Nova 2.0 doesn't support validation sets
    print("\n[Step 2/3] Uploading training data to S3...")
    training_s3_uri = s3_manager.upload_file(training_file, "guard-rail-nova-2-lite/train/training_data.jsonl")
    output_s3_uri = f"s3://{s3_bucket_name}/guard-rail-nova-2-lite/output"

    # Step 3: Get/Create IAM role
    print("\n[Step 3/3] Setting up IAM role and creating fine-tuning job...")
    role_arn = bedrock_manager.get_or_create_role_arn()

    # Create fine-tuning job (no validation data for Nova 2.0)
    job_name = f"guard-rail-nova-2-lite-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    job_info = bedrock_manager.create_fine_tuning_job(
        job_name=job_name,
        training_s3_uri=training_s3_uri,
        output_s3_uri=output_s3_uri,
        role_arn=role_arn
    )

    # Save job info
    output_dir = os.path.join(base_dir, "evaluation", "results")
    os.makedirs(output_dir, exist_ok=True)
    job_info_file = os.path.join(output_dir, "nova_2_lite_job_info.json")
    with open(job_info_file, "w") as f:
        json.dump(job_info, f, indent=2, default=str)
    print(f"\n✓ Job info saved to {job_info_file}")

    # Monitor job
    print("\n" + "="*60)
    print("Monitoring Fine-Tuning Job")
    print("="*60)

    final_status = bedrock_manager.wait_for_job_completion(job_name)

    # Save final status
    final_status_file = os.path.join(output_dir, "nova_2_lite_final_status.json")
    with open(final_status_file, "w") as f:
        json.dump(final_status, f, indent=2, default=str)

    print(f"\n✓ Final status saved to {final_status_file}")
    print("\n" + "="*60)
    print("Fine-Tuning Pipeline Complete!")
    print("="*60 + "\n")

    if final_status.get("status") in ["Completed", "COMPLETED"]:
        print("Next steps:")
        print("1. Create a custom model deployment for on-demand inference")
        print("2. Or purchase Provisioned Throughput for production use")
        print(f"\nCustom Model: {final_status.get('custom_model_name')}")
        print(f"Model ARN: {final_status.get('output_model_arn')}")


if __name__ == "__main__":
    main()
