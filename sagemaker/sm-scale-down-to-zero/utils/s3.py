import os
import boto3
import logging
from sagemaker.s3 import S3Uploader
from botocore.exceptions import ClientError

class s3_handler():
    
    def __init__(self, region_name=None):
        
        self.region_name = region_name
        self.resource = boto3.resource('s3', region_name=self.region_name)
        self.client = boto3.client('s3', region_name=self.region_name)
        
        print (f"This is a S3 handler with [{self.region_name}] region.")
        
    def create_bucket(self, bucket_name):
        """Create an S3 bucket in a specified region

        If a region is not specified, the bucket is created in the S3 default
        region (us-east-1).

        :param bucket_name: Bucket to create
        :return: True if bucket created, else False
        """

        try:
            if self.region_name is None:
                self.client.create_bucket(Bucket=bucket_name)
            else:
                location = {'LocationConstraint': self.region_name}
                self.client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration=location
                )
            
            print (f"CREATE:[{bucket_name}] Bucket was created successfully")
            
        except ClientError as e:
            logging.error(e)
            print (f"ERROR: {e}")
            return False
        
        return True
    
    def copy_object(self, source_obj, target_bucket, target_obj):
        
        '''
        Copy S3 to S3
        '''
        
        try:
            response = self.client.copy_object(
                Bucket=target_bucket,#'destinationbucket',
                CopySource=source_obj,#'/sourcebucket/HappyFacejpg',
                Key=target_obj,#'HappyFaceCopyjpg',
            )
            
        except ClientError as e:
            logging.error(e)
            print (f"ERROR: {e}")
            return False
        
    def upload_dir(self, source_dir, target_bucket, target_dir):
        
        inputs = S3Uploader.upload(source_dir, "s3://{}/{}".format(target_bucket, target_dir))
        
        print (f"Upload:[{source_dir}] was uploaded to [{inputs}]successfully")
    
    def delete_bucket(self, bucket_name):
        
        try:
            self._delete_all_object(bucket_name=bucket_name)
            response = self.client.delete_bucket(
                Bucket=bucket_name,
            )
            
            print (f"DELETE: [{bucket_name}] Bucket was deleted successfully")
        
        except ClientError as e:
            logging.error(e)
            print (f"ERROR: {e}")
            return False
        
        return True
    
    def _delete_all_object(self, bucket_name):
        
        bucket = self.resource.Bucket(bucket_name)
        bucket.object_versions.delete() ## delete versioning
        bucket.objects.all().delete() ## delete all objects in the bucket
