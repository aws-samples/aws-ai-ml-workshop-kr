


def get_s3_prefix_name(s3_path, verbose=True):
    file_name = s3_path.split('/')[-1]
    file_name = '/' + file_name
    desired_s3_uri = s3_path.split(file_name)[0]

    if verbose:
        print("file_name: ", file_name)
        print("desired_s3_uri: ", desired_s3_uri)
    return desired_s3_uri

from sagemaker.s3 import S3Uploader

def upload_data_s3(desired_s3_uri, file_name, verbose=True):
    # upload the model yaml file to s3
    
    file_s3_path = S3Uploader.upload(local_path=file_name, desired_s3_uri=desired_s3_uri)

    print(f"{file_name} is uploaded to:")
    print(file_s3_path)

    return file_s3_path
