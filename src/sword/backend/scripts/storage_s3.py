import os
from pathlib import Path
import boto3

def s3():
    """
    Construct an S3 client from environment variables.
    Expected envs:
      - AWS_ACCESS_KEY_ID
      - AWS_SECRET_ACCESS_KEY
      - AWS_DEFAULT_REGION (optional, defaults to us-east-1)
    """
    return boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )

def upload(local: Path, bucket: str, key: str) -> str:
    """
    Upload a local file to s3://{bucket}/{key} and return the URL.
    """
    s3().upload_file(str(local), bucket, key)
    return f"s3://{bucket}/{key}"

def download(bucket: str, key: str, dest: Path):
    """
    Download s3://{bucket}/{key} to dest path and return dest.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    s3().download_file(bucket, key, str(dest))
    return dest

def presigned_url(bucket: str, key: str, expires=3600) -> str:
    """
    Return a presigned GET url valid for 'expires' seconds.
    """
    return s3().generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires
    )