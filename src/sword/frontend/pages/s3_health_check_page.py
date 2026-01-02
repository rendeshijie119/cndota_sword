import os
from pathlib import Path
import io
import time

import streamlit as st
import pandas as pd
import boto3
import s3fs

def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return str(v) if v is not None else default

def s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=_env("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=_env("AWS_SECRET_ACCESS_KEY"),
        region_name=_env("AWS_DEFAULT_REGION", "us-east-1"),
    )

def s3_fs():
    return s3fs.S3FileSystem(
        key=_env("AWS_ACCESS_KEY_ID"),
        secret=_env("AWS_SECRET_ACCESS_KEY"),
        client_kwargs={"region_name": _env("AWS_DEFAULT_REGION", "us-east-1")},
    )

def page():
    st.title("S3 Health Check")

    bucket = _env("S3_BUCKET")
    prefix = _env("S3_PREFIX", "")
    wards_source = _env("WARDS_SOURCE", "local")

    st.write("Config")
    st.write(f"S3_BUCKET = {bucket}")
    st.write(f"S3_PREFIX = {prefix}")
    st.write(f"WARDS_SOURCE = {wards_source}")

    if not bucket:
        st.error("Missing S3_BUCKET. Ensure secrets are set correctly.")
        return

    st.markdown("---")
    st.subheader("List objects under prefix")

    cli = s3_client()
    base_prefix = prefix
    if not base_prefix.endswith("/") and base_prefix:
        base_prefix += "/"

    try:
        resp = cli.list_objects_v2(Bucket=bucket, Prefix=base_prefix, MaxKeys=50)
        count = resp.get("KeyCount", 0)
        st.write(f"Found {count} objects under '{base_prefix}'")
        objs = resp.get("Contents", []) or []
        if objs:
            df = pd.DataFrame([{"key": o.get("Key"), "size": o.get("Size"), "last_modified": o.get("LastModified")} for o in objs])
            st.dataframe(df, width="stretch")
        else:
            st.info("No objects yet. Try uploading a test file below.")
    except Exception as e:
        st.error(f"list_objects_v2 failed: {e}")
        return

    st.markdown("---")
    st.subheader("Upload a small test CSV")

    team = st.text_input("Team name for key path", value="Team Falcons")
    if st.button("Upload test"):
        # create a tiny CSV in memory
        df = pd.DataFrame({"match_id": ["test"], "value": [42]})
        csv_bytes = df.to_csv(index=False).encode("utf-8")

        key = f"{base_prefix}obs_logs/teams/{''.join(c if c.isalnum() or c in '-_ .' else '_' for c in team)}/test_{int(time.time())}.csv"
        try:
            cli.put_object(Bucket=bucket, Key=key, Body=csv_bytes)
            st.success(f"Uploaded: s3://{bucket}/{key}")
            # generate presigned URL
            url = cli.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=3600)
            st.write("Presigned URL (1h):")
            st.write(url)
            # read back via s3fs
            fs = s3_fs()
            with fs.open(f"s3://{bucket}/{key}", "rb") as f:
                df_back = pd.read_csv(f)
            st.write("Read-back preview:")
            st.dataframe(df_back, width="stretch")
        except Exception as e:
            st.error(f"Upload/read failed: {e}")

if __name__ == "__main__":
    page()