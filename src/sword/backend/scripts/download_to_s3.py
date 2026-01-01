import requests, tempfile, os
from pathlib import Path
from .storage_s3 import upload

def download_to_s3(url: str, bucket: str, key: str, max_mb: int = 200, chunk_size: int = 1024 * 512) -> tuple[bool, str]:
    """
    流式下载到临时文件，超过 max_mb 中止，然后上传到 S3 并删除临时文件。
    """
    tf = tempfile.NamedTemporaryFile(delete=False)
    tmp_path = Path(tf.name)
    tf.close()
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            size_hdr = r.headers.get("Content-Length")
            if size_hdr is not None and int(size_hdr) > max_mb * 1024 * 1024:
                return False, f"skip: content-length {int(size_hdr)/1e6:.1f} MB > max {max_mb} MB"
            written = 0
            with open(tmp_path, "wb") as fh:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if not chunk: continue
                    fh.write(chunk)
                    written += len(chunk)
                    if written > max_mb * 1024 * 1024:
                        return False, f"abort: wrote {written/1e6:.1f} MB > max {max_mb} MB"
        s3_url = upload(tmp_path, bucket, key)
        return True, f"ok: {s3_url} ({tmp_path.stat().st_size/1e6:.1f} MB)"
    except Exception as e:
        return False, f"error: {e}"
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass