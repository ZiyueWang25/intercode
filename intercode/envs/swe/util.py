import io
import os
import tarfile

from docker.models.containers import Container


def copy_to_container(container: Container, src: str, dst_dir: str):
    """src shall be an absolute path"""
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w|") as tar, open(src, "rb") as f:
        info = tar.gettarinfo(fileobj=f)
        info.name = os.path.basename(src)
        tar.addfile(info, f)

    container.put_archive(dst_dir, stream.getvalue())
