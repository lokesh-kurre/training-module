from __future__ import annotations

import glob
import io
import os
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from training.utils.logger import get_logger


LOGGER = get_logger("training.utils.io")

_AUTO_S3_CLIENT = threading.local()


def _is_s3_uri(uri: str) -> bool:
    return uri.startswith("s3://")


def _is_file_uri(uri: str) -> bool:
    return uri.startswith("file://")


def _file_uri_to_path(file_uri: str) -> str:
    parsed = urlparse(file_uri)
    return unquote(parsed.path)


def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    return bucket, prefix


def _resolve_s3_client(s3_client: Any = None, **kwargs: Any) -> Any:
    """Resolve S3 client with precedence:

    1) explicit `s3_client` arg
    2) kwargs['s3_client'] / kwargs['boto3_client']
    3) auto-created cached client (thread/process local)
    """
    if s3_client is not None:
        return s3_client

    if "s3_client" in kwargs and kwargs["s3_client"] is not None:
        return kwargs["s3_client"]

    if "boto3_client" in kwargs and kwargs["boto3_client"] is not None:
        return kwargs["boto3_client"]

    # Thread/process-local cached client to avoid repeated boto3.client("s3") creation.
    state = getattr(_AUTO_S3_CLIENT, "state", None)
    pid = os.getpid()
    if isinstance(state, dict) and state.get("pid") == pid and state.get("client") is not None:
        return state["client"]

    try:
        import boto3  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return None

    session = boto3.session.Session()
    client = session.client("s3")
    _AUTO_S3_CLIENT.state = {"pid": pid, "client": client}
    return client


def list_files(
    directory: str | None = None,
    /,
    recursive: bool = False,
    count: int = 1000,
    verbose: bool = False,
    s3_client: Any = None,
    **kwargs: Any,
) -> Iterator[str]:
    """List files from local/file URI paths or S3 prefixes.

    For local paths and file:// URIs, this uses glob-based listing.
    For s3:// URIs, this uses list_objects_v2 pagination and returns keys and prefixes.
    """
    max_count = max(0, int(count))
    if max_count == 0:
        return

    yielded = 0

    if directory is None:
        bucket = kwargs.get("Bucket")
        prefix = kwargs.get("Prefix", "")
        if not bucket:
            if verbose:
                LOGGER.warning("list_files: directory is None and no S3 Bucket was provided")
            return
        directory = f"s3://{bucket}/{prefix}" if prefix else f"s3://{bucket}"

    if _is_s3_uri(directory):
        bucket, prefix = _parse_s3_uri(directory)
        client = _resolve_s3_client(s3_client=s3_client, **kwargs)
        if client is None:
            if verbose:
                LOGGER.warning("list_files: boto3 is unavailable and no s3_client was provided")
            return

        paginator = client.get_paginator("list_objects_v2")
        params: dict[str, Any] = dict(kwargs)
        params.setdefault("Bucket", bucket)
        params.setdefault("Prefix", prefix)

        for page in paginator.paginate(**params):
            for item in page.get("CommonPrefixes", []):
                value = item.get("Prefix")
                if value:
                    yield f"s3://{params['Bucket']}/{value}"
                    yielded += 1
                    if yielded >= max_count:
                        return

            for item in page.get("Contents", []):
                key = item.get("Key")
                if key:
                    yield f"s3://{params['Bucket']}/{key}"
                    yielded += 1
                    if yielded >= max_count:
                        return

        return

    local_dir = _file_uri_to_path(directory) if _is_file_uri(directory) else directory
    local_dir = str(Path(local_dir).expanduser())

    if any(token in local_dir for token in ["*", "?", "["]):
        pattern = local_dir
    else:
        pattern = str(Path(local_dir) / ("**/*" if recursive else "*"))

    for path in glob.iglob(pattern, recursive=recursive):
        path_obj = Path(path)
        if path_obj.is_file():
            yield str(path_obj)
            yielded += 1
            if yielded >= max_count:
                return


def read_file(
    filepath: str | None = None,
    verbose: bool = False,
    s3_client: Any = None,
    as_stream: bool = False,
    **kwargs: Any,
) -> bytes | Any | None:
    """Read bytes from local/file URI/S3 path.

    If as_stream=True, returns a binary stream-like object.
    """
    target = filepath
    if target is None:
        bucket = kwargs.get("Bucket")
        key = kwargs.get("Key")
        if bucket and key:
            target = f"s3://{bucket}/{key}"

    if not target:
        if verbose:
            LOGGER.warning("read_file: no filepath (or S3 bucket/key) provided")
        return None

    if _is_s3_uri(target):
        bucket, key = _parse_s3_uri(target)
        client = _resolve_s3_client(s3_client=s3_client, **kwargs)
        if client is None:
            if verbose:
                LOGGER.warning("read_file: boto3 is unavailable and no s3_client was provided")
            return None

        params: dict[str, Any] = dict(kwargs)
        params.setdefault("Bucket", bucket)
        params.setdefault("Key", key)
        response = client.get_object(**params)
        body = response.get("Body")
        if body is None:
            return None

        if as_stream:
            return body

        return body.read()

    local_path = _file_uri_to_path(target) if _is_file_uri(target) else target
    path = Path(local_path).expanduser()

    try:
        data = path.read_bytes()
    except (FileNotFoundError, PermissionError, OSError) as exc:
        if verbose:
            LOGGER.warning("read_file: failed to read '%s': %s", path, exc)
        return None

    if as_stream:
        return io.BytesIO(data)

    return data


def read_binary(
    filepath: str | None = None,
    verbose: bool = False,
    s3_client: Any = None,
    as_stream: bool = False,
    **kwargs: Any,
) -> bytes | Any | None:
    """Alias for read_file for explicit binary reads."""
    return read_file(
        filepath=filepath,
        verbose=verbose,
        s3_client=s3_client,
        as_stream=as_stream,
        **kwargs,
    )


def read_image(
    filepath: str | None = None,
    as_pillow: bool = False,
    cv2_imdecode_mode: int = -1,
    verbose: bool = False,
    s3_client: Any = None,
    **kwargs: Any,
) -> Any | None:
    """Read and decode image from local/file URI/S3 using PIL or OpenCV."""
    stream = read_file(
        filepath=filepath,
        verbose=verbose,
        s3_client=s3_client,
        as_stream=True,
        **kwargs,
    )
    if stream is None:
        return None

    if isinstance(stream, (bytes, bytearray, memoryview)):
        data = bytes(stream)
    elif hasattr(stream, "read"):
        data = stream.read()
    else:
        data = None

    if data is None:
        return None

    if as_pillow:
        try:
            from PIL import Image
        except ModuleNotFoundError:
            if verbose:
                LOGGER.warning("read_image: Pillow is not installed")
            return None

        try:
            return Image.open(io.BytesIO(data))
        except Exception as exc:
            if verbose:
                LOGGER.warning("read_image: Pillow decode failed: %s", exc)
            return None

    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np
    except ModuleNotFoundError:
        if verbose:
            LOGGER.warning("read_image: OpenCV or numpy is not installed")
        return None

    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2_imdecode_mode)
    if image is None and verbose:
        LOGGER.warning("read_image: OpenCV decode failed")
    return image
