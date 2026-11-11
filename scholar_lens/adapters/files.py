"""The :class:`~scholar_lens.application.ports.Files` adapter: a local path
or ``s3://<bucket>/<key>`` in the run's bucket."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from scholar_lens.application.ports import FileLocationError, ObjectStore

logger = logging.getLogger(__name__)

S3_SCHEME = "s3://"


class LocalOrS3Files:
    """Files on the local disk, and in the run's bucket when one is set."""

    def __init__(self, bucket: ObjectStore | None = None) -> None:
        self._bucket = bucket

    def _key(self, location: str) -> str | None:
        """The bucket key of an ``s3://`` location (None: a local path).

        Raises:
            FileLocationError: Another bucket, no bucket, or no key.
        """
        if not location.startswith(S3_SCHEME):
            return None
        name, _, key = location.removeprefix(S3_SCHEME).partition("/")
        if self._bucket is None or name != self._bucket.bucket_name or not key:
            raise FileLocationError(f"{location!r} is not in this run's S3 bucket.")
        return key

    def read_text(self, location: str) -> str | None:
        if (key := self._key(location)) is not None:
            assert self._bucket is not None
            return self._bucket.get_text(key)
        path = Path(location).expanduser()
        return path.read_text(encoding="utf-8") if path.is_file() else None

    def read_bytes(self, location: str) -> bytes | None:
        if (key := self._key(location)) is not None:
            assert self._bucket is not None
            return self._bucket.get_bytes(key)
        path = Path(location).expanduser()
        return path.read_bytes() if path.is_file() else None

    def write_text(self, location: str, text: str) -> None:
        if (key := self._key(location)) is not None:
            assert self._bucket is not None
            self._bucket.put_text(key, text)
            return
        path = Path(location).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def write_bytes(self, location: str, data: bytes) -> None:
        if self._key(location) is not None:
            raise FileLocationError(f"Bytes are written locally only: {location!r}.")
        path = Path(location).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def modified(self, location: str) -> datetime | None:
        if self._key(location) is not None:
            return None
        path = Path(location).expanduser()
        if not path.is_file():
            return None
        return datetime.fromtimestamp(path.stat().st_mtime, UTC)

    def make_dir(self, location: str) -> None:
        if self._key(location) is None:
            Path(location).expanduser().mkdir(parents=True, exist_ok=True)
