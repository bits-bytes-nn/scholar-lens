"""The :class:`Files` port: one adapter resolves a local path or an ``s3://``
location, and the application reaches files only through it."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import scholar_lens.application as application
import scholar_lens.domain as domain
from scholar_lens.adapters.files import LocalOrS3Files
from scholar_lens.application.ports import FileLocationError

# Calls that touch the disk directly.
_FILE_IO = {
    "read_text",
    "write_text",
    "read_bytes",
    "write_bytes",
    "mkdir",
    "rename",
    "unlink",
    "rmdir",
    "is_file",
    "is_dir",
    "exists",
    "stat",
    "iterdir",
    "glob",
}


class _Bucket:
    bucket_name = "bucket"

    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}

    def get_text(self, key: str) -> str | None:
        data = self.blobs.get(key)
        return None if data is None else data.decode("utf-8")

    def put_text(self, key: str, text: str) -> None:
        self.blobs[key] = text.encode("utf-8")

    def get_bytes(self, key: str) -> bytes | None:
        return self.blobs.get(key)


@pytest.mark.parametrize("text", ["", "{}", "한글 본문\n"])
def test_text_round_trips_locally_and_in_the_bucket(tmp_path: Path, text: str) -> None:
    bucket = _Bucket()
    files = LocalOrS3Files(bucket)
    for location in (str(tmp_path / "a" / "b.txt"), "s3://bucket/a/b.txt"):
        assert files.read_text(location) is None
        files.write_text(location, text)
        assert files.read_text(location) == text
    assert files.modified(str(tmp_path / "a" / "b.txt")) is not None
    assert files.modified("s3://bucket/a/b.txt") is None


@pytest.mark.parametrize(
    "location", ["s3://other/a.txt", "s3://bucket/", "s3://bucket"]
)
@pytest.mark.parametrize("with_bucket", [True, False])
def test_a_location_outside_the_runs_bucket_is_refused(
    location: str, with_bucket: bool
) -> None:
    files = LocalOrS3Files(_Bucket() if with_bucket else None)
    with pytest.raises(FileLocationError):
        files.read_text(location)
    with pytest.raises(FileLocationError):
        files.write_text(location, "x")


def test_bytes_are_read_from_the_bucket_and_written_locally(tmp_path: Path) -> None:
    bucket = _Bucket()
    bucket.blobs["img/a.png"] = b"\x89PNG"
    files = LocalOrS3Files(bucket)
    data = files.read_bytes("s3://bucket/img/a.png")
    assert data == b"\x89PNG"
    files.write_bytes(str(tmp_path / "staged" / "a.png"), data)
    assert (tmp_path / "staged" / "a.png").read_bytes() == data
    with pytest.raises(FileLocationError):
        files.write_bytes("s3://bucket/img/b.png", data)
    files.make_dir(str(tmp_path / "x" / "y"))
    files.make_dir("s3://bucket/prefix")
    assert (tmp_path / "x" / "y").is_dir()


def _direct_file_io(path: Path) -> list[str]:
    """Calls of ``path`` that touch the disk without the Files port."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "open":
            out.append(f"{path.name}:{node.lineno} open()")
        if isinstance(func, ast.Attribute) and func.attr in _FILE_IO:
            receiver = ast.unparse(func.value)
            # The port itself, and string methods that share a name.
            if receiver.endswith("files") or receiver.startswith(("'", '"')):
                continue
            out.append(f"{path.name}:{node.lineno} {receiver}.{func.attr}()")
    return out


@pytest.mark.parametrize("layer", [application, domain], ids=["application", "domain"])
def test_the_inner_layers_reach_files_only_through_the_port(layer: object) -> None:
    # The domain reads no file at all (the conventions, too, are passed in);
    # the application only through Files.
    root = Path(layer.__file__).parent  # type: ignore[attr-defined]
    found = [c for p in sorted(root.rglob("*.py")) for c in _direct_file_io(p)]
    if layer is domain:
        found += [
            f"{p.name}: __file__"
            for p in sorted(root.rglob("*.py"))
            if "Path(__file__" in p.read_text(encoding="utf-8")
        ]
    assert found == []
