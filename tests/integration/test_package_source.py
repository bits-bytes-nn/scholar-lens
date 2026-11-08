"""A release's code read from an in-memory archive (no network), and the
product's distributions chosen by exact name."""

from __future__ import annotations

import io
import zipfile
from types import SimpleNamespace

import httpx
import pytest

from scholar_lens.adapters.packages.pypi import package_source
from scholar_lens.application.use_cases import guide as guide_writer
from scholar_lens.domain.guide.sdk_index import PackageSource


def _wheel(members: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for path, text in members.items():
            zf.writestr(path, text)
    return buffer.getvalue()


def _client(blob: bytes) -> httpx.Client:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/json"):
            return httpx.Response(
                200,
                json={
                    "info": {"version": "1.0"},
                    "urls": [
                        {
                            "filename": "gear-1.0-py3-none-any.whl",
                            "size": len(blob),
                            "url": "https://files.example/gear.whl",
                        }
                    ],
                },
            )
        return httpx.Response(200, content=blob)

    return httpx.Client(transport=httpx.MockTransport(handle))


WHEEL = {
    "gear/__init__.py": "A = 1\n",
    "gear/core.py": "B = 2\n",
    "space/part/__init__.py": "C = 3\n",
    "gear-1.0.dist-info/top_level.txt": "gear\nspace\n",
    "gear-1.0.dist-info/METADATA": "Name: gear\n",
}


def test_the_whole_source_is_complete_with_its_top_level_names() -> None:
    found = package_source("gear", client=_client(_wheel(WHEEL)))
    assert found is not None and found.complete and found.version == "1.0"
    assert set(found.files) == {
        "gear/__init__.py",
        "gear/core.py",
        "space/part/__init__.py",
    }
    assert found.top_levels == {"gear", "space"}


def test_a_source_over_the_size_cap_is_marked_partial() -> None:
    found = package_source("gear", client=_client(_wheel(WHEEL)), max_chars=8)
    assert found is not None and not found.complete
    assert set(found.files) == {"gear/__init__.py"}
    # The layout is still known past the cap.
    assert found.top_levels == {"gear", "space"}


def _source(top: str) -> PackageSource:
    return PackageSource("1.0", {f"{top}/__init__.py": "def run():\n    pass\n"})


@pytest.fixture
def fetched(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    names: list[str] = []
    tops = {"gear-kit": "gear", "gearbox": "gearbox", "gear": "gear"}

    def fake(name: str) -> PackageSource:
        names.append(name)
        return _source(tops.get(name, name.replace("-", "_")))

    _INDEX.source = fake
    return names


# The package index product_sdk reads (its source lookup set by ``fetched``).
_INDEX = SimpleNamespace(source=None, extras=lambda _name: None)


def test_a_distribution_of_the_products_name_is_matched_exactly(
    fetched: list[str],
) -> None:
    provided = {n: ("1.0", frozenset()) for n in ("gearbox", "Gear", "other")}
    index = guide_writer.product_sdk(provided, ["gear"], _INDEX)
    # "gearbox" starts with the name but is another distribution.
    assert fetched == ["Gear"] and index is not None


def test_names_compare_as_pep_503_normalises_them(fetched: list[str]) -> None:
    provided = {"Gear_Kit": ("1.0", frozenset())}
    assert guide_writer.product_sdk(provided, ["gear.kit"], _INDEX) is not None
    assert fetched == ["Gear_Kit"]


def test_without_a_distribution_of_the_name_its_import_package_decides(
    fetched: list[str],
) -> None:
    provided = {n: ("1.0", frozenset()) for n in ("gearbox", "gear-kit")}
    index = guide_writer.product_sdk(provided, ["gear"], _INDEX)
    assert index is not None and index.packages == {"gear"}
    assert (
        guide_writer.product_sdk({"gearbox": ("1.0", frozenset())}, ["gear"], _INDEX)
        is None
    )
