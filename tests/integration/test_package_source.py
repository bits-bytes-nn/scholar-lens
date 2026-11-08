"""A release's code read from an in-memory archive (no network), the run's
target release read instead of the latest, and the product's distributions
chosen by exact name."""

from __future__ import annotations

import io
import zipfile
from types import SimpleNamespace

import httpx
import pytest

from scholar_lens.adapters.packages.pypi import package_extras, package_source
from scholar_lens.application.use_cases import guide as guide_writer
from scholar_lens.domain.guide.sdk_index import PackageSource
from scholar_lens.domain.sources import ReleaseUnavailable


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

    def fake(name: str, release: str | None = None) -> PackageSource:
        names.append(name)
        return _source(tops.get(name, name.replace("-", "_")))

    _INDEX.source = fake
    return names


# The package index product_sdk reads (its source lookup set by ``fetched``).
_INDEX = SimpleNamespace(source=None, extras=lambda _name, _release=None: None)


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


# ---- the run's target release ---------------------------------------------------


def _release_client(
    releases: dict[str, bytes | None], latest: str, asked: list[str]
) -> httpx.Client:
    """PyPI with ``releases`` (version -> wheel, None: no files) of gear."""

    def entry(version: str) -> dict[str, object]:
        blob = releases[version]
        return {
            "info": {"version": version, "provides_extra": [f"x{version}"]},
            "urls": (
                []
                if blob is None
                else [
                    {
                        "filename": f"gear-{version}-py3-none-any.whl",
                        "size": len(blob),
                        "url": f"https://files.example/gear-{version}.whl",
                    }
                ]
            ),
        }

    def handle(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        asked.append(path)
        if path.startswith("/gear-"):
            version = path.removeprefix("/gear-").removesuffix(".whl")
            return httpx.Response(200, content=releases[version] or b"")
        parts = path.strip("/").split("/")  # pypi, gear, [version], json
        if parts[1] != "gear":
            return httpx.Response(404)
        if len(parts) == 3:
            return httpx.Response(
                200, json={**entry(latest), "releases": {v: [] for v in releases}}
            )
        if parts[2] not in releases:
            return httpx.Response(404)
        return httpx.Response(200, json=entry(parts[2]))

    return httpx.Client(transport=httpx.MockTransport(handle))


def _gear(version: str) -> bytes:
    return _wheel({"gear/__init__.py": f"VERSION = {version!r}\n"})


RELEASES = {v: _gear(v) for v in ("0.6.1", "0.6.3", "0.6.4rc1", "0.7.0", "1.0")}


@pytest.mark.parametrize(
    ("release", "used"),
    [
        (None, "1.0"),
        ("0.7.0", "0.7.0"),
        ("v0.7.0", "0.7.0"),
        # A docs series is its newest final release.
        ("0.6", "0.6.3"),
    ],
)
def test_the_target_release_is_read_not_the_latest(
    release: str | None, used: str
) -> None:
    asked: list[str] = []
    client = _release_client(RELEASES, "1.0", asked)
    found = package_source("gear", release=release, client=client)
    assert found is not None and found.version == used
    assert found.files["gear/__init__.py"] == f"VERSION = {used!r}\n"
    assert asked[-1] == f"/gear-{used}.whl"
    assert package_extras("gear", release=release, client=client) == (
        used,
        frozenset({f"x{used}"}),
    )
    if release:
        assert f"/pypi/gear/{used}/json" in asked


@pytest.mark.parametrize("release", ["0.5", "2.0.0"])
def test_a_release_the_index_lacks_is_unavailable_not_the_latest(
    release: str,
) -> None:
    client = _release_client(RELEASES, "1.0", [])
    with pytest.raises(ReleaseUnavailable):
        package_source("gear", release=release, client=client)
    with pytest.raises(ReleaseUnavailable):
        package_extras("gear", release=release, client=client)
    # An unknown package is still no answer, not a missing release.
    assert package_source("other", release=release, client=client) is None


def test_a_release_without_readable_files_is_unavailable() -> None:
    client = _release_client({"0.9": None, "1.0": _gear("1.0")}, "1.0", [])
    with pytest.raises(ReleaseUnavailable):
        package_source("gear", release="0.9", client=client)
    # Its extras are still the release's.
    assert package_extras("gear", release="0.9", client=client) == (
        "0.9",
        frozenset({"x0.9"}),
    )


class _ReleaseIndex:
    """A package index with gear at 0.9 and 1.0 (the latest); a release
    outside ``releases`` is unavailable."""

    def __init__(self, releases: tuple[str, ...] = ("0.9", "1.0")) -> None:
        self.releases = releases
        self.asked: list[tuple[str, str, str | None]] = []

    def _at(self, name: str, release: str | None) -> str:
        if release is not None and release not in self.releases:
            raise ReleaseUnavailable(f"no {name} {release}")
        return release or "1.0"

    def extras(
        self, name: str, release: str | None = None
    ) -> tuple[str, frozenset[str]] | None:
        self.asked.append(("extras", name, release))
        version = self._at(name, release)
        return version, frozenset({f"x{version}"})

    def source(self, name: str, release: str | None = None) -> PackageSource | None:
        self.asked.append(("source", name, release))
        version = self._at(name, release)
        return PackageSource(version, {"gear/__init__.py": "def run():\n    pass\n"})


def test_the_products_code_and_extras_are_read_at_the_target_release() -> None:
    index = _ReleaseIndex()
    latest = {"gear": ("1.0", frozenset({"x1.0"})), "other": ("3.0", frozenset())}
    notes: list[str] = []
    sdk = guide_writer.product_sdk(latest, ["gear"], index, "0.9", notes)
    assert sdk is not None and sdk.version == "gear 0.9"
    provided = guide_writer.release_packages(latest, ["gear"], "0.9", index, notes)
    # The product's distribution at the release, the others as they were.
    assert provided == {
        "gear": ("0.9", frozenset({"x0.9"})),
        "other": ("3.0", frozenset()),
    }
    assert ("source", "gear", None) not in index.asked and notes == []


def test_a_target_release_the_index_lacks_skips_the_checks_with_a_note() -> None:
    index = _ReleaseIndex(releases=("1.0",))
    latest = {"gear": ("1.0", frozenset({"x1.0"}))}
    notes: list[str] = []
    assert guide_writer.product_sdk(latest, ["gear"], index, "0.9", notes) is None
    assert guide_writer.release_packages(latest, ["gear"], "0.9", index, notes) == {}
    # Never the latest release in its place.
    assert ("source", "gear", None) not in index.asked
    assert len(notes) == 2 and all("gear 0.9" in n for n in notes)


class _UnreachableAtRelease(_ReleaseIndex):
    """The index answers for the latest release, not for the target (an
    outage or an unknown release reported as None by the port)."""

    def extras(
        self, name: str, release: str | None = None
    ) -> tuple[str, frozenset[str]] | None:
        self.asked.append(("extras", name, release))
        return None if release else ("1.0", frozenset({"x1.0"}))

    def source(self, name: str, release: str | None = None) -> PackageSource | None:
        self.asked.append(("source", name, release))
        if release:
            return None
        return PackageSource("1.0", {"gear/__init__.py": "def run():\n    pass\n"})


def test_a_target_release_the_index_cannot_answer_is_noted_never_replaced() -> None:
    index = _UnreachableAtRelease()
    latest = {"gear": ("1.0", frozenset({"x1.0"})), "other": ("3.0", frozenset())}
    notes: list[str] = []
    assert guide_writer.product_sdk(latest, ["gear"], index, "0.9", notes) is None
    provided = guide_writer.release_packages(latest, ["gear"], "0.9", index, notes)
    # The latest release's extras never stand in for the target's.
    assert provided == {"other": ("3.0", frozenset())}
    assert len(notes) == 2 and all("gear 0.9" in n for n in notes)
