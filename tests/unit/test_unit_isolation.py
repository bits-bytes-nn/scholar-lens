"""The unit tests run on the domain, the application and fake ports only
(docs/engine.md, 6): no unit test module imports an adapter, the sections
engine, the configuration or an entrypoint, directly or through a shared
test module (``tests.fakes``, another test file)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
UNIT = ROOT / "tests" / "unit"
FORBIDDEN = (
    "scholar_lens.adapters",
    "scholar_lens.sections_engine",
    "scholar_lens.configs",
    "scholar_lens.bootstrap",
    "scholar_lens.entrypoints",
    "scholar_lens.cli",
)


def _imports(path: Path) -> set[str]:
    """Every module a file imports, at any depth of its code."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found.add(node.module)
            # ``from tests import fakes`` imports the module ``tests.fakes``.
            found |= {f"{node.module}.{alias.name}" for alias in node.names}
    return found


def _test_module(name: str) -> Path | None:
    """The file of a ``tests.*`` module, None for any other module."""
    if name != "tests" and not name.startswith("tests."):
        return None
    base = ROOT.joinpath(*name.split("."))
    for path in (base.with_suffix(".py"), base / "__init__.py"):
        if path.is_file():
            return path
    return None


def _reached(path: Path) -> dict[str, list[str]]:
    """Module -> the chain of test files through which ``path`` imports it."""
    reached: dict[str, list[str]] = {}
    seen: set[Path] = set()
    stack = [(path, [path.relative_to(ROOT).as_posix()])]
    while stack:
        current, chain = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        for name in _imports(current):
            reached.setdefault(name, chain)
            if (module := _test_module(name)) is not None:
                stack.append((module, [*chain, module.relative_to(ROOT).as_posix()]))
    return reached


UNIT_FILES = sorted(p for p in UNIT.glob("*.py") if p.name != "__init__.py")


@pytest.mark.parametrize("path", UNIT_FILES, ids=lambda p: p.name)
def test_a_unit_test_module_imports_no_adapter(path: Path) -> None:
    bad = {
        name: chain
        for name, chain in _reached(path).items()
        if any(name == f or name.startswith(f + ".") for f in FORBIDDEN)
    }
    assert not bad, bad


def test_the_check_sees_an_import_through_a_shared_test_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check follows ``tests.*`` imports: a fake that imports an adapter
    is found from the unit test that imports the fake."""
    root = tmp_path
    (root / "tests" / "unit").mkdir(parents=True)
    (root / "tests" / "__init__.py").write_text("")
    (root / "tests" / "shared.py").write_text(
        "def helper():\n    from scholar_lens.adapters.x import y\n"
    )
    test = root / "tests" / "unit" / "test_x.py"
    test.write_text("from tests.shared import helper\n")
    monkeypatch.setattr(__import__(__name__, fromlist=["_"]), "ROOT", root)
    reached = _reached(test)
    assert reached["scholar_lens.adapters.x"] == [
        "tests/unit/test_x.py",
        "tests/shared.py",
    ]
