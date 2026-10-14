"""Import contracts between the pipeline modules, read from the source (AST):
the survey stages and the shared section loop stay independent of the guide
and review generators, and the web research layer of the review plan."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "scholar_lens" / "src"
PACKAGE = "scholar_lens.src"


def _module_name(path: Path) -> str:
    parts = path.relative_to(SRC).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join((PACKAGE, *parts))


def _imports(source: str, package: str) -> set[str]:
    """Every module ``source`` (in ``package``) imports, top-level or inside a
    function, as an absolute name; ``from . import x`` counts as importing
    ``x``."""
    parts = package.split(".")
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = parts[: len(parts) - node.level + 1]
                module = ".".join([*base, *([node.module] if node.module else [])])
            else:
                module = node.module or ""
            found.add(module)
            found.update(f"{module}.{alias.name}" for alias in node.names)
    return found


def imported_modules(path: Path) -> set[str]:
    """The modules the source file ``path`` imports (:func:`_imports`)."""
    name = _module_name(path)
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    return _imports(path.read_text(encoding="utf-8"), package)


def _modules(pattern: str) -> list[Path]:
    paths = sorted(SRC.glob(pattern))
    assert paths, pattern
    return paths


# (modules, the modules none of them may import)
CONTRACTS = [
    ("survey_*.py", ("tech_guide", "explainer")),
    ("web_research.py", ("review_plan",)),
    ("section_loop.py", ("tech_guide", "explainer", "survey_*")),
]


@pytest.mark.parametrize(
    ("path", "forbidden"),
    [
        pytest.param(path, forbidden, id=path.stem)
        for pattern, forbidden in CONTRACTS
        for path in _modules(pattern)
    ],
)
def test_import_contract(path: Path, forbidden: tuple[str, ...]) -> None:
    banned = {
        _module_name(target) for name in forbidden for target in _modules(f"{name}.py")
    }
    assert not imported_modules(path) & banned


def test_imports_resolve_every_import_form() -> None:
    # The contracts are only as good as the resolution.
    source = (
        "from .tech_guide import X\n"
        "from . import explainer\n"
        "from ..src.survey_plan import Y\n"
        "import scholar_lens.src.review_plan\n"
        "def f():\n"
        "    from .prompts.base import Z\n"
    )
    imports = _imports(source, PACKAGE)
    for module in ("tech_guide", "explainer", "survey_plan", "review_plan"):
        assert f"{PACKAGE}.{module}" in imports
    assert f"{PACKAGE}.prompts.base" in imports
    assert f"{PACKAGE}.prompts.tech_guide" not in imports
