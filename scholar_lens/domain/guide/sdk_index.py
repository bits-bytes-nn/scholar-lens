"""What a guide's subject package really defines, read from its source (never
run): its modules, the names each module binds, and the parameters of its
functions and classes. A guide's Python code is checked against it (an
import path or keyword argument the package does not have fails as pasted),
and the definitions a section's code uses are shown to the checks, so a
default, a signature or a return value is the code's, not a page's.

Parsing uses the standard library's ``ast``; nothing is imported.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

# Archive paths: "pkg-1.0/src/strands/agent.py" or "strands/agent.py".
_ARCHIVE_ROOT = re.compile(r"^[^/]+-\d[^/]*/")
_SOURCE_PREFIX = re.compile(r"^(?:src|lib)/")
MAX_DEFINITION_CHARS = 2_500


@dataclass(frozen=True)
class Callable_:
    """A function's or class's keyword surface: its parameter names and
    whether it takes any keyword (``**kwargs``)."""

    params: frozenset[str]
    any_keyword: bool


@dataclass
class SdkIndex:
    """Modules (dotted names) -> the names they bind; qualified names
    ("strands.agent.agent.Agent") -> their keyword surface and source."""

    version: str = ""
    modules: dict[str, set[str]] = field(default_factory=dict)
    callables: dict[str, Callable_] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    # Names a module re-exports: "strands.Agent" -> "strands.agent.agent.Agent".
    aliases: dict[str, str] = field(default_factory=dict)

    @property
    def packages(self) -> set[str]:
        return {m.split(".")[0] for m in self.modules}

    @classmethod
    def from_sources(cls, files: Mapping[str, str], version: str = "") -> SdkIndex:
        index = cls(version=version)
        for path, text in files.items():
            module = _module_name(path)
            if not module:
                continue
            try:
                tree = ast.parse(text)
            except (SyntaxError, ValueError):
                continue
            package = (
                module if path.endswith("__init__.py") else module.rpartition(".")[0]
            )
            index._add(module, tree, text, package)
        return index

    def _add(self, module: str, tree: ast.Module, text: str, package: str) -> None:
        names = self.modules.setdefault(module, set())
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
                self._callable(f"{module}.{node.name}", node.args, node, text)
            elif isinstance(node, ast.ClassDef):
                names.add(node.name)
                init = next(
                    (
                        n
                        for n in node.body
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and n.name == "__init__"
                    ),
                    None,
                )
                if init is not None:
                    self._callable(f"{module}.{node.name}", init.args, node, text)
                else:
                    self.sources[f"{module}.{node.name}"] = _segment(text, node)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                names |= {t.id for t in targets if isinstance(t, ast.Name)}
            elif isinstance(node, ast.ImportFrom) and node.module:
                base = _resolve(package, node.module, node.level)
                for alias in node.names:
                    bound = alias.asname or alias.name
                    names.add(bound)
                    self.aliases[f"{module}.{bound}"] = f"{base}.{alias.name}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    names.add((alias.asname or alias.name).split(".")[0])

    def _callable(
        self, qualified: str, args: ast.arguments, node: ast.AST, text: str
    ) -> None:
        params = {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
        params.discard("self")
        self.callables[qualified] = Callable_(frozenset(params), args.kwarg is not None)
        self.sources[qualified] = _segment(text, node)

    def resolve(self, qualified: str) -> str:
        """``qualified`` through re-exports to where it is defined."""
        seen: set[str] = set()
        while qualified in self.aliases and qualified not in seen:
            seen.add(qualified)
            qualified = self.aliases[qualified]
        return qualified

    def has_module(self, module: str) -> bool:
        return module in self.modules

    def binds(self, module: str, name: str) -> bool:
        return (
            name in self.modules.get(module, set())
            or f"{module}.{name}" in self.modules
        )


def _module_name(path: str) -> str:
    path = _SOURCE_PREFIX.sub("", _ARCHIVE_ROOT.sub("", path))
    if not path.endswith(".py"):
        return ""
    parts = path[: -len(".py")].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(p for p in parts if p)


def _resolve(package: str, target: str, level: int) -> str:
    """The absolute module a ``from <dots><target> import`` names, from the
    importing module's ``package`` (level 1 is the package itself)."""
    if not level:
        return target
    parts = package.split(".") if package else []
    base = parts[: len(parts) - (level - 1)] if level > 1 else parts
    return ".".join([*base, target]) if target else ".".join(base)


def _segment(text: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(text, node) or ""
    return segment[:MAX_DEFINITION_CHARS]


@dataclass(frozen=True)
class ApiUse:
    module: str
    name: str
    keywords: tuple[str, ...] = ()


def api_problems(code_blocks: Iterable[str], index: SdkIndex) -> list[str]:
    """What the package does not have that the code uses: a module of the
    package that does not exist, a name a module does not bind, a keyword a
    function or class of the package does not take. Only the package's own
    names are judged; a block that does not parse is skipped (the reading
    check reports it)."""
    problems: list[str] = []
    for code in code_blocks:
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError):
            continue
        imported: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                top = node.module.split(".")[0]
                if top not in index.packages:
                    continue
                if not index.has_module(node.module):
                    problems.append(
                        f"module `{node.module}` does not exist in "
                        f"{top} {index.version}".rstrip()
                    )
                    continue
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    if not index.binds(node.module, alias.name):
                        problems.append(
                            f"`{alias.name}` is not in `{node.module}` "
                            f"({top} {index.version})".rstrip()
                        )
                    else:
                        imported[alias.asname or alias.name] = index.resolve(
                            f"{node.module}.{alias.name}"
                        )
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            target = imported.get(node.func.id)
            surface = index.callables.get(target or "")
            if surface is None or surface.any_keyword:
                continue
            unknown = [
                k.arg for k in node.keywords if k.arg and k.arg not in surface.params
            ]
            if unknown:
                problems.append(
                    f"`{node.func.id}()` takes no keyword {', '.join(map(repr, unknown))}"
                    f" (it takes {', '.join(sorted(surface.params)) or 'none'})"
                )
    return list(dict.fromkeys(problems))


_IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")


def definitions_for(text: str, index: SdkIndex, limit_chars: int) -> str:
    """The package's own source of the functions and classes ``text`` names
    (its code and code spans), most named first, within ``limit_chars``."""
    counts: dict[str, int] = {}
    for word in _IDENTIFIER.findall(text):
        counts[word] = counts.get(word, 0) + 1
    by_name: dict[str, list[str]] = {}
    for qualified in index.sources:
        by_name.setdefault(qualified.rsplit(".", 1)[-1], []).append(qualified)
    parts: list[str] = []
    left = limit_chars
    for name in sorted(counts, key=lambda n: -counts[n]):
        for qualified in by_name.get(name, [])[:2]:
            source = index.sources[qualified]
            if not source or left <= 0:
                continue
            block = f"# {qualified}\n{source}"[:left]
            left -= len(block)
            parts.append(block)
    return "\n\n".join(parts)
