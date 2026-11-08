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


# Top-level names that are a source tree's tooling, not its package: build
# output, docs, tests and scripts directories, setup and task files.
_TOOLING = frozenset(
    {
        "build",
        "dist",
        "doc",
        "docs",
        "test",
        "tests",
        "testing",
        "example",
        "examples",
        "benchmark",
        "benchmarks",
        "script",
        "scripts",
        "tools",
        "ci",
        "setup",
        "conftest",
        "noxfile",
        "tasks",
        "fabfile",
    }
)
# Statements whose bodies run at import time on some path: names bound under
# ``if TYPE_CHECKING:``, ``try: ... except ImportError:`` or a version test
# are names of the module.
_COMPOUND = (ast.If, ast.Try, ast.TryStar, ast.With)


@dataclass
class SdkIndex:
    """Modules (dotted names) -> the names they bind; qualified names
    ("strands.agent.agent.Agent") -> their keyword surface and source.

    ``partial`` says the source was cut (a size cap), so a module that is
    not in ``modules`` may still exist; ``open_modules`` bind names no
    reading can list (a module ``__getattr__``, a computed ``__all__``, a
    star import from outside the index, a file that does not parse). Neither
    ever yields a "does not exist" verdict."""

    version: str = ""
    modules: dict[str, set[str]] = field(default_factory=dict)
    callables: dict[str, Callable_] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    # Names a module re-exports: "strands.Agent" -> "strands.agent.agent.Agent".
    aliases: dict[str, str] = field(default_factory=dict)
    partial: bool = False
    open_modules: set[str] = field(default_factory=set)
    # Packages (an ``__init__.py`` or a namespace directory): their
    # submodules are names of them too.
    package_modules: set[str] = field(default_factory=set)
    # module -> the modules it star-imports.
    stars: dict[str, list[str]] = field(default_factory=dict)

    @property
    def packages(self) -> set[str]:
        return {m.split(".")[0] for m in self.modules}

    @classmethod
    def from_sources(
        cls,
        files: Mapping[str, str],
        version: str = "",
        *,
        partial: bool = False,
        top_levels: Iterable[str] = (),
    ) -> SdkIndex:
        """The index of ``files`` (archive path -> source). Only importable
        modules count: those under a top-level package (a directory with an
        ``__init__.py``, or a namespace package the distribution names in
        ``top_levels``) and single-file top-level modules; build output and
        tooling directories and files do not. ``partial`` says ``files`` are
        not all of the distribution's code."""
        index = cls(version=version, partial=partial)
        known = set(top_levels) | {
            module
            for path in files
            if path.endswith("__init__.py")
            and (module := _module_name(path))
            and "." not in module
            and module not in _TOOLING
        }
        for path, text in files.items():
            module = _module_name(path)
            if not module:
                continue
            is_package = path.endswith("__init__.py")
            top = module.split(".")[0]
            single_file = "." not in module and not is_package
            if top not in known and not (single_file and top not in _TOOLING):
                continue
            if is_package:
                index.package_modules.add(module)
            try:
                tree = ast.parse(text)
            except (SyntaxError, ValueError):
                # It exists, but what it binds cannot be read.
                index.modules.setdefault(module, set())
                index.open_modules.add(module)
                continue
            package = module if is_package else module.rpartition(".")[0]
            index._add(module, tree.body, text, package)
        # A namespace package (no __init__.py) exists through its modules.
        for module in list(index.modules):
            parts = module.split(".")
            for n in range(1, len(parts)):
                parent = ".".join(parts[:n])
                if parent not in index.modules:
                    index.modules[parent] = set()
                    index.package_modules.add(parent)
        return index

    def _add(self, module: str, body: list[ast.stmt], text: str, package: str) -> None:
        names = self.modules.setdefault(module, set())
        for node in body:
            if isinstance(node, _COMPOUND):
                for part in _bodies(node):
                    self._add(module, part, text, package)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
                if node.name == "__getattr__":
                    self.open_modules.add(module)
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
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                bound = {n.id for t in targets for n in _stored(t)}
                names |= bound
                if "__getattr__" in bound:
                    self.open_modules.add(module)
                if "__all__" in bound and node.value is not None:
                    if (listed := _literal_names(node.value)) is None:
                        self.open_modules.add(module)
                    else:
                        names |= listed
            elif isinstance(node, ast.Expr) and _extends_all(node.value):
                # ``__all__.extend(...)``: names computed at import time.
                self.open_modules.add(module)
            elif isinstance(node, ast.ImportFrom) and (node.module or node.level):
                base = _resolve(package, node.module or "", node.level)
                for alias in node.names:
                    if alias.name == "*":
                        self.stars.setdefault(module, []).append(base)
                        continue
                    bound_name = alias.asname or alias.name
                    names.add(bound_name)
                    self.aliases[f"{module}.{bound_name}"] = f"{base}.{alias.name}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        names.add(alias.asname)
                        self.aliases[f"{module}.{alias.asname}"] = alias.name
                    else:
                        names.add(alias.name.split(".")[0])

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
        """Whether ``module`` was read (it exists)."""
        return module in self.modules

    def module_missing(self, module: str) -> bool:
        """Whether ``module`` surely does not exist: the index read all of
        the distribution's code and it is not there."""
        return not self.partial and module not in self.modules

    def attribute(self, module: str, name: str) -> str | None:
        """What ``module.name`` is, through re-exports and star imports (a
        submodule or a definition's qualified name); None when the index
        does not show the module binding it."""
        return self._attribute(module, name, set())

    def _attribute(self, module: str, name: str, seen: set[str]) -> str | None:
        if f"{module}.{name}" in self.modules:
            return f"{module}.{name}"
        if name in self.modules.get(module, set()):
            return self.resolve(f"{module}.{name}")
        seen.add(module)
        for source in self.stars.get(module, []):
            if source not in seen and (found := self._attribute(source, name, seen)):
                return found
        return None

    def complete(self, module: str) -> bool:
        """Whether every name ``module`` binds is known, so a name it does
        not show does not exist: it was read, is not open, star-imports
        only complete modules, and (in a partial index) is not a package,
        whose unread submodules are names of it too."""
        return self._complete(module, set())

    def _complete(self, module: str, seen: set[str]) -> bool:
        if module not in self.modules or module in self.open_modules:
            return False
        if self.partial and module in self.package_modules:
            return False
        seen.add(module)
        return all(
            s in seen or self._complete(s, seen) for s in self.stars.get(module, [])
        )

    def binds(self, module: str, name: str) -> bool:
        """Whether ``module`` binds ``name`` or may (it is not complete)."""
        return self.attribute(module, name) is not None or not self.complete(module)


def _bodies(node: ast.stmt) -> list[list[ast.stmt]]:
    if isinstance(node, (ast.Try, ast.TryStar)):
        return [
            node.body,
            *(h.body for h in node.handlers),
            node.orelse,
            node.finalbody,
        ]
    if isinstance(node, ast.If):
        return [node.body, node.orelse]
    if isinstance(node, ast.With):
        return [node.body]
    return []


def _stored(target: ast.expr) -> list[ast.Name]:
    """The names an assignment target binds (``a``, ``a, (b, c)``)."""
    if isinstance(target, ast.Name):
        return [target]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [n for t in target.elts for n in _stored(t)]
    if isinstance(target, ast.Starred):
        return _stored(target.value)
    return []


def _literal_names(value: ast.expr) -> set[str] | None:
    """The names a literal ``__all__`` lists; None when it is computed."""
    if not isinstance(value, (ast.List, ast.Tuple)):
        return None
    names: set[str] = set()
    for element in value.elts:
        if not (isinstance(element, ast.Constant) and isinstance(element.value, str)):
            return None
        names.add(element.value)
    return names


def _extends_all(value: ast.expr) -> bool:
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and isinstance(value.func.value, ast.Name)
        and value.func.value.id == "__all__"
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


def api_problems(code_blocks: Iterable[str], index: SdkIndex) -> list[str]:
    """What the package does not have that the code uses: a module of the
    package that does not exist, a name a module does not bind, a keyword a
    function or class of the package does not take. Calls are resolved
    through the code's imports (``from m import n as a``, ``import m.sub as
    x``), attribute chains to a module (``x.func(...)``, ``m.Class(...)``)
    and decorators with arguments (``@x.func(...)``). Only the package's own
    names are judged, and a name the code rebinds is not; a block that does
    not parse is skipped (the reading check reports it)."""
    problems: list[str] = []
    for code in code_blocks:
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError):
            continue
        problems += _block_problems(tree, index)
    return list(dict.fromkeys(problems))


def _block_problems(tree: ast.Module, index: SdkIndex) -> list[str]:
    problems: list[str] = []
    version = f" {index.version}" if index.version else ""
    # Local name -> the package's qualified name (a module or a definition).
    env: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            top = node.module.split(".")[0]
            if top not in index.packages:
                continue
            if not index.has_module(node.module):
                # A partial index has not read every module: no verdict.
                if index.module_missing(node.module):
                    problems.append(
                        f"module `{node.module}` does not exist in {top}{version}"
                    )
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                if not index.binds(node.module, alias.name):
                    problems.append(
                        f"`{alias.name}` is not in `{node.module}` ({top}{version})"
                    )
                else:
                    env[alias.asname or alias.name] = index.attribute(
                        node.module, alias.name
                    ) or index.resolve(f"{node.module}.{alias.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in index.packages:
                    continue
                if not index.has_module(alias.name):
                    if index.module_missing(alias.name):
                        problems.append(
                            f"module `{alias.name}` does not exist in {top}{version}"
                        )
                    continue
                # ``import a.b`` binds ``a``; ``import a.b as x`` binds ``a.b``.
                env[alias.asname or top] = alias.name if alias.asname else top
    for name in _rebound(tree):
        env.pop(name, None)
    if not env:
        return problems
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            owner = _qualified(node.value, env, index)
            if owner is None or not index.has_module(owner):
                continue
            if index.attribute(owner, node.attr) is None and index.complete(owner):
                top = owner.split(".")[0]
                problems.append(f"`{node.attr}` is not in `{owner}` ({top}{version})")
        elif isinstance(node, ast.Call):
            target = _qualified(node.func, env, index)
            surface = index.callables.get(target or "")
            if surface is None or surface.any_keyword:
                continue
            unknown = [
                k.arg for k in node.keywords if k.arg and k.arg not in surface.params
            ]
            if unknown:
                problems.append(
                    f"`{ast.unparse(node.func)}()` takes no keyword "
                    f"{', '.join(map(repr, unknown))}"
                    f" (it takes {', '.join(sorted(surface.params)) or 'none'})"
                )
    return problems


def _rebound(tree: ast.Module) -> set[str]:
    """Names the code assigns or defines itself (then an import of the same
    name no longer says what they are)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return names


def _qualified(node: ast.expr, env: Mapping[str, str], index: SdkIndex) -> str | None:
    """The package's qualified name an expression of imported names and
    attributes refers to (``x.sub.func`` -> ``pkg.sub.api.func``), or None
    when it is not one the index knows."""
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.Attribute):
        owner = _qualified(node.value, env, index)
        if owner is not None and index.has_module(owner):
            return index.attribute(owner, node.attr)
    return None


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


@dataclass(frozen=True)
class PackageSource:
    """A release's Python code as read from its archive.

    ``files`` maps archive paths to source. ``complete`` is False when the
    size cap stopped the reading, so a module absent from ``files`` may
    still exist. ``top_levels`` are the import names the wheel installs at
    its root (or its metadata lists): namespace packages count as packages
    of the distribution, build and tooling directories do not."""

    version: str
    files: dict[str, str]
    complete: bool = True
    top_levels: frozenset[str] = frozenset()
