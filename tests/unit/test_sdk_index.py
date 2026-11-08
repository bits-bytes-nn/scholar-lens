"""A guide's code checked against the product's own source (read with ast,
never run), and the definitions shown to its checks."""

from __future__ import annotations

import pytest

from scholar_lens.domain.guide.sdk_index import SdkIndex, api_problems, definitions_for

FILES = {
    "pkg-1.2.0/src/toolkit/__init__.py": "from .agent import Agent\n__all__ = ['Agent']\n",
    "pkg-1.2.0/src/toolkit/agent.py": (
        "class Agent:\n"
        "    def __init__(self, model=None, *, system_prompt='', max_delay=240):\n"
        "        self.model = model\n\n"
        "def run(prompt, **kwargs):\n    return prompt\n"
    ),
    "pkg-1.2.0/src/toolkit/hooks/__init__.py": "class HookProvider:\n    pass\n",
}


def _index() -> SdkIndex:
    return SdkIndex.from_sources(FILES, "toolkit 1.2.0")


def test_import_paths_names_and_keywords_are_checked_against_the_code() -> None:
    code = (
        "from toolkit import Agent\nfrom toolkit.hooks import HookProvider, Missing\n"
        "from toolkit.nowhere import X\nfrom toolkit.agent import run\nimport json\n"
        "agent = Agent(model='m', system_prompt='s', made_up=1)\nrun('p', anything=1)\n"
    )
    problems = api_problems([code], _index())
    assert any("`Missing` is not in `toolkit.hooks`" in p for p in problems)
    assert any("module `toolkit.nowhere` does not exist" in p for p in problems)
    assert any("`Agent()` takes no keyword 'made_up'" in p for p in problems)
    # **kwargs takes anything; other packages and unparsable code are not judged.
    assert not any("run()" in p for p in problems)
    assert api_problems(["from other import X", "x = <key>"], _index()) == []


def test_the_definitions_a_section_names_are_shown_with_their_defaults() -> None:
    found = definitions_for("`Agent(max_delay=...)` 기본값", _index(), 2_000)
    assert "max_delay=240" in found and "toolkit.agent.Agent" in found
    assert definitions_for("nothing here", _index(), 2_000) == ""


# A package whose callables are reached through attributes: a submodule
# re-exported by the package, a function re-exported by the submodule, a
# class defined in it.
NESTED = {
    "kit/__init__.py": "from . import serve\n",
    "kit/serve/__init__.py": (
        "from .api import deployment\n\n"
        "class Router:\n    def __init__(self, policy, *, limit=1):\n        pass\n"
    ),
    "kit/serve/api.py": (
        "def deployment(func=None, *, num_replicas=1, router_class=None):\n"
        "    return func\n"
    ),
}


def _nested() -> SdkIndex:
    return SdkIndex.from_sources(NESTED, "kit 0.1")


def test_an_attribute_calls_unknown_keyword_is_reported() -> None:
    code = "from kit import serve\nserve.deployment(num_replicas=2, made_up=1)\n"
    problems = api_problems([code], _nested())
    assert any("`serve.deployment()` takes no keyword 'made_up'" in p for p in problems)
    assert not any("'num_replicas'" in p for p in problems)


def test_a_class_reached_through_its_module_is_checked() -> None:
    code = "import kit.serve\nkit.serve.Router('p', limit=2, bogus=3)\n"
    problems = api_problems([code], _nested())
    assert any("`kit.serve.Router()` takes no keyword 'bogus'" in p for p in problems)


def test_a_decorator_calls_unknown_keyword_is_reported() -> None:
    code = (
        "from kit import serve\n\n"
        "@serve.deployment(router_class=None, replicas=3)\n"
        "class App:\n    pass\n"
    )
    problems = api_problems([code], _nested())
    assert any(
        "`serve.deployment()` takes no keyword 'replicas'" in p for p in problems
    )


def test_a_module_alias_resolves_its_attributes() -> None:
    code = "import kit.serve as ks\nks.deployment(wrong=1)\nks.nothing()\n"
    problems = api_problems([code], _nested())
    assert any("`ks.deployment()` takes no keyword 'wrong'" in p for p in problems)
    assert any("`nothing` is not in `kit.serve`" in p for p in problems)


def test_a_missing_module_of_a_plain_import_is_reported() -> None:
    problems = api_problems(["import kit.nowhere\n"], _nested())
    assert any("module `kit.nowhere` does not exist" in p for p in problems)


def test_a_name_the_code_rebinds_is_not_judged() -> None:
    code = (
        "from kit import serve\nserve = object()\n"
        "serve.deployment(anything=1)\nserve.whatever\n"
    )
    assert api_problems([code], _nested()) == []


@pytest.mark.parametrize(
    "body",
    [
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n"
        "    from .core import Widget\n",
        "try:\n    from .fast import Widget\nexcept ImportError:\n"
        "    from .core import Widget\n",
        "import sys\nif sys.version_info >= (3, 12):\n    Widget = int\n"
        "else:\n    Widget = float\n",
        "__all__ = ['Widget']\n",
    ],
    ids=["type-checking", "try-import", "version-test", "literal-all"],
)
def test_a_name_bound_in_a_compound_statement_or_listed_in_all_exists(
    body: str,
) -> None:
    index = SdkIndex.from_sources(
        {"gear/__init__.py": body, "gear/core.py": "class Widget:\n    pass\n"}
    )
    assert api_problems(["from gear import Widget\n"], index) == []
    assert api_problems(["from gear import Gadget\n"], index)


@pytest.mark.parametrize(
    "body",
    [
        "def __getattr__(name):\n    raise AttributeError(name)\n",
        "__getattr__, __dir__ = make_lazy(__name__)\n",
        "__all__ = sorted(_EXPORTS)\n",
        "__all__ = []\n__all__.extend(_EXPORTS)\n",
        "from outside_dependency import *\n",
    ],
    ids=["getattr", "assigned-getattr", "computed-all", "extended-all", "star"],
)
def test_an_open_module_never_yields_a_missing_name_problem(body: str) -> None:
    index = SdkIndex.from_sources({"gear/__init__.py": body})
    code = "import gear\nfrom gear import Anything\ngear.whatever()\n"
    assert api_problems([code], index) == []


def test_a_star_import_from_the_package_brings_its_names() -> None:
    index = SdkIndex.from_sources(
        {
            "gear/__init__.py": "from .core import *\n",
            "gear/core.py": "def spin(*, speed=1):\n    pass\n",
        }
    )
    assert api_problems(["from gear import spin\n"], index) == []
    assert api_problems(["from gear import missing\n"], index)
    problems = api_problems(["import gear\ngear.spin(force=2)\n"], index)
    assert any("`gear.spin()` takes no keyword 'force'" in p for p in problems)


def test_a_partial_index_does_not_report_modules_it_did_not_read() -> None:
    files = {
        "gear/__init__.py": "",
        "gear/core.py": "def spin():\n    pass\n",
    }
    partial = SdkIndex.from_sources(files, partial=True)
    code = (
        "from gear.unread import Thing\nimport gear.other\n"
        "from gear import unread_sub\nfrom gear.core import spin, missing\n"
    )
    problems = api_problems([code], partial)
    # Only the plain module it read can be judged.
    assert problems == ["`missing` is not in `gear.core` (gear)"]
    full = api_problems([code], SdkIndex.from_sources(files))
    assert any("module `gear.unread` does not exist" in p for p in full)
    assert any("`unread_sub` is not in `gear`" in p for p in full)


def test_only_importable_packages_count() -> None:
    files = {
        "gear-1.0/setup.py": "from setuptools import setup\n",
        "gear-1.0/build/lib/gear/__init__.py": "",
        "gear-1.0/tools/release.py": "def bump():\n    pass\n",
        "gear-1.0/gear/__init__.py": "",
        "gear-1.0/single.py": "def helper():\n    pass\n",
    }
    index = SdkIndex.from_sources(files)
    assert index.packages == {"gear", "single"}
    assert api_problems(["from tools.release import nothing\n"], index) == []


def test_a_namespace_package_counts_when_the_distribution_names_it() -> None:
    files = {"space/gear/__init__.py": "def spin():\n    pass\n"}
    assert SdkIndex.from_sources(files).packages == set()
    index = SdkIndex.from_sources(files, top_levels={"space"})
    assert index.packages == {"space"}
    assert api_problems(["from space.gear import spin\n"], index) == []
    assert api_problems(["from space.gear import spun\n"], index)
