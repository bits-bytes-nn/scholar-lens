"""A guide's code checked against the product's own source (read with ast,
never run), and the definitions shown to its checks."""

from __future__ import annotations

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
