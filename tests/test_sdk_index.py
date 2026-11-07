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
