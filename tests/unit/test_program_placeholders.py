"""A runnable section's code with a ``...`` placeholder is a problem; an
explicit excerpt and ``.pyi``-style protocol stubs are not
(docs/engine.md, 4.1 program)."""

from __future__ import annotations

import pytest

from scholar_lens.domain.program import (
    EXCERPT_MARKER,
    is_excerpt,
    python_code_problems,
)
from scholar_lens.domain.prompts.guide_evidence import GUIDE_PRINCIPLES

PLACEHOLDER = "the block has placeholders; show runnable code or mark it as an excerpt"

# Each holds a ``...`` a reader cannot run.
PLACEHOLDERS = [
    "def handler(request):\n    ...\n",
    "class Model:\n    ...\n\nmodel = Model()\n",
    "class Model:\n    '''A model.'''\n    ...\n",
    "async def main():\n    ...\n",
    "...\n",
    "x = 1\n...\nprint(x)\n",
    "def f():\n    x = 1\n    ...\n    return x\n",
    "for item in range(3):\n    ...\n",
    "if True:\n    ...\nelse:\n    pass\n",
    "try:\n    ...\nexcept ValueError:\n    pass\n",
    "with open('a') as fh:\n    ...\n",
    "class Outer:\n    def run(self):\n        ...\n",
]

# Ellipsis that is not a statement, or a stub by design.
RUNNABLE = [
    "import numpy as np\n\narr = np.zeros((2, 2))\nprint(arr[..., 0])\n",
    "x = ...\nprint(x)\n",
    "from typing import Protocol\n\nclass Store(Protocol):\n"
    "    def get(self, key: str) -> str: ...\n",
    "import typing\n\nclass Store(typing.Protocol):\n    def get(self) -> str:\n        ...\n",
    "from typing import overload\n\n@overload\ndef f(x: int) -> int: ...\n"
    "@overload\ndef f(x: str) -> str: ...\ndef f(x):\n    return x\n",
    "from abc import ABC, abstractmethod\n\nclass Base(ABC):\n"
    "    @abstractmethod\n    def run(self) -> None: ...\n",
    "import abc\n\nclass Base(abc.ABC):\n    @abc.abstractmethod\n    def run(self):\n        ...\n",
    "def handler(request):\n    return request\n",
]


def _section(code: str, *, checked: bool = True) -> list[tuple[str, str, bool]]:
    return [("Usage", f"## Usage\n\n```python\n{code}```\n", checked)]


def _placeholder_problems(code: str, *, checked: bool = True) -> list[str]:
    return [
        p
        for _, p in python_code_problems(_section(code, checked=checked))
        if p.startswith(PLACEHOLDER)
    ]


@pytest.mark.parametrize("code", PLACEHOLDERS)
def test_a_placeholder_in_a_runnable_section_is_a_problem(code: str) -> None:
    assert len(_placeholder_problems(code)) == 1


@pytest.mark.parametrize("code", PLACEHOLDERS)
@pytest.mark.parametrize(
    "marker", [EXCERPT_MARKER, EXCERPT_MARKER + ": the signature", "# Excerpt"]
)
def test_an_excerpt_marked_block_may_hold_placeholders(code: str, marker: str) -> None:
    assert _placeholder_problems(f"{marker}\n{code}") == []
    assert _placeholder_problems(f"\n{marker}\n{code}") == []


@pytest.mark.parametrize("code", PLACEHOLDERS)
def test_a_section_nobody_runs_is_not_checked(code: str) -> None:
    assert _placeholder_problems(code, checked=False) == []


@pytest.mark.parametrize("code", RUNNABLE)
def test_ellipsis_values_and_protocol_stubs_are_not_placeholders(code: str) -> None:
    assert _placeholder_problems(code) == []


def test_the_marker_only_counts_on_the_first_line() -> None:
    code = "def handler(request):\n    ...\n" + EXCERPT_MARKER + "\n"
    assert not is_excerpt(code)
    assert len(_placeholder_problems(code)) == 1


def test_the_writer_prompt_documents_the_marker() -> None:
    principles = " ".join(GUIDE_PRINCIPLES.split())
    assert f"`{EXCERPT_MARKER}" in principles
    assert "no `...` placeholder" in principles


def test_a_call_that_does_not_bind_to_the_posts_own_definition_is_a_problem() -> None:
    from scholar_lens.domain.program import python_code_problems

    defined = (
        "## 정의\n\n```python\ndef handle(ticket, *, user_id):\n    return ticket\n\n"
        "class Agent:\n    def __init__(self, name, tools=None):\n        self.name = name\n"
        "\n\ndef log(*args, **kwargs):\n    print(args, kwargs)\n```\n"
    )
    used = (
        "## 사용\n\n```python\nhandle('t')\nAgent('a', tools=[], model='x')\n"
        "handle('t', user_id=1)\nAgent('ok')\nlog(1, 2, x=3)\nhandle(*['t'], **{})\n```\n"
    )
    problems = [
        p
        for _, p in python_code_problems(
            [("정의", defined, True), ("사용", used, True)]
        )
    ]
    assert len(problems) == 2
    assert "handle (missing required argument(s) user_id)" in problems[0]
    assert "Agent (unknown keyword argument(s) model)" in problems[1]

    # A later definition replaces the earlier one.
    redefined = "## 다시\n\n```python\ndef handle(ticket):\n    return ticket\n\nhandle('t')\n```\n"
    assert not [
        p
        for _, p in python_code_problems(
            [("정의", defined, True), ("다시", redefined, True)]
        )
        if "does not match" in p
    ]


@pytest.mark.parametrize(
    "blocks",
    [
        # A decorator may change what a call takes.
        [
            (
                "import click\n@click.command()\n@click.option('--name')\ndef hello(name):\n    pass\n\nhello()",
                True,
            )
        ],
        # A name rebound by an import, an assignment or a lambda.
        [
            ("def attention(q, k, v):\n    return q", True),
            (
                "from torch.nn.functional import scaled_dot_product_attention as attention\n"
                "attention(1, 2, 3, is_causal=True)",
                True,
            ),
        ],
        [
            (
                "import functools\ndef query(db, sql):\n    pass\nquery = functools.partial(query, 'db')\nquery('select 1')",
                True,
            )
        ],
        [("def make(x):\n    pass\nmake = lambda: 3\nmake()", True)],
        # A call in a function body runs under its own scope.
        [("def load(path):\n    pass\ndef run(load):\n    load()\nrun(print)", True)],
        # A call before the block redefines the name binds to the earlier one.
        [("def f(a):\n    pass\nf(1)\ndef f(a, b):\n    pass", True)],
        # @overload stubs are not the constructor.
        [
            (
                "from typing import overload\nclass C:\n    @overload\n    def __init__(self, a: str): ...\n"
                "    @overload\n    def __init__(self, a: str, b: int): ...\n    def __init__(self, a, b=0):\n"
                "        pass\nC('a', 1)",
                True,
            )
        ],
        # Other bindings: except-as, match capture, a star import, global.
        [
            (
                "def f(a):\n    pass\ntry:\n    pass\nexcept Exception as f:\n    pass\nf()",
                True,
            )
        ],
        [("def f(a):\n    pass\nmatch 1:\n    case f:\n        pass\nf()", True)],
        [("def dumps(a):\n    pass\nfrom json import *\ndumps({}, indent=2)", True)],
        # A base may change what calling the class does.
        [
            (
                "from enum import Enum\nclass Planet(Enum):\n    EARTH = (1.0, 2.0)\n"
                "    def __init__(self, mass, radius):\n        self.mass = mass\nPlanet((1.0, 2.0))",
                True,
            )
        ],
        # An internals sketch (an unchecked section) defines nothing a reader runs.
        [("def handle(t, *, user_id):\n    pass", False), ("handle('t')", True)],
    ],
)
def test_calls_the_signature_check_must_not_flag(
    blocks: list[tuple[str, bool]],
) -> None:
    from scholar_lens.domain.program import python_code_problems

    sections = [
        (f"s{i}", f"## s{i}\n\n```python\n{code}\n```\n", checked)
        for i, (code, checked) in enumerate(blocks)
    ]
    assert not [p for _, p in python_code_problems(sections) if "does not match" in p]


def test_calls_that_run_at_definition_are_checked() -> None:
    from scholar_lens.domain.program import python_code_problems

    code = "def f(a):\n    pass\ndef g(x=f()):\n    pass\nclass C:\n    v = f()\n"
    found = [
        p
        for _, p in python_code_problems(
            [("s", f"## s\n\n```python\n{code}```\n", True)]
        )
        if "does not match" in p
    ]
    assert len(found) == 2
