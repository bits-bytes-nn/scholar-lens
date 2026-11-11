"""The definitions a guide's runnable code shows, read with ast: what later
sections' code must call as defined."""

from __future__ import annotations

from scholar_lens.domain.guide.evidence_guide import code_definitions

FIRST = (
    "## Setup\n\n```python\nfrom kit import serve\n\n"
    "LIMIT: int = 3\nclient = make_client(region='x', retries=LIMIT)\n\n"
    "@serve.deployment(replicas=2)\nclass Echo(Base, metaclass=Meta):\n"
    "    size = 1\n\n"
    "    async def reply(self, text: str, *, loud: bool = False) -> str:\n"
    "        def inner():\n            return text\n"
    "        return inner()\n\n"
    "def build(name, **options):\n    helper = 1\n    return Echo()\n```\n"
)
SECOND = (
    "## Use\n\n```python\ndef build(name, *, strict=True) -> Echo:\n"
    "    return Echo()\n\n"
    "if __name__ == '__main__':\n    app = build('a')\n```\n"
)


def test_signatures_and_top_level_names_are_listed_in_reading_order() -> None:
    assert code_definitions([FIRST]) == [
        "LIMIT: int = 3",
        "client = make_client(region='x', retries=LIMIT)",
        "@serve.deployment(replicas=2) class Echo(Base, metaclass=Meta)",
        "async def Echo.reply(self, text: str, *, loud: bool=False) -> str",
        "def build(name, **options)",
    ]


def test_a_redefinition_replaces_the_earlier_one_in_its_place() -> None:
    lines = code_definitions([FIRST, SECOND])
    assert "def build(name, **options)" not in lines
    assert lines[-2:] == ["def build(name, *, strict=True) -> Echo", "app = build('a')"]


def test_the_last_lines_are_kept_within_the_limit() -> None:
    assert (
        code_definitions([FIRST, SECOND], limit=2)
        == code_definitions([FIRST, SECOND])[-2:]
    )
    assert code_definitions([FIRST], limit=0) == []


def test_only_python_code_that_parses_or_runs_in_a_session_counts() -> None:
    markdown = (
        "```bash\nexport TOKEN=1\n```\n\n"
        "```javascript\nfunction start() {}\n```\n\n"
        "```python\ndef broken(:\n```\n\n"
        "```python\n>>> total = 1 + 2\n>>> total\n3\n```\n\n"
        "```python\nvalue = '" + "v" * 200 + "'\n```\n"
    )
    lines = code_definitions([markdown])
    assert lines[0] == "total = 1 + 2"
    assert lines[1].startswith("value = ") and lines[1].endswith("…")
    assert len(lines) == 2
