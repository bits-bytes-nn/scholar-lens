"""Running a guide's code as a reader pastes it: the program, its outcome
lines and whose failure it is."""

from __future__ import annotations

import subprocess
import sys
import tempfile

import pytest

from scholar_lens.domain.program_run import (
    BlockRun,
    blamed,
    judge_run,
    program_blocks,
    program_script,
    read_outcomes,
    run_mark,
    run_requirements,
)


def _run(source: str, timeout: int = 60) -> str:
    # In a scratch directory: a block may write files (a %%writefile cell).
    with tempfile.TemporaryDirectory() as scratch:
        return subprocess.run(
            [sys.executable, "-I", "-c", source],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=scratch,
        ).stdout


def _py(code: str) -> str:
    return f"```python\n{code}\n```"


# ---- the blocks and the program ---------------------------------------------------


def test_excerpts_run_unjudged_sessions_run_and_every_section_counts() -> None:
    sections = [
        ("설치", _py("x = 1"), False),
        ("사용", _py("# excerpt\nfoo()") + "\n\n" + _py(">>> y = x\n>>> y\n1"), True),
        ("실행", _py("print(y)"), True),
    ]
    assert [(t, c.strip(), r) for t, c, r in program_blocks(sections)] == [
        ("설치", "x = 1", False),
        ("사용", "# excerpt\nfoo()", False),
        ("사용", "y = x\ny", True),
        ("실행", "print(y)", True),
    ]


def test_the_program_shares_state_and_runs_on_after_a_failure() -> None:
    script = program_script(
        [
            "import asyncio\nasync def f():\n    return 2\nx = asyncio.run(f())",
            "if __name__ == '__main__':\n    print('main', x, end='')",
            "z = 3\ny = x.nope\nprint('skipped')",
            "print('after', z)",
        ],
        block_timeout_s=10,
    )
    out = _run(script)
    assert "skipped" not in out and "after 3" in out
    runs = read_outcomes(out)
    assert [r.ok for r in runs] == [True, True, False, True]
    assert runs[2].error_type == "AttributeError"
    # The block's print without a newline does not hide the outcome line.
    assert "main 2" in out


def test_a_block_over_its_time_is_stopped() -> None:
    script = program_script(["import time\ntime.sleep(30)"], block_timeout_s=1)
    (run,) = read_outcomes(_run(script, timeout=20))
    assert run.error_type == "BlockTimeout"


def test_outcomes_out_of_order_or_malformed_are_ignored() -> None:
    mark = "@@SCHOLAR_LENS_RESULT@@"
    out = "\n".join(
        [
            f'{mark}{{"index": 1, "ok": true}}',
            f"{mark}not json",
            f'{mark}{{"index": 3, "ok": true}}',
            f'noise{mark}{{"index": 2, "ok": false, "error_type": "NameError"}}',
        ]
    )
    assert [(r.index, r.ok) for r in read_outcomes(out)] == [(1, True), (2, False)]


def _blocks() -> list[tuple[str, str, bool]]:
    return [("설치", "import os", False), ("사용", "b = os.c", True), ("끝", "d", True)]


def test_the_sandbox_installs_the_docs_pins_and_what_the_post_installs() -> None:
    post = (
        '```bash\npip install "ray[serve]" transformers torch\n```\n'
        "```bash\npip install Strands_Agents[otel] -U\n```\n"
        "Then pip install the SDK if needed.\n"
        "```bash\npip install -r requirements.txt 'boto3<2' -t vendor pkg.whl\n```\n"
    )
    assert run_requirements([("ray", "2.40.0"), ("strands-agents", "1.2")], post) == [
        "ray[serve]==2.40.0",
        "strands-agents[otel]==1.2",
        "transformers",
        "torch",
        "boto3<2",
    ]


def test_a_message_loses_terminal_colours_and_line_breaks() -> None:
    run = BlockRun(
        index=1, ok=False, message="\x1b[36mray::Ctl\x1b[39m (pid=1)\n  File x"
    )
    assert run.message == "ray::Ctl (pid=1) File x"


def test_blocks_run_as_a_script_with_a_main_module() -> None:
    script = program_script(
        [
            "from pathlib import Path\nhere = Path(__file__).name",
            "from __future__ import annotations\nimport typing\n"
            "class A:\n    x: int\nprint(typing.get_type_hints(A))",
            "import sys\nsys.exit(0)",
            "print('after exit', here)",
        ],
        block_timeout_s=10,
    )
    runs = read_outcomes(_run(script))
    assert [r.ok for r in runs] == [True, True, True, True], runs


def test_the_code_cannot_hide_or_forge_its_outcomes() -> None:
    """A block that swallows exceptions, replaces stdout or prints a
    marker of its own does not change what is read."""
    mark = run_mark("n0nce")
    script = program_script(
        [
            "import time\nwhile True:\n    try:\n        time.sleep(5)\n"
            "    except Exception:\n        pass",
            "import io, sys\nsys.stdout = io.StringIO()",
            "print('@@SCHOLAR_LENS_RESULT@@{\"index\": 4, \"ok\": true}')\nraise NameError('x')",
            "sys.stdout.close()",
        ],
        block_timeout_s=1,
        mark=mark,
    )
    out = _run(script, timeout=30)
    runs = read_outcomes(out, mark)
    assert [(r.index, r.ok, r.error_type) for r in runs] == [
        (1, False, "BlockTimeout"),
        (2, True, ""),
        (3, False, "NameError"),
        (4, True, ""),
    ]


def test_top_level_await_runs() -> None:
    script = program_script(
        ["import asyncio\nawait asyncio.sleep(0)\nx = 1", "print(x)"],
        block_timeout_s=10,
    )
    assert [r.ok for r in read_outcomes(_run(script))] == [True, True]


def test_post_installs_pairs_fences() -> None:
    post = (
        "```python\nx = 1\n```\n\nAvoid running pip install numpy==1.0 in prose.\n\n"
        "```bash\npip install strands-agents-tools\n```\n"
        "```\npip install boto3\n```\n"
    )
    assert run_requirements([], post) == ["strands-agents-tools", "boto3"]


def test_spawned_children_write_no_outcomes_and_futures_carry_over(tmp_path) -> None:  # type: ignore[no-untyped-def]
    mark = run_mark("x1")
    script = program_script(
        [
            "from __future__ import annotations\nimport multiprocessing as mp\n"
            "def sq(x):\n    return x * x",
            "class A:\n    b: B\nclass B:\n    pass\nimport typing\ntyping.get_type_hints(A)",
            "if __name__ == '__main__':\n    ctx = mp.get_context('spawn')\n"
            "    with ctx.Pool(2) as pool:\n        print(pool.map(sq, [1, 2]))",
            "import inspect\ndef f():\n    pass\ninspect.getsource(f)",
        ],
        block_timeout_s=60,
        mark=mark,
    )
    path = tmp_path / "prog.py"
    path.write_text(script)
    out = subprocess.run(
        [sys.executable, "-I", str(path)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    ).stdout
    assert out.count(mark) == 4, out
    assert [r.ok for r in read_outcomes(out, mark)] == [True, True, True, True]


def test_a_notebook_cell_keeps_its_code_without_the_magics() -> None:
    from scholar_lens.domain.program import python_code_problems

    sections = [
        (
            "설치",
            _py("%pip install requests\nimport json\ndata = json.dumps({})"),
            True,
        ),
        ("사용", _py("print(data)"), True),
    ]
    assert [c.strip() for _, c, _ in program_blocks(sections)] == [
        "import json\ndata = json.dumps({})",
        "print(data)",
    ]
    assert python_code_problems(sections) == []


@pytest.mark.parametrize(
    ("cell", "runs"),
    [
        ("%%writefile requirements.txt\nstrands-agents\nbedrock-agentcore", "file"),
        ("%%bash\nls -la", ""),
        ("%%time\nx = sum(range(3))", "x = sum(range(3))"),
        ("resp = 1\nok = (resp\n      != 200)", None),
    ],
)
def test_cell_magics_run_as_a_notebook_runs_them(cell: str, runs: str | None) -> None:
    blocks = program_blocks([("셀", _py(cell), True)])
    if runs is None:  # a plain script: run as written
        assert blocks[0][1].strip() == cell
    elif runs == "file":
        ((_, code, _),) = blocks
        script = program_script(
            [code, "print(open('requirements.txt').read())"], block_timeout_s=10
        )
        assert [r.ok for r in read_outcomes(_run(script))] == [True, True]
    else:
        assert [c.strip() for _, c, _ in blocks] == ([runs] if runs else [])


def test_a_module_a_cell_writes_is_importable_and_runnable() -> None:
    blocks = program_blocks(
        [
            (
                "도구",
                _py("%%writefile tools.py\ndef add(a, b):\n    return a + b"),
                True,
            ),
            ("사용", _py("from tools import add\nprint(add(1, 2))"), True),
            ("실행", _py("%run tools.py\nprint(add(2, 3))"), True),
        ]
    )
    script = program_script([c for _, c, _ in blocks], block_timeout_s=10)
    assert [r.ok for r in read_outcomes(_run(script))] == [True, True, True]


def test_a_timing_magic_runs_its_statement() -> None:
    from scholar_lens.domain.program import python_code_problems

    sections = [
        ("시작", _py("%time result = sum(range(3))"), True),
        ("출력", _py("print(result)"), True),
    ]
    assert [c.strip() for _, c, _ in program_blocks(sections)] == [
        "result = sum(range(3))",
        "print(result)",
    ]
    assert python_code_problems(sections) == []


def test_an_indented_shell_line_keeps_the_cell_valid() -> None:
    cell = "try:\n    import json5\nexcept ImportError:\n    !pip install -q json5\nprint('ready')"
    ((_, code, _),) = program_blocks([("설정", _py(cell), True)])
    script = program_script([code], block_timeout_s=10)
    assert [r.ok for r in read_outcomes(_run(script))] == [True]


def test_a_transcript_showing_a_traceback_is_not_judged() -> None:
    cell = (
        ">>> Config(nme='x')\nTraceback (most recent call last):\n  ...\n"
        "TypeError: Config.__init__() got an unexpected keyword argument 'nme'"
    )
    ((_, _, checked),) = program_blocks([("오류", _py(cell), True)])
    assert not checked


# ---- what the post is blamed for -------------------------------------------------

_PRODUCT = {"strands", "strands.hooks", "llama_index.core"}


def _failed(error_type: str, message: str, index: int = 1) -> BlockRun:
    return BlockRun(index=index, ok=False, error_type=error_type, message=message)


@pytest.mark.parametrize(
    ("error_type", "message", "is_blamed"),
    [
        # The product's API itself.
        ("TypeError", "Agent.__init__() got an unexpected keyword argument 'x'", True),
        ("TypeError", "Agent() missing 1 required positional argument: 'model'", True),
        (
            "ImportError",
            "cannot import name 'HookOrder' from 'strands.hooks' (/x.py)",
            True,
        ),
        ("ModuleNotFoundError", "No module named 'strands.tool'", True),
        ("AttributeError", "module 'strands' has no attribute 'Agnet'", True),
        (
            "ParamValidationError",
            'Unknown parameter in inferenceConfig: "max_tokens"',
            True,
        ),
        ("ParamValidationError", 'Missing required parameter in input: "Key"', True),
        ("UnknownServiceError", "Unknown service: 'bedrock-agent-runtimes'", True),
        # Anything else leaves the check partial.
        ("TypeError", "helper() got an unexpected keyword argument 'x'", False),
        (
            "TypeError",
            "Client.__init__() got an unexpected keyword argument 'proxies'",
            False,
        ),
        ("TypeError", "Could not resolve authentication method", False),
        ("ImportError", "cannot import name 'X' from 'numpy'", False),
        ("ModuleNotFoundError", "No module named 'strands'", False),
        ("ModuleNotFoundError", "No module named 'llama_index.llms'", False),
        ("AttributeError", "module 'numpy' has no attribute 'float_'", False),
        ("AttributeError", "'str' object has no attribute 'text'", False),
        ("ParamValidationError", "Invalid type for parameter id, value: None", False),
        ("NameError", "name 'client' is not defined", False),
        ("KeyError", "'OPENAI_API_KEY'", False),
        ("NoCredentialsError", "Unable to locate credentials", False),
    ],
)
def test_only_the_products_api_is_blamed(
    error_type: str, message: str, is_blamed: bool
) -> None:
    assert blamed(_failed(error_type, message), _PRODUCT, {"Agent"}) is is_blamed


def test_a_finding_names_the_first_blamed_block_and_the_rest_limits_the_check() -> None:
    blocks = [
        ("설치", "from strands import Agent", True),
        ("호출", "agent = Agent()\nagent('hi')", True),
        ("설정", "agent2 = Agent(bad=1)", True),
    ]
    outcomes = [
        BlockRun(index=1, ok=True),
        _failed("NoCredentialsError", "Unable to locate credentials", 2),
        _failed(
            "TypeError", "Agent.__init__() got an unexpected keyword argument 'bad'", 3
        ),
    ]
    run = judge_run(blocks, outcomes, _PRODUCT)
    (finding,) = run.findings
    assert finding.section == "설정" and "NoCredentialsError" in run.limited
    # Off the docs' versions, the product's API may differ: nothing is blamed.
    assert not judge_run(blocks, outcomes, _PRODUCT, exact=False).findings
    # A section not meant to run is not blamed.
    blocks[2] = ("원리", blocks[2][1], False)
    assert not judge_run(blocks, outcomes, _PRODUCT).findings


def test_a_callee_the_post_redefines_is_its_own() -> None:
    blocks = [
        ("설치", "from strands import Agent", True),
        ("원리", "class Agent:\n    def __init__(self, model):\n        pass", False),
        ("사용", "Agent(a=1)", True),
    ]
    failed = _failed(
        "TypeError", "Agent.__init__() got an unexpected keyword argument 'a'", 3
    )
    outcomes = [BlockRun(index=1, ok=True), BlockRun(index=2, ok=True), failed]
    assert not judge_run(blocks, outcomes, _PRODUCT).findings


def test_a_module_attribute_a_skipped_import_binds_is_not_blamed() -> None:
    blocks = [
        ("설치", "import strands", True),
        ("설정", "import os\nos.environ['X']\nimport strands.storage", True),
        ("저장", "strands.storage.save(1)", True),
    ]
    attr = _failed("AttributeError", "module 'strands' has no attribute 'storage'", 3)
    outcomes = [BlockRun(index=1, ok=True), _failed("KeyError", "'X'", 2), attr]
    assert not judge_run(blocks, outcomes, _PRODUCT).findings
    blocks[1] = ("설정", "import os\nos.environ['X']", True)
    assert judge_run(blocks, outcomes, _PRODUCT).findings


def test_a_cells_pip_lines_install() -> None:
    post = _py("%pip install llama-index-core llama-index-llms-bedrock-converse\nx = 1")
    assert run_requirements([("llama-index-core", "0.12.0")], post) == [
        "llama-index-core==0.12.0",
        "llama-index-llms-bedrock-converse",
    ]


def test_an_excerpt_is_not_judged_by_the_static_check() -> None:
    from scholar_lens.domain.program import python_code_problems

    sections = [
        (
            "발췌",
            _py(
                "# excerpt: 앞 클래스의 머리만\nclass Translator:\n    # 본문은 앞과 같습니다"
            ),
            True,
        ),
        ("발췌2", _py("# excerpt\nresult = handle.remote(x)"), True),
    ]
    assert python_code_problems(sections) == []
