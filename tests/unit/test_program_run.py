"""Running a guide's code as a reader pastes it: the program, its outcome
lines and whose failure it is."""

from __future__ import annotations

import subprocess
import sys

import pytest

from scholar_lens.domain.program_run import (
    BlockRun,
    judge_run,
    program_blocks,
    program_script,
    read_outcomes,
    run_mark,
    run_requirements,
    verdict,
)


def _run(source: str, timeout: int = 60) -> str:
    return subprocess.run(
        [sys.executable, "-I", "-c", source],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
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
    assert verdict(run, ()) == "environment"


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


# ---- whose failure ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error_type", "module", "message", "expected"),
    [
        ("NameError", "builtins", "name 'client' is not defined", "code"),
        ("SyntaxError", "builtins", "unexpected token", "code"),
        ("TypeError", "builtins", "missing 1 required argument: 'model'", "code"),
        # An SDK without its key raises a TypeError: the environment's.
        (
            "TypeError",
            "builtins",
            "Could not resolve authentication method. Expected either api_key",
            "environment",
        ),
        ("AttributeError", "builtins", "'Agent' object has no attribute 'go'", "code"),
        (
            "AttributeError",
            "builtins",
            "'NoneType' object has no attribute 'x'",
            "unclear",
        ),
        ("ModuleNotFoundError", "builtins", "No module named 'strands.tools'", "code"),
        ("ModuleNotFoundError", "builtins", "No module named 'torch'", "environment"),
        ("ImportError", "builtins", "cannot import name 'X' from 'y'", "unclear"),
        (
            "ImportError",
            "builtins",
            "cannot import name 'HookOrder' from 'strands.hooks' (/x.py)",
            "code",
        ),
        # The product itself missing is an install that failed.
        ("ModuleNotFoundError", "builtins", "No module named 'strands'", "environment"),
        ("SystemExit", "builtins", "2", "unclear"),
        # A namespace package's part: the product's install failed.
        (
            "ModuleNotFoundError",
            "builtins",
            "No module named 'google.adk'",
            "environment",
        ),
        (
            "NoCredentialsError",
            "botocore.exceptions",
            "Unable to locate",
            "environment",
        ),
        ("KeyError", "builtins", "'OPENAI_API_KEY'", "unclear"),
        ("KeyError", "builtins", "missing api key: 'OPENAI_API_KEY'", "environment"),
        (
            "TypeError",
            "builtins",
            "make_agent() got an unexpected keyword argument 'max_tokens'",
            "code",
        ),
        (
            "TypeError",
            "builtins",
            "call() got an unexpected keyword argument 'timeout'",
            "code",
        ),
        (
            "AttributeError",
            "builtins",
            "'Result' object has no attribute 'token_usage'",
            "code",
        ),
        (
            "TypeError",
            "builtins",
            "f() missing 1 required positional argument: 'api_key'",
            "code",
        ),
        ("ConnectionError", "builtins", "refused", "environment"),
        ("ValueError", "builtins", "bad shape", "unclear"),
    ],
)
def test_verdict(error_type: str, module: str, message: str, expected: str) -> None:
    run = BlockRun(
        index=1, ok=False, error_type=error_type, error_module=module, message=message
    )
    assert verdict(run, {"strands", "strands.hooks", "google.adk"}) == expected


def _blocks() -> list[tuple[str, str, bool]]:
    return [("설치", "import os", False), ("사용", "b = os.c", True), ("끝", "d", True)]


def test_a_code_failure_in_a_runnable_section_is_a_finding() -> None:
    run = judge_run(
        _blocks(),
        [
            BlockRun(index=1, ok=True),
            BlockRun(index=2, ok=False, error_type="AttributeError", message="no c"),
        ],
        (),
    )
    (finding,) = run.findings
    assert finding.section == "사용"
    assert "AttributeError" in finding.problem
    assert (run.blocks_run, run.blocks_total, run.limited) == (2, 3, "")


def test_failures_the_post_is_not_blamed_for_limit_the_check() -> None:
    not_runnable = judge_run(
        _blocks(), [BlockRun(index=1, ok=False, error_type="NameError")], ()
    )
    environment = judge_run(
        _blocks(),
        [
            BlockRun(index=1, ok=True),
            BlockRun(index=2, ok=False, error_type="ConnectionError"),
        ],
        (),
    )
    killed = judge_run(_blocks(), [BlockRun(index=1, ok=True)], ())
    complete = judge_run(_blocks(), [BlockRun(index=i, ok=True) for i in (1, 2, 3)], ())
    assert not (not_runnable.findings or environment.findings or killed.findings)
    assert "not meant to run" in not_runnable.limited
    assert "lacks" in environment.limited
    assert "killed" in killed.limited
    assert complete.limited == "" and complete.blocks_run == 3


@pytest.mark.parametrize(
    ("error_type", "message", "counts"),
    [
        ("NameError", "name 'result' is not defined", False),
        ("AttributeError", "'Agent' object has no attribute 'go'", False),
        ("AttributeError", "module 'strands' has no attribute 'Agnet'", True),
        ("TypeError", "Agent.__init__() got an unexpected keyword argument 'x'", True),
        ("TypeError", "unsupported operand type(s)", False),
        ("ModuleNotFoundError", "No module named 'strands.tool'", True),
    ],
)
def test_after_a_gap_only_failures_that_cannot_come_from_it_count(
    error_type: str, message: str, counts: bool
) -> None:
    """The first block cannot reach the model; what a later block does not
    find may be what that block would have made."""
    run = judge_run(
        [("시작", "r = agent('hi')", True), ("다음", "r.message", True)],
        [
            BlockRun(
                index=1,
                ok=False,
                error_type="NoCredentialsError",
                error_module="botocore.exceptions",
            ),
            BlockRun(index=2, ok=False, error_type=error_type, message=message),
        ],
        {"strands"},
    )
    assert bool(run.findings) is counts
    assert ("does not depend on it" in run.findings[0].problem) if counts else True
    assert "NoCredentialsError" in run.limited


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


def test_a_failure_through_an_illustrations_definition_is_not_blamed() -> None:
    """A section not meant to run shows a simplified class of the product;
    the usage section's call fails against it, not against the product."""
    blocks = [
        ("설치", "from strands import Agent", True),
        ("원리", "class Agent:\n    def __init__(self, model):\n        pass", False),
        ("사용", "agent = Agent(a=1)", True),
    ]
    fails = [
        BlockRun(index=1, ok=True),
        BlockRun(index=2, ok=True),
        BlockRun(
            index=3,
            ok=False,
            error_type="TypeError",
            message="Agent.__init__() got an unexpected keyword argument 'a'",
        ),
    ]
    run = judge_run(blocks, fails, {"strands"})
    assert not run.findings and "cannot attribute" in run.limited
    # Imported again in a runnable block, the name is the product's.
    blocks.insert(2, ("다시", "from strands import Agent", True))
    fails = [BlockRun(index=i, ok=True) for i in (1, 2, 3)] + [
        fails[-1].model_copy(update={"index": 4})
    ]
    assert judge_run(blocks, fails, {"strands"}).findings


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


def test_a_name_the_run_skipped_binding_is_not_blamed() -> None:
    blocks = [
        (
            "호출",
            "try:\n    resp = client.list()\nexcept Exception as e:\n    print(e)",
            True,
        ),
        ("출력", "print(resp['Buckets'])\nprint(typo)", True),
    ]
    skipped = BlockRun(
        index=2, ok=False, error_type="NameError", message="name 'resp' is not defined"
    )
    typo = skipped.model_copy(update={"message": "name 'typo' is not defined"})
    ok = BlockRun(index=1, ok=True)
    assert not judge_run(blocks, [ok, skipped], ()).findings
    assert judge_run(blocks, [ok, typo], ()).findings


def test_packages_off_their_versions_blame_only_the_codes_text() -> None:
    blocks = [("사용", "import pandas\npandas.new_api()", True)]
    attr = BlockRun(
        index=1,
        ok=False,
        error_type="AttributeError",
        message="module 'pandas' has no attribute 'new_api'",
    )
    syntax = BlockRun(index=1, ok=False, error_type="SyntaxError", message="bad")
    assert judge_run(blocks, [attr], {"pandas"}).findings
    assert not judge_run(blocks, [attr], {"pandas"}, exact=False).findings
    assert judge_run(blocks, [syntax], (), exact=False).findings


def test_a_module_attribute_a_skipped_import_binds_is_not_blamed() -> None:
    blocks = [
        ("설치", "import pkg", True),
        ("설정", "import os\nos.environ['PKG_ENDPOINT']\nimport pkg.storage", True),
        ("저장", "pkg.storage.save(1)", True),
    ]
    gap = BlockRun(index=2, ok=False, error_type="KeyError", message="'PKG_ENDPOINT'")
    attr = BlockRun(
        index=3,
        ok=False,
        error_type="AttributeError",
        message="module 'pkg' has no attribute 'storage'",
    )
    assert not judge_run(
        blocks, [BlockRun(index=1, ok=True), gap, attr], {"pkg"}
    ).findings
    blocks[1] = ("설정", "import os\nos.environ['PKG_ENDPOINT']", True)
    assert judge_run(blocks, [BlockRun(index=1, ok=True), gap, attr], {"pkg"}).findings


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


def test_a_gap_limits_only_the_blocks_that_use_what_it_binds() -> None:
    """An internals excerpt that cannot run alone leaves the unrelated
    usage block judged in full."""
    blocks = [
        ("구조", "result = self.event_loop_cycle(messages)", False),
        ("사용", "import json\njson.dumps({}).decode()", True),
        ("이어서", "print(result.stop_reason)", True),
    ]
    excerpt = BlockRun(
        index=1, ok=False, error_type="NameError", message="name 'self' is not defined"
    )
    attr = BlockRun(
        index=2,
        ok=False,
        error_type="AttributeError",
        message="'str' object has no attribute 'decode'",
    )
    run = judge_run(blocks, [excerpt, attr], ())
    assert run.findings and "not depend" not in run.findings[0].problem
    uses = BlockRun(
        index=3,
        ok=False,
        error_type="AttributeError",
        message="'Result' object has no attribute 'stop_reason'",
    )
    assert not judge_run(
        blocks, [excerpt, BlockRun(index=2, ok=True), uses], ()
    ).findings


def test_state_a_failed_call_would_have_set_is_not_blamed() -> None:
    blocks = [
        (
            "색인",
            "class DocumentIndex:\n    def build(self, docs):\n        self.vectors = embed(docs)\nindex = DocumentIndex()",
            True,
        ),
        ("빌드", "index.build(['a'])", True),
        ("확인", "print(len(index.vectors))", True),
    ]
    runs = [
        BlockRun(index=1, ok=True),
        BlockRun(
            index=2,
            ok=False,
            error_type="NoCredentialsError",
            error_module="botocore.exceptions",
            message="Unable to locate credentials",
        ),
        BlockRun(
            index=3,
            ok=False,
            error_type="AttributeError",
            message="'DocumentIndex' object has no attribute 'vectors'",
        ),
    ]
    assert not judge_run(blocks, runs, ()).findings


def test_a_gap_does_not_spread_through_builtins_or_modules() -> None:
    blocks = [
        ("설정", "import json\ncfg = {'a': 1}", True),
        ("호출", "print(json.dumps(cfg))\nfetch()", True),
        ("사용", "print(json.dumps([1]).decode())", True),
    ]
    runs = [
        BlockRun(index=1, ok=True),
        BlockRun(index=2, ok=False, error_type="ConnectionError", message="refused"),
        BlockRun(
            index=3,
            ok=False,
            error_type="AttributeError",
            message="'str' object has no attribute 'decode'",
        ),
    ]
    assert judge_run(blocks, runs, ()).findings


@pytest.mark.parametrize(
    "message",
    [
        "requires_access_token() got an unexpected keyword argument 'scope'",
        "count_tokens() missing 1 required positional argument: 'text'",
        "Client.get_credentials() takes 1 positional argument but 2 were given",
    ],
)
def test_a_signature_error_is_the_codes_whatever_the_callee_is_called(
    message: str,
) -> None:
    run = BlockRun(index=1, ok=False, error_type="TypeError", message=message)
    assert verdict(run, ()) == "code"


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
