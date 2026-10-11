"""Tests for CodeRetriever's clone-URL safety guard (no network)."""

from __future__ import annotations

import pytest

from scholar_lens.src.code_retriever import CodeRetriever
from scholar_lens.src.url_guard import UnsafeUrlError


class TestAssertClonable:
    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "ssh://git@github.com/org/repo.git",
            "git://github.com/org/repo.git",
            "http://github.com/org/repo.git",  # non-https downgrade
            "http://169.254.169.254/repo.git",  # internal literal
        ],
    )
    def test_rejects_unsafe_urls(self, url: str) -> None:
        with pytest.raises(UnsafeUrlError):
            CodeRetriever._assert_clonable(url)

    def test_accepts_public_https_forge_url(self) -> None:
        # A normal public https forge URL passes (github.com resolves publicly).
        CodeRetriever._assert_clonable("https://github.com/org/repo.git")


class TestPythonFiles:
    def test_skips_symlinks_escapes_and_vendored_dirs(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from scholar_lens.src.code_retriever import python_files

        repo = tmp_path / "repo"
        (repo / "pkg").mkdir(parents=True)
        (repo / "pkg" / "model.py").write_text("x = 1")
        (repo / ".venv" / "lib").mkdir(parents=True)
        (repo / ".venv" / "lib" / "dep.py").write_text("y = 2")
        (repo / "node_modules").mkdir()
        (repo / "node_modules" / "n.py").write_text("z = 3")
        secret = tmp_path / "secret.txt"
        secret.write_text("SECRET=hunter2")
        (repo / "evil.py").symlink_to(secret)

        assert [p.relative_to(repo).as_posix() for p in python_files(repo)] == [
            "pkg/model.py"
        ]

    def test_lora_like_repo_is_bounded_and_ranked(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """microsoft/LoRA vendors a whole transformers copy under examples/;
        only the repo's own code should be indexed, README-mentioned first."""
        from scholar_lens.src.code_retriever import python_files

        repo = tmp_path / "LoRA"

        def write(rel: str, text: str = "x = 1") -> None:
            path = repo / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

        write("README.md", "Install loralib, then see loralib/layers.py.")
        write("loralib/__init__.py")
        write("loralib/layers.py")
        write("loralib/utils.py")
        write("setup.py")  # the repo's own project file is not "vendored"
        write("examples/NLU/setup.py")  # a packaged copy of another project
        write("examples/NLU/src/transformers/modeling_bert.py")
        write("examples/NLG/src/transformers/x.py")
        write("examples/NLG/src/gpt2_ft.py")
        write("tests/test_layers.py")
        write("loralib/big.py", "x = 1\n" * 30_000)  # > 100 KB

        files = [p.relative_to(repo).as_posix() for p in python_files(repo)]
        assert files[:3] == [
            "loralib/__init__.py",
            "loralib/layers.py",
            "loralib/utils.py",
        ]
        assert "examples/NLG/src/gpt2_ft.py" in files
        assert not any(
            "transformers" in f or f.startswith("examples/NLU") for f in files
        )
        assert "tests/test_layers.py" not in files and "loralib/big.py" not in files
        assert len(python_files(repo, max_files=2)) == 2

    async def test_chunks_are_capped(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from scholar_lens.src.code_retriever import CodeRetriever

        repo = tmp_path / "r"
        (repo / "pkg").mkdir(parents=True)
        for i in range(5):
            (repo / "pkg" / f"m{i}.py").write_text("def f():\n    return 1\n\n" * 200)
        retriever = CodeRetriever.__new__(CodeRetriever)
        retriever.max_files, retriever.max_chunks = 10, 7
        retriever._initialize_text_splitter(200, 20)
        chunks = await retriever._create_documents([repo])
        assert len(chunks) == 7
        assert chunks[0].metadata["source"] == "pkg/m0.py"


class TestGitClone:
    async def test_timeout_kills_git_and_args_disable_redirects(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:  # type: ignore[no-untyped-def]
        import os

        import scholar_lens.src.code_retriever as cr

        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        args_file = tmp_path / "args.txt"
        fake_git = bin_dir / "git"
        fake_git.write_text(f'#!/bin/sh\necho "$@" > {args_file}\nexec sleep 30\n')
        fake_git.chmod(0o755)
        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
        monkeypatch.setattr(cr, "_CLONE_TIMEOUT_SECONDS", 1)

        with pytest.raises(TimeoutError, match="exceeded 1s"):
            await cr.CodeRetriever._git_clone(
                "https://github.com/o/r", tmp_path / "out"
            )
        args = args_file.read_text()
        assert "http.followRedirects=false" in args and "core.symlinks=false" in args
        assert "--depth=1" in args

    async def test_nonzero_exit_raises(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
        import os

        import scholar_lens.src.code_retriever as cr

        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "git").write_text(
            "#!/bin/sh\necho 'fatal: not found' >&2\nexit 128\n"
        )
        (bin_dir / "git").chmod(0o755)
        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
        with pytest.raises(RuntimeError, match="fatal: not found"):
            await cr.CodeRetriever._git_clone(
                "https://github.com/o/r", tmp_path / "out"
            )
