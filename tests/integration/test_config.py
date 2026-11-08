"""Tests for configuration loading and the cover-image helper."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from scholar_lens.configs import Code, Config, Explanation, Github
from scholar_lens.configs import TechGuide as TechGuideConfig
from scholar_lens.configs.config import CONFIG_PATH, ConfigFileError
from scholar_lens.domain.constants import LanguageModelId


class TestConfigValidation:
    def test_chunk_overlap_must_be_less_than_size(self) -> None:
        with pytest.raises(ValidationError):
            Code(chunk_size=1024, chunk_overlap=2048)

    def test_chunk_size_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            Code(chunk_size=0, chunk_overlap=0)

    def test_valid_chunk_config_accepted(self) -> None:
        c = Code(chunk_size=1024, chunk_overlap=256)
        assert c.chunk_size == 1024 and c.chunk_overlap == 256

    def test_max_total_tokens_must_be_positive_when_set(self) -> None:
        with pytest.raises(ValidationError):
            Explanation(
                paper_finalization_model_id=LanguageModelId.CLAUDE_V4_5_HAIKU,
                max_total_tokens=-1000,
            )

    def test_max_total_tokens_none_is_allowed(self) -> None:
        e = Explanation(
            paper_finalization_model_id=LanguageModelId.CLAUDE_V4_5_HAIKU,
            max_total_tokens=None,
        )
        assert e.max_total_tokens is None

    def test_every_artifact_has_a_default_spend_cap(self) -> None:
        config = Config()
        for section in (config.explanation, config.summary):
            assert section.max_total_tokens
        assert config.tech_guide.max_cost_usd and config.survey.write_max_cost_usd

    @pytest.mark.parametrize(
        ("section", "key"),
        [
            ("tech_guide", "engine"),
            ("survey", "engine"),
            ("survey", "verify_grounding"),
        ],
    )
    def test_a_key_of_a_removed_engine_fails_at_load(
        self, tmp_path: Path, section: str, key: str
    ) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(f"{section}:\n  {key}: x\n", encoding="utf-8")
        # The error names the file and the key, without pydantic's dump.
        with pytest.raises(ConfigFileError, match=f"config.yaml: {section}.{key}"):
            Config.from_yaml(str(cfg_file))

    def test_guide_research_defaults_read_the_top_results_of_each_query(
        self,
    ) -> None:
        # Pages are taken round-robin across the queries, so this many reads
        # each planned query's (each research intent's) top results.
        guide = Config().tech_guide
        assert guide.fetch_top_results >= 2 * guide.max_research_queries

    @pytest.mark.parametrize("value", [2, 31])
    def test_guide_max_sections_is_bounded(self, value: int) -> None:
        with pytest.raises(ValidationError):
            TechGuideConfig(max_sections=value)
        assert TechGuideConfig(max_sections=20).max_sections == 20

    def test_category_labels_default_and_override(self) -> None:
        assert Github().review_category == "Paper Reviews"
        assert Github().summary_category == "Paper Summaries"
        assert Github().tech_guide_category == "Tech Guides"
        # A blog whose summary tab reads site.categories['Summaries'] can align.
        gh = Github(summary_category="Summaries", tech_guide_category="Guides")
        assert gh.summary_category == "Summaries"
        assert gh.tech_guide_category == "Guides"


class TestGithubConfig:
    def test_cover_keys_are_gone(self) -> None:
        # Covers come from the blog's post conventions now; stale keys fail.
        with pytest.raises(ValueError):
            Github(cover_images={"x": "y.jpg"})  # type: ignore[call-arg]


class TestRetiredKeys:
    def test_a_retired_key_is_ignored_with_a_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "resources:\n  project_name: p\n  github:\n    public_assets: true\n",
            encoding="utf-8",
        )
        with caplog.at_level("WARNING", logger="scholar_lens.configs.config"):
            config = Config.from_yaml(str(cfg_file))
        assert "public_assets" not in config.resources.github.model_dump()
        assert "public_assets" in caplog.text and "ignored" in caplog.text

    def test_an_unknown_key_still_fails(self) -> None:
        with pytest.raises(ValidationError, match="public_asset"):
            Github(public_asset=True)  # type: ignore[call-arg]

    def test_retired_keys_are_per_section(self) -> None:
        # Only the section that had the key accepts it.
        with pytest.raises(ValidationError, match="public_assets"):
            Config.model_validate({"paper": {"public_assets": True}})


class TestConfigLoading:
    def test_from_yaml_parses_models(self, tmp_path: Path) -> None:
        cfg_text = textwrap.dedent("""
            resources:
              project_name: test-lens
              github:
                enabled: true
                repo_name: owner/blog
            explanation:
              paper_finalization_model_id: anthropic.claude-haiku-4-5-20251001-v1:0
              paper_synthesis_model_id: anthropic.claude-opus-4-8
            """)
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(cfg_text, encoding="utf-8")
        config = Config.from_yaml(str(cfg_file))
        assert config.resources.project_name == "test-lens"
        assert config.resources.github.enabled is True
        assert (
            config.explanation.paper_synthesis_model_id.value
            == "anthropic.claude-opus-4-8"
        )

    def test_empty_yaml_uses_defaults(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("", encoding="utf-8")
        config = Config.from_yaml(str(cfg_file))
        assert config.resources.project_name == "scholar-lens"

    @pytest.mark.parametrize(
        "model_id",
        [
            "anthropic.claude-opus-5-5",
            "anthropic.claude-sonnet-5-5",
            "anthropic.claude-haiku-5-5",
            "anthropic.claude-fable-5-1",
        ],
    )
    @pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
    def test_latest_models_and_efforts_selectable(
        self, tmp_path: Path, model_id: str, effort: str
    ) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            f"summary:\n  summary_model_id: {model_id}\n  thinking_effort: {effort}\n",
            encoding="utf-8",
        )
        config = Config.from_yaml(str(cfg_file))
        assert config.summary.summary_model_id.value == model_id
        assert config.summary.thinking_effort == effort

    def test_unknown_effort_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Explanation(thinking_effort="extreme")  # type: ignore[arg-type]

    def test_invalid_model_id_rejected(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "explanation:\n" "  paper_finalization_model_id: not-a-real-model\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigFileError, match="paper_finalization_model_id"):
            Config.from_yaml(str(cfg_file))


def test_scholar_lens_config_names_another_config_file(tmp_path, monkeypatch) -> None:
    from scholar_lens.configs import Config
    from scholar_lens.configs.config import CONFIG_PATH_ENV

    path = tmp_path / "eval.yaml"
    path.write_text("survey:\n  min_papers: 7\n", encoding="utf-8")
    # Loading .env would leak this checkout's API keys into later tests.
    monkeypatch.setattr("scholar_lens.configs.config.load_dotenv", lambda: None)
    monkeypatch.setenv(CONFIG_PATH_ENV, str(path))
    assert Config.load().survey.min_papers == 7


def test_a_github_config_that_cannot_open_a_pr_fails_at_load() -> None:
    import pytest
    from pydantic import ValidationError

    from scholar_lens.configs.config import Github

    with pytest.raises(ValidationError, match="owner/name"):
        Github(enabled=True)
    with pytest.raises(ValidationError, match="branch_prefix"):
        Github(branch_prefix="/")
    assert Github(enabled=True, repo_name="owner/owner.github.io").enabled


@pytest.mark.parametrize("section", ["explanation", "summary", "tech_guide"])
def test_a_length_range_upside_down_fails_at_load(section: str) -> None:
    from pydantic import ValidationError

    from scholar_lens.configs.config import Config

    with pytest.raises(ValidationError, match="min_total_chars"):
        Config.model_validate(
            {section: {"min_total_chars": 60_000, "max_total_chars": 30_000}}
        )


@pytest.mark.parametrize("prefix", ["*", "a b", "a//b", "/x", "x/", ""])
def test_an_s3_prefix_outside_the_key_grammar_fails(prefix: str) -> None:
    from pydantic import ValidationError

    from scholar_lens.configs.config import Resources

    with pytest.raises(ValidationError):
        Resources(project_name="p", s3_prefix=prefix)


def test_a_repo_name_needs_an_owner_and_a_name() -> None:
    from pydantic import ValidationError

    from scholar_lens.configs.config import Github

    for name in ("owner/", "/repo", "a/b/c"):
        with pytest.raises(ValidationError):
            Github(enabled=True, repo_name=name)


def _leaves(data: object, path: str = "") -> dict[str, object]:
    """Every leaf value of a dumped config, by its dotted key."""
    if isinstance(data, dict):
        out: dict[str, object] = {}
        for key, value in data.items():
            out |= _leaves(value, f"{path}.{key}" if path else str(key))
        return out
    return {path: data}


def test_the_whole_config_template_is_the_code_defaults() -> None:
    # Every key the template sets has the code's default, so a config with
    # a section left out behaves as the template documents it.
    template = Config.from_yaml(CONFIG_PATH.with_name("config-template.yaml"))
    given = _leaves(template.model_dump(mode="json"))
    defaults = _leaves(Config().model_dump(mode="json"))
    differ = {
        key: (given.get(key), defaults.get(key))
        for key in given.keys() | defaults.keys()
        if given.get(key) != defaults.get(key)
    }
    assert differ == {}
    assert template == Config()


def test_default_demoted_domains_are_copies() -> None:
    first, second = Config().tech_guide, Config().tech_guide
    first.demoted_domains["forum"].append("example.org")
    assert "example.org" not in second.demoted_domains["forum"]
    assert set(second.demoted_domains) == {"forum", "social", "aggregator"}
