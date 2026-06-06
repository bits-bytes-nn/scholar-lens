"""Tests for configuration loading and the cover-image helper."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from scholar_lens.configs import Code, Config, Explanation, Github
from scholar_lens.src.constants import LanguageModelId


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

    def test_category_labels_default_and_override(self) -> None:
        assert Github().review_category == "Paper Reviews"
        assert Github().summary_category == "Paper Summaries"
        assert Github().tech_guide_category == "Tech Guides"
        # A blog whose summary tab reads site.categories['Summaries'] can align.
        gh = Github(summary_category="Summaries", tech_guide_category="Guides")
        assert gh.summary_category == "Summaries"
        assert gh.tech_guide_category == "Guides"


class TestGithubCoverImages:
    def test_known_category_maps_to_configured_image(self) -> None:
        gh = Github(
            cover_images={"language-models": "language-models.jpg"},
            default_cover_image="default.jpg",
        )
        assert gh.cover_image_for("Language Models") == "language-models.jpg"

    def test_slug_normalisation_is_case_and_space_insensitive(self) -> None:
        gh = Github(cover_images={"multimodal-learning": "mm.jpg"})
        assert gh.cover_image_for("  Multimodal   Learning ") == "mm.jpg"

    def test_unknown_category_falls_back_to_default(self) -> None:
        gh = Github(default_cover_image="fallback.jpg")
        assert gh.cover_image_for("Quantum Widgets") == "fallback.jpg"

    def test_none_cover_images_coerced_to_empty_dict(self) -> None:
        # Mirrors a YAML key written as `cover_images:` with no value.
        gh = Github(cover_images=None)  # type: ignore[arg-type]
        assert gh.cover_images == {}
        assert gh.cover_image_for("anything") == "default.jpg"


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

    def test_invalid_model_id_rejected(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "explanation:\n" "  paper_finalization_model_id: not-a-real-model\n",
            encoding="utf-8",
        )
        with pytest.raises(Exception):
            Config.from_yaml(str(cfg_file))
