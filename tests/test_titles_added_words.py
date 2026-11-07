"""A title with a content word the other lacks names another work."""

from __future__ import annotations

import pytest

from scholar_lens.domain.titles import titles_match


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Attention Is All You Need", "Attention Is Not All You Need"),
        (
            "Deep Residual Learning for Image Recognition",
            "Deep Residual Learning for Image Recognition: A Survey",
        ),
        ("Training Compute-Optimal LLMs", "Training Compute-Optimal LLMs Revisited"),
    ],
)
def test_an_added_word_is_another_work(a: str, b: str) -> None:
    assert not titles_match(a, b) and not titles_match(b, a)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Few-Shot Learners", "Few Shot Learners"),
        ("Visualizing Attention", "Visualising Attention"),
        (
            "Language Models are Few-Shot Learners",
            "Language models are few-shot learners.",
        ),
    ],
)
def test_spelling_variants_still_match(a: str, b: str) -> None:
    assert titles_match(a, b)
