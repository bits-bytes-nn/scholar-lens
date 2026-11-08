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


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (
            "LoRA: Low-Rank Adaptation of Large Language Models",
            "Low-Rank Adaptation of Large Language Models",
        ),
        (
            "Efficient Serving with PagedAttention (vLLM)",
            "Efficient Serving with PagedAttention",
        ),
        ("Cats vs. Dogs Classification", "Cats versus Dogs Classification"),
        ("GPT-4 Technical Report", "GPT4 Technical Report"),
        ("Llama 2: Open Foundation Models", "Llama2: Open Foundation Models"),
    ],
)
def test_a_short_name_spelling_or_version_spacing_is_the_same_paper(
    a: str, b: str
) -> None:
    assert titles_match(a, b) and titles_match(b, a)


def test_two_different_short_names_are_two_papers() -> None:
    assert not titles_match(
        "Alpaca: A Strong, Replicable Instruction-Following Model",
        "Vicuna: A Strong, Replicable Instruction-Following Model",
    )
    assert not titles_match("Fast Serving (vLLM)", "Fast Serving (SGLang)")


def test_a_title_that_differs_by_without_is_another_paper() -> None:
    from scholar_lens.domain.titles import titles_match

    assert not titles_match(
        "Learning with Noisy Labels", "Learning without Noisy Labels"
    )
    assert titles_match("Learning with Noisy Labels", "Learning with noisy labels")


@pytest.mark.parametrize(
    "a, b",
    [
        # The same words in another order are another paper.
        ("Graphs for Deep Learning", "Deep Learning for Graphs"),
        ("Language Models for Code Generation", "Code Generation for Language Models"),
    ],
)
def test_titles_with_the_same_words_in_another_order_do_not_match(
    a: str, b: str
) -> None:
    from scholar_lens.domain.titles import titles_match

    assert not titles_match(a, b)
