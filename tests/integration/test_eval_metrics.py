"""The evaluation's deterministic metrics run on a post (they rotted once
unnoticed: ADR 0002 reads them before any judge)."""

from __future__ import annotations

from eval.metrics import measure

POST = """---
layout: post
title: "DPO"
---

## TL;DR

### 무엇을 해결하나요?

DPO는 보상 모델 없이 선호를 직접 최적화합니다. 승률은 61%입니다.

## 보상 모델 없이 정책을 직접 맞추는 목적 함수

선호 쌍만으로 정책을 학습합니다. 이 방식은 PPO보다 단순하고, 학습이 안정적입니다.
요약 과제에서 사람 평가 승률은 61%였습니다.
"""


def test_the_metrics_measure_a_post() -> None:
    metrics = measure(POST, "review")
    assert metrics["prose_chars"] > 50
    assert metrics["sections"] >= 2
