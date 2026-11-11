---
layout: post
title: "ToolOrchestra: Elevating Intelligence via Efficient Model and Tool Orchestration"
seo_title: "ToolOrchestra"
subtitle: "ToolOrchestra: 효율적인 모델·도구 오케스트레이션으로 지능의 상한 끌어올리기"
date: 2025-11-26 18:59:46
paper_author: "NVIDIA"
paper_doi: "10.48550/arXiv.2511.21689"
description: "ToolOrchestra는 8B 오케스트레이터를 강화학습으로 훈련해 도구와 GPT-5 같은 강한 모델을 골라 호출하게 합니다. Orchestrator-8B는 HLE에서 37.1%로 GPT-5(35.1%)를 앞서면서 2.5배 효율적입니다."
categories: ["Paper Summaries", "Agentic-AI"]
tags: ["Tool-Orchestration", "Orchestrator-Model", "Multi-Objective-Reward", "Efficiency-Aware-Reinforcement-Learning", "ToolScale-Dataset", "Agentic-AI", "Reasoning-Models"]
cover: /assets/images/agentic-ai.jpg
use_math: true
---
## TL;DR

ToolOrchestra는 80억 파라미터 소형 모델을 강화학습으로 훈련해, 웹 검색이나 코드 인터프리터 같은 도구는 물론 GPT-5 같은 더 강한 모델까지 '도구'로 골라 호출하게 만드는 방법입니다. 정답 여부, 비용·지연, 사용자 선호를 함께 보상합니다. 이렇게 만든 Orchestrator-8B는 HLE에서 37.1%를 기록해 GPT-5(35.1%)를 앞서면서도 2.5배 효율적이고, τ²-Bench와 FRAMES에서는 약 30%의 비용으로 GPT-5를 큰 차이로 앞섭니다.

## 연구 배경

기존 도구 사용 에이전트는 강한 단일 모델에 검색기나 계산기 같은 유틸리티 도구를 붙이는 구조가 대부분입니다. 저자들은 사람이 전문가나 정교한 소프트웨어에 일을 맡기듯, 지능 수준과 가격이 서로 다른 모델들을 도구로 두고 작은 '지휘자' 모델이 하위 문제를 적절히 위임하는 오케스트레이션 패러다임을 제안합니다.

기성 모델에 프롬프트만 주는 방식으로는 이 역할을 잘 해내지 못합니다. 저자들은 HLE 300문제로 예비 실험을 했습니다. 오케스트레이터에게 GPT-5, GPT-5-mini, Qwen3-32B, Qwen2.5-Coder-32B를 도구로 주고 "최저 비용으로 최고 성능을 내라"고 지시했습니다.

![프롬프트 기반 오케스트레이터와 RL 학습 오케스트레이터의 모델별 호출 비율 막대그래프](/assets/images/toolorchestra-elevating-intelligence-via-efficient-model-and-tool-orchestration-summary-1.png)

막대그래프를 보면 호출이 한쪽으로 크게 쏠립니다. GPT-5는 호출의 약 66%를 자기 계열인 GPT-5-mini에 몰아주고, Qwen3-8B는 73%를 가장 강한 GPT-5에 넘깁니다. 저자들은 이를 두 가지 편향으로 해석합니다. 하나는 자기 계열 모델을 선호하는 자기 강화 편향이고, 다른 하나는 비용을 무시하고 가장 강한 도구로 직행하는 경향입니다. 반면 RL로 학습한 Orchestrator-8B는 호출을 비교적 고르게 분산합니다. 비용 효율과 사용자 선호에 따른 제어 가능성도 기존 도구 에이전트 연구에서 거의 다뤄지지 않았습니다.

## 제안 방법

![Orchestrator가 추론과 도구 호출을 반복하며 기본 도구, 전문 LLM, 범용 LLM을 사용하고 세 가지 보상으로 학습되는 구조도](/assets/images/toolorchestra-elevating-intelligence-via-efficient-model-and-tool-orchestration-summary-2.png)

전체 구조도에서 볼 부분은 도구 풀이 세 층으로 나뉜다는 점입니다. 기본 도구(웹 검색, 로컬 검색, 코드 인터프리터, `get_flight_status` 같은 도메인 함수), 전문 LLM(수학·코딩 모델), 범용 LLM(GPT-5, Claude Opus 4.1 등)이 모두 같은 인터페이스로 노출됩니다. 오케스트레이터는 매 턴 체인 오브 소트로 상황을 분석하고, 도구와 인자를 골라 호출한 뒤, 그 결과를 관찰값으로 받습니다. 이 루프는 종료 신호가 오거나 최대 50턴에 이를 때까지 반복됩니다. 각 도구는 이름, 설명, 타입이 지정된 파라미터 스키마를 담은 JSON으로 정의합니다. LLM 도구의 설명은 자동으로 만듭니다. 학습 과제 10개를 샘플링해 해당 LLM이 푼 궤적과 성공 여부를 구한 다음, 다른 LLM이 이를 보고 강점과 약점을 서술합니다.

학습의 핵심은 보상 설계입니다. 궤적 $$\tau$$마다 다음 벡터를 구성합니다.

$$
M^{\tau}=[m^{\tau}_{t_{1}},\ldots,m^{\tau}_{t_{n}},\,r_{\text{outcome}}(\tau),\,r_{\text{compute}}(\tau),\,r_{\text{latency}}(\tau)]
$$

$$m^{\tau}_{t_i}$$는 도구 $$t_i$$의 호출 횟수입니다. $$r_{\text{outcome}}\in\{0,1\}$$은 GPT-5 심판이 판정한 정답 여부입니다. $$r_{\text{compute}}=-\$(\tau)$$는 입출력 토큰을 API 가격으로 환산한 비용의 음수이고, $$r_{\text{latency}}=-\mathit{Clock}(\tau)$$는 소요 시간의 음수입니다. 각 성분은 롤아웃 배치 안에서 min-max 정규화하고, 배치 내 값이 모두 같으면 0으로 둡니다. 최종 보상은 사용자 선호 벡터 $$P=[p_{t_1},\ldots,p_{t_n},p_{\text{outcome}},p_{\text{compute}},p_{\text{latency}}]$$와의 내적입니다.

$$
R(\tau)=\begin{cases}M_{\text{normalized}}^{\tau}\cdot P & \text{if } r_{\text{outcome}}(\tau)=1\\ 0 & \text{otherwise}\end{cases}
$$

> **핵심:** 비용과 선호는 정답을 맞힌 궤적끼리만 비교됩니다. 틀린 궤적은 아무리 싸도 보상이 0이므로, 비용을 아끼려고 정답을 포기하는 정책이 생기기 어렵습니다.

예시 값으로 보면 이해가 쉽습니다. $$p_{\text{outcome}}=p_{\text{compute}}=1$$이고 나머지가 0인 사용자를 가정합니다. 같은 배치에서 두 궤적이 정답을 맞혔고 비용이 각각 10센트와 40센트라면, 정규화 후 싼 궤적은 비용 항에서 1점, 비싼 궤적은 0점을 받습니다. 그래서 싼 궤적이 더 높은 보상을 얻습니다. 특정 도구에 $$p_{t_i}=1$$을 주면 그 도구를 많이 쓴 궤적이 우대됩니다. 예를 들어 사내 기밀 때문에 로컬 검색과 자체 호스팅 모델만 쓰고 싶은 사용자의 선호를 이렇게 반영할 수 있습니다.

정책 최적화에는 GRPO를 사용합니다. 같은 과제에서 생성한 궤적 그룹 안에서 보상을 평균과 표준편차로 정규화해 어드밴티지 $$A(\tau)=(R(\tau)-\text{mean}\,R)/\text{std}\,R$$를 구하고, PPO와 같은 클리핑된 대리 목적함수를 최대화합니다.

## 구현 방식

백본은 Qwen3-8B입니다. GeneralThought-430K와 자체 합성 데이터로 학습했습니다. 설정은 학습률 1e-6, 최대 입력 24,000토큰, 최대 생성 8,000토큰, 학습 배치 16, 롤아웃 배치 8, 롤아웃당 최대 50턴이며 H100 16장을 사용했습니다. KL 손실 폭주를 막기 위해 역전파에서 세 종류의 샘플을 제외합니다.

- 배치 보상의 표준편차가 0.1 미만인 경우(신호가 약함)
- 도구 호출 형식이 어긋난 경우
- 유효한 답이나 출력이 없는 경우

검증 가능한 다중 턴 도구 사용 데이터가 부족하기 때문에, 저자들은 합성 데이터셋 ToolScale을 만들었습니다. LLM이 금융, 의료, 철도 등 10개 도메인마다 데이터베이스 스키마와 레코드, 자주 쓰는 도구 API를 생성합니다. 이어서 도메인 의도를 제안받아 구체적 과제로 바꾸고, 다른 LLM이 제약을 추가해 과제를 더 어렵게 만듭니다. 각 과제는 지시문, 정답 함수 호출 시퀀스, 풀이 중 반드시 언급해야 할 정보로 구성됩니다. 다음 중 하나에 해당하는 과제는 제거합니다. 정답 호출 실행 시 오류가 나는 경우, LLM이 pass@8로도 풀지 못하는 경우, 도구 없이 풀리는 경우입니다. 성공 판정은 세 조건을 모두 만족해야 합니다. 실행 후 DB 상태가 일치해야 하고, 필수 정보를 언급해야 하며, 정답이 조작한 DB 항목을 빠짐없이 조작해야 합니다.

일반화를 위해 학습 인스턴스마다 사용 가능한 도구의 부분집합을 무작위로 바꿉니다. 가격표도 TogetherAI, Fireworks AI 등 10개 서드파티 API 기준으로 다양하게 적용합니다. 선호 학습용으로는 LLM이 (선호 지시문, 선호 벡터) 쌍을 생성하고 다른 LLM이 일관성을 검증합니다. 이 쌍은 학습용과 평가용으로 나눕니다.

## 실험 결과

평가에서는 도구 세트를 고정했습니다. 기본 도구는 Tavily 웹 검색, Python 샌드박스, Qwen3-Embedding-8B 기반 Faiss 로컬 검색입니다. 전문 LLM은 코드 작성용 GPT-5, GPT-5-mini, Qwen2.5-Coder-32B와 수학용 Qwen2.5-Math-72B/7B이고, 범용 LLM은 GPT-5, GPT-5-mini, Llama-3.3-70B, Qwen3-32B입니다. 비교 대상은 같은 도구를 받은 프롬프트 기반 오케스트레이터와 도구 없음 또는 기본 도구만 가진 단일 모델입니다. 추론 온도는 0이고, HLE는 텍스트 전용 부분집합을 사용했습니다.

### 주요 벤치마크

| 모델(도구 구성) | HLE | FRAMES | τ²-Bench | 비용(센트) | 지연(분) |
| --- | --- | --- | --- | --- | --- |
| GPT-5 (도구 없음) | 23.4 | 66.3 | –∗ | 6.2 | 4.1 |
| GPT-5 (기본 도구) | 35.1 | 74.0 | 77.7 | 30.2 | 19.8 |
| Qwen3-8B (도구+모델) | 30.6 | 68.9 | 72.3 | 27.6 | 18.3 |
| Qwen3-235B-A22B (도구+모델) | 32.8 | 74.2 | 75.6 | 29.7 | 21.2 |
| Claude Opus 4.1 (도구+모델) | 34.6 | 72.8 | 76.8 | 52.5 | 25.6 |
| GPT-5 (도구+모델) | 21.2 | 57.5 | 62.3 | 17.8 | 13.6 |
| Orchestrator-8B (도구+모델) | 37.1 | 76.3 | 80.2 | 9.2 | 8.2 |

비용과 지연은 HLE와 FRAMES의 평균이며, 다른 기준선 행은 생략했습니다. ∗τ²-Bench는 도구 없이 풀 수 없습니다.

Orchestrator-8B는 세 벤치마크 모두에서 가장 높습니다. 가장 강한 기준선인 기본 도구를 쓴 GPT-5와 비교하면 비용은 30.2센트에서 9.2센트로 약 30% 수준이고, 지연은 19.8분에서 8.2분으로 줄었습니다. 같은 도구와 모델을 준 프롬프트 기반 GPT-5는 오히려 21.2로 떨어지는데, 저자들은 그 원인을 GPT-5-mini로 기본 위임하는 편향으로 봅니다. τ²-Bench에서 Orchestrator는 GPT-5를 전체 단계의 약 40%에서만 호출하고 나머지는 더 싼 도구로 처리하면서도, 매 단계 GPT-5를 쓰는 에이전트보다 높은 점수를 냈습니다.

### 비용-성능 곡선과 도구 분포

![최대 턴 수를 바꿔 가며 측정한 비용 대비 HLE 정확도 곡선](/assets/images/toolorchestra-elevating-intelligence-via-efficient-model-and-tool-orchestration-summary-3.png)

최대 턴 수를 10, 20, 50, 100으로 바꾸며 HLE 정확도를 비용에 대해 그린 곡선입니다. 같은 예산에서 Orchestrator 곡선이 GPT-5, Claude Opus 4.1, Qwen3-235B-A22B보다 항상 위에 있습니다. 세 벤치마크 평균 호출 분포에서도 차이가 드러납니다. Claude Opus 4.1은 호출의 56.9%를, GPT-5는 52.3%를 각각 한 모델(GPT-5, GPT-5-mini)에 몰아줍니다. Orchestrator-8B는 로컬 검색 22.0%, GPT-5-mini 20.7%, GPT-5 19.5%, Qwen3-32B 15.9%로 고르게 나눕니다.

### 미학습 도구, 가격, 선호에 대한 일반화

학습 때 보지 못한 모델(Claude Sonnet 4.1, Codestral-22B, Gemma-3-27b 등)로 도구 세트를 바꾼 경우에도 Orchestrator-8B는 HLE 22.0, FRAMES 73.8, τ²-Bench 48.8, 비용 34.8을 기록했습니다. GPT-5 오케스트레이터는 16.4, 54.8, 44.8, 비용 81.3이었습니다. 학습에 쓰지 않은 deepinfra 가격표에서도 HLE 36.9, 비용 7.5로 우위가 유지되었습니다. 선호 지시문을 붙인 평가에서는 선호 반영 점수가 HLE 46.7, FRAMES 68.4, τ²-Bench 79.5로, 2위인 Claude Opus 4.1(40.2, 63.4, 73.5)보다 높았습니다.

## 의의와 한계

이 연구는 작은 모델을 "모든 걸 직접 푸는 모델"이 아니라 "누구에게 맡길지 아는 모델"로 훈련하면, 프런티어 모델 단독 사용보다 정확도와 비용을 동시에 개선할 수 있음을 실증했습니다. 보상을 선호 벡터와의 내적으로 정의했기 때문에, 정확도·비용·지연·특정 도구 선호를 하나의 틀에서 사용자별로 조정할 수 있습니다. 모델, 데이터(ToolScale), 코드도 공개되었습니다.

한계도 분명합니다.

- **비교 조건 차이:** 기존 보고된 GPT-5 SOTA(HLE 35.2)는 전체 세트 기준이고, 이 논문의 수치는 텍스트 전용 부분집합 기준이라 직접 비교할 수 없습니다. τ²-Bench에서도 GPT-5의 보고치 84.2를 재현하지 못하고 77.7을 얻었습니다.
- **약한 기준선 가능성:** 비교 대상 오케스트레이터는 모두 프롬프트 기반입니다. 프롬프트 기반 GPT-5가 도구를 더 받고 오히려 성능이 떨어진 결과를 보면, 기준선의 프롬프트 설계가 우위 폭에 영향을 줬을 수 있습니다.
- **분석 부족:** 보상 성분별 기여를 분리한 절제 실험은 제시되지 않았습니다.
- **합성 데이터와 LLM 판정 의존:** 학습 데이터, 도구 설명, 선호 쌍을 모두 LLM이 생성·검증하고, 정답 판정도 GPT-5 심판에 맡기므로 생성 모델의 편향이 그대로 섞일 수 있습니다.
- **학습·평가 지표 불일치:** 선호 평가 지표는 Qwen3-8B 출발점 대비 상대값으로 정의되어 학습 보상과 형태가 다릅니다. 따라서 점수의 절대적 의미를 해석할 때 주의가 필요합니다.
- **외부 모델 의존:** 비용 우위는 여전히 외부 GPT-5 호출을 포함한 값이므로, 사용하는 API의 가격 구조에 따라 달라집니다.

- - -

### References
* [ToolOrchestra: Elevating Intelligence via Efficient Model and Tool Orchestration](https://arxiv.org/pdf/2511.21689v1)
* [Code](https://github.com/NVlabs/ToolOrchestra/)
* [Model](https://huggingface.co/nvidia/Orchestrator-8B)
* [Data](https://huggingface.co/datasets/nvidia/ToolScale)
* [Webpage](https://research.nvidia.com/labs/lpr/ToolOrchestra)
