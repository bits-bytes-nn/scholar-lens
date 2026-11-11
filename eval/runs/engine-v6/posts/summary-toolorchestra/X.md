---
layout: post
title: "ToolOrchestra: Elevating Intelligence via Efficient Model and Tool Orchestration"
seo_title: "ToolOrchestra: Efficient Model and Tool Orchestration for Agents"
subtitle: "ToolOrchestra: 효율적인 모델·도구 오케스트레이션으로 지능의 상한을 끌어올리기"
date: 2025-11-26 18:59:46
paper_author: "NVIDIA"
paper_doi: "10.48550/arXiv.2511.21689"
description: "ToolOrchestra는 8B 오케스트레이터를 강화학습으로 훈련해 웹 검색, 코드 실행기, 전문 모델, GPT-5 같은 범용 LLM을 비용과 사용자 선호에 맞춰 조율합니다. HLE에서 37.1%를 기록해 GPT-5(35.1%)를 앞서면서 2.5배 효율적입니다."
categories: ["Paper Summaries", "Agentic-AI"]
tags: ["Tool-Orchestration", "Orchestrator-Model", "Efficiency-Aware-Reward", "User-Preference-Alignment", "Synthetic-Tool-Use-Environments", "Agentic-AI", "Reasoning-Models"]
cover: /assets/images/agentic-ai.jpg
use_math: true
---
## TL;DR

ToolOrchestra는 80억 파라미터 소형 모델을 '오케스트레이터'로 강화학습시켜, 웹 검색과 코드 실행기 같은 기본 도구부터 수학·코딩 전문 모델, GPT-5 같은 범용 LLM까지 필요할 때만 골라 호출하게 만드는 방법입니다. 보상에 정답 여부, 비용·지연 시간, 사용자 선호를 함께 넣었습니다. 그 결과 Orchestrator-8B는 HLE에서 37.1%로 GPT-5(35.1%)를 넘으면서 2.5배 효율적이었고, τ²-Bench와 FRAMES에서는 약 30%의 비용으로 GPT-5를 앞섰습니다.

## 연구 배경

기존 도구 사용 에이전트 연구는 대부분 강력한 단일 모델에 웹 검색이나 계산기 같은 유틸리티를 붙이는 구조였습니다. 저자들은 사람이 전문가나 정교한 소프트웨어처럼 자신보다 뛰어난 자원을 빌려 문제를 풀듯이, 모델도 다른 모델을 '지능형 도구'로 부를 수 있다고 봅니다. 이때 핵심 과제는 비용이 서로 다른 도구 가운데 무엇을 언제 부를지 정하는 일입니다.

가장 쉬운 방법은 기성 LLM에 프롬프트만 주어 오케스트레이터로 쓰는 것인데, 저자들의 예비 실험에서 이 방식은 심하게 편향되었습니다. HLE 300문항에서 GPT-5, GPT-5-mini, Qwen3-32B, Qwen2.5-Coder-32B를 도구로 주고 "비용을 최소화하며 최선의 결과를 내라"고 지시했지만 결과는 아래와 같았습니다.

![프롬프트 기반 GPT-5·Qwen3-8B와 RL로 훈련한 Orchestrator-8B가 각 하위 모델을 호출한 비율 막대 그래프](/assets/images/toolorchestra-elevating-intelligence-via-efficient-model-and-tool-orchestration-summary-1.png)

GPT-5는 호출의 대부분을 GPT-5-mini에 몰아주었고, Qwen3-8B는 약 73%를 GPT-5에 넘겼습니다. 반면 훈련된 Orchestrator-8B는 GPT-5, GPT-5-mini, Qwen3-32B에 호출을 비교적 고르게 나눕니다. 저자들은 자기 계열 모델을 과하게 쓰는 경향과 비용과 무관하게 가장 강한 도구로 쏠리는 경향을 문제로 지적하고, 이런 상황에는 별도의 훈련이 필요하다고 결론 내립니다.

## 제안 방법

문제는 다중 턴 마르코프 결정 과정으로 정의됩니다. 매 턴 오케스트레이터는 체인 오브 소트로 상황을 분석하고, 도구를 골라 인자를 채운 뒤, 환경이 돌려준 결과를 다음 턴의 입력으로 받습니다. 이 루프는 종료 신호가 오거나 최대 50턴에 이를 때까지 반복됩니다. 목표는 정답률과 사용자 선호 부합도를 높이고 총비용과 총지연 시간은 줄이는 것입니다.

![사용자 질의가 오케스트레이터의 추론–도구 호출–응답 루프를 거쳐 기본 도구, 전문 LLM, 범용 LLM과 상호작용하고, 결과·효율·선호 보상이 RL로 되돌아가는 구조도](/assets/images/toolorchestra-elevating-intelligence-via-efficient-model-and-tool-orchestration-summary-2.png)

구조도에서 주목할 부분은 세 종류의 도구가 같은 인터페이스로 묶여 있다는 점입니다. 모든 도구는 이름, 설명, 타입이 붙은 파라미터 스키마를 담은 JSON으로 정의됩니다. LLM 도구의 설명은 자동으로 만듭니다. 훈련 과제 10개를 무작위로 뽑아 해당 LLM이 푼 궤적과 성공 여부를 수집하고, 다른 LLM에게 이를 바탕으로 "수학은 강하지만 화학 명명법에 약하다"는 식의 능력 설명을 쓰게 합니다.

방법의 중심은 보상 설계입니다. 궤적 $$\tau$$마다 다음 벡터를 구성합니다.

$$
M^{\tau}=[m^{\tau}_{t_{1}},\ldots,m^{\tau}_{t_{n}},\,r_{\text{outcome}}(\tau),\,r_{\text{compute}}(\tau),\,r_{\text{latency}}(\tau)]
$$

$$m^{\tau}_{t_i}$$는 도구 $$t_i$$의 호출 횟수입니다. $$r_{\text{outcome}}\in\{0,1\}$$은 GPT-5 판정기가 매긴 정답 여부이고, $$r_{\text{compute}}=-\$(\tau)$$는 토큰을 API 가격으로 환산한 비용에 음수를 취한 값, $$r_{\text{latency}}=-\mathit{Clock}(\tau)$$는 소요 시간에 음수를 취한 값입니다. 각 성분은 롤아웃 배치 $$\mathrm{T}$$ 안에서 최솟값과 최댓값으로 정규화하며, 배치 내 값이 모두 같으면 0으로 둡니다. 최종 보상은 다음과 같습니다.

$$
R(\tau)=\begin{cases}M^{\tau}_{\text{normalized}}\cdot P & \text{if } r_{\text{outcome}}(\tau)=1\\ 0 & \text{otherwise}\end{cases}
$$

$$P=[p_{t_1},\ldots,p_{t_n},p_{\text{outcome}},p_{\text{compute}},p_{\text{latency}}]$$는 0과 1 사이 값으로 이루어진 선호 벡터입니다. 예를 들어 $$p_{t_1}=1$$은 도구 $$t_1$$을 쓰고 싶다는 뜻이고, $$p_{\text{compute}}=0$$은 비용을 신경 쓰지 않는다는 뜻입니다. 틀린 궤적은 다른 성분이 아무리 좋아도 보상이 0입니다.

설명용 가상 수치로 예를 들어 보겠습니다. 선호가 "정확도와 비용만 중시"($$p_{\text{outcome}}=p_{\text{compute}}=1$$)이고 배치 안에 정답 궤적 A(비용 $0.10)와 B(비용 $0.40), 오답 궤적 C가 있다고 하겠습니다. 비용 성분은 A가 1, B가 0으로 정규화되고 정답 성분은 둘 다 1이므로 $$R(A)=2$$, $$R(B)=1$$, $$R(C)=0$$이 됩니다. 같은 정답이라도 더 싸게 푼 궤적이 더 큰 보상을 받습니다.

> **핵심:** 비용과 도구 선호를 고정 가중치로 섞지 않고, 배치 내 정규화 후 사용자별 선호 벡터와 내적합니다. 그래서 같은 정책이 질의마다 붙는 선호 지시에 따라 다른 도구 조합을 고르도록 학습됩니다.

이 보상으로 GRPO를 적용합니다. 같은 과제에서 생성한 궤적 그룹 안에서 $$A(\tau)=(R(\tau)-\text{mean}\,R)/\text{std}\,R$$로 어드밴티지를 구하고, 클리핑된 대리 목적 함수를 최대화합니다.

## 구현 방식

백본은 Qwen3-8B이고, GeneralThought-430K와 합성 데이터를 함께 써서 훈련합니다. 학습률은 1e-6, 최대 입력 길이는 24,000, 최대 생성 길이는 8,000 토큰입니다. 훈련 배치 크기는 16, 롤아웃 배치 크기는 8이고, 롤아웃은 최대 50턴이며 H100 16장을 사용했습니다. 에이전트 RL에서 KL 손실이 폭주하지 않도록 역전파 전에 세 종류의 샘플을 걸러냅니다.

- 배치 내 보상 표준편차가 0.1 미만인 경우: 신호가 약하기 때문입니다.
- 도구 호출 형식을 어긴 출력
- 유효한 답을 내지 못한 출력

검증 가능한 다중 턴 도구 사용 데이터가 부족해서 저자들은 합성 데이터셋 ToolScale을 만들었습니다. 금융, 스포츠, 의료, 철도, 여행 등 10개 도메인마다 LLM이 데이터베이스 스키마와 레코드, 자주 쓰이는 도구 API를 생성합니다. 이어서 도메인의 전형적인 의도를 제안하고, 이를 구체적인 과제로 바꾼 뒤 제약을 추가해 난도를 높입니다. 각 과제에는 지시문, 정답 함수 호출 시퀀스, 풀이 중 반드시 언급해야 할 정보가 함께 들어갑니다. 정답 호출을 실행하면 오류가 나는 과제, LLM이 pass@8로도 못 푸는 과제, 행동 없이도 풀리는 과제는 제거합니다. 궤적이 성공으로 인정되려면 실행 후 DB 상태가 정답과 일치하고, 필수 정보를 언급했고, 정답이 건드린 레코드를 모두 처리해야 합니다.

일반화를 위해 훈련 인스턴스마다 사용 가능한 도구 부분집합과 가격표(10개 서드파티 API 공급자 기준)를 무작위로 바꿉니다. 사용자 선호 데이터는 LLM이 "기밀 정보가 있으니 API 호출을 피하고 로컬 모델을 써 달라" 같은 선호 지시와 대응하는 선호 벡터 쌍을 만들고, 다른 LLM이 둘의 일관성을 검증하는 방식으로 구성합니다.

## 실험 결과

평가에서는 도구 집합을 고정합니다. 기본 도구는 Tavily 웹 검색, Python 샌드박스, Qwen3-Embedding-8B 기반 Faiss 로컬 검색, 도메인 함수입니다. 전문 LLM으로 GPT-5/GPT-5-mini/Qwen2.5-Coder-32B 코드 작성기와 Qwen2.5-Math-72B/7B를 두고, 범용 LLM으로 GPT-5, GPT-5-mini, Llama-3.3-70B, Qwen3-32B를 둡니다. 비교 대상은 같은 도구 집합을 프롬프트로 다루는 여러 LLM과 도구 없이 또는 기본 도구만 쓰는 단일 모델입니다. 온도는 0이고 오픈 모델 가격은 TogetherAI 기준입니다.

### 주요 비교

| 도구 구성 | 모델 | HLE | FRAMES | τ²-Bench | 비용(센트) | 지연(분) |
| --- | --- | --- | --- | --- | --- | --- |
| 기본 도구 | GPT-5 | 35.1 | 74.0 | 77.7 | 30.2 | 19.8 |
| 기본 도구 | Claude Opus 4.1 | 19.8 | 63.5 | 46.0 | 76.2 | 32.5 |
| 기본+전문+범용 LLM | Qwen3-8B | 30.6 | 68.9 | 72.3 | 27.6 | 18.3 |
| 기본+전문+범용 LLM | Qwen3-235B-A22B | 32.8 | 74.2 | 75.6 | 29.7 | 21.2 |
| 기본+전문+범용 LLM | Claude Opus 4.1 | 34.6 | 72.8 | 76.8 | 52.5 | 25.6 |
| 기본+전문+범용 LLM | GPT-5 | 21.2 | 57.5 | 62.3 | 17.8 | 13.6 |
| 기본+전문+범용 LLM | Orchestrator-8B | 37.1 | 76.3 | 80.2 | 9.2 | 8.2 |

도구 없음 구성과 Llama 계열 일부 행은 생략했습니다. 비용과 지연은 HLE와 FRAMES의 평균이며, HLE는 텍스트 전용 부분집합입니다.

첫째, Orchestrator-8B는 세 벤치마크 모두에서 가장 높은 점수를 기록했습니다. HLE에서 기본 도구를 쓴 GPT-5보다 2.0점 높으면서 비용은 30.2센트에서 9.2센트로 약 30% 수준이고, 지연은 19.8분에서 8.2분으로 줄었습니다. 같은 도구 집합을 프롬프트로 다룬 GPT-5는 21.2로 오히려 떨어졌는데, 저자들은 그 원인으로 GPT-5-mini에 기대는 편향을 꼽습니다. 인스턴스당 평균 호출 횟수를 보면 이 GPT-5는 GPT-5-mini를 5.6회, Claude Opus 4.1은 GPT-5를 6.2회 부르는 반면 Orchestrator-8B는 GPT-5를 1.6회만 부르고 로컬 검색(1.8회)과 Qwen3-32B(1.3회)를 섞어 씁니다.

![최대 허용 턴 수를 늘려 가며 측정한 HLE 정확도와 비용의 관계를 Orchestrator-8B, GPT-5, Claude Opus 4.1, Qwen3-235B-A22B 네 곡선으로 비교한 그래프](/assets/images/toolorchestra-elevating-intelligence-via-efficient-model-and-tool-orchestration-summary-3.png)

둘째, 최대 턴 수를 10, 20, 50, 100으로 바꿔 가며 예산별로 비교하면 Orchestrator-8B 곡선이 모든 비용 구간에서 다른 시스템보다 위에 있습니다. 같은 예산에서는 더 정확하고, 같은 정확도는 더 적은 비용으로 달성합니다.

셋째, 훈련 때 보지 못한 모델 구성(Claude Opus/Sonnet 4.1, o3-mini, GPT-4o, Codestral-22B, Gemma-3-27B 등)으로 바꿔도 Orchestrator-8B는 HLE 22.0, FRAMES 73.8, τ²-Bench 48.8을 기록했습니다. Claude Opus 4.1 오케스트레이터는 각각 17.8, 53.6, 43.4이고, 비용은 102.4 대 34.8입니다. 처음 보는 가격표(deepinfra)에서도 성능 우위가 유지되었습니다. 선호 지시를 붙인 평가에서는 선호 반영 보상이 HLE 46.7, FRAMES 68.4, τ²-Bench 79.5로, 차순위인 Claude Opus 4.1(40.2, 63.4, 73.5)보다 높았습니다.

## 의의와 한계

이 연구는 "더 큰 모델 하나"가 아니라 작은 조율 모델이 이질적인 모델과 도구를 엮는 복합 시스템으로도 프런티어 모델을 넘을 수 있음을 보여 줍니다. 비용과 선호를 보상에 직접 넣어 사용자가 정확도·비용·프라이버시 사이의 균형을 질의 단위로 조절할 수 있게 한 점도 실용적입니다. 모델, 데이터셋 ToolScale, 코드가 함께 공개되었습니다.

한계도 분명합니다. 먼저 비교 대상이 공정하게 맞춰져 있지 않습니다. Orchestrator는 RL로 훈련되었지만 비교 대상 오케스트레이터는 모두 프롬프트만 받았으므로, 같은 방식으로 훈련한 더 큰 모델이나 다른 라우팅 기법과는 비교하지 않았습니다. 보상 성분별 기여를 분리한 절제 실험도 본문에 없습니다. 측정 조건도 들쭉날쭉합니다. HLE는 텍스트 전용 부분집합이라 공개된 GPT-5 성적(전체 집합 35.2)과 직접 비교할 수 없고, τ²-Bench에서는 저자들이 GPT-5의 보고치 84.2를 재현하지 못해 77.7을 기준으로 삼았습니다. 정답 판정과 합성 데이터 생성이 LLM, 특히 GPT-5 판정기에 의존하고, 비용은 특정 공급자의 가격표로 환산한 값입니다. 선호 평가 지표 역시 기본 Qwen3-8B 대비 상대값이라 해석에 주의해야 합니다. 마지막으로 성능의 상당 부분은 여전히 GPT-5 같은 프런티어 모델 호출에서 나오므로, 이 방법은 강한 모델을 대체한다기보다 강한 모델을 덜, 더 잘 쓰게 만드는 방법에 가깝습니다.

- - -

### References
* [ToolOrchestra: Elevating Intelligence via Efficient Model and Tool Orchestration](https://arxiv.org/pdf/2511.21689v1)
* [Code](https://github.com/NVlabs/ToolOrchestra/)
* [Model](https://huggingface.co/nvidia/Orchestrator-8B)
* [Data](https://huggingface.co/datasets/nvidia/ToolScale)
* [Webpage](https://research.nvidia.com/labs/lpr/ToolOrchestra)
