---
layout: post
title: "ToolOrchestra: Elevating Intelligence via Efficient Model and Tool Orchestration"
seo_title: "ToolOrchestra"
subtitle: "ToolOrchestra: 효율적인 모델·도구 오케스트레이션으로 지능을 끌어올리다"
date: 2025-11-26 18:59:46
paper_author: "NVIDIA"
paper_doi: "10.48550/arXiv.2511.21689"
description: "ToolOrchestra는 강화학습으로 작은 8B 오케스트레이터가 도구와 강한 LLM 호출을 조율하도록 학습합니다. Orchestrator-8B는 HLE에서 37.1%로 GPT-5(35.1%)를 앞서며 2.5배 더 효율적입니다."
categories: ["Paper Summaries", "Agentic-AI"]
tags: ["Tool-Orchestration", "Orchestrator-Model", "Multi-Objective-Reward", "Cost-Aware-Reinforcement-Learning", "Synthetic-Tool-Use-Data", "Agentic-AI", "Reasoning-Models"]
cover: /assets/images/agentic-ai.jpg
use_math: true
---
## TL;DR

ToolOrchestra는 80억 파라미터 규모의 작은 모델을 오케스트레이터로 두고 강화학습으로 끝까지 학습시킵니다. 이 오케스트레이터는 웹 검색이나 코드 인터프리터(code interpreter) 같은 도구와 더 강한 LLM을 언제, 어떻게 호출할지 정하고, 그 결과를 여러 턴에 걸쳐 조합합니다. 보상은 답이 맞았을 때에만 비용·지연·사용자 선호를 반영하도록 설계했습니다. 이렇게 학습한 Orchestrator-8B는 HLE에서 37.1%를 기록해 GPT-5(35.1%)를 앞서면서도 2.5배 더 효율적이었습니다. τ²-Bench와 FRAMES에서도 GPT-5보다 크게 높은 점수를 냈고, 비용은 GPT-5의 약 30%만 썼습니다.

## 연구 배경

LLM에 도구를 붙이면 정확도가 오르는 경우가 있지만, 비용과 지연(latency)이 함께 늘어날 수 있습니다. 여기에 더해 어떤 도구를 언제 부를지도 제대로 고르지 못합니다. Claude Opus 4.1은 도구를 쓰면 고난도 추론 벤치마크 Humanity's Last Exam(HLE) 점수가 11.7에서 19.8로 오르지만, 대가로 비용은 2.8배, 지연은 4배가 됩니다.

검색 같은 기본 도구 외에 성능이 제각각인 LLM까지 도구로 쥐여주고, 오프더셸프(off-the-shelf) 모델에 프롬프트만으로 오케스트레이터(orchestrator) 역할을 맡기면 체계적인 편향이 나타납니다. HLE, FRAMES, $$\tau^{2}$$-Bench에서 인스턴스당 평균 호출 수를 세어 보면, GPT-5는 GPT-5-mini를 5.6회, GPT-5를 2.7회 호출해 자기 계열 모델에 일을 몰아줍니다. 저자들은 이 현상을 자기 강화 편향(self-enhancement bias)이라고 부릅니다.

Qwen3-8B는 비용이나 실제 효용과 상관없이 가장 강한 GPT-5를 인스턴스당 6.0회 불러 거기에 의존합니다. 저자들은 오케스트레이터가 자신보다 약한 모델과 강한 모델을 모두 부를 수 있는 이 상황에는 프롬프트만으로는 부족하고 별도의 학습 방법이 필요하다고 결론짓습니다.

## 제안 방법

ToolOrchestra의 보상은 답이 맞았을 때에만 효율과 사용자 선호를 따집니다. 오케스트레이터는 더 강한 LLM과 웹 검색·코드 인터프리터(code interpreter) 같은 도구를 언제, 어떻게 부를지 정하고, 그 결과를 여러 턴에 걸쳐 조합합니다. 프롬프트만 준 오프더셸프 오케스트레이터에서 앞서 본 호출 편향을, 프롬프트가 아니라 보상으로 직접 교정하려는 접근입니다.

이 구조에서 눈여겨볼 부분은 도구 집합의 구성입니다. 웹 검색이나 `get_flight_status` 같은 기본 도구 옆에 수학·코딩 전문 LLM과 앞서 소개한 GPT-5 같은 범용 LLM이 같은 층위의 도구로 나란히 놓여 있습니다. 오케스트레이터는 매 턴 추론한 뒤 이들 중 하나를 호출하고, 응답을 받아 다음 턴으로 넘어갑니다. 예를 들어 가상의 경로를 하나 들면, Humanity's Last Exam(HLE)이라는 고난도 추론 벤치마크의 문제에서 웹 검색으로 자료를 찾고 코드 인터프리터로 계산을 확인한 다음 꼭 필요한 단계에서만 GPT-5를 부르는 식입니다(논문에 실린 궤적이 아니라 설명을 위한 예시입니다).

학습 신호는 세 가지입니다. 결과 보상 $$r_{\text{outcome}}(\tau)\in\{0,1\}$$은 궤적 $$\tau$$가 문제를 풀었는지를 나타내며, 정답 비교는 GPT-5가 판정자로 맡습니다. 효율 보상은 $$r_{\text{compute}}(\tau)=-\$(\tau)$$와 $$r_{\text{latency}}(\tau)=-\mathit{Clock}(\tau)$$입니다. 앞의 것은 입력·출력 토큰을 외부 API 가격으로 환산한 금전 비용이고, 뒤의 것은 실제로 걸린 시간입니다. 여기에 도구 $$t_1,\dots,t_n$$별 호출 횟수 $$m^{\tau}_{t_i}$$를 더해 벡터 $$M^{\tau}=[m^{\tau}_{t_1},\dots,m^{\tau}_{t_n},r_{\text{outcome}},r_{\text{compute}},r_{\text{latency}}]$$를 만듭니다. 학습 중에는 각 원소를 롤아웃 배치 안의 최솟값과 최댓값으로 min-max 정규화하고, 배치 안에서 값이 모두 같은 원소는 0으로 둡니다. 최종 보상은 다음과 같습니다.

$$R(\tau)=\begin{cases}M_{\text{normalized}}^{\tau}\cdot P&\text{if }r_{\text{outcome}}(\tau)\\
0&\text{otherwise}.\end{cases}$$

$$M_{\text{normalized}}^{\tau}$$는 정규화된 벡터이고, $$P=[p_{t_1},\dots,p_{t_n},p_{\text{outcome}},p_{\text{compute}},p_{\text{latency}}]$$는 각 항목을 얼마나 최적화하고 싶은지를 $$0\le p\le 1$$로 적은 사용자 선호 벡터(preference vector)입니다. 예컨대 $$p_{\text{outcome}}=1$$, $$p_{\text{compute}}=0$$이면 비용은 무시하고 정확도만 따지겠다는 뜻이고, $$p_{t_1}=1$$이면 도구 $$t_1$$을 강하게 선호한다는 뜻입니다. 틀린 궤적은 아무리 싸고 빨라도 보상이 0입니다.

예를 들어 가상의 배치에서 정답 궤적 둘 가운데 GPT-5를 한 번 부른 궤적의 비용이 1달러, 세 번 부른 궤적의 비용이 3달러이고 이 값이 배치의 최소·최대라면, $$-\$$$로 바꾼 뒤 정규화한 비용 항목은 싼 궤적이 1, 비싼 궤적이 0입니다. 이때 $$p_{\text{compute}}$$가 클수록 GPT-5를 한 번만 부르고 같은 답에 도달한 싼 궤적이 더 큰 보상을 받습니다.

## 구현 방식

평가에서는 시작 체크포인트인 Qwen3-8B를 같은 예시에 먼저 돌려, 도구별 호출 수·정답 여부·비용·지연으로 이루어진 기준 벡터를 얻습니다. 도구별 호출 수와 정답 여부 성분은 기준값으로 나누고, 비용과 지연은 기준값을 새 값으로 나눠 기준보다 싸고 빠를수록 커집니다. 정답을 맞힌 예시만 선호 벡터 $$P$$와 내적한 보상을 받고, 벤치마크 점수는 예시별 보상의 합입니다.

HLE, FRAMES, τ²-Bench 인스턴스당 평균으로 Orchestrator-8B는 GPT-5를 1.6회, GPT-5-mini를 1.7회 호출했습니다. 앞서 본 두 오케스트레이터의 쏠림과 달리 Orchestrator-8B는 어느 한쪽에 쏠리지 않습니다. τ²-Bench에서는 단계의 약 40%에서만 GPT-5를 부르고 나머지는 더 싼 모델과 도구로 처리하고도, 매 단계 GPT-5를 쓰는 에이전트보다 성능이 높았습니다.

## 실험 결과

Orchestrator-8B는 세 벤치마크 모두에서 GPT-5보다 높은 점수를 내면서 비용은 더 적게 씁니다. HLE에서는 37.1%로 GPT-5의 35.1%를 2.0%p 앞서고, 효율은 2.5배 높습니다. τ²-Bench와 FRAMES에서는 GPT-5를 큰 차이로 앞서는데, 쓰는 비용은 약 30%에 그칩니다.

τ²-Bench에서 비교 대상별 점수와 비용, 지연은 다음과 같습니다.

| 모델 | 도구 구성 | τ²-Bench 점수 | 비용 | 지연 |
| --- | --- | --- | --- | --- |
| GPT-5 | 기본 도구 | 77.7 | 31.3 | 20.2 |
| Claude Opus 4.1 | 기본 도구·전문 LLM·범용 LLM | 76.8 | 52.8 | 24.1 |
| Orchestrator-8B | 기본 도구·전문 LLM·범용 LLM | 80.2 | 10.3 | 8.6 |

같은 도구를 쓰는 Claude Opus 4.1은 점수가 비슷한데도 비용이 다섯 배를 넘습니다.

학습 때 본 적 없는 조건에서도 우위가 이어집니다. 저자들은 먼저 학습에 없던 모델로 도구 구성을 바꿔 시험했습니다. 다음으로 질문마다 사용자 선호 지시를 붙인 테스트 세트에서, 앞서 정의한 선호 반영 보상으로 점수를 매겼습니다.

| 조건 | 모델 | HLE | FRAMES | τ²-Bench |
| --- | --- | --- | --- | --- |
| 미학습 도구 구성 | Orchestrator-8B | 22.0 | 73.8 | 48.8 |
| 미학습 도구 구성 | GPT-5 | 16.4 | 54.8 | 44.8 |
| 선호 반영 보상 | Orchestrator-8B | 46.7 | 68.4 | 79.5 |
| 선호 반영 보상 | GPT-5 | 34.6 | 62.3 | 70.3 |

미학습 도구 구성에서 Orchestrator-8B의 비용은 34.8로 가장 낮아, GPT-5의 81.3보다 훨씬 적었습니다. 선호 반영 평가에서는 GPT-5도 사용자 선호를 충실히 따르지 못했습니다.

![HLE 정확도와 비용의 관계를 모델별 곡선으로 그린 그래프](/assets/images/toolorchestra-elevating-intelligence-via-efficient-model-and-tool-orchestration-summary-1.png)

예산을 늘려도 우위는 유지됩니다. 저자들은 과제당 허용 최대 턴 수를 10, 20, 50, 100으로 바꿔 가며 HLE 비용-성능 곡선을 그렸습니다. 같은 예산에서 Orchestrator-8B는 GPT-5, Claude Opus 4.1, Qwen3-235B-A22B보다 일관되게 높은 성능을 냈고, 비슷한 성능은 훨씬 낮은 비용으로 달성했습니다.

## 의의와 한계

ToolOrchestra는 큰 모델 하나를 키우지 않아도, 작은 오케스트레이터가 도구와 모델을 골라 쓰는 방식으로 더 높은 정확도와 더 낮은 비용을 함께 얻을 수 있음을 보여 줍니다.

저자들이 향후 과제로 제시하는 방향은 더 정교한 재귀적 오케스트레이터 시스템입니다. 이를 통해 지능의 상한을 끌어올리는 동시에, 점점 복잡해지는 에이전트 작업을 더 효율적으로 풀려는 것입니다.

이 글의 판단으로는, 지금의 성과는 상한을 넘어섰다기보다 강한 모델을 적게 부르는 방식에 가깝습니다. 앞서 본 GPT-5 호출 감소가 이를 뒷받침합니다. 보고된 결과는 주로 GPT-5 같은 강한 모델을 덜 부르면서 같은 수준 이상의 정확도를 얻는 효과를 보여 줍니다. 실무에서는 여러 등급의 모델을 API로 쓸 수 있고 호출 비용이 큰 환경이 이 접근에 잘 맞습니다. 다만 도구 풀이나 가격 체계가 크게 바뀌면 재학습이 얼마나 필요한지 따로 점검하는 편이 안전합니다.

- - -

### References
* [ToolOrchestra: Elevating Intelligence via Efficient Model and Tool Orchestration](https://arxiv.org/pdf/2511.21689v1)
