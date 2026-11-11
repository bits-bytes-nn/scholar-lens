---
layout: post
title: "From RLHF to DPO and GRPO: Preference Optimization for LLM Alignment"
date: 2026-10-14 01:47:37
description: "사람의 비교 판단으로 언어 모델을 정렬하는 선호 최적화 방법을 다룹니다. PPO 기반 RLHF, AI 피드백, DPO와 후속 손실, 참조 모델 없는 손실, 선호 게임과 온라인 샘플링, GRPO까지 여섯 계보를 과최적화와 분포 이동 문제를 중심으로 정리합니다."
categories: ["Paper Surveys", "Finetuning"]
tags: ["Reward-Overoptimization", "Constitutional-AI", "KL-Regularization", "Nash-Learning-from-Human-Feedback", "Self-Play-Preference-Optimization", "Alignment"]
cover: /assets/images/finetuning.jpg
use_math: true
---
## 가능도 대신 사람의 비교 판단을 최적화하는 정렬 방법들

Stiennon 등이 지적했듯, 사람이 쓴 텍스트의 가능도를 최대화하는 훈련 목표는 사람이 실제로 좋다고 판단하는 고품질 출력을 만드는 목표와 어긋납니다 ([Stiennon et al., 2020](https://arxiv.org/abs/2009.01325)). 그래서 사람의 비교 판단을 직접 학습 신호로 쓰는 선호 최적화(preference optimization)가 사후 훈련(post training)의 중심이 되었습니다. Stiennon 등은 요약 과제에서 두 목표가 어긋나는 문제를 짚었습니다. 이들은 사람의 선호 비교로 보상 모델(reward model)을 학습하고, 그 보상을 강화학습으로 최적화하는 방식으로 이 간극을 메웠습니다 ([Stiennon et al., 2020](https://arxiv.org/abs/2009.01325)). 이 방식을 인간 피드백 기반 강화학습(reinforcement learning from human feedback, RLHF)이라 부릅니다. 이렇게 정렬한다고 일반 능력을 잃는 것은 아닙니다. Bai 등(Anthropic)은 RLHF로 유용하고 무해한 어시스턴트를 훈련했고, 그 결과 거의 모든 NLP 평가에서 성능이 올랐습니다 ([Bai et al., 2022](https://arxiv.org/abs/2204.05862)). 파이썬 코딩이나 요약 같은 전문 기술 학습과 함께 진행해도 문제가 없었습니다 ([Bai et al., 2022](https://arxiv.org/abs/2204.05862)).

이 글 전체의 공통 예는 Reddit TL;DR 요약 과제입니다. 한 게시글에 대한 요약 두 개를 보여 주고 사람이 더 나은 쪽을 고르게 해서 비교 데이터를 만듭니다. 이 비교 데이터로 정책을 정렬합니다.

글 끝의 연표는 2017년부터 2025년까지의 방법을 여섯 갈래로 나눠 정리합니다. 각 행에는 연도, 논문, 계보, 핵심 아이디어가 들어 있습니다. 계보 열을 보면 흐름이 드러납니다. 2017~2022년에는 RLHF 기초가 대부분을 차지합니다. 2023년부터는 AI 피드백, 직접 선호 최적화, 그리고 참조 모델(KL 제약의 기준이 되는 고정된 SFT 정책) 없는 손실이 갈라져 나옵니다. 2024년 이후에는 두 갈래가 늘어납니다. 하나는 현재 정책에서 새로 뽑은 샘플로 학습하거나 정렬을 정책 간 게임으로 보는 온라인·게임 이론 방법이고, 다른 하나는 PPO의 가치 함수(크리틱) 없이 그룹 내 상대 보상 등으로 이점을 추정하는 크리틱 없는 RL입니다.

설계도 같은 방향으로 바뀌었습니다. 초기 방법들은 보상 모델과 PPO(Proximal Policy Optimization, RLHF에서 보상을 최대화하는 데 쓰는 정책 경사 알고리즘)를 따로 두었습니다. 이후 방법들은 이 둘을 하나의 손실로 합치거나 참조 모델을 덜어 냈습니다. 최근 방법들은 다시 RL로 돌아가되, 크리틱 없이 현재 정책의 샘플로 학습하는 방식을 씁니다.

| 연도 | 논문 | 계보 | 핵심 아이디어 |
| --- | --- | --- | --- |
| 2017 | Deep RL from Preferences ([Christiano et al., 2017](https://arxiv.org/abs/1706.03741)) | RLHF 기초 | 궤적 구간 비교로 학습한 보상 예측기와 비동기 RL |
| 2019 | LM Fine-Tuning from Preferences ([Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)) | RLHF 기초 | KL 페널티를 붙인 PPO로 사전학습 언어모델을 미세조정 |
| 2020 | Summarize from HF ([Stiennon et al., 2020](https://arxiv.org/abs/2009.01325)) | RLHF 기초 | 사람 비교로 학습한 보상 모델을 이용한 요약 RLHF |
| 2022 | InstructGPT ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)) | RLHF 기초 | SFT, 보상 모델, PPO-ptx로 이어진 3단계 지시 따르기 정렬 |
| 2022 | Anthropic HH-RLHF ([Bai et al., 2022](https://arxiv.org/abs/2204.05862)) | RLHF 기초 | 반복 온라인 RLHF로 학습한 유용하고 무해한 어시스턴트 |
| 2022 | RM Overoptimization ([Gao et al., 2022](https://arxiv.org/abs/2210.10760)) | RLHF 기초 | 합성 gold 보상 모델로 측정한 보상 과최적화의 스케일링 법칙 |
| 2022 | Constitutional AI ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)) | AI 피드백 | 원칙 목록에 기반한 자기 비판과 AI 선호 라벨 |
| 2023 | RRHF ([Yuan et al., 2023](https://arxiv.org/abs/2304.05302)) | 참조 모델 없는 손실 | 길이 정규화 로그확률에 대한 랭킹 손실 |
| 2023 | DPO ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)) | 직접 선호 최적화 | 보상을 정책 로그비로 재매개화한 이진 교차엔트로피 손실 |
| 2023 | Llama 2 ([Touvron et al., 2023](https://arxiv.org/abs/2307.09288)) | RLHF 기초 | 거부 샘플링과 안전·유용성 두 보상 모델을 결합한 반복 RLHF |
| 2023 | RLAIF ([Lee et al., 2023](https://arxiv.org/abs/2309.00267)) | AI 피드백 | LLM 라벨러를 이용한 RLAIF와 직접 점수 보상 |
| 2023 | IPO ([Azar et al., 2023](https://arxiv.org/abs/2310.12036)) | 직접 선호 최적화 | 선호 확률의 Ψ 변환으로 일반화한 ΨPO와 IPO 제곱 손실 |
| 2023 | NLHF ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)) | 온라인·게임 이론 | 쌍대 선호 모델의 정규화된 내시 균형을 찾는 Nash-MD |
| 2024 | KTO ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)) | 직접 선호 최적화 | 프로스펙트 이론 효용에 기반한 이진 피드백 정렬 손실 |
| 2024 | GRPO ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)) | 크리틱 없는 RL | 그룹 평균 보상을 기준선으로 삼는 크리틱 없는 GRPO |
| 2024 | ORPO ([Hong et al., 2024](https://arxiv.org/abs/2403.07691)) | 참조 모델 없는 손실 | SFT 손실에 오즈비 페널티를 더한 단일 단계 정렬 |
| 2024 | Online IPO ([Calandriello et al., 2024](https://arxiv.org/abs/2403.08635)) | 온라인·게임 이론 | 현재 정책 샘플로 학습하는 Online IPO와 IPO-MD |
| 2024 | SPPO ([Wu et al., 2024](https://arxiv.org/abs/2405.00675)) | 온라인·게임 이론 | 승률 기반 곱셈 가중 갱신을 근사하는 셀프 플레이 최적화 |
| 2024 | RLHF Workflow ([Dong et al., 2024](https://arxiv.org/abs/2405.07863)) | 온라인·게임 이론 | 프록시 선호 모델에 기반한 재현 가능한 반복 DPO 레시피 |
| 2024 | SimPO ([Meng et al., 2024](https://arxiv.org/abs/2405.14734)) | 참조 모델 없는 손실 | 목표 마진을 둔 길이 정규화 평균 로그우도 보상 |
| 2024 | DAA Overoptimization ([Rafailov et al., 2024](https://arxiv.org/abs/2406.02900)) | 직접 선호 최적화 | 직접 정렬 알고리즘의 보상 과최적화 스케일링 분석 |
| 2025 | GRPO Dynamics ([Mroueh, 2025](https://arxiv.org/abs/2503.06639)) | 크리틱 없는 RL | 검증 가능한 이진 보상에서 GRPO를 가중 대조 손실로 해석 |

## 비교에서 보상 모델로, 보상 모델에서 KL 페널티가 붙은 PPO로 이어진 파이프라인

RLHF는 2017년 Atari와 MuJoCo에서 궤적 구간을 비교하던 보상 학습에서 시작해 사전학습 언어 모델로 넘어왔고, InstructGPT에 이르러 SFT→보상 모델→PPO라는 3단계 파이프라인으로 자리 잡았습니다. 같은 시기에 Stiennon 등은 보상 모델을 지나치게 최적화하면 실제 목표가 오히려 나빠지는 현상을 관찰했고, 이후 그 함수 형태까지 정량화되었습니다.

### 보상 함수 대신 궤적 구간 비교로 학습한 보상 추정기

Christiano 등의 출발점은 보상 함수를 직접 쓰기 어려운 강화학습 과제입니다. 이들의 시스템은 정책과 보상 추정기 $$\hat{r}$$를 함께 유지하고, 세 프로세스를 비동기로 돌립니다 ([Christiano et al., 2017](https://arxiv.org/abs/1706.03741)).

1. 정책은 예측 보상 $$\hat{r}$$를 최대화하도록 일반 RL로 갱신됩니다.
2. 정책이 만든 궤적에서 구간 쌍을 골라 사람에게 어느 쪽이 나은지 묻습니다.
3. 쌓인 비교 데이터에 지도학습으로 $$\hat{r}$$를 적합합니다.

궤적은 (1)에서 (2)로, 사람의 비교는 (2)에서 (3)으로, 갱신된 $$\hat{r}$$의 파라미터는 다시 (1)로 흘러갑니다 ([Christiano et al., 2017](https://arxiv.org/abs/1706.03741)).

![궤적 구간 비교로 보상 예측기를 학습하고 에이전트가 예측 보상을 최대화하는 구조도](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-1.png)

*출처: [Christiano et al., 2017](https://arxiv.org/abs/1706.03741)*

구조도에서 두 학습은 따로 돌면서 서로를 먹입니다. 보상 예측기는 궤적 구간 비교로 비동기 학습되고, 에이전트는 그 예측 보상만 봅니다 ([Christiano et al., 2017](https://arxiv.org/abs/1706.03741)).

절대 점수를 회귀하지 않고 비교를 택한 이유는 실험에 근거합니다. 사람은 절대 점수보다 비교를 훨씬 일관되게 매겼고, 연속 제어 과제에서는 비교를 예측하는 쪽이 점수를 예측하는 쪽보다 훨씬 잘 작동했습니다 ([Christiano et al., 2017](https://arxiv.org/abs/1706.03741)). 같은 실험에서 긴 클립이 단일 프레임보다 클립당 훨씬 많은 정보를 주었습니다 ([Christiano et al., 2017](https://arxiv.org/abs/1706.03741)). 비교를 보상으로 바꾸는 장치는 브래들리-테리 모델(Bradley-Terry model)입니다. 구간 $$\sigma^{1}$$이 $$\sigma^{2}$$보다 선호될 확률을 구간 안 추정 보상 합의 지수에 비례하도록 놓습니다 ([Christiano et al., 2017](https://arxiv.org/abs/1706.03741)).

$$
\hat{P}\left[\sigma^{1}\succ\sigma^{2}\right]=\frac{\exp{\sum\hat{r}\left(o^{1}_{t},a^{1}_{t}\right)}}{\exp{\sum\hat{r}\left(o^{1}_{t},a^{1}_{t}\right)}+\exp{\sum\hat{r}\left(o^{2}_{t},a^{2}_{t}\right)}}
$$

$$o^{i}_{t}$$와 $$a^{i}_{t}$$는 구간 $$\sigma^{i}$$의 시점 $$t$$에서 관측과 행동이고, 합은 구간 안의 모든 시점에 걸칩니다. 보상 추정기는 이 예측 확률과 사람 라벨 사이의 교차 엔트로피를 최소화합니다 ([Christiano et al., 2017](https://arxiv.org/abs/1706.03741)).

$$
\mathrm{loss}\left(\hat{r}\right)=-\sum_{\left(\sigma^{1},\sigma^{2},\mu\right)\in\mathcal{D}}\mu\left(1\right)\log\hat{P}\left[\sigma^{1}\succ\sigma^{2}\right]+\mu\left(2\right)\log\hat{P}\left[\sigma^{2}\succ\sigma^{1}\right]
$$

$$\mathcal{D}$$는 비교 데이터셋이고, $$\mu$$는 사람 라벨을 두 구간 위의 분포로 쓴 것입니다. 한쪽을 고르면 그쪽에 질량 1이 실립니다. 가상의 예로, 첫 구간의 추정 보상 합이 2.0이고 둘째 구간이 1.0이면 $$\hat{P}[\sigma^{1}\succ\sigma^{2}]=e^{2}/(e^{2}+e^{1})\approx 0.73$$입니다. 사람이 첫 구간을 골랐다면 손실은 $$-\log 0.73\approx 0.31$$이고, 둘째 구간을 골랐다면 $$-\log 0.27\approx 1.31$$입니다. 보상의 절대값은 정해지지 않고 차이만 학습되는데, 이 성질은 뒤의 모든 방법에 그대로 이어집니다.

어떤 쌍을 사람에게 물을지도 설계해야 했습니다. 저자들은 보상 추정기 앙상블의 예측 분산이 가장 큰 구간 쌍을 골라 불확실성을 근사했습니다 ([Christiano et al., 2017](https://arxiv.org/abs/1706.03741)). 저자들 스스로 이 방식을 조잡한 근사라고 불렀고, 절제 실험에서는 일부 과제의 성능을 오히려 떨어뜨렸습니다 ([Christiano et al., 2017](https://arxiv.org/abs/1706.03741)).

![MuJoCo 과제에서 구성 요소를 하나씩 제거한 절제 실험 결과](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-2.png)

*출처: [Christiano et al., 2017](https://arxiv.org/abs/1706.03741)*

절제 실험은 MuJoCo 과제에서 앙상블, 온라인 질의, 정규화, 구간 길이 같은 구성 요소를 하나씩 뺀 성능을 비교하며, 모든 곡선은 합성 라벨로 학습한 여러 실행의 평균입니다 ([Christiano et al., 2017](https://arxiv.org/abs/1706.03741)).

결과를 보면 보상 함수에 접근하지 않고도 Atari 게임과 시뮬레이션 로봇 보행 같은 복잡한 과제를 풀었고, 피드백은 에이전트 상호작용의 1% 미만에만 주었습니다 ([Christiano et al., 2017](https://arxiv.org/abs/1706.03741)). Atari 7개 게임에서는 사람 질의 5,500개로 학습했습니다 ([Christiano et al., 2017](https://arxiv.org/abs/1706.03741)). 진짜 보상으로 학습한 RL을 따라잡기는 MuJoCo보다 어려웠지만 대부분의 게임에서 상당히 학습했고, Enduro처럼 RL을 넘어선 게임도 있었습니다 ([Christiano et al., 2017](https://arxiv.org/abs/1706.03741)).

![Atari 게임에서 실제 사람 피드백, 합성 피드백, 진짜 보상 RL의 성능 비교](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-3.png)

*출처: [Christiano et al., 2017](https://arxiv.org/abs/1706.03741)*

Atari 결과 역시 사람 피드백과 합성 피드백으로 학습한 정책, 진짜 보상으로 학습한 RL을 각 게임의 진짜 보상으로 잽니다 ([Christiano et al., 2017](https://arxiv.org/abs/1706.03741)).

![MuJoCo 로봇 과제에서 실제 사람 피드백, 합성 오라클 피드백, 진짜 보상 RL의 성능 비교](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-4.png)

*출처: [Christiano et al., 2017](https://arxiv.org/abs/1706.03741)*

MuJoCo 그래프는 실제 사람 피드백, 오라클이 준 합성 피드백, 진짜 보상 함수를 쓴 RL을 모두 진짜 보상으로 비교합니다 ([Christiano et al., 2017](https://arxiv.org/abs/1706.03741)). 사람 라벨의 오류와 불일치 때문에 실제 사람 피드백은 합성 피드백과 대체로 비슷하거나 약간 낮았습니다 ([Christiano et al., 2017](https://arxiv.org/abs/1706.03741)). Qbert에서는 실제 사람 피드백으로 첫 레벨도 깨지 못했는데, 저자들은 짧은 클립이 헷갈리고 평가하기 어려웠기 때문으로 봅니다 ([Christiano et al., 2017](https://arxiv.org/abs/1706.03741)).

### 사전학습 언어 모델 위에 얹은 보상 모델과 KL 페널티

Ziegler 등은 같은 비교 기반 보상 학습을 시뮬레이션 환경 대신 사전학습 언어 모델에 적용했습니다 ([Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)). 이들이 다룬 과제는 사람의 판단으로만 정의되기 때문에, 보상을 알려면 사람에게 물을 수밖에 없습니다. 그래서 먼저 사람 라벨로 보상 모델을 학습하고, 그 보상 모델을 최적화합니다 ([Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)). 라벨러는 한 입력 $$x$$에 대한 여러 후보 $$y_{i}$$ 가운데 가장 좋은 응답을 고릅니다 ([Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)). 그래서 보상 모델 손실은 쌍이 아니라 네 후보에 대한 소프트맥스 로그우도입니다 ([Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)).

$$
\textrm{loss}(r)=\mathbb{E}_{\left(x,\left\{y_{i}\right\}_{i},b\right)\sim S}\left[\log{\frac{e^{r(x,y_{b})}}{\sum_{i}e^{r(x,y_{i})}}}\right]
$$

$$S$$는 라벨 데이터, $$b$$는 사람이 고른 후보의 인덱스, $$r(x,y)$$는 보상 모델의 스칼라 출력입니다. 이 식은 로그우도이므로 보상 모델은 이를 최대화하도록 학습합니다. 후보가 둘이면 앞서 본 브래들리-테리 형태로 돌아갑니다.

Atari에서 넘어오면서 가장 크게 달라진 부분은 정책이 사전학습 모델에서 출발한다는 사실입니다. 저자들은 PPO가 최적화할 보상을 보상 모델 점수 그대로 쓰지 않고, 초기 모델 $$\rho$$에서 멀어질수록 깎는 형태로 바꿨습니다 ([Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)).

$$
R(x,y)=r(x,y)-\beta\log\frac{\pi(y\mid x)}{\rho(y|x)}.
$$

$$\pi$$는 학습 중인 정책, $$\rho$$는 사전학습된 초기 언어 모델, $$\beta$$는 페널티 강도입니다. 표본마다 로그 비율을 빼므로, 정책 분포에서 기댓값을 취하면 $$\pi$$와 $$\rho$$ 사이의 KL 발산에 비례하는 KL 페널티(KL penalty)가 됩니다. 저자들은 이 항이 세 가지 일을 한다고 설명합니다. 엔트로피 보너스처럼 작동하고, 정책이 보상 모델이 유효한 범위 밖으로 벗어나지 않게 막으며, 스타일 이어 쓰기 과제에서는 일관성과 주제성을 지키는 과제 정의의 일부가 됩니다 ([Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)). 두 번째 역할은 보상 모델이 학습 분포 바깥에서 믿을 수 없다는 사실을 처음부터 전제한 설계입니다.

![KL로 잰 초기 정책과의 거리가 커질수록 보상은 오르지만 샘플이 부자연스러워지는 관계(모의 감성 과제)](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-5.png)

*출처: [Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)*

모의 감성 과제에서 이 맞교환이 바로 드러납니다. 정책이 $$\operatorname{KL}(\pi,\rho)$$ 기준으로 초기 정책에서 더 멀리 가도록 허용할수록 보상은 높아지지만, 그 대가로 샘플이 덜 자연스러워집니다 ([Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)).

전체 절차는 정책 샘플을 생성해 사람이 최선을 고르게 하고, $$\rho$$로 초기화한 보상 모델을 학습한 뒤, 앞의 수정 보상 $$R$$로 PPO를 돌리는 순서입니다. 온라인 설정에서는 정책 샘플로 비교를 다시 모아 보상 모델을 재학습합니다 ([Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)).

![보상 모델 학습과 정책 학습 과정을 나타낸 개요도](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-6.png)

*출처: [Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)*

개요도는 두 학습 과정을 나란히 그리며, 온라인에서는 두 과정이 교차하며 진행됩니다 ([Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)). Christiano 등의 비동기 루프가 언어 모델 버전으로 다시 나타난 모습입니다.

사람 데이터의 양이 결과를 얼마나 바꾸는지는 오프라인 설정에서 따로 쟀습니다. 제로샷 모델과, 사람 데이터 양을 달리해 오프라인으로 미세조정한 모델들을 사람 평가로 비교하고, 라벨러 3명 중 다수가 미세조정 모델을 선호한 비율을 보고합니다 ([Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)).

![사람 데이터 양을 달리한 오프라인 미세조정 모델과 제로샷 모델의 사람 평가 비교](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-7.png)

*출처: [Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)*

구현에서 얻은 교훈도 있습니다. 보상 모델과 정책이 파라미터를 공유하게 하자 과적합으로 실패했습니다. 보상 모델 데이터는 최대 6만 개인데 정책 에피소드는 200만 개라 데이터 불균형이 컸기 때문이고, 결국 두 네트워크를 따로 학습했습니다 ([Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)). 요약 과제에서는 성능의 의미가 모호했습니다. ROUGE 기준으로는 TL;DR과 CNN/DM 모두에서 지도학습+RL 미세조정이 가장 좋았지만, 순수 RL 미세조정은 모든 경우 지도학습 기준선보다 낮았습니다 ([Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)). 사람 라벨러 기준으로 매우 좋았던 요약 모델은 거의 전부 문장을 그대로 복사하는 추출식이었고, 저자들은 라벨러가 단순한 휴리스틱에 기대는 습성을 모델이 악용했을 수 있다고 인정했습니다 ([Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)). 실행마다 편차가 크고 사람 라벨로 여러 시드를 돌리기는 비싸서, 온라인과 오프라인 사이의 차이도 상당 부분 노이즈일 수 있습니다 ([Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)).

### TL;DR 요약에서 굳어진 비교 수집→보상 모델→PPO의 순환

Stiennon 등은 앞의 공통 예인 TL;DR 요약에서 이 구도를 정리했습니다. 절차는 세 단계이고 반복할 수 있습니다. 사람 비교를 수집하고, 게시글과 후보 요약이 주어졌을 때 그 요약이 더 나을 로그 오즈를 예측하는 보상 모델을 학습한 뒤, 이 보상 모델에 맞춰 PPO로 정책을 최적화합니다 ([Stiennon et al., 2020](https://arxiv.org/abs/2009.01325)).

![사람 피드백 수집, 보상 모델 학습, 정책 학습 절차를 보여 주는 도식](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-8.png)

*출처: [Stiennon et al., 2020](https://arxiv.org/abs/2009.01325)*

도식에 그려진 사람 피드백 수집, 보상 모델 학습, 정책 학습의 세 상자가 이후 RLHF 논문들이 공유하는 뼈대가 됩니다 ([Stiennon et al., 2020](https://arxiv.org/abs/2009.01325)). 비교는 두 요약 사이에서 이루어지므로 보상 모델 손실도 쌍대 로지스틱 형태로 단순해집니다 ([Stiennon et al., 2020](https://arxiv.org/abs/2009.01325)).

$$
\text{loss}(r_{\theta})=-E_{(x,y_{0},y_{1},i)\sim D}[\log(\sigma(r_{\theta}(x,y_{i})-r_{\theta}(x,y_{1-i})))]
$$

$$x$$는 Reddit 게시글, $$y_{0}$$와 $$y_{1}$$은 후보 요약, $$i$$는 사람이 고른 요약의 인덱스, $$\sigma$$는 시그모이드입니다. 두 보상의 차이에 시그모이드를 씌우면 앞서 본 브래들리-테리 확률과 같아집니다. PPO 보상은 앞서 정의한 KL 수정 보상과 같은 형태이되, 페널티의 기준이 사전학습 모델이 아니라 SFT 정책 $$\pi^{\text{SFT}}$$로 바뀌었습니다 ([Stiennon et al., 2020](https://arxiv.org/abs/2009.01325)). 이후 참조 정책(reference policy)이라 불리는 자리가 여기서 SFT 모델로 정해집니다. 정책은 SFT 모델로 초기화하고, PPO의 가치 함수는 정책과 파라미터를 전혀 공유하지 않는 별도 Transformer로 두며 보상 모델 가중치로 초기화합니다 ([Stiennon et al., 2020](https://arxiv.org/abs/2009.01325)). 파라미터를 분리하면 학습 초기에 가치 함수 갱신이 사전학습된 정책을 망가뜨리는 일을 막을 수 있습니다 ([Stiennon et al., 2020](https://arxiv.org/abs/2009.01325)).

![TL;DR 데이터셋에서 모델별 요약이 사람이 쓴 참조 요약보다 선호되는 비율](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-9.png)

*출처: [Stiennon et al., 2020](https://arxiv.org/abs/2009.01325)*

주요 결과 그래프의 세로축은 사람이 모델 요약을 사람이 쓴 참조 요약보다 선호한 비율입니다 ([Stiennon et al., 2020](https://arxiv.org/abs/2009.01325)). 같은 데이터로 ROUGE와 사람 판단이 얼마나 일치하는지도 쟀습니다. 지도학습 기준선의 샘플을 비교할 때 ROUGE와 라벨러의 일치율은 약 57%였지만, 인간 피드백 모델의 샘플에서는 약 50%로 떨어졌습니다 ([Stiennon et al., 2020](https://arxiv.org/abs/2009.01325)). 반면 보상 모델의 일치율은 여전히 ROUGE보다 높았습니다 ([Stiennon et al., 2020](https://arxiv.org/abs/2009.01325)). 모델이 좋아질수록 고정된 자동 지표는 사람 선호를 놓치게 되고, 학습된 보상 모델이 그 자리를 대신합니다.

그러나 학습된 보상 모델에도 한계가 있었습니다. 가볍게 최적화하면 라벨러 기준으로 요약이 좋아지지만, 더 밀어붙이면 실제 선호가 보상 모델의 예측보다 처지고, 끝내 보상 모델이 사람 선호와 반상관이 됩니다 ([Stiennon et al., 2020](https://arxiv.org/abs/2009.01325)).

![보상 모델 최적화 강도에 따른 선호 점수 변화](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-10.png)

*출처: [Stiennon et al., 2020](https://arxiv.org/abs/2009.01325)*

가로축은 보상 모델 최적화의 정도이고, 세로축의 선호 점수는 처음에 오르다가 과적합이 시작되면서 떨어져 더 나쁜 요약을 만듭니다 ([Stiennon et al., 2020](https://arxiv.org/abs/2009.01325)). 한편 TL;DR에서 학습한 6.7B 인간 피드백 모델은 뉴스 기사 요약을 학습한 적이 없는데도, 비슷한 요약 길이에서 CNN/DM의 T5와 거의 같은 수준에 도달했습니다 ([Stiennon et al., 2020](https://arxiv.org/abs/2009.01325)).

![CNN/DM 전이 결과: 모델 크기별 전체 요약 품질과 길이별 점수](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-11.png)

*출처: [Stiennon et al., 2020](https://arxiv.org/abs/2009.01325)*

전이 결과의 (a)는 CNN/DM에서 모델 크기에 따른 전체 요약 품질이고, (b)는 6.7B TL;DR 모델 등의 요약 길이별 전체 점수입니다 ([Stiennon et al., 2020](https://arxiv.org/abs/2009.01325)). 비용은 컸습니다. 6.7B 모델의 RL 미세조정에만 약 320 GPU-일이 들었고, 데이터 수집에도 수천 시간의 라벨러 작업이 필요했습니다 ([Stiennon et al., 2020](https://arxiv.org/abs/2009.01325)).

### 지시 따르기 전반으로 넓어진 SFT→보상 모델→PPO 3단계

InstructGPT는 요약에서 다듬어진 절차를 범용 지시 따르기로 넓혔습니다. 사람 시연으로 SFT를 하고, 라벨러가 어떤 모델 출력을 선호할지 예측하는 보상 모델을 비교 데이터로 학습한 뒤, 이 보상 모델을 보상 함수로 PPO 최적화합니다 ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)).

![지도 미세조정, 보상 모델 학습, PPO 강화학습의 3단계를 나타낸 다이어그램](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-12.png)

*출처: [Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)*

다이어그램의 둘째 단계에서 라벨러는 한 쌍이 아니라 A부터 D까지 여러 샘플에 순위를 매깁니다 ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)). 순위 하나에서 나오는 비교를 어떻게 쓰는지가 보상 모델 손실에 반영됩니다 ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)).

$$
\operatorname{loss}\left(\theta\right)=-\frac{1}{% raw %}{{K\choose 2}}{% endraw %}E_{\left(x,y_{w},y_{l}\right)\sim D}\left[\log\left(\sigma\left(r_{\theta}\left(x,y_{w}\right)-r_{\theta}\left(x,y_{l}\right)\right)\right)\right]
$$

$$y_{w}$$는 선호된 응답, $$y_{l}$$은 덜 선호된 응답, $$K$$는 프롬프트당 순위가 매겨진 응답 수입니다. 한 프롬프트에서 나온 $${K\choose 2}$$개 비교를 모두 하나의 배치 원소로 묶어 학습하며, 저자들은 이렇게 묶으면 과적합을 막을 수 있다고 보고했습니다 ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)). 가상의 예로 $$K=4$$이면 순위 하나에서 비교 6개가 나오고, 손실은 이 6개의 평균입니다. 175B 보상 모델은 학습이 불안정해 RL의 가치 함수로 쓰기 부적합했고, 그래서 작은 보상 모델만 썼습니다 ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)).

RL 단계는 밴딧 환경으로 정의됩니다. 무작위 고객 프롬프트에 응답 하나를 내면 보상 모델이 점수를 매기고 에피소드가 끝납니다 ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)). 토큰마다 SFT 모델에 대한 KL 페널티를 걸어 보상 모델의 과최적화를 완화하고, 가치 함수는 보상 모델로 초기화합니다 ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)). 여기에 한 항이 더 붙습니다 ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)).

$$
\begin{aligned}
\operatorname{objective}\left(\phi\right)=&E_{\left(x,y\right)\sim D_{\pi_{\phi}^{\mathrm{RL}}}}\left[r_{\theta}(x,y)-\beta\log\left(\pi_{\phi}^{\mathrm{RL}}(y\mid x)/\pi^{\mathrm{SFT}}(y\mid x)\right)\right]+\\
&\gamma E_{x\sim D_{\textrm{pretrain}}}\left[\log(\pi_{\phi}^{\mathrm{RL}}(x))\right]
\end{aligned}
$$

$$\pi_{\phi}^{\mathrm{RL}}$$은 학습 중인 정책, $$D_{\pi_{\phi}^{\mathrm{RL}}}$$은 그 정책이 만든 프롬프트-응답 분포, $$D_{\textrm{pretrain}}$$은 사전학습 분포, $$\gamma$$는 사전학습 로그우도 항의 계수입니다. $$\gamma>0$$인 변형을 PPO-ptx라 부릅니다. 이 항은 RLHF 뒤에 공개 NLP 데이터셋 성능이 떨어지는 현상을 되돌리기 위한 장치입니다.

![사전학습 손실 계수에 따른 공개 NLP 데이터셋 평가](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-13.png)

*출처: [Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)*

계수를 바꿔 가며 공개 NLP 데이터셋을 평가해 보면, DROP과 SQuAD를 크게 개선하면서 다른 지표는 별로 해치지 않는 계수가 존재합니다 ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)).

사람 평가 결과도 뚜렷했습니다.

![API 프롬프트 분포에서 175B SFT 모델 대비 각 모델 출력이 선호된 비율](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-14.png)

*출처: [Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)*

이 그래프는 API 프롬프트 분포에서 각 모델의 출력이 175B SFT 모델보다 선호된 비율을 보여 줍니다 ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)). PPO와 PPO-ptx는 GPT-3 기준선을 크게 앞서고, 파라미터가 100분의 1인 1.3B PPO-ptx의 출력이 175B GPT-3보다 선호됩니다 ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)). 정렬에 드는 연산은 사전학습보다 훨씬 작았습니다. 175B SFT는 4.9 petaflops/s-days, 175B PPO-ptx는 60 petaflops/s-days였고, GPT-3 사전학습은 3,640 petaflops/s-days였습니다 ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)). 데이터 쪽에는 약점이 남았습니다. 선호 데이터가 영어 중심의 소규모 계약 라벨러 집단에서 나와 대표성이 없고, 비용 때문에 대부분의 비교를 라벨러 한 명만 매겨서 라벨러 간 불일치를 파악하기 어렵습니다 ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)).

### 반복 온라인 수집과 보정 문제를 드러낸 HH-RLHF와 Llama 2

Anthropic의 Bai 등은 같은 파이프라인을 유용하고 무해한 어시스턴트에 적용하면서 학습 동역학을 들여다봤습니다. RLHF 학습 중 선호 모델 점수는 초기 정책과의 KL 발산의 제곱근에 대략 선형으로 비례했습니다 ([Bai et al., 2022](https://arxiv.org/abs/2204.05862)). 보상이 $$\sqrt{D_{\text{KL}}}$$에 비례하면 보상을 두 배로 올리는 데 KL은 네 배가 들므로, 보상 향상의 대부분은 정책이 초기 정책에서 조금만 움직인 구간에서 얻어집니다 ([Bai et al., 2022](https://arxiv.org/abs/2204.05862)). 이들이 부딪힌 더 큰 문제는 보정이었습니다. 선호 모델 점수가 높은 구간으로 갈수록 보정과 견고성이 떨어졌고, 크라우드워커 Elo가 선호 모델의 예측과 어긋났습니다 ([Bai et al., 2022](https://arxiv.org/abs/2204.05862)). 온라인 선호 모델의 테스트 정확도는 기본 분포 74%, 거부 샘플링 분포 70%, 온라인 분포 67%로, 데이터가 높은 품질 쪽으로 갈수록 낮아졌습니다 ([Bai et al., 2022](https://arxiv.org/abs/2204.05862)).

해법으로 택한 방식은 반복 온라인 RLHF입니다. 그때까지 가장 좋은 RLHF 정책으로 크라우드워커 비교 데이터를 모으고, 기존 데이터와 섞어 새 선호 모델과 정책을 다시 학습하는 과정을 주 단위로 되풀이했습니다 ([Bai et al., 2022](https://arxiv.org/abs/2204.05862)). 높은 점수 구간의 데이터가 부족해 보정이 나빠지는 문제를 겨냥한 방법이며, 52B 모델과 유용성(helpfulness) 데이터에서 실행했습니다 ([Bai et al., 2022](https://arxiv.org/abs/2204.05862)). 여기서 '온라인'은 같은 모델을 이어서 학습하지 않고 반복마다 새 모델을 재학습한다는 뜻입니다 ([Bai et al., 2022](https://arxiv.org/abs/2204.05862)).

![데이터 수집과 모델 학습 전체 워크플로를 요약한 도식](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-15.png)

*출처: [Bai et al., 2022](https://arxiv.org/abs/2204.05862)*

선호 모델 학습, RLHF, 온라인 반복으로 이어지는 데이터 수집과 모델 학습의 전체 워크플로가 이 도식 하나에 정리되어 있습니다 ([Bai et al., 2022](https://arxiv.org/abs/2204.05862)).

![컨텍스트 증류 모델, 정적 데이터 RLHF 모델, 반복 온라인 RLHF 모델에 대한 크라우드워커 선호 Elo 요약](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-16.png)

*출처: [Bai et al., 2022](https://arxiv.org/abs/2204.05862)*

크라우드워커 선호를 Elo로 모아 보면, 반복 온라인 방식으로 학습한 RLHF 모델이 컨텍스트 증류 모델과 '정적' 데이터셋으로 학습한 RLHF 모델보다 더 선호되었습니다 ([Bai et al., 2022](https://arxiv.org/abs/2204.05862)). 정책이 만든 고품질 응답 근처에서 비교를 새로 모아야 보상 모델이 그 영역에서 쓸 만해진다는 결론이 Christiano 등의 비동기 루프와 이어집니다.

Llama 2는 이 흐름을 대형 오픈 모델에서 반복했습니다. 보상 모델은 선택 응답 $$y_{c}$$와 거부 응답 $$y_{r}$$ 쌍에 대한 이진 랭킹 손실로 학습하고, 유용성 보상 모델에는 선호 등급에 따른 마진 $$m(r)$$을 더했습니다 ([Touvron et al., 2023](https://arxiv.org/abs/2307.09288)).

$$
\mathcal{L}_{\text{ranking}}=-\text{log}(\sigma(r_{\theta}(x,y_{c})-r_{\theta}(x,y_{r})-m(r)))
$$

$$r$$는 라벨러가 매긴 선호 등급이고, 마진 $$m(r)$$는 등급이 뚜렷할수록 두 보상의 차이를 더 크게 벌리도록 요구합니다. PPO 보상은 안전과 유용성 두 보상 모델을 조건부로 결합합니다. 결합한 점수에 시그모이드의 역함수인 로짓을 씌워 0과 1 사이로 눌린 값을 실수 전체로 다시 펼치고, 평균 0·분산 1로 정규화하는 화이트닝을 거친 다음, 기준 정책에 대한 KL 페널티를 뺍니다 ([Touvron et al., 2023](https://arxiv.org/abs/2307.09288)). 저자들은 이 KL 제약이 학습 안정성에 유용하고, 보상 모델 점수는 높은데 사람 평가는 낮은 보상 해킹을 줄인다고 보고했습니다 ([Touvron et al., 2023](https://arxiv.org/abs/2307.09288)). KL 계수 $$\beta$$는 7B와 13B에서 0.01, 34B와 70B에서 0.005였습니다 ([Touvron et al., 2023](https://arxiv.org/abs/2307.09288)).

PPO와 함께 쓴 거부 샘플링(rejection sampling)은 70B 모델에서 답을 $$K$$개 뽑아 보상이 가장 높은 응답으로 미세조정하고, 더 작은 모델은 이 데이터로 학습해 큰 모델의 능력을 증류받는 방식입니다 ([Touvron et al., 2023](https://arxiv.org/abs/2307.09288)). V3까지 직전 반복의 샘플만 쓰자 시 짓기 같은 능력이 퇴보했고, 이후로는 이전 모든 반복의 상위 샘플을 함께 넣었습니다 ([Touvron et al., 2023](https://arxiv.org/abs/2307.09288)). RLHF는 샘플링 설정도 바꿉니다. Llama 2-Chat-RLHF에서 10~100개를 샘플링할 때 최적 온도는 1.2~1.3이었고, 그래서 반복마다 온도를 점진적으로 다시 맞춰야 합니다 ([Touvron et al., 2023](https://arxiv.org/abs/2307.09288)).

![반복 미세조정을 거치며 변하는 Llama 2-Chat의 ChatGPT 대비 승률](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-17.png)

*출처: [Touvron et al., 2023](https://arxiv.org/abs/2307.09288)*

반복 미세조정 버전이 올라갈수록 Llama 2-Chat의 ChatGPT 대비 승률이 높아졌고, 판정자로는 자체 보상 모델과 GPT-4를 함께 썼습니다 ([Touvron et al., 2023](https://arxiv.org/abs/2307.09288)). 다만 사람 평가 프롬프트에는 코딩이나 추론 프롬프트가 없고, 평가 자체도 주관적이며 잡음이 있습니다 ([Touvron et al., 2023](https://arxiv.org/abs/2307.09288)).

### 합성 gold 보상으로 측정한 과최적화의 함수 형태

Stiennon 등이 관찰한 과최적화를 Gao, Schulman, Hilton은 정량적인 법칙으로 만들었습니다. 학습된 프록시 보상 모델을 지나치게 최적화하면 진짜 목표가 결국 나빠지고, 저자들은 이를 과최적화라 불렀습니다 ([Gao et al., 2022](https://arxiv.org/abs/2210.10760)). 저자들은 InstructGPT 환경에 '진짜 보상' 역할을 하는 gold 보상 모델을 두는 합성 설정을 썼습니다 ([Gao et al., 2022](https://arxiv.org/abs/2210.10760)). 두 궤적 가운데 gold 보상 모델이 더 높은 점수를 준 쪽을 항상 선호로 표기해 비교를 결정론적으로 만들고, 이 데이터로 프록시 보상 모델을 학습합니다 ([Gao et al., 2022](https://arxiv.org/abs/2210.10760)). 그런 다음 최고 n개 고르기(BoN)와 RL로 프록시를 최적화하면서 gold 점수를 추적합니다 ([Gao et al., 2022](https://arxiv.org/abs/2210.10760)).

![정책 크기를 고정하고 보상 모델 크기를 바꾼 BoN 실험에서 gold 보상이 오르다 떨어지는 곡선](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-18.png)

*출처: [Gao et al., 2022](https://arxiv.org/abs/2210.10760)*

이 실험은 정책을 1.2B로 고정하고 보상 모델 크기만 바꿉니다 ([Gao et al., 2022](https://arxiv.org/abs/2210.10760)). 가로축은 제곱근 눈금입니다. 프록시를 최적화할수록 gold 보상은 처음에 오르다가 나중에 떨어지고, 저자들이 제안한 함수 형태가 이 곡선에 잘 맞습니다 ([Gao et al., 2022](https://arxiv.org/abs/2210.10760)). 최적화량은 $$d=\sqrt{\text{KL}}$$, 곧 초기 정책과의 KL의 제곱근으로 잽니다. BoN의 gold 보상은 $$d$$의 이차 함수를 따릅니다 ([Gao et al., 2022](https://arxiv.org/abs/2210.10760)).

$$
R_{\text{bo}n}\left(d\right)=d\left(\alpha_{\text{bo}n}-\beta_{\text{bo$n$}}d\right)
$$

RL의 gold 보상은 이차항 대신 로그항이 들어간 형태를 따릅니다 ([Gao et al., 2022](https://arxiv.org/abs/2210.10760)).

$$
R_{\text{RL}}\left(d\right)={d}\left(\alpha_{\text{RL}}-\beta_{\text{RL}}\log{d}\right)
$$

두 식에서 $$\alpha_{\text{bon}}$$와 $$\alpha_{\text{RL}}$$은 초기 기울기, 곧 프록시를 따라갈 때 gold 보상이 처음 얼마나 빨리 오르는지를 정하고, $$\beta_{\text{bon}}$$와 $$\beta_{\text{RL}}$$은 과최적화로 꺾이는 정도를 정하는 곡선 적합 계수입니다. 이 $$\beta$$들은 앞서 본 KL 페널티 계수 $$\beta$$와 다른 양이며, 아래에서 아래첨자 없이 쓴 $$\alpha$$, $$\beta$$도 이 적합 계수를 뜻합니다. 두 식 모두 최고점이 하나 있습니다. 미분해서 0으로 놓으면 BoN의 최고점은 $$d^{*}=\alpha/(2\beta)$$, RL의 최고점은 $$d^{*}=\exp(\alpha/\beta-1)$$입니다. 가상의 계수로 BoN에서 $$\alpha=1$$, $$\beta=0.25$$라면 $$d^{*}=2$$, 곧 KL 4에서 gold 보상이 최대 1에 이르고, 그 뒤로는 더 최적화할수록 손해입니다. 정책 1.2B와 보상 모델 데이터 90,000개를 고정했을 때, BoN의 $$\alpha$$와 $$\beta$$는 보상 모델 크기에 따라 매끄럽게 변했습니다 ([Gao et al., 2022](https://arxiv.org/abs/2210.10760)). RL은 $$\alpha$$를 고정해도 $$\beta$$가 깔끔하게 스케일링되어 최고 gold 점수를 예측할 수 있었습니다 ([Gao et al., 2022](https://arxiv.org/abs/2210.10760)).

KL을 최적화량의 척도로 쓰는 데에는 함정이 있습니다. RL은 같은 gold 보상에 이르기까지 BoN보다 훨씬 많은 KL을 '소모'합니다 ([Gao et al., 2022](https://arxiv.org/abs/2210.10760)). BoN의 KL은 $$\log n$$에 비례하지만 RL의 KL은 학습 스텝에 대해 대략 이차로 늘어나므로, 저자들은 KL이 최적화량의 척도로 부적절하다고 봅니다 ([Gao et al., 2022](https://arxiv.org/abs/2210.10760)). 정책 크기의 영향도 쟀습니다. 보상 모델을 12M으로 두고 정책을 1.2B와 6B로 비교하면, 큰 정책은 최적화로 얻는 이득이 작지만 과최적화가 더 심해지지는 않았습니다 ([Gao et al., 2022](https://arxiv.org/abs/2210.10760)). 두 정책의 gold 점수는 거의 같은 KL에서 최고점에 이르고, 프록시와 gold의 격차도 거의 같았습니다 ([Gao et al., 2022](https://arxiv.org/abs/2210.10760)).

KL 페널티에 대한 결과는 앞선 논문들의 기대와 결이 다릅니다.

![KL 페널티 강도를 바꾼 RL 실험에서 프록시 보상과 gold 보상의 변화](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-19.png)

*출처: [Gao et al., 2022](https://arxiv.org/abs/2210.10760)*

정책과 보상 모델 크기를 고정하고 KL 페널티만 바꾼 실험에서, 점선은 프록시 보상이고 실선은 gold 보상입니다 ([Gao et al., 2022](https://arxiv.org/abs/2210.10760)). 페널티가 gold 점수에 주는 효과는 조기 종료와 같았습니다 ([Gao et al., 2022](https://arxiv.org/abs/2210.10760)). 페널티는 gold 점수를 더 일찍 수렴시킬 뿐 KL 대 gold 보상의 프런티어는 바꾸지 못했습니다 ([Gao et al., 2022](https://arxiv.org/abs/2210.10760)). 저자들은 이 결론이 하이퍼파라미터에 민감할 수 있다고 덧붙였습니다 ([Gao et al., 2022](https://arxiv.org/abs/2210.10760)). Ziegler 등과 Llama 2가 KL 항을 보상 모델의 유효 범위를 지키고 보상 해킹을 줄이는 장치로 설명한 것과 견주면, 적어도 이 합성 설정에서 KL 페널티는 새로운 해를 주지 않고 최적화를 어디서 멈출지만 정합니다.

이 정량화에도 빈칸이 있습니다. InstructGPT 환경의 합성 gold 보상 모델만 썼으므로, 실제 사람의 의도와 라벨 사이의 불일치에서 오는 과최적화는 다루지 않았고, 합성 설정의 결과가 현실로 그대로 옮겨진다는 보장도 없습니다 ([Gao et al., 2022](https://arxiv.org/abs/2210.10760)). 정책 크기 스케일링도 제한적으로만 탐색해서 다른 추세를 놓쳤을 수 있습니다 ([Gao et al., 2022](https://arxiv.org/abs/2210.10760)).

> **핵심:** 2017년부터 2023년까지 RLHF는 같은 두 장치를 계속 다듬었습니다. 하나는 비교를 브래들리-테리 형태의 보상 차이로 바꾸는 보상 모델이고, 다른 하나는 정책을 참조 정책 근처에 묶어 두는 KL 항입니다. 이 시기에 남은 숙제도 같은 곳을 가리킵니다. 보상 모델은 정책이 새로 도달한 분포에서 보정이 무너지고, 그 분포에서 비교를 다시 모으지 않으면 최적화는 일정 지점을 넘는 순간 진짜 목표를 해칩니다.

이 글의 판단으로는, 비싼 PPO 루프와 별도의 보상 모델이 이후 방법들이 덜어 내려 한 대상이었습니다. 반면 반복 온라인 수집과 과최적화 곡선이 보여 준 '현재 정책 근처의 데이터' 문제는 덜어 낼 수 없는 제약으로 남았습니다.

## 원칙 목록과 AI 라벨로 대체된 사람의 무해성 판단

Constitutional AI는 RLHF 파이프라인을 그대로 둔 채 무해성 비교 라벨을 붙이는 주체만 사람에서 원칙에 따라 판단하는 피드백 모델로 바꿨고, RLAIF는 AI 라벨만으로도 SFT 기준선을 넘을 수 있으며 라벨 비용은 사람 주석보다 10배 이상 싸다고 추정했습니다.

### 헌법이라는 원칙 목록과 두 단계 학습

Constitutional AI에서 사람이 개입하는 곳은 규칙과 원칙을 적은 목록 하나뿐이며, 저자들은 이 목록을 '헌법'이라고 부릅니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)). 이 목록만을 감독 신호로 삼아, 무해하면서도 회피하지 않는 어시스턴트를 두 단계로 학습합니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)).

![지도학습 단계와 강화학습 단계로 이루어진 Constitutional AI 전체 과정](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-20.png)

*출처: [Bai et al., 2022](https://arxiv.org/abs/2212.08073)*

이 개요도에서는 위쪽 SL 단계(비판→수정→미세조정)와 아래쪽 RL 단계(AI 비교 평가→선호 모델→RL)가 어떻게 이어지는지 보면 됩니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)). SL-CAI는 유해한 답을 유도하는 프롬프트에 모델이 낸 응답을 모델 스스로 비판하고 고쳐 쓰게 합니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)). 수정 단계마다 원칙을 무작위로 하나씩 뽑고, 모든 단계의 수정본으로 사전학습 모델을 미세조정합니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)). 유용성(helpfulness)을 지키려고, 유용성 RLHF 모델이 크라우드워커 프롬프트에 낸 응답도 미세조정 데이터에 섞습니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)).

수정을 거듭할수록 무해성 선호 모델(PM) 점수는 단조롭게 올랐지만, 순수 유용성 점수는 떨어졌습니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)). PM 점수가 높은 구간에서는 보정도 나빠졌습니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)). 비판 단계는 작은 모델에서만 무해성 점수를 올렸고 큰 모델에서는 눈에 띄는 차이가 없었습니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)). 비판 내용도 부정확하거나 과장될 때가 많았는데, 저자들은 투명성을 이유로 이 단계를 남겼습니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)).

### 객관식 확률을 그대로 쓰는 소프트 라벨

RL-CAI의 핵심은 비교 라벨을 만드는 방식입니다. 피드백 모델은 원칙 하나를 받아 두 응답 중 더 무해한 쪽을 객관식으로 고릅니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)). 이때 (A)와 (B)의 로그확률을 정규화한 확률이 선호 모델 학습의 목표값, 곧 소프트 라벨이 됩니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)). 예를 들어 가상의 비교에서 정규화 결과가 (A) 0.7, (B) 0.3이라면, 이 쌍은 'A 승'이라는 하드 라벨 대신 목표 확률 0.7인 비교 예제가 됩니다. 저자들은 이 소프트 라벨이 비교적 잘 보정되어 있으리라 기대했습니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)). 모든 라벨에 같은 원칙을 쓰지 않고 비교마다 원칙을 무작위로 뽑아 앙상블하자, PM의 거동이 눈에 띄게 견고해졌습니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)). 라벨이 준비된 뒤의 선호 모델 학습과 RL은 RLHF와 똑같습니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)). 유용성 비교는 사람 라벨을, 무해성 비교는 AI 라벨을 섞어 씁니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)).

피드백 모델이 이런 판단을 해낼 수 있는지는 따로 확인했습니다.

![HHH 이진 비교 438개에서 사람 피드백으로 학습한 선호 모델과 사전학습 언어 모델의 정확도 비교](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-21.png)

*출처: [Bai et al., 2022](https://arxiv.org/abs/2212.08073)*

유용성·정직성·무해성을 묻는 이진 비교 438개에서, 체인 오브 소트(chain-of-thought, CoT)를 쓰면 사전학습 언어 모델의 정확도가 크게 올랐습니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)). 저자들은 52B보다 큰 모델이면 사람 피드백으로 학습한 PM과 대등해지리라 전망했습니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)). CoT 라벨은 확률을 일정 범위로 잘라 냈는데(클램핑), 범위를 좁힐수록 결과가 좋아져 논문의 주요 결과에는 40-60%를 택했습니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)). 예를 들어 가상의 CoT 라벨이 0.95라면 목표값은 0.60으로 줄어듭니다.

### 회피 없이 더 무해해진 정책과 굿하트의 흔적

![52B RL 실행들의 무해성 대 유용성 Elo 비교](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-22.png)

*출처: [Bai et al., 2022](https://arxiv.org/abs/2212.08073)*

이 그림은 크라우드워커의 모델 비교로 계산한 무해성과 유용성 Elo를 두 축으로 놓고, 52B RL 실행 전체를 한 평면에 배치합니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)). Elo는 값 자체보다 모델 사이의 차이만 의미가 있습니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)). RL-CAI는 RLHF와 SL-CAI보다 훨씬 무해했습니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)). CoT를 쓰면 유용성은 조금 낮아지고 무해성은 조금 높아졌습니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)). 질적으로도 차이가 있었습니다. HH RLHF는 회피적으로 답하는 경향이 있었지만, RL-CAI는 레드팀 프롬프트 대부분에 거의 회피하지 않고 미묘하면서도 무해한 답을 냈습니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)).

![RL 학습 시퀀스 수에 따른 유용성 Elo와 무해성 Elo 곡선](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-23.png)

*출처: [Bai et al., 2022](https://arxiv.org/abs/2212.08073)*

학습이 진행되는 동안의 궤적은 이 곡선에서 확인할 수 있습니다. 크라우드워커 비교로 잰 유용성과 무해성 Elo를 누적 RL 학습 시퀀스 수에 따라 그렸습니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)). 다만 RL-CAI도 과하게 학습하면 앞서 본 보상 과최적화와 같은 굿하트 현상이 나타났습니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)). 유해한 프롬프트에 지나치게 가혹하게 답하거나 상투적인 문구를 붙이는 식입니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)). 원칙의 개수를 늘려도 무해성 점수는 크게 달라지지 않았습니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)). 원칙이 많을수록 행동이 다양해지리라는 기대는 정량적으로 검증하지 않았습니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)). 저자들은 AI 피드백(AI feedback)으로 반복적 온라인 학습을 하는 방향을 다음 과제로 남겼습니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)).

### 사람 라벨과 맞먹는 AI 라벨, 열 배 이상 싼 비용

![위쪽 RLAIF와 아래쪽 RLHF 파이프라인을 나란히 놓은 도식](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-24.png)

*출처: [Lee et al., 2023](https://arxiv.org/abs/2309.00267)*

RLAIF 논문의 도식은 위쪽에 RLAIF, 아래쪽에 RLHF 파이프라인을 나란히 그려 둡니다 ([Lee et al., 2023](https://arxiv.org/abs/2309.00267)). Lee 등은 AI 라벨러가 정책과 같은 크기이거나 초기 정책과 똑같은 체크포인트여도 RLAIF가 SFT 기준선을 넘는다는 것을 보였고, 이를 자기 개선의 가능성으로 해석했습니다 ([Lee et al., 2023](https://arxiv.org/abs/2309.00267)). 변형인 d-RLAIF는 보상 모델 학습을 아예 건너뜁니다. LLM에게 생성 결과를 1~10점으로 평가하게 하고, 점수 토큰 확률의 가중합을 곧바로 보상으로 씁니다 ([Lee et al., 2023](https://arxiv.org/abs/2309.00267)). 정책이 바뀌어도 고정된 보상 모델이 낡아 가는 문제를 이렇게 피하며, 성능도 표준 RLAIF보다 나았습니다 ([Lee et al., 2023](https://arxiv.org/abs/2309.00267)). 저자들은 LLM 라벨링 비용이 사람 주석보다 10배 이상 싸다고 추정했습니다 ([Lee et al., 2023](https://arxiv.org/abs/2309.00267)).

이 글의 판단으로는, 두 연구가 덜어 낸 것은 라벨 비용이지 앞서 본 분포 이동 문제가 아닙니다. 피드백 모델도 결국 고정된 판정자이고, CAI의 굿하트 현상이 보여 주듯 그 판정자를 과하게 최적화하면 같은 방식으로 무너집니다. 그래서 d-RLAIF처럼 현재 정책의 출력을 그때그때 채점하는 설계가 이후 온라인 방법들과 맞닿아 있습니다.

## 보상 모델을 정책 안에 숨긴 닫힌 형태의 재매개화와 그 균열

DPO는 KL 정규화 보상 최대화 문제의 최적 정책을 닫힌 형태로 풀어 보상 모델 손실을 정책 손실로 바꿨고, 그 결과 RL 루프가 사라졌습니다. 그러나 IPO, KTO, 직접 정렬 알고리즘(direct alignment algorithm, DAA) 과최적화 연구가 이어지면서 이 단순화의 대가가 드러났습니다. 결정적 선호에서 생기는 과적합, 쌍 데이터에 대한 의존, 한 에폭이 끝나기도 전에 시작되는 성능 열화가 차례로 확인되었습니다.

### 최적 정책의 닫힌 형태에서 나온 정책 손실

출발점은 RLHF의 RL 단계 목적함수입니다. 학습된 보상을 최대화하되 참조 정책(reference policy)에서 멀어지는 만큼 KL 페널티(KL penalty)를 받습니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)).

$$\max_{\pi_{\theta}}\mathbb{E}_{x\sim\mathcal{D},y\sim\pi_{\theta}(y\mid x)}\bigl[r_{\phi}(x,y)\bigr]-\beta\mathbb{D}_{\textrm{KL}}\bigl[\pi_{\theta}(y\mid x)\mid\mid\pi_{\text{ref}}(y\mid x)\bigr]$$

$$x$$는 프롬프트, $$y$$는 응답, $$r_{\phi}$$는 학습된 보상 모델(reward model), $$\pi_{\text{ref}}$$는 보통 SFT 모델인 참조 정책입니다. $$\beta$$는 KL 페널티의 세기입니다. 보상 모델은 그보다 먼저 브래들리-테리 모델(Bradley-Terry model)의 음의 로그우도로 학습합니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)).

$$\mathcal{L}_{R}(r_{\phi},\mathcal{D})=-\mathbb{E}_{(x,y_{w},y_{l})\sim\mathcal{D}}\bigl[\log\sigma(r_{\phi}(x,y_{w})-r_{\phi}(x,y_{l}))\bigr]$$

$$y_w$$는 선호 응답, $$y_l$$은 비선호 응답, $$\sigma$$는 시그모이드입니다. RLHF는 이 두 문제를 차례로 풉니다. 먼저 보상 모델을 맞추고, 그 보상으로 PPO를 돌립니다.

Rafailov 등은 이 두 단계를 하나로 합쳤습니다. 보상 함수에서 최적 정책으로 가는 해석적 매핑을 이용하면, 보상에 대한 손실을 정책에 대한 손실로 변수 변환할 수 있습니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)). 위 KL 제약 문제의 최적 정책은 닫힌 형태로 주어집니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)).

$$\pi_{r}(y\mid x)=\frac{1}{Z(x)}\pi_{\text{ref}}(y\mid x)\exp\left(\frac{1}{\beta}r(x,y)\right)$$

$$Z(x)$$는 모든 응답에 걸친 합으로 확률을 정규화하는 분배함수입니다. 실제 언어 모델에서는 응답 공간이 너무 커서 이 값을 계산할 수 없습니다. 그래서 이 식을 그대로 쓰지 않고, 양변에 로그를 취해 보상에 대해 다시 씁니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)).

$$r(x,y)=\beta\log\frac{\pi_{r}(y\mid x)}{\pi_{\text{ref}}(y\mid x)}+\beta\log Z(x)$$

이 식을 브래들리-테리 손실에 넣으면 계산 불가능한 항이 사라집니다. 브래들리-테리 확률은 두 응답의 보상 차이에만 의존하고, 같은 프롬프트의 두 응답은 $$\beta\log Z(x)$$를 공유하므로 차이를 취할 때 상쇄됩니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)). 남는 손실은 다음과 같습니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)).

$$\mathcal{L}_{\text{DPO}}(\pi_{\theta};\pi_{\text{ref}})=-\mathbb{E}_{(x,y_{w},y_{l})\sim\mathcal{D}}\left[\log\sigma\left(\beta\log\frac{\pi_{\theta}(y_{w}\mid x)}{\pi_{\text{ref}}(y_{w}\mid x)}-\beta\log\frac{\pi_{\theta}(y_{l}\mid x)}{\pi_{\text{ref}}(y_{l}\mid x)}\right)\right]$$

$$\beta\log(\pi_\theta/\pi_{\text{ref}})$$는 정책 안에 숨어 있는 암묵적 보상입니다. 손실 전체는 그 차이에 대한 이진 교차엔트로피입니다. 이론적으로 이 재매개화는 표현력을 잃지 않습니다. 브래들리-테리(더 일반적으로는 Plackett-Luce)와 일치하는 모든 보상 클래스를 $$\beta\log(\pi/\pi_{\text{ref}})$$ 형태로 나타낼 수 있습니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)).

![보상 모델 학습 뒤 RL을 거치는 RLHF와 선호 데이터에 이진 교차엔트로피를 바로 적용하는 DPO 파이프라인 비교](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-25.png)

*출처: [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)*

두 파이프라인을 나란히 그린 도식에서 달라진 곳은 가운데 단계입니다. RLHF는 보상 모델을 학습한 뒤 RL로 정책을 최적화하고, DPO는 선호 데이터에서 곧바로 정책을 최적화합니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)). 정책 네트워크가 언어 모델이자 암묵적 보상 모델을 겸하므로 별도의 보상 모델도, 샘플링 루프도 필요 없습니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)). 데이터 준비도 단순합니다. 참조 정책에서 응답 쌍을 뽑아 선호를 라벨링한 오프라인 데이터셋을 만들고, 주어진 $$\pi_{\text{ref}}$$와 $$\beta$$로 손실을 최소화합니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)). 가능하면 $$\pi_{\text{ref}}$$를 SFT 모델로 두고, SFT 모델이 없으면 선호 응답의 우도를 최대화해 $$\pi_{\text{ref}}$$를 초기화합니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)).

그래디언트를 풀어 보면 손실이 무엇을 하는지 보입니다. 선호 응답의 우도는 높이고 비선호 응답의 우도는 낮추며, 암묵적 보상이 순서를 거꾸로 매긴 예제일수록 더 큰 가중치를 줍니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)). 이 가중치를 빼고 우도만 밀고 당기는 단순한 버전은 모델을 퇴화시킬 수 있습니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)). 예를 들어 가상의 쌍에서 $$\beta=0.1$$이고, 선호 응답의 로그비가 $$+2$$, 비선호 응답의 로그비가 $$-1$$이라고 해 봅니다. 그러면 암묵적 보상 차이는 $$0.1\times(2-(-1))=0.3$$이고, $$\sigma(0.3)\approx0.574$$이므로 이 쌍의 손실은 $$-\log 0.574\approx0.555$$입니다. 차이가 $$-0.3$$으로 뒤집혀 있었다면 손실은 약 $$0.855$$로 커지고, 이 쌍이 그래디언트에서 차지하는 몫도 커집니다.

PPO 기반 RLHF가 불안정했던 이유도 같은 틀에서 설명됩니다. 참조 정책의 소프트 가치에 해당하는 정규화 항이 없으면 정책 그래디언트의 분산이 커집니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)). 기존 방법은 학습된 가치함수나 사람 응답을 기준선으로 써서 이 분산을 줄였습니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)). DPO 재매개화로 얻은 보상은 기준선이 아예 필요 없습니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)).

### PPO를 앞선 보상-KL 프런티어와 요약·대화 승률

정답 보상을 아는 IMDb 감성 생성에서 저자들은 22회 실행 스윕으로 보상과 KL의 맞교환을 쟀습니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)). DPO의 프런티어는 PPO를 엄격히 지배했고, 정답 보상에 접근할 수 있는 PPO-GT보다도 앞섰습니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)).

![왼쪽은 참조 정책 대비 KL에 따른 기대 보상 프런티어, 오른쪽은 TL;DR 요약 승률과 샘플링 온도의 관계](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-26.png)

*출처: [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)*

왼쪽 프런티어에서 확인할 것은 모든 KL 값에서 DPO의 기대 보상이 가장 높다는 사실이고, 저자들은 이를 최적화 품질의 근거로 듭니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)). 오른쪽은 Reddit TL;DR 요약 과제에서 샘플링 온도에 따라 승률이 어떻게 변하는지 보여 줍니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)). TL;DR로 학습한 정책을 분포 밖의 CNN/DailyMail 기사에 적용해도 순위는 같았습니다. 정답 요약 대비 GPT-4 승률은 온도 0에서 DPO 0.36, PPO 0.26이었고, 더 높은 온도에서도 앞서 본 비교와 마찬가지로 DPO가 PPO를 앞섰습니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)).

![Anthropic-HH 단일 턴 대화의 GPT-4 승률과 학습 중 온도별 승률 변화](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-27.png)

*출처: [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)*

대화 과제에서는 기준이 더 까다롭습니다. Anthropic-HH 테스트셋에서 데이터셋의 선호 응답보다 나은 응답을 낸 방법은 DPO뿐이었고, 학습이 진행되는 동안 온도별 개선도 비교적 안정적이었습니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)). Pythia-2.8B 단일 턴 설정의 GPT-4 승률로 보면, DPO는 선호 응답을 넘어선 유일한 계산 효율적 방법이었고 계산량이 큰 Best-of-128과 비슷하거나 더 나았습니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)). 이 비교들은 대부분 GPT-4 판정에 기댑니다. 저자들은 GPT-4로 계산한 승률이 프롬프트에 따라 달라진다고 밝혔습니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)).

### 선호가 결정적일수록 사라지는 정규화

IPO 논문의 Azar 등은 RLHF와 DPO를 하나의 목적함수 아래에 놓고 비교했습니다. 선호 확률의 비선형 단조함수 $$\Psi$$를 최대화하면서 참조 정책과의 KL로 정규화하는 $$\Psi$$PO입니다 ([Azar et al., 2023](https://arxiv.org/abs/2310.12036)).

$$\max_{\pi}\mathop{\mathbb{E}}_{\begin{subarray}{c}x\sim\rho\\
y\sim\pi(.\mid x)\\
y^{\prime}\sim\mu(.|x)\end{subarray}}[\Psi(p^{*}(y\succ y^{\prime}\mid x))]-\tau D_{\text{KL}}(\pi\;||\;\pi_{\text{ref}})$$

$$\rho$$는 프롬프트 분포, $$\mu$$는 비교 대상 응답 $$y'$$을 내는 행동 정책, $$p^{*}(y\succ y'\mid x)$$는 참 선호 확률, $$\tau$$는 정규화 세기입니다. $$\Psi(q)=\log(q/(1-q))$$로 두면 브래들리-테리 가정 아래에서 이 목적함수가 RLHF와 DPO를 모두 포함합니다 ([Azar et al., 2023](https://arxiv.org/abs/2310.12036)).

문제는 이 로그오즈 변환에서 생깁니다. 예를 들어 가상의 쌍에서 선호 확률이 0.99라면 로그오즈는 $$\log 99\approx4.6$$이고, 확률이 1이 되면 로그오즈는 무한대로 발산합니다. 선호가 결정적일수록 브래들리-테리가 요구하는 보상 차이가 끝없이 커지고, 그만큼 KL 정규화가 상대적으로 약해져 DPO가 과적합합니다 ([Azar et al., 2023](https://arxiv.org/abs/2310.12036)). 데이터셋의 비교 대부분이 한 번씩만 라벨링되므로 경험적 선호는 사실상 0 아니면 1입니다. 그래서 이 상황은 극단적 예외가 아니라 평범한 경우에 가깝습니다. RLHF가 이 문제를 덜 겪는 이유도 같은 분석에서 나옵니다. RLHF에서는 보상 모델이 과소적합되면서 정책을 정규화하는 효과가 생기지만, DPO는 보상 학습을 건너뛰면서 이 효과를 잃습니다 ([Azar et al., 2023](https://arxiv.org/abs/2310.12036)).

IPO는 $$\Psi$$를 항등함수로 둡니다. 최적 정책의 로그비가 만족해야 할 방정식들을 하나의 제곱 회귀 문제로 바꾸고, 이를 오프라인으로 학습합니다 ([Azar et al., 2023](https://arxiv.org/abs/2310.12036)).

$$L(\pi)=\mathop{\mathbb{E}}_{y,y^{\prime}\sim\mu}\left[\left(h_{\pi}(y,y^{\prime})-\frac{p^{*}(y\succ\mu)-p^{*}(y^{\prime}\succ\mu)}{\tau}\right)^{2}\right]$$

$$h_{\pi}(y,y')$$는 두 응답 각각의 정책 대 참조 정책 로그비를 서로 뺀 값입니다. $$p^{*}(y\succ\mu)$$는 $$y$$가 $$\mu$$에서 뽑은 응답을 이길 확률입니다. 실제 데이터에는 참 선호 확률이 없으므로, 평균이 $$p^{*}(y\succ y')$$인 베르누이 지시자 $$I(y,y')$$로 바꾼 불편 추정 손실을 씁니다 ([Azar et al., 2023](https://arxiv.org/abs/2310.12036)).

$$\mathop{\mathbb{E}}_{y,y^{\prime}\sim\mu}\Big[\big(h_{\pi}(y,y^{\prime})-\tau^{-1}I(y,y^{\prime})\big)^{2}\Big]$$

DPO 손실과의 차이는 목표값에 있습니다. 시그모이드 안의 차이는 손실을 줄이려면 무한히 커져야 하지만, IPO의 로그비 차이는 유한한 목표 $$\tau^{-1}$$ 근처에 머물도록 회귀합니다.

![세 행동 총순서 데이터에서 정규화 세기별 IPO와 DPO의 행동 확률 학습 곡선](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-28.png)

*출처: [Azar et al., 2023](https://arxiv.org/abs/2310.12036)*

세 행동에 총순서가 있는 장난감 데이터에서 그린 학습 곡선이 이 차이를 보여 줍니다. DPO는 $$\tau$$를 어떻게 잡아도 결정적 정책으로 수렴했고, IPO는 정규화가 강할 때 탐욕 정책으로 무너지지 않았습니다 ([Azar et al., 2023](https://arxiv.org/abs/2310.12036)). 다만 실험은 이런 최소한의 장난감 예제가 전부입니다. 저자들도 실제 사람 선호 데이터로 언어 모델을 학습하는 실험을 향후 과제로 남겼습니다 ([Azar et al., 2023](https://arxiv.org/abs/2310.12036)).

### 쌍 없이 이진 신호만 쓰는 KTO

KTO는 다른 균열을 짚었습니다. DPO는 같은 프롬프트에 대한 선호·비선호 응답 쌍이 있어야 학습할 수 있습니다. KTO는 출력 하나가 바람직한지 아닌지만 알려 주는 이진 신호로 학습하고, 저자들은 이런 데이터가 실제로 더 풍부하고 싸고 빨리 모인다고 봅니다 ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)).

Ethayarajh 등은 먼저 기존 손실들을 프로스펙트 이론의 언어로 다시 분류했습니다. 인간 인식 손실(human-aware loss, HALO)은 암묵적 보상에서 기준점 보상의 기댓값을 뺀 값에 비감소 가치함수를 씌운 손실군입니다 ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)). 이 정의로 보면 DPO와 PPO-Clip은 HALO이고, SLiC와 CSFT는 HALO가 아닙니다 ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)).

![여러 HALO가 함축하는 결과 효용 곡선과 프로스펙트 이론 가치함수 비교](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-29.png)

*출처: [Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)*

효용 곡선을 겹쳐 보면, 각 HALO가 함축하는 가치함수에도 프로스펙트 이론의 인간 가치함수처럼 손실 회피 같은 성질이 있습니다 ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)). KTO는 여기서 두 가지를 바꿉니다. DPO가 비선호 응답 하나를 기준점으로 삼는 것과 달리, KTO는 KL을 기준점으로 두고 선호 로그우도가 아니라 효용을 직접 최대화합니다 ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)). 보상 동치류는 프롬프트마다 상수 $$h(x)$$를 더해도 브래들리-테리 선호 확률이 변하지 않는 보상들의 모임입니다. 저자들은 그 근거로, 이 동치류 안에서 보상을 $$r+h(x)$$처럼 입력별로 옮기면 선호 가능도는 그대로인데 가치 분포는 달라진다는 사실을 듭니다 ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)). 따라서 선호 가능도만으로는 프로스펙트 이론의 가치가 정해지지 않습니다 ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)).

$$L_{\text{KTO}}(\pi_{\theta},\pi_{\text{ref}})=\mathbb{E}_{x,y\sim D}[\lambda_{y}-v(x,y)]$$

가치함수 $$v$$는 바람직한 응답에서 $$\lambda_{D}\sigma(\beta(r_{\theta}(x,y)-z_{0}))$$, 바람직하지 않은 응답에서 $$\lambda_{U}\sigma(\beta(z_{0}-r_{\theta}(x,y)))$$입니다 ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)). $$r_\theta$$는 정책과 참조 정책으로 정의한 암묵적 보상이고, $$z_0$$는 $$\mathrm{KL}(\pi_\theta\lVert\pi_{\text{ref}})$$로 둔 기준점입니다 ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)). $$\lambda_y$$는 응답의 종류에 따라 $$\lambda_D$$ 또는 $$\lambda_U$$가 되는 가중치입니다 ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)). 원래 Kahneman-Tversky 가치함수의 지수 형태는 수치적으로 불안정해서 로지스틱 함수로 바꿨고, $$\beta$$로 위험 회피 정도를 조절합니다 ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)). $$z_0$$로는 역전파하지 않습니다. 이 항은 손실의 포화를 조절하는 용도로만 씁니다 ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)).

![모델 크기별로 HALO와 비HALO 방법의 GPT-4 판정 승률을 비교한 그래프](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-30.png)

*출처: [Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)*

이 분류는 실험에서도 의미가 있었습니다. Llama 계열에서 HALO인 DPO와 오프라인 PPO 변형이 비HALO인 SLiC와 CSFT보다, 이기기 어려운 기준선 출력과 맞붙인 GPT-4 판정 승률이 높았습니다 ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)). 그러나 차이가 $$p<0.05$$ 수준에서 유의한 구간은 13B 이상뿐이었고, 이 기준선 출력에 맞먹은 것도 HALO로 정렬한 Llama-13B와 30B뿐이었습니다 ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)).

![1B~30B 규모별 KTO와 DPO의 GPT-4 판정 승률 비교](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-31.png)

*출처: [Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)*

KTO와 DPO를 직접 맞붙인 비교에서는 1B~30B 전 규모에서 KTO가 DPO와 같거나 더 나았습니다 ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)). Llama에서는 SFT 없이 KTO만 쓴 모델이 SFT 뒤 DPO를 적용한 모델과 대등했습니다 ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)). Llama-13B와 30B에서는 SFT를 건너뛴 KTO 모델이 SFT+KTO 모델과도 대등했고, 시험한 방법 중 이런 성질을 보인 것은 KTO뿐이었습니다 ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)).

![SFT 없이 DPO 또는 KTO로 정렬한 모델의 응답 길이 비교](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-32.png)

*출처: [Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)*

SFT를 건너뛰면 DPO는 다르게 무너집니다. 응답 길이 비교에 따르면 SFT 없이 DPO로 정렬한 모델은 말을 늘어놓다가 대화 전체를 지어내는 환각을 보였고, KTO 모델에서는 이런 문제가 나타나지 않았습니다 ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)).

![바람직한 예를 줄인 불균형 데이터에서 Llama-7B의 KTO와 DPO 성능 비교](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-33.png)

*출처: [Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)*

데이터 형태에 대한 의존도도 실험했습니다. 같은 데이터셋을 더 작고 크게 불균형하게 만들어도, KTO로 정렬한 Llama-7B는 DPO 모델과 같거나 더 나았습니다 ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)). 바람직한 예 하나에 바람직하지 않은 예가 열 개인 비율에서도 마찬가지였습니다 ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)). Llama-7B에서는 바람직한 데이터를 최대 90%까지 버려도 DPO를 앞섰습니다 ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)).

저자들은 KTO가 늘 낫다고 주장하지 않습니다. 선호 데이터에 잡음과 비추이성이 충분히 적으면 DPO가 더 낫고, KTO는 과소적합할 위험이 있습니다 ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)). 잡음이 많을 때는 KTO의 최악 경우 보장이 유리합니다 ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)). 참조 모델을 없앤 KTO 변형은 메모리 효율이 좋고 ORPO보다 나았지만, 표준 KTO보다 성능이 낮았습니다 ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)). 이 참조 모델 없는 변형은 손실 회피 하이퍼파라미터에도 더 민감해서, $$\lambda_D$$를 1.5나 2.0으로 두면 GSM8K와 BBH 성능이 몇 포인트씩 떨어졌습니다 ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)).

### 보상 모델 없이도 한 에폭 안에서 시작되는 과최적화

DPO가 보상 모델을 없앴으니 앞서 본 보상 과최적화(reward overoptimization)도 함께 사라졌을까요. DAA Overoptimization 논문의 답은 그렇지 않다는 것입니다. DPO를 비롯한 직접 정렬 알고리즘은 별도 보상 모델이 없어도 KL 예산이 커지면 고전적 RLHF와 비슷한 열화 패턴을 보입니다 ([Rafailov et al., 2024](https://arxiv.org/abs/2406.02900)). 열화는 넓은 범위의 KL 예산에서 나타나고, 데이터셋 한 에폭을 다 돌기 전에 시작되는 경우도 많습니다 ([Rafailov et al., 2024](https://arxiv.org/abs/2406.02900)).

![Pythia 6.9B에서 DPO·SLiC·IPO 학습 중 에폭 진행률에 따른 승률과 KL 변화](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-34.png)

*출처: [Rafailov et al., 2024](https://arxiv.org/abs/2406.02900)*

에폭 안의 학습 동역학을 그린 그래프는 위쪽 행에 에폭 진행률별 승률을, 아래쪽 행에 그에 대응하는 KL을 놓습니다 ([Rafailov et al., 2024](https://arxiv.org/abs/2406.02900)). KL 제약이 낮은 설정에서도 대부분의 실험이 에폭의 첫 25% 안에 최고 성능에 도달했습니다 ([Rafailov et al., 2024](https://arxiv.org/abs/2406.02900)). 그 뒤로는 학습을 더 할수록 성능이 떨어지고, 모델은 참조 정책에서 점점 멀어졌습니다 ([Rafailov et al., 2024](https://arxiv.org/abs/2406.02900)). 앞서 장난감 예제로만 확인했던 IPO도 이 결과를 피하지 못했습니다 ([Rafailov et al., 2024](https://arxiv.org/abs/2406.02900)).

저자들은 원인을 암묵적 보상 목적이 과소제약되어 있다는 데서 찾습니다. 손실은 데이터에 들어 있는 쌍에서의 암묵적 보상 값만 보므로, 대략 전체 프롬프트-응답 공간 위의 값을 데이터 쌍이 건드리는 값으로 옮기는 행렬로 손실의 구조를 쓸 수 있습니다. 프롬프트-응답 공간의 크기 $$\lvert\mathcal{X}\times\mathcal{Y}\rvert$$가 선호 데이터셋의 크기 $$\lvert\mathcal{D}\rvert$$보다 압도적으로 크기 때문에, 이 행렬에는 자명하지 않은 영공간이 생길 가능성이 높고 문제가 엄격한 볼록이 아니게 됩니다 ([Rafailov et al., 2024](https://arxiv.org/abs/2406.02900)). 영공간 방향으로 정책을 움직여도 손실 값은 변하지 않으므로, 손실을 똑같이 최소화하는 해가 여럿 생깁니다. 그 결과 최적해가 여럿 존재하고, 그중 일부는 데이터에 없는 분포 밖 응답에 큰 확률을 줍니다 ([Rafailov et al., 2024](https://arxiv.org/abs/2406.02900)). 정량적으로도 RLHF와 닮았습니다. RLHF 연구에서 보상 과최적화를 기술하던 스케일링 법칙의 함수형을 DAA의 승률에 적용하자 잘 맞았고, KL과 승률 사이의 이차 적합보다 RMSE가 절반으로 줄었습니다 ([Gao et al., 2022](https://arxiv.org/abs/2210.10760))([Rafailov et al., 2024](https://arxiv.org/abs/2406.02900)).

> **핵심:** DPO는 보상 모델이라는 부품을 지웠지만, 고정된 오프라인 데이터로 KL에 묶인 정책을 밀어붙이는 구조는 그대로 남겼습니다. 그래서 과최적화도 보상 모델 대신 암묵적 보상 안에서 다시 나타납니다.

이 글의 판단으로는, 세 후속 연구가 짚은 균열은 하나의 원인으로 모입니다. IPO의 결정적 선호, KTO가 보여 준 SFT 없는 DPO의 환각, DAA의 분포 밖 응답은 모두 데이터가 덮지 못하는 영역에서 암묵적 보상이 제약 없이 움직인 결과로 읽힙니다. 그렇다면 실무에서 쓸 처방은 손실 함수를 갈아끼우는 것만으로 끝나지 않습니다. 짧은 학습, 에폭 중간의 체크포인트 평가, 그리고 IPO가 장난감 수준에서만 보인 정규화 효과가 실제 규모에서도 유지되는지에 대한 검증이 함께 필요합니다.

## 참조 모델과 SFT 단계를 덜어낸 랭킹·오즈비·길이 정규화 손실

RRHF는 PPO의 RL 루프를, ORPO와 SimPO는 DPO의 참조 모델을 덜어 냈으며, ORPO는 별도의 선호 최적화 단계까지 SFT 안으로 합쳤고, SimPO는 보상을 생성할 때 쓰이는 평균 로그우도에 맞췄습니다 ([Yuan et al., 2023](https://arxiv.org/abs/2304.05302))([Hong et al., 2024](https://arxiv.org/abs/2403.07691))([Meng et al., 2024](https://arxiv.org/abs/2405.14734)). 앞서 정의한 DPO 손실에서 보상은 참조 정책 대비 로그 비율이었습니다 ([Meng et al., 2024](https://arxiv.org/abs/2405.14734)). 이 설계 때문에 학습 내내 고정된 모델 하나를 메모리에 더 올려야 하고, 참조 정책을 먼저 SFT로 만들어 두는 단계도 필요합니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290))([Meng et al., 2024](https://arxiv.org/abs/2405.14734)). 세 방법은 이 비용을 줄이는 대신, 참조 정책이 맡던 정규화 역할을 다른 항으로 대신해야 했습니다.

### 여러 응답의 순서를 맞추는 RRHF의 랭킹 손실

RRHF는 PPO의 RL 루프를 순위 학습 문제로 바꿨습니다. 질의 하나에 대해 여러 정책에서 샘플한 응답들을 보상으로 점수화하고, 정책이 각 응답에 주는 확률의 순서가 그 보상 순서를 따르도록 학습합니다 ([Yuan et al., 2023](https://arxiv.org/abs/2304.05302)).

![RRHF와 PPO의 학습 작업 흐름 비교도](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-35.png)

*출처: [Yuan et al., 2023](https://arxiv.org/abs/2304.05302)*

이 작업 흐름도는 RRHF와 PPO의 학습 과정을 나란히 놓고 비교합니다 ([Yuan et al., 2023](https://arxiv.org/abs/2304.05302)). 손실은 힌지 형태입니다 ([Yuan et al., 2023](https://arxiv.org/abs/2304.05302)).

$$
L_{rank}=\sum_{r_{i}<r_{j}}\max(0,p_{i}-p_{j})
$$

여기서 $$r_i$$는 응답 $$i$$의 보상 점수이고, $$p_i$$는 정책이 응답 $$i$$에 주는 길이 정규화 로그확률입니다 ([Yuan et al., 2023](https://arxiv.org/abs/2304.05302)). 보상이 낮은 응답 $$i$$의 $$p_i$$가 보상이 높은 응답 $$j$$의 $$p_j$$보다 크면 그 차이만큼 벌점을 받고, 순서가 맞으면 벌점이 0입니다. 마진 항은 없습니다 ([Yuan et al., 2023](https://arxiv.org/abs/2304.05302)). 전체 손실은 이 랭킹 손실과 최고 보상 응답에 대한 SFT 교차엔트로피를 가중치 없이 더한 값입니다 ([Yuan et al., 2023](https://arxiv.org/abs/2304.05302)). 그래서 RL 없이 기존 SFT 코드에 몇 줄만 더하면 구현됩니다 ([Yuan et al., 2023](https://arxiv.org/abs/2304.05302)).

대가도 분명합니다. 질의마다 여러 응답을 입력으로 넣어야 해서, 질의 하나당 GPU 사용량이 PPO보다 늘어납니다 ([Yuan et al., 2023](https://arxiv.org/abs/2304.05302)). 온라인이나 반복 샘플링으로 돌리면 보상 모델을 과최적화하는 경향도 있습니다 ([Yuan et al., 2023](https://arxiv.org/abs/2304.05302)). RRHF는 참조 모델을 쓰지 않는 대신 외부 보상 모델에 기대므로, 초기 RLHF가 겪은 과최적화 문제를 그대로 물려받습니다.

### SFT 안으로 들어간 ORPO의 오즈비 페널티

ORPO의 출발점은 SFT 자체의 부작용입니다. HH-RLHF 데이터로 OPT-350M을 미세조정하면서 선호 응답과 거절 응답의 로그확률을 추적하자, 감독 신호로는 선호 응답만 썼는데도 거절 응답의 로그확률이 함께 올라갔습니다 ([Hong et al., 2024](https://arxiv.org/abs/2403.07691)).

![HH-RLHF에서 OPT-350M을 SFT할 때 선호 응답과 거절 응답의 로그확률 추이](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-36.png)

*출처: [Hong et al., 2024](https://arxiv.org/abs/2403.07691)*

학습이 진행될수록 두 곡선이 나란히 오르고, 때로는 거절 응답 쪽이 더 높아지기도 합니다 ([Hong et al., 2024](https://arxiv.org/abs/2403.07691)). SFT는 원하는 응답의 형식과 문체를 가르치지만, 비슷한 형식의 나쁜 응답까지 함께 끌어올립니다. 그렇다면 SFT를 하는 바로 그 단계에서 거절 응답에 약한 벌점을 주면 별도의 선호 최적화 단계가 필요 없다는 것이 ORPO의 설계입니다 ([Hong et al., 2024](https://arxiv.org/abs/2403.07691)).

![참조 모델 사용 여부와 학습 단계 수로 본 정렬 기법 비교도](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-37.png)

*출처: [Hong et al., 2024](https://arxiv.org/abs/2403.07691)*

여러 정렬 기법을 비교한 도식에서 ORPO만 참조 모델 없이 한 단계로 끝나며, 거절 응답에는 약한 페널티를, 선호 응답에는 강한 적응 신호를 줍니다 ([Hong et al., 2024](https://arxiv.org/abs/2403.07691)). 벌점의 단위는 확률이 아니라 오즈(odds)입니다 ([Hong et al., 2024](https://arxiv.org/abs/2403.07691)).

$$
\textbf{odds}_{\theta}(y|x)=\frac{P_{\theta}(y\mid x)}{1-P_{\theta}(y\mid x)}
$$

$$P_\theta(y\mid x)$$는 정책이 입력 $$x$$에서 응답 $$y$$를 생성할 확률입니다. 두 응답의 오즈를 나눈 오즈비(odds ratio)는 선호 응답이 비선호 응답보다 얼마나 더 생성되기 쉬운지를 나타냅니다 ([Hong et al., 2024](https://arxiv.org/abs/2403.07691)). 오즈비 손실은 이 비의 로그에 로그 시그모이드를 씌웁니다 ([Hong et al., 2024](https://arxiv.org/abs/2403.07691)).

$$
\mathcal{L}_{OR}=-\log\sigma\left(\log\frac{\textbf{odds}_{\theta}(y_{w}|x)}{\textbf{odds}_{\theta}(y_{l}|x)}\right)
$$

$$y_w$$와 $$y_l$$은 각각 선호 응답과 비선호 응답이고, $$\sigma$$는 시그모이드 함수입니다. 식에 $$\pi_{\text{ref}}$$가 없으므로 참조 모델이 필요 없습니다 ([Hong et al., 2024](https://arxiv.org/abs/2403.07691)). 최종 목적함수는 SFT 손실에 이 항을 가중해 더합니다 ([Hong et al., 2024](https://arxiv.org/abs/2403.07691)).

$$
\mathcal{L}_{ORPO}=\mathbb{E}_{(x,y_{w},y_{l})}\left[\mathcal{L}_{SFT}+\lambda\cdot\mathcal{L}_{OR}\right]
$$

$$\mathcal{L}_{SFT}$$는 선호 응답에 대한 일반적인 음의 로그우도이고, $$\lambda$$는 오즈비 항의 비중을 정하는 가중치입니다 ([Hong et al., 2024](https://arxiv.org/abs/2403.07691)).

확률비 대신 오즈비를 고른 이유는 페널티의 세기에 있습니다. 같은 확률 쌍을 넣었을 때 로그 오즈비가 로그 확률비보다 넓은 범위에 분포하고, 그래서 더 부드러운 페널티가 됩니다 ([Hong et al., 2024](https://arxiv.org/abs/2403.07691)).

![같은 확률 쌍에서 샘플한 로그 확률비와 로그 오즈비의 분포](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-38.png)

*출처: [Hong et al., 2024](https://arxiv.org/abs/2403.07691)*

두 분포를 겹쳐 보면 같은 입력 확률 쌍 $$(X_1, X_2)$$에서 로그 오즈비의 범위가 더 넓게 퍼집니다 ([Hong et al., 2024](https://arxiv.org/abs/2403.07691)). 예를 들어 가상의 두 응답 확률이 0.2와 0.1이면 확률비는 2이고, 오즈는 각각 0.25와 약 0.111이어서 오즈비는 약 2.25가 됩니다. 두 확률이 함께 커질수록 오즈의 분모 $$1-P$$가 작아져서 오즈비는 확률비보다 더 빠르게 벌어집니다.

그래디언트를 풀면 이 설계의 동작이 드러납니다. 그래디언트 안의 $$\delta(d)$$ 항은 모델이 거절 응답을 더 생성하려 할 때 파라미터 업데이트를 키우는 페널티로 작동합니다 ([Hong et al., 2024](https://arxiv.org/abs/2403.07691)). 다른 항 $$h(d)$$는 선호 응답과 거절 응답의 그래디언트를 각각 $$1-P(y\mid x)$$로 나눈 가중 대비이며, 분모 때문에 우도가 낮은 쪽의 그래디언트가 증폭됩니다 ([Hong et al., 2024](https://arxiv.org/abs/2403.07691)). 선호 응답의 우도가 아직 낮으면 그쪽을 더 세게 끌어올리고, 거절 응답이 득세하면 더 세게 누르는 구조입니다.

저자들은 검증 범위의 한계도 밝힙니다. 비교한 정렬 알고리즘의 폭이 좁았고, 7B를 넘는 규모로의 확장은 향후 과제로 남겼습니다 ([Hong et al., 2024](https://arxiv.org/abs/2403.07691)). 앞서 본 KTO 연구에서는 참조 모델을 없앤 KTO 변형이 ORPO보다 나았는데, 이 결과도 ORPO의 우위가 아직 확정되지 않았음을 보여 줍니다 ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)).

### 생성 지표에 보상을 맞춘 SimPO

SimPO는 DPO 보상과 생성 과정 사이의 불일치를 짚습니다. 디코딩은 대체로 평균 로그우도가 높은 응답을 고르는데, DPO의 암묵적 보상은 참조 모델 대비 로그 비율이라 이 지표와 일치하지 않습니다 ([Meng et al., 2024](https://arxiv.org/abs/2405.14734)). 저자들이 학습 세트에서 DPO로 학습한 뒤 확인하자, 보상 순위와 평균 로그우도 순위가 일치하는 삼중항은 약 50%뿐이었습니다 ([Meng et al., 2024](https://arxiv.org/abs/2405.14734)). 보상으로는 선호 응답이 이기도록 학습했는데, 생성 기준으로 보면 절반은 여전히 거절 응답을 더 그럴듯하게 본다는 뜻입니다.

![UltraFeedback에서 DPO의 평균 로그확률-길이 상관과 보상 순위-우도 순위 분할표](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-39.png)

*출처: [Meng et al., 2024](https://arxiv.org/abs/2405.14734)*

UltraFeedback에서 측정한 이 비교의 한쪽은 DPO의 평균 로그확률과 응답 길이 사이의 Spearman 상관이고, 다른 쪽은 보상 기준 순위와 평균 로그우도 기준 순위의 분할표입니다 ([Meng et al., 2024](https://arxiv.org/abs/2405.14734)). 해법은 보상을 생성 지표와 같은 형태로 정의하는 것입니다 ([Meng et al., 2024](https://arxiv.org/abs/2405.14734)).

$$
r_{\text{SimPO }}(x,y)=\frac{\beta}{|y|}\log\pi_{\theta}(y\mid x)=\frac{\beta}{|y|}\sum_{i=1}^{|y|}\log\pi_{\theta}(y_{i}\mid x,y_{<i})
$$

$$\lvert y\rvert$$는 응답의 토큰 수, $$y_i$$는 $$i$$번째 토큰, $$\beta$$는 보상의 스케일입니다. 길이로 나눈 평균 로그우도 자체가 보상이고, 참조 모델은 등장하지 않습니다 ([Meng et al., 2024](https://arxiv.org/abs/2405.14734)). 이 보상을 브래들리-테리 손실에 넣고 목표 마진 $$\gamma$$를 뺀 것이 SimPO 목적함수입니다 ([Meng et al., 2024](https://arxiv.org/abs/2405.14734)).

$$
\mathcal{L}_{\text{SimPO }}(\pi_{\theta})=-\mathbb{E}_{(x,y_{w},y_{l})\sim\mathcal{D}}\left[\log\sigma\left(\frac{\beta}{|y_{w}|}\log\pi_{\theta}(y_{w}\mid x)-\frac{\beta}{|y_{l}|}\log\pi_{\theta}(y_{l}\mid x)-\gamma\right)\right]
$$

$$\mathcal{D}$$는 선호 쌍 데이터셋입니다. 마진 $$\gamma$$는 승리 응답의 보상이 패배 응답보다 최소 $$\gamma$$만큼 크도록 강제합니다 ([Meng et al., 2024](https://arxiv.org/abs/2405.14734)). $$\gamma$$를 키우면 처음에는 생성 품질이 오르지만 너무 크면 다시 떨어집니다 ([Meng et al., 2024](https://arxiv.org/abs/2405.14734)). IPO도 비슷한 마진을 쓰지만, 전체 목적함수로 비교하면 SimPO보다 효과가 낮았습니다 ([Meng et al., 2024](https://arxiv.org/abs/2405.14734)).

길이로 나누는 이유는 TL;DR 요약 같은 과제에서 쉽게 보입니다. 예를 들어 가상의 요약 두 개가 있어서, 60토큰짜리 요약의 로그확률 합이 −48이고 20토큰짜리 요약은 −20이라고 합시다. 합으로 비교하면 짧은 요약이 앞서지만, 토큰당 평균으로는 긴 요약이 −0.8로 짧은 요약의 −1.0보다 높습니다. 합을 보상으로 쓰면 보상이 길이와 엉키고, 평균을 쓰면 디코딩이 실제로 보는 값과 같은 척도에서 비교합니다.

길이 정규화의 효과는 상관계수로 확인됩니다. 홀드아웃 세트에서 평균 로그우도와 응답 길이의 Spearman 상관은 길이 정규화를 뺀 SimPO가 0.82, DPO가 0.59, SimPO가 0.34였습니다 ([Meng et al., 2024](https://arxiv.org/abs/2405.14734)). 정규화를 빼면 DPO보다 길이 편향이 커지고, 넣으면 DPO보다 줄어듭니다 ([Meng et al., 2024](https://arxiv.org/abs/2405.14734)).

![보상 정의만 다른 SimPO와 DPO의 AlpacaEval 2·Arena-Hard 성능 비교](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-40.png)

*출처: [Meng et al., 2024](https://arxiv.org/abs/2405.14734)*

두 방법의 차이를 요약한 도식에서 음영으로 표시한 보상 정의만 다르고 나머지 틀은 같으며, AlpacaEval 2와 Arena-Hard의 여러 설정에서 SimPO가 DPO를 크게 앞섭니다 ([Meng et al., 2024](https://arxiv.org/abs/2405.14734)). 다만 두 벤치마크 모두 GPT-4 계열 판정에 기대는 평가이므로, 앞서 DPO에서 확인한 자동 평가의 프롬프트 민감성이 여기에도 적용됩니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)). 저자들은 한계도 적어 둡니다. 목표 마진이라는 하이퍼파라미터를 수동으로 조정해야 하고, 왜 잘 작동하는지에 대한 이론적 분석이 부족합니다 ([Meng et al., 2024](https://arxiv.org/abs/2405.14734)).

세 방법을 함께 보면 무엇을 얻고 무엇을 새로 맞춰야 하는지가 정리됩니다. 참조 모델의 순전파와 SFT 단계를 덜어 메모리와 파이프라인은 가벼워지지만, 참조 정책이 하던 정규화는 사라지지 않고 다른 손잡이로 옮겨 갑니다. RRHF에서는 SFT 교차엔트로피 항, ORPO에서는 $$\lambda$$와 오즈비의 부드러운 페널티, SimPO에서는 길이 정규화와 마진 $$\gamma$$가 그 역할을 맡습니다. 이 글의 판단으로는, 이 손잡이들이 KL 제약만큼 분포 밖 응답을 막아 주는지는 아직 어느 논문도 직접 보여 주지 않았습니다. 앞서 본 직접 정렬 알고리즘의 과최적화 분석은 참조 모델을 둔 손실에서도 고정된 오프라인 데이터의 과소제약이 문제였음을 보였는데 ([Rafailov et al., 2024](https://arxiv.org/abs/2406.02900)), 참조 모델까지 뺀 손실에서 같은 측정을 하는 일은 남은 과제입니다.

## 브래들리-테리를 넘어선 선호 게임과 현재 정책 샘플링

비추이적인 선호와 분포 이동을 함께 다루려면 두 가지를 바꿔야 합니다. 목표는 응답마다 점수를 매기는 보상 대신 두 정책이 맞붙는 선호 게임의 균형이 되고, 데이터는 고정된 오프라인 쌍 대신 현재 정책이 방금 생성한 샘플이 됩니다.

### 점수 하나로 담기지 않는 선호와 정규화된 내시 균형

NLHF는 보상 모델 학습을 건너뛰고, 두 응답을 받아 어느 쪽이 나은지의 확률을 내는 선호 모델을 학습합니다 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)). 정렬 목표도 보상 최대화가 아니라 이 선호 모델의 내시 균형(Nash equilibrium) 정책을 찾는 것입니다 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)). 선호 모델을 고른 이유는 표현력에 있습니다. 보상 모델은 응답마다 점수 하나를 매기므로 비추이적 선호를 표현할 수 없지만, 쌍을 직접 받는 선호 모델은 표현할 수 있습니다 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)).

TL;DR 요약으로 옮기면 비추이성이 어떤 모습인지 드러납니다. 예를 들어 가상의 요약 A, B, C가 있고, A는 B보다 핵심을 잘 짚어 이기고, B는 C보다 간결해서 이기고, C는 A보다 원문의 어조를 잘 살려 이긴다고 합시다. 점수가 $$s_A > s_B > s_C > s_A$$를 동시에 만족할 수는 없으므로, 어떤 보상 모델도 이 세 비교를 모두 맞히지 못합니다.

저자들의 세 행동 예시에서 이 차이는 해의 모양으로 나타납니다. 브래들리-테리 보상을 최대화하는 RLHF 해는 $$y_1$$ 하나만 결정적으로 고르지만, 선호 모델의 내시 균형은 $$y_1$$과 $$y_2$$에 각각 $$1/3+\epsilon/2$$, $$y_3$$에 $$1/3-\epsilon$$의 확률을 줍니다 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)). 선호를 조금 바꾸는 $$\epsilon$$에 대해 RLHF 해는 한 행동으로 튀지만 내시 해는 거의 균등한 분포에서 매끄럽게 움직입니다 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)). 저자들은 이 성질 덕분에 내시 균형이 사람 선호의 다양성에 더 잘 맞을 수 있다고 봅니다 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)).

두 번째 근거는 데이터 분포 의존성입니다. 표현력이 완벽한 이상적인 경우, 보상 모델은 학습 데이터를 만든 프롬프트 분포와 두 정책에 따라 달라지지만 선호 모델은 그 분포에 영향을 받지 않습니다 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)). 그래서 반복 라운드마다 새 정책이 데이터를 만들어도 선호 모델을 버리지 않고 재사용하거나 확장할 수 있습니다 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)).

게임에도 KL 제약이 필요합니다. NLHF는 참조 정책과의 로그 비율을 선호 자체에 넣어 정규화된 선호를 정의합니다 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)).

$$
\mathcal{P}_{\tau}^{\pi,\pi^{\prime}}(y\succ y^{\prime}\mid x)\stackrel{\text{def}}{=}\mathcal{P}(y\succ y^{\prime}\mid x)-\tau\log\frac{\pi(y\mid x)}{\mu(y\mid x)}+\tau\log\frac{\pi^{\prime}(y^{\prime}\mid x)}{\mu(y^{\prime}\mid x)}
$$

$$y\sim\pi(\cdot\mid x)$$와 $$y^{\prime}\sim\pi^{\prime}(\cdot\mid x)$$는 두 정책이 낸 응답, $$\mathcal{P}$$는 학습된 선호 모델, $$\mu$$는 참조 정책, $$\tau$$는 정규화 세기입니다 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)). 참조 정책에서 멀어진 응답은 선호에서 그만큼 감점되므로, 이 게임의 균형이 정규화된 내시 균형이 됩니다 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)).

균형을 찾는 알고리즘이 Nash-MD입니다. 현재 정책과 참조 정책을 기하 평균으로 섞은 정책 $$\pi_t^{\mu}$$를 기준으로 미러 디센트 한 스텝을 밟습니다 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)). 명시적으로 쓰면 다음과 같습니다 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)).

$$
\pi_{t+1}(y)\propto\pi_{t}^{\mu}(y)\exp\left(\eta_{t}\mathcal{P}(y\succ\pi_{t}^{\mu})\right)
$$

$$\eta_t$$는 스텝 크기이고, $$\mathcal{P}(y\succ\pi_{t}^{\mu})$$는 응답 $$y$$가 혼합 정책의 샘플을 이길 확률입니다 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)). 혼합 정책을 상대로 많이 이기는 응답일수록 지수적으로 확률이 커집니다. 표 형태 설정에서 $$\eta_t=2/(\tau(t+2))$$로 두면 마지막 반복이 $$\mathrm{KL}(\pi^{*}_{\tau},\pi_{T})\leq 8/(\tau^{2}(T+1))$$로, 즉 $$O(1/T)$$ 속도로 정규화된 내시 균형에 수렴합니다 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)). 이 보장은 평균 반복이 아니라 마지막 반복에 대한 것입니다. 다만 언어 모델에 적용한 Nash-MD-PG는 파라미터화된 정책에서 경사 스텝 하나만 밟으므로, 미러 디센트를 구현한 것이 아니라 정규화된 선호에 대한 단순 경사 하강이고 이 수렴 보장도 그대로 옮겨 오지 않습니다 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)).

![선호 모델 크기별 학습·테스트 정확도 학습 곡선](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-41.png)

*출처: [Munos et al., 2023](https://arxiv.org/abs/2312.00886)*

선호 모델의 품질은 크기에 따라 갈립니다. 110M, 3B, 11B 세 크기를 나란히 놓은 학습 곡선에서 클수록 정확도가 높지만, 3B에서 11B로 갈 때의 이득은 작습니다 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)).

![같은 크기의 선호 모델과 보상 모델의 학습·테스트 정확도 비교](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-42.png)

*출처: [Munos et al., 2023](https://arxiv.org/abs/2312.00886)*

같은 크기의 선호 모델과 보상 모델을 학습 세트와 테스트 세트에서 겨룬 정확도 곡선도 함께 제시됩니다 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)). 정책 실험의 성격에 대해서는 저자들이 직접 선을 긋습니다. 실험은 NLHF가 RLHF보다 낫다는 주장이 아니라 구현이 가능하다는 시연이고, 하이퍼파라미터도 최적화하지 않았습니다 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)).

### 고정 데이터를 현재 정책 샘플로 바꾼 Online IPO

Online IPO의 변경점은 하나입니다. 앞서 본 샘플 기반 IPO 손실 ([Azar et al., 2023](https://arxiv.org/abs/2310.12036))에서 고정된 데이터 분포 자리에, 현재 정책 $$\pi$$가 생성하고 기울기를 멈춘(stop-gradient) 응답 쌍을 넣습니다 ([Calandriello et al., 2024](https://arxiv.org/abs/2403.08635)). 샘플 단위의 손실 식은 오프라인 IPO와 똑같고, 데이터를 어디서 뽑느냐만 다릅니다 ([Calandriello et al., 2024](https://arxiv.org/abs/2403.08635)). 앞서 IPO가 지적한 결정적 선호에서의 과적합과 직접 정렬 알고리즘이 고정 데이터의 좁은 커버리지 때문에 과소제약된다는 진단 ([Rafailov et al., 2024](https://arxiv.org/abs/2406.02900))을 떠올리면, 손실은 그대로 두고 데이터를 정책 근처로 옮기는 처방이 자연스럽습니다.

IPO-MD는 Nash-MD의 혼합 정책 아이디어를 데이터 생성에 빌려 옵니다. 온라인 정책과 참조 정책을 기하 평균으로 섞은 정책에서 쌍을 뽑으며, 혼합 비율 $$\beta=0$$이면 셀프 플레이(self-play)인 Online IPO, $$\beta=1$$이면 고정된 참조 정책을 상대로 개선하는 IPO 변형이 됩니다 ([Calandriello et al., 2024](https://arxiv.org/abs/2403.08635)). 하나의 손잡이로 셀프 플레이와 오프라인에 가까운 학습 사이를 오갈 수 있습니다. 저자들은 결과가 요약 과제 하나에 한정되며 일반 대화 에이전트에서는 추가 검증이 필요하다고 밝힙니다 ([Calandriello et al., 2024](https://arxiv.org/abs/2403.08635)).

### 승률을 직접 회귀하는 셀프 플레이, SPPO

SPPO는 정규화 없는 선호 게임의 이상적 갱신에서 출발합니다. 현재 정책을 상대로 한 승률로 가중하는 곱셈 가중치 갱신입니다 ([Wu et al., 2024](https://arxiv.org/abs/2405.00675)).

$$
\pi_{t+1}(\mathbf{y}\mid\mathbf{x})\propto\pi_{t}(\mathbf{y}\mid\mathbf{x})\exp(\eta\mathbb{P}(\mathbf{y}\succ\pi_{t}\mid\mathbf{x}))
$$

$$\eta$$는 학습률, $$\mathbb{P}(\mathbf{y}\succ\pi_{t}\mid\mathbf{x})$$는 응답 $$\mathbf{y}$$가 현재 정책의 응답을 이길 확률입니다 ([Wu et al., 2024](https://arxiv.org/abs/2405.00675)). 비례식을 로그 비율로 풀면 정규화 상수 $$Z_{\pi_t}(\mathbf{x})$$가 드러납니다 ([Wu et al., 2024](https://arxiv.org/abs/2405.00675)).

$$
\log\bigg(\frac{\pi_{t+1}(\mathbf{y}\mid\mathbf{x})}{\pi_{t}(\mathbf{y}\mid\mathbf{x})}\bigg)=\eta\cdot\mathbb{P}(\mathbf{y}\succ\pi_{t}\mid\mathbf{x})-\log Z_{\pi_{t}}(\mathbf{x})
$$

SPPO는 이 등식의 양변 차이를 L2 거리로 직접 줄입니다 ([Wu et al., 2024](https://arxiv.org/abs/2405.00675)). 설계상 갈림길은 여기 있습니다. DPO와 IPO는 두 응답의 로그 비율 차이를 써서 정규화 상수를 지우지만, SPPO는 응답마다 로그 비율을 따로 목표값에 맞춥니다 ([Wu et al., 2024](https://arxiv.org/abs/2405.00675)). 그래서 승자와 패자의 간격만 1로 벌리는 것이 아니라 승자를 $$1/2$$ 근처, 패자를 $$-1/2$$ 근처로 각각 밀어, 승자 우도가 거의 변하지 않는 DPO의 문제를 덜어 줍니다 ([Wu et al., 2024](https://arxiv.org/abs/2405.00675)). 저자들은 DPO와 KTO에는 내시 균형으로의 수렴 보장이 없다는 점도 대비시킵니다 ([Wu et al., 2024](https://arxiv.org/abs/2405.00675)).

한 라운드의 절차는 네 단계입니다 ([Wu et al., 2024](https://arxiv.org/abs/2405.00675)).

1. 프롬프트마다 현재 정책 $$\pi_t$$에서 응답 $$K$$개를 샘플링합니다.
2. 선호 모델로 응답 쌍마다 승률을 매기며, 질의 수는 $$O(K^2)$$입니다.
3. 응답별 승률 추정치를 붙여 데이터셋을 만듭니다.
4. 앞서 본 제곱 손실로 다음 정책 $$\pi_{t+1}$$을 학습합니다.

예를 들어 가상의 TL;DR 프롬프트에서 $$K=3$$개 요약의 추정 승률이 0.8, 0.5, 0.2이고 $$\eta=1$$, $$\log Z$$를 상수 0.5로 근사한다면, 각 요약의 로그 비율 목표는 0.3, 0, −0.3이 됩니다. 평균보다 자주 이기는 요약만 확률이 오르고, 평균 수준인 요약은 그대로 남습니다. 이 상수 근사는 이론상 soft value에 가까울 때만 분산을 줄이며, 이론 전체도 모델 클래스의 표현력과 데이터 커버리지를 가정합니다 ([Wu et al., 2024](https://arxiv.org/abs/2405.00675)).

![반복마다 오르는 AlpacaEval 2.0 길이 통제·원 승률](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-43.png)

*출처: [Wu et al., 2024](https://arxiv.org/abs/2405.00675)*

AlpacaEval 2.0에서 GPT-4-Turbo 대비 승률을 반복별로 그린 결과는 길이 통제(LC) 승률과 원 승률이 모두 꾸준히 오르는 흐름을 보여 줍니다 ([Wu et al., 2024](https://arxiv.org/abs/2405.00675)). Mistral-7B 기반 SPPO의 LC 승률 상승폭은 논문이 퍼센트로 보고한 값으로 Mistral-7B-Instruct에서 1회차까지 7.69, 1회차에서 2회차까지 2.10, 2회차에서 3회차까지 1.64입니다 ([Wu et al., 2024](https://arxiv.org/abs/2405.00675)). 반복을 거듭할수록 이득이 빠르게 줄어듭니다. Open LLM Leaderboard 평균 점수는 Mistral-7B가 2회차에서 66.75, Llama-3-8B가 70.29였습니다 ([Wu et al., 2024](https://arxiv.org/abs/2405.00675)).

![승률 추정 미니배치 크기 K별 AlpacaEval 2.0 승률 비교](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-44.png)

*출처: [Wu et al., 2024](https://arxiv.org/abs/2405.00675)*

승률 $$\mathbb{P}(\mathbf{y}\succ\pi_{t}\mid\mathbf{x})$$를 추정하는 미니배치 크기를 $$K=2$$와 $$K=5$$로 바꾼 ablation은 LC 승률과 원 승률 양쪽에서 비교됩니다 ([Wu et al., 2024](https://arxiv.org/abs/2405.00675)). 선호 모델 질의가 $$K$$의 제곱으로 늘어나므로 이 비교가 곧 비용 결정입니다. 실험은 계산 자원 제약으로 UltraFeedback 한 데이터셋과 일부 벤치마크에 한정됩니다 ([Wu et al., 2024](https://arxiv.org/abs/2405.00675)).

### 오픈소스 환경을 위한 반복 DPO 레시피

RLHF Workflow는 오픈소스 RLHF가 오프라인 설정에 머물러 있다는 간극에서 출발합니다 ([Dong et al., 2024](https://arxiv.org/abs/2405.07863)). 온라인으로 사람 피드백을 받기 어려우므로, 여러 공개 데이터셋으로 학습한 프록시 선호 모델로 사람 피드백을 근사하는 재현 가능한 온라인 반복 레시피를 제시합니다 ([Dong et al., 2024](https://arxiv.org/abs/2405.07863)).

![이력 데이터가 누적되는 반복 직접 선호 학습 구현도](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-45.png)

*출처: [Dong et al., 2024](https://arxiv.org/abs/2405.07863)*

구현도에서 따라갈 지점은 이력 데이터셋이 자라는 방식입니다. 첫 반복에는 이력 데이터가 비어 있어 정책이 초기화 상태인 SFT 체크포인트와 같고, 이후 반복마다 앞선 반복의 선호 데이터가 쌓입니다 ([Dong et al., 2024](https://arxiv.org/abs/2405.07863)). 각 반복에서는 누적 데이터로 SFT 참조 모델 기준 DPO를 하고, 프롬프트마다 응답 8개를 샘플링해 보상 모델이 매긴 최고·최저 응답을 선호 쌍으로 데이터에 더하며, 다음 학습은 이전 반복 모델에서 이어 갑니다 ([Dong et al., 2024](https://arxiv.org/abs/2405.07863)). 쌍을 고르는 데에는 결국 보상 모델의 순위가 쓰이므로, 보상 모델은 RL 루프에서 빠졌다가 데이터 생성기로 돌아온 셈입니다.

이 글의 판단으로는, 네 방법을 실용적으로 만든 공통분모는 게임 이론적 목표보다 현재 정책 샘플링 쪽에 있습니다. NLHF와 SPPO는 비추이적 선호를 다루는 이론을 갖췄지만, 보고된 실험만으로는 내시 균형이라는 목표 자체가 브래들리-테리 기반 반복 DPO보다 얼마나 이득인지 분리되지 않습니다. NLHF는 우위 비교를 의도하지 않았고 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)), Online IPO는 요약 과제 하나 ([Calandriello et al., 2024](https://arxiv.org/abs/2403.08635)), SPPO는 데이터셋 하나 ([Wu et al., 2024](https://arxiv.org/abs/2405.00675))에서 평가했습니다.

## 가치 모델 없이 그룹 평균을 기준선으로 삼는 RL과 검증 가능한 보상

GRPO는 PPO의 가치 모델(value model)을 같은 질문에 대한 출력 그룹의 평균 보상으로 대신해 학습 자원을 줄이고 수학 추론 성능을 끌어올렸습니다. 후속 분석은 정답 여부로 판정하는 이진 보상에서 GRPO가 성공 확률에 따라 가중치가 달라지는 대조 손실로 작동하며, 그 결과 성공 확률을 증폭한다는 것을 보였습니다.

### 크리틱을 그룹 점수로 바꾼 이유

PPO의 가치 모델은 보통 정책 모델과 크기가 비슷해 메모리와 계산 부담이 큽니다 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)). 언어 모델 RL에서는 보상이 마지막 토큰에만 주어져 토큰별 가치를 학습하기도 어렵습니다 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)). GRPO는 별도의 가치 함수 근사를 아예 두지 않고, 같은 질문에 대해 샘플한 여러 출력의 평균 보상을 기준선으로 씁니다 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)).

![PPO와 GRPO의 학습 구조 비교도](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-46.png)

*출처: [Shao et al., 2024](https://arxiv.org/abs/2402.03300)*

두 구조를 나란히 놓은 비교도에서 짚을 지점은 GRPO 쪽에서 가치 모델이 빠지고, 그 자리를 그룹 점수에서 추정한 기준선이 채운다는 것입니다 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)). 저자들은 이 교체로 학습 자원이 크게 줄어든다고 설명합니다 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)).

목적함수는 PPO의 클리핑 비율 구조를 유지하되 어드밴티지(advantage)를 그룹 상대값으로 바꿉니다 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)).

$$
\begin{aligned}\mathcal{J}_{GRPO}(\theta)&=\mathbb{E}{[q\sim P(Q),\{o_{i}\}_{i=1}^{G}\sim\pi_{\theta_{old}}(O\mid q)]}\\
&\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_{i}|}\sum_{t=1}^{|o_{i}|}\left\{\min\left[\frac{\pi_{\theta}(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}\mid q,o_{i,<t})}\hat{A}_{i,t},\text{clip}\left(\frac{\pi_{\theta}(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}\mid q,o_{i,<t})},1-\varepsilon,1+\varepsilon\right)\hat{A}_{i,t}\right]-\beta\mathbb{D}_{KL}\left[\pi_{\theta}\Vert\pi_{ref}\right]\right\}\end{aligned}
$$

$$q$$는 질문, $$o_i$$는 이전 정책 $$\pi_{\theta_{old}}$$에서 뽑은 $$G$$개 출력 중 $$i$$번째, $$o_{i,t}$$는 그 출력의 $$t$$번째 토큰입니다. $$\hat{A}_{i,t}$$는 토큰별 어드밴티지, $$\varepsilon$$은 클리핑 범위, $$\beta$$는 참조 정책 $$\pi_{ref}$$에 대한 KL 페널티 계수입니다. PPO가 KL 페널티를 보상에 섞는 것과 달리, GRPO는 KL 발산을 손실에 직접 더해 $$\hat{A}_{i,t}$$ 계산을 복잡하게 만들지 않습니다 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)). KL 항은 Schulman의 불편추정량으로 계산하며, 이 값은 항상 양수입니다 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)).

$$
\mathbb{D}_{KL}\left[\pi_{\theta}\Vert\pi_{ref}\right]=\frac{\pi_{ref}(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta}(o_{i,t}\mid q,o_{i,<t})}-\log\frac{\pi_{ref}(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta}(o_{i,t}\mid q,o_{i,<t})}-1
$$

결과 감독(outcome supervision)에서는 출력마다 받은 보상을 그룹 평균과 표준편차로 정규화해, 그 값을 출력 안의 모든 토큰에 같은 어드밴티지로 줍니다 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)).

$$
\hat{A}_{i,t}=\widetilde{r}_{i}=\frac{r_{i}-{\rm mean}(\mathbf{r})}{% raw %}{{{% endraw %}\rm std}(\mathbf{r})}
$$

여기서 $$\mathbf{r}$$은 그룹 $$G$$개 출력의 보상 벡터입니다. 과정 감독(process supervision)에서는 추론 단계마다 보상을 주고 정규화한 뒤, 각 토큰의 어드밴티지를 그 토큰 이후 단계들의 정규화 보상 합으로 계산합니다 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)).

정답이면 1, 오답이면 0을 주는 가상의 그룹 $$G=4$$를 예로 들어 보겠습니다. 두 개가 맞고 두 개가 틀리면 평균 0.5, 모집단 표준편차 0.5이므로 정답 출력의 어드밴티지는 +1, 오답은 −1입니다. 세 개가 맞고 하나가 틀리면 평균 0.75, 표준편차 약 0.433이 되어 정답은 약 +0.58, 하나뿐인 오답은 약 −1.73을 받습니다. 그룹이 이미 대부분 맞힐 때는 드문 실패에 큰 벌점이 몰립니다. 실제 DeepSeekMath-RL 학습은 질문당 출력 64개를 샘플하고 KL 계수를 0.04로 둡니다 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)).

### online RFT보다 나은 이유와 RL이 바꾸는 것

![DeepSeekMath-Instruct 1.3B를 여러 방법으로 추가 학습한 두 벤치마크 성능](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-47.png)

*출처: [Shao et al., 2024](https://arxiv.org/abs/2402.03300)*

DeepSeekMath-Instruct 1.3B를 RFT, online RFT, GRPO 등으로 추가 학습해 두 벤치마크에서 비교한 결과에서 GRPO가 online RFT를 앞섭니다 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)). 저자들은 그 차이를 그래디언트 계수에서 찾습니다. online RFT는 정답 출력만 같은 강도로 강화하지만, GRPO는 보상 크기에 따라 계수가 달라져 정답은 강화하고 오답은 억제합니다 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)). 단계별 보상을 쓰는 GRPO+PS가 결과 보상만 쓰는 GRPO+OS보다 낫다는 결과도 같은 비교에 들어 있습니다 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)). 보상 모델을 갱신하며 RL을 되풀이하는 반복 RL은 성능을 크게 올리고, 효과는 첫 반복에서 특히 큽니다 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)).

![SFT와 RL 모델의 GSM8K·MATH Maj@K와 Pass@K 비교](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-48.png)

*출처: [Shao et al., 2024](https://arxiv.org/abs/2402.03300)*

DeepSeekMath 7B의 SFT 모델과 RL 모델을 GSM8K와 MATH에서 온도 0.7로 비교하면, RL은 Maj@K를 높이지만 Pass@K는 높이지 않습니다 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)). K개 샘플 중 하나라도 맞히는 능력은 그대로이고, 다수결로 맞히는 비율만 오릅니다.

### 성공 확률로 가중하는 대조 손실이라는 해석

Mroueh의 GRPO Dynamics는 이 현상에 이론적 설명을 붙입니다. 검증 가능한 보상(verifiable reward)이 이진일 때, 평균과 분산으로 보상을 보정하는 GRPO는 이전 정책이 만든 합성 샘플을 대조 샘플로 쓰는 대조 손실과 같습니다 ([Mroueh, 2025](https://arxiv.org/abs/2503.06639)). 가중치는 이전 정책 $$\pi_{\theta_{old}}(\cdot\mid q)$$의 성공 확률 $$p(q)$$에 따라 달라집니다 ([Mroueh, 2025](https://arxiv.org/abs/2503.06639)). $$p$$가 높으면 실패 샘플에 큰 벌점이 가고, $$p$$가 낮으면 성공 샘플에 큰 가중치가 갑니다 ([Mroueh, 2025](https://arxiv.org/abs/2503.06639)).

![이전 정책의 성공 확률에 따른 GRPO의 성공·실패 샘플 가중치](/assets/images/from-rlhf-to-dpo-and-grpo-preference-optimization-for-llm-alignment-49.png)

*출처: [Mroueh, 2025](https://arxiv.org/abs/2503.06639)*

가중치 곡선은 앞의 가상 그룹 계산이 우연이 아님을 보여 줍니다. 성공 확률이 움직이는 방향에 따라 성공 샘플과 실패 샘플 중 어느 쪽 가중치가 커지는지 읽으면 됩니다 ([Mroueh, 2025](https://arxiv.org/abs/2503.06639)). 정책을 반복 갱신하면 성공 확률은 고정점으로 수렴합니다 ([Mroueh, 2025](https://arxiv.org/abs/2503.06639)). 참조 정책의 성공 확률 $$p_{\mathrm{ref}}(q)$$가 0이나 1 같은 양 끝이 아니면, 고정점 $$p^{*}(q)$$는 $$p_{\mathrm{ref}}(q)$$보다 큽니다 ([Mroueh, 2025](https://arxiv.org/abs/2503.06639)). 반대로 참조 성공 확률이 양 끝값이면 고정점이 반드시 안정적이지 않아, 안정성을 위해 $$\beta$$에 조건이 필요합니다 ([Mroueh, 2025](https://arxiv.org/abs/2503.06639)).

이 글의 관점에서는 두 논문의 결과가 한 그림으로 맞물립니다. 이론이 말하는 증폭은 참조 모델이 이미 어느 정도 맞히는 질문에서만 작동하고, 실험에서 RL이 Pass@K가 아니라 Maj@K만 올린 것도 그 범위와 들어맞습니다. 다만 GRPO Dynamics의 분석은 이진 보상과 질문 단위 성공 확률이라는 단순화 위에 서 있어, 대규모 실험의 Pass@K 정체를 직접 검증한 것은 아닙니다.

## 보상 모델 유무와 상관없이 되풀이되는 분포 이동과 과최적화

계보마다 손실의 모양은 달라도, 이 방법들이 실제로 다투는 질문은 하나입니다. KL로 참조 정책에 묶인 정책이 어떤 분포에서 나온 데이터로 학습하는가입니다.

### 정적 피드백의 실패에서 나온 첫 교훈

이 질문은 언어 모델 이전에 이미 드러났습니다. Deep RL from Preferences에서 학습 초기에만 선호를 모은 오프라인 보상 학습은 정책이 움직이면서 상태 점유 분포가 계속 바뀌는 비정상성 때문에 보상의 일부만 포착했습니다 ([Christiano et al., 2017](https://arxiv.org/abs/1706.03741)). 그 결과 Pong에서 점수를 내지 않고 랠리만 끝없이 이어 가는 이상 행동이 나왔고, 저자들은 인간 피드백을 처음에 한 번 주는 대신 RL과 번갈아 진행해야 한다고 결론지었습니다 ([Christiano et al., 2017](https://arxiv.org/abs/1706.03741)). LM Fine-Tuning from Preferences는 이 교훈을 언어 모델에 그대로 적용했습니다. RL을 진행하는 동안 현재 정책의 샘플에 라벨을 계속 모으고 보상 모델을 다시 학습해 분포 이동을 막았습니다 ([Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)). 이 온라인 수집은 요약 과제에서는 중요했지만 단순한 스타일 과제에서는 필요하지 않았습니다 ([Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)). 두 요약 중 더 나은 쪽을 고르는 TL;DR 같은 과제가 이후 계보의 공통 무대가 된 데에도 이런 사정이 있습니다. 출력 공간이 넓어서, 정책이 조금만 움직여도 처음 모은 비교 데이터가 다루지 못하는 영역으로 금방 넘어갑니다.

### 보상 모델을 없애도 사라지지 않는 과최적화

DPO 계열의 직접 정렬 알고리즘(direct alignment algorithm)은 보상 모델을 정책 안에 숨겼지만, 데이터는 여전히 고정된 오프라인 쌍입니다. 앞서 본 DAA 과최적화 연구는 명시적 보상 모델이 없어도 KL 예산이 커질수록 품질이 꺾이는 현상이 반복된다는 사실을 보였습니다. 같은 연구는 길이 정규화(length normalization)로도 이 문제가 풀리지 않는다고 보고합니다. 목적함수에 길이 보정을 넣으면 KL과 승률 사이의 파레토 경계는 달라지지만, 보상 과최적화는 완화되지 않고 오히려 악화될 수 있습니다 ([Rafailov et al., 2024](https://arxiv.org/abs/2406.02900)). 용량이 작은 모델이나 KL 예산이 작은 설정에서는 정책이 길이 같은 단순한 특징을 따라 더 강하게 외삽했습니다 ([Rafailov et al., 2024](https://arxiv.org/abs/2406.02900)). 앞서 본 SimPO의 길이 정규화는 평균 로그우도와 응답 길이의 상관을 낮췄습니다 ([Meng et al., 2024](https://arxiv.org/abs/2405.14734)). 하지만 이 효과는 학습 데이터 분포 안의 길이 편향을 다룬 결과입니다. 정책이 데이터 밖으로 나가면서 생기는 외삽까지 막아 주지는 않습니다.

### 오프라인 손실을 온라인화하면 도착하는 내시 균형

온라인 계보는 데이터 분포를 바꾸는 쪽에서 답을 찾았습니다. Online IPO는 앞서 설명한 대로 IPO 손실을 그대로 두고 고정 데이터만 현재 정책의 샘플로 바꿨습니다 ([Calandriello et al., 2024](https://arxiv.org/abs/2403.08635)). 이 목적함수의 최소해는 정규화된 선호 게임의 내시 균형이고, Nash-MD-PG가 푸는 해와 같습니다 ([Calandriello et al., 2024](https://arxiv.org/abs/2403.08635)). 브래들리-테리를 버리고 선호 게임을 정의한 NLHF와, 쌍대 손실만 다듬던 IPO가 데이터 분포 하나를 바꾸자 같은 지점에서 만났습니다. 손실의 형태보다 샘플이 어느 정책에서 오느냐가 해의 성격을 더 크게 좌우했습니다.

### 그룹 비교로 다시 들어온 상대 평가

GRPO는 반대 방향에서 같은 결론에 이릅니다. 같은 질문에 대한 출력 그룹 안에서 어드밴티지를 계산하는 방식은, 같은 질문의 출력끼리 비교하도록 학습된 보상 모델의 성격과 잘 맞습니다 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)). PPO는 토큰별 KL을 보상에 섞어 넣지만, GRPO는 KL을 손실에 직접 더합니다 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)). 결과적으로 GRPO는 현재 정책에서 뽑은 출력들 사이의 상대 비교를 KL로 묶은 형태가 됩니다. 정답 검증으로 보상을 주는 점만 다를 뿐, 구조는 온라인 쌍대 선호 학습과 가깝습니다.

> **핵심:** 오프라인 직접 정렬의 과최적화, 길이 보정의 한계, 온라인 IPO와 내시 균형의 동치, GRPO의 그룹 비교는 모두 "현재 정책의 분포에서 비교하라"는 같은 처방을 가리킵니다.

이 글은 세 갈래를 서로의 한계에 대한 답으로 읽습니다. 오프라인 직접 정렬은 RL 루프를 없앤 대신 초기 RLHF가 피하려던 정적 피드백 문제를 다시 떠안았습니다. 온라인 셀프 플레이는 데이터를 현재 정책으로 되돌려 그 문제를 풀지만, 매 라운드 샘플링과 선호 판정에 비용이 듭니다. 크리틱 없는 RL은 그 비용 중 가치 모델 몫만 덜어 내고 온라인성은 그대로 지킵니다. 다만 Online IPO의 동치는 해의 성질에 관한 결과일 뿐입니다. 어느 접근이 실제로 과최적화를 덜 겪는지는 같은 KL 예산에서 직접 비교한 연구가 아직 부족합니다.

## 데이터 형태, 메모리 비용, 온라인성 사이의 맞교환

비교표의 핵심 방법들은 하나의 축에서 우열을 가릴 수 없습니다. 어떤 데이터를 받는지, 학습 중에 몇 개의 모델을 메모리에 올리는지, 데이터를 언제 만드는지, 무엇으로 평가받는지가 방법마다 다르고, 각자 다른 지점에서 비용과 성능을 맞바꿉니다. 아래 표는 계보별 대표 논문의 주요 결과와 한계를 한 줄씩 모은 것입니다. 표를 읽을 때는 주요 결과 열의 숫자끼리 바로 견주지 말고, 각 숫자 옆에 붙은 모델, 과제, 지표가 무엇인지부터 확인해야 합니다.

### 쌍, 이진, 그룹으로 갈리는 데이터 형태

DPO, IPO, SimPO, ORPO는 모두 선호 응답과 거절 응답이 짝을 이룬 쌍 데이터를 전제로 합니다 ([Meng et al., 2024](https://arxiv.org/abs/2405.14734))([Hong et al., 2024](https://arxiv.org/abs/2403.07691)). KTO는 입력마다 바람직한지 아닌지만 표시된 이진 라벨로 학습하므로, 쌍을 만들기 어려운 데이터에서도 쓸 수 있습니다. SPPO는 프롬프트당 응답을 여러 개 뽑고 선호 모델로 모든 쌍의 승률을 매기는데, 이 질의 수는 응답 수의 제곱에 비례합니다 ([Wu et al., 2024](https://arxiv.org/abs/2405.00675)). GRPO는 쌍 비교 없이 같은 질문에 대한 출력 그룹의 보상을 평균과 표준편차로 정규화해 어드밴티지로 씁니다 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)). 쌍 데이터를 요구하지 않는 방법일수록 라벨 수집은 쉬워지지만, 그만큼 샘플링이나 보상 계산 쪽으로 비용이 옮겨 갑니다.

### 참조 모델과 크리틱이 정하는 메모리 비용

PPO 기반 RLHF는 정책, 참조 정책, 보상 모델, 가치 모델을 함께 메모리에 올립니다. GRPO는 이 중 정책과 크기가 비슷한 가치 모델을 그룹 평균 기준선으로 대신했지만 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)), KL 항을 계산하려면 참조 정책은 여전히 필요합니다 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)). DPO는 RL 루프를 없앴어도 참조 모델의 순전파가 남아 있습니다. SimPO와 ORPO는 이 순전파까지 걷어 냈습니다. SimPO는 길이 정규화된 평균 로그확률을 보상으로 쓰고, ORPO는 SFT 손실에 오즈비 항을 더합니다 ([Meng et al., 2024](https://arxiv.org/abs/2405.14734))([Hong et al., 2024](https://arxiv.org/abs/2403.07691)).

같은 조건에서 손실만 바꾼 비교를 보면 이 단순화의 효과가 드러납니다. Mistral-Base 7B를 AlpacaEval 2 LC 승률로 평가했을 때 SimPO는 21.5%로 DPO의 15.1%보다 6.4%p 높았고, 같은 설정에서 R-DPO는 17.4%, ORPO는 14.7%, KTO는 13.1%, IPO는 11.8%였습니다 ([Meng et al., 2024](https://arxiv.org/abs/2405.14734)). 다만 이 결과는 GPT-4가 판정하는 지시 따르기 벤치마크에서 나왔습니다. SimPO 자체도 수동으로 조정해야 하는 마진 하이퍼파라미터를 새로 들여왔습니다 ([Meng et al., 2024](https://arxiv.org/abs/2405.14734)).

### 오프라인 한 번과 온라인 여러 라운드

오프라인 방법은 데이터를 한 번 모아 한 번 학습하므로 파이프라인이 가장 단순합니다. 대신 앞서 본 과최적화와 분포 이동의 문제를 그대로 떠안습니다 ([Rafailov et al., 2024](https://arxiv.org/abs/2406.02900)). 온라인 계보는 매 라운드 현재 정책에서 새로 샘플링하고 판정자를 다시 거칩니다. RLHF Workflow는 프롬프트마다 응답을 여러 개 뽑아 보상 모델로 가장 좋은 것과 가장 나쁜 것을 골라 쌍을 만듭니다 ([Dong et al., 2024](https://arxiv.org/abs/2405.07863)). SPPO는 반복 초기에 큰 향상을 얻고 이후 라운드에서는 이득이 점점 줄어듭니다 ([Wu et al., 2024](https://arxiv.org/abs/2405.00675)). 온라인 라운드를 몇 번 돌릴지는 이 체감 곡선과 샘플링 비용을 함께 보고 정해야 합니다.

### 사람, GPT-4 판정, 정답 검증이라는 서로 다른 잣대

평가 방식도 방법마다 다릅니다. DPO는 TL;DR 요약에서 PPO와 비교되었고, Online IPO의 온라인 실험도 요약 과제에서 이루어졌습니다. 그러나 어떤 결과는 사람 선호로, 어떤 결과는 GPT-4 승률로, 어떤 결과는 쌍대 side-by-side 평가로 보고되어 있어 숫자를 서로 바꿔 읽을 수 없습니다. NLHF 저자들은 자신들의 실험이 방법 간 우위를 보이려는 것이 아니라 구현 가능성을 시연한 것이라고 밝혔습니다 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)). Online IPO의 결과도 요약 과제 하나에 한정됩니다 ([Calandriello et al., 2024](https://arxiv.org/abs/2403.08635)). GRPO는 정답 검증이 가능한 수학 과제에서 평가되었고, 이 설정에서 RL은 Maj@K를 높였지만 Pass@K는 높이지 못했습니다 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)).

이 글의 관점에서 정리하면 선택 기준은 대략 이렇습니다. 쌍 데이터가 이미 있고 계산 자원이 빠듯하다면 참조 모델이 없는 SimPO 계열이 가장 싼 출발점입니다. 판정자를 반복해서 호출할 여력이 있다면 반복 DPO나 SPPO 같은 온라인 방식이 분포 이동에 더 강합니다. 정답을 자동으로 검증할 수 있는 과제라면 GRPO가 자연스러운 선택입니다. 표의 숫자는 이 세 경우를 서로 다른 잣대로 잰 결과라서, 한 방법이 다른 방법을 대체한다는 근거로 읽으면 안 됩니다.

| 논문 | 계보 | 주요 결과 | 한계 |
| --- | --- | --- | --- |
| Deep RL from Preferences | RLHF 기초 | MuJoCo 합성 라벨 1400개 진짜 보상 RL 상회 | Qbert 사람 피드백 실패 |
| LM Fine-Tuning from Preferences | RLHF 기초 | 60k fine-tune vs 참조 요약 TL;DR 승률 96%; 실제 인간 피드백 vs 감성 분류기 보상 선호 77% | 라벨러 휴리스틱 악용(추출식 복사) |
| Summarize from HF | RLHF 기초 | 1.3B RLHF TL;DR 참조 대비 선호 61%; 6.7B RLHF CNN/DM 전이 품질(지도 미세조정에 근접) | 6.7B RL 미세조정 비용 약 320 GPU-일 |
| InstructGPT | RLHF 기초 | 175B InstructGPT 대 GPT-3 선호율 85±3%; RM 홀드아웃 라벨러 정확도 69.6% | 소수 라벨러 편향 |
| Anthropic HH-RLHF | RLHF 기초 | 52B 반복 온라인 vs 기본 데이터 통제 실험 (PM 비교 약 44k) | 온라인 PM 정확도 74/70/67% |
| RM Overoptimization | RLHF 기초 | RM 데이터 임계치 약 2,000 비교 | 합성 gold RM 설정의 일반화 한계 |
| Constitutional AI | AI 피드백 | HHH 이진 비교 90% 초과 정확도 | RL-CAI 과학습 시 굿하트 현상 |
| RRHF | 참조 모델 없는 손실 | Alpaca-RRHF_SP HH 평균 보상 -0.96 | 다중 응답 필요와 과최적화 |
| DPO | 직접 선호 최적화 | GPT-J TL;DR GPT-4 승률 DPO 61% vs PPO 57%; TL;DR 인간 선호 DPO 대 PPO 58% | GPT-4 승률의 프롬프트 민감성 |
| RLAIF | AI 피드백 | RLAIF TL;DR 요약 SFT 대비 승률 71% (RLHF 73%); RLAIF 무해 대화 무해율 88% | 인간+AI 피드백 결합 이득 없음 |
| IPO | 직접 선호 최적화 | IPO 최적 정책 π*(y1)=σ(0.5/τ) | 장난감 실험 한정 |
| NLHF | 온라인·게임 이론 | T5X-XL 선호 모델 정확도 0.78 vs 보상 모델 0.76; Nash-MD-PG β 0.125–0.375 요약 쌍대 평가 최고 | 소규모 요약 실험, 우위 미입증 |
| KTO | 직접 선호 최적화 | Zephyr-β-SFT GSM8K KTO vs DPO +13.5점; Mistral-7B OpenAssistant KTO(입력당 하나) 승률 0.631 | 깨끗한 선호 데이터에서 KTO 과소적합 위험 |
| GRPO | 크리틱 없는 RL | DeepSeekMath-RL 7B GSM8K 88.2% / MATH 51.7% | RL의 Maj@K 향상, Pass@K 불변 |
| ORPO | 참조 모델 없는 손실 | Mistral-ORPO-β AlpacaEval 2.0 12.20%; Phi-2 ORPO AlpacaEval 2.0 6.35% | 비교 범위와 7B 초과 규모 미검증 |
| Online IPO | 온라인·게임 이론 | 온라인 IPO/IPO-MD 요약 side-by-side 승률 우위 | 요약 과제 한정 평가 |
| SPPO | 온라인·게임 이론 | Llama-3-8B-SPPO 총 향상 15.85%; Mistral-7B-SPPO Iter3 Arena-Hard 23.3 | 단일 데이터셋·제한된 벤치마크 |
| RLHF Workflow | 온라인·게임 이론 | Iterative DPO 8B AlpacaEval-2 LC 31.3 (vanilla DPO 22.5) | 반복 RLHF의 길이 편향 |
| SimPO | 참조 모델 없는 손실 | Llama-3-Instruct SimPO AlpacaEval 2 LC 44.7%; SimPO 실행시간 약 20% 감소, 메모리 약 10% 감소 | 수동 조정이 필요한 마진 하이퍼파라미터 |
| DAA Overoptimization | 직접 선호 최적화 | Pythia 6.9B 최고 성능 시점 데이터 25% | 해결책을 제시하지 않는 진단 연구 |

## 단순한 손실, 온라인 데이터, 검증 가능한 보상으로 모이는 흐름과 남은 숙제

여섯 계보를 나란히 놓으면, RL을 걷어 내는 데서 출발한 분야가 온라인 샘플링과 가벼운 RL로 되돌아오는 흐름이 보입니다. 그러나 길이 편향, 추론 성능 저하, 과최적화, RL 효과의 범위라는 네 가지 문제는 어느 계보도 아직 풀지 못했습니다.

### RL을 걷어 낸 뒤 다시 돌아온 현재 정책 샘플

DPO는 보상 모델과 RL 루프를 정책 손실 하나로 접었고, SimPO와 ORPO는 참조 모델의 순전파까지 덜어 냈습니다. 손실을 단순하게 만드는 일은 여기서 거의 끝에 이르렀습니다. 그 뒤의 개선은 손실이 아니라 데이터 쪽에서 나왔습니다. 오프라인 방법인 IPO를 현재 정책 샘플로 학습하도록 바꾼 Online IPO가 대표적입니다. 이렇게 바꾼 목적함수의 최소해는 정규화된 선호 게임의 내시 균형과 같습니다 ([Calandriello et al., 2024](https://arxiv.org/abs/2403.08635)). GRPO Dynamics는 정답 여부를 검증할 수 있는 이진 보상 아래에서 GRPO를 분석했고, 이때 그룹 평균·분산 보정이 이전 정책의 샘플을 대조 샘플로 쓰는 대조 손실이 된다는 결과를 보였습니다 ([Mroueh, 2025](https://arxiv.org/abs/2503.06639)). Atari 보상 학습에서 나온 교훈, 즉 피드백이 RL과 섞여 진행되어야 한다는 결론이 형태만 바뀌어 다시 나타난 셈입니다 ([Christiano et al., 2017](https://arxiv.org/abs/1706.03741)).

판정자를 누구로 두느냐는 질문은 아직 정리되지 않았습니다. RLHF Workflow는 온라인 사람 피드백을 구하기 어려워 오픈 데이터로 학습한 프록시 선호 모델에 의존합니다 ([Dong et al., 2024](https://arxiv.org/abs/2405.07863)). RLAIF에서는 사람 피드백과 AI 피드백을 결합해도 사람 피드백만 쓸 때보다 나아지지 않았습니다 ([Lee et al., 2023](https://arxiv.org/abs/2309.00267)). AI 라벨은 사람 라벨을 대신할 수는 있어도, 사람 라벨에 더해 이득을 쌓는 수단으로 쓰일 수 있는지는 확인되지 않았습니다.

### 어느 계보도 풀지 못한 네 가지 숙제

첫째 숙제는 길이 편향입니다. RLHF Workflow의 저자들은 반복 RLHF 초기 실험에서 상당한 길이 편향을 관찰했습니다 ([Dong et al., 2024](https://arxiv.org/abs/2405.07863)). 온라인 데이터든 길이 정규화 보상이든 길이라는 단순한 특징을 따라가는 경향을 완전히 끊어 내지는 못했습니다.

둘째 숙제는 추론 성능 저하입니다. SimPO 논문은 선호 최적화 뒤 GSM8k 같은 추론 과제의 성능이 떨어지며, 이런 과제에서는 SimPO가 DPO와 비슷하거나 더 나쁠 때도 있다고 보고했습니다 ([Meng et al., 2024](https://arxiv.org/abs/2405.14734)). 지시 따르기 승률에서 얻은 이득이 다른 능력의 손실로 상쇄될 수 있습니다.

셋째 숙제는 과최적화입니다. DAA 과최적화 연구는 이 문제를 지적하고 이론적 근거를 제시했지만 해결책은 내놓지 않았고, 인간 선호 모델이 존재한다는 가정도 그대로 남겨 두었습니다 ([Rafailov et al., 2024](https://arxiv.org/abs/2406.02900)).

넷째 숙제는 RL이 실제로 무엇을 바꾸느냐입니다. Maj@K는 K개 샘플의 다수결 답이 맞는 비율이고, Pass@K는 K개 샘플 중 하나라도 맞는 비율입니다. DeepSeekMath에서 RL은 Maj@K를 높였지만 Pass@K는 높이지 못했습니다. 저자들은 이를 근본 능력이 늘어서가 아니라 출력 분포가 견고해져 TopK 안의 정답이 위로 올라온 결과로 해석합니다 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)). GRPO Dynamics가 보인 성공 확률 증폭은 참조 모델의 성공 확률보다 높은 고정점으로 수렴한다는 결과인데 ([Mroueh, 2025](https://arxiv.org/abs/2503.06639)), 이 결과도 참조 모델이 이미 어느 정도 맞히는 문제에서 출발한다는 전제 위에 서 있습니다.

단순한 손실, 현재 정책 데이터, 검증 가능한 보상으로 분야가 모이고 있지만, 이 조합이 정렬하는 대상은 여전히 모델이 이미 낼 수 있는 출력의 분포입니다.

이 글의 관점에서 보면, 다음 진전은 손실 함수를 하나 더 고안하는 데서 나오기 어렵습니다. 길이와 과최적화를 같은 KL 예산 아래에서 함께 재는 평가, 그리고 정렬 뒤의 추론 능력을 Pass@K처럼 분포 다듬기와 구분해 측정하는 평가가 먼저 갖춰져야 어떤 계보가 실제로 앞서는지 판단할 수 있습니다.

## 보상 모델을 지우려던 시도가 다시 돌아온 지점

RLHF에서 GRPO까지 바뀐 것은 보상 모델과 RL을 다루는 방식이었고, KL 정규화와 현재 정책 데이터라는 두 요소는 형태만 바꾼 채 그대로 남았습니다.

바뀐 쪽은 분명합니다. 앞서 본 DPO와 SimPO에서 별도의 보상 모델은 정책 쪽으로 흡수되었습니다. NLHF는 단일 점수 대신 쌍대 선호 모델을 학습해 비추이적 선호까지 표현하려 했습니다 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)). PPO의 가치 모델도 앞서 본 GRPO의 그룹 baseline으로 대체되었습니다. 판정자도 사람에서 원칙을 따르는 AI 피드백 모델과 프록시 선호 모델로 넓어졌습니다 ([Dong et al., 2024](https://arxiv.org/abs/2405.07863)).

남은 쪽도 그만큼 분명합니다. 첫째 요소인 KL 정규화는 이름과 자리만 옮겼습니다. NLHF는 KL 항을 선호 자체에 넣어 정규화된 게임을 정의했고 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)), GRPO는 앞서 본 대로 KL을 손실 쪽으로 옮겼습니다 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)). 둘째 요소인 현재 정책 데이터는 앞서 본 Atari 실험의 교훈에서 출발합니다. 그 뒤 요약 과제에서는 RL 도중 현재 정책 샘플에 라벨을 계속 모아 보상 모델을 다시 학습했고, 이 온라인 수집은 요약에는 중요했지만 단순한 스타일 과제에는 필요하지 않았습니다 ([Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)). 이 흐름은 Online IPO, 반복 DPO, GRPO의 그룹 샘플로 계속 이어졌습니다. 앞서 본 Online IPO의 동치 결과도 이 해석과 맞습니다. IPO가 현재 정책에서 샘플하면서 KL 정규화를 유지하면 NLHF의 정규화된 게임과 같은 해에 이릅니다. 따라서 손실의 겉모양이 달라도 두 요소가 같으면 같은 해에 도달하며, 두 요소가 손실 형태보다 더 근본적인 자리에 있음을 이 결과가 보여 줍니다.

바뀐 것은 비교 판단을 정책으로 옮기는 경로였고, 무엇을 정렬하는가는 바뀌지 않았습니다. 이 글은 앞으로의 비교도 손실 이름보다 두 질문을 기준으로 삼는 편이 낫다고 봅니다. 하나는 KL로 묶인 정책이 어떤 분포의 데이터로 학습하는가입니다. 다른 하나는 그 결과를 같은 KL 예산 아래에서 측정했는가, 곧 참조 정책과의 KL 발산을 같은 수준으로 맞춘 상태에서 방법들을 비교했는가입니다.

- - -

### References
* Paul Christiano, Jan Leike, Tom B. Brown et al. (2017). [Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741). NeurIPS.
* Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu et al. (2019). [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593). arXiv.
* Nisan Stiennon, Long Ouyang, Jeff Wu et al. (2020). [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325). NeurIPS.
* Long Ouyang, Jeff Wu, Xu Jiang et al. (2022). [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155). NeurIPS.
* Yuntao Bai, Andy Jones, Kamal Ndousse et al. (2022). [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862). arXiv.
* Leo Gao, John Schulman, Jacob Hilton (2022). [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760). ICML.
* Yuntao Bai, Saurav Kadavath, Sandipan Kundu et al. (2022). [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073). arXiv.
* Zheng Yuan, Hongyi Yuan, Chuanqi Tan et al. (2023). [RRHF: Rank Responses to Align Language Models with Human Feedback without tears](https://arxiv.org/abs/2304.05302). NeurIPS.
* Rafael Rafailov, Archit Sharma, Eric Mitchell et al. (2023). [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290). NeurIPS.
* Hugo Touvron, Louis Martin, Kevin Stone et al. (2023). [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288). arXiv.
* Harrison Lee, Samrat Phatale, Hassan Mansoor et al. (2023). [RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/abs/2309.00267). ICML.
* Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot et al. (2023). [A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036). International Conference on Artificial Intelligence and Statistics.
* Rémi Munos, Michal Valko, Daniele Calandriello et al. (2023). [Nash Learning from Human Feedback](https://arxiv.org/abs/2312.00886). ICML.
* Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff et al. (2024). [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306). arXiv.
* Zhihong Shao, Peiyi Wang, Qihao Zhu et al. (2024). [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300). arXiv.
* Jiwoo Hong, Noah Lee, James Thorne (2024). [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691). EMNLP.
* Daniele Calandriello, Daniel Guo, Remi Munos et al. (2024). [Human Alignment of Large Language Models through Online Preference Optimisation](https://arxiv.org/abs/2403.08635). ICML.
* Yue Wu, Zhiqing Sun, Huizhuo Yuan et al. (2024). [Self-Play Preference Optimization for Language Model Alignment](https://arxiv.org/abs/2405.00675). ICLR.
* Hanze Dong, Wei Xiong, Bo Pang et al. (2024). [RLHF Workflow: From Reward Modeling to Online RLHF](https://arxiv.org/abs/2405.07863). TMLR.
* Yu Meng, Mengzhou Xia, Danqi Chen (2024). [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734). NeurIPS.
* Rafael Rafailov, Yaswanth Chittepu, Ryan Park et al. (2024). [Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms](https://arxiv.org/abs/2406.02900). NeurIPS.
* Youssef Mroueh (2025). [Reinforcement Learning with Verifiable Rewards: GRPO's Effective Loss, Dynamics, and Success Amplification](https://arxiv.org/abs/2503.06639). arXiv.

### 참고 자료
* [Illustrating Reinforcement Learning from Human Feedback (RLHF)](https://huggingface.co/blog/rlhf)
