---
layout: post
title: "From RLHF to DPO and GRPO: Lineages of LLM Preference Optimization"
date: 2026-10-14 01:47:37
description: "LLM 선호 최적화를 보상 모델과 PPO 기반 RLHF, AI 피드백, DPO 계열 직접 최적화, 참조 모델 없는 목적함수, 내시 균형 셀프 플레이, GRPO의 여섯 갈래로 나눕니다. 과최적화와 트레이드오프를 함께 계보별로 정리합니다."
categories: ["Paper Surveys", "Finetuning"]
tags: ["Reward-Overoptimization", "Constitutional-AI", "RLAIF", "SimPO", "Self-Play-Preference-Optimization", "Alignment"]
cover: /assets/images/finetuning.jpg
use_math: true
---
## 사람의 비교 판단을 학습 신호로 바꾸는 여섯 갈래의 계보

좋은 텍스트는 손실 함수로 적기 어렵습니다. 그래서 LLM 정렬은 "두 응답 중 어느 쪽이 나은가"라는 사람의 비교 판단을 학습 신호로 바꾸는 선호 최적화(preference optimization)에 기대 왔습니다. 다음 토큰 교차 엔트로피는 참조 텍스트를 흉내 내도록 학습시킬 뿐입니다. BLEU나 ROUGE도 참조와 겹치는 정도를 정해진 규칙으로 셉니다. 어느 쪽도 창의성, 사실성, 실행 가능성처럼 맥락마다 달라지는 기준을 담지 못합니다. 반면 사람은 절대 점수를 매기기보다 둘 중 하나를 고르는 판단을 훨씬 쉽게 내립니다.

예를 들어 가상의 Reddit 게시물 하나에 요약 A와 B가 있고, 라벨러가 A를 골랐다고 해 보겠습니다. 이 한 쌍, 곧 게시물과 "A가 B보다 낫다"는 판정이 이후 모든 방법이 공유하는 입력입니다. 방법마다 다른 것은 이 판정을 무엇으로 바꾸느냐입니다. 보상 모델(reward model)의 점수 차로 바꾸는 방법도 있고, 정책의 참조 모델 대비 로그확률 비로 바꾸는 방법도 있고, 다른 정책(또는 자기 자신의 이전 버전)을 이기는 확률로 바꾸는 방법도 있습니다.

출발점은 브래들리-테리 모델(Bradley-Terry model)로 학습한 보상 모델과 KL 페널티(KL penalty)를 단 PPO로 이루어진 다단계 파이프라인입니다. 이후 연구는 이 파이프라인의 무거운 구성요소를 하나씩 덜어내거나 바꾸는 방향으로 갈라졌습니다.

- **RLHF 기초**: 보상 모델, 참조 모델(reference model), PPO를 모두 갖춘 원형입니다.
- **AI 피드백**: 사람 라벨을 원칙 목록과 AI 판정으로 바꿉니다.
- **직접 선호 최적화**: 보상 모델과 RL 루프를 닫힌 형태의 손실로 대신합니다.
- **참조 모델 없는 단순화**: 참조 모델과 별도의 SFT 단계까지 덜어냅니다.
- **온라인·게임 이론**: 브래들리-테리 가정 대신 일반 선호의 내시 균형(Nash equilibrium)을 목표로 삼습니다.
- **크리틱 없는 RL**: 가치 모델을 없애고 검증 가능한 보상을 씁니다.

이 갈래들은 앞선 방법을 차례로 대체한 순서라기보다, 같은 문제의 서로 다른 구성요소를 동시에 공략한 흐름으로 읽는 편이 맞습니다.

| 연도 | 논문 | 계보 | 핵심 아이디어 |
| --- | --- | --- | --- |
| 2017 | Deep RL from Human Preferences ([Christiano et al., 2017](https://arxiv.org/abs/1706.03741)) | RLHF 기초 | 궤적 구간 쌍 비교로 학습한 브래들리-테리 보상 모델과 딥 RL |
| 2019 | Fine-Tuning LMs from Preferences ([Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)) | RLHF 기초 | KL 페널티를 단 PPO로 사전학습 언어 모델을 사람 선호에 미세조정 |
| 2020 | Learning to Summarize ([Stiennon et al., 2020](https://arxiv.org/abs/2009.01325)) | RLHF 기초 | SFT, 보상 모델, PPO로 이어지는 대규모 요약 RLHF 레시피 |
| 2022 | InstructGPT ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)) | RLHF 기초 | 지시 수행 어시스턴트를 위한 3단계 RLHF 파이프라인과 사전학습 혼합 |
| 2022 | Anthropic HH RLHF ([Bai et al., 2022](https://arxiv.org/abs/2204.05862)) | RLHF 기초 | 반복 온라인 RLHF로 만든 도움되고 무해한 어시스턴트 |
| 2022 | RM Overoptimization Laws ([Gao et al., 2022](https://arxiv.org/abs/2210.10760)) | RLHF 기초 | gold 보상 모델로 측정한 보상 과최적화의 스케일링 법칙 |
| 2022 | Constitutional AI ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)) | AI 피드백 | 원칙 기반 자기 비평과 AI 선호로 얻는 무해성 |
| 2023 | RRHF ([Yuan et al., 2023](https://arxiv.org/abs/2304.05302)) | 참조 모델 없는 단순화 | 길이 정규화 로그확률에 대한 응답 랭킹 손실 |
| 2023 | DPO ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)) | 직접 선호 최적화 | KL 정규화 목적의 닫힌 해로 얻은 보상 모델 없는 선호 손실 |
| 2023 | Llama 2 ([Touvron et al., 2023](https://arxiv.org/abs/2307.09288)) | RLHF 기초 | 안전·도움 보상 모델 결합과 거부 샘플링을 쓰는 오픈 채팅 모델 RLHF |
| 2023 | RLAIF ([Lee et al., 2023](https://arxiv.org/abs/2309.00267)) | AI 피드백 | 기성 LLM 선호 라벨로 보상 모델을 학습하는 RLAIF |
| 2023 | IPO ([Azar et al., 2023](https://arxiv.org/abs/2310.12036)) | 직접 선호 최적화 | 선호 확률의 일반 함수 Ψ를 최적화하는 ΨPO와 제곱 손실 IPO |
| 2023 | NLHF ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)) | 온라인·게임 이론 | 일반 선호 모델의 내시 균형을 미러 디센트로 찾는 정렬 |
| 2024 | KTO ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)) | 직접 선호 최적화 | 전망 이론 효용으로 이진 바람직함 신호를 학습하는 KTO |
| 2024 | GRPO ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)) | 크리틱 없는 RL | 가치 모델 없이 그룹 상대 어드밴티지를 쓰는 GRPO |
| 2024 | ORPO ([Hong et al., 2024](https://arxiv.org/abs/2403.07691)) | 참조 모델 없는 단순화 | SFT 손실에 오즈비 페널티를 더한 단일 단계 정렬 |
| 2024 | Online IPO ([Calandriello et al., 2024](https://arxiv.org/abs/2403.08635)) | 온라인·게임 이론 | 현재 정책 샘플로 학습하는 온라인 IPO와 IPO-MD |
| 2024 | SPPO ([Wu et al., 2024](https://arxiv.org/abs/2405.00675)) | 온라인·게임 이론 | 승률 기반 승법 가중치 갱신을 근사하는 셀프 플레이 선호 최적화 |
| 2024 | RLHF Workflow ([Dong et al., 2024](https://arxiv.org/abs/2405.07863)) | 온라인·게임 이론 | 보상 모델 순위로 쌍을 만드는 온라인 반복 DPO 워크플로 |
| 2024 | SimPO ([Meng et al., 2024](https://arxiv.org/abs/2405.14734)) | 참조 모델 없는 단순화 | 길이 정규화 평균 로그우도 보상과 목표 마진의 참조 모델 없는 손실 |
| 2024 | DAA Overoptimization ([Rafailov et al., 2024](https://arxiv.org/abs/2406.02900)) | 직접 선호 최적화 | 직접 정렬 알고리즘에서의 보상 과최적화 스케일링 |
| 2025 | GRPO Dynamics ([Mroueh, 2025](https://arxiv.org/abs/2503.06639)) | 크리틱 없는 RL | 검증 가능한 보상 아래 GRPO의 실효 손실과 성공 증폭 |

## 보상 모델과 KL 페널티로 쌓아 올린 PPO 파이프라인과 그 굿하트 한계

표준 RLHF 파이프라인은 세 단계를 거쳐 만들어졌습니다. Atari와 로봇 제어에서 궤적 비교로 보상을 배우던 방법이 먼저 있었고, 이것이 KL 페널티를 단 PPO를 통해 언어 모델로 옮겨졌으며, InstructGPT에서 SFT, 보상 모델, PPO의 3단계로 굳어졌습니다. 같은 계보는 프록시 보상을 너무 밀어붙이면 진짜 선호가 떨어지는 한계도 직접 측정했습니다.

### 궤적 구간 비교로 학습한 보상 예측기

Christiano et al.이 다룬 문제는 보상 함수 없이, 비전문가가 두 궤적 구간 중 어느 쪽이 나은지 고른 판단만으로 복잡한 RL 과제를 풀 수 있느냐였습니다. 답은 두 부분으로 이루어졌습니다. 선호 데이터로 보상 모델을 지도학습하고, 그 보상을 딥 RL로 최적화하는 것입니다. 사람이 피드백을 준 비율은 에이전트와 환경 사이 상호작용의 1% 미만이었습니다 ([Christiano et al., 2017](https://arxiv.org/abs/1706.03741)).

절대 점수 대신 비교를 받은 데는 실험 근거가 있습니다. 연속 제어 과제에서 비교를 예측하는 쪽이 점수를 회귀하는 쪽보다 훨씬 잘 작동했고, 사람도 절대 점수보다 비교를 더 일관되게 주었습니다. 같은 실험에서 긴 클립을 비교하게 하면 클립 하나당 얻는 정보가 더 많았습니다 ([Christiano et al., 2017](https://arxiv.org/abs/1706.03741)).

비교를 확률로 바꾸는 장치가 브래들리-테리 모델입니다. 사람이 구간을 선호할 확률은 그 구간에서 추정 보상을 합한 값의 지수에 비례한다고 놓습니다 ([Christiano et al., 2017](https://arxiv.org/abs/1706.03741)).

$$\hat{P}\left[\sigma^{1}\succ\sigma^{2}\right]=\frac{\exp{\sum\hat{r}\left(o^{1}_{t},a^{1}_{t}\right)}}{\exp{\sum\hat{r}\left(o^{1}_{t},a^{1}_{t}\right)}+\exp{\sum\hat{r}\left(o^{2}_{t},a^{2}_{t}\right)}}$$

$$\sigma^{1}, \sigma^{2}$$는 비교 대상인 두 궤적 구간이고, $$o_t, a_t$$는 시점 $$t$$의 관측과 행동, $$\hat{r}$$는 학습 중인 보상 추정기입니다. 보상 추정기는 이 예측 확률과 사람 라벨 사이의 교차 엔트로피를 최소화하도록 학습합니다 ([Christiano et al., 2017](https://arxiv.org/abs/1706.03741)).

$$\mathrm{loss}\left(\hat{r}\right)=-\sum_{\left(\sigma^{1},\sigma^{2},\mu\right)\in\mathcal{D}}\mu\left(1\right)\log\hat{P}\left[\sigma^{1}\succ\sigma^{2}\right]+\mu\left(2\right)\log\hat{P}\left[\sigma^{2}\succ\sigma^{1}\right]$$

$$\mathcal{D}$$는 수집한 비교 데이터이고, $$\mu$$는 라벨러가 두 구간에 준 선호 분포입니다. 한쪽을 확실히 고르면 그쪽에 질량이 모두 실리고, 비슷하다고 답하면 질량이 나뉩니다. 보상 학습은 결국 "어느 쪽이 이길까"를 맞히는 이진 분류이고, 보상 값 자체는 그 분류기의 로짓으로 나타납니다.

시스템은 세 프로세스가 비동기로 돌아가는 구조였습니다. 첫째 프로세스는 정책이 예측 보상을 최대화하도록 RL로 갱신합니다. 둘째는 그 궤적에서 구간 쌍을 골라 사람에게 비교를 요청하고, 셋째는 모인 비교로 보상 추정기를 다시 적합합니다. 궤적은 첫째에서 둘째로, 사람의 비교는 둘째에서 셋째로, 보상 추정기의 파라미터는 셋째에서 다시 첫째로 흐릅니다 ([Christiano et al., 2017](https://arxiv.org/abs/1706.03741)).

![궤적 구간 비교로 보상 예측기를 비동기 학습하고 에이전트가 예측 보상을 최대화하는 구조도](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-1.png)

*출처: [Christiano et al., 2017](https://arxiv.org/abs/1706.03741)*

도식에서 사람은 환경과 직접 상호작용하지 않습니다. 사람은 구간 비교에만 관여하고, 에이전트가 최대화하는 대상은 보상 예측기의 출력입니다 ([Christiano et al., 2017](https://arxiv.org/abs/1706.03741)). 이 분리 구조가 이후 언어 모델 RLHF에서도 그대로 이어집니다.

실제 알고리즘에는 몇 가지 장치가 더 붙었습니다. 보상 예측기를 앙상블로 두고, 데이터의 $$1/e$$를 검증용으로 떼어 l2 정규화 강도를 맞췄습니다. 앞서 본 브래들리-테리 확률에 소프트맥스를 그대로 쓰지 않고, 사람이 10% 확률로 무작위 응답한다고 가정한 것도 그런 장치입니다. 어떤 쌍을 물을지는 많은 구간 쌍을 샘플링한 뒤 앙상블 구성원 사이 예측 분산이 가장 큰 쌍을 고르는 방식으로 정했습니다. 저자들 스스로 이 분산 기반 선택은 거친 근사라고 밝혔고, 일부 과제에서는 오히려 성능을 떨어뜨렸다고 보고했습니다 ([Christiano et al., 2017](https://arxiv.org/abs/1706.03741)).

![MuJoCo 과제에서 구성요소를 하나씩 제거했을 때의 성능 비교](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-2.png)

*출처: [Christiano et al., 2017](https://arxiv.org/abs/1706.03741)*

절제 실험은 앙상블, 온라인 질의, 정규화, 구간 길이 같은 구성요소를 하나씩 빼고 MuJoCo 과제의 성능을 비교합니다. 특히 "온라인 질의 제거"가 눈여겨볼 축입니다. 정책이 바뀌는 동안 계속 새 구간을 물어야 하는지가 여기서 갈리고, 오프라인과 온라인 데이터 문제는 이 계보 내내 다시 등장합니다.

![MuJoCo 로봇 과제에서 사람 피드백, 합성 피드백, 참 보상 RL의 참 보상 기준 학습 곡선](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-3.png)

*출처: [Christiano et al., 2017](https://arxiv.org/abs/1706.03741)*

MuJoCo 결과는 실제 사람 피드백, 오라클이 준 합성 피드백(라벨 350, 700, 1400개), 참 보상으로 학습한 RL을 모두 과제의 참 보상 기준으로 비교합니다. Atari 결과도 같은 세 조건을 참 보상으로 견줍니다 ([Christiano et al., 2017](https://arxiv.org/abs/1706.03741)).

![Atari 게임에서 사람 피드백, 합성 피드백, 참 보상 RL의 참 보상 기준 학습 곡선](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-4.png)

*출처: [Christiano et al., 2017](https://arxiv.org/abs/1706.03741)*

저자들은 보상 모델을 지도학습으로 따로 학습하면 상호작용 복잡도를 약 3자릿수 줄일 수 있다고 보고했습니다. 이를 근거로, 이 기법이 최신 RL 시스템에 경제적으로 확장될 수 있다는 첫 증거라고 주장했습니다 ([Christiano et al., 2017](https://arxiv.org/abs/1706.03741)).

### KL 페널티를 덧붙인 수정 보상과 언어 모델로의 이식

Ziegler et al.은 Christiano et al.의 비교 기반 보상 학습을 사전학습 언어 모델에 그대로 가져왔습니다. 라벨러는 입력 $$x$$에 대한 여러 응답 후보 $$y_i$$ 가운데 가장 좋은 것을 고릅니다. 언어 모델에서 새로 필요해진 것은 정책이 출발점에서 너무 멀어지지 않게 묶는 장치였고, 그 장치가 보상에 직접 붙은 KL 페널티입니다 ([Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)).

$$R(x,y)=r(x,y)-\beta\log\frac{\pi(y\mid x)}{\rho(y|x)}.$$

$$r$$은 학습된 보상 모델, $$\pi$$는 학습 중인 정책, $$\rho$$는 초기 언어 모델, $$\beta$$는 페널티 강도입니다. PPO는 $$r$$이 아니라 이 수정 보상 $$R$$을 최대화합니다 ([Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)). 예를 들어 가상의 요약 $$y$$에 대해 정책 확률이 0.02, 초기 모델 확률이 0.01이고 $$\beta=0.1$$이면, 로그비는 $$\ln 2 \approx 0.693$$이므로 보상에서 약 0.069가 깎입니다. 초기 모델이 거의 내지 않을 출력에 정책이 확률을 몰아줄수록 이 차감이 커집니다.

저자들은 이 항의 목적을 셋으로 설명했습니다. 엔트로피 보너스 역할을 하고, 정책이 $$r$$이 유효한 범위 밖으로 나가지 않게 막으며, 스타일 이어쓰기 과제에서는 과제 정의의 일부를 이룹니다 ([Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)). 두 번째 목적이 이 계보의 핵심 긴장입니다. 보상 모델은 자기가 본 분포 안에서만 믿을 만하다는 가정이 처음부터 설계에 들어가 있었습니다.

절차는 네 단계였습니다. 먼저 $$\rho$$에서 샘플을 모아 사람에게 라벨을 받고, $$\rho$$로 초기화한 보상 모델을 학습합니다. 이어서 앞의 수정 보상 $$R$$로 PPO를 돌려 정책을 학습하며, 온라인 설정이라면 보상 모델을 주기적으로 다시 학습합니다 ([Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)).

![보상 모델 학습과 정책 학습 과정의 도식](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-5.png)

*출처: [Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)*

이 도식은 보상 모델 학습과 정책 학습을 별개의 두 루프로 그립니다. 온라인 설정에서는 두 루프가 번갈아 진행됩니다 ([Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)).

![정책이 초기 정책에서 멀어질수록 보상은 오르지만 샘플이 덜 자연스러워지는 KL-보상 프런티어](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-6.png)

*출처: [Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)*

$$\operatorname{KL}(\pi,\rho)$$를 더 허용하면 보상은 높아지지만 샘플은 덜 자연스러워집니다 ([Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)). KL은 공짜 정규화가 아니라, 보상과 자연스러움 사이의 다이얼입니다.

요약 실험에서는 경고 신호가 하나 나왔습니다. 저자들은 이 결과를 성취보다는 우려로 받아들였고, 모델이 라벨러의 판단 휴리스틱을 악용했을 가능성을 지적했습니다 ([Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)).

실패 사례도 기록되어 있습니다. 완전 온라인 데이터 수집은 소프트웨어와 ML 양쪽에서 복잡도가 크고 품질 관리도 어려워서, 저자들은 오프라인과 온라인의 절충안으로 배치 수집을 제안했습니다. 보상 모델과 정책이 파라미터를 공유하게 한 시도는 과적합으로 실패했습니다. 보상 모델의 샘플은 최대 60k였는데 정책의 에피소드는 2M이어서, 데이터 불균형이 원인이었습니다. 보상 부호가 뒤집히는 버그는 KL 항의 부호까지 함께 뒤집었습니다. 그 결과 모델은 자연스러운 언어로 정규화된 채 부정 감성을 최적화했고, 일관되게 최악의 출력(노골적인 성적 내용)을 학습했습니다 ([Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)). 이 사건은 보상 신호가 무엇이든 PPO가 그것을 충실히 최적화한다는 사실을 극단적인 형태로 보여 줍니다.

### TL;DR 요약에서 정리된 쌍별 손실과 처음 관측된 과최적화 곡선

Stiennon et al.은 Reddit TL;DR 요약을 무대로 보상 모델의 손실을 응답 단위 점수의 차로 정리했습니다 ([Stiennon et al., 2020](https://arxiv.org/abs/2009.01325)).

$$\text{loss}(r_{\theta})=-E_{(x,y_{0},y_{1},i)\sim D}[\log(\sigma(r_{\theta}(x,y_{i})-r_{\theta}(x,y_{1-i})))]$$

$$x$$는 게시물, $$y_0, y_1$$은 두 요약, $$i$$는 사람이 고른 요약의 인덱스, $$\sigma$$는 시그모이드입니다. 앞서 본 Christiano et al.의 브래들리-테리 확률을 두 후보로 좁히고, 구간 보상의 합을 요약 전체에 대한 스칼라 점수 하나로 바꾸면 그대로 이 형태가 됩니다. 소개 부분에서 세운 가상의 게시물에 대입하면, 보상 모델이 고른 요약 A에 1.5, B에 0.5를 준 경우 $$\sigma(1.0)\approx 0.731$$이 되고 손실은 $$-\ln 0.731 \approx 0.313$$입니다. 같은 쌍에서 A와 B 점수가 같으면 손실은 $$\ln 2 \approx 0.693$$이 되고, B를 더 높게 매기면 손실이 그보다 커집니다. 이 손실은 점수의 차만 보기 때문에, 보상의 절대 수준은 어떤 상수를 더해도 그대로 남습니다.

정책 쪽에서는 앞서 본 KL 수정 보상과 같은 구조를 쓰되, 기준 분포를 초기 언어 모델이 아니라 지도 미세조정(supervised fine-tuning)을 거친 SFT 정책으로 두었습니다. 저자들은 KL 항의 역할을 두 가지로 설명했습니다. 엔트로피 보너스로서 탐색을 유도하고 단일 모드로 붕괴하는 것을 막는 역할, 그리고 보상 모델이 학습 때 본 출력과 크게 다른 출력을 정책이 배우지 않게 하는 역할입니다. 가치 함수는 정책과 분리된 파라미터로 두고 보상 모델에서 초기화했습니다 ([Stiennon et al., 2020](https://arxiv.org/abs/2009.01325)).

절차는 단계별로 진행됩니다. 여러 정책에서 요약을 샘플링해 사람에게 비교를 받고, 그 비교로 보상 모델을 학습한 다음, SFT로 초기화한 정책을 보상 모델의 로짓을 보상으로 삼아 PPO로 최적화합니다 ([Stiennon et al., 2020](https://arxiv.org/abs/2009.01325)).

![사람 피드백 수집, 보상 모델 학습, 정책 학습으로 이어지는 요약 RLHF 절차도](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-7.png)

*출처: [Stiennon et al., 2020](https://arxiv.org/abs/2009.01325)*

절차도를 따라가 보면 사람 피드백, 보상 모델 학습, 정책 학습이 하나의 흐름으로 이어집니다 ([Stiennon et al., 2020](https://arxiv.org/abs/2009.01325)). 이후 InstructGPT가 앞에 SFT 단계를 명시적으로 세우기 전까지, 언어 모델 RLHF는 대체로 이 모양이었습니다.

결과 가운데 두 가지가 오래 인용됩니다. 훨씬 짧은 요약을 내면서도 CNN/DM 참조 요약으로 미세조정한 6.7B 모델에 거의 근접했습니다. 둘째, 보상 모델의 학습 데이터를 두 배로 늘리면 검증 정확도가 약 1.1% 오르고, 모델 크기를 두 배로 늘리면 약 1.8% 올랐습니다 ([Stiennon et al., 2020](https://arxiv.org/abs/2009.01325)). 보상 모델 품질이 데이터와 모델 크기 양쪽에 반응한다는 관측입니다.

![보상 모델 최적화 정도에 따라 처음엔 오르다가 과적합으로 떨어지는 선호 점수](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-8.png)

*출처: [Stiennon et al., 2020](https://arxiv.org/abs/2009.01325)*

가로축을 따라 보상 모델에 대한 최적화를 늘려 가면, 요약은 처음에는 좋아지다가 결국 과적합해 더 나빠집니다. 저자들의 서술은 더 직접적입니다. 최적화를 계속하면 진짜 선호가 보상 모델의 예측에서 떨어져 나가고, 결국 보상 모델이 사람 선호와 반상관하게 됩니다 ([Stiennon et al., 2020](https://arxiv.org/abs/2009.01325)).

### InstructGPT에서 굳어진 SFT, 보상 모델, PPO의 3단계

InstructGPT는 이 흐름을 세 단계 레시피로 고정했습니다. 시연 데이터로 SFT를 하고, 더 넓은 API 프롬프트 집합에서 모델 출력 사이의 비교 데이터를 모아 라벨러가 어떤 출력을 선호할지 예측하는 보상 모델(RM)을 학습한 뒤, 그 RM을 보상으로 PPO 최적화를 합니다 ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)).

![SFT, 보상 모델 학습, PPO 강화학습의 3단계 방법 도식](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-9.png)

*출처: [Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)*

도식에서 세 단계는 (1) SFT, (2) RM 학습, (3) 그 RM에 대한 PPO 강화학습으로 이어집니다 ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)). 이후 대부분의 후속 방법이 "무엇을 덜어냈는가"를 말할 때 기준으로 삼는 그림이 이것입니다.

보상 모델 손실은 앞서 본 쌍별 로지스틱 손실과 형태가 같지만, 데이터를 다루는 방식이 다릅니다 ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)).

$$\operatorname{loss}\left(\theta\right)=-\frac{1}{% raw %}{{K\choose 2}}{% endraw %}E_{\left(x,y_{w},y_{l}\right)\sim D}\left[\log\left(\sigma\left(r_{\theta}\left(x,y_{w}\right)-r_{\theta}\left(x,y_{l}\right)\right)\right)\right]$$

$$r_\theta(x,y)$$는 프롬프트 $$x$$와 응답 $$y$$에 대한 보상 모델의 스칼라 출력이고, $$y_w$$는 쌍에서 선호된 응답, $$y_l$$은 덜 선호된 응답입니다. $$K$$는 라벨러가 한 프롬프트에서 순위를 매긴 응답 수이고, 그 순위에서 $${K\choose 2}$$개의 쌍 비교가 나옵니다. 저자들은 이 쌍들을 따로 섞지 않고 한 프롬프트의 모든 비교를 하나의 배치 원소로 묶어 학습했습니다. 그 이유로 과적합 방지와 계산 효율을 들었습니다 ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)). 예를 들어 가상의 프롬프트 하나에 응답 4개를 순위 매기면 쌍 비교가 6개 나오고, 정규화 계수 $$1/6$$이 그 6개 손실의 평균을 냅니다.

RL 환경은 밴딧 설정입니다. 프롬프트 하나에 응답 하나를 내면 RM이 보상을 주고 에피소드가 끝납니다. 여기에 SFT 모델 대비 토큰별 KL 페널티를 더해 RM 과최적화를 완화했고, 가치 함수는 RM에서 초기화했습니다 ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)).

![API 프롬프트 분포에서 각 모델이 175B SFT 모델보다 선호된 비율을 보여 주는 사람 평가](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-10.png)

*출처: [Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)*

PPO와 PPO-ptx는 GPT-3 기준선을 크게 앞섰고, 1.3B PPO-ptx의 출력이 175B GPT-3의 출력보다 더 선호되었습니다 ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)). 다만 이 결과는 OpenAI API 프롬프트 분포에서 라벨러가 내린 선호이고, 범용 능력 비교는 아닙니다.

다른 지표들은 이득이 더 작거나 비용이 따른다는 쪽을 가리킵니다. TruthfulQA 사람 평가에서 PPO 모델은 GPT-3보다 진실하고 유익한 출력을 작지만 유의미하게 더 많이 냈고, 폐쇄 도메인 과제의 환각도 줄었습니다. 사전학습 업데이트를 섞은 PPO-ptx가 이 하락을 완화했고, KL 계수를 키우는 단순한 해법보다 결과가 나았습니다 ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)).

![사전학습 손실 계수에 따른 공개 NLP 데이터셋 성능](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-11.png)

*출처: [Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)*

사전학습 손실 계수를 바꿔 가며 보면, DROP과 SQuAD 성능을 크게 회복시키는 계수 값이 있습니다 ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)). 이와 대조하려고 저자들은 같은 공개 데이터셋 평가를 KL 보상 계수의 함수로도 그렸습니다.

![KL 보상 계수에 따른 공개 NLP 데이터셋 성능](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-12.png)

*출처: [Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)*

KL 계수를 키워도 DROP과 SQuAD의 성능 하락은 완전히 해소되지 않습니다 ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)). KL 페널티는 SFT 분포 근처에 머물게 할 뿐이고, 사전학습 분포에서 얻은 능력을 지켜 주지는 못한다는 해석과 맞는 결과입니다.

### 반복 온라인 수집과 Llama 2의 운영형 변형

가장 좋은 RLHF 정책을 학습하고, 그 정책으로 크라우드워커의 비교 데이터를 새로 모은 다음, 기존 데이터와 섞어 선호 모델(PM)과 정책을 처음부터 다시 학습하는 과정을 되풀이했습니다. 목표는 PM이 높은 점수 구간에서 보정이 잘 되도록 만드는 것이었습니다 ([Bai et al., 2022](https://arxiv.org/abs/2204.05862)). 정책이 좋아질수록 그 정책이 내는 응답은 기존 데이터에 드물어지므로, 그 구간의 비교를 새로 채워 넣는 셈입니다.

![문맥 증류 모델, 정적 데이터 RLHF 모델, 반복 온라인 RLHF 모델에 대한 크라우드워커 선호 Elo 비교](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-13.png)

*출처: [Bai et al., 2022](https://arxiv.org/abs/2204.05862)*

Elo 비교는 문맥 증류(distillation) 모델, 정적 데이터셋으로 학습한 RLHF 모델, 반복 온라인 방식으로 학습한 RLHF 모델에 대한 크라우드워커 선호를 한데 요약합니다. 온라인의 이득이 단지 데이터가 늘어서인지 확인하려고, 저자들은 데이터 크기를 같게 맞춘 실험도 했습니다. base, 거부 샘플링, 온라인 데이터를 섞어 학습한 정책이 base 데이터만으로 학습한 정책보다 크라우드워커에게 분명히 더 선호되었습니다. 같은 연구는 RLHF 학습 중 PM 점수가 정책과 초기 정책 사이 KL의 제곱근에 대략 선형으로 늘어난다는 관계도 관측했습니다. 저자들은 이 관계를 RL이 초기 정책에서 크게 벗어나지 않는 작은 변화 영역에 머무르기 때문으로 해석했습니다 ([Bai et al., 2022](https://arxiv.org/abs/2204.05862)).

Llama 2는 같은 뼈대 위에 운영상의 변형을 여럿 얹었습니다. 보상 모델은 InstructGPT와 같은 이진 순위 손실로 학습하되, 라벨러가 매긴 선호 등급에 따라 이산 마진을 더했습니다 ([Touvron et al., 2023](https://arxiv.org/abs/2307.09288)).

$$\mathcal{L}_{\text{ranking}}=-\text{log}(\sigma(r_{\theta}(x,y_{c})-r_{\theta}(x,y_{r})-m(r)))$$

$$y_c$$는 선택된 응답, $$y_r$$은 거부된 응답, $$m(r)$$은 선호 등급 $$r$$에 따라 정해지는 이산 마진입니다. 두 응답의 차이가 크다고 라벨된 쌍일수록 점수 차를 더 벌리도록 학습되며, 이 마진을 넣자 유용성(helpfulness) 보상 모델의 정확도가 개선되었습니다 ([Touvron et al., 2023](https://arxiv.org/abs/2307.09288)). 예를 들어 가상의 TL;DR 쌍에서 A가 "훨씬 낫다"로 라벨되어 마진이 1이라면, 점수 차가 1.0이어도 시그모이드 안의 값은 0이 되어 손실은 앞서 계산한 0.313이 아니라 $$\ln 2$$로 남습니다.

PPO 보상도 단일 RM이 아닙니다 ([Touvron et al., 2023](https://arxiv.org/abs/2307.09288)).

$$R(g\mid p)=\tilde{R}_{c}(g\mid p)-\beta D_{KL}(\pi_{\theta}(g\mid p)\parallel\pi_{0}(g\mid p))$$

$$p$$는 프롬프트, $$g$$는 생성 결과, $$\pi_0$$은 원래 정책입니다. $$\tilde{R}_c$$는 안전 RM과 유용성 RM을 조건부로 결합한 점수입니다. 안전 관련 프롬프트이거나 안전 점수가 임계값보다 낮으면 안전 RM을 쓰고, 그 점수를 로짓으로 바꾼 뒤 whitening한 값입니다. 마지막 항은 원래 정책에서 멀어지는 것에 대한 페널티이고, 학습 안정성과 보상 해킹 완화에 쓰입니다 ([Touvron et al., 2023](https://arxiv.org/abs/2307.09288)).

PPO 앞에는 거부 샘플링 미세조정이 들어갑니다. 반복 단계마다 가장 최근 모델에서 프롬프트당 $$K$$개 답을 샘플링하고, RM 점수가 가장 높은 답으로 미세조정합니다. 이 샘플링은 가장 큰 모델에서만 하고, 작은 모델들은 그 데이터로 증류했습니다 ([Touvron et al., 2023](https://arxiv.org/abs/2307.09288)).

![N개 샘플 중 최대 보상과 중앙값 보상의 차이](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-14.png)

*출처: [Touvron et al., 2023](https://arxiv.org/abs/2307.09288)*

$$N$$을 1부터 100까지 늘리면서 학습 프롬프트에 대해 평균한 최대 보상과 중앙값 보상을 나란히 그렸고, 두 값의 차이는 거부 샘플링으로 얻을 수 있는 잠재 이득으로 읽힙니다. 샘플링 조건도 RLHF와 함께 움직였습니다. Llama 2-Chat-RLHF에서 10~100개 출력을 샘플링할 때 최적 온도는 $$T\in[1.2,1.3]$$이었고, 그래서 반복마다 온도를 다시 맞춰야 했습니다. 반복 방식에는 대가도 있었습니다. 직전 반복의 샘플만으로 반복 미세조정하자 전반적인 개선이 이어지는 와중에도 운율 있는 시 작성 같은 일부 능력이 퇴보했습니다 ([Touvron et al., 2023](https://arxiv.org/abs/2307.09288)).

### 굿하트를 함수형으로 적은 과최적화 스케일링 법칙

앞서 요약 실험에서 곡선으로만 관측된 보상 과최적화(reward overoptimization)를 Gao, Schulman, Hilton은 크기와 스케일링으로 정량화했습니다 ([Gao et al., 2022](https://arxiv.org/abs/2210.10760)).

![실제 RLHF와 gold RM을 정답으로 쓰는 합성 설정의 보상 모델 학습 구도 비교](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-15.png)

*출처: [Gao et al., 2022](https://arxiv.org/abs/2210.10760)*

이 구도에서 실제 RLHF는 사람이 만든 비교 데이터로 프록시 RM을 학습하고 그것을 RL이나 BoN으로 최적화합니다. 합성 설정에서는 gold RM이 정답 자리를 대신합니다. gold RM으로는 InstructGPT 보상 모델을 썼습니다. 비교 데이터는 gold RM 점수가 높은 궤적을 항상 선호로 표시하는 결정적 방식으로 만들고, 이 데이터로 프록시 RM을 학습했습니다 ([Gao et al., 2022](https://arxiv.org/abs/2210.10760)).

최적화량은 $$d=\sqrt{\mathrm{KL}(\pi \,\Vert\, \pi_{\text{init}})}$$로 잽니다. Best-of-$$n$$(BoN) 샘플링에서 gold 보상은 다음 함수형을 따릅니다 ([Gao et al., 2022](https://arxiv.org/abs/2210.10760)).

$$R_{\text{bo}n}\left(d\right)=d\left(\alpha_{\text{bo}n}-\beta_{\text{bo}n}d\right)$$

RL에서는 형태가 다릅니다 ([Gao et al., 2022](https://arxiv.org/abs/2210.10760)).

$$R_{\text{RL}}\left(d\right)={d}\left(\alpha_{\text{RL}}-\beta_{\text{RL}}\log{d}\right)$$

$$\alpha$$는 최적화 초기에 gold 보상이 오르는 기울기를, $$\beta$$는 과최적화로 깎이는 정도를 정하는 계수입니다. 두 함수형 모두 $$d$$가 작을 때는 첫 항이 지배해 보상이 늘지만, $$d$$가 커지면 두 번째 항이 커져 결국 보상을 끌어내립니다. 예를 들어 가상의 계수 $$\alpha_{\text{bo}n}=1$$, $$\beta_{\text{bo}n}=0.25$$라면 gold 보상은 $$d=2$$에서 최대 1에 도달하고, $$d=4$$에서는 0으로 돌아옵니다.

![보상 모델 크기를 바꾼 BoN과 RL 실험에서 프록시 최적화에 따른 gold 보상 변화](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-16.png)

*출처: [Gao et al., 2022](https://arxiv.org/abs/2210.10760)*

가로축은 제곱근 KL 척도이고, 학습된 프록시를 최적화하면 gold 보상이 먼저 오르다가 나중에 떨어지며, 앞의 함수형이 이 궤적에 잘 맞습니다 ([Gao et al., 2022](https://arxiv.org/abs/2210.10760)). 계수가 RM 크기에 따라 어떻게 변하는지도 따로 그렸습니다.

![RM 파라미터 수에 따른 BoN과 RL 과최적화 법칙 계수의 스케일링](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-17.png)

*출처: [Gao et al., 2022](https://arxiv.org/abs/2210.10760)*

$$\alpha_{\text{bo}n}$$, $$\beta_{\text{bo}n}$$, $$\beta_{\text{RL}}$$가 프록시 보상(점선)과 gold 보상(실선) 각각에 대해 파라미터 수와 함께 그려져 있습니다. RL에서는 $$\alpha_{\text{RL}}$$을 모든 RM 크기에서 고정해도 되어, $$\beta_{\text{RL}}$$ 하나로 깔끔한 스케일링 곡선이 나왔습니다 ([Gao et al., 2022](https://arxiv.org/abs/2210.10760)).

KL을 비교의 공통 화폐로 쓰는 관행에도 단서가 붙었습니다. BoN은 초기 정책 주변을 아주 국소적으로 탐색해서 KL이 대략 $$\log(n)$$으로만 늘어납니다. 반면 RL은 KL 페널티가 없으면 스텝 수에 대해 대략 이차로 KL이 늘어나, BoN보다 KL 효율이 훨씬 낮습니다. 저자들은 이 차이 때문에 KL이 방법 사이의 최적화량 비교에 적절하지 않다고 결론지었습니다 ([Gao et al., 2022](https://arxiv.org/abs/2210.10760)).

KL 페널티 자체의 효과는 이 계보의 설계 전제를 정면으로 흔듭니다.

![여러 KL 페널티 강도에서 프록시 보상과 gold 보상의 변화를 비교한 RL 실험](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-18.png)

*출처: [Gao et al., 2022](https://arxiv.org/abs/2210.10760)*

저자들은 KL 페널티가 gold 점수에 주는 효과가 조기 종료와 동등하다고 관측했습니다. 더 정확히는, KL 페널티는 gold RM 점수를 더 일찍 수렴하게 만들 뿐 KL 대 gold 보상의 경계 자체는 바꾸지 못했습니다. 저자들은 이 결론이 하이퍼파라미터에 민감할 수 있다고 덧붙였습니다. 정책 크기 쪽 결과도 직관과 어긋났습니다. 큰 정책은 최적화로 얻는 gold 이득이 더 작았지만 더 빨리 과최적화하지는 않았고, 두 정책의 gold 점수는 거의 같은 KL에서 정점을 찍었습니다. 모든 실험이 InstructGPT 환경에서 이루어졌으므로, 이 법칙이 다른 환경에 일반화되는지는 저자들도 열어 두었습니다 ([Gao et al., 2022](https://arxiv.org/abs/2210.10760)).

KL 페널티를 넣은 이유는 처음부터 "보상 모델이 유효한 범위"를 지키는 것이었습니다 ([Ziegler et al., 2019](https://arxiv.org/abs/1909.08593))([Stiennon et al., 2020](https://arxiv.org/abs/2009.01325))([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)). 과최적화 법칙 연구는 그 장치가 경계를 옮기지 못하고 멈추는 지점만 앞당긴다는 결과를 내놓았습니다 ([Gao et al., 2022](https://arxiv.org/abs/2210.10760)). 이 글의 판단으로는, 이 계보가 남긴 유산이 둘입니다. 하나는 SFT, 보상 모델, 참조 모델, 가치 함수를 모두 갖춘 무거운 레시피이고, 다른 하나는 그 무게를 늘려도 프록시와 진짜 선호 사이의 간극이 사라지지 않는다는 정량적 증거입니다. 반복 온라인 수집이 정적 데이터보다 낫다는 관측 ([Bai et al., 2022](https://arxiv.org/abs/2204.05862))과 맞물려, 이 간극은 목적함수의 형태보다 보상 신호가 어떤 분포에서 얼마나 믿을 만한지에 달린 문제로 보입니다.

## 사람 라벨을 원칙과 AI 판정으로 대체하는 확장 가능한 감독

RLHF의 비교 라벨은 사람이 아니라 모델이 매길 수도 있으며, 그렇게 해도 파이프라인의 나머지는 바뀌지 않습니다. Constitutional AI는 사람의 감독을 원칙 목록 하나로 줄이고 무해성 라벨을 모델의 판정으로 채웠고, RLAIF는 기성 LLM이 매긴 선호 라벨로도 요약 과제의 사람 평가에서 사람 피드백과 통계적으로 구별되지 않는 결과가 나온다는 것을 보였습니다.

### 원칙 목록만 남긴 사람의 감독

Constitutional AI(CAI)는 유해성에 대한 사람 라벨을 하나도 쓰지 않고 무해하면서도 회피적이지 않은 어시스턴트를 학습합니다. 사람이 제공하는 감독은 규칙이나 원칙의 목록, 즉 "헌법"뿐이고, 방법의 이름도 여기서 나왔습니다. 학습은 두 단계입니다. 지도학습(SL) 단계에서 모델이 스스로 비평하고 수정한 응답으로 미세조정하고, RL 단계에서 AI 피드백(AI feedback)으로 만든 선호 라벨로 강화학습을 합니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)).

![SL 단계의 비평·수정·지도학습과 RL 단계의 AI 비교 평가·선호 모델·RL로 이어지는 Constitutional AI 전체 과정 개요도](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-19.png)

*출처: [Bai et al., 2022](https://arxiv.org/abs/2212.08073)*

개요도의 위쪽 줄이 SL 단계이고, 그 아래로 RL 단계가 이어집니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)). 앞서 본 InstructGPT의 3단계 도식과 겹쳐 보면, 사람이 비교 데이터를 만들던 자리를 모델의 판정이 차지했을 뿐 뼈대는 그대로입니다.

### 비평과 수정으로 만든 지도학습 데이터

SL-CAI 단계는 유해한 응답을 끌어내는 프롬프트로 시작합니다. 모델이 응답을 샘플링하고, 원칙에 비추어 그 응답을 스스로 비평한 뒤 수정하며, 수정본으로 사전학습 모델을 미세조정합니다. 저자들이 쓴 원칙은 모두 16개이고, 수정할 때마다 그중 하나를 무작위로 뽑았습니다. 유용성이 떨어지지 않도록 helpful RLHF 모델의 응답도 학습 데이터에 섞었습니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)).

수정을 거듭하면 선호 모델(preference model, PM, 응답 쌍 중 어느 쪽이 나은지 점수를 매기도록 학습한 모델)이 매긴 무해성 점수는 단조롭게 올라갔지만, 순수 유용성 점수는 내려갔습니다. 원칙의 개수는 PM 점수에 영향이 없었습니다. 그래도 저자들은 원칙이 다양하면 응답도 다양해져 RL 단계의 탐색에 도움이 되리라고 예상했습니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)). 수정만으로는 무해성과 유용성을 동시에 밀어 올리지 못했고, 이 균형을 맞추는 일은 RL 단계로 넘어갔습니다.

### 객관식 판정의 로그확률로 만든 소프트 선호 라벨

RL-CAI는 피드백 모델에 원칙 하나와 두 응답을 보여 주고, 둘 중 어느 쪽이 더 무해한지 객관식으로 묻습니다. 정답 토큰 하나만 뽑아 쓰지 않고, 선택지 (A)와 (B)의 로그확률을 정규화한 값을 소프트 선호 라벨로 씁니다. 예를 들어 가상의 피드백 모델이 (A)에 로그확률 −0.4, (B)에 −1.2를 주었다면, 지수를 취해 정규화한 (A)의 확률은 약 0.69가 됩니다. 이 0.69는 앞서 본 Christiano et al.의 교차 엔트로피 손실에서 라벨러 선호 분포 $$\mu$$가 맡던 자리에 들어가므로, 확신이 덜한 비교는 손실에 약하게 반영됩니다. 선호 모델은 유용성 비교에 사람 라벨을, 무해성 비교에 AI 라벨을 섞어 학습했습니다. 라벨을 얻은 다음의 PM 학습과 RL은 RLHF와 똑같습니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)).

모델의 판정을 믿을 근거로 저자들은 HHH(도움, 정직, 무해) 비교 판별 실험을 내놓았습니다. 체인 오브 소트(chain-of-thought, CoT)로 추론하게 하면 언어 모델의 비교 판별 성능이 크게 올랐고, 추세로 보면 52B보다 큰 모델은 사람 피드백으로 학습한 PM과 경쟁할 만하다고 저자들은 내다봤습니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)).

![유용성·정직성·무해성 이진 비교 문항에서 사람 피드백 선호 모델과 객관식·CoT 방식 사전학습 언어 모델의 정확도를 모델 크기별로 비교한 그래프](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-20.png)

*출처: [Bai et al., 2022](https://arxiv.org/abs/2212.08073)*

평가 문항은 유용성·정직성·무해성을 묻는 438개의 이진 비교입니다. 모델 크기가 커질수록 객관식으로 답하는 사전학습 언어 모델, 특히 CoT를 더한 언어 모델의 곡선이 사람 피드백 선호 모델의 곡선에 다가갑니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)).

CoT에는 대가가 있었습니다. CoT 샘플은 어느 선택지가 낫다고 명시적으로 말하기 때문에 확률 목표가 0이나 1에 몰렸고, 보정이 맞지 않았습니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)). 위의 가상 예시로 말하면 0.69 같은 중간값이 사라지고 거의 1.0만 남아, 소프트 라벨의 장점이 없어집니다.

### 회피하지 않는 무해성과 개선된 Elo 프런티어

RL-CAI를 red team 프롬프트로 시험하자, 이 모델은 회피하는 일이 거의 없었고 대부분의 프롬프트에 뉘앙스 있는 무해한 답을 냈습니다. 반면 HH RLHF 모델은 "I can't answer that" 같은 회피 답변을 자주 냈습니다. RL-CAI는 RLHF 모델과 SL-CAI 모델보다 유의하게 더 무해했고, CoT를 쓴 버전은 조금 덜 도움이 되는 대신 조금 더 무해했습니다. 저자들은 CoT 없이 얻은 피드백 모델의 로그확률이 비교적 잘 보정되어 있다고 보고했습니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)).

![52B RL 실행 전체에 대해 크라우드워커 비교로 계산한 무해성 대 유용성 Elo 점수](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-21.png)

*출처: [Bai et al., 2022](https://arxiv.org/abs/2212.08073)*

이 산점도는 Elo 점수의 차이만 의미가 있고 절대값은 의미가 없습니다. 읽어야 할 축은 프런티어의 위치입니다. RL-CAI의 점들이 RLHF보다 같은 유용성에서 더 무해한 쪽으로 파레토 프런티어를 밀어냅니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)).

![RL 학습 시퀀스 수에 따른 유용성(왼쪽)과 무해성(오른쪽) Elo 점수 변화](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-22.png)

*출처: [Bai et al., 2022](https://arxiv.org/abs/2212.08073)*

학습이 진행되는 동안의 궤적을 담은 그래프에서는 유용성 Elo(왼쪽)와 무해성 Elo(오른쪽)가 RL 학습 시퀀스 수의 함수로 나옵니다. RL-CAI 스냅샷의 무해성이 높게 유지되어, 앞의 산점도와 같은 RLHF 대비 프런티어 차이를 시간 축에서 다시 확인할 수 있습니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)).

### TL;DR 비교 라벨을 기성 LLM이 매긴 RLAIF

CAI는 사람 라벨을 무해성에서만 대체했습니다. RLAIF(Lee et al.)는 범위를 넓혀, 기성 LLM이 매긴 선호 라벨만으로 보상 모델을 학습해 요약, 도움 되는 대화, 무해한 대화의 세 과제에서 사람 피드백 RLHF와 견줄 만한 성능을 냈다고 보고합니다 ([Lee et al., 2023](https://arxiv.org/abs/2309.00267)). 아래에서 수치로 확인하는 것은 그중 요약 과제뿐입니다.

![LLM이 선호 라벨을 만드는 RLAIF 파이프라인과 사람이 라벨을 만드는 RLHF 파이프라인 비교 도식](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-23.png)

*출처: [Lee et al., 2023](https://arxiv.org/abs/2309.00267)*

위아래 두 파이프라인에서 달라지는 부분은 비교 라벨을 만드는 주체 하나입니다 ([Lee et al., 2023](https://arxiv.org/abs/2309.00267)). 소개에서 세운 TL;DR 무대로 옮기면, 같은 게시물에 대한 두 요약 중 어느 쪽이 나은지를 사람 대신 LLM이 고르고, 그 비교로 보상 모델과 정책을 학습합니다.

Reddit TL;DR 요약 과제의 사람 평가에서 SFT 정책 대비 승률은 RLAIF가 71%, RLHF가 73%였고, 두 방법의 차이는 통계적으로 유의하지 않았습니다. 변형인 d-RLAIF(direct RLAIF)는 요약 과제에서 사람 평가자가 두 정책을 직접 비교했을 때 같은 크기의 RLAIF보다 60% 선호되었습니다 ([Lee et al., 2023](https://arxiv.org/abs/2309.00267)).

이 글의 판단으로는, 두 연구가 덜어낸 것은 라벨 수집 비용이고 파이프라인의 무게는 아닙니다. 앞서 본 대로 라벨 이후의 파이프라인이 RLHF와 같으므로 프록시 과최적화도 따라오고, CoT 라벨의 보정 문제처럼 판단 품질은 따로 검증해야 합니다.

## 보상 모델을 지우는 닫힌 형태 재매개화와 그 과적합

DPO는 KL 정규화 보상 최대화 문제의 최적 정책을 뒤집어 보상을 정책의 로그비로 표현하고, 그 덕분에 보상 모델과 RL 루프 없이 선호 데이터에 교차 엔트로피 하나만 걸어 정책을 학습합니다. 이후 연구는 이 단순화의 한계와 전제를 각각 다른 쪽에서 드러냈습니다. ΨPO/IPO 분석은 선호가 결정적일수록 KL 정규화가 힘을 잃는다는 것을 보였고, KTO는 쌍 비교 데이터가 없어도 된다는 것을 보였으며, DAA 스케일링 연구는 보상 모델이 없어도 과최적화가 그대로 남는다는 것을 확인했습니다.

### 최적 정책을 뒤집어 얻는 정책 속의 보상

DPO 저자들은 보상 함수에서 최적 정책으로 가는 해석적 대응을 이용해, 보상 함수에 대한 손실을 정책에 대한 손실로 바꿉니다. 앞서 본 KL 페널티를 단 보상 최대화 목적에는 닫힌 형태의 최적 정책이 있고, 이 식을 보상에 대해 풀면 다음과 같습니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)).

$$r(x,y)=\beta\log\frac{\pi_{r}(y\mid x)}{\pi_{\text{ref}}(y\mid x)}+\beta\log Z(x)$$

여기서 $$\pi_r$$는 보상 $$r$$에 대한 최적 정책, $$\pi_{\text{ref}}$$는 참조 모델, $$\beta$$는 KL 페널티의 세기, $$Z(x)$$는 프롬프트 $$x$$에만 의존하는 분배함수입니다. 계산하기 어려운 항은 $$Z(x)$$인데, 브래들리-테리 모델은 같은 프롬프트에 대한 두 응답의 보상 차이만 보기 때문에 이 항이 상쇄됩니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)). 그러면 선호 확률 전체를 정책과 참조 모델의 로그확률만으로 쓸 수 있습니다.

저자들은 이 재매개화가 학습되는 보상 모델의 클래스를 제한하지 않으며 최적 정책을 정확히 복원한다고 보였습니다. 정책이 곧 암묵적 보상(implicit reward) 모델이 되는 셈입니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)).

### 로그비 차이에 거는 이진 교차 엔트로피

재매개화한 보상을 브래들리-테리 선호 확률에 넣고 음의 로그우도를 취하면 DPO 손실이 나옵니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)).

$$\mathcal{L}_{\text{DPO}}(\pi_{\theta};\pi_{\text{ref}})=-\mathbb{E}_{(x,y_{w},y_{l})\sim\mathcal{D}}\left[\log\sigma\left(\beta\log\frac{\pi_{\theta}(y_{w}\mid x)}{\pi_{\text{ref}}(y_{w}\mid x)}-\beta\log\frac{\pi_{\theta}(y_{l}\mid x)}{\pi_{\text{ref}}(y_{l}\mid x)}\right)\right]$$

$$y_w$$는 선호된 응답, $$y_l$$은 선호되지 않은 응답, $$\mathcal{D}$$는 오프라인 선호 데이터셋, $$\sigma$$는 로지스틱 함수입니다. 괄호 안의 두 항이 각 응답의 암묵적 보상이고, 손실은 그 차이에 대한 이진 교차 엔트로피입니다.

TL;DR 비교 한 쌍으로 옮겨 보면 계산이 구체적으로 보입니다. 예를 들어 가상의 게시물에 대한 두 요약에서, 학습 중인 정책이 참조 모델보다 선호 요약 $$y_w$$의 로그확률을 1.0 높였고 비선호 요약 $$y_l$$의 로그확률을 0.5 낮췄다고 해 봅시다. $$\beta=0.1$$이면 괄호 안의 값은 $$0.1\times(1.0-(-0.5))=0.15$$이고, $$\sigma(0.15)\approx0.537$$이므로 이 쌍의 손실은 약 0.621입니다. 보상 모델이 이 쌍을 점수로 바꾸고 PPO가 그 점수를 쫓던 과정이, 로그확률 네 개로 계산하는 손실 하나로 줄었습니다.

그래디언트를 풀어 보면 선호 응답의 우도는 올라가고 비선호 응답의 우도는 내려가며, 각 예시는 암묵적 보상이 비선호 응답을 얼마나 높게 평가하는지에 따라 가중됩니다. 순서를 틀리게 매긴 쌍일수록 더 세게 고치는 구조이고, 저자들은 이 가중치를 뺀 단순 버전이 모델을 퇴화시킬 수 있다고 보고했습니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)). 위 가상 예시의 가중치는 $$\sigma(-0.15)\approx0.463$$이며, 두 요약의 순서가 뒤집혀 있었다면 0.537로 커집니다.

PPO와 비교하면 차이가 하나 더 있습니다. 저자들의 분석에서 PPO는 참조 정책의 soft value에 해당하는 정규화 항이 없으면 분산이 커지지만, DPO 재매개화로 얻은 보상에는 베이스라인이 필요 없습니다. 여기서 soft value는 프롬프트에만 의존하는 값으로, 앞의 재매개화 식에서는 $$\beta\log Z(x)$$ 항에 해당합니다. 같은 프롬프트의 두 응답을 비교하면 앞서 본 분배함수처럼 이 항이 상쇄되므로, DPO는 따로 베이스라인을 추정하지 않아도 됩니다. 파이프라인도 짧습니다. 오프라인 선호 데이터셋을 만들고 $$\pi_{\text{ref}}$$를 SFT 모델로 초기화한 뒤 앞의 손실을 최소화하며, SFT 모델이 없으면 선호 응답의 우도를 최대화해 $$\pi_{\text{ref}}$$를 만듭니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)).

![보상 모델을 학습한 뒤 RL로 최적화하는 RLHF와 선호 데이터에 교차 엔트로피를 직접 거는 DPO를 나란히 놓은 도식](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-24.png)

*출처: [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)*

두 경로를 나란히 놓은 도식에서 DPO 쪽에는 보상 모델 학습과 RL 샘플링 단계가 아예 없습니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)). 앞서 본 InstructGPT의 3단계 레시피에서 뒤의 두 단계가 손실 함수 하나로 합쳐졌습니다.

### 보상-KL 프런티어와 TL;DR 승률

최적화가 얼마나 잘 되는지는 감성 생성 과제에서 먼저 확인했습니다.

![참조 정책 대비 KL과 기대 보상의 프런티어에서 방법별 곡선을 비교한 그래프](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-25.png)

*출처: [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)*

왼쪽 패널은 같은 KL 예산에서 방법별 기대 보상을 비교합니다. 모든 KL 값에서 DPO의 보상이 가장 높습니다. IMDb 감성 생성에서 DPO의 보상-KL 트레이드오프는 PPO를 엄격하게 지배했고, 정답 보상에 접근하는 PPO-GT보다도 나은 프런티어를 그렸습니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)).

대화와 요약에서도 결과는 비슷했습니다.

![Anthropic-HH 1단계 대화에서 GPT-4가 판정한 방법별 승률과 학습 중 온도별 승률 변화](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-26.png)

*출처: [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)*

Anthropic-HH 1단계 대화의 GPT-4 판정 승률에서 테스트셋의 선택 응답을 넘어선 방법은 DPO 하나였고, 오른쪽의 학습 중 승률도 온도별로 안정적이었습니다. TL;DR 요약에서 PPO 요약과 비교했을 때 사람 평가자는 DPO 요약을 58% 선호했고, GPT-4(C) 판정에서는 54%였습니다. 이 비교에서 GPT-4(C) 판정은 사람 판정에 가까웠습니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)).

### 결정적 선호에서 무너지는 KL 정규화

IPO 저자들(Azar et al.)은 RLHF가 두 가지 근사에 기대고 있다고 정리합니다. 하나는 쌍별 선호를 점별 보상으로 대체하는 근사이고, 다른 하나는 보상 모델이 분포 밖으로 일반화한다는 가정입니다. DPO는 두 번째 근사만 우회하고 첫 번째에는 여전히 크게 의존합니다. 두 근사를 모두 피하려고 저자들은 선호 확률의 비감소 함수 $$\Psi$$를 KL 정규화와 함께 최대화하는 일반 목적함수 ΨPO를 정의했습니다 ([Azar et al., 2023](https://arxiv.org/abs/2310.12036)).

$$\max_{\pi}\mathop{\mathbb{E}}_{\begin{subarray}{c}x\sim\rho\\
y\sim\pi(.\mid x)\\
y^{\prime}\sim\mu(.|x)\end{subarray}}[\Psi(p^{*}(y\succ y^{\prime}\mid x))]-\tau D_{\text{KL}}(\pi\;||\;\pi_{\text{ref}})\,.$$

$$\rho$$는 프롬프트 분포, $$\mu$$는 비교 상대 응답 $$y'$$를 내는 행동 정책, $$p^{*}(y\succ y'\mid x)$$는 참 선호 확률, $$\tau$$는 정규화 세기입니다. $$\Psi(q)=\log(q/(1-q))$$로 두고 브래들리-테리 가정을 더하면 이 목적은 RLHF와 DPO로 환원됩니다 ([Azar et al., 2023](https://arxiv.org/abs/2310.12036)).

이 틀에서 문제가 드러납니다. 선호가 결정적이면, 즉 한 응답이 다른 응답을 항상 이기면 브래들리-테리 보상 차이가 무한대로 가고, 최적 정책은 $$\tau$$와 상관없이 열등한 응답의 확률을 0으로 만듭니다. 선호가 결정적일수록 KL 정규화가 약해지고, 그래서 DPO가 과적합합니다. 현실의 유한 데이터에서도 같은 일이 생깁니다. 한 쌍을 몇 번만 관측하면 표본으로 추정한 선호가 0이나 1 같은 극단값이 되기 쉽기 때문입니다. RLHF에서는 보상 모델이 이런 극단값에 과소적합하면서 정책에 정규화를 걸어 주지만, DPO는 보상 모델 학습을 건너뛰면서 그 정규화까지 잃습니다 ([Azar et al., 2023](https://arxiv.org/abs/2310.12036)).

처방으로 저자들은 $$\Psi$$를 항등함수로 바꾼 IPO를 제안합니다. DPO와 같은 방식으로 최적 정책 찾기를 근 찾기 문제로 바꾸는데, 이때 $$h_{\pi}(y,y')=\log\frac{\pi(y)\pi_{\text{ref}}(y')}{\pi(y')\pi_{\text{ref}}(y)}$$를 정의하고 이를 제곱 손실 하나로 표현합니다 ([Azar et al., 2023](https://arxiv.org/abs/2310.12036)).

$$L(\pi)=\mathop{\mathbb{E}}_{y,y^{\prime}\sim\mu}\left[\left(h_{\pi}(y,y^{\prime})-\frac{p^{*}(y\succ\mu)-p^{*}(y^{\prime}\succ\mu)}{\tau}\right)^{2}\right]$$

$$p^{*}(y\succ\mu)$$는 $$y$$가 $$\mu$$에서 뽑은 응답을 이길 확률입니다. $$h_\pi$$, 즉 두 응답의 로그비 차이가 선호 확률의 차이를 $$\tau$$로 나눈 값에 맞도록 회귀하는 손실입니다. 참 선호 확률은 알 수 없으므로, 실제로는 선호 데이터셋에서 뽑은 쌍의 베르누이 선호 지시자 $$I(y,y')$$로 바꾼 표본 손실을 씁니다 ([Azar et al., 2023](https://arxiv.org/abs/2310.12036)).

$$\mathop{\mathbb{E}}_{y,y^{\prime}\sim\mu}\Big[\big(h_{\pi}(y,y^{\prime})-\tau^{-1}I(y,y^{\prime})\big)^{2}\Big]$$

두 손실의 차이는 결정적 쌍 하나로 보입니다. 예를 들어 가상의 TL;DR 데이터에서 요약 A가 요약 B를 매번 이긴다고 해 봅시다. DPO 손실 $$-\log\sigma(\cdot)$$는 마진이 무한대일 때만 0이 되므로, B의 로그확률을 끝없이 내리는 쪽으로 그래디언트가 계속 흐릅니다. IPO에서는 목표값이 $$\tau^{-1}I$$라는 유한한 수여서, $$\tau=0.5$$이고 $$I=1$$이면 $$h_\pi$$가 2에 닿는 순간 제곱 손실의 그래디언트가 0이 되고 B의 확률은 그 자리에 멈춥니다.

![데이터셋 D1에서 IPO와 DPO가 학습하는 행동 확률의 곡선을 정규화 세기별로 비교한 그래프](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-27.png)

*출처: [Azar et al., 2023](https://arxiv.org/abs/2310.12036)*

세 행동짜리 장난감 데이터셋 D1의 학습 곡선이 이 차이를 그대로 보여 줍니다. DPO는 모든 $$\tau$$에서 결정적 정책으로 수렴해 참조 정책을 무시한 반면, IPO는 정규화가 강하면 탐욕적 정책으로 가지 않았습니다 ([Azar et al., 2023](https://arxiv.org/abs/2310.12036)).

![데이터셋 D3에서 한 번도 이기지 못한 행동의 확률이 IPO와 DPO에서 어떻게 변하는지 비교한 그래프](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-28.png)

*출처: [Azar et al., 2023](https://arxiv.org/abs/2310.12036)*

데이터셋 D3의 곡선에서 주목할 대상은 승리가 한 번도 관측되지 않은 행동입니다. DPO는 정규화를 아무리 세게 걸어도 이 행동의 확률을 없애 $$\pi_{\text{ref}}$$를 완전히 무시했고, IPO는 $$\tau$$에 따라 확률을 점진적으로 줄였습니다. 두 결과 모두 장난감 예제에서 나온 것입니다 ([Azar et al., 2023](https://arxiv.org/abs/2310.12036)).

### 쌍 대신 바람직함 여부만 묻는 KTO

KTO(Ethayarajh et al.)는 다른 쪽을 건드립니다. 정렬을 전망 이론(prospect theory)의 관점에서 보고, 선호 쌍 대신 출력이 바람직한지 아닌지라는 이진 신호만으로 생성물의 효용을 직접 최대화합니다. 저자들은 이런 데이터가 실제 환경에서 더 풍부하고, 싸고, 빨리 모인다고 봅니다 ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)). TL;DR 무대로 말하면, 두 요약을 나란히 놓고 고를 필요 없이 요약 하나마다 좋다, 나쁘다만 표시하면 됩니다.

저자들은 먼저 인간 인지 손실(human-aware loss, HALO)이라는 손실 계열을 정의합니다. 암묵적 보상에서 참조점의 기대 보상을 뺀 값에 비감소 가치함수를 씌운 손실이며, DPO와 PPO-Clip이 여기에 속합니다 ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)).

![여러 HALO가 내포하는 효용 함수를 전망 이론의 인간 가치함수와 겹쳐 그린 그래프](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-29.png)

*출처: [Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)*

곡선들을 겹쳐 보면 HALO들이 내포하는 가치함수가 손실회피처럼 전망 이론의 표준 가치함수와 같은 성질을 공유합니다. KTO 손실은 이 틀을 이진 신호에 맞춘 형태입니다 ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)).

$$L_{\text{KTO}}(\pi_{\theta},\pi_{\text{ref}})=\mathbb{E}_{x,y\sim D}[\lambda_{y}-v(x,y)]$$

$$v(x,y)$$는 가치함수로, 바람직한 출력에는 암묵적 보상 $$r$$에서 참조점 $$z_0$$를 뺀 값에, 바람직하지 않은 출력에는 $$z_0-r$$에 로지스틱 함수를 적용합니다. $$\lambda_y$$는 출력의 종류에 따라 달라지는 손실회피 계수로, $$\lambda_D$$는 바람직한(desirable) 출력에, $$\lambda_U$$는 바람직하지 않은(undesirable) 출력에 쓰이고, 로지스틱 함수 안의 $$\beta$$는 위험회피를 조절합니다. 원래 전망 이론의 가치함수에 쓰이는 지수 대신 로지스틱 함수를 썼습니다. 참조점 $$z_0$$는 정책과 참조 모델 사이의 KL이며, 마이크로배치 안에서 다음처럼 추정합니다 ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)).

$$\hat{z}_{\text{0}}=\max\left(0,\frac{1}{m}\sum_{1\leq i\leq m}\log\frac{\pi_{\theta}(y_{j}\mid x_{i})}{\pi_{\text{ref}}(y_{j}\mid x_{i})}\right)$$

$$m$$은 마이크로배치 크기이고, $$y_j$$는 $$x_i$$가 아닌 다른 프롬프트의 출력입니다. 마이크로배치 안에서 출력을 섞어 짝이 맞지 않는 쌍을 만들고, 이것으로 편향된 KL 추정치를 얻습니다. 학습을 안정시키려고 $$z_0$$로는 역전파하지 않으며, 이 값은 손실의 포화를 조절하는 데만 쓰입니다 ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)).

![GPT-4-0613이 판정한 승률에서 HALO에 속하는 방법들과 속하지 않는 방법들을 비교한 그래프](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-30.png)

*출처: [Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)*

HALO라는 분류가 실제로 의미가 있는지는 이 비교에서 드러납니다. 이기기 어려운 기준 출력에 대한 GPT-4 판정 승률에서, HALO인 DPO와 오프라인 PPO 변형이 HALO가 아닌 두 방법인 SLiC와 CSFT를 앞섰습니다 ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)).

![모델 규모별로 KTO와 DPO의 GPT-4-0613 판정 승률을 비교한 그래프](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-31.png)

*출처: [Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)*

규모별 비교에서 SFT에 쓰였을 출력을 상대로 잰 승률은 모든 규모에서 KTO가 DPO와 같거나 높았습니다. SFT+KTO는 SFT+DPO와 경쟁할 만했고, Llama-7B/13B/30B에서는 KTO 단독이 DPO 단독보다 나았으며, 이 차이는 다중 비교를 보정한 뒤에도 7B와 30B에서 유의했습니다 ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)).

SFT를 건너뛴 설정에서도 두 방법은 갈렸습니다. Llama-13B/30B에서 SFT 없이 KTO만으로 정렬한 모델이 SFT+KTO와 경쟁할 만했고, 시험한 방법 가운데 이런 거동을 보인 것은 KTO뿐이었습니다. DPO는 SFT 없이 쓰면 응답 길이가 폭증했습니다 ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)).

![SFT 없이 DPO와 KTO로 정렬한 모델의 응답 길이를 비교한 그래프](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-32.png)

*출처: [Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)*

응답 길이 그래프에서 SFT를 거치지 않은 DPO 모델은 장황하게 늘어놓다가 대화 전체를 환각하는 경향을 보였고, KTO 모델에는 이 문제가 없었습니다 ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)).

![바람직한 예시를 크게 줄인 불균형 데이터에서 KTO로 정렬한 Llama-7B와 DPO로 정렬한 모델의 성능 비교](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-33.png)

*출처: [Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)*

불균형 데이터 실험에서는 같은 데이터셋을 작고 치우치게 만든 경우를 봅니다. 바람직한 예시를 크게 줄인 불균형 데이터에서도 KTO로 정렬한 Llama-7B는 DPO로 정렬한 모델과 같거나 나은 성능을 냈습니다. 바꿔 말하면 바람직한 데이터를 최대 90%까지 버려도 DPO와 같거나 나은 성능을 유지했습니다 ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)). 참조 모델까지 없앤 KTO 변형은 메모리를 아끼지만 표준 KTO보다 성능이 낮고 손실회피 하이퍼파라미터에 더 민감했습니다.

### 보상 모델 없이도 남는 과최적화

보상 모델이 없으면 앞서 본 굿하트 현상도 사라질 것이라는 기대에 DAA 과최적화 연구(Rafailov et al., 2024)는 반대 증거를 내놓습니다. DPO, IPO, SLiC 같은 직접 정렬 알고리즘(DAA)은 별도 보상 모델이 없는데도 높은 KL 예산에서 고전적 RLHF와 비슷한 성능 저하를 보였고, 한 에폭이 끝나기 전에 열화하기도 했습니다 ([Rafailov et al., 2024](https://arxiv.org/abs/2406.02900)). 저자들은 Gao et al.이 RL에 대해 찾은 과최적화 함수형, 즉 앞서 본 $$\sqrt{\text{KL}}$$에 대한 혹 모양 곡선을 DAA에 그대로 적용했고, 정답 보상 대신 승률을 대용으로 썼습니다 ([Rafailov et al., 2024](https://arxiv.org/abs/2406.02900))([Gao et al., 2022](https://arxiv.org/abs/2210.10760)).

![6.9B 모델에서 DPO, SLiC, IPO의 에폭 내 진행 비율에 따른 승률과 KL 변화](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-34.png)

*출처: [Rafailov et al., 2024](https://arxiv.org/abs/2406.02900)*

6.9B 모델의 에폭 내 동역학을 담은 그래프에서는 위쪽 줄의 승률과 아래쪽 줄의 KL을 함께 읽어야 합니다. KL 제약이 낮을 때 대부분의 실험이 에폭의 처음 25% 안에서 최고 성능에 도달했고, 그 뒤로 학습을 더 할수록 나빠졌습니다. 세 알고리즘 모두 승률이 KL에 대해 혹 모양을 그렸고, DPO와 SLiC는 비슷했으며, IPO는 과최적화에 덜 취약하고 같은 제약에서 KL도 대체로 낮았습니다. 모델이 클수록 과최적화에 덜 취약했다는 관찰도 함께 나왔습니다 ([Rafailov et al., 2024](https://arxiv.org/abs/2406.02900)).

저자들이 제시한 원인은 손실의 형태와 데이터의 범위입니다. 선호 데이터의 비교 쌍들을 프롬프트-응답 공간 위에 늘어놓은 행렬을 생각하면, 프롬프트-응답 공간 $$\lvert\mathcal{X}\times\mathcal{Y}\rvert$$이 데이터셋 $$\lvert\mathcal{D}\rvert$$보다 훨씬 커서 이 행렬은 행(관측된 비교)보다 열(프롬프트-응답 조합)이 훨씬 많고, 따라서 자명하지 않은 영공간을 갖기 쉽습니다. 영공간 방향으로 암묵적 보상을 바꿔도 손실은 변하지 않으므로 DAA 손실은 엄격한 볼록이 아니게 되고, 데이터가 닿지 않는 방향의 보상은 정해지지 않습니다. 그 결과 암묵적 보상이 과소 제약되어, 정책이 데이터가 덮지 않는 분포 밖 응답에 확률을 실을 수 있습니다. 이 분석은 계산 자원의 한계로 더 큰 모델 규모까지는 확장되지 않았습니다 ([Rafailov et al., 2024](https://arxiv.org/abs/2406.02900)).

이 글의 판단으로는, 네 연구를 겹쳐 읽으면 DPO가 줄인 것은 파이프라인의 단계 수이고 최적화 대상의 취약성은 줄지 않았습니다. IPO가 지적한 결정적 선호의 문제, KTO에서 SFT 없이 DPO의 응답 길이가 폭증한 현상, DAA 연구의 에폭 내 열화는 모두 KL 정규화가 기대만큼 정책을 붙잡아 두지 못한다는 같은 증상으로 읽힙니다 ([Azar et al., 2023](https://arxiv.org/abs/2310.12036))([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306))([Rafailov et al., 2024](https://arxiv.org/abs/2406.02900)). 실무에서는 DPO 계열을 쓰더라도 에폭 초반부터 승률과 KL을 함께 추적하고, 선호가 0이나 1에 몰린 데이터에는 IPO처럼 목표를 유한하게 묶는 손실을 먼저 검토할 만합니다.

## 참조 모델과 별도 SFT 단계를 덜어낸 랭킹·오즈비·길이 정규화 목적함수

참조 모델을 빼고 지도 미세조정까지 정렬과 합치려면, 목적함수가 두 가지를 직접 해결해야 합니다. 하나는 응답 길이에 휘둘리지 않는 점수이고, 다른 하나는 선호 응답을 올리는 동안 비선호 응답이 함께 올라가지 않게 막는 장치입니다. RRHF는 길이 정규화(length normalization)한 로그확률로 응답을 줄 세웠습니다. ORPO는 오즈비(odds ratio) 페널티를 SFT 손실에 붙여 두 단계를 하나로 합쳤고, SimPO는 생성 지표와 같은 평균 로그우도에 목표 마진을 더해 참조 모델 없이 DPO를 앞섰습니다.

### 길이 정규화 로그확률 랭킹으로 시작한 RRHF

RRHF의 출발점은 단순합니다. 더 좋은 응답에는 더 큰 확률을, 더 나쁜 응답에는 더 작은 확률을 주도록 모델을 학습시킵니다. 응답은 초기 모델, ChatGPT, 사람처럼 여러 정책이 만든 $$k$$개이고, 각 응답 $$y_i$$에는 정책이 매기는 다음 점수가 붙습니다 ([Yuan et al., 2023](https://arxiv.org/abs/2304.05302)).

$$p_{i}=\frac{\sum_{t}\log P_{\pi}(y_{i,t}\mid x,y_{i,<t})}{\|y_{i}\|}$$

$$\|y_i\|$$는 응답의 토큰 수이고, 분자는 응답 전체의 로그확률 합입니다. 따라서 $$p_i$$는 토큰당 평균 로그확률입니다. 보상 $$r_i$$의 순서와 이 점수의 순서가 어긋나면 벌점을 주는 손실은 다음과 같습니다 ([Yuan et al., 2023](https://arxiv.org/abs/2304.05302)).

$$L_{rank}=\sum_{r_{i}<r_{j}}\max(0,p_{i}-p_{j})$$

보상이 낮은 $$i$$의 점수가 보상이 높은 $$j$$보다 크면 그 차이만큼 손실이 생기고, 순서가 맞으면 0입니다. RRHF는 여기에 보상이 가장 높은 응답에 대한 SFT 교차 엔트로피를 더하며, 마진은 두지 않습니다 ([Yuan et al., 2023](https://arxiv.org/abs/2304.05302)).

![RRHF와 PPO의 학습 흐름을 나란히 비교한 도식](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-35.png)

*출처: [Yuan et al., 2023](https://arxiv.org/abs/2304.05302)*

PPO와 나란히 놓은 흐름도에서 RRHF는 여러 정책의 샘플을 랭킹 손실로 바로 학습합니다 ([Yuan et al., 2023](https://arxiv.org/abs/2304.05302)).

길이 정규화가 하는 일은 TL;DR 비교 한 쌍으로 보면 분명해집니다. 예를 들어 가상의 게시물에 대해 사람이 고른 요약이 40토큰에 로그확률 합 −60이고, 고르지 않은 요약이 20토큰에 합 −36이라고 해 봅시다. 합으로 비교하면 비선호 요약의 점수가 더 높아 순서가 뒤집힙니다. 토큰당 평균으로 바꾸면 선호 요약은 −1.5, 비선호 요약은 −1.8이 되어 순서가 맞고 랭킹 손실도 0입니다. 이 정규화 점수가 뒤에서 SimPO 보상의 원형이 됩니다.

### 오즈비 페널티로 한 단계에 합친 ORPO

ORPO(Hong et al.)는 SFT 자체의 약점에서 시작합니다. 교차 엔트로피만으로는 선호 응답을 학습하는 동안 거부 응답에 벌을 줄 장치가 없고, 그래서 거부 응답 토큰의 로그확률도 선호 응답과 함께 올라갑니다 ([Hong et al., 2024](https://arxiv.org/abs/2403.07691)).

![HH-RLHF에서 OPT-350M을 미세조정하는 동안 선호 응답과 거부 응답의 로그확률 변화](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-36.png)

*출처: [Hong et al., 2024](https://arxiv.org/abs/2403.07691)*

OPT-350M을 HH-RLHF로 미세조정한 곡선을 보면, 감독 신호로는 선호 응답만 썼는데도 거부 응답의 로그확률이 같이 오릅니다. 저자들은 SFT 단계에서 비선호 생성 스타일에 약한 페널티만 주어도 선호에 맞춘 SFT가 된다고 보고, 참조 모델 없이 SFT와 선호 정렬을 한 단계로 합칩니다. 전체 목적함수는 두 손실의 가중합입니다 ([Hong et al., 2024](https://arxiv.org/abs/2403.07691)).

$$\mathcal{L}_{ORPO}=\mathbb{E}_{(x,y_{w},y_{l})}\left[\mathcal{L}_{SFT}+\lambda\cdot\mathcal{L}_{OR}\right]$$

$$\mathcal{L}_{SFT}$$는 선호 응답에 대한 음의 로그우도이고, $$\lambda$$는 오즈비 손실의 가중치입니다. 오즈는 길이 정규화한 평균 로그우도에서 얻은 확률 $$P_\theta$$로 정의합니다 ([Hong et al., 2024](https://arxiv.org/abs/2403.07691)).

$$\textbf{odds}_{\theta}(y|x)=\frac{P_{\theta}(y\mid x)}{1-P_{\theta}(y\mid x)}$$

오즈비 손실은 선호 응답과 비선호 응답 오즈의 로그비에 log-sigmoid를 씌운 형태이며, 로그 오즈비를 키울수록 줄어듭니다 ([Hong et al., 2024](https://arxiv.org/abs/2403.07691)).

$$\mathcal{L}_{OR}=-\log\sigma\left(\log\frac{\textbf{odds}_{\theta}(y_{w}|x)}{\textbf{odds}_{\theta}(y_{l}|x)}\right)$$

이 손실의 그래디언트는 두 항의 곱으로 나뉩니다 ([Hong et al., 2024](https://arxiv.org/abs/2403.07691)).

$$\nabla_{\theta}\mathcal{L}_{OR}=\delta(d)\cdot h(d)$$

$$d$$는 선호 쌍 하나이고, $$\delta(d)$$는 틀린 예측에 벌을 주는 항, $$h(d)$$는 선호 응답과 거부 응답의 그래디언트를 대조하는 항입니다 ([Hong et al., 2024](https://arxiv.org/abs/2403.07691)).

비율로 확률비 대신 오즈비를 고른 데에는 이유가 있습니다. 저자들은 확률비를 쓰면 SFT와 묶인 설정에서 비선호 토큰의 로짓을 지나치게 억누르고, 오즈비는 비선호 응답을 완만하게 구분하면서 선호 응답의 생성을 우선한다고 판단했습니다 ([Hong et al., 2024](https://arxiv.org/abs/2403.07691)).

![같은 입력 확률쌍에서 표본으로 얻은 로그 확률비와 로그 오즈비의 분포](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-37.png)

*출처: [Hong et al., 2024](https://arxiv.org/abs/2403.07691)*

같은 확률쌍을 넣어 두 값의 분포를 그린 시뮬레이션에서는 로그 오즈비의 범위가 로그 확률비보다 넓습니다 ([Hong et al., 2024](https://arxiv.org/abs/2403.07691)).

![RLHF, DPO, ORPO의 정렬 파이프라인을 비교한 도식](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-38.png)

*출처: [Hong et al., 2024](https://arxiv.org/abs/2403.07691)*

도식처럼 참조 모델이 빠지면서 RLHF나 DPO보다 메모리가 줄고, 배치마다 필요한 순전파 횟수도 절반이 됩니다 ([Hong et al., 2024](https://arxiv.org/abs/2403.07691)).

![UltraFeedback 테스트셋에서 OPT 세 규모별로 SFT, RLHF, DPO, ORPO 모델의 보상 분포 비교](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-39.png)

*출처: [Hong et al., 2024](https://arxiv.org/abs/2403.07691)*

UltraFeedback 테스트셋의 보상 분포는 OPT 모델 규모별로 방법을 나눠 그렸습니다. ORPO는 SFT의 분포를 높은 보상 쪽으로 옮겼고, 기대 보상을 높이는 효과는 RLHF나 DPO와 같거나 더 컸습니다. 같은 OPT 모델들의 보상 모델 기준 승률에서 ORPO는 SFT를 상대로 최대 80.5%, PPO를 상대로 최대 85.8%였습니다. AlpacaEval 2.0에서는 Phi-2(2.7B) ORPO가 6.35%, Llama-2(7B) ORPO가 9.44%를 기록했습니다 ([Hong et al., 2024](https://arxiv.org/abs/2403.07691)).

### 생성 지표와 맞춘 SimPO의 평균 로그우도 보상

SimPO(Meng et al.)는 DPO 보상의 정의를 문제 삼습니다. 앞서 본 DPO 손실의 암묵적 보상은 정책과 참조 모델의 로그비인데, 생성을 이끄는 지표는 대략 정책이 낸 응답의 평균 로그우도라서 둘이 일치하지 않습니다 ([Meng et al., 2024](https://arxiv.org/abs/2405.14734))([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)). 저자들은 이 학습과 추론의 불일치를 없애려고 보상을 생성 지표에 맞췄습니다. 불일치는 실제로 작지 않았습니다. UltraFeedback 학습 세트에서 DPO 보상 순서가 맞은 쌍 가운데 거의 절반은 평균 로그우도 순서가 반대였습니다 ([Meng et al., 2024](https://arxiv.org/abs/2405.14734)).

![UltraFeedback에서 DPO의 평균 로그확률과 길이의 상관, 보상과 우도 순위의 분할표, 보상 정확도를 SimPO와 비교한 그림](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-40.png)

*출처: [Meng et al., 2024](https://arxiv.org/abs/2405.14734)*

이 비교는 DPO의 평균 로그확률과 응답 길이 사이의 Spearman 상관, 보상 기준 순위와 우도 기준 순위의 분할표, 보상 정확도를 한데 모았습니다. SimPO는 보상을 정책의 길이 정규화 로그확률로 바꿉니다 ([Meng et al., 2024](https://arxiv.org/abs/2405.14734)).

$$r_{\text{SimPO }}(x,y)=\frac{\beta}{|y|}\log\pi_{\theta}(y\mid x)=\frac{\beta}{|y|}\sum_{i=1}^{|y|}\log\pi_{\theta}(y_{i}\mid x,y_{<i})$$

$$\vert y\vert$$는 응답 길이, $$\beta$$는 보상의 스케일입니다. 참조 모델 항이 없으므로 순전파는 정책 하나만 거칩니다. 이 보상을 브래들리-테리 모델에 넣고 목표 마진 $$\gamma$$를 빼면 목적함수가 됩니다 ([Meng et al., 2024](https://arxiv.org/abs/2405.14734)).

$$\mathcal{L}_{\text{SimPO }}(\pi_{\theta})=-\mathbb{E}_{(x,y_{w},y_{l})\sim\mathcal{D}}\left[\log\sigma\left(\frac{\beta}{|y_{w}|}\log\pi_{\theta}(y_{w}\mid x)-\frac{\beta}{|y_{l}|}\log\pi_{\theta}(y_{l}\mid x)-\gamma\right)\right]$$

RRHF 설명에 쓴 가상의 TL;DR 요약 한 쌍을 그대로 넣어 보면, 평균 로그확률이 −1.5와 −1.8이고 $$\beta=2.0$$, $$\gamma=0.5$$일 때 괄호 안은 $$-3.0+3.6-0.5=0.1$$이고 손실은 $$-\log\sigma(0.1)\approx0.644$$입니다. 순서가 맞아도 보상 차이가 마진을 겨우 넘었기 때문에, 손실은 두 요약을 더 벌리는 쪽으로 계속 그래디언트를 보냅니다.

길이 정규화는 장식이 아닙니다. 저자들은 보상에서 정규화 항을 빼면 길지만 품질이 낮은 응답 쪽으로 편향된다고 보고했습니다. UltraFeedback으로 학습한 모델을 보류 세트에서 재 보니, 평균 로그우도와 응답 길이의 Spearman 상관은 정규화를 뺀 SimPO가 0.82, DPO가 0.59, SimPO가 0.34였습니다. Mistral-Base 7B에서 정규화를 빼면 AlpacaEval 2 LC 승률이 21.5%에서 11.9%로 떨어졌습니다 ([Meng et al., 2024](https://arxiv.org/abs/2405.14734)).

![마진 크기에 따른 보상 정확도와 AlpacaEval 2 LC 승률, 그리고 보상 차이 분포의 변화](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-41.png)

*출처: [Meng et al., 2024](https://arxiv.org/abs/2405.14734)*

마진 실험에서는 두 지표가 어떻게 갈라지는지 확인하면 됩니다. $$\gamma$$를 키울수록 보류 세트의 보상 정확도는 올라갔지만 AlpacaEval 2 승률은 처음에 오르다가 이후 내려갔습니다. 생성 품질이 보상 마진만으로 정해지지 않는다는 뜻입니다. IPO도 SimPO와 비슷한 목표 마진을 두지만, 목적함수 전체로는 SimPO만큼 효과적이지 않았습니다 ([Meng et al., 2024](https://arxiv.org/abs/2405.14734)).

![보상 정식화만 다른 SimPO와 DPO를 여러 설정의 AlpacaEval 2와 Arena-Hard에서 비교한 그림](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-42.png)

*출처: [Meng et al., 2024](https://arxiv.org/abs/2405.14734)*

두 방법은 주로 보상 정식화에서만 다르며, 여러 설정의 AlpacaEval 2와 Arena-Hard에서 SimPO가 DPO를 크게 앞섰습니다. AlpacaEval 2 LC 리더보드에서 Llama-3-8B-Instruct-SimPO는 LC 44.7%로 Llama-3-8B-Instruct-DPO의 40.3%보다 높았습니다. 한계도 분명합니다. GSM8k 같은 추론 과제에서는 성능이 떨어져 때로 DPO와 비슷하거나 더 나빴고, 안전성과 정직성은 명시적으로 다루지 않았습니다 ([Meng et al., 2024](https://arxiv.org/abs/2405.14734)).

이 글의 판단으로는, 세 방법이 참조 모델을 빼고도 버틴 비결은 같은 곳에 있습니다. 응답 길이를 점수에서 떼어 내고, 비선호 응답을 밀어내는 힘을 오즈비든 마진이든 명시적인 형태로 넣었습니다. 다만 앞서 본 마진 실험처럼 목적함수 위의 정확도와 생성 품질은 따로 움직입니다.

## 브래들리-테리를 넘어 내시 균형을 쫓는 온라인 셀프 플레이

선호가 비이행적이거나 사람마다 갈릴 때는 점별 보상의 최댓값보다 "어떤 상대에게도 지지 않는 정책", 곧 선호 게임의 내시 균형(Nash equilibrium)을 목표로 삼는 편이 낫습니다. NLHF가 정렬 문제를 이렇게 다시 정의했고, 온라인 IPO는 이 균형 찾기가 현재 정책 샘플로 하는 셀프 플레이(self-play)와 같다는 것을 보였습니다. SPPO와 반복 DPO는 매 라운드 새로 샘플링한 응답으로 이 아이디어를 Mistral-7B, Llama-3-8B 같은 오픈 모델 학습에 적용했습니다 ([Wu et al., 2024](https://arxiv.org/abs/2405.00675))([Dong et al., 2024](https://arxiv.org/abs/2405.07863)).

### 점별 보상이 지우는 선호의 다양성

앞서 IPO가 지적한 쌍별→점별 근사를 NLHF(Munos et al.)는 버리고, 두 응답을 함께 입력받는 일반 선호 모델의 내시 균형을 계산하는 과제로 정렬을 정의합니다. 선호 모델에는 참조 정책 $$\mu$$와의 KL 페널티를 넣어 정규화하고, 이렇게 정규화한 선호 모델에서는 균형이 존재하고 유일합니다 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)).

차이는 사람들이 세 유형으로 나뉘고, 유형마다 행동 $$y_1, y_2, y_3$$ 가운데 서로 다른 하나를 선호하는 예에서 드러납니다. 보상 모델의 최적해, 곧 RLHF의 해는 $$y_1$$ 하나만 결정적으로 고릅니다. 반면 내시 해는 각 집단이 선호하는 세 행동에 거의 균등한 확률을 주므로 선호의 다양성에 더 잘 맞고 분포 변화에도 덜 민감합니다. 선호를 조금 비트는 섭동 $$\epsilon$$을 넣어 보면, RLHF 해는 $$\epsilon>0$$이면 $$y_1$$만, $$\epsilon<0$$이면 $$y_2$$만 고르도록 급변하지만 내시 균형은 $$\epsilon$$에 따라 매끄럽게 움직입니다 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)).

데이터 분포에 대한 의존성도 다릅니다. 표현력이 완벽한 이상적 경우에도 보상 모델의 최적해는 데이터를 만든 정책 분포에 따라 달라집니다. 선호 모델은 분포와 무관하게 사람의 선호 확률로 수렴합니다. 그래서 반복 RLHF에서는 정책이 바뀔 때마다 보상 모델을 다시 학습해야 하지만, NLHF는 같은 선호 모델을 계속 쓰면서 확장할 수 있습니다 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)).

![크기가 다른 선호 모델들의 학습 세트와 테스트 세트 정확도 학습 곡선](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-43.png)

*출처: [Munos et al., 2023](https://arxiv.org/abs/2312.00886)*

선호 모델 자체의 학습 곡선에서는 규모 효과가 눈에 띕니다. 모델이 클수록 정확도가 높지만 3B에서 11B로 키울 때 얻는 이득은 작습니다 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)).

![같은 크기의 선호 모델과 보상 모델의 학습·테스트 정확도 곡선 비교](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-44.png)

*출처: [Munos et al., 2023](https://arxiv.org/abs/2312.00886)*

저자들은 같은 크기의 보상 모델과 선호 모델을 학습 세트와 테스트 세트에서 나란히 비교해, 쌍을 통째로 보는 모델과 점수 하나를 내는 모델의 정확도 곡선을 함께 보여 줍니다 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)).

### 혼합 정책을 상대로 한 미러 디센트, Nash-MD

균형을 찾는 알고리즘인 Nash-MD는 현재 정책과 참조 정책 $$\mu$$를 기하 혼합한 정규화 정책 $$\pi_t^{\mu}$$를 상대로 미러 디센트 한 스텝을 밟습니다 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)).

$$
\pi_{t+1}\stackrel{\text{def}}{=}\arg\max_{\pi}\left[\eta_{t}\mathcal{P}(\pi\succ\pi_{t}^{\mu})-\text{KL}(\pi,\pi_{t}^{\mu})\right]
$$

$$\mathcal{P}(\pi\succ\pi_{t}^{\mu})$$는 $$\pi$$의 응답이 혼합 정책의 응답보다 선호될 확률이고, $$\eta_t$$는 스텝 크기입니다 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)). 혼합 정책을 이기는 쪽으로 움직이되, 그 혼합 정책에서 멀리 벗어나지 않게 KL로 묶습니다.

표 형태 정책에서는 강한 보장이 붙습니다. $$\eta_t=2/(\tau(t+2))$$로 두면 마지막 반복 정책이 정규화 내시 균형 $$\pi^*_\tau$$에 $$O(1/T)$$ 속도로 KL 수렴합니다 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)).

$$
\text{KL}(\pi^{*}_{\tau},\pi_{T})\leq\frac{8}{\tau^{2}(T+1)}
$$

$$\tau$$는 정규화 강도, $$T$$는 반복 횟수입니다. 수렴하는 대상이 과거 정책들의 평균이 아니라 마지막 반복 정책이어서 과거 정책을 저장할 필요가 없습니다. 기존 후회 최소화 알고리즘은 과거 정책의 혼합이 수렴하므로 그 정책들을 저장해야 하는데, 이 차이는 메모리가 큰 LLM에서 중요합니다 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)).

LLM용 딥러닝 버전인 Nash-MD-PG와 Nash-EMA-PG는 정책 경사를 씁니다. 상대 응답 $$y'$$는 기하 혼합 정책(Nash-MD-PG)이나 파라미터 지수이동평균 정책(Nash-EMA-PG)에서 뽑습니다. 선호 확률 $$\mathcal{P}(y\succ y'\mid x)$$에서 $$1/2=\mathcal{P}(y\succ y\mid x)$$를 베이스라인으로 빼는데, 이 베이스라인은 그래디언트의 기댓값을 바꾸지 않으면서 가치 함수를 학습하지 않고도 분산을 줄입니다 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)). 요약 응답 비교 한 쌍에 대입하면 이렇습니다. 예를 들어 가상의 선호 모델이 정책의 요약이 상대 요약보다 나을 확률을 0.7로 본다면, 그 요약의 로그확률을 올리는 신호는 $$0.7-0.5=0.2$$이고, 0.4로 본다면 $$-0.1$$이 되어 로그확률을 내립니다.

요약 과제에서 LLM이 판정하는 쌍별 평가를 해 보니, 혼합 계수 $$\beta$$를 0.125~0.375 사이의 중간값으로 둔 Nash-MD-PG가 Self-Play와 Best-Response 기준선보다 일관되게 우수했습니다. 다만 수렴 보장은 표 형태 정책에만 있습니다. 딥러닝 버전은 이론과 달리 $$\beta$$를 감쇠시키지 않는 근사입니다 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)).

### 온라인 IPO: 같은 손실, 현재 정책의 샘플

Calandriello et al.의 온라인 IPO는 앞서 본 IPO 손실에서 정적 데이터 분포만 현재 정책 $$\pi$$가 만든 분포로 바꿉니다 ([Calandriello et al., 2024](https://arxiv.org/abs/2403.08635))([Azar et al., 2023](https://arxiv.org/abs/2310.12036)). 데이터 분포 쪽의 $$\pi$$에는 stop-gradient를 씌웁니다. 손실을 만들 데이터는 $$\pi$$에서 생성하지만, 생성 과정 자체로는 미분하지 않는다는 뜻입니다 ([Calandriello et al., 2024](https://arxiv.org/abs/2403.08635)).

이 작은 변경에서 이론적 연결이 나옵니다. 두 생성물을 모두 현재 정책에서 뽑아 선호 모델로 라벨을 다는 이 온라인 설정에서는 IPO 손실의 최적화가 Nash-MD, 곧 셀프 플레이로 선호 모델의 내시 균형을 찾는 일과 동치가 됩니다. 이를 확장한 IPO-MD는 온라인 정책과 참조 정책의 혼합 정책에서 데이터를 생성합니다. 반면 온라인 DPO에는 이 셀프 플레이 동치 성질이 없습니다 ([Calandriello et al., 2024](https://arxiv.org/abs/2403.08635)).

요약 과제에서 온라인 변형끼리 나란히 비교하니, 3개 시드로 9번 비교한 평균에서는 IPO가 모든 온라인 알고리즘을 이겼습니다. 표준편차까지 고려하면 IPO와 IPO-MD는 통계적으로 구분되지 않았고, 둘 다 DPO와 SLiC를 포함한 나머지를 일관되게 이겼습니다. 실험은 770M 모델과 단일 요약 과제에 한정되어 있어, 대규모 대화 에이전트로 일반화되는지는 검증되지 않았습니다 ([Calandriello et al., 2024](https://arxiv.org/abs/2403.08635)).

### SPPO: 승률에 맞춘 제곱 회귀

SPPO(Wu et al.)는 균형을 향한 이상적인 갱신을 승률로 가중한 승법 가중치 업데이트로 씁니다 ([Wu et al., 2024](https://arxiv.org/abs/2405.00675)).

$$
\pi_{t+1}(\mathbf{y}\mid\mathbf{x})\propto\pi_{t}(\mathbf{y}\mid\mathbf{x})\exp(\eta\mathbb{P}(\mathbf{y}\succ\pi_{t}\mid\mathbf{x}))
$$

$$\mathbb{P}(\mathbf{y}\succ\pi_t\mid\mathbf{x})$$는 응답 $$\mathbf{y}$$가 현재 정책의 응답을 이길 확률이고, $$\eta$$는 학습률입니다. 현재 정책을 상대로 자주 이기는 응답일수록 확률이 지수적으로 커집니다. 문제는 정규화 상수입니다. DPO와 IPO는 쌍의 차이를 취해 정규화 항을 상쇄하지만, SPPO는 로그 정규화 상수를 $$\eta/2$$로 대체하고 응답마다 로그비를 직접 회귀합니다 ([Wu et al., 2024](https://arxiv.org/abs/2405.00675)).

$$
\boldsymbol{\theta}_{t+1}\leftarrow\mathop{\mathrm{argmin}}_{\boldsymbol{\theta}}\mathbb{E}_{(\mathbf{x},\mathbf{y},\widehat{P}(\mathbf{y}\succ\pi_{t}\mid\mathbf{x}))\sim\mathcal{D}_{t}}\bigg(\log\bigg(\frac{\pi_{\boldsymbol{\theta}}(\mathbf{y}\mid\mathbf{x})}{\pi_{t}(\mathbf{y}\mid\mathbf{x})}\bigg)-\eta\bigg(\widehat{P}(\mathbf{y}\succ\pi_{t}\mid\mathbf{x})-\frac{1}{2}\bigg)\bigg)^{2}
$$

$$\widehat{P}$$는 추정한 승률이고, $$\mathcal{D}_t$$는 라운드 $$t$$에 만든 데이터셋입니다. 참조 정책 자리에 직전 라운드 정책 $$\pi_t$$가 들어가는 점이 DPO와 다릅니다. 예를 들어 가상의 요약이 다른 후보들을 상대로 승률 0.8을 받으면 로그비의 목표는 $$0.3\eta$$이고, 승률 0.3을 받으면 $$-0.2\eta$$입니다. 그래서 SPPO는 승자와 패자 사이의 격차만 맞추지 않습니다. $$\eta$$로 나눈 로그비 기준으로 승자는 $$+1/2$$ 근처로, 패자는 $$-1/2$$ 근처로 밀어 승자의 우도가 실제로 오르도록 유도합니다. 이는 위 손실식에서 승률이 1과 0일 때의 목표 $$\pm\eta/2$$에 해당합니다. 저자들의 정리에 따르면 DPO와 KTO는 본래 반복적인 방법이 아니고 내시 균형 보장도 없지만, IPO와 SPPO는 설계상 내시 해를 반복적으로 풉니다 ([Wu et al., 2024](https://arxiv.org/abs/2405.00675)).

한 라운드는 네 단계로 돌아갑니다 ([Wu et al., 2024](https://arxiv.org/abs/2405.00675)).

1. 현재 정책 $$\pi_t$$에서 프롬프트마다 응답 $$K$$개를 샘플링합니다.
2. 선호 오라클로 $$K$$개 응답끼리 승률을 주석합니다. 질의 수는 $$O(K^2)$$입니다.
3. (프롬프트, 응답, 승률 추정) 튜플로 데이터셋을 만듭니다.
4. 앞의 제곱 손실로 정책을 갱신합니다.

관련 연구와 비교하면 위치가 분명해집니다. Swamy 등의 SPO는 PPO나 SAC로 같은 지수 갱신을 근사하지만, 대규모 LLM 미세조정에서 안정적으로 돌리기가 어렵습니다. DNO와 REBEL은 DPO처럼 정규화 항을 상쇄하는 쌍 손실을 쓰는 반면, SPPO의 손실은 정책 경사와 연결됩니다 ([Wu et al., 2024](https://arxiv.org/abs/2405.00675)).

![AlpacaEval 2.0에서 GPT-4-Turbo 상대 길이 통제 승률과 일반 승률의 SPPO 반복별 변화](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-45.png)

*출처: [Wu et al., 2024](https://arxiv.org/abs/2405.00675)*

반복 라운드를 따라가 보면, AlpacaEval 2.0에서 GPT-4-Turbo를 상대로 한 승률이 길이 통제(LC) 기준과 통제하지 않은 기준 모두에서 꾸준히 오릅니다. Mistral-7B 기반으로 AlpacaEval 2.0에서 GPT-4-Turbo를 상대한 SPPO의 LC 승률은 28.53%로, 반복 DPO 계열에서 가장 좋은 Snorkel 모델의 26.39%와 IPO의 25.45%보다 높았습니다. MT-Bench에서는 Mistral-7B-SPPO Iter3가 평균 7.59로 모든 기준선을 넘었지만, 처음 두 라운드에서는 점수가 오히려 떨어졌습니다. Open LLM Leaderboard 평균은 Mistral-7B SPPO가 66.75, Llama-3-8B SPPO가 70.29까지 올랐고, Mistral에서는 반복을 거듭하면서 점수가 내려간 라운드도 있었습니다 ([Wu et al., 2024](https://arxiv.org/abs/2405.00675)).

![PairRM-0.4B를 심판으로 한 베이스 모델, DPO, IPO, SPPO 모델 간 쌍별 승률 히트맵](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-46.png)

*출처: [Wu et al., 2024](https://arxiv.org/abs/2405.00675)*

베이스 모델인 Mistral-7B-Instruct-v0.2와 DPO, IPO, SPPO 모델들의 쌍별 승률 히트맵은 PairRM-0.4B가 심판입니다. 이 심판은 긴 출력을 내는 모델에 유리할 수 있습니다 ([Wu et al., 2024](https://arxiv.org/abs/2405.00675)).

![승률 추정 미니배치 크기 K=2와 K=5에 따른 SPPO의 AlpacaEval 2.0 일반·LC 승률](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-47.png)

*출처: [Wu et al., 2024](https://arxiv.org/abs/2405.00675)*

$$O(K^2)$$ 질의 비용은 $$K$$를 줄이면 내려갑니다. 미니배치 크기를 바꾼 실험에서 첫 라운드에는 $$K=5$$가 $$K=2$$보다 나았지만 이후 라운드에서는 차이가 줄었고, 작은 $$K$$일수록 응답 길이가 덜 늘었습니다. 저자들은 이를 근거로 승률 추정의 노이즈에 SPPO가 강건하다고 봅니다 ([Wu et al., 2024](https://arxiv.org/abs/2405.00675)).

### 누적 데이터로 굴리는 반복 DPO

RLHF Workflow(Dong et al.)는 같은 온라인화를 DPO 손실 그대로 구현합니다. 일반 정식화에서는 반복마다 지금까지 모은 데이터 전체로 정책을 갱신하고, 새 프롬프트에 대해 응답 쌍을 모아 선호 신호를 받아 데이터셋에 추가합니다. 탐색이 계속되므로 분포 밖 응답 문제가 완화됩니다 ([Dong et al., 2024](https://arxiv.org/abs/2405.07863)).

![반복 직접 선호 학습의 구현 도식: 첫 반복은 SFT 모델에서 시작하고 이후 반복마다 선호 데이터가 누적됨](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-48.png)

*출처: [Dong et al., 2024](https://arxiv.org/abs/2405.07863)*

구현 도식에서는 첫 반복을 눈여겨볼 만합니다. 이력 데이터가 비어 있어 첫 정책은 초기화 그대로인 SFT 체크포인트이고, 이후 반복부터 이전 라운드의 선호 데이터가 쌓입니다. 실용 알고리즘은 반복마다 SFT 모델을 참조로 둔 DPO로 정책을 학습합니다. 선호 쌍은 보상 모델로 순위를 매긴 응답 가운데 best-of-8과 worst-of-8로 만듭니다 ([Dong et al., 2024](https://arxiv.org/abs/2405.07863)). 내시 균형을 직접 겨냥하지는 않지만, 매 라운드 현재 정책 분포에서 피드백을 새로 받는다는 골격은 SPPO와 같습니다.

이 글은 이 계보를 이렇게 읽습니다. 균형 개념이 준 실질적인 이득은 이론보다 운영 방식에서 나왔습니다. 고정된 선호 모델이나 오라클을 두고 현재 정책 샘플로 신호를 계속 갱신하는 방식이 SPPO에서 오픈 모델의 벤치마크 향상으로 이어졌습니다. 다만 PairRM 심판은 긴 출력을 내는 모델에 유리할 수 있고, MT-Bench에서는 처음 두 라운드에 점수가 떨어졌습니다 ([Wu et al., 2024](https://arxiv.org/abs/2405.00675)).

## 크리틱 없는 그룹 상대 어드밴티지와 검증 가능한 보상의 성공 증폭

GRPO는 PPO의 가치 모델을 떼어 내고, 같은 질문에서 뽑은 출력 여러 개의 평균 보상을 기준선으로 삼아 수학 추론용 RL을 가볍게 만들었습니다 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)). 이후 Mroueh의 분석은 정답 여부만 알려 주는 이진 보상 아래에서 GRPO가 성공확률로 가중된 대조 손실임을 보였습니다. 이 손실은 참조 모델보다 높은 성공확률로 수렴하지만, 성공 사례를 새로 만들어 내지는 못합니다 ([Mroueh, 2025](https://arxiv.org/abs/2503.06639)).

### 가치 모델 대신 그룹 평균을 쓰는 기준선

앞서 본 PPO 기반 RLHF는 토큰별 KL 페널티를 보상에 섞고, 보상 모델은 마지막 토큰에만 점수를 줍니다. Shao et al.은 이 구조 때문에 모든 토큰에서 정확한 가치 함수를 학습하기 어렵고, 정책과 크기가 비슷한 모델을 하나 더 메모리에 올려야 한다고 지적합니다. 대안은 두 가지입니다. 가치 모델을 더 잘 학습하거나, 기준선을 다른 데서 구하는 것입니다. GRPO는 후자를 택했습니다. 질문 하나에 출력 $$G$$개를 샘플링하고, 그 그룹의 보상 통계를 기준선으로 씁니다 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)).

![PPO와 GRPO의 학습 구조 비교도: GRPO는 가치 모델 없이 그룹 점수로 기준선을 추정함](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-49.png)

*출처: [Shao et al., 2024](https://arxiv.org/abs/2402.03300)*

두 구조를 나란히 놓은 도식에서 GRPO 쪽에는 가치 모델이 빠져 있고, 그 자리를 그룹 점수에서 추정한 기준선이 채웁니다. 저자들은 이 변경으로 학습 자원이 크게 줄었다고 밝힙니다. 결과만 채점하는 결과 감독에서 그룹 상대 어드밴티지(group relative advantage)는 다음과 같습니다 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)).

$$
\hat{A}_{i,t}=\widetilde{r}_{i}=\frac{r_{i}-{\rm mean}(\mathbf{r})}{% raw %}{{{% endraw %}\rm std}(\mathbf{r})}
$$

$$r_i$$는 그룹의 $$i$$번째 출력이 받은 보상이고, $$\mathbf{r}$$는 그룹 전체의 보상 벡터입니다. $$\hat{A}_{i,t}$$는 출력 $$i$$의 $$t$$번째 토큰에 주는 어드밴티지로, 한 출력 안의 모든 토큰에 같은 값이 들어갑니다. 예를 들어 가상의 그룹 다섯 개 가운데 네 개가 정답이어서 보상이 1, 하나가 오답이어서 0이라면, 그룹 평균과 표준편차로 정규화한 뒤 정답 출력의 어드밴티지는 $$0.5$$, 오답 출력은 $$-2$$가 됩니다. 저자들은 이 방식이 보상 모델의 성격과도 맞는다고 봅니다. 보상 모델은 대개 같은 질문에 대한 출력끼리의 비교로 학습되기 때문입니다 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)).

KL은 보상에 넣지 않고 손실에 따로 더합니다. 추정량은 항상 양수인 비편향 추정량을 씁니다 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)).

$$
\mathbb{D}_{KL}\left[\pi_{\theta}||\pi_{ref}\right]=\frac{\pi_{ref}(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta}(o_{i,t}\mid q,o_{i,<t})}-\log\frac{\pi_{ref}(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta}(o_{i,t}\mid q,o_{i,<t})}-1
$$

$$q$$는 질문, $$o_{i,t}$$는 출력 $$i$$의 $$t$$번째 토큰, $$\pi_{ref}$$는 참조 모델입니다. 이 두 항을 합치면 토큰 하나의 로그확률에 곱해지는 그래디언트 계수가 나옵니다 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)).

$$
GC_{GRPO}(q,o,t,\pi_{\theta_{rm}})=\hat{A}_{i,t}+\beta\left(\frac{\pi_{ref}(o_{i,t}\mid o_{i,<t})}{\pi_{\theta}(o_{i,t}\mid o_{i,<t})}-1\right)
$$

$$\beta$$는 KL 계수입니다. 인자로 적힌 $$\pi_{\theta_{rm}}$$는 우변에 나타나지 않으므로, 이 계수는 $$\hat{A}_{i,t}$$와 KL 항만으로 정해집니다. 둘째 항은 정책이 참조 모델보다 그 토큰에 확률을 더 많이 주면 음수가 되어 당기는 힘으로 작용합니다. 실제 RL 단계에서는 질문마다 출력 64개를 뽑고 KL 계수를 0.04로 둡니다. 추론 단계마다 보상을 주는 과정 감독 버전도 있습니다. 여기서는 토큰의 어드밴티지를 그 뒤에 오는 단계들의 정규화 보상의 합으로 계산합니다 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)).

### 수학 벤치마크에서 본 효과와 그 범위

![DeepSeekMath-Instruct 1.3B를 RFT, Online RFT, GRPO 결과 감독·과정 감독으로 추가 학습한 두 벤치마크 성능](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-50.png)

*출처: [Shao et al., 2024](https://arxiv.org/abs/2402.03300)*

DeepSeekMath-Instruct 1.3B를 RFT, Online RFT, GRPO 결과 감독·과정 감독으로 추가 학습해 두 벤치마크에서 비교하면 두 가지 순서가 드러납니다. GRPO는 Online RFT를 앞섰습니다. 저자들은 그 원인을 보상 크기에 따라 양·음의 그래디언트 계수를 달리 주는 데서 찾습니다. 또 과정 감독 GRPO가 결과 감독 GRPO보다 나았습니다. 7B 모델은 GSM8K와 MATH의 체인 오브 소트(chain-of-thought) 데이터만으로 RL을 했는데도 모든 평가 지표에서 DeepSeekMath-Instruct 7B를 넘었고, 학습 도메인 밖 벤치마크도 함께 좋아졌습니다 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)).

![DeepSeekMath-Instruct 7B의 반복 강화학습 성능 곡선, 두 벤치마크](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-51.png)

*출처: [Shao et al., 2024](https://arxiv.org/abs/2402.03300)*

DeepSeekMath-Instruct 7B의 반복 RL 곡선에서는 특히 첫 반복에서 큰 향상이 나타납니다 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)).

![SFT와 RL DeepSeekMath 7B의 GSM8K·MATH Maj@K와 Pass@K 비교(온도 0.7)](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-52.png)

*출처: [Shao et al., 2024](https://arxiv.org/abs/2402.03300)*

이 상승을 해석할 때는 SFT 모델과 RL 모델의 Maj@K, Pass@K를 함께 봐야 합니다. Maj@K는 샘플 $$K$$개의 다수결 답이 맞는 비율이고, Pass@K는 $$K$$개 중 하나라도 맞는 비율입니다. GSM8K와 MATH에서 RL은 Maj@K를 올리지만 Pass@K는 올리지 못합니다 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)). 후보 $$K$$개 안에 정답이 하나라도 들어갈 확률은 그대로이고, 이미 나오던 정답이 다수결에서 이기는 빈도만 늘었습니다.

### 이진 검증 보상 아래의 가중 대조 손실

Mroueh는 이진 보상 아래 GRPO의 동역학을 이론으로 분석합니다. 보상이 정답 1, 오답 0인 검증 가능한 보상(verifiable reward)일 때, 평균과 분산으로 보상을 보정하는 GRPO는 대조 손실을 유도합니다. 대조 샘플은 이전 정책이 생성한 합성 데이터입니다. 손실의 가중치는 이전 정책 $$\pi_{\theta_{\text{old}}}$$가 그 질문에서 성공할 확률 $$p$$로 정해집니다. $$p$$가 높으면 성공 샘플에는 작은 가중치가 붙고 실패 샘플은 크게 벌점을 받습니다. $$p$$가 낮으면 드문 성공 샘플을 강하게 강화합니다 ([Mroueh, 2025](https://arxiv.org/abs/2503.06639)). 앞의 가상 예시에서 오답 하나가 $$-2$$를 받은 것도 이 구조를 따른 결과입니다.

![이전 정책의 성공확률에 따른 GRPO의 성공 샘플·실패 샘플 가중치](/assets/images/from-rlhf-to-dpo-and-grpo-lineages-of-llm-preference-optimization-53.png)

*출처: [Mroueh, 2025](https://arxiv.org/abs/2503.06639)*

성공확률을 가로축에 둔 가중치 곡선을 보면, 성공 샘플과 실패 샘플의 가중치가 $$p$$에 따라 서로 반대 방향으로 움직입니다 ([Mroueh, 2025](https://arxiv.org/abs/2503.06639)).

저자는 한 반복의 성공확률을 다음 반복의 성공확률로 보내는 사상 $$h_{\varepsilon,p_{\mathrm{ref}}(q)}$$를 두고, 이 사상이 참조 모델의 성공확률 $$p_{\mathrm{ref}}(q)$$와 $$\varepsilon$$에 의존하도록 정식화합니다. 그 고정점 $$p^*(q)$$는 $$p_{\mathrm{ref}}(q)$$보다 큽니다. 직관적으로 보면, 앞서 본 가중치가 매 반복 성공 샘플을 실패 샘플보다 상대적으로 밀어 올리므로 균형점이 참조 모델보다 위에 놓이는 셈입니다. 저자는 이를 성공 증폭이라 부릅니다. 클리핑을 쓰는 표준 GRPO 대신 직전 반복에 대한 KL만 거는 Mirror GRPO를 쓰면, 퇴화하지 않은 $$p_{\mathrm{ref}}$$에서 단조 개선과 성공확률 1로의 수렴이 보장됩니다. 반대로 참조 정책 정규화를 섞으면 단조 개선은 보장되지 않습니다 ([Mroueh, 2025](https://arxiv.org/abs/2503.06639)).

한계도 같은 분석에서 나옵니다. 어떤 경우에도 GRPO는 성공을 새로 만들지 않습니다. $$p_{\mathrm{ref}}(q)=0$$인 질문에서는 0이 흡수 고정점이어서, 이를 벗어나려면 별도의 탐색이 필요합니다 ([Mroueh, 2025](https://arxiv.org/abs/2503.06639)). 그룹 전체가 오답이면 평균을 빼는 순간 모든 어드밴티지가 0이 되므로, 학습 신호가 아예 생기지 않습니다.

이 글은 두 논문을 서로를 보완하는 근거로 읽습니다. 성공 증폭 정리와 흡수점 한계는 Maj@K만 오르고 Pass@K는 그대로인 관찰과 잘 들어맞습니다. GRPO는 가치 모델을 없앤 덕분에 싸게 돌아가지만, 그 대가로 기준선을 그룹 안의 성공 분포에 전적으로 의존합니다. 그래서 모델이 이미 가끔은 맞히는 문제에서는 강력하고, 한 번도 맞히지 못하는 문제에서는 무력합니다.

## 계보마다 되풀이되는 과최적화와 오프라인·온라인 데이터 분포의 문제

어느 계보가 어떤 구성요소를 덜어냈든, 프록시를 과하게 밀어붙이면 굿하트 현상은 형태만 바꿔 다시 나타납니다. 그 처방으로 여러 계보가 공통으로 내놓은 답은 현재 정책이 만든 출력에 피드백을 새로 받는 온라인화였습니다.

### 구성요소를 바꿔도 돌아오는 굿하트

출발점은 앞서 본 RLHF의 TL;DR 요약 실험입니다. 그 실험에서 보상 모델을 과최적화하면 진짜 사람 선호가 떨어졌고, Gao et al.이 그 크기를 스케일링 법칙으로 정량화했습니다 ([Stiennon et al., 2020](https://arxiv.org/abs/2009.01325))([Gao et al., 2022](https://arxiv.org/abs/2210.10760)). 이후 계보들은 이 문제의 원인으로 보이는 구성요소를 하나씩 바꿨지만, 현상은 매번 다시 보고되었습니다.

사람 라벨을 AI 판정으로 바꾼 Constitutional AI에서는 RL-CAI를 과학습하면 굿하트 행동이 나타났습니다. 모델이 유해 프롬프트에 지나치게 가혹하게 반응하거나 "you are valid, valued, and cared for" 같은 상투 문구를 끼워 넣었습니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)). 보상 모델을 지운 직접 정렬 알고리즘에서도 앞서 본 DAA 과최적화가 나타났습니다.

길이 정규화도 해법이 되지 못했습니다. DPO 실험에서 목적식에 길이 보정을 넣자 KL-승률 파레토 경계는 달라졌지만 과최적화는 해소되지 않았고, 오히려 악화될 수도 있었습니다. 모델이 작거나 KL 예산이 작을수록 정책은 길이 같은 단순한 특징으로 더 강하게 외삽했습니다 ([Rafailov et al., 2024](https://arxiv.org/abs/2406.02900)). 참조 모델까지 없앤 SimPO의 저자들도 같은 위험을 인정합니다. 참조 모델에 대한 명시적 정규화가 없으므로 원칙적으로 보상 해킹이 생길 수 있다는 것입니다. 다만 실험에서는 관찰되지 않았습니다 ([Meng et al., 2024](https://arxiv.org/abs/2405.14734)).

### 정책 분포를 따라가는 피드백

온라인화의 필요성은 계보의 맨 처음에 이미 나왔습니다. Christiano et al.은 온라인 질의 없이 보상을 학습하면 성능이 나빴다고 보고합니다. 점유 분포가 비정상적이어서 보상의 일부만 학습되었고, 그 결과 Pong에서 끝없이 랠리만 이어 가는 행동이 나타났습니다. 그래서 저자들은 피드백을 정적으로 주지 말고 RL과 엮어야 한다고 결론 내렸습니다 ([Christiano et al., 2017](https://arxiv.org/abs/1706.03741)).

GRPO 논문은 이 축을 여러 방법을 한데 묶는 그래디언트 형태로 정리합니다 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)).

$$
\nabla_{\theta}\mathcal{J}_{% raw %}{{\color[rgb]{1,0,0}\mathcal{A}}{% endraw %}}(\theta)=\mathbb{E}[\underbrace{(q,o)\sim{\color[rgb]{1,0,0}\mathcal{D}}}_{Data\ Source}]\left(\frac{1}{|o|}\sum_{t=1}^{|o|}\underbrace{GC_{% raw %}{{\mathcal{A}}{% endraw %}}(q,o,t,{\color[rgb]{1,0,0}\pi_{% raw %}{{rf}}{% endraw %}})}_{Gradient\ Coefficient}\nabla_{\theta}\log\pi_{\theta}(o_{t}\mid q,o_{<t})\right)
$$

$$\mathcal{A}$$는 학습 방법, $$\mathcal{D}$$는 질문 $$q$$와 출력 $$o$$를 공급하는 데이터 소스, $$\pi_{rf}$$는 보상 함수, $$GC_{\mathcal{A}}$$는 토큰 $$o_t$$의 로그확률에 곱하는 그래디언트 계수입니다. 이 틀에서 SFT, RFT, DPO, PPO, GRPO는 세 요소의 조합으로만 구분됩니다. 데이터 소스를 기준으로 나누면 DPO는 SFT 모델이 오프라인으로 샘플링한 선호·비선호 쌍을 쓰고, Online RFT, PPO, GRPO는 학습 중인 정책에서 실시간으로 샘플링합니다 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)). 예를 들어 가상의 TL;DR 데이터셋에서 비교 쌍이 모두 SFT 모델의 요약이라면, 학습이 진행되어 정책이 만드는 요약이 달라질수록 그 요약은 비교 쌍 어디에도 없는 분포로 옮겨 갑니다.

여러 계보는 이 간극을 같은 방식으로 메웠습니다. Ziegler et al.은 온라인 설정에서 보상 모델을 주기적으로 재학습했습니다 ([Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)). Anthropic HH RLHF는 최신 정책으로 비교를 새로 모았고, 그 이득이 데이터 양 때문이 아님을 보였습니다 ([Bai et al., 2022](https://arxiv.org/abs/2204.05862)). Constitutional AI는 후속 방향으로 AI 감독을 이용한 반복 온라인 학습을 제안합니다. 선호 모델을 새 AI 피드백으로 정책 분포에 맞춰 갱신하면 사람 없이도 자동화할 수 있다는 구상입니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)). 온라인 IPO는 정적 데이터 분포를 현재 정책의 샘플로 바꿨고, 그 결과가 셀프 플레이와 같아짐을 보였습니다 ([Calandriello et al., 2024](https://arxiv.org/abs/2403.08635)). SPPO와 반복 DPO는 라운드마다 응답을 새로 샘플링합니다 ([Wu et al., 2024](https://arxiv.org/abs/2405.00675))([Dong et al., 2024](https://arxiv.org/abs/2405.07863)). 비용은 앞서 본 Ziegler et al.의 배치 수집 절충안이 이미 보여 줍니다.

이 글의 해석으로는, 온라인화가 해결하는 것은 데이터가 정책을 따라가지 못하는 분포 불일치이고, 프록시 자체의 결함은 그대로 남습니다. 온라인으로 갱신해도 판정자가 보상 모델이나 AI 피드백 모델이라면, 정책은 그 판정자의 약점을 새 분포에서 다시 찾아낼 수 있습니다. NLHF가 분포와 무관하게 수렴하는 선호 모델을 내세운 것도 이 지점과 맞닿아 있습니다 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)).

## 단계 수, 참조 모델, 데이터 형태로 갈리는 방법별 트레이드오프

핵심 방법들은 파이프라인 단계 수, 참조 모델과 크리틱의 유무, 요구하는 피드백 형태, 온라인 여부의 네 축에서 서로 다른 선택을 했습니다. 그래서 아래 표의 승률은 비용과 견고성을 무엇과 맞바꿨는지와 함께 읽어야 합니다.

### 덜어낸 구성요소와 그 대가

단계 수부터 보면 InstructGPT의 SFT, 보상 모델, PPO 세 단계가 기준선입니다 ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)). DPO는 보상 모델 학습과 RL 루프를 하나의 분류 손실로 합쳤지만 SFT 모델을 참조 정책으로 두는 단계는 그대로 남겼습니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)). ORPO는 오즈비 페널티를 SFT 손실에 더해 정렬 전체를 한 단계로 줄였습니다 ([Hong et al., 2024](https://arxiv.org/abs/2403.07691)). KTO는 충분히 큰 Llama 모델에서 SFT를 건너뛰어도 경쟁력이 있었습니다. 같은 조건에서 DPO는 SFT를 빼면 응답 길이가 폭증했습니다 ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)). 단계를 줄일수록 SFT가 맡던 분포 고정을 목적함수가 대신 떠안게 됩니다.

참조 모델과 크리틱 쪽에서는 PPO가 가장 무겁습니다. 보상 모델로 초기화한 가치 함수를 정책과 별도로 학습해야 하기 때문입니다 ([Stiennon et al., 2020](https://arxiv.org/abs/2009.01325))([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)). GRPO는 그룹 평균 보상을 기준선으로 삼아 크리틱을 없앴지만 참조 정책에 대한 KL 항은 손실에 남겼습니다 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)). SimPO와 ORPO는 참조 모델까지 지워 메모리와 계산을 아꼈습니다 ([Hong et al., 2024](https://arxiv.org/abs/2403.07691))([Meng et al., 2024](https://arxiv.org/abs/2405.14734)). 그 대신 SimPO 저자들이 인정했듯이 명시적 정규화가 없어서 보상 해킹을 원칙적으로 막을 장치도 없습니다 ([Meng et al., 2024](https://arxiv.org/abs/2405.14734)).

피드백 형태는 데이터 비용을 직접 좌우합니다. 앞서 본 TL;DR 비교 한 쌍을 예로 들면 방법마다 이 쌍을 소비하는 방식이 다릅니다. RLHF는 이 쌍으로 브래들리-테리 보상 모델을 학습한 뒤 그 점수를 PPO 보상으로 씁니다 ([Stiennon et al., 2020](https://arxiv.org/abs/2009.01325)). DPO는 같은 쌍을 참조 정책 대비 로그비 차이에 바로 넣습니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)). KTO는 쌍을 요구하지 않고 출력마다 붙은 바람직/바람직하지 않음 신호만 받습니다 ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)). 예를 들어 가상의 TL;DR 데이터에서 한 쌍을 '선택된 요약은 바람직, 버려진 요약은 바람직하지 않음'으로 쪼개면 그대로 KTO 데이터가 됩니다. NLHF의 선호 모델은 두 요약을 함께 입력받아 선호 확률을 냅니다 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)). SPPO는 현재 정책이 프롬프트당 뽑은 여러 응답의 모든 쌍을 오라클에 물어야 하므로 질의 수가 응답 수의 제곱으로 늘어납니다 ([Wu et al., 2024](https://arxiv.org/abs/2405.00675)). GRPO 분석이 전제한 검증 가능한 이진 보상은 정답 판정기가 있는 수학 같은 영역의 신호입니다 ([Mroueh, 2025](https://arxiv.org/abs/2503.06639)). 요약처럼 정답이 없는 과제에는 이런 판정기가 없습니다. 사람 라벨을 줄이려면 Constitutional AI와 RLAIF처럼 AI 피드백으로 라벨 출처를 바꾸는 선택지가 남습니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073))([Lee et al., 2023](https://arxiv.org/abs/2309.00267)).

온라인 여부는 앞서 다룬 분포 불일치의 축입니다. 오프라인 DPO는 SFT 모델의 샘플에 묶이고, 온라인 IPO, SPPO, 반복 DPO는 매 라운드 샘플링 비용과 판정 비용을 치르는 대신 정책을 따라가는 데이터를 얻습니다 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300))([Calandriello et al., 2024](https://arxiv.org/abs/2403.08635))([Wu et al., 2024](https://arxiv.org/abs/2405.00675))([Dong et al., 2024](https://arxiv.org/abs/2405.07863)).

### 승률표를 읽는 방법

표의 '주요 결과' 열에는 서로 다른 기준선, 심판, 벤치마크가 섞여 있습니다. DPO 실험에서는 GPT-4 판정과 사람 판정의 승률이 달랐습니다 ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)). SPPO의 쌍별 비교는 PairRM을 심판으로 써서 그쪽에 유리할 수 있습니다 ([Wu et al., 2024](https://arxiv.org/abs/2405.00675)). 그래서 행 사이의 수치를 직접 비교하면 안 됩니다. 같은 행 안에서 무엇을 기준으로 이겼는지를 먼저 확인해야 합니다.

'한계' 열은 대부분 규모와 범위의 제약입니다. 온라인 IPO는 770M 모델과 요약 한 과제에 머물렀고 ([Calandriello et al., 2024](https://arxiv.org/abs/2403.08635)), DAA 과최적화 분석도 계산 자원 때문에 더 큰 규모로 가지 못했습니다 ([Rafailov et al., 2024](https://arxiv.org/abs/2406.02900)). NLHF의 수렴 보장은 표 형태 정책에만 성립합니다 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)). 이 글의 해석으로는, 덜어낸 구성요소가 많은 방법일수록 표의 높은 승률은 '같은 비용에서 얼마나 잘했나'로 읽고, 과최적화나 보상 해킹에 대한 견고성은 표 밖의 증거로 따로 확인해야 합니다. 쌍 데이터가 이미 있고 계산이 빠듯하면 참조 모델 없는 오프라인 목적함수가 합리적입니다. 판정기나 선호 오라클을 반복해서 부를 수 있으면 온라인 방법이 분포 불일치를 줄이는 쪽으로 비용을 쓰는 선택입니다.

| 논문 | 계보 | 주요 결과 | 한계 |
| --- | --- | --- | --- |
| Deep RL from Human Preferences | RLHF 기초 | MuJoCo 700 라벨: RL 거의 일치 | Atari Qbert 사람 피드백 실패 |
| Fine-Tuning LMs from Preferences | RLHF 기초 | 60k RL vs 지도학습 TL;DR 승률 97%; 인간 피드백 vs 감성 분류기 77% 선호 | 요약 정책은 추출형 'smart copier' |
| Learning to Summarize | RLHF 기초 | 1.3B RLHF TL;DR 참조 대비 선호 61% (SFT 10배 모델 43%); 6.7B RLHF TL;DR 길이통제 선호 ~65% | RL 학습 비용 320 GPU-일 |
| InstructGPT | RLHF 기초 | 175B InstructGPT vs GPT-3 선호율 85±3%; 1.3B InstructGPT가 175B GPT-3보다 선호 (100배 작음) | PPO-ptx의 DROP/SQuADv2/번역 잔존 성능 하락 |
| Anthropic HH RLHF | RLHF 기초 | 온라인 PM base/RS/online 정확도 74%/70%/67% | 최악 사례 강건성 미검증과 RL 불안정성 |
| RM Overoptimization Laws | RLHF 기초 | RM 데이터 임계값 약 2,000개 비교 | 합성 설정의 한계 |
| Constitutional AI | AI 피드백 | HHH 원본 221문항 정확도 90% 초과 | 절대 유해성 점수 보정 한계 |
| RRHF | 참조 모델 없는 단순화 | Alpaca-RRHF_SP HH 평균 보상 -0.96 | 다중 응답 비용 및 과최적화 |
| DPO | 직접 선호 최적화 | TL;DR DPO 대 PPO 인간 선호 58%; CNN/DailyMail DPO 승률 0.36 vs PPO 0.26 | 모델 규모 6B 한계 |
| RLAIF | AI 피드백 | 무해 대화 RLAIF 무해율 88%; AI 라벨링 비용 10배 이상 절감 | 길이 편향 및 피드백 결합 이득 없음 |
| IPO | 직접 선호 최적화 | 장난감 예제 D1 관측 선호 3개 | 장난감 예제 한정 평가 |
| NLHF | 온라인·게임 이론 | T5X-XL 선호 모델 정확도 0.78 vs 보상 모델 0.76 | 소규모 요약 실험, 공정 비교 불가 |
| KTO | 직접 선호 최적화 | Mistral-7B OpenAssistant 입력당 하나 KTO 승률 0.631 | 저노이즈 선호 데이터에서 KTO 과소적합 위험 |
| GRPO | 크리틱 없는 RL | DeepSeekMath-RL 7B GSM8K 88.2% / MATH 51.7% | RL의 Maj@K 개선·Pass@K 정체 |
| ORPO | 참조 모델 없는 단순화 | Mistral-ORPO-β AlpacaEval 12.20%; Mistral-ORPO-β MT-Bench 7.32 | 비교 방법 범위·7B 이하 규모 한계 |
| SPPO | 온라인·게임 이론 | Mistral-7B-SPPO Iter3 AlpacaEval LC 승률 28.52%; Llama-3-8B-SPPO AlpacaEval LC 승률 38.77% | 단일 데이터셋·제한된 벤치마크 |
| RLHF Workflow | 온라인·게임 이론 | vanilla DPO 8B Arena-Hard 22.4 | 길이 페널티 모델 길이 편향 ablation |
| SimPO | 참조 모델 없는 단순화 | Gemma-2-9B-it-SimPO AlpacaEval 2 LC 72.4%; SimPO 실행시간 약 20%, 메모리 약 10% 절감 | 마진 γ 수동 튜닝 필요 |

## 단순화와 온라인화 사이에서 남는 보상 신호의 품질 문제

목적함수는 계보마다 가벼워졌고 샘플링은 정책 분포 쪽으로 옮겨 갔지만, 신호 자체의 품질은 어느 계보도 풀지 못했습니다. 누구의 선호를 신호로 삼는지, 최악 사례에서 얼마나 버티는지, 라벨의 척도를 얼마나 믿을 수 있는지가 그 품질을 이룹니다.

### 가벼워진 소비 방식, 그대로인 신호

계보를 가로지르는 흐름은 앞서 본 구성요소 덜어내기와 온라인화 두 가지입니다.

두 흐름이 바꾼 것은 신호를 소비하는 방식입니다. 그래도 "사람이 고른 요약이 더 낫다"는 라벨을 정답으로 받아들이는 전제는 모든 방법에 공통입니다.

### 누구의 선호인가

InstructGPT 저자들은 소수의 계약 라벨러, 그것도 주로 영어 사용자가 선호를 결정했다고 밝히면서 이 집단이 배포된 모델을 쓰고 그 영향을 받을 사람 전체를 대표하지 못한다고 인정했습니다 ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)). 라벨러 집단의 습관도 신호에 섞입니다. TL;DR에서 RLHF 모델이 참조 요약보다 지나치게 자주 선호되자, 저자들은 모델이 라벨러의 휴리스틱을 악용했을 가능성을 지적했습니다 ([Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)).

AI 피드백은 이 질문을 없애지 않고 위치만 옮깁니다. Constitutional AI는 사람 라벨 대신 원칙 목록을 감독으로 삼았고 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)), RLAIF는 기성 LLM의 판정이 요약에서 사람 피드백과 비슷한 결과를 낸다고 보였습니다 ([Lee et al., 2023](https://arxiv.org/abs/2309.00267)). 그러면 질문은 "원칙을 누가 썼는가", "판정 모델은 누구의 선호를 학습했는가"로 바뀝니다. NLHF는 서로 다른 선호가 섞인 상황에서 내시 해가 한 행동으로 쏠리지 않는다는 사례를 보였습니다. 이 사례는 선호가 섞인 뒤에 무엇을 최적화하느냐를 다룹니다 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)).

### 평균 승률 뒤에 가려진 최악 사례

앞서 본 승률 수치는 모두 평균을 잽니다. 그러나 실패는 꼬리에서 일어났습니다. 보상 부호가 뒤집힌 버그 하나로 모델이 일관되게 최악의 출력을 학습한 일이 있었고 ([Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)), RL-CAI를 과학습하자 유해 프롬프트에 지나치게 가혹한 답이나 상투적 문구가 나왔습니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)). 보상 모델의 최적해는 선호 차이의 부호가 바뀌면 한 응답에서 다른 응답으로 급변합니다 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)). 앞서 본 DAA의 과소 제약과 SimPO의 정규화 부재도 같은 꼬리 위험입니다. 검증 가능한 보상을 쓰는 GRPO도 참조 정책의 성공확률이 0인 문제에서는 성공 샘플을 새로 만들지 못합니다 ([Mroueh, 2025](https://arxiv.org/abs/2503.06639)). 평균 승률이 오른다고 해서 이런 꼬리가 줄었는지는 알 수 없습니다.

### 라벨 척도의 신뢰도

신호의 세 번째 약점은 척도입니다. 계보의 출발점부터 사람은 절대 점수보다 비교를 더 일관되게 준다는 관찰이 있었습니다. 그 비교조차 사람이 일정 확률로 무작위 응답한다고 가정하고 모델링해야 했습니다 ([Christiano et al., 2017](https://arxiv.org/abs/1706.03741)). Llama 2는 선호 등급을 마진으로 넣어 비교의 강도를 반영하려 했습니다 ([Touvron et al., 2023](https://arxiv.org/abs/2307.09288)). 반대 방향의 문제도 있습니다. 라벨이 0과 1로 굳으면 ΨPO 분석이 보였듯 DPO의 KL 정규화가 약해집니다 ([Azar et al., 2023](https://arxiv.org/abs/2310.12036)). 그런데 체인 오브 소트를 쓴 AI 판정은 확률이 0이나 1에 몰려 보정되지 않았습니다 ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)). 판정자를 바꾸면 척도도 흔들리며, 이는 앞서 승률표를 읽으며 본 대로입니다.

### 신호 품질이 남은 병목

이 글은 단순화와 온라인화가 신호 품질 문제를 오히려 더 앞으로 드러냈다고 봅니다. 보상 모델의 과소적합이 정규화 역할을 하던 완충 장치는 직접 최적화에서 사라졌습니다 ([Azar et al., 2023](https://arxiv.org/abs/2310.12036)). 온라인화는 분포 불일치를 줄이는 대신 판정자의 편향을 매 라운드 그대로 정책에 되먹입니다. 그래서 다음 진전은 새 목적함수보다 라벨 집단의 구성, 꼬리 위험의 측정, 판정 확률의 보정에서 나올 가능성이 큽니다. 정답 판정기가 있는 영역이 크리틱 없는 RL로 빠르게 옮겨 간 흐름도, 이 문제를 푼 결과라기보다 문제를 우회할 수 있는 곳부터 먼저 옮겨 간 결과로 읽힙니다.

## 선호 데이터에서 검증 가능한 보상으로 옮겨 가는 정렬의 무게중심

쌍별 선호로 보상을 배우던 RLHF는 보상 모델 없는 직접 최적화와 게임 이론적 재정의를 거쳐, 정답을 검증할 수 있는 영역에서는 크리틱 없는 RL로 무게중심을 옮기고 있습니다.

출발점은 앞서 본 SFT, 보상 모델, PPO의 3단계 파이프라인이었습니다. 이후 갈래들은 이 비교 한 쌍을 다루는 방식을 바꾸었습니다. DPO는 앞서 다룬 대로 보상 모델을 지웠습니다. NLHF는 비교 자체를 선호 모델로 남기고 정렬을 그 선호 모델의 내시 균형 찾기로 재정의했습니다 ([Munos et al., 2023](https://arxiv.org/abs/2312.00886)). 셀프 플레이와의 동치는 앞서 본 온라인 IPO에서 성립합니다. 수학처럼 정답을 판정할 수 있는 영역에서는 사람의 비교 대신 검증기가 신호를 맡았습니다. GRPO는 앞서 본 그룹 평균 기준선으로 가치 모델을 덜어낸 방법이며, 앞서 본 성공 증폭이 그 이론적 근거입니다 ([Mroueh, 2025](https://arxiv.org/abs/2503.06639)).

계보 전체를 한 문장으로 줄이면 이렇게 됩니다. 정렬 연구는 사람의 비교 판단을 더 싸고 안정적으로 소비하는 방법을 찾아 왔고, 비교 판단 자체를 대체할 수 있는 곳부터 검증 가능한 보상으로 넘어가고 있습니다. 다만 이 이동이 끝난 것은 아닙니다. 검증기가 없는 대화, 요약, 무해성 같은 과제에서는 여전히 선호 신호가 유일한 감독입니다. 검증 가능한 보상도 앞서 본 흡수점 한계, 곧 참조 정책의 성공확률이 0인 문제에서는 힘을 쓰지 못합니다.

이 글의 관점에서 실무 선택은 신호의 종류가 먼저 결정합니다. 정답을 판정할 수 있으면 GRPO류의 크리틱 없는 RL이 가장 가볍습니다. 판정할 수 없으면 직접 최적화나 온라인 선호 최적화 중에서 고르게 됩니다. 어느 경우든 학습된 보상 모델이나 검증기 같은 대리 보상을 참조 정책에서 얼마나 멀리 최적화할지(과최적화와 보상 해킹의 위험), 그리고 신호를 누가 만들었는지(소수 라벨러의 편향이나 검증기의 오류)를 함께 따져야 합니다.

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
