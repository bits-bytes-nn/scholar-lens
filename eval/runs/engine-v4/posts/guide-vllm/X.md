---
layout: post
title: "vLLM in Practice: PagedAttention, V1 Engine, and Production Serving"
date: 2026-11-08 23:27:00
description: "vLLM의 PagedAttention으로 KV 캐시를 블록 단위로 관리하는 방식과 V1 엔진의 프로세스 구조를 설명합니다. 스케줄러, 프리픽스 캐싱, 설치, 엔진 인자, CPU 백엔드, Helm 배포까지 실무 설정을 정리합니다."
categories: ["Tech Guides", "Language-Models"]
tags: ["PagedAttention", "KV-Cache-Management", "vLLM-V1-Engine", "Prefix-Caching", "Helm-Deployment", "Model-Serving", "Efficient-Attention"]
cover: /assets/images/language-models.jpg
use_math: true
---
## TL;DR

vLLM은 LLM 추론과 서빙을 위한 오픈소스 엔진입니다. KV 캐시 메모리가 낭비되면 한 배치에 묶을 수 있는 요청 수와 처리량이 줄어들기 때문에, vLLM은 그 메모리를 관리하는 방식을 바꿔 처리량을 올립니다. 기존 시스템은 요청마다 KV 캐시를 최대 길이만큼 연속 메모리에 미리 할당합니다. vLLM 논문 실험에서 FasterTransformer와 Orca는 KV 캐시 메모리의 20.4%–38.2%만 실제 토큰 상태 저장에 썼습니다.

PagedAttention의 핵심은 KV 캐시를 운영체제의 페이지처럼 작은 단위로 쪼개어 다루는 데 있습니다. 블록 테이블이 연속된 논리 블록을 흩어진 물리 블록에 매핑하고, 물리 블록은 토큰이 생성될 때마다 필요한 만큼 할당합니다. 이렇게 하면 낭비는 시퀀스의 마지막 블록에서만 생겨 4% 미만으로 줄고, 남는 메모리만큼 더 많은 요청을 한 배치에 묶을 수 있습니다. 물리 블록의 참조 카운트와 쓰기 시 복사(copy-on-write)를 이용하면 여러 시퀀스가 같은 프롬프트 블록을 공유할 수 있습니다. 병렬 샘플링과 빔 서치에서는 이 공유로 메모리 사용이 최대 55% 줄고 처리량은 최대 2.2배 높아집니다.

vLLM의 현재 엔진 아키텍처인 V1은 스케줄러와 모델 실행기만 담당하는 엔진 코어 루프를 별도 프로세스로 떼어 내며, 기본 구성은 세 종류의 프로세스입니다. HTTP와 입력 처리를 맡는 API 서버, 스케줄러와 모델 실행기를 돌리는 엔진 코어, GPU마다 하나씩 붙어 GPU 메모리를 관리하는 워커입니다. 데이터 병렬 크기가 1보다 크면 DP 코디네이터 프로세스가 하나 더 붙습니다. 스케줄러는 프리필과 디코드를 따로 구분하지 않습니다. 스텝마다 `max_num_batched_tokens`만큼의 토큰 예산을 실행 중인 요청과 대기 요청으로 채웁니다. 캐시가 바닥나면 요청 하나를 선점해 블록을 비우고, 그 요청은 나중에 재계산으로 되살립니다.

Linux와 Python 3.11–3.14 환경의 NVIDIA GPU라면 uv로 wheel을 설치하면 됩니다. AMD ROCm에서는 추가 인덱스를 지정하고, Google TPU에서는 별도 패키지 `vllm-tpu`를 설치합니다. 이후 `LLM` 클래스로 오프라인 배치 추론을 돌리거나 `vllm serve`로 OpenAI 호환 서버를 띄울 수 있습니다.

## 연속 메모리 사전 할당이 낭비하는 GPU 메모리와 vLLM의 자리

LLM 서빙의 처리량을 결정하는 요인은 GPU 연산 능력이 아니라, 한 번에 몇 개의 요청을 배치로 묶을 수 있는지를 정하는 KV 캐시(KV cache) 관리 방식입니다. 자기회귀 Transformer는 프롬프트와 지금까지 만든 출력을 바탕으로 토큰을 하나씩 생성합니다. 이 과정을 종료 토큰이 나올 때까지 요청마다 반복하므로 워크로드가 메모리 대역폭에 묶이고 GPU 연산 자원은 다 쓰이지 못합니다. 여러 요청을 한 배치로 묶으면 처리량을 끌어올릴 수 있습니다. 다만 그러려면 요청마다 쓰는 메모리 공간을 효율적으로 관리해야 합니다.

![A100에서 13B 모델을 서빙할 때의 GPU 메모리 배치](/assets/images/vllm-in-practice-pagedattention-v1-engine-and-production-serving-1.png)

이 배치도는 세 영역의 크기를 비교해 보면 이해가 쉽습니다. 40GB A100에서 13B 파라미터 모델을 서빙하면 메모리의 약 65%를 모델 가중치(회색)가 차지하며, 이 영역은 서빙 내내 그대로 상주합니다. 30% 가까이는 요청의 동적 상태인 KV 캐시(빨간색)가 쓰고, 이 메모리는 요청 단위로 할당되고 해제됩니다. KV 캐시는 어텐션(attention)의 key·value 텐서로, 앞선 토큰들의 문맥을 담고 있어 다음 출력 토큰을 만들 때 쓰입니다. 남은 작은 부분(노란색)은 모델을 평가하는 동안 잠깐 생겼다 사라지는 활성화 텐서 몫입니다. 가중치 크기는 고정되어 있고 활성화가 차지하는 비중은 작습니다. 그래서 최대 배치 크기는 KV 캐시를 어떻게 관리하느냐에 달려 있고, 이 관리가 비효율적이면 배치 크기와 처리량이 크게 제한됩니다.

KV 캐시는 일반 딥러닝 텐서와 성질이 다릅니다. 모델이 토큰을 생성하는 동안 크기가 계속 늘었다 줄었다 하고, 요청이 얼마나 오래 살아 있을지, 최종 길이가 얼마가 될지도 미리 알 수 없습니다. 그런데 대부분의 딥러닝 프레임워크는 텐서를 연속 메모리에 두도록 요구합니다. 이 때문에 FasterTransformer나 Orca 같은 기존 서빙 시스템은 요청 하나의 KV 캐시를 연속 공간에 저장하고, 그 공간을 최대 길이(예: 2048 토큰)만큼 미리 잡아 둡니다. 이 방식에서 낭비가 생기는 경로는 세 가지입니다.

- **내부 단편화**: 실제 요청 길이가 최대 길이보다 훨씬 짧으면 예약한 공간의 대부분이 비어 있게 됩니다.
- **예약 낭비**: 실제 길이를 미리 안다고 해도 요청이 살아 있는 동안 청크 전체가 묶여 있습니다. 그래서 다른 짧은 요청은 지금 쓰이지 않는 부분조차 사용할 수 없습니다.
- **외부 단편화**: 요청마다 미리 잡는 크기가 달라서 청크 사이에 쓸 수 없는 틈이 생깁니다.

간단한 가상의 예로 계산해 보면 규모가 감이 옵니다. 2048 토큰을 예약했는데 프롬프트와 출력을 합쳐 300 토큰에서 생성이 끝나면, 예약 공간 중 실제 토큰 상태에 쓰인 비율은 약 15%이고 나머지 1748 토큰분은 요청이 끝날 때까지 아무도 쓰지 못합니다. PagedAttention 논문에서 기존 시스템을 프로파일링한 결과에도 비슷한 현상이 나타납니다. KV 캐시 메모리 가운데 실제 토큰 상태를 저장하는 데 쓰인 비율은 20.4%–38.2%에 그쳤습니다.

![기존 LLM 서빙 시스템별 평균 메모리 낭비 비율 분해](/assets/images/vllm-in-practice-pagedattention-v1-engine-and-production-serving-2.png)

이 막대그래프는 같은 실험에서 각 서빙 시스템의 KV 캐시 메모리가 평균적으로 얼마나 낭비되었는지를 항목별로 나눠 보여 줍니다. 시스템마다 막대에서 실제 토큰 상태에 쓰인 몫이 얼마나 작은지를 보면 됩니다. 연속 공간 방식에는 다른 문제도 있습니다. 병렬 샘플링이나 빔 서치(beam search)처럼 요청 하나가 여러 시퀀스를 만드는 경우, 시퀀스들의 KV 캐시가 서로 떨어진 연속 공간에 따로 저장됩니다. 그래서 겹치는 부분이 있어도 공유하지 못합니다.

vLLM은 바로 이 지점을 겨냥한 고처리량 분산 LLM 서빙 엔진입니다. 운영체제의 가상 메모리 페이징에서 착안한 PagedAttention 위에 만들어졌고, 블록 단위 메모리 관리와 선점형 요청 스케줄링을 PagedAttention에 맞춰 함께 설계했습니다. KV 캐시를 블록으로 나누고 비연속 메모리에 배치하는 구체적인 방식은 뒤의 PagedAttention과 블록 테이블 설명에서 다룹니다.

## OS 페이징을 KV 캐시에 옮긴 PagedAttention과 블록 테이블

PagedAttention은 시퀀스의 KV 캐시를 고정 토큰 수의 블록으로 나눕니다. 그래서 논리적으로 연속된 키·값을 물리적으로는 흩어진 메모리에 저장할 수 있고, 어텐션 커널은 계산 중에 필요한 블록을 찾아 가져옵니다. 프로세스가 연속된 주소 공간을 쓰는 것처럼 시퀀스는 연속된 논리 블록(logical block)을 봅니다. 이 논리 블록은 블록 테이블(block table)을 거쳐 메모리 여기저기에 있는 물리 블록(physical block)에 매핑되고, 물리 블록은 새 토큰이 생성될 때 필요한 만큼만 할당됩니다.

| 운영체제 가상 메모리 | PagedAttention |
|---|---|
| 페이지 | KV 블록 |
| 바이트 | 토큰 |
| 프로세스 | 시퀀스 |
| 페이지 테이블 | 블록 테이블 |

### 블록 단위로 쪼갠 어텐션 계산

블록이 메모리에서 붙어 있지 않아도 어텐션 계산은 블록마다 따로 진행할 수 있습니다. 블록 하나에 담기는 토큰 수를 KV 블록 크기 $$B$$라고 하고, $$j$$번째 키 블록을 $$K_{j}=(k_{(j-1)B+1},\ldots,k_{jB})$$, 값 블록을 $$V_{j}=(v_{(j-1)B+1},\ldots,v_{jB})$$로 두면 표준 어텐션은 블록 단위 계산으로 다시 쓸 수 있습니다.

$$
A_{ij}=\frac{\exp(q_{i}^{\top}K_{j}/\sqrt{d})}{\sum_{t=1}^{\lceil i/B\rceil}\exp(q_{i}^{\top}K_{t}\mathbf{1}/\sqrt{d})},\ o_{i}=\sum_{j=1}^{\lceil i/B\rceil}V_{j}A_{ij}^{\top}
$$

$$q_{i}$$는 $$i$$번째 토큰의 쿼리 벡터이고, $$d$$는 키 벡터의 차원으로 스케일링에 쓰입니다. $$A_{ij}=(a_{i,(j-1)B+1},\ldots,a_{i,jB})$$는 $$j$$번째 KV 블록에 대한 어텐션 점수 행 벡터입니다. $$\mathbf{1}$$은 블록 안의 점수를 모두 더하는 1 벡터입니다. $$\lceil i/B\rceil$$는 $$i$$번째 토큰까지를 덮는 블록 수입니다. 분모는 이 블록들 전체에서 소프트맥스 정규화를 하고, 출력 $$o_{i}$$는 블록별 값 벡터에 그 블록의 점수를 곱해 더한 값입니다. Transformer에서 토큰 하나는 레이어와 헤드마다 키·값 벡터를 따로 가집니다. 모든 벡터를 한 블록에 담는 설계와 헤드·레이어별로 블록과 블록 테이블을 따로 두는 설계 사이에 성능 차이는 없었고, 저자들은 구현이 쉬운 후자를 택했습니다.

![비연속 메모리에 흩어진 KV 블록을 커널이 블록별로 읽어 어텐션을 계산하는 과정](/assets/images/vllm-in-practice-pagedattention-v1-engine-and-production-serving-3.png)

이 그림에서는 쿼리 토큰 "forth"의 키·값이 세 블록에 나뉘어 있고, 세 블록이 물리 메모리에서 서로 떨어져 있습니다. 커널은 한 번에 블록 하나씩 처리합니다. 먼저 쿼리 벡터 $$q_{i}$$와 블록 0의 키 벡터("Four score and seven")를 곱해 $$A_{ij}$$를 구하고, 그 점수를 같은 블록의 값 벡터 $$V_{j}$$에 곱해 최종 출력 $$o_{i}$$에 누적합니다.

### 블록 테이블이 기록하는 것과 디코딩 중의 할당

메모리 관리는 두 계층이 나눠 맡습니다. 요청의 KV 캐시는 논리 KV 블록의 연속으로 표현되고, 새 토큰이 생길 때마다 왼쪽부터 채워집니다. GPU 워커에서는 block engine이 GPU DRAM의 연속된 덩어리를 하나 잡아 물리 KV 블록으로 잘라 둡니다. 스와핑(swapping)에 쓰려고 CPU RAM에도 같은 작업을 합니다. KV 블록 매니저는 요청마다 블록 테이블을 유지하며, 테이블 항목 하나에는 논리 블록에 대응하는 물리 블록과 그 블록에서 채워진 위치 수(#filled)가 기록됩니다. 논리 블록과 물리 블록을 분리했기 때문에 모든 위치를 미리 예약하지 않고도 KV 캐시를 동적으로 키울 수 있습니다.

![프롬프트 7토큰 요청의 논리 블록이 블록 테이블을 거쳐 물리 블록 7, 1, 3으로 변환되는 예](/assets/images/vllm-in-practice-pagedattention-v1-engine-and-production-serving-4.png)

이 예시는 블록 크기가 4인 경우이며, 블록 테이블의 #filled 값이 단계마다 어떻게 바뀌는지 따라가면 이해하기 쉽습니다.

1. vLLM은 처음부터 최대 생성 길이만큼 메모리를 예약하지 않고, 프롬프트 계산에 필요한 블록만 잡습니다. 프롬프트가 7토큰이므로 논리 블록 0과 1 두 개를 물리 블록 7과 1에 매핑합니다. 프리필 단계에서 앞 4토큰은 논리 블록 0에, 다음 3토큰은 논리 블록 1에 저장되고, 블록 1의 남은 한 칸은 이후 생성용으로 남겨 둡니다.
2. 첫 디코딩 스텝에서는 물리 블록 7과 1 위에서 PagedAttention으로 새 토큰을 만듭니다. 새 KV는 마지막 블록의 빈칸에 들어가고 #filled가 갱신됩니다.
3. 두 번째 디코딩 스텝에서는 마지막 논리 블록이 이미 차 있습니다. 그래서 새 논리 블록을 만들고 물리 블록 3을 할당한 뒤 그 매핑을 블록 테이블에 기록합니다.

![요청의 생성이 진행되며 논리 블록이 비연속 물리 블록에 차례로 매핑되는 애니메이션](https://vllm.ai/blog-assets/figures/annimation1.gif)

이 애니메이션에서는 연속된 논리 블록이 순서대로 채워지는 동안 물리 블록은 메모리 곳곳에서 하나씩 잡히는 모습을 볼 수 있습니다. 디코딩 반복마다 vLLM은 배치에 넣을 후보 시퀀스를 고르고, 새로 필요해진 논리 블록에 물리 블록을 할당합니다. 그다음 프롬프트 단계 요청의 모든 토큰과 생성 단계 요청의 최신 토큰을 하나의 시퀀스로 이어 모델에 넣습니다. 요청이 끝나면 그 블록은 해제되어 다른 요청의 KV 캐시를 담는 데 쓰입니다.

### 마지막 블록에만 남는 낭비

블록을 왼쪽부터 채우고 앞 블록이 모두 찼을 때만 새 물리 블록을 할당하므로, 요청 하나의 낭비는 블록 하나 안으로 제한됩니다. 실제로 낭비는 시퀀스의 마지막 블록에서만 생기며 4% 미만입니다. 덕분에 더 많은 시퀀스를 한 배치에 넣을 수 있어 GPU 활용도와 처리량이 오릅니다.

### 블록 크기 16이 기본값인 까닭

블록 크기에는 트레이드오프가 있습니다. 블록이 너무 작으면 KV 캐시를 읽고 처리할 때 GPU 병렬성을 충분히 활용하지 못합니다. 반대로 너무 크면 내부 단편화가 늘고 블록을 공유할 확률이 줄어듭니다. Alpaca 트레이스에서는 16과 32가 잘 동작했지만, 시퀀스가 블록보다 짧아지는 더 큰 크기에서는 성능이 크게 떨어졌습니다. 16은 대부분의 워크로드에서 GPU를 효율적으로 쓰기에 충분히 크면서 내부 단편화를 피할 만큼 작았고, 그래서 vLLM의 기본 블록 크기가 16으로 정해졌습니다.

### 비연속 접근을 감당하는 커널 최적화

기존 시스템은 PagedAttention이 만드는 메모리 접근 패턴을 효율적으로 지원하지 못합니다. 이 때문에 vLLM은 전용 GPU 커널을 세 가지 만들었습니다.

- **reshape와 블록 쓰기 융합**: 레이어마다 새 KV 캐시를 블록으로 나누고 블록 읽기에 맞는 레이아웃으로 바꾼 뒤 블록 테이블이 가리키는 위치에 저장합니다. 이 과정을 커널 하나로 합쳐 실행 오버헤드를 줄였습니다.
- **블록 읽기와 어텐션 융합**: FasterTransformer의 어텐션 커널을 고쳐 블록 테이블에 따라 KV를 읽으면서 바로 어텐션을 계산합니다. 메모리 접근을 병합하려고 블록마다 워프 하나를 배정하며, 한 배치 안에서 서로 다른 시퀀스 길이도 지원합니다.
- **fused block copy**: 쓰기 시 복사(copy-on-write)가 일으키는 블록 복사는 서로 떨어진 블록을 대상으로 할 수 있습니다. 이를 `cudaMemcpyAsync`로 처리하면 작은 전송이 수없이 반복되므로, 여러 블록의 복사를 커널 한 번에 묶었습니다.

이 블록 관리를 맡는 구현은 버전에 따라 바뀌어 왔습니다. v0.6.4에서는 v1 블록 매니저가 제거되고 `SelfAttnBlockSpaceManager`(블록 매니저 v2)가 기본이 되었습니다.

## 참조 카운트와 쓰기 시 복사로 이루어지는 블록 공유와 프리픽스 캐싱

블록 테이블에는 논리 블록과 물리 블록 사이에 간접 계층이 하나 있습니다. 이 계층 덕분에 여러 시퀀스가 같은 물리 블록을 가리킬 수 있고, 프롬프트처럼 내용이 같은 KV 캐시는 한 번만 저장하면 됩니다. 운영체제에서 여러 프로세스가 물리 페이지를 공유하듯, PagedAttention에서도 서로 다른 시퀀스가 각자의 논리 블록을 같은 물리 블록에 매핑해 블록을 공유합니다. 공유한 블록에 누군가 써야 할 때가 문제인데, PagedAttention은 물리 블록마다 참조 카운트를 추적하고 쓰기 시 복사를 구현해 공유를 안전하게 유지합니다.

![여러 출력을 샘플링하는 요청에서 출력 시퀀스들이 프롬프트 블록을 함께 참조하며 생성되는 과정](https://vllm.ai/blog-assets/figures/annimation3.gif)

애니메이션에서는 출력 시퀀스들이 프롬프트 부분에서 같은 물리 블록을 함께 가리키다가, 생성된 토큰이 서로 달라지는 지점부터 각자 블록을 따로 갖게 됩니다.

### 참조 카운트가 2인 블록에 처음 쓰는 샘플이 받는 복사본

병렬 샘플링에서는 요청 하나가 같은 입력 프롬프트로 여러 샘플을 만듭니다. 그래서 프롬프트의 KV 캐시도 공유할 수 있습니다. 프롬프트 단계에서는 프롬프트 상태를 한 벌만 저장합니다. 두 시퀀스의 논리 블록 0과 1이 모두 물리 블록 7과 1에 매핑되고, 두 물리 블록의 참조 카운트는 각각 2가 됩니다.

생성 단계에 들어서면 두 출력은 서로 다른 토큰을 샘플링하므로 KV 캐시를 따로 저장해야 합니다. 순서는 다음과 같습니다.

1. A1이 마지막 논리 블록(논리 블록 1)에 쓰려고 합니다. vLLM은 대응하는 물리 블록 1의 참조 카운트가 1보다 큰 것을 확인합니다.
2. 새 물리 블록 3을 할당하고, block engine에 물리 블록 1의 내용을 복사하라고 지시한 뒤, 물리 블록 1의 참조 카운트를 1로 낮춥니다. A1의 논리 블록 1은 이제 물리 블록 3을 가리킵니다.
3. 이어서 A2가 물리 블록 1에 쓸 때는 참조 카운트가 이미 1입니다. 그래서 복사하지 않고 새로 생성한 KV 캐시를 물리 블록 1에 바로 씁니다.

쓰기 시 복사는 이처럼 블록 단위로 일어나고, 운영체제가 프로세스를 fork할 때 쓰는 기법과 비슷합니다. 프롬프트 KV 캐시 가운데 마지막 논리 블록만 쓰기 시 복사로 관리되고 나머지는 모든 샘플이 그대로 공유합니다. 그래서 입력 프롬프트가 길수록 절약되는 메모리가 커집니다. 이때 생기는 블록 복사는 앞서 다룬 fused block copy 커널이 한 번에 묶어 처리합니다.

절약 규모를 감 잡기 위해 가상의 숫자로 계산해 보겠습니다. 블록 크기 16에서 500토큰 프롬프트로 샘플 4개를 만드는 경우입니다. 프롬프트는 블록 32개를 차지하며 마지막 블록에는 4토큰만 들어 있습니다. 공유하면 가득 찬 31개는 한 벌만 두고, 마지막 블록만 원본 1개와 복사본 3개로 늘어나 총 35개면 됩니다.

### 후보가 갈라지고 다시 모이는 빔 서치의 블록 트리

빔 서치는 병렬 샘플링보다 공유 범위가 넓습니다. 처음 프롬프트 블록뿐 아니라 후보 사이의 다른 블록도 공유하고, 디코딩이 진행되면서 공유 패턴이 계속 바뀝니다. 운영체제에서 fork가 겹쳐 만들어지는 프로세스 트리와 비슷한 구조입니다. 빔 폭 $$k=4$$인 예를 보면, 어느 반복 직전까지 각 후보는 가득 찬 논리 블록 4개를 썼습니다. 모든 후보가 블록 0(프롬프트)을 공유하고, 후보 3은 두 번째 블록에서 갈라졌으며, 후보 0–2는 앞 세 블록을 공유하다가 네 번째 블록에서 갈라졌습니다.

다음 반복에서 상위 4개 후보는 모두 원래 후보 1과 2에서 나옵니다. 원래 후보 0과 3은 상위에서 밀려나므로 그 논리 블록이 해제되고, 대응하는 물리 블록의 참조 카운트가 줄어듭니다. 참조 카운트가 0이 된 물리 블록 2, 4, 5, 8은 해제되고, 새 후보의 KV 캐시를 담을 블록 9–12가 새로 할당됩니다. 그 결과 공유 관계는 아래 표와 같이 정리됩니다.

| 공유 범위 | 공유하는 물리 블록 |
|---|---|
| 후보 0–3 전체 | 0, 1, 3 |
| 후보 0과 1 | 6 |
| 후보 2와 3 | 7 |
| 새 후보 각각 | 9–12 중 하나씩 새로 할당 |

기존 서빙 시스템이었다면 이 시점에 후보 3이 후보 2의 KV 캐시 대부분을 복사해야 생성을 이어 갈 수 있습니다. 후보 간 KV 복사가 자주 일어나는 구조입니다. vLLM에서는 서로 다른 빔 후보의 블록 대부분이 공유됩니다. 쓰기 시 복사는 새로 생성된 토큰이 기존 공유 블록 안에 들어갈 때만 적용되고, 복사량도 블록 하나 분량에 그칩니다. 이런 공유로 병렬 샘플링과 빔 서치의 메모리 사용량이 최대 55% 줄었고, 처리량은 최대 2.2배까지 향상되었습니다.

> **핵심:** 모델과 커널은 시퀀스마다 물리 블록 ID 목록만 받고, 시퀀스 간 공유 관계는 알지 못합니다. 공유와 해제는 모두 블록 테이블과 참조 카운트 수준에서 끝납니다.

이 분리 덕분에 디코딩 방식이 서로 다른 요청도 한 배치에 함께 넣을 수 있습니다. 탐욕 디코딩, 병렬 샘플링, 빔 서치 요청이 섞여 있어도 논리 블록을 물리 블록으로 바꾸는 공통 매핑 계층이 공유의 복잡성을 감춰 줍니다.

### 공유 라이브러리처럼 미리 잡아 두는 공유 프리픽스

같은 원리는 요청 사이에도 적용됩니다. 시스템 프롬프트처럼 많은 사용자 프롬프트가 같은 프리픽스로 시작하는 서비스에서는, 서비스 제공자가 미리 정의한 공유 프리픽스용 물리 블록을 예약해 둘 수 있습니다. 운영체제가 여러 프로세스에 공유 라이브러리를 제공하는 방식과 같습니다. 이 프리픽스로 시작하는 사용자 프롬프트는 자기 논리 블록을 캐시된 물리 블록에 매핑하기만 하면 되고, 마지막 블록은 쓰기 시 복사 대상으로 표시됩니다. 그러면 프롬프트 단계 계산은 사용자의 실제 입력 부분에서만 실행됩니다.

### V1에서 기본으로 켜지는 해시 기반 프리픽스 캐싱

운영 관점에서 보면, 공통 시스템 프롬프트가 긴 채팅 서비스나 같은 지시문을 반복하는 배치 작업일수록 프리픽스 캐싱의 이득이 커집니다.

## API 서버·엔진 코어·GPU 워커로 나뉜 V1 프로세스 아키텍처

V1에서 요청 하나는 세 종류의 프로세스를 차례로 지납니다. HTTP와 입력 처리를 맡는 API 서버, 스케줄러와 KV 캐시를 쥔 엔진 코어, GPU마다 하나씩 붙는 워커입니다. 데이터 병렬을 켤 때만 여기에 DP 코디네이터가 하나 더 붙습니다.

### 중앙 스케줄러 하나가 GPU 워커들을 지휘하는 원래 설계

PagedAttention 논문의 vLLM은 중앙 스케줄러 하나가 분산된 GPU 워커들의 실행을 조율하는 구조였습니다. KV 캐시 매니저는 이 스케줄러가 보내는 지시에 따라 GPU 워커의 물리 KV 캐시 메모리를 관리합니다. 아래 개요도는 중앙 스케줄러와 KV 캐시 매니저, GPU/CPU 블록 할당자, GPU 워커로 이루어진 구성을 보여 줍니다.

![중앙 스케줄러와 KV 캐시 매니저 아래에 블록 할당자와 여러 GPU 워커가 연결된 vLLM 시스템 개요도](/assets/images/vllm-in-practice-pagedattention-v1-engine-and-production-serving-5.png)

구현에서는 FastAPI 프론트엔드가 OpenAI API 인터페이스를 확장합니다. 스케줄러와 블록 매니저 같은 제어 계층은 Python으로 작성했고, PagedAttention처럼 핵심이 되는 연산은 커스텀 CUDA 커널로 만들었습니다. 워커 간 텐서 통신에는 NCCL을 씁니다. 

### CPU 작업을 실행 루프 밖으로 빼낸 엔진 코어

V1은 멀티프로세싱 구조를 AsyncLLM의 핵심부까지 넓혔습니다. 그 결과 스케줄러와 모델 실행기만 도는 격리된 엔진 코어 실행 루프가 생겼습니다. 토크나이징, 멀티모달 입력 처리, 디토크나이징, 요청 스트리밍처럼 CPU를 많이 쓰는 작업은 이 루프와 겹쳐 실행되고, 덕분에 모델 처리량을 최대로 끌어올릴 수 있습니다. 프로세스는 역할에 따라 네 종류로 나뉩니다.

- **API 서버 프로세스**: HTTP 요청(OpenAI 호환 API 등)을 받고, 토크나이징과 멀티모달 데이터 로딩 같은 입력 처리를 하며, 결과를 클라이언트에 스트리밍합니다. 엔진 코어와는 ZMQ 소켓으로 통신합니다. 각 API 서버는 모든 엔진 코어와 다대다로 연결되므로 어느 API 서버든 요청을 어느 엔진 코어로나 보낼 수 있습니다. 미디어 로딩에는 API 서버 프로세스마다 여러 CPU 스레드를 쓰며, 스레드 수는 `VLLM_MEDIA_LOADING_THREAD_COUNT`로 정하고 기본값은 8입니다.
- **엔진 코어 프로세스**: 스케줄러를 돌리고 KV 캐시를 관리하며 GPU 워커들의 모델 실행을 조율합니다. 요청을 계속 스케줄링해 워커에 작업을 넘기는 바쁜 루프(busy loop)가 이 프로세스 안에서 돕니다.
- **GPU 워커 프로세스**: GPU마다 전용 워커가 하나씩 있어 모델 가중치를 올리고, forward pass를 실행하고, GPU 메모리를 관리합니다. 워커는 자기를 소유한 엔진 코어와 통신합니다.
- **DP 코디네이터 프로세스**: `--data-parallel-size > 1`일 때만 하나 추가됩니다. DP rank 사이의 로드 밸런싱을 맡고, MoE 모델에서는 동기화된 forward pass를 조율합니다.

### TP·PP·DP 크기에서 바로 계산되는 프로세스 수

프로세스 수는 병렬화 설정에서 바로 계산됩니다. API 서버는 기본 1개이고, 데이터 병렬을 쓰면 DP 크기에 맞춰 자동으로 늘어납니다. 이 개수는 `--api-server-count` 플래그로 직접 정할 수도 있습니다. 엔진 코어는 DP rank마다 하나이므로 `--data-parallel-size 4`면 4개입니다. GPU 워커는 엔진 코어마다 `tensor_parallel_size x pipeline_parallel_size`개입니다. GPU가 $$N$$개, API 서버가 $$A$$개인 배포의 프로세스 구성을 표로 정리했습니다.

| 프로세스 종류 | 개수 | 역할 |
|---|---|---|
| API 서버 | `A` (기본 `DP`) | HTTP 요청과 입력 처리 |
| 엔진 코어 | `DP` (기본 1) | 스케줄러와 KV 캐시 관리 |
| GPU 워커 | `N` (= `DP x PP x TP`) | GPU당 하나, 모델 forward pass 실행 |
| DP 코디네이터 | `DP > 1`이면 1, 아니면 0 | DP rank 간 로드 밸런싱 |

전체 프로세스 수는 $$A + DP + N$$이고, $$DP > 1$$이면 코디네이터 몫으로 1을 더합니다.

GPU 4개를 쓰는 단일 노드 배포 `vllm serve -tp=4`는 API 서버 1개, 엔진 코어 1개, GPU 워커 4개로 모두 6개 프로세스입니다. 그림을 보면 API 서버와 엔진 코어가 하나씩 있고, 엔진 코어 아래에 워커 네 개가 매달려 있습니다.

![TP=4 배포에서 API 서버 1개와 엔진 코어 1개, GPU 워커 4개가 연결된 V1 프로세스 구성도](https://docs.vllm.ai/en/latest/assets/design/arch_overview/v1_process_architecture_tp4.png)

GPU 8개로 `vllm serve -tp=2 -dp=4`를 띄우면 API 서버 4개, 엔진 코어 4개, GPU 워커 8개에 DP 코디네이터 1개가 더해져 17개 프로세스가 됩니다. 이 구성의 그림에서는 엔진 코어마다 워커가 두 개씩 붙고, API 서버 네 개가 엔진 코어 네 개와 엇갈려 연결되며, 그 옆에 코디네이터가 따로 서 있습니다.

![TP=2, DP=4 배포에서 API 서버 4개와 엔진 코어 4개, 워커 8개, DP 코디네이터 1개로 이루어진 V1 프로세스 구성도](https://docs.vllm.ai/en/latest/assets/design/arch_overview/v1_process_architecture_tp2_dp4.png)

GPU 하나에서 기본 설정으로 띄우는 경우를 같은 식에 넣으면 API 서버 1개, 엔진 코어 1개, 워커 1개로 3개 프로세스가 나옵니다. 다음 함수는 앞의 표에 나온 식을 그대로 옮긴 예시로, vLLM API는 쓰지 않습니다.

```python
def count_processes(tp: int = 1, pp: int = 1, dp: int = 1, api_servers: int | None = None) -> dict:
    a = dp if api_servers is None else api_servers
    workers = dp * pp * tp
    coordinator = 1 if dp > 1 else 0
    return {
        "api_server": a,
        "engine_core": dp,
        "gpu_worker": workers,
        "dp_coordinator": coordinator,
        "total": a + dp + workers + coordinator,
    }

print(count_processes())             # opt-125m, GPU 1개
print(count_processes(tp=4))         # vllm serve -tp=4
print(count_processes(tp=2, dp=4))   # vllm serve -tp=2 -dp=4
```

세 호출의 합계는 각각 3, 6, 17입니다. 뒤의 두 값은 위에서 본 두 배포 예와 같습니다.

### 블록 테이블을 브로드캐스트하고 all-reduce로 맞추는 텐서 병렬 한 스텝

엔진 코어 하나에 워커가 여럿 붙는 텐서 병렬에서 한 스텝이 어떻게 도는지는 논문의 분산 실행 설계에 나와 있습니다. vLLM은 Megatron-LM 방식의 텐서 모델 병렬을 지원합니다. 선형 계층을 나눠 블록 단위 행렬 곱을 하고, 어텐션 연산자는 헤드 차원으로 쪼개서 각 프로세스가 헤드 일부만 맡는 SPMD(Single Program Multiple Data) 실행입니다. SPMD에서는 모든 워커가 같은 프로그램을 돌리되 각자 다른 샤드를 처리합니다. 모델을 나눠도 각 샤드는 같은 입력 토큰을 처리하므로 같은 위치의 KV 캐시가 필요합니다. 그래서 KV 캐시 매니저는 중앙 스케줄러 안에 하나만 두고, 모든 워커가 논리 블록에서 물리 블록으로 가는 매핑을 함께 씁니다. 워커들은 같은 물리 블록 ID를 쓰지만, 각자 실제로 저장하는 것은 자기가 맡은 어텐션 헤드 몫의 KV 캐시뿐입니다.

논문 설계에서 스텝 하나는 다음 순서로 진행됩니다.

1. 스케줄러가 배치 안의 요청마다 입력 토큰 ID와 블록 테이블을 담은 제어 메시지를 만듭니다.
2. 이 메시지를 모든 GPU 워커에 브로드캐스트합니다.
3. 워커들은 입력 토큰 ID로 모델을 실행하고, 어텐션 계층에서는 메시지에 담긴 블록 테이블을 따라 KV 캐시를 읽습니다.
4. 실행 중간 결과는 스케줄러를 거치지 않고 워커끼리 all-reduce로 동기화합니다.
5. 이번 반복에서 샘플링한 토큰을 스케줄러에 돌려줍니다.

메모리 관리 정보는 매 디코딩 반복이 시작될 때 입력과 함께 한 번에 전달됩니다. 그래서 워커끼리 메모리 관리를 위해 따로 동기화할 필요가 없습니다.

### 커널이 아니라 CPU 오버헤드에서 나온 V1의 처리량 향상

vLLM 블로그는 V1이 V0보다 처리량이 최대 1.7배 높다고 보고합니다. 비교 대상은 멀티스텝 스케줄링을 끈 V0입니다. 측정에는 ShareGPT 데이터셋과 Llama 3.1 8B, Llama 3.3 70B를 썼습니다. VisionArena에서 측정한 Qwen2-VL은 속도 향상 폭이 더 컸습니다. 두 버전의 커널은 거의 같으므로, 이 차이는 주로 CPU 오버헤드가 줄어든 데서 나옵니다. 앞서 본 엔진 코어 분리가 이 오버헤드를 겨냥한 설계입니다.

GPU 배포에서 CPU 코어를 산정할 때도 이 구조를 기준으로 삼으면 됩니다. API 서버마다 미디어 로딩 스레드가 따로 돌고 엔진 코어도 바쁜 루프로 코어를 점유하므로, DP를 키우면 GPU만큼 CPU 수요도 늘어납니다.

## 매 스텝 토큰 예산을 채우는 스케줄러와 재계산으로 되살아나는 선점

V1 스케줄러는 스텝마다 `max_num_batched_tokens`만큼의 토큰 예산을 실행 중인 요청과 대기 요청에 나눠 채웁니다. KV 캐시가 바닥나면 정책에 따라 고른 요청의 블록을 비우고, 그 요청은 나중에 재계산해서 되살립니다. 이 스케줄러는 앞서 본 엔진 코어 프로세스의 바쁜 루프 안에서 돌면서, 매 스텝 어떤 요청을 실행할지 결정합니다.

### prefill과 decode를 가르지 않는 토큰 예산

V1 스케줄러에는 prefill 단계와 decode 단계의 구분이 따로 없습니다. 사용자가 보낸 프롬프트 토큰과 모델이 생성한 출력 토큰을 똑같이 다룹니다. 요청마다 지금까지 계산을 마친 토큰 수를 기록해 두고, 스케줄러는 그 부족분을 메우도록 토큰을 배정합니다. 청크 프리필(chunked prefill), 앞서 본 프리픽스 캐싱(prefix caching), 추측 디코딩(speculative decoding)도 모두 이 하나의 루프에서 처리됩니다.

V0와 비교하면 차이가 분명합니다. vLLM 블로그의 "Inside vLLM" 글(2025-09-05)은 V1 스케줄러가 prefill 요청과 decode 요청을 같은 스텝에 섞을 수 있는 반면, V0 엔진은 한 번에 둘 중 한 종류만 처리했다고 설명합니다.

### 남은 예산에 맞춰 잘리는 청크 프리필

청크 프리필을 켜면 prefill 요청이 남은 `max_num_batched_tokens`에 맞춰 여러 조각으로 나뉩니다. 이 옵션은 `--enable-chunked-prefill`과 `--disable-chunked-prefill`로 켜고 끕니다. 긴 프롬프트가 들어와도 한 스텝의 예산을 혼자 다 차지하지 않습니다.

예를 들어 예산을 16토큰으로 잡았다고 하겠습니다. opt-125m 서버에서 요청 A와 B가 디코딩 중인데, 30토큰짜리 프롬프트 C가 새로 도착한 상황입니다. 아래 코드는 요청별 토큰 수 딕셔너리를 만드는 과정을 단순하게 흉내 낸 예시이며, vLLM API는 쓰지 않습니다.

```python
from dataclasses import dataclass

@dataclass
class Req:
    name: str
    prompt_len: int
    computed: int = 0

    def need(self) -> int:
        # 프롬프트가 남았으면 남은 만큼, 다 끝났으면 디코딩 1토큰
        return max(self.prompt_len - self.computed, 1)

def schedule_step(running: list[Req], waiting: list[Req], budget: int) -> dict[str, int]:
    plan: dict[str, int] = {}
    for req in running + waiting:
        if budget == 0:
            break
        n = min(req.need(), budget)
        plan[req.name] = n
        req.computed += n
        budget -= n
    for req in list(waiting):
        if req.name in plan:
            waiting.remove(req)
            running.append(req)
    return plan

running = [Req("A", 5, 5), Req("B", 8, 8)]
waiting = [Req("C", 30)]
for step in range(4):
    print(step + 1, schedule_step(running, waiting, budget=16))
```

첫 두 스텝의 배정은 `{'A': 1, 'B': 1, 'C': 14}`입니다. 세 번째 스텝에서 C는 남은 2토큰만 받고, 네 번째 스텝부터는 A, B와 똑같이 1토큰씩 디코딩합니다. 이 장난감 모델에는 prefill에서 decode로 넘어가는 분기가 없습니다. C가 단계를 옮겨 가는 과정은 계산한 토큰 수가 프롬프트 길이를 따라잡는 것뿐입니다.

### 도착 순서와 priority 값으로 갈리는 정책

정책은 `--scheduling-policy`로 고르며, 선택지는 `fcfs`와 `priority`이고 기본값은 `fcfs`입니다. 두 정책은 실행 순서를 정할 뿐 아니라 KV 캐시가 고갈됐을 때 누구를 내보낼지도 정합니다.

| 정책 | 처리 순서 | KV 캐시 고갈 시 선점 대상 | 선점 후 대기 큐 위치 |
|---|---|---|---|
| `fcfs` (기본) | 도착 순서 | 가장 나중에 들어온 실행 중 요청 | 큐의 맨 앞 |
| `priority` | `priority` 값이 낮을수록 먼저, 동률이면 도착 시간 | `priority` 값이 가장 크고, 그중 가장 늦게 도착한 요청 | `(priority, arrival_time)` 순서에 맞는 자리 |

`priority` 정책에서는 요청마다 정수 `priority` 필드가 붙습니다. 숫자가 클수록 우선순위가 낮다는 점을 착각하기 쉬우므로, 이 정책을 쓴다면 클라이언트 쪽 규약부터 맞춰 두는 편이 안전합니다.

PagedAttention 논문도 모든 요청에 FCFS를 적용했습니다. 선점할 때는 먼저 도착한 요청을 먼저 서비스하고, 가장 늦게 온 요청부터 내보냈습니다. 논문이 내세운 목적은 공정성과 기아(starvation) 방지입니다.

### 블록을 통째로 비우고 다시 계산하는 선점

요청에 새 KV 캐시 블록을 할당할 수 없으면, 스케줄러는 정책이 고른 실행 중 요청을 하나씩 선점(preemption)합니다. 블록이 충분히 확보될 때까지 이 과정을 반복합니다. 선점된 요청에는 다음 일이 차례로 일어납니다.

1. 그 요청의 KV 캐시 블록이 해제됩니다.
2. 진행 상태가 초기화되고, 앞의 표에 적은 자리로 대기 큐에 돌아갑니다.
3. 다시 투입될 때는 CPU 메모리에서 블록을 되가져오지 않고 재계산(recomputation)합니다. 이때 프리픽스 캐싱이나 KV 커넥터가 진행분의 일부를 곧바로 복원할 수 있습니다.

한 시퀀스의 블록은 언제나 함께 접근되므로, 논문은 시퀀스의 블록을 모두 축출하거나 하나도 축출하지 않는 all-or-nothing 정책을 택했습니다.

### 스와핑과 재계산을 비교한 논문의 측정

논문은 축출한 블록을 복구하는 방법으로 두 가지를 다뤘습니다. 스와핑(swapping)은 블록을 CPU RAM으로 복사해 두는 방식입니다. CPU로 옮기는 블록 수는 GPU의 전체 물리 블록 수를 넘지 않으므로, 스왑 공간은 KV 캐시에 할당한 GPU 메모리 크기로 제한됩니다. 재계산은 디코딩 중에 생성한 토큰을 원래 프롬프트 뒤에 이어 붙여 새 프롬프트로 만들고, 모든 위치의 KV를 한 번의 프롬프트 단계에서 다시 만듭니다.

블록 크기가 작을 때는 스와핑의 오버헤드가 재계산보다 훨씬 컸습니다. 작은 CPU-GPU 전송이 많아져 유효 PCIe 대역폭이 제한됐기 때문입니다. 블록 크기가 16에서 64 사이일 때는 두 방식의 종단 성능이 비슷했습니다.

v0.6.4에서는 선점 방식을 재계산(`recompute`)과 블록 스왑(`swap`) 중에서 고를 수 있었습니다. 현재 V1의 선점은 앞서 설명한 대로 재계산만 씁니다.

### GPU 사용률의 공백을 줄이는 비동기 스케줄링

`--async-scheduling`은 비동기 스케줄링을 제어하며, `--no-async-scheduling`으로 끌 수 있습니다. 비동기 스케줄링은 GPU 사용률에 생기는 공백을 줄여서 지연 시간과 처리량을 함께 개선합니다. 스텝마다 스케줄링 결정을 내리는 CPU 작업이 GPU 실행 사이에 끼어드는 구조에서는, 이 공백이 곧 GPU가 놀고 있는 시간입니다. 앞서 본 V1의 처리량 향상이 주로 CPU 오버헤드 감소에서 나왔으므로, 이 옵션도 같은 방향의 설정으로 볼 수 있습니다.

## wheel·소스·Docker 설치와 OPT-125M 첫 실행

NVIDIA GPU에서는 uv로 사전 빌드 wheel을 설치하고 커널을 내려받기만 하면, `LLM` 클래스로 오프라인 배치 추론을 돌리고 `vllm serve`로 OpenAI 호환 서버를 띄울 수 있습니다. CUDA 버전이 다르거나 기존 PyTorch를 그대로 써야 하면 소스에서 빌드하고, 공식 Docker 이미지로 OpenAI 호환 서버를 바로 띄울 수도 있습니다.

### Linux와 Python 3.11–3.14를 전제로 하는 지원 플랫폼

vLLM이 요구하는 OS는 Linux이고, Python은 3.11부터 3.14까지 지원합니다. Windows는 네이티브로 지원하지 않으므로 WSL에 호환되는 Linux 배포판을 올려서 씁니다. GPU 플랫폼은 NVIDIA CUDA, AMD ROCm, Intel XPU, 그리고 vLLM-Metal을 거치는 Apple Silicon입니다. CPU 백엔드는 Intel/AMD x86, ARM AArch64, Apple silicon, IBM Z(S390X)에서 돌아갑니다. CPU 백엔드 설정은 CPU 백엔드를 다루는 부분에서 따로 정리합니다. 메인 저장소 밖에서 관리하는 서드파티 하드웨어 플러그인도 지원하며, 이 플러그인들은 Hardware-Pluggable RFC를 따릅니다.

### uv로 까는 사전 빌드 wheel과 커널 내려받기

NVIDIA GPU에서 가장 짧은 경로는 uv로 새 가상환경을 만들고 wheel을 설치한 다음 커널을 받는 순서입니다.

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
vllm download-kernels
```

`--torch-backend=auto`를 주면 uv가 설치된 CUDA 드라이버 버전을 확인하고 그에 맞는 PyTorch 인덱스를 고릅니다. 환경변수 `UV_TORCH_BACKEND=auto`도 효과가 같습니다. `vllm download-kernels`는 설치된 FlashInfer와 CUDA 버전에 맞게 미리 컴파일된 FlashInfer 커널을 받아 옵니다. 이 단계를 생략하면 vLLM이 시작할 때 커널을 내려받아 직접 컴파일합니다. 이 과정은 Hopper 이후 GPU, 특히 Blackwell에서 수 분이 더 걸리고 경고 로그를 남깁니다. 커널은 vLLM을 설치하거나 업그레이드할 때마다 같은 Python 환경에서 다시 받아야 하며, 커스텀 컨테이너 이미지를 만들 때도 마찬가지입니다.

conda를 쓰는 팀이라면 conda 환경을 만들고 그 안에 `pip install --upgrade uv`로 uv를 설치한 뒤 같은 `uv pip install` 명령을 실행합니다. 다만 conda로 설치한 PyTorch는 NCCL을 정적으로 링크하기 때문에 vLLM이 NCCL을 쓸 때 문제가 생길 수 있습니다. vLLM이 컴파일한 CUDA 커널은 다른 CUDA 버전이나 PyTorch 버전과 바이너리 호환이 되지 않습니다. PyTorch 버전이 같아도 빌드 설정이 다르면 호환되지 않으므로 새 환경에 설치하기를 권장합니다.

GPU가 NVIDIA가 아니면 명령이 조금 바뀝니다. AMD ROCm에서는 `uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/`로 추가 인덱스를 지정하며, 이 경로가 지원하는 조합은 Python 3.12, ROCm 7.0, `glibc >= 2.35`입니다. Google TPU에서는 별도 패키지인 `vllm-tpu`를 설치합니다.

### `LLM` 클래스로 돌리는 OPT-125M 오프라인 배치

설치가 끝나면 서버를 띄우지 않고 프롬프트 목록을 한 번에 생성할 수 있습니다. 앞서 정한 예제 모델로 짠 스크립트는 이렇습니다.

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="facebook/opt-125m")

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

`LLM` 생성자가 엔진과 모델을 초기화합니다. `llm.generate`는 네 프롬프트를 엔진의 대기 큐에 넣고 엔진을 실행한 다음, 출력 토큰을 담은 `RequestOutput` 객체 리스트를 돌려줍니다.

샘플링 기본값은 주의해서 봐야 합니다. Hugging Face 모델 저장소에 `generation_config.json`이 있으면 vLLM은 기본으로 그 파일의 권장값을 적용합니다. vLLM 자체 기본값을 쓰려면 `LLM`을 만들 때 `generation_config="vllm"`을 넘깁니다. `llm.generate`는 채팅 템플릿을 적용하지 않습니다. Instruct 모델이나 Chat 모델에는 템플릿을 직접 적용하거나, OpenAI `client.chat.completions`와 같은 형식의 메시지 리스트를 `llm.chat`에 넘깁니다.

### `vllm serve`로 띄우는 OpenAI 호환 서버

같은 모델을 HTTP로 서빙하는 명령은 한 줄입니다.

```bash
vllm serve facebook/opt-125m
```

아래 예제의 클라이언트는 `http://localhost:8000`으로 서버에 요청합니다. 한 서버는 한 번에 모델 하나만 호스팅하고, 모델 목록(`/v1/models`), Chat Completions, Completions 같은 엔드포인트를 구현합니다. 서버도 `generation_config.json`을 기본으로 적용하므로, 끄려면 띄울 때 `--generation-config vllm`을 넘깁니다.

Completions 요청은 OpenAI API와 같은 형식으로 보냅니다.

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "facebook/opt-125m",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```

`openai` Python 패키지에서는 API 키와 API base를 vLLM 서버에 맞게 바꿉니다.

```python
from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
completion = client.completions.create(
    model="facebook/opt-125m",
    prompt="San Francisco is a",
)
print("Completion result:", completion)
```

Chat Completions 엔드포인트는 기본으로 토크나이저에 저장된 채팅 템플릿을 씁니다. 채팅 요청은 Instruct 모델로 확인하는 편이 적절합니다. 서버를 `vllm serve Qwen/Qwen2.5-1.5B-Instruct`로 띄웠다면 요청의 `model` 값도 같은 이름으로 맞춰서 보냅니다.

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"}
        ]
    }'
```

외부에 서버를 노출한다면 인증부터 켭니다. `--api-key` 인자나 환경변수 `VLLM_API_KEY`를 주면 서버가 요청 헤더의 API 키를 검사합니다. `--api-key` 뒤에 키를 여러 개 넘기면 그중 어느 키든 받아들이므로 키 교체에 활용할 수 있습니다. 인증을 켰다면 위 클라이언트의 `openai_api_key`도 `"EMPTY"` 대신 실제 키로 바꿔야 합니다.

어텐션 백엔드는 시스템과 모델에 맞춰 자동으로 선택됩니다. 직접 고르려면 `--attention-backend`를 쓰며, NVIDIA CUDA에서 고를 수 있는 값에는 `FLASH_ATTN`과 `FLASHINFER`가 있습니다. 메모리 비율과 병렬화를 정하는 나머지 엔진 인자는 엔진 인자를 다루는 부분에서 정리합니다.

### Python만 고칠 때와 커널까지 고칠 때로 갈리는 소스 빌드

CUDA 버전이 다르거나 기존 PyTorch 설치를 그대로 써야 한다면 소스에서 빌드해야 합니다. Python 코드만 고치는 경우에는 컴파일을 건너뛰고, 사전 컴파일된 wheel을 재사용하는 editable 설치를 씁니다.

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 uv pip install --editable . --torch-backend=auto
vllm download-kernels
```

C++이나 커널 코드를 고쳤다면 이 방식은 쓸 수 없습니다. 쓰면 라이브러리를 찾을 수 없다거나 undefined symbol이라는 import 오류가 납니다. 특정 wheel을 강제로 쓰려면 `VLLM_PRECOMPILED_WHEEL_LOCATION`에 정확한 URL이나 로컬 경로를 지정합니다. 이 변수를 주면 다른 탐색 로직은 모두 건너뜁니다. 전체 빌드는 클론한 디렉터리에서 `uv pip install -e . --torch-backend=auto`로 하며, GCC/G++ 13 이상이 필요합니다. 빌드를 반복한다면 `ccache`를 PATH에 두면 빌드 시스템이 자동으로 사용합니다. `pip install -e .`와 함께 쓸 때는 `CCACHE_NOHASHDIR="true" pip install --no-build-isolation -e .`로 실행합니다.

### 엔진 인자를 태그 뒤에 붙이는 Docker 이미지

NVIDIA CUDA용 공식 이미지 `vllm/vllm-openai`는 OpenAI 호환 서버를 바로 실행하는 이미지입니다. 엔진 인자는 이미지 태그 뒤에 붙입니다.

```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model facebook/opt-125m
```

`--ipc=host`나 `--shm-size` 플래그를 주면 컨테이너가 호스트의 공유 메모리에 접근할 수 있습니다. vLLM이 쓰는 PyTorch가 프로세스 사이, 특히 텐서 병렬 추론에서 공유 메모리로 데이터를 주고받기 때문입니다. 선택적 의존성은 라이선스 문제를 피하려고 이미지에 넣지 않았으므로, 필요하면 이 이미지를 베이스로 레이어를 하나 더 얹습니다. 이미지를 직접 빌드할 때는 저장소의 `docker/Dockerfile`에서 `vllm-openai` 타깃을 지정합니다.

```bash
DOCKER_BUILDKIT=1 docker build . \
    --target vllm-openai \
    --tag vllm/vllm-openai \
    --file docker/Dockerfile
```

C++이나 CUDA 커널을 건드리지 않았다면 `--build-arg VLLM_USE_PRECOMPILED="1"`로 사전 컴파일 wheel을 써서 빌드 시간을 크게 줄일 수 있습니다. 특정 커밋의 wheel이 필요하면 `--build-arg VLLM_PRECOMPILED_WHEEL_COMMIT=<commit_hash>`를 추가합니다.

## 메모리 한도와 병렬화를 결정하는 엔진 인자

엔진 인자는 하나의 집합이어서, 오프라인 추론에서는 `LLM` 클래스의 인자로, 온라인 서빙에서는 `vllm serve`의 인자로 똑같이 넘깁니다. 엔진 인자 클래스인 `EngineArgs`와 `AsyncEngineArgs`는 `vllm.config`의 설정 클래스들을 합쳐 놓은 것입니다. 그래서 타입, 기본값, docstring은 이 설정 클래스들을 기준으로 봅니다.

### 인스턴스마다 따로 잡히는 GPU 메모리 비율

`--gpu-memory-utilization`은 모델 실행기가 쓸 GPU 메모리의 비율을 0에서 1 사이 값으로 정하며, 기본값은 `0.92`입니다. 이 한도는 현재 vLLM 인스턴스에만 적용됩니다. 같은 GPU에서 다른 인스턴스가 돌고 있는지는 상관이 없으므로, 한 GPU에 인스턴스 두 개를 올린다면 각각 `0.5`로 설정합니다. v0.6.4에서는 이 값이 전역 한도였습니다. vLLM이 시작하기 전에 GPU 메모리의 50%가 이미 쓰이고 있는데 0.9를 주면, 모델 실행기에는 40%만 할당되었습니다. 예전 설정을 옮겨 올 때는 이 차이 때문에 값을 다시 계산해야 합니다.

블록 하나가 담는 토큰 수는 `--block-size`로 정하며, 앞서 본 블록 크기 트레이드오프가 이 값에 그대로 적용됩니다. 메모리 부족이 나면 이 비율을 낮추라고 Unsloth 문서는 권합니다. 같은 문서는 float8 KV 캐시를 쓰면 메모리 사용이 50% 줄어든다고 보고합니다.

### 컨텍스트 길이·양자화·실행 모드를 정하는 모델 인자

`--max-model-len`은 프롬프트와 출력을 합친 컨텍스트 길이입니다. 지정하지 않으면 모델 설정에서 자동으로 가져옵니다. 값은 k/m/g/K/M/G를 붙인 사람이 읽기 쉬운 형식으로 쓸 수 있습니다. `-1`이나 `auto`를 주면 GPU 메모리에 들어가는 최대 길이를 고릅니다. 모델의 최대 컨텍스트가 들어가면 그 값을 쓰고, 들어가지 않으면 수용할 수 있는 가장 긴 길이를 찾습니다.

`--quantization`(`-q`)은 가중치 양자화 방법을 지정합니다. 값이 `None`이면 먼저 모델 설정 파일의 `quantization_config`를 확인합니다. 그것도 `None`이면 가중치가 양자화되지 않았다고 보고 `dtype`으로 가중치의 데이터 타입을 정합니다. 따라서 사전 양자화된 체크포인트라면 이 인자를 따로 줄 필요가 없습니다. `--enforce-eager`의 기본값은 `False`이며, 이때는 CUDA 그래프와 eager 실행을 섞어 씁니다. 켜면 `torch.compile`과 CUDA 그래프가 모두 꺼지며, `-cc.mode=none -cc.cudagraph_mode=none`과 효과가 같습니다.

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="facebook/opt-125m",
    gpu_memory_utilization=0.5,
    max_model_len=1024,
    enforce_eager=True,
)
outputs = llm.generate(
    ["Hello, my name is"],
    SamplingParams(temperature=0.8, top_p=0.95),
)
print(outputs[0].outputs[0].text)
```

서버 쪽에서는 같은 설정을 플래그로 넘깁니다.

```bash
vllm serve facebook/opt-125m \
    --gpu-memory-utilization 0.5 \
    --max-model-len 1K \
    --enforce-eager
```

### 스텝당 처리량을 묶는 스케줄러 인자

`--max-num-seqs`는 한 반복에서 처리할 최대 시퀀스 수이고, `--max-num-batched-tokens`는 한 반복에서 처리할 최대 토큰 수입니다. 앞서 본 토큰 예산 스케줄러가 매 스텝 채우는 예산이 바로 이 값입니다. 두 인자의 문서상 기본값은 테스트 편의를 위한 값입니다. 실제 사용에서는 `EngineArgs.create_engine_config`에서 값이 정해집니다.

### 배포 형태를 정하는 TP·PP·DP 크기

`--tensor-parallel-size`(`-tp`)는 텐서 병렬 그룹 수이고, `--data-parallel-size`(`-dp`)는 데이터 병렬 그룹 수입니다. 둘 다 기본값은 `1`입니다. MoE 모델에서는 텐서·프리필 컨텍스트·데이터 병렬 크기를 곱한 값에 따라 MoE 레이어가 샤딩됩니다. 이 크기들이 프로세스를 몇 개 띄우는지는 앞서 본 `-tp=2 -dp=4` 사례의 17개 프로세스 계산을 따릅니다.

워커를 어디서 돌릴지는 `--distributed-executor-backend`가 정합니다. 선택지는 `external_launcher`, `mp`, `ray`, `uni`입니다. `pipeline_parallel_size`와 `tensor_parallel_size`의 곱이 사용 가능한 GPU 수 이하이면 단일 호스트에 머물도록 `mp`(멀티프로세싱)를 쓰고, 그렇지 않으면 오류가 납니다. `mp`를 쓰려면 `nnodes`도 설정해야 하고, `ray`는 이 인자를 직접 `ray`로 지정해야 합니다. TPU 플랫폼은 분산 추론에 Ray만 지원합니다. 여러 노드에 걸칠 때는 Unsloth 문서의 안내를 참고할 만합니다. 이 문서는 TP를 노드 안에서, PP를 노드 사이에서 쓰고 PP를 노드 수로 맞추라고 권합니다.

## KV 캐시 공간과 NUMA 바인딩이 성능을 가르는 CPU 백엔드

CPU 백엔드에서는 엔진 인자보다 환경변수 두 개가 먼저 성능을 좌우합니다. KV 캐시 크기를 정하는 `VLLM_CPU_KVCACHE_SPACE`와 OpenMP 스레드를 코어에 묶는 `VLLM_CPU_OMP_THREADS_BIND`입니다. CPU 백엔드가 지원하는 아키텍처는 앞서 설치 과정에서 다뤘습니다. x86에서는 CPU 플래그로 `avx512f`를 권장하며, `avx2`만 있으면 일부 기능이 제한됩니다.

### GiB 단위로 직접 잡는 KV 캐시와 NUMA 노드 한도

`VLLM_CPU_KVCACHE_SPACE`는 KV 캐시 크기를 GiB 단위로 지정합니다. 값을 크게 잡을수록 더 많은 요청을 병렬로 처리합니다. 기본값 `0`은 이 변수로 KV 캐시 크기를 덮어쓰지 않는다는 뜻입니다.

상한은 머신 전체 메모리가 아니라 NUMA 노드 하나의 용량입니다. TP rank 하나가 쓰는 메모리는 가중치 샤드 크기에 `VLLM_CPU_KVCACHE_SPACE`를 더한 값입니다. 이 합이 NUMA 노드 하나의 용량을 넘으면 TP 워커가 메모리 부족으로 `exitcode 9`를 내며 종료됩니다. 노드당 메모리가 64GiB인 서버를 가정해 계산해 보면, 가중치 샤드가 30GiB일 때 KV 캐시를 40GiB로 잡으면 합이 70GiB가 되어 노드 용량을 넘습니다. 이 변수는 레거시 설정이며, 새로 설정할 때는 `--kv-cache-memory-bytes`를 씁니다.

### 프레임워크 몫을 남기는 스레드 바인딩

`VLLM_CPU_OMP_THREADS_BIND`에 넣을 수 있는 값은 셋입니다. CPU id 목록, 기본값인 `auto`, 그리고 코어 바인딩을 끄고 사용자가 정한 OpenMP 변수를 따르는 `nobind`입니다. 텐서 병렬을 켜면 앞서 본 것처럼 `|`로 rank마다 코어 구간을 나눕니다. 앞 구간은 rank0의 OpenMP 스레드가, 뒤 구간은 rank1의 스레드가 씁니다. `auto`는 각 rank의 스레드를 NUMA 노드별 코어에 묶습니다. `nobind`에서는 표준 `OMP_NUM_THREADS`가 스레드 수를 정합니다.

온라인 서빙에서는 CPU 과다 할당을 피하려고 서빙 프레임워크용으로 코어 1–2개를 남겨 두길 권장합니다. 마지막 코어 하나를 프레임워크 몫으로 남기고 나머지 코어에 추론 스레드를 묶으면 아래 명령이 됩니다.

```bash
export VLLM_CPU_KVCACHE_SPACE=40
export VLLM_CPU_OMP_THREADS_BIND=0-30
vllm serve facebook/opt-125m --dtype=bfloat16
```

코어 번호를 직접 고르지 않고 `auto`를 유지하려면, OpenMP에 쓰지 않을 코어 수를 `VLLM_CPU_NUM_OF_RESERVED_CPU`로 정합니다. 이 변수는 `auto`일 때만 적용됩니다. 지정하지 않으면 x86, ARM, RISC-V는 노드 한 대에서 실행되는 rank(워커 프로세스)의 수인 `local_world_size`만큼 예약하고, PowerPC와 S390X는 1개를 예약합니다. `world_size == 1`일 때는 프런트엔드 프로세스용으로 1개를 직접 예약하길 권장합니다.

```bash
export VLLM_CPU_KVCACHE_SPACE=40
export VLLM_CPU_NUM_OF_RESERVED_CPU=1
vllm serve facebook/opt-125m --dtype=bfloat16
```

대부분의 경우에는 `auto`를 권장합니다. 성능이 기대에 못 미치거나 바인딩이 예상과 다르게 잡힐 때 수동 지정을 시도합니다. 하이퍼스레딩이 켜진 서버에서는 `lscpu -e`로 논리 코어와 물리 코어의 대응을 확인합니다. 그다음 물리 코어 하나당 논리 코어 하나만 쓰도록 구간을 고릅니다. 논리 코어 두 개가 물리 코어 하나를 공유하는 서버라면, 각 물리 코어의 논리 코어 짝 가운데 한쪽 묶음에만 스레드를 묶습니다. 바인딩이 실제로 적용됐는지는 벤치마크를 돌리면서 `htop`에서 코어 사용률을 보고 확인합니다.

### 한 rank는 한 NUMA 노드에

멀티 소켓 NUMA 머신에서 TP나 PP를 켜면 NUMA 노드 하나가 rank 하나로 취급됩니다. 따라서 한 rank의 코어는 모두 같은 노드에 두어야 노드를 넘나드는 메모리 접근을 피합니다. 소켓과 메모리 노드가 충분하면 DP, TP, PP를 함께 쓰길 권장합니다.

### CPU에서 고를 dtype과 양자화

CPU 백엔드는 따로 지정하지 않으면 모델 설정에 적힌 `dtype`을 그대로 씁니다. 다만 torch CPU의 float16 지원이 불안정하므로, 성능이나 정확도에 문제가 있으면 `dtype=bfloat16`을 명시하라고 권장합니다. AMD Zen의 `ZenCpuPlatform`에서는 float16에 AVX512-FP16이 필요하며, 이 명령어는 Venice(6세대 EPYC)부터 지원됩니다. Turin(5세대 EPYC) 이하는 `bfloat16`과 `float32`만 받고, float16으로 선언된 모델은 로드할 때 `bfloat16`으로 자동 다운캐스트됩니다.

양자화 인자가 `None`이면 모델 설정의 `quantization_config`를 먼저 봅니다. CPU 백엔드가 지원하는 방식을 아키텍처별로 정리하면 아래 표가 됩니다.

| 양자화 방식 | 지원 아키텍처 |
|---|---|
| AWQ | x86, s390x, Power |
| GPTQ | x86, s390x, Power |
| compressed-tensor INT8 W8A8 | x86, s390x |

### GPU와 다른 블록 크기와 배치 기본값

CPU 문서는 배치 한도의 기본값도 따로 밝힙니다. `--max-num-batched-tokens`의 오프라인·온라인 기본값은 앞서 소개한 `world_size` 배수 값입니다. `--max-num-seqs`는 오프라인에서 `256 * world_size`, 온라인에서 `128 * world_size`입니다. 엔진 인자 설명에 적힌 최대 배치 토큰 수의 기본값은 테스트 편의를 위한 값이고, 실제 값은 `EngineArgs.create_engine_config`에서 정해집니다. `--max-num-batched-tokens`는 첫 토큰 성능에, `--max-num-seqs`는 출력 토큰 성능에 더 크게 영향을 줍니다. 배치가 크면 처리량이 오르고 작으면 지연이 줄어들므로, 기본값에서 출발해 조정하라고 안내합니다.

### 컨테이너에서 막히는 NUMA 시스템콜

Docker 같은 컨테이너에서는 런타임의 기본 seccomp·capability 설정이 vLLM이 쓰는 `get_mempolicy`, `migrate_pages` 같은 NUMA 시스템콜을 막습니다. 그러면 `get_mempolicy: Operation not permitted` 경고가 뜰 수 있습니다. 기능에는 영향이 없지만, NUMA 메모리 바인딩과 마이그레이션 최적화가 적용되지 않아 성능이 떨어질 수 있습니다. 최소 권한으로 이 최적화를 켜려면 다음 두 옵션을 더합니다. `SYS_NICE`는 `get_mempolicy`의 EPERM 문제를 풀고, `seccomp=unconfined`는 `migrate_pages`를 허용합니다.

```bash
# excerpt: docker run에 더하는 NUMA 관련 옵션
docker run ... --cap-add SYS_NICE --security-opt seccomp=unconfined ...
```

seccomp를 통째로 해제하기 어려운 환경이라면, 기본 프로필을 바탕으로 `migrate_pages`만 `SCMP_ACT_ALLOW` 목록에 추가한 프로필을 씁니다. `--privileged=true`도 동작은 하지만 권한 범위가 넓어 일반적으로 권장하지 않습니다. Kubernetes에서는 워크로드 YAML의 `securityContext`로 같은 효과를 냅니다.

```yaml
securityContext:
  seccompProfile:
    type: Unconfined
  capabilities:
    add:
      - SYS_NICE
```

## values.yaml 하나로 갈아 끼우는 Helm 배포와 S3 모델 다운로드

vLLM 저장소의 Helm 차트를 쓰면 같은 서빙 구조를 그대로 두고 `values.yaml` 값만 바꿔 Kubernetes에 배포합니다. Helm은 Kubernetes 패키지 매니저입니다. 변수 값을 덮어쓰는 방식으로, 같은 구조를 설정만 달리해 여러 네임스페이스에 올릴 수 있습니다. 차트에는 배포, 오토스케일링, 리소스 관리 같은 구성이 들어 있습니다.

사전에 준비할 것은 넷입니다. 실행 중인 Kubernetes 클러스터, NVIDIA `k8s-device-plugin`, 클러스터에서 쓸 수 있는 GPU 자원, 그리고 자동 모델 다운로드를 쓸 경우 가중치를 담은 S3 버킷입니다. 차트 디렉터리에는 `Chart.yaml`, 기본값을 담은 `values.yaml`, 이 값을 검증하는 `values.schema.json`이 있습니다. `templates` 아래에는 configmap, deployment, hpa, job, poddisruptionbudget, pvc, secrets, service, custom-objects 템플릿이 있습니다.

![vLLM Helm 배포 아키텍처 구성도](https://docs.vllm.ai/en/stable/assets/deployment/architecture_helm_deployment.png)

위 그림은 vLLM의 Helm 배포 아키텍처를 나타냅니다.

### 릴리스 하나의 설치와 제거

차트는 `examples/deployment/chart-helm`에 있습니다. 아래 명령은 릴리스 이름 `test-vllm`으로 설치하면서, S3 접속 정보를 `--set`으로 시크릿에 넘깁니다.

```bash
helm upgrade --install --create-namespace \
  --namespace=ns-vllm test-vllm . \
  -f values.yaml \
  --set secrets.s3endpoint=$ACCESS_POINT \
  --set secrets.s3bucketname=$BUCKET \
  --set secrets.s3accesskeyid=$ACCESS_KEY \
  --set secrets.s3accesskey=$SECRET_KEY
```

제거는 `helm uninstall test-vllm --namespace=ns-vllm` 한 줄입니다. 이 명령은 영구 볼륨까지 포함해 차트와 연결된 Kubernetes 구성요소를 모두 지우고 릴리스를 삭제합니다. 내려받아 둔 모델 가중치도 함께 사라지므로, 다시 설치하면 다운로드부터 새로 시작합니다.

### 두 페이지가 다르게 적은 기본 실행 명령

컨테이너는 공식 이미지 `vllm/vllm-openai:latest`로 뜨며, `/data/`에 받은 모델을 `opt-125m`이라는 이름으로 서빙합니다. 다만 기본 `image.command`는 두 페이지가 다르게 적습니다. Helm 가이드의 값 표는 `["vllm","serve","/data/","--served-model-name","opt-125m","--host","0.0.0.0","--port","8000"]`로 적습니다. 차트 저장소의 `values.yaml` 파일에는 여기에 `--enforce-eager`, `--dtype bfloat16`, `--block-size 16`이 더 붙어 있습니다. 실제로 렌더링되는 값은 설치에 쓴 `values.yaml`이 정합니다. 앞서 본 대로 eager 강제는 `torch.compile`과 CUDA 그래프를 모두 끕니다. 처리량이 중요하면 이 플래그를 뺀 명령으로 덮어씁니다. 엔진 인자는 이 배열 뒤에 이어 붙입니다.

### init job과 wait 컨테이너로 나뉜 S3 다운로드

`extraInit.modelDownload.enabled`의 기본값은 `true`입니다. 켜져 있으면 차트가 자동 다운로드 job과 wait 컨테이너를 만듭니다. 둘 다 `{% raw %}{{ .Release.Name }}{% endraw %}-secrets` 시크릿에서 `S3_ENDPOINT_URL`, `S3_BUCKET_NAME` 같은 환경변수를 주입받습니다. job이 `aws s3 sync`로 모델을 PVC의 `/data`에 받습니다. 그동안 vLLM 파드의 init 컨테이너는 `--dryrun` 동기화 결과에 받을 파일이 남아 있는 동안 10초씩 기다립니다.

```bash
while aws --endpoint-url $S3_ENDPOINT_URL s3 sync --dryrun s3://$S3_BUCKET_NAME/$S3_PATH /data | grep -q download; do sleep 10; done
```

기본값에서 모델 경로는 `relative_s3_model_path/opt-125m`, PVC 크기는 `1Gi`이고, EC2 메타데이터 서비스는 꺼져 있습니다. 더 큰 모델을 받으려면 PVC 크기와 경로를 함께 올립니다. 다음 값 파일이 그런 예입니다.

```yaml
extraInit:
  modelDownload:
    enabled: true
  pvcStorage: "10Gi"
  s3modelpath: "models/llama-7b"
```

llm-d처럼 모델 다운로드 없이 사이드카만 필요하면 다운로드를 끄고 `initContainers`를 직접 적습니다. 다운로드를 켠 채 `initContainers`를 함께 적으면, 커스텀 init 컨테이너는 다운로드 대기 컨테이너 뒤에 붙습니다.

```yaml
extraInit:
  modelDownload:
    enabled: false
  initContainers:
    - name: llm-d-routing-proxy
      image: ghcr.io/llm-d/llm-d-routing-sidecar:v0.2.0
      imagePullPolicy: IfNotPresent
      ports:
        - containerPort: 8080
          name: proxy
      securityContext:
        runAsUser: 1000
      restartPolicy: Always
  pvcStorage: "10Gi"
```

### 리소스, 프로브, 롤링 업데이트와 오토스케일링

`resources`의 요청과 제한은 기본으로 CPU 4, 메모리 16Gi, GPU 1개입니다. GPU를 요청하면 deployment 템플릿이 `runtimeClassName: nvidia`를 넣습니다. 또 `nvidia.com/gpu.product` 키로 노드 어피니티를 겁니다. 이 어피니티의 후보 값은 `gpuModels`에서 오는데, 기본값이 자리표시자 `"TYPE_GPU_USED"`입니다. 그대로 두면 파드가 노드에 배치되지 않으므로 실제 GPU 제품명으로 바꿉니다.

liveness 프로브는 포트 8000의 `/health`를 호출합니다. liveness는 15초 뒤 시작해 10초마다 확인하고, 연속 3회 실패하면 실패로 봅니다. readiness는 5초 뒤 시작해 5초마다 확인하며, 실패 임계값은 같은 3회입니다. 모델이 커서 기동이 오래 걸리면 liveness의 초기 지연을 늘립니다. `deploymentStrategy`를 비워 두면 롤링 업데이트가 `maxSurge: 100%`, `maxUnavailable: 0`으로 잡힙니다. 이 설정에서는 교체 중에 새 파드가 기존 파드 수만큼 함께 떠 있으므로, 그만큼 GPU가 클러스터에 남아 있어야 합니다.

오토스케일링은 기본으로 꺼져 있습니다. 켜면 HPA가 1~100개 레플리카 범위에서 CPU 사용률 80%를 목표로 조정합니다. LLM 서빙의 부하는 주로 GPU에 걸리므로, CPU 사용률만으로는 스케일 시점을 놓치기 쉽습니다. 켤 때는 부하 테스트로 CPU 지표가 실제 요청량을 따라가는지부터 확인합니다.

값을 바꾼 뒤에는 차트에 포함된 helm-unittest 테스트로 템플릿 렌더링을 검증합니다.

```bash
helm plugin install https://github.com/helm-unittest/helm-unittest
helm unittest .
```

## 처리량 중심 서빙에서 vLLM이 유리한 경우와 한계

vLLM이 가장 크게 앞서는 곳은 요청이 많고 길이가 제각각이라 KV 캐시 관리가 배치 크기를 막는 워크로드입니다. 반대로 메모리에 여유가 있고 시퀀스가 짧은 환경에서는 격차가 줄어듭니다. 플랫폼 제약과 버전별 차이도 선택에 함께 반영해야 합니다.

### 비교 대상별 처리량 격차

PagedAttention 논문과 vLLM 블로그는 비교 기준과 조건이 서로 다릅니다. 아래 표는 그 수치를 한곳에 모은 것입니다.

| 비교 대상 | vLLM의 우위 | 조건 |
|---|---|---|
| FasterTransformer, Orca | 처리량 2–4배, 모델 정확도 영향 없음 | 논문, 여러 모델·워크로드 |
| FasterTransformer | 유지 가능한 요청률 최대 22배 | 논문, ShareGPT |
| Orca (Oracle) | 요청률 1.7–2.7배, 지연은 비슷 | 논문, ShareGPT |
| Orca (Max) | 요청률 2.7–8배, 지연은 비슷 | 논문, ShareGPT |
| HuggingFace Transformers | 처리량 최대 24배, 모델 구조 변경 없음 | vLLM 블로그 |
| 초기 HF 백엔드 | 처리량 최대 30배 | LMSYS 내부 마이크로 벤치마크 |

논문은 FasterTransformer와 벌어진 격차의 원인으로 두 가지를 듭니다. FasterTransformer는 세밀한 스케줄링을 쓰지 않고, 메모리를 Orca (Max)처럼 비효율적으로 관리합니다. Orca와 비교한 이점은 더 많은 요청을 한 배치에 담는 데서 나옵니다. OPT-13B에서 vLLM은 동시에 처리하는 요청 수가 Orca (Oracle)의 2.2배, Orca (Max)의 4.3배입니다. 앞서 본 대로 낭비가 마지막 블록 하나로 제한되기 때문입니다. 논문은 시퀀스가 길수록, 모델이 클수록, 디코딩 알고리즘이 복잡할수록 개선 폭이 커진다고 정리합니다.

### 이점이 줄어드는 조건

Alpaca 데이터셋에서 OPT-175B를 돌린 경우에는 Orca (Oracle)과 Orca (Pow2) 대비 우위가 뚜렷하지 않았습니다. 이 서버 구성은 KV 캐시에 쓸 GPU 메모리가 넉넉했고, Alpaca는 시퀀스가 짧았습니다. 그래서 메모리 관리가 비효율적인 시스템도 요청을 많이 묶을 수 있었고, 병목이 메모리에서 연산으로 옮겨갔습니다.

### 운영 규모의 도입 사례

LMSYS의 Chatbot Arena는 HF Transformers 기반 백엔드로 출발했습니다. 피크 트래픽이 몇 배로 늘면서 이 백엔드가 병목이 되었고, FastChat-vLLM 통합으로 최대 5배 많은 트래픽을 받게 되었습니다. 같은 글은 서빙에 쓰는 GPU 수를 50% 줄였고, 하루 평균 3만, 피크 6만 요청을 처리했다고 밝힙니다.

### 플랫폼, 바이너리, 버전의 제약

앞서 설치 과정에서 본 Linux 전용 지원과 CUDA·PyTorch 바이너리 비호환이 그대로 도입 제약이 됩니다.

버전에 따라 동작도 달라집니다. 앞서 본 GPU 메모리 사용 비율과 선점 방식의 v0.6.4 차이가 그 예입니다. vLLM V1 블로그는 당시 알파였던 V1이 파이프라인 병렬, 구조화 디코딩, 추측 디코딩, LoRA, Prometheus 메트릭 등을 지원하지 않았고 Ampere 이후 NVIDIA GPU만 지원했다고 보고했습니다. 이는 그 시점의 보고이므로, 블로그나 튜토리얼을 따라 설정할 때는 그 글이 다룬 버전을 먼저 확인합니다.

### 선택 기준

이 글의 판단으로는 다음 경우에 vLLM을 고릅니다. 동시 요청이 많고 입력·출력 길이가 들쭉날쭉한 경우, 병렬 샘플링이나 빔 서치처럼 KV 캐시를 공유할 여지가 큰 경우, OpenAI 호환 엔드포인트로 기존 클라이언트를 그대로 붙이려는 경우입니다. 반면 요청이 드물어 배치가 거의 생기지 않거나, 짧은 시퀀스를 큰 메모리에서 돌려 이미 연산이 병목인 환경에서는 이점이 작습니다. 이런 경우에는 HF Transformers처럼 익숙한 스택을 유지하는 편이 단순합니다. 지금까지 따라온 작은 예제 모델은 동작 확인용입니다. 도입을 결정하기 전에는 실제 모델과 실제 트래픽 분포로 요청률 대비 지연 곡선을 직접 측정합니다. 운영에서는 앞서 다룬 `--gpu-memory-utilization`과 `--max-model-len` 조합, CUDA·PyTorch 버전 고정, 그리고 CPU 사용률만 보는 기본 오토스케일링 설정을 가장 먼저 점검합니다.

- - -

### Sources
* [Architecture Overview - vLLM](https://docs.vllm.ai/en/latest/design/arch_overview/)
* [Quickstart - vLLM](https://docs.vllm.ai/en/latest/getting_started/quickstart/)
* [Engine Arguments - vLLM](https://docs.vllm.ai/en/stable/configuration/engine_args/)
* [GPU - vLLM](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
* [Installation - vLLM](https://docs.vllm.ai/en/latest/getting_started/installation/)
* [CPU - vLLM](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/)
* [Helm - vLLM](https://docs.vllm.ai/en/stable/deployment/frameworks/helm/)
* [Helm Charts - vLLM](https://docs.vllm.ai/en/stable/examples/deployment/chart-helm/)
* [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/html/2309.06180)
* [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention &#124; vLLM Blog](https://vllm.ai/blog/2023-06-20-vllm)
* [vLLM V1: A Major Upgrade to vLLM's Core Architecture &#124; vLLM Blog](https://vllm.ai/blog/2025-01-27-v1-alpha-release)
* [Inside vLLM: Anatomy of a High-Throughput LLM Inference System &#124; vLLM Blog](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)
* [vLLM Engine Arguments &#124; Unsloth Documentation](https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide/vllm-engine-arguments)
* [Engine Arguments — vLLM](https://docs.vllm.ai/en/v0.6.4/models/engine_args.html)
