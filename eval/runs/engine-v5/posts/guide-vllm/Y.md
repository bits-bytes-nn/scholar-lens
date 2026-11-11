---
layout: post
title: "vLLM Internals and Setup: PagedAttention, Scheduling, Prefix Caching"
date: 2026-11-09 14:59:23
description: "vLLM의 PagedAttention과 V1 프로세스 구조, 통합 스케줄러, 자동 프리픽스 캐싱의 동작 원리를 정리하고, uv 설치부터 OpenAI 호환 서버, 하드웨어별 설치와 메트릭 관측까지 설정 전반을 다룹니다."
categories: ["Tech Guides", "Language-Models"]
tags: ["PagedAttention", "Prefix-Caching", "KV-Cache-Management", "Continuous-Batching", "OpenAI-Compatible-Server", "Model-Serving"]
cover: /assets/images/language-models.jpg
use_math: true
---
## TL;DR

vLLM은 LLM 추론·서빙 엔진으로, 서빙 처리량을 묶는 KV 캐시 메모리 낭비를 줄이는 데 설계의 초점이 있습니다. 기존 시스템은 KV 캐시를 최대 길이만큼 연속 메모리에 미리 잡아 두었고, 단편화와 과다 예약 때문에 메모리의 60~80%를 낭비했습니다. PagedAttention은 OS 가상 메모리의 페이징처럼 KV 캐시를 고정 크기 블록으로 나눕니다. 이 블록을 비연속 공간에 흩어 두고 블록 테이블로 위치를 추적하며, 토큰이 생성될 때마다 필요한 만큼만 할당합니다. 이렇게 하면 낭비가 시퀀스의 마지막 블록에서만 생기고 그 비율은 4% 미만으로 떨어집니다. 남은 메모리에는 더 많은 시퀀스를 묶어 처리합니다. 배치는 매 반복마다 다시 짜는 연속 배칭으로 구성합니다. PagedAttention 논문(Kwon et al., SOSP 2023)의 평가에서 vLLM은 FasterTransformer, Orca보다 처리량이 2~4배 높았고 모델 정확도에는 영향이 없었습니다.

현재 vLLM에서 기본으로 켜지는 엔진 아키텍처인 V1에서는 요청이 세 프로세스를 거칩니다. API 서버가 HTTP를 받고, 엔진 코어가 스케줄링을 맡고, GPU마다 하나씩 있는 워커가 forward를 실행합니다. 스케줄러는 prefill과 decode를 따로 나누지 않습니다. 매 스텝 `max_num_batched_tokens` 예산을 요청들에 나눠 줍니다. KV 캐시가 바닥나면 실행 중인 요청을 선점하고, 그 요청이 다시 승인되면 CPU 스왑 대신 재계산합니다. 처리가 끝난 요청의 KV 블록은 해시로 찾아 같은 프리픽스를 가진 새 요청에 다시 쓰며, 이때 모델 출력은 달라지지 않습니다. 그다음 `vllm serve Qwen/Qwen2.5-1.5B-Instruct`처럼 서빙할 모델 이름을 넣어 OpenAI 호환 API 서버를 띄웁니다. 동시 요청이 많아 처리량이 중요한 서빙에서, 그리고 시스템 프롬프트처럼 공통 프리픽스를 공유하는 요청이 많은 워크로드에서 vLLM의 장점이 가장 크게 드러납니다. 반대로 서버 하나가 한 번에 한 모델만 호스팅하므로 한 프로세스에서 여러 모델을 돌려야 하는 경우에는 맞지 않습니다. 요청이 드문 단일 사용자 환경에서도 배칭으로 얻는 이득이 작습니다.

## 연속 메모리 사전 할당이 KV 캐시의 60~80%를 낭비하는 서빙 문제

LLM 서빙의 처리량을 묶는 것은 연산량이 아니라 KV 캐시(KV cache)를 담을 GPU 메모리입니다. 기존 시스템은 그 메모리의 대부분을 실제 토큰이 쓰지 않는 공간으로 잡아 두고 있었습니다.

자기회귀 트랜스포머(transformer)는 프롬프트와 지금까지 만든 출력을 바탕으로 토큰을 하나씩 생성합니다. 이렇게 순차적으로 생성하기 때문에 워크로드는 메모리 병목이 되고, GPU 연산 능력은 다 쓰이지 못합니다. 여러 요청을 한 배치로 묶으면 처리량이 오르지만, 그러려면 요청마다 쓰는 메모리를 효율적으로 관리해야 합니다. 서빙 중 GPU 메모리에서 가중치는 내내 고정되어 있고, 요청의 동적 상태, 곧 어텐션(attention)의 키·값 텐서인 KV 캐시는 요청마다 할당되고 해제됩니다. 활성값은 작은 비중에 그치므로 최대 배치 크기는 KV 캐시를 어떻게 관리하느냐에 달려 있습니다. KV 캐시는 크기도 크고 변동도 심합니다. LLaMA-13B에서는 시퀀스 하나가 최대 1.7GB를 차지하고, 그 크기는 예측할 수 없는 시퀀스 길이에 따라 달라집니다.

FasterTransformer와 Orca 같은 기존 시스템은 대부분의 딥러닝 프레임워크처럼 텐서를 연속 메모리에 두어야 했습니다. 그래서 요청마다 최대 길이(예: 2048 토큰)만큼의 연속 공간을 미리 할당했고, 이 방식에서 세 종류의 낭비가 생깁니다.

- **예약(reservation)**: 덩어리 전체가 요청이 끝날 때까지 묶여 있어서, 지금 비어 있는 부분이 있어도 더 짧은 다른 요청이 쓰지 못합니다.
- **내부 단편화(internal fragmentation)**: 실제 출력 길이가 최대 길이보다 훨씬 짧으면 남는 공간이 그대로 버려집니다.
- **외부 단편화(external fragmentation)**: 요청마다 미리 할당하는 크기가 달라서 덩어리 사이에 쓰지 못하는 틈이 생깁니다.

가상의 예를 들어, 최대 2048 토큰으로 예약한 요청이 300 토큰에서 끝나면 1748칸, 약 85%는 한 번도 쓰이지 않습니다. 같은 요청에서 병렬 샘플링이나 빔 서치로 여러 시퀀스를 만들면 KV 캐시 일부를 공유할 여지가 있습니다. 하지만 시퀀스마다 별도의 연속 공간에 저장하는 구조에서는 그 공유도 불가능했습니다.

시스템별 평균 메모리 낭비 비율을 세 범주로 나눈 그래프에서, 실제 토큰 상태가 차지하는 몫이 얼마나 작은지 눈여겨보면 됩니다.

![LLM 서빙 시스템별 KV 캐시 메모리의 예약·내부 단편화·외부 단편화 낭비 비율 막대그래프](/assets/images/vllm-internals-and-setup-pagedattention-scheduling-prefix-caching-1.png)

논문의 프로파일링에서 기존 시스템이 실제 토큰 상태 저장에 쓴 KV 캐시 메모리는 20.4%~38.2%뿐이었습니다. vLLM 블로그는 이 낭비를 단편화와 과다 예약으로 인한 메모리의 60~80%로 제시합니다.

vLLM은 이 메모리 관리를 PagedAttention으로 바꿉니다. KV 캐시를 고정 크기 블록으로 나누어 비연속 공간에 저장하며, 블록 테이블이 동작하는 방식은 다음 내용에서 다룹니다. 블로그 발표 시점에 vLLM은 모델 구조를 바꾸지 않고도 HuggingFace Transformers(HF)보다 최대 24배 높은 처리량을 냈습니다. 논문 평가에서는 FasterTransformer·Orca 대비 처리량이 2~4배 높았고 모델 정확도에는 영향이 없었습니다. 시퀀스가 길수록, 모델이 클수록, 디코딩 알고리즘이 복잡할수록 향상 폭이 컸습니다.

블로그의 벤치마크는 A10G의 LLaMA-7B와 A100(40GB)의 LLaMA-13B를 대상으로, ShareGPT에서 입출력 길이를 샘플링해 측정했습니다.

| 요청당 출력 수 | HF 대비 처리량 | TGI 대비 처리량 |
|---|---|---|
| 1개 | 14배~24배 | 2.2배~2.5배 |
| 병렬 3개 | 8.5배~15배 | 3.3배~3.5배 |

출력을 여러 개 요청하면 HF 대비 배수는 줄지만 TGI 대비 격차는 오히려 벌어집니다. 그 가운데 A100에서 요청마다 출력 세 개를 병렬 생성한 경우를 그래프로 확인할 수 있습니다.

![A100에서 요청당 병렬 출력 세 개를 생성할 때 HF·TGI·vLLM의 서빙 처리량 비교 그래프](https://vllm.ai/blog-assets/figures/perf_a100_n3_light.png)

실제 운영 사례로는 LMSYS가 있습니다. LMSYS의 초기 내부 마이크로벤치마크에서 vLLM 백엔드는 초기 HF 백엔드보다 최대 30배 높은 처리량을 냈습니다. 교체 후 같은 트래픽을 서빙하는 GPU 수는 50% 줄었고, vLLM은 하루 평균 3만 건, 피크 6만 건의 요청을 처리했습니다.

## 블록 테이블과 비연속 KV 블록으로 단편화와 복사를 없애는 PagedAttention

PagedAttention은 KV 캐시를 일정한 수의 토큰을 담는 블록으로 나눕니다. 이 블록을 메모리 어디에나 흩어 두고 블록 테이블(block table)로 위치를 추적해서, 앞서 본 세 가지 낭비와 시퀀스 사이에서 공유할 수 없던 문제를 함께 해결합니다.

### 페이지·바이트·프로세스로 옮겨 놓은 KV 캐시

PagedAttention은 운영체제 가상 메모리의 페이징에서 착안했습니다. 논문은 블록을 페이지로, 토큰을 바이트로, 요청을 프로세스로 대응시킵니다. 블로그는 같은 비유에서 프로세스 자리에 시퀀스를 놓습니다. 블록 하나에는 정해진 개수의 토큰에 해당하는 키·값 벡터가 들어가며, 이 개수가 블록 크기 $$B$$입니다.

블록 크기가 모두 같으므로 외부 단편화는 사라집니다. 비교적 작은 블록을 필요할 때마다 할당하므로 내부 단편화도 줄어듭니다. 공유는 블록 단위로 이루어집니다. 트랜스포머의 각 토큰에는 층과 헤드마다 키·값 벡터가 있습니다. 이를 한 블록에 모두 담는 설계와 층·헤드별로 블록 테이블을 따로 두는 설계 사이에 성능 차이가 없어서, 저자들은 구현하기 쉬운 후자를 택했습니다.

### 블록 단위로 나눠 계산하는 어텐션

블록이 연속되어 있지 않아도 어텐션 결과는 같아야 합니다. 그래서 논문은 어텐션 식을 블록 합으로 다시 씁니다.

$$A_{ij}=\frac{\exp(q_{i}^{\top}K_{j}/\sqrt{d})}{\sum_{t=1}^{\lceil i/B\rceil}\exp(q_{i}^{\top}K_{t}\mathbf{1}/\sqrt{d})},\ o_{i}=\sum_{j=1}^{\lceil i/B\rceil}V_{j}A_{ij}^{\top}$$

$$q_i$$는 $$i$$번째 토큰의 쿼리 벡터입니다. $$K_{j}=(k_{(j-1)B+1},\ldots,k_{jB})$$와 $$V_{j}=(v_{(j-1)B+1},\ldots,v_{jB})$$는 $$j$$번째 키 블록과 값 블록입니다. $$A_{ij}$$는 $$j$$번째 KV 블록에 대한 어텐션 점수 행벡터이고, $$d$$는 어텐션의 차원, $$\mathbf{1}$$은 모든 원소가 1인 벡터입니다. 분모의 합은 $$i$$번째 토큰까지 쓰인 $$\lceil i/B\rceil$$개 블록 전체를 훑습니다. 커널은 블록마다 따로 키를 가져와 점수를 구하고, 같은 블록의 값과 곱해 출력에 더합니다.

이 그림에서는 쿼리 토큰 "forth"가 물리 메모리 곳곳에 흩어진 세 블록을 차례로 읽는 모습을 보면 됩니다. 블록 0에는 "Four score and seven"의 키가 들어 있습니다.

![쿼리 벡터가 메모리에 비연속으로 흩어진 세 개의 키·값 블록을 블록별로 읽어 어텐션을 계산하는 PagedAttention 구조도](/assets/images/vllm-internals-and-setup-pagedattention-scheduling-prefix-caching-2.png)

### 블록 테이블과 필요할 때만 늘어나는 물리 블록

vLLM은 생성 가능한 최대 길이만큼 메모리를 미리 잡지 않습니다. 프롬프트 계산에서 나오는 KV 캐시를 담을 만큼의 블록만 예약하고, 블록 테이블이 시퀀스의 연속된 논리 블록을 비연속 물리 블록(physical block)에 매핑합니다. 논문의 예시는 블록 크기 4, 프롬프트 7토큰으로 단계를 밟습니다.

| 단계 | 일어나는 일 | 블록 테이블 상태 |
|---|---|---|
| 프리필 | 논리 블록 0·1을 물리 블록 7·1에 매핑하고, 앞 4토큰과 뒤 3토큰의 KV를 저장합니다 | 블록 1에 빈 슬롯 하나 |
| 첫 디코딩 | 물리 블록 7·1로 PagedAttention을 수행하고, 새 토큰의 KV를 남은 슬롯에 씁니다 | `#filled` 기록을 갱신합니다 |
| 둘째 디코딩 | 마지막 논리 블록이 가득 차서 새 논리 블록용으로 물리 블록 3을 할당합니다 | 새 매핑을 추가합니다 |

아래 그림에서는 왼쪽 논리 블록의 순서가 오른쪽 물리 블록 번호의 순서와 전혀 다르게 놓인 모습에 주목하면 됩니다.

![논리 KV 블록 0·1·2가 블록 테이블을 거쳐 물리 블록 7·1·3으로 변환되는 과정을 보여 주는 도식](/assets/images/vllm-internals-and-setup-pagedattention-scheduling-prefix-caching-3.png)

블로그의 애니메이션은 같은 흐름을 시간 순서로 보여 줍니다. 토큰이 생성되면서 필요한 시점에 물리 블록이 할당됩니다.

![토큰 생성에 따라 블록 테이블에 물리 블록이 하나씩 할당되는 요청 생성 과정 애니메이션](https://vllm.ai/blog-assets/figures/annimation1.gif)

블록은 왼쪽부터 채워지고, 앞의 블록이 모두 찬 뒤에만 새 블록을 할당합니다. 그래서 요청 하나의 낭비는 블록 하나를 넘지 않습니다. 요청이 끝나면 그 블록은 다른 요청의 KV 캐시를 저장하는 데 쓰입니다. 블로그는 실제 낭비가 4% 미만이며, 그만큼 더 많은 시퀀스를 배치에 넣을 수 있어 처리량이 오른다고 밝힙니다.

### 반복마다 새로 짜는 배치

배치는 매 디코딩 반복마다 다시 구성됩니다. vLLM은 반복마다 배치할 후보 시퀀스를 고르고, 새로 필요한 논리 블록에 물리 블록을 할당합니다. 그런 다음 프롬프트 단계 요청의 토큰 전체와 생성 단계 요청의 최신 토큰을 하나의 시퀀스로 이어 붙여 모델에 넣습니다. 이것이 연속 배칭(continuous batching)의 출발점입니다.

### 블록 크기 16의 근거

블록 하나에 여러 토큰을 담으면 커널이 더 많은 위치를 병렬로 처리하므로 하드웨어 활용도가 오르고 지연이 줄어듭니다. 너무 작으면 KV 캐시를 읽고 처리할 때 GPU 병렬성을 다 쓰지 못합니다. 너무 크면 내부 단편화가 늘고 블록을 공유할 확률이 낮아집니다. 이 둘 사이의 균형을 고려해 논문 시점의 vLLM은 기본 블록 크기를 16으로 정했습니다.

### 세 가지 GPU 커널 융합

블록 단위 접근은 기존 시스템이 효율적으로 지원하지 못하는 메모리 접근 패턴이어서, 저자들은 전용 커널을 따로 만들었습니다.

1. **reshape와 블록 쓰기 융합**: 층마다 새 KV 캐시를 블록으로 나누고, 블록 읽기에 맞는 레이아웃으로 바꾼 뒤, 블록 테이블이 가리키는 위치에 씁니다. 이 과정을 커널 하나로 합쳐 실행 오버헤드를 줄입니다.
2. **블록 읽기와 어텐션 융합**: FasterTransformer의 어텐션 커널을 고쳐 블록 테이블을 따라 KV를 읽으면서 바로 어텐션을 계산합니다. 병합된 메모리 접근을 위해 워프 하나가 블록 하나를 읽고, 배치 안에서 서로 다른 시퀀스 길이도 지원합니다.
3. **블록 복사 융합**: 카피 온 라이트가 일으키는 복사는 비연속 블록에 걸쳐 있습니다. `cudaMemcpyAsync`로 처리하면 작은 전송이 수없이 생기므로, 여러 블록의 복사를 커널 한 번 실행으로 묶습니다.

### 참조 카운트와 카피 온 라이트로 공유하는 블록

블록 테이블이 있으면 서로 다른 시퀀스의 논리 블록을 같은 물리 블록에 매핑할 수 있습니다. 안전하게 공유하려고 물리 블록마다 참조 카운트(reference count)를 두고 카피 온 라이트(copy-on-write)를 적용합니다.

**병렬 샘플링**에서는 두 출력이 프롬프트를 공유합니다. 두 시퀀스의 논리 블록 0·1이 모두 물리 블록 7·1을 가리키므로 두 블록의 참조 카운트는 2입니다. 샘플 A1이 마지막 논리 블록에 쓰려고 하면, vLLM은 물리 블록 1의 참조 카운트가 1보다 큰 것을 확인합니다. 그러면 새 물리 블록 3을 할당해 내용을 복사하고 카운트를 1로 낮춥니다. 이어서 A2는 카운트가 이미 1이므로 물리 블록 1에 바로 씁니다. 출력을 여러 개 샘플링하는 애니메이션에서는 프롬프트 블록이 한 벌만 존재하다가, 갈라지는 지점에서만 블록이 복제되는 모습을 확인할 수 있습니다.

![여러 출력 시퀀스가 같은 물리 블록을 공유하다가 쓰기 시점에 카피 온 라이트로 블록을 복제하는 과정 애니메이션](https://vllm.ai/blog-assets/figures/annimation3.gif)

**빔 서치**에서는 프롬프트 블록뿐 아니라 후보 사이의 다른 블록도 공유되며, 공유 패턴은 디코딩이 진행되면서 계속 바뀝니다. 논문의 $$k=4$$ 예시를 보면, 후보 0과 3이 상위 후보에서 탈락하자 그 논리 블록이 해제됩니다. 이어서 참조 카운트가 0이 된 물리 블록 2·4·5·8을 풀고, 새 후보를 위해 블록 9~12를 할당합니다. 기존 시스템이라면 후보 3이 후보 2의 KV 캐시 상당 부분을 복사해야 했습니다. vLLM에서는 새 토큰이 기존 공유 블록 안에 들어갈 때만 블록 하나를 복사합니다.

**공유 프리픽스**는 프로세스들이 공유 라이브러리를 쓰는 방식과 닮았습니다. 서비스 제공자가 미리 정한 시스템 프롬프트용 물리 블록을 예약해 두면, 사용자 프롬프트는 논리 블록을 그 블록에 매핑하기만 하면 됩니다. 이때 마지막 블록에는 카피 온 라이트 표시를 합니다. 이 방식은 논문 시점의 설계입니다.

> **핵심:** 모델과 실행 커널은 시퀀스마다 물리 블록 ID 목록만 받고 공유 패턴은 알지 못합니다. 덕분에 탐욕 디코딩, 병렬 샘플링, 빔 서치처럼 디코딩 방식이 서로 다른 요청도 한 배치에 함께 들어갑니다.

블로그는 이 메모리 공유로 병렬 샘플링과 빔 서치의 메모리 사용량이 최대 55% 줄고, 처리량은 최대 2.2배 높아진다고 주장합니다.

## API 서버·엔진 코어·GPU 워커로 나뉜 V1 프로세스 구조

V1에서 요청 하나는 HTTP를 받는 API 서버, 스케줄링과 KV 캐시를 맡는 엔진 코어, 실제 forward를 돌리는 GPU 워커라는 서로 다른 프로세스를 차례로 지납니다. 프로세스 수는 텐서 병렬과 데이터 병렬 크기에 따라 정해집니다.

### 커널은 그대로 두고 다시 짠 핵심 시스템

V1이 V0에서 그대로 가져온 부분은 모델 구현, GPU 커널, 유틸리티입니다. 스케줄러, KV 캐시 매니저, 워커, 샘플러, API 서버는 새로 설계했습니다. 앞서 본 PagedAttention 커널은 유지되었고, 그 위에서 요청을 받고 배치를 짜는 층이 바뀐 셈입니다. 재설계 목표는 네 가지입니다. 단순하고 모듈화된 코드베이스, CPU 오버헤드가 거의 없는 높은 성능, 핵심 최적화를 한 아키텍처로 통합하는 것, 기능과 최적화를 기본으로 켜서 설정이 필요 없게 하는 것입니다.

### 세 종류의 상주 프로세스와 조건부 코디네이터

**API 서버**는 OpenAI 호환 API 같은 HTTP 요청을 받아 토큰화와 멀티모달 데이터 로딩 같은 입력 처리를 하고, 결과를 클라이언트로 스트리밍합니다. 엔진 코어와는 ZMQ 소켓으로 통신합니다. 기본값은 프로세스 하나이고, 데이터 병렬을 쓰면 DP 크기에 맞춰 자동으로 늘어납니다. 각 API 서버는 모든 엔진 코어와 many-to-many로 연결되어 어느 엔진 코어로든 요청을 보낼 수 있습니다. 개수를 직접 정하려면 `--api-server-count` 플래그를 씁니다. 미디어 로딩에는 API 서버마다 CPU 스레드를 여러 개 쓰는데, 이 스레드 수는 `VLLM_MEDIA_LOADING_THREAD_COUNT`로 조절하며 기본값은 8입니다.

**엔진 코어**는 스케줄러를 돌리고 KV 캐시를 관리하면서 GPU 워커들의 모델 실행을 조율합니다. 요청을 쉬지 않고 스케줄링해 워커에 일을 넘기는 busy loop로 동작합니다. 엔진 코어는 DP rank마다 하나씩 있어서, `--data-parallel-size 4`이면 엔진 코어 프로세스도 넷입니다.

**GPU 워커**는 GPU 하나를 전담합니다. 모델 가중치를 올리고, forward를 실행하고, GPU 메모리를 관리하며, 자신을 소유한 엔진 코어와 통신합니다. 엔진 코어 하나에 딸린 워커 수는 `tensor_parallel_size x pipeline_parallel_size`입니다.

**DP 코디네이터**는 `--data-parallel-size > 1`일 때만 생깁니다. DP rank 사이의 로드 밸런싱을 맡고, MoE 모델에서는 동기화된 forward를 조율합니다.

### TP·DP 설정에서 나오는 프로세스 개수

GPU 수를 $$N$$, 텐서 병렬 크기를 $$TP$$, 데이터 병렬 크기를 $$DP$$, API 서버 수를 $$A$$라고 두면 프로세스 수는 아래 표처럼 정해집니다.

| 프로세스 | 개수 | 역할 |
|---|---|---|
| API 서버 | `A` (기본값 `DP`) | HTTP 요청과 입력 처리 |
| 엔진 코어 | `DP` (기본값 1) | 스케줄러와 KV 캐시 관리 |
| GPU 워커 | `N` (= `DP x PP x TP`) | GPU당 하나, 모델 forward 실행 |
| DP 코디네이터 | `DP > 1`이면 1, 아니면 0 | DP rank 간 로드 밸런싱 |
| 합계 | `A + DP + N` (DP > 1이면 +1) | - |

GPU 4장 단일 노드에서 `vllm serve -tp=4`로 띄우면 API 서버 1, 엔진 코어 1, GPU 워커 4로 모두 6개 프로세스가 뜹니다. 이 구성은 DP가 1이므로 DP 코디네이터는 생기지 않습니다. 아래 구조도는 이 6개 프로세스 구성을 그린 것입니다.

![텐서 병렬 4 단일 노드 배포에서 API 서버 하나와 엔진 코어 하나가 GPU 워커 네 개에 연결된 V1 프로세스 구조도](https://docs.vllm.ai/en/latest/assets/design/arch_overview/v1_process_architecture_tp4.png)

GPU 8장에서 `vllm serve -tp=2 -dp=4`를 쓰면 API 서버 4, 엔진 코어 4, GPU 워커 8에 DP 코디네이터 1이 더해져 17개가 됩니다. 이 그림에서는 API 서버 넷이 엔진 코어 넷 모두와 엇갈려 연결된 many-to-many 배선과, 엔진 코어마다 워커가 둘씩 붙은 모습을 비교하면 됩니다.

![텐서 병렬 2·데이터 병렬 4 배포에서 API 서버 4개, 엔진 코어 4개, GPU 워커 8개, DP 코디네이터 1개로 이루어진 V1 프로세스 구조도](https://docs.vllm.ai/en/latest/assets/design/arch_overview/v1_process_architecture_tp2_dp4.png)

### 블록 테이블을 공유하는 텐서 병렬 워커

엔진 코어 하나가 워커 여러 개를 거느려도 KV 캐시 관리가 복잡해지지 않는 이유는 논문의 분산 실행 설계에서 찾을 수 있습니다. vLLM은 Megatron-LM 방식의 텐서 모델 병렬을 지원합니다. 선형층은 블록 단위 행렬 곱으로 나뉘고, 어텐션은 헤드 차원으로 나뉘어 각 프로세스가 헤드 일부를 맡습니다. 모델을 나눠도 모든 샤드는 같은 입력 토큰을 처리하므로 같은 위치의 KV 캐시가 필요합니다. 그래서 중앙 스케줄러 안에 KV 캐시 관리자를 하나만 두고, 모든 워커가 논리 블록과 물리 블록의 매핑을 공유합니다. 워커들의 물리 블록 ID는 같지만, 각 워커는 자기가 맡은 어텐션 헤드의 KV만 저장합니다.

스케줄러는 매 스텝 요청별 입력 토큰 ID와 블록 테이블을 담은 제어 메시지를 워커에 브로드캐스트합니다. 워커는 어텐션 층에서 이 블록 테이블을 따라 KV 캐시를 읽고, 중간 결과는 스케줄러를 거치지 않고 all-reduce로 동기화한 뒤, 샘플링한 토큰을 스케줄러로 돌려보냅니다. 메모리 관리 정보는 각 반복이 시작될 때 입력과 함께 받으므로, 워커끼리 메모리 관리를 위해 따로 동기화할 필요가 없습니다. 참고로 v0.7.3 문서에서는 `pipeline_parallel_size`와 `tensor_parallel_size`의 곱이 사용 가능한 GPU 수 이하이면 단일 호스트에서 멀티프로세싱(`mp`)으로 분산 워커를 구성했습니다.

## 토큰 예산 하나로 prefill과 decode를 함께 다루는 통합 스케줄러와 재계산 기반 선점

V1 스케줄러는 매 스텝 정해진 토큰 예산을 요청별 토큰 수로 나눠 주는 루프 하나로 배치를 짜고, KV 캐시가 바닥나면 정책에 따라 실행 중인 요청을 내보낸 뒤 나중에 처음부터 다시 계산합니다.

### 계산된 토큰 수의 차이를 메우는 단일 루프

스케줄러는 앞서 본 엔진 코어 프로세스 안에서 돌며, 스텝마다 어떤 요청을 실행할지 정합니다. 이때 쓰는 기준은 `max_num_batched_tokens`라는 토큰 예산 하나입니다. 스케줄러는 이 예산을 실행 중(running) 요청과 대기 중(waiting) 요청 양쪽에 채워 넣고, prefill 단계와 decode 단계를 따로 두지 않습니다. 각 요청은 자기 토큰 가운데 몇 개가 이미 계산되었는지 기록해 두고, 스케줄러는 그 차이를 메울 만큼 토큰을 배정합니다. 배정 결과는 `{request_id: num_tokens}` 형태의 딕셔너리이며, 프롬프트 토큰과 출력 토큰을 똑같이 취급합니다. 청크 프리필, 프리픽스 캐싱, 추측 디코딩이 모두 이 한 루프 안에서 처리됩니다.

이 방식이 앞서 본 연속 배칭과 어떻게 이어지는지는 작은 예로 보면 분명해집니다. 예를 들어 예산이 512 토큰이고, decode 중인 요청 셋과 새로 들어온 2,000 토큰짜리 프롬프트 하나가 있다고 가정해 보겠습니다. 아래 표는 이 가상의 스텝에서 딕셔너리가 채워지는 모습입니다.

| 요청 | 상태 | 남은 미계산 토큰 | 이번 스텝 배정 |
|---|---|---|---|
| `req-A` | decode | 1 | 1 |
| `req-B` | decode | 1 | 1 |
| `req-C` | decode | 1 | 1 |
| `req-D` | 새 프롬프트 | 2,000 | 509 |
| 합계 | decode 3개 + 새 프롬프트 1개 | 2,003 | 512 |

`req-D`는 이번 스텝에서 509 토큰만 계산하고, 나머지 1,491 토큰은 다음 스텝부터 같은 방식으로 이어서 배정받습니다. 프롬프트를 이렇게 남은 예산에 맞춰 쪼개는 동작이 청크 프리필이며, `--enable-chunked-prefill`을 켜면 prefill 요청을 남은 `max_num_batched_tokens`에 맞춰 나눌 수 있습니다. V1에서는 청크 프리필이 가능한 경우 기본으로 켜집니다. V0에서는 모델 특성에 따라 조건부로만 켜졌습니다. 여러 요청의 부분 prefill을 동시에 처리하는 Concurrent Partial Prefills는 V1 기능 표에 "In Progress"로 올라 있습니다.

### 배치 토큰 수와 시퀀스 수라는 두 상한

한 스텝의 크기는 두 인자가 제한합니다. `--max-num-batched-tokens`는 한 iteration에서 처리할 수 있는 최대 토큰 수이고, `--max-num-seqs`는 한 iteration에서 처리할 최대 시퀀스 수입니다. 토큰 예산이 남아 있어도 시퀀스 수 상한에 먼저 닿으면 대기 요청을 더 받지 않으므로, decode 위주의 짧은 요청이 많을 때는 두 번째 상한이 실제 병목이 됩니다. 두 인자의 표기 기본값은 주로 테스트 편의를 위한 값이며, 실제 사용에서는 엔진 인자로 엔진 설정을 만드는 메서드인 `EngineArgs.create_engine_config`에서 값이 정해집니다. 청크 프리필 플래그도 마찬가지입니다. `--max-num-batched-tokens`는 `1k`, `25.6k` 같은 사람이 읽기 쉬운 정수 표기를 받으며, `1k`는 1,000, `1K`는 1,024로 해석됩니다.

예를 들어 실행 예제의 Qwen2.5-1.5B 서버에 두 상한을 직접 지정하면 명령은 아래처럼 됩니다. 숫자는 설명용으로 고른 값입니다.

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --max-num-batched-tokens 8k \
  --max-num-seqs 128
```

### KV 캐시가 고갈될 때 내보낼 요청을 정하는 두 정책

`--scheduling-policy`의 선택지는 `fcfs`와 `priority`이고 기본값은 `fcfs`입니다. 두 정책은 요청을 받는 순서뿐 아니라 KV 캐시가 바닥났을 때 누구를 선점할지도 정합니다.

- **FCFS**: 도착 순서대로 스케줄링하며, KV 캐시가 고갈되면 가장 마지막에 승인된 실행 중 요청을 선점합니다.
- **Priority**: 요청마다 정수 `priority` 필드가 있고, 값이 낮을수록 먼저 처리하며 같은 값이면 도착 시각으로 순서를 가립니다. KV 캐시가 고갈되면 우선순위가 가장 낮은 요청, 즉 `priority` 값이 가장 크고 그다음으로 가장 늦게 도착한 실행 중 요청을 선점합니다.

값의 방향을 반대로 이해하면 중요한 요청부터 선점되는 역효과가 납니다.

### CPU 스왑 없이 다시 계산하는 선점

새 KV 캐시 블록을 할당할 수 없는 요청이 생기면, 스케줄러는 정책이 고른 실행 중 요청을 블록이 충분히 빌 때까지 차례로 선점합니다. 순서는 다음과 같습니다.

1. 선점된 요청의 KV 캐시 블록을 해제합니다.
2. 요청의 진행 상태를 리셋하고 대기 큐로 돌려보냅니다. FCFS에서는 큐의 맨 앞에 들어가고, Priority에서는 `(priority, arrival_time)` 순서에 맞춰 다시 삽입됩니다.
3. 다시 승인되면 CPU 메모리에서 KV를 되가져오지 않고 재계산합니다. 프리픽스 캐싱이나, 외부 저장소나 다른 인스턴스에서 KV 캐시를 불러오는 구성 요소인 KV connector가 있으면 진행분 일부를 즉시 복원할 수 있습니다.

FCFS에서 선점된 요청이 큐 맨 앞으로 가기 때문에, 블록이 다시 생기면 이 요청이 새 요청보다 먼저 재승인됩니다. PagedAttention 논문은 디코딩 중 생성한 토큰을 원래 프롬프트에 이어 붙여 새 프롬프트로 삼으면 모든 위치의 KV 캐시를 한 번의 prompt 단계 iteration으로 만들 수 있어, 재계산 지연이 원래 지연보다 크게 낮을 수 있다고 설명합니다. V1에서 재승인된 요청은 진행 상태가 리셋된 채 앞서 본 토큰 예산 루프에서 다시 토큰을 배정받습니다.

### 논문과 v0.x에 있던 스왑과 재계산 두 갈래

처음부터 스왑이 없었던 것은 아닙니다. 논문의 vLLM도 FCFS를 쓰고 가장 늦게 도착한 요청부터 선점했습니다. 시퀀스의 블록은 함께 접근되므로 전부 축출하거나 하나도 축출하지 않는 all-or-nothing 방식을 썼고, 빔 서치 후보처럼 한 요청에 속한 시퀀스들은 sequence group으로 묶어 함께 선점하고 함께 재스케줄링했습니다. 축출된 블록을 되살리는 방법으로는 두 가지를 두었습니다. 하나는 블록을 CPU RAM으로 복사하고 별도의 CPU 블록 할당자로 관리하는 스왑이고, 다른 하나는 재스케줄링 때 KV 캐시를 다시 계산하는 재계산입니다. 스왑 방식에서는 CPU로 옮긴 블록 수가 GPU의 전체 물리 블록 수를 넘지 않아 스왑 공간이 KV 캐시용 GPU 메모리 크기로 묶였고, 선점이 일어나면 선점된 시퀀스가 모두 끝날 때까지 새 요청을 받지 않았습니다.

이 두 갈래는 v0.x 엔진 인자에도 그대로 남아 있었습니다. v0.8.3에도 `fcfs`(기본)와 `priority` 정책이 이미 있었습니다. V1 기능 표에서 GPU <> CPU KV 캐시 스와핑은 "Removed"로 표시되어 있고, 단순해진 코어 구조 덕분에 선점 처리에 스와핑이 더 이상 필요 없어졌다고 설명합니다. 시점별로 선점된 요청의 KV를 되살리는 방식을 나란히 놓으면 이렇게 갈립니다.

| 시점 | 선점된 요청의 KV 복구 방식 |
|---|---|
| 논문 | CPU RAM으로 스왑 또는 재계산 |
| v0.4.1 | GPU당 CPU 스왑 공간, 기본 4GiB |
| v0.7.3, v0.8.3 | `recompute` 또는 `swap` 중 선택 |
| V1 | 재계산, KV 캐시 스와핑 제거 |

 따라서 예전 문서에서 가져온 스왑 공간 설정은 V1 서버에서 선점 비용을 줄이는 수단이 되지 못합니다. V1에서 선점 비용은 재계산량과 프리픽스 캐시 적중 여부로 정해집니다.

### GPU 실행 뒤에서 다음 스텝을 짜는 비동기 스케줄링

비동기 스케줄링(`--async-scheduling`)을 켜면 CPU 스케줄링과 GPU 실행이 겹쳐 돕니다. GPU가 스텝 N을 실행하는 동안 CPU는 스텝 N+1을 스케줄링하고, 스텝 N이 만들 토큰 자리는 자리표시자로 채워 둡니다. 이렇게 하면 GPU 활용 사이의 공백이 줄어 지연 시간과 처리량이 함께 좋아지며, 끄려면 `--no-async-scheduling`을 씁니다. 선점과 맞물리는 지점도 정해져 있습니다. 아직 도착하지 않은 출력이 남은 상태에서 요청이 선점되어도 그 출력은 보통 도착하는 대로 전달됩니다. 다만 `reset_prefix_cache`가 같은 스텝 안에서 요청을 선점했다가 재개하는 경우처럼, 그 출력을 전달하는 것이 유효하지 않을 때는 출력을 버립니다.

## 해시 체인과 LRU free queue로 KV 블록을 재사용하는 Automatic Prefix Caching

자동 프리픽스 캐싱(automatic prefix caching)은 처리가 끝난 요청의 KV 캐시 블록을 지우지 않고 남겨 두었다가, 같은 프리픽스로 시작하는 새 요청이 오면 그 블록을 다시 씁니다. 모델 출력은 달라지지 않습니다. 블록을 찾아내는 키는 해시 체인이고, 아무도 쓰지 않는 블록은 free queue에 머물다가 공간이 모자랄 때 오래된 것부터 축출(eviction)됩니다.

### 부모 해시를 이어 받는 블록 키

vLLM은 블록마다 블록 안의 토큰과 그 앞 프리픽스 전체를 반영한 해시를 키로 씁니다. 키는 `hash(tuple[components])`이고 구성요소는 셋입니다.

- **부모 해시**: 바로 앞 블록의 해시 값입니다.
- **블록 토큰**: 이 블록에 든 토큰의 튜플입니다. 정확한 토큰을 넣어 해시 충돌 가능성을 줄입니다.
- **추가 해시**: LoRA ID, 멀티모달 입력 해시, cache salt처럼 블록을 고유하게 만드는 데 필요한 값입니다.

블록에 부모 해시가 들어가므로, 세 번째 블록의 키에는 앞 두 블록의 토큰 정보가 사슬처럼 묶여 있습니다. 문장 하나를 세 블록에 나눠 담으면 블록 키는 아래처럼 자랍니다.

```text
Block 1 Block 2 Block 3
[A gentle breeze stirred] [the leaves as children] [laughed in the distance]
Block 1: |<--- block tokens ---->|
Block 2: |<------- prefix ------>| |<--- block tokens --->|
Block 3: |<------------------ prefix -------------------->| |<--- block tokens ---->|
```

그래서 두 요청에서 같은 블록 토큰이 나와도 앞선 프리픽스가 다르면 키가 달라 재사용되지 않습니다. 캐시 대상은 가득 찬 블록뿐입니다.

### 첫 블록에 섞는 cache_salt와 이미지 해시

추가 해시 자리를 쓰는 대표적인 경우가 둘 있습니다. 첫째는 멀티테넌트 격리입니다. 요청에 `cache_salt`를 넣으면 그 값이 첫 블록의 해시에 주입됩니다. 그러면 salt가 같은 요청끼리만 캐시된 블록을 재사용할 수 있고, 공격자가 지연 시간 차이를 관찰해 다른 사용자의 캐시 내용을 추론하는 타이밍 공격을 막을 수 있습니다. 해시 체인 구조상 첫 블록 키가 달라지면 그 뒤 블록의 키도 모두 달라지므로, salt는 첫 블록에만 넣어도 충분합니다. Chat Completions 요청 본문에는 다음처럼 필드 하나를 더 넣습니다.

```json
{
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Here is a document with details about the world series: ..."},
    {"role": "user", "content": "Who won the world series in 2020?"}
  ],
  "cache_salt": "your-cache-salt"
}
```

둘째는 멀티모달 입력입니다. 토큰화 단계에서 `[IMG]`는 플레이스홀더 토큰 열로 바뀌고, 이 플레이스홀더는 prefill 중에 이미지 임베딩으로 대체됩니다. 이미지가 달라도 플레이스홀더 토큰은 똑같기 때문에, vLLM은 프론트엔드 이미지 프로세서가 만든 이미지 해시를 추가 해시로 넣어 이미지를 구분합니다. 블록 크기 16에 플레이스홀더가 41개인 프롬프트라면 블록 0부터 3까지 네 블록이 모두 같은 이미지 해시를 추가 해시로 가집니다.

### 블록 풀, free queue, 두 매핑으로 이루어진 KV 캐시 매니저

V1의 프리픽스 캐싱은 KV 캐시 매니저 안에 구현되어 있습니다. 기본 단위는 `KVCacheBlock` 데이터 클래스입니다.

```python
# excerpt: KVCacheBlock 데이터 클래스의 단순화된 필드
class KVCacheBlock:
    # The block ID (immutable)
    block_id: int
    # The block hash (will be assigned when the block is full,
    # and will be reset when the block is evicted).
    block_hash: BlockHash
    # The number of requests using this block now.
    ref_cnt: int

    # The pointers to form a doubly linked list for the free queue.
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None
```

`block_hash`는 블록이 가득 찰 때 붙고 축출될 때 지워지며, `ref_cnt`는 지금 이 블록을 쓰는 요청 수입니다. 매니저는 시작할 때 모든 `KVCacheBlock`을 한꺼번에 만들어 블록 풀로 둡니다. 그러면 Python 객체 생성 오버헤드가 없고 모든 블록을 항상 추적할 수 있습니다. free queue는 블록 안의 `prev`/`next` 포인터로 바로 엮은 이중 연결 리스트입니다. 이 구조에서는 큐 중간의 원소를 $$O(1)$$에 꼬리로 옮길 수 있고, 원소를 감싸는 `deque` 같은 별도 Python 큐가 필요 없습니다.

![KV 캐시 매니저 초기화 직후의 블록 풀, free queue, 캐시 블록, 요청 블록 구성도](https://docs.vllm.ai/en/stable/assets/design/prefix_caching/overview.png)

구성도에서 볼 요소는 넷입니다. Block Pool은 `KVCacheBlock` 리스트이고, Free Block Queue는 조작에 필요한 head와 tail 블록 포인터만 저장하며, Cache blocks는 해시 키에서 블록 ID로, Request blocks는 요청 ID에서 할당된 블록 ID로 가는 매핑입니다. 참조 수가 0인 블록만 free queue에 들어갑니다. 따라서 free queue에는 아직 해시를 지닌 채 재사용을 기다리는 블록과 해시가 없는 빈 블록이 섞여 있습니다.

### 적중 블록을 먼저 붙잡고 head에서 꺼내는 할당

새 요청이 오면 스케줄러는 먼저 `kv_cache_manager.get_computed_blocks()`를 호출합니다. 이 메서드는 프롬프트 토큰을 해시해 Cache blocks를 조회하고, 이미 계산된 블록의 열을 돌려줍니다. 이어서 `kv_cache_manager.allocate_slots()`가 다음 순서로 동작합니다.

1. 새로 필요한 블록 수를 계산하고, 할당할 블록이 부족하면 그대로 반환합니다.
2. 계산된 블록을 "touch"합니다. 참조 수를 1 올리고, 다른 요청이 쓰지 않던 블록이면 free queue에서 뺍니다. 이렇게 해서 적중한 블록이 이번 할당 중에 축출되지 않습니다.
3. free queue의 head에서 새 블록을 꺼냅니다. head 블록이 캐시된 블록이면 이 시점에 축출되어, 이후로는 어떤 요청도 그 블록을 재사용하지 못합니다.
4. 할당한 블록이 이미 토큰으로 가득 차 있으면 즉시 Cache blocks에 넣어, 같은 배치의 다른 요청도 재사용할 수 있게 합니다.

실행 중인 요청은 1단계와 3단계를 거친 뒤, 기존 블록과 새 블록의 슬롯에 토큰 ID를 덧붙이고, 가득 찬 블록이 생기면 그때 캐시에 넣습니다. 축출은 세 단계입니다. free queue의 head, 곧 가장 오래 쓰이지 않은 블록을 꺼내고, Cache blocks에서 그 블록 ID를 지우고, 블록 해시를 제거합니다.

요청이 끝나면 참조 수가 0이 된 블록을 해제하는데, 블록은 역순으로 free queue의 tail에 붙습니다. 요청의 마지막 블록은 더 많은 토큰을 해시에 담고 있어 다른 요청이 재사용할 가능성이 낮으므로, 가장 먼저 축출되도록 head 쪽에 가깝게 둡니다.

![요청 하나를 해제한 뒤 블록이 역순으로 tail에 붙은 free queue](https://docs.vllm.ai/en/stable/assets/design/prefix_caching/free.png)

이 그림에서는 요청 1과 함께 블록 2, 3, 4, 8이 해제되는데, 꼬리에 붙는 순서가 블록 번호의 역순이라는 데 주목하면 됩니다.

### 블록 10개로 따라가는 할당과 축출

블록 크기 4, 전체 블록 10개인 매니저에서 흐름을 따라가 보면 위 규칙이 어떻게 맞물리는지 보입니다. Time 1에 빈 캐시로 요청 0이 들어와 블록 4개를 받는데, 그중 3개는 가득 차서 캐시되고 네 번째 블록은 4토큰 중 3토큰만 찹니다. Time 2에 요청 0이 블록 3을 채우면서 블록 3이 캐시되고, 디코딩을 이어 가도록 블록 4가 할당됩니다.

![요청 1의 프롬프트가 앞 두 블록에서만 캐시에 적중한 Time 3 상태](https://docs.vllm.ai/en/stable/assets/design/prefix_caching/example-time-4.png)

Time 3의 요청 1은 프롬프트 14토큰 가운데 앞 10토큰이 요청 0과 같습니다. 그런데 적중은 앞 2블록, 8토큰에서 멈춥니다. 세 번째 블록은 4토큰 중 2토큰만 일치하기 때문입니다. 블록 단위 키에서는 일부만 겹친 블록이 적중으로 이어지지 않습니다.

![요청 0이 끝난 뒤 블록 2·3·4만 free queue로 들어간 Time 4 상태](https://docs.vllm.ai/en/stable/assets/design/prefix_caching/example-time-5.png)

Time 4에 요청 0이 끝나면 블록 2, 3, 4가 역순으로 free queue에 들어가는데, 블록 2와 3은 여전히 캐시된 상태입니다. 블록 0과 1은 요청 1이 쓰고 있어 큐에 들어가지 않습니다. Time 5에는 요청 1도 끝납니다.

![요청 2가 적중 블록을 touch한 뒤 새 블록을 받고 캐시 블록 하나가 축출된 Time 6 상태](https://docs.vllm.ai/en/stable/assets/design/prefix_caching/example-time-7.png)

Time 6에 들어온 요청 2는 프롬프트 앞부분이 요청 0과 같습니다. 할당 전 free queue는 `7 - 8 - 9 - 4 - 3 - 2 - 6 - 5 - 1 - 0`이었지만, 적중 블록 0, 1, 2를 touch해 큐에서 빼고 나면 `7 - 8 - 9 - 4 - 3 - 6 - 5`가 남습니다. 나머지 5블록은 head부터 꺼내므로 최종 할당은 0(캐시), 1(캐시), 2(캐시), 7, 8, 9, 4, 3(축출)입니다. touch가 할당보다 먼저 일어나서, 큐 중간에 있던 블록 2가 head 쪽 블록보다 먼저 꺼내지는 일은 없습니다.

### append-only 블록 테이블이 남기는 중복 블록

V1의 블록 테이블은 append-only라서, 같은 내용의 블록이 두 번 캐시되는 경우가 생깁니다. 예를 들어 블록 크기 4에서 요청 1이 프롬프트 `ABCDEF`로 `GHI`를 생성하면 블록 0(`ABCD`)과 블록 1(`EFGH`)이 캐시됩니다. 같은 요청을 greedy 샘플링으로 다시 보내면 요청 2는 블록 0에 적중하지만, `EF`만 든 두 번째 블록은 가득 차지 않아 적중하지 못하고 새 블록 3을 받습니다. 디코딩으로 블록 3이 `EFGH`로 차면 블록 1과 같은 키로 한 번 더 캐시됩니다. v0은 이 중복을 감지하면 블록 3을 해제하고 블록 테이블을 `[0, 1]`로 바꿨지만, V1에서는 `[0, 3]`을 `[0, 1]`로 고칠 수 없습니다. 그래서 키 E-H의 블록이 둘 존재하다가, 요청이 해제될 때 중복이 사라집니다.

### 캐시를 건너뛰는 prompt logprobs와 해시 알고리즘 선택

프리픽스 캐싱을 켜 둔 상태에서도 prompt logprobs를 요청할 수 있지만, V1은 logprobs를 캐시하지 않습니다. prompt logprobs가 필요한 요청이 오면 엔진은 프리픽스 캐시를 무시하고 프롬프트 전체의 prefill을 다시 계산합니다. 평가 파이프라인처럼 prompt logprobs를 대량으로 뽑는 작업에서는 공유 프리픽스가 있어도 캐시 이득이 사라집니다.

블록 키를 만드는 해시는 `vllm serve`의 `--prefix-caching-hash-algo`로 고르며, v0.11부터 기본값인 `sha256`은 이전 버전에서 보장되지 않던 충돌 위험을 해소합니다. 선택지는 넷입니다.

| 알고리즘 | 직렬화 | 특징 |
|---|---|---|
| `sha256` | Python `pickle` | 기본값입니다. Python이나 vLLM 버전이 다르면 해시가 재현되지 않을 수 있습니다. |
| `sha256_cbor` | `cbor2` | 재현 가능하고 언어 간 호환되는 해시입니다. 환경 간 결정적 캐싱에 권장됩니다. |
| `xxhash` | Pickle | 128비트 xxHash로 더 빠른 비암호학적 해시입니다. 선택 패키지 `xxhash`가 필요합니다. |
| `xxhash_cbor` | 정규 CBOR | xxHash로 재현 가능한 해시를 만듭니다. 선택 패키지 `xxhash`가 필요합니다. |

`xxhash`나 `xxhash_cbor`를 고르려면 먼저 `uv pip install xxhash`로 이 선택 패키지를 설치합니다. 암호학적으로 안전하지 않은 해시는 이론상 충돌 위험을 키우고, 충돌은 정의되지 않은 동작이나 멀티테넌트 환경의 정보 유출로 이어질 수 있습니다. 앞서 본 `cache_salt` 격리도 해시가 충돌하지 않아야 의미가 있으므로, salt로 테넌트를 나누는 서버라면 `xxhash` 계열의 속도 이득과 보안 위험을 함께 따져 봐야 합니다. 여러 인스턴스가 같은 키를 만들어야 하는 구성이라면, 실행 예제 서버에 CBOR 직렬화를 지정해 두면 됩니다.

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --prefix-caching-hash-algo sha256_cbor
```

예전 버전에서는 선택지가 달랐습니다. v0.8.3에서는 Python 내장 해시인 `builtin`과, 충돌에 강하지만 오버헤드가 있는 `sha256` 가운데서 골랐습니다.

![KV 캐시 매니저 초기화 시의 구성요소(블록 풀, free queue, cache blocks, request blocks) 개요도.](https://docs.vllm.ai/en/latest/assets/design/prefix_caching/overview.png)

![요청이 free된 뒤 free queue 상태: 블록이 역순으로 tail에 추가된다.](https://docs.vllm.ai/en/latest/assets/design/prefix_caching/free.png)

![Time 6 그림: 요청 2의 할당 후 블록 상태(적중 블록 0,1,2와 새로 할당·evict된 블록).](https://docs.vllm.ai/en/latest/assets/design/prefix_caching/example-time-7.png)

## uv 설치 한 줄에서 오프라인 배치 추론과 OpenAI 호환 서버까지 가는 Quickstart

NVIDIA GPU가 있는 Linux 머신에서는 명령 네 줄로 설치를 마치고, 같은 모델 하나로 Python 스크립트 안의 배치 생성과 HTTP 서버를 모두 돌릴 수 있습니다.

### uv 가상환경과 FlashInfer 커널 다운로드

필요한 환경은 Linux와 Python 3.11~3.14입니다. 다른 하드웨어용 설치 경로는 이어지는 하드웨어별 설치 경로에서 다룹니다. uv로 가상환경을 만든 뒤 vLLM을 설치하고, 마지막으로 사전 컴파일 커널을 내려받습니다.

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
vllm download-kernels
```

`--torch-backend=auto`를 주면 uv가 설치된 CUDA 드라이버 버전을 보고 맞는 PyTorch 인덱스를 고릅니다. 특정 백엔드를 원하면 `--torch-backend=cu126`처럼 지정하고, 환경 변수 `UV_TORCH_BACKEND`도 같은 값을 받습니다.

마지막 줄은 생략하기 쉬워도 빼지 않는 편이 좋습니다. FlashInfer 자체는 vLLM과 함께 설치되지만 사전 컴파일 커널은 포함되지 않습니다. `vllm download-kernels`는 설치된 FlashInfer와 CUDA 버전에 맞는 커널을 받아 둡니다. 이 커널이 없으면 vLLM이 시작할 때 다운로드와 컴파일을 하는데, Hopper 이후 GPU에서는 이 과정에 몇 분이 더 걸릴 수 있습니다. vLLM을 새로 설치하거나 업그레이드할 때마다 다시 실행해야 시작 시 컴파일을 피할 수 있습니다.

### LLM 클래스로 돌리는 오프라인 배치 생성

서버 없이 프롬프트 목록을 한꺼번에 처리할 때는 `LLM`과 `SamplingParams` 두 클래스를 씁니다. `LLM`은 vLLM 엔진으로 오프라인 추론을 실행하는 주 클래스이고, `SamplingParams`에는 샘플링 설정을 담습니다. `LLM`을 생성하는 순간 엔진이 초기화되고 모델이 로드됩니다.

실행 예제에서는 `chat`을 씁니다.

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")

messages_list = [
    [{"role": "user", "content": prompt}]
    for prompt in prompts
]
outputs = llm.chat(messages_list, sampling_params)

for idx, output in enumerate(outputs):
    prompt = prompts[idx]
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

결과는 `RequestOutput` 객체의 리스트로 돌아오고, 각 객체에 출력 토큰이 모두 들어 있습니다. 템플릿이 필요 없는 베이스 모델이라면 마지막 호출을 `llm.generate(prompts, sampling_params)`로 바꾸고 `output.prompt`로 프롬프트를 읽으면 됩니다.

샘플링 기본값은 미리 확인해 두는 편이 좋습니다. 모델 저장소에 `generation_config.json`이 있으면 모델 제작자가 권장한 샘플링 파라미터가 기본으로 적용됩니다. vLLM 자체 기본값을 쓰려면 `LLM` 인스턴스를 만들 때 `generation_config="vllm"`을 설정합니다. `SamplingParams`를 생략했는데 실행 환경마다 출력 분포가 달라 보인다면 이 설정부터 의심해 볼 만합니다.

### `vllm serve` 한 줄로 띄우는 OpenAI 호환 서버

같은 모델을 HTTP로 제공하려면 명령 하나면 됩니다.

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

이 서버는 OpenAI API 프로토콜을 구현하므로, OpenAI API를 쓰던 애플리케이션이 주소만 바꿔 연결할 수 있습니다. 기본 주소는 `http://localhost:8000`이고 `--host`와 `--port`로 바꿉니다. 서버 하나는 한 번에 모델 하나만 호스팅합니다. 오프라인 경로와 마찬가지로 서버도 모델 저장소의 `generation_config.json` 값을 기본으로 적용하며, 끄려면 시작할 때 `--generation-config vllm`을 넘깁니다.

서버가 뜨면 Completions 엔드포인트에 바로 요청을 보낼 수 있습니다. 아래 요청은 `max_tokens`를 7로, `temperature`를 0으로 둡니다.

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "prompt": "San Francisco is a",
    "max_tokens": 7,
    "temperature": 0
  }'
```

Chat Completions는 `/v1/chat/completions`에 `messages` 배열을 보내는 방식입니다.

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
 애플리케이션 코드에서는 `openai` Python 패키지의 `base_url`만 vLLM 서버로 돌리면 되고, API 키 자리에는 `"EMPTY"`를 넣습니다. 아래 스크립트는 두 엔드포인트를 차례로 호출합니다.

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
    model="Qwen/Qwen2.5-1.5B-Instruct",
    prompt="San Francisco is a",
)
print("Completion result:", completion)

chat_response = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ],
)
print("Chat response:", chat_response)
```

두 호출 모두 `model` 필드에는 `vllm serve`에 넘긴 저장소 이름을 그대로 씁니다.

## 소스 빌드·Docker·CPU·Apple Silicon으로 넓히는 하드웨어별 설치 경로

NVIDIA용 wheel이 맞지 않는 환경이라면 하드웨어마다 정해진 설치 경로를 따릅니다. 공식 설치 경로가 있는 GPU는 NVIDIA CUDA, AMD ROCm, Intel XPU, 그리고 vLLM-Metal을 거치는 Apple Silicon입니다. CPU 백엔드는 Intel/AMD x86, ARM AArch64, Apple silicon, IBM Z(S390X)를 지원합니다. V1 기준으로 NVIDIA, AMD, Intel GPU, TPU, CPU가 지원 대상이고, Ascend NPU 같은 그 밖의 하드웨어는 `vllm-ascend`, `vllm-spyre`, `vllm-gaudi`, `vllm-openvino` 같은 플러그인이 맡습니다. 이 플러그인은 메인 `vllm` 저장소 밖에서 관리되며 Hardware-Pluggable RFC를 따릅니다.

| 구분 | 설치 경로 | 비고 |
| --- | --- | --- |
| NVIDIA, Python만 수정 | `VLLM_USE_PRECOMPILED=1` 편집 가능 설치 | C++·커널 수정 불가 |
| NVIDIA, 배포 | `vllm/vllm-openai` 이미지 | `--ipc=host` 또는 `--shm-size` |
| AMD ROCm | `--extra-index-url https://wheels.vllm.ai/rocm/` | Python 3.12, ROCm 7.0 |
| Google TPU | `vllm-tpu` 패키지 | 별도 패키지 |
| x86 CPU | `+cpu` wheel, 소스 빌드, `vllm/vllm-openai-cpu` 이미지 | `LD_PRELOAD` 설정 |
| Apple Silicon GPU | vLLM-Metal | MLX 모델 필요 |

### 컴파일 없이 Python만 고치는 소스 빌드

vLLM의 Python 코드만 수정하려면 컴파일 없이 편집 가능 모드로 설치합니다.

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 uv pip install --editable . --torch-backend=auto
vllm download-kernels
```

이 명령은 현재 브랜치에 대응하는 main 브랜치의 기준 커밋을 찾고, 그 커밋의 사전 빌드 wheel을 내려받아 컴파일된 라이브러리와 `vllm-rs` 바이너리를 설치에 씁니다. 그래서 C++나 커널 코드를 바꾼 경우에는 이 방식을 쓸 수 없습니다. 그대로 쓰면 import 시점에 라이브러리를 찾지 못하거나 undefined symbol 오류가 납니다. 개발 브랜치를 rebase했다면 vLLM을 지우고 같은 명령을 다시 실행해 라이브러리를 최신으로 맞추라고 권합니다.

### 컨테이너로 옮기는 GPU 서버와 공유 메모리

배포용으로는 Docker Hub의 공식 이미지 `vllm/vllm-openai`가 OpenAI 호환 서버를 바로 띄웁니다. 이미지 태그 뒤에 엔진 인자를 붙이는 구조라서, 실행 예제 모델로 바꾸면 명령이 이렇게 됩니다.

```bash
docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=$HF_TOKEN" \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-1.5B-Instruct
```

`--ipc=host`는 빠뜨리기 쉬운 옵션입니다. vLLM이 쓰는 PyTorch는 프로세스 간 데이터 공유, 특히 텐서 병렬 추론에 공유 메모리를 쓰므로, 컨테이너가 호스트 공유 메모리에 접근하도록 `ipc=host`나 `--shm-size` 중 하나를 줘야 합니다. 같은 이미지는 Podman 같은 다른 컨테이너 엔진에서도 실행됩니다.

AMD GPU에서는 uv에 ROCm 인덱스를 추가해 설치하며, 현재 Python 3.12, ROCm 7.0, `glibc >= 2.35`를 지원합니다. uv를 권하는 이유는 추가 인덱스에 기본 인덱스보다 높은 우선순위를 주기 때문입니다. Google TPU는 이름이 다른 패키지를 씁니다.

```bash
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/
uv pip install vllm-tpu
```

### CPU 서버의 wheel·소스 빌드·이미지와 환경 변수

x86 CPU용 사전 빌드 wheel은 AVX512/AVX2 대상으로 0.17.0부터 나옵니다. 최신 릴리스 번호를 받아 `+cpu` wheel을 설치합니다.

```bash
export VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest | jq -r .tag_name | sed 's/^v//')

uv pip install "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_34_x86_64.whl" --torch-backend cpu
```

wheel로 설치했다면 실행 전에 Intel OpenMP 라이브러리 `libiomp5.so`의 경로를 찾아 `LD_PRELOAD`에 추가해야 하고, 소스 빌드라면 TCMalloc도 함께 넣습니다. 소스 빌드는 클론한 저장소에서 빌드 의존성을 설치한 다음 `VLLM_TARGET_DEVICE=cpu uv pip install . --no-build-isolation`으로 끝납니다. macOS에서는 `VLLM_TARGET_DEVICE`가 자동으로 `cpu`로 잡히며, 현재 지원되는 장치는 CPU뿐입니다. 컨테이너를 원하면 `vllm/vllm-openai-cpu:latest-x86_64` 이미지를 받아 GPU 이미지처럼 태그 뒤에 인자를 붙여 실행합니다.

CPU에서는 환경 변수로 KV 캐시 크기를 정할 수 있습니다. 실행 예제 서버를 32코어 머신에 올린다면, 서빙 프레임워크 몫으로 코어 하나를 남기고 0~30번 코어에 추론 스레드를 묶습니다.

```bash
export VLLM_CPU_KVCACHE_SPACE=40
export VLLM_CPU_OMP_THREADS_BIND=0-30
vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype=bfloat16
```

`VLLM_CPU_KVCACHE_SPACE`는 KV 캐시 크기를 GiB 단위로 지정하며, 값이 클수록 더 많은 요청을 병렬로 처리합니다. 기본값 `0`은 이 변수로 KV 캐시 크기를 덮어쓰지 않는다는 뜻입니다. 다만 TP rank 하나의 메모리 사용량은 가중치 샤드 크기와 이 값의 합이고, 이 합이 NUMA 노드 하나의 용량을 넘으면 워커가 `exitcode 9`로 종료됩니다. 이 변수는 레거시 설정이므로 새 구성에서는 `--kv-cache-memory-bytes`를 씁니다. `--dtype=bfloat16`을 명시한 이유는 torch CPU의 float16 지원이 불안정해서입니다. vLLM CPU는 기본적으로 모델의 dtype을 쓰지만, 성능이나 정확도 문제가 보이면 bfloat16을 지정하라고 권합니다.

Docker나 Kubernetes에서는 `get_mempolicy: Operation not permitted` 경고가 뜰 수 있습니다. 기능에는 영향이 없지만 NUMA 메모리 바인딩 최적화가 적용되지 않아 성능이 떨어질 수 있으며, `--cap-add SYS_NICE`와 `--security-opt seccomp=unconfined`로 풉니다.

### MLX 위에서 도는 Apple Silicon의 vLLM-Metal

Mac의 GPU를 쓰려면 별도 패키지인 vLLM-Metal을 설치합니다. 설치 스크립트가 Python 환경 구성, MLX와 의존성 설치, vLLM-Metal 패키지 설치를 차례로 진행합니다. vLLM-Metal은 PyTorch 대신 MLX를 연산 백엔드로 쓰므로, 실행 예제의 Hugging Face 저장소 대신 mlx-community의 MLX 최적화 모델이 필요합니다. 서버는 전용 가상환경을 활성화한 뒤 `vllm serve`로 시작하며, 모델을 지정하지 않으면 기본 모델이 쓰입니다. 대화형으로 바로 확인하려면 같은 환경에서 `vllm chat`을 실행합니다.

```bash
source ~/.venv-vllm-metal/bin/activate

# Start the API server (specify your mlx-community model or it will use default)
vllm serve
```

`INFO: Application startup complete.`가 출력될 때까지 2~3분 정도 걸립니다. 그 뒤에는 앞서 만든 openai 클라이언트 코드에서 `model` 이름만 MLX 모델로 바꾸면 됩니다.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",  # No auth required for local server
)

response = client.chat.completions.create(
    model="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    messages=[{"role": "user", "content": "Hello!"}],
)

print(response.choices[0].message.content)
```

## EngineArgs로 맞추는 컨텍스트 길이·양자화·병렬화·GPU 메모리 비율

서빙 동작을 바꾸는 손잡이는 엔진 인자 하나로 모여 있습니다. 엔진 인자는 vLLM 엔진의 동작을 제어합니다. 오프라인 추론에서는 `LLM` 클래스의 인자로, 온라인 서빙에서는 `vllm serve`의 인자로 들어갑니다. 따라서 앞서 만든 오프라인 배치 스크립트와 `vllm serve` 서버는 같은 인자 집합으로 튜닝됩니다. 인자 클래스인 `EngineArgs`와 `AsyncEngineArgs`는 `vllm.config`에 정의된 설정 클래스들을 합쳐 만든 것입니다. 타입, 기본값, 설명의 기준도 이 설정 클래스들입니다. 인자마다 `ModelConfig`, `CacheConfig`, `ParallelConfig` 같은 소속 클래스가 정해져 있습니다.

예를 들어 실행 예제 서버의 컨텍스트 길이, 메모리 비율, 어텐션 백엔드를 한꺼번에 지정하면 다음처럼 됩니다. 각 값은 아래에서 하나씩 설명합니다.

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --max-model-len 8k \
  --gpu-memory-utilization 0.85 \
  --attention-backend FLASH_ATTN
```

### 모델과 연산 경로를 정하는 인자

`--max-model-len`은 프롬프트와 출력을 합친 컨텍스트 길이입니다. 값을 주지 않으면 모델 설정에서 자동으로 정해집니다. `1k`, `1K`, `25.6k` 같은 사람이 읽기 쉬운 표기도 받습니다. `-1`이나 `auto`를 주면 GPU 메모리에 들어가는 최대 길이를 고릅니다. 모델의 최대 컨텍스트가 들어가면 그 값을 쓰고, 들어가지 않으면 수용 가능한 가장 긴 길이를 찾습니다.

`--quantization`(`-q`)은 가중치 양자화 방식입니다. `None`이면 먼저 모델 설정 파일의 `quantization_config`를 확인합니다. 그것도 없으면 가중치가 양자화되지 않았다고 보고 `dtype`으로 가중치 자료형을 정합니다. 사전 양자화된 체크포인트라면 보통 인자를 생략해도 설정 파일이 방식을 알려 줍니다. v0.4.1에서는 선택지가 aqlm, awq, fp8, gptq, squeezellm, marlin이었습니다.

`--attention-backend`는 `auto`나 `None`일 때 자동으로 선택됩니다. 이때 vLLM은 시스템과 모델에 호환되는 백엔드 중 가장 성능이 좋은 것을 고릅니다. NVIDIA CUDA에서는 `FLASH_ATTN`이나 `FLASHINFER`를 직접 지정할 수 있고, AMD ROCm과 Intel XPU에는 각각 다른 선택지가 있습니다. 서빙에서는 위 명령처럼 플래그를 붙입니다. 앞서 만든 오프라인 배치 스크립트는 `LLM(...)`을 직접 만들고 명령줄 인자를 읽지 않습니다. 그래서 같은 값을 `LLM(...)`을 만들 때 엔진 인자로 넘겨야 합니다.

`--pipeline-parallel-size`(`-pp`)는 파이프라인 병렬 그룹 수이고 기본값은 `1`입니다.

`--block-size`는 연속된 캐시 블록 하나에 담기는 토큰 수입니다. `None`을 받으면 기본값을 쓰고, 생성이 끝난 뒤에는 항상 정수가 됩니다. 크기를 정할 때의 트레이드오프는 앞서 본 GPU 병렬성 대 내부 단편화의 문제 그대로입니다. 이전 버전에서는 범위가 좁았습니다. v0.4.1의 선택지는 8, 16, 32였고 기본값은 16이었습니다. v0.7.3에서는 CUDA 장치에서 32 이하만 지원했습니다.

### 시작 시점에 KV 캐시 풀을 정하는 메모리 비율

`--gpu-memory-utilization`은 모델 실행기가 쓸 GPU 메모리 비율로, 0에서 1 사이 값이며 기본값은 `0.92`입니다. 이 값은 현재 vLLM 인스턴스에만 걸리는 한도라서 같은 GPU에 다른 인스턴스가 떠 있어도 상관없습니다. 한 GPU에 인스턴스 두 개를 올린다면 각각 `0.5`를 주면 됩니다. GPU가 아닌 설치에서는 해당 장치 메모리의 비율을 정합니다.

이 비율이 동시성과 어떻게 이어지는지는 ScaleOps의 L4 GPU 관측에서 볼 수 있습니다. `0.90`으로 띄운 서버는 유휴 상태든 포화 상태든 DCGM 기준 약 23 GiB를 계속 점유했는데, KV 캐시 풀을 필요할 때 키우지 않고 시작 시 한 번에 예약하기 때문입니다. 같은 글은 vLLM v0.8.5에서 비율을 바꿔 가며 KV 용량을 측정했습니다.

| `--gpu-memory-utilization` | KV 용량(토큰) | 결과 |
| --- | --- | --- |
| 0.90 | 33,616 | 정상 기동 |
| 0.95 | 42,640 | 정상 기동 |
| 0.98 | 48,048 | 정상 기동 |
| 0.99 | 없음 | 샘플러 워밍업 중 크래시 |

0.90에서 0.98로 올리자 동시성이 43% 늘었지만, 0.99에서는 기동하지 못했습니다. 반대로 `--gpu-memory-utilization`을 기본값보다 낮추라는 권장도 있습니다. 그만큼 KV 캐시 용량은 줄지만, 동시 부하에서 KV 캐시가 급증할 때 OOM으로 프로세스가 죽는 일을 막는 여유가 생긴다는 설명입니다. 같은 GPU에 다른 프로세스가 없다면 기본값 근처에서 시작해 기동과 부하 테스트로 상한을 확인하고, 다른 프로세스와 GPU를 나눠 쓴다면 낮은 값부터 잡는 편이 안전합니다.

### 샘플링 이전 값을 돌려주는 logprobs 기본 동작

V1에서 반환되는 logprobs는 기본적으로 temperature나 페널티 같은 후처리를 적용하기 전의 값이라, 실제 샘플링에 쓰인 확률과 다를 수 있습니다. `--logprobs-mode`로 이 동작을 바꿀 수 있으며 지원 모드는 넷입니다.

- `raw_logprobs`: 기본값이며, 금지어 같은 로짓 처리기를 적용하기 전의 logprobs입니다.
- `processed_logprobs`: temperature와 top_k/top_p를 포함해 모든 처리기를 적용한 뒤의 logprobs입니다.
- `raw_logits`: 처리기를 적용하기 전의 로짓입니다.
- `processed_logits`: 모든 처리기를 적용한 뒤의 로짓입니다.

temperature를 적용해 샘플링하면서 반환 확률을 평가나 보정에 쓰려면 `processed_` 계열을 고릅니다.

## /metrics의 KV 캐시 사용률·선점·TTFT로 하는 관측과 Kubernetes 오토스케일링

vLLM 서버의 상태는 VRAM 그래프에 드러나지 않습니다. `/metrics`가 노출하는 KV 캐시 사용률, 대기 요청 수, 선점 횟수를 봐야 상태를 알 수 있습니다. 이 지표는 OpenAI 호환 API 서버의 `/metrics` 엔드포인트로 나옵니다. V1에서는 Prometheus 호환 형식에 `vllm:` 접두사를 붙여 노출합니다. 이 형식으로 내보내는 컴포넌트는 `PrometheusStatLogger`입니다. Prometheus가 이 엔드포인트를 주기적으로(예를 들어 1초마다) 폴링해 시계열로 저장하며, 모든 지표에는 서빙 중인 모델 이름이 `model_name` 레이블로 붙습니다.

```bash
curl http://0.0.0.0:8000/metrics
```

### 먼저 볼 다섯 가지 신호

지표 타입은 셋입니다. Counter는 증가만 하다가 인스턴스가 재시작되면 0으로 돌아갑니다. Gauge는 값이 오르내립니다. Histogram은 표본을 버킷별로 셉니다. 운영에서 먼저 볼 지표를 표로 정리합니다.

| 지표 | 타입 | 의미 |
| --- | --- | --- |
| `vllm:kv_cache_usage_perc` | Gauge | KV 캐시 사용률(1이 100%) |
| `vllm:num_requests_running` | Gauge | 모델 실행 배치에 들어간 요청 수 |
| `vllm:num_requests_waiting` | Gauge | 처리를 기다리는 요청 수 |
| `vllm:num_preemptions` | Counter | 엔진의 누적 선점 횟수 |
| `vllm:time_to_first_token_seconds` | Histogram | 첫 토큰까지의 시간(초) |
| `vllm:prefix_cache_hits` | Counter | 프리픽스 캐시 적중(캐시된 토큰 수) |

프리픽스 캐시 적중률은 일부러 Gauge로 두지 않았습니다. 질의와 적중을 각각 Counter로 기록해 두면 사용자가 원하는 구간의 적중률을 PromQL로 직접 계산할 수 있기 때문입니다. 5분 구간이라면 다음처럼 계산합니다.

```text
rate(cache_query_hit[5m]) / rate(cache_query_total[5m])
```

vLLM에서는 분자에 `vllm:prefix_cache_hits`, 분모에 같은 토큰 단위 Counter인 `vllm:prefix_cache_queries`를 넣습니다.

앞서 본 LRU 축출이 실제로 어떻게 돌아가는지는 KV 블록 체류 지표로 확인합니다. 블록 수명(`vllm:kv_block_lifetime_seconds`), 마지막 접근 뒤 축출까지의 유휴 시간(`vllm:kv_block_idle_before_evict_seconds`), 연속 접근 사이 간격(`vllm:kv_block_reuse_gap_seconds`)이 각각 히스토그램으로 나옵니다. 표본 추출은 `--kv-cache-metrics-sample`이 제어합니다. 엔진 코어는 `SchedulerStats`로 원시 축출 이벤트만 보내고, 프런트엔드가 이를 Prometheus 관측값으로 바꿉니다. 로깅이 켜져 있으면 같은 데이터를 `LLM.get_metrics()`로도 볼 수 있습니다. 수명과 유휴 시간을 한 차트에 겹쳐 그리면 방치된 캐시나 긴 decode 동안 프롬프트를 붙잡아 두는 워크로드가 잘 보입니다.

### 엔진 코어 타임스탬프로 나누는 지연 구간

TTFT나 prefill 시간 같은 구간 지표는 엔진 코어가 남기는 요청별 이벤트 타임스탬프로 계산합니다. 프런트엔드는 `QUEUED`와 `SCHEDULED` 시점을 볼 수 없습니다. 게다가 구간은 같은 프로세스의 단조 시계로 재야 하므로 엔진 코어가 모든 이벤트의 시각을 기록합니다. 이벤트는 넷입니다. `QUEUED`는 스케줄러 큐에 들어간 시점, `SCHEDULED`는 처음 실행에 배정된 시점입니다. `PREEMPTED`는 대기 큐로 되돌아간 시점으로, 이후 다시 스케줄되면 prefill부터 새로 시작합니다. `NEW_TOKENS`는 출력이 생성된 시점입니다. 큐 구간은 `QUEUED`부터 가장 최근 `SCHEDULED`까지이고, prefill 구간은 그 `SCHEDULED`부터 첫 `NEW_TOKENS`까지입니다. decode 구간은 첫 `NEW_TOKENS`부터 마지막 `NEW_TOKENS`까지입니다.

아래 그림에서 각 구간이 어느 이벤트 쌍 사이에 걸치는지 확인할 수 있습니다.

![선점이 없는 요청의 큐·prefill·decode·토큰 간 구간 계산도](https://docs.vllm.ai/en/stable/assets/design/metrics/intervals-1.png)

앞서 본 재계산 기반 선점이 끼어들면 구간을 계산하는 기준이 바뀝니다. decode 중에 선점되면 이미 생성한 토큰은 재사용되므로, 그 선점은 토큰 간·decode·inference 구간에 반영됩니다.

![decode 도중 선점된 요청의 구간 계산도](https://docs.vllm.ai/en/stable/assets/design/metrics/intervals-2.png)

prefill 중에 선점되면 TTFT와 prefill 구간에 반영됩니다. 이 그림에서는 재스케줄 이후의 prefill이 첫 토큰 시점을 얼마나 뒤로 미는지 보면 됩니다.

![prefill 도중 선점된 요청의 구간 계산도](https://docs.vllm.ai/en/stable/assets/design/metrics/intervals-3.png)

그래서 선점 Counter가 오르는 시점에 TTFT 꼬리가 같이 길어진다면 같은 원인에서 나온 증상으로 읽으면 됩니다.

### 버킷·MFU·폐기 정책을 다루는 플래그

코어 엔진 히스토그램은 `--custom-histogram-buckets`로 버킷 경계를 패밀리 단위로 교체합니다. 요청 단계 히스토그램은 기본 최소 경계가 0.3초라서, 300ms 미만 SLO를 추적하려면 경계를 바꿔야 합니다.

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --custom-histogram-buckets '{"request_latency": [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0, 30.0]}'
```

값은 양수이고 유한해야 하며 엄격히 증가해야 합니다. 알 수 없는 패밀리 키는 시작 시 거부됩니다. 경계 하나가 레이블 조합마다 시계열을 하나씩 더 만들기 때문에, 실제로 모니터링하는 패밀리만 짧은 목록으로 바꾸는 편이 좋습니다. MFU 계산용 FLOPs·메모리 읽기/쓰기 바이트 Counter는 `--enable-mfu-metrics`를 켜야 나옵니다. X.Y에서 폐기 예정이 된 지표는 X.Y+1에서 숨겨지고, `--show-hidden-metrics-for-version=X.Y`로 다시 켤 수 있으며, X.Y+2에서 제거됩니다. 대시보드와 알림 규칙은 업그레이드 두 번 안에 새 이름으로 옮겨야 합니다.

### VRAM이 아닌 KV 사용률로 하는 확장

같은 글은 오토스케일링에는 `vllm:kv_cache_usage_perc`와 `vllm:num_requests_waiting`을 쓰고, VRAM은 용량 계획에만 쓰며 VRAM에는 알림을 걸지 말라고 권합니다.

포화 상태는 장애로 드러나지 않습니다. 그 상태로 스케줄러가 선점을 이어 가면서 `vllm:num_preemptions_total`이 0에서 887로 올랐지만 파드는 재시작 없이 계속 서빙했습니다. 콜드 스타트 비용은 ScaleOps 테스트에서 파드만 6.3분, 새 노드에서 이미지를 처음 받을 때는 처음부터 끝까지 약 11분이었습니다.

정리하면 알림은 선점 Counter의 증가율과 TTFT 꼬리에 겁니다. 새 파드가 준비되기까지 수 분이 걸리므로, 그동안 기존 파드가 선점으로 버티는 구간을 감안해 임계값을 여유 있게 잡아야 합니다.

## vLLM을 고를 때와 피할 때: 처리량 이득과 V1의 제약

vLLM은 KV 캐시 메모리가 배치 크기를 묶는 고동시성 서빙에서 가장 큰 이득을 줍니다. 대신 V1에서 사라진 기능, CUDA graph가 차지하는 메모리, 긴 콜드 스타트, 서버당 모델 하나라는 조건을 받아들여야 합니다.

### 메모리가 병목인 워크로드에서 커지는 이득

PagedAttention은 낭비를 요청당 마지막 블록 하나로 줄여 같은 GPU에 더 많은 시퀀스를 배칭합니다. 한 요청이 여러 출력을 만드는 병렬 샘플링과 빔 서치에서는 블록 공유로 메모리를 더 아낍니다. 시스템 프롬프트처럼 프리픽스가 겹치는 요청은 자동 프리픽스 캐싱으로 앞부분 블록을 재사용합니다. 그래서 다음 워크로드라면 vLLM을 먼저 검토할 만합니다.

- 동시 요청이 많고 응답 길이가 제각각인 채팅·API 서빙
- `n`개 샘플을 뽑거나 빔 서치를 쓰는 생성
- 긴 공통 시스템 프롬프트나 few-shot 예시를 붙이는 에이전트·RAG 요청

반대로 요청이 드물게 하나씩 들어오는 환경에서는 배칭할 대상이 적어서, 메모리 관리의 이점이 처리량으로 이어질 여지도 작습니다.

### HF Transformers·TGI·FasterTransformer·Orca와 비교하면

| 대안 | vLLM과 비교한 결과 |
| --- | --- |
| HF Transformers | 모델 구조 변경 없이 최대 24배 처리량입니다. |
| TGI | 출력 1개 요청 기준으로 앞서 본 2.2~2.5배 처리량입니다. |
| FasterTransformer, Orca | 정확도 손실 없이 2~4배 처리량입니다. |

이 수치는 논문과 출시 블로그 시점의 비교입니다. 지금 비교하려면 대상 시스템의 최신 버전과 실제 트래픽 분포로 다시 측정해야 합니다. LMSYS가 GPU 수를 절반으로 줄인 앞서 본 사례도 그들 트래픽에서 나온 결과입니다.

### V1에서 사라지거나 바뀐 기능

V0 시절 코드를 옮길 때는 몇 가지 기능이 없어졌습니다. `best_of`는 사용이 적어 제거되었습니다. 요청마다 넘기던 logits processor도 제거되었고, 그 자리를 시작 시점에 설정하는 전역 logits processor가 대신합니다. 요청별로 로짓을 다르게 조작하던 애플리케이션은 설계를 다시 해야 합니다. prompt logprobs를 요청하면 프리픽스 캐시를 건너뛰고 프롬프트 전체를 다시 prefill합니다. 평가 파이프라인처럼 프롬프트 logprobs를 대량으로 뽑는 작업에서는 캐시 이득을 기대하기 어렵습니다.

### CUDA graph 메모리와 시작 시간의 맞교환

V1에서는 CUDA graph 캡처가 V0보다 메모리를 더 씁니다. 이 메모리는 앞서 다룬 `--gpu-memory-utilization` 안에서 KV 캐시 풀과 같은 공간을 나눠 씁니다. 2026년 6월 29일자 Kubernetes 배포 가이드(vLLM on Kubernetes)는 `--enforce-eager`로 캡처를 건너뛰면 메모리 사용과 시작 시간이 줄지만, 정상 상태의 토큰당 추론은 다소 느려진다고 보고합니다. 파드가 자주 뜨고 내려가는 환경이라면 시작 시간을, 오래 떠 있는 서버라면 정상 상태 속도를 우선하는 쪽으로 기웁니다. 앞서 본 수 분 단위의 콜드 스타트는 이 선택과 별개로 이미지 풀과 모델 로드에서도 생깁니다.

### 한계와 실패 양상

하나의 서버는 한 번에 모델 하나만 호스팅합니다. 여러 모델이나 LoRA가 아닌 별개 체크포인트를 섞어 서빙하려면 인스턴스를 나누고 앞단에 라우터를 둬야 합니다. 과부하가 걸려도 프로세스가 죽지는 않습니다. KV 캐시 풀이 100%까지 찬 뒤에는 선점이 쌓이면서 지연이 길어지는 방식으로 나빠집니다. 그래서 장애 알림만 걸어 두면 문제를 늦게 발견합니다. 메모리 비율을 한계까지 올리면 서빙 도중이 아니라 시작 단계의 워밍업에서 OOM이 날 수 있습니다. CPU 백엔드에서는 float16이 불안정할 수 있어 bfloat16을 명시해야 합니다. 커널 코드를 고친 뒤 Python 전용 빌드를 쓰면 import 오류가 납니다.

이 글의 판단으로는, Qwen2.5-1.5B-Instruct 같은 모델 하나를 OpenAI 호환 API로 많은 사용자에게 내보내는 용도라면 vLLM이 기본 선택입니다. 요청별 logits 조작이나 프로세스 하나에서 여러 모델을 다루는 기능이 핵심이라면 다른 경로를 먼저 확인하는 편이 낫습니다. 도입한 뒤에는 앞서 정한 KV 캐시 사용률과 선점 지표를 기준으로 용량과 메모리 비율을 조정하면 됩니다.

- - -

### Sources
* [vLLM V1 - vLLM](https://docs.vllm.ai/en/stable/usage/v1_guide/)
* [Architecture Overview - vLLM](https://docs.vllm.ai/en/latest/design/arch_overview/)
* [Automatic Prefix Caching - vLLM](https://docs.vllm.ai/en/stable/design/prefix_caching/)
* [Quickstart - vLLM](https://docs.vllm.ai/en/latest/getting_started/quickstart/)
* [Engine Arguments - vLLM](https://docs.vllm.ai/en/stable/configuration/engine_args/)
* [Automatic Prefix Caching - vLLM](https://docs.vllm.ai/en/latest/design/prefix_caching/)
* [GPU - vLLM](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
* [Installation - vLLM](https://docs.vllm.ai/en/latest/getting_started/installation/)
* [CPU - vLLM](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/)
* [Production Metrics - vLLM](https://docs.vllm.ai/en/latest/usage/metrics/)
* [Metrics - vLLM](https://docs.vllm.ai/en/stable/design/metrics/)
* [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/html/2309.06180)
* [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention &#124; vLLM Blog](https://vllm.ai/blog/2023-06-20-vllm)
* [vLLM on Kubernetes: Deploy, Scale, and Monitor LLM Inference](https://scaleops.com/blog/vllm-kubernetes/)
* [vLLM Kubernetes: 7 Proven Production Patterns 2026](https://thegoodshell.com/vllm-kubernetes/)
* [vLLM Prometheus metrics for self-hosted LLM tuning &#124; Elastic Observability Labs](https://www.elastic.co/observability-labs/blog/tune-vllm-prometheus-metrics-elastic)
* [Engine Arguments — vLLM](https://docs.vllm.ai/en/v0.7.3/serving/engine_args.html)
* [Engine Arguments — vLLM](https://docs.vllm.ai/en/v0.8.3/serving/engine_args.html)
* [Engine Arguments — vLLM](https://docs.vllm.ai/en/v0.4.1/models/engine_args.html)
