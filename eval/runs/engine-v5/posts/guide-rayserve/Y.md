---
layout: post
title: "Ray Serve on Ray Actors: Deployments, Composition, Config, KubeRay"
date: 2026-11-09 15:35:08
description: "Ray Serve를 Ray actor 위에서 다루는 방법을 정리합니다. deployment와 replica, 모델 컴포지션, Serve config 파일, KubeRay RayService 배포 구조와 설정을 설명합니다."
categories: ["Tech Guides", "Data-Architecture"]
tags: ["Ray-Serve", "KubeRay", "Model-Composition", "DeploymentHandle", "Serve-Config", "Model-Serving"]
cover: /assets/images/data-architecture.jpg
---
## TL;DR

Ray Serve는 온라인 추론 API를 만드는 확장 가능한 모델 서빙 라이브러리입니다. 특정 프레임워크에 묶이지 않고 임의의 Python 코드를 그대로 실행합니다. 정적 그래프를 YAML로 정의하는 도구와 다르게, 모델 호출을 함수 호출처럼 코드로 작성합니다. 모델마다 다른 자원을 쓰게 하거나 다른 머신에서 실행할 수도 있습니다. 기본 단위는 deployment입니다. `@serve.deployment`를 붙인 클래스가 deployment가 되고, 런타임에는 별도의 Ray actor 프로세스로 뜬 replica 여러 개로 복제됩니다. 처리량은 이 복제본 개수를 고정하거나 오토스케일링으로 조절합니다. deployment 하나 이상이 모여 업그레이드 단위인 application이 됩니다. 그중 ingress deployment가 모든 인바운드 트래픽을 받습니다. 내부에서는 전역 Controller actor가 다른 actor를 관리하고, 요청은 replica actor에서 실행됩니다. 오토스케일러도 Controller 안에서 돕니다. deployment를 호출하는 핸들(`DeploymentHandle`)과 replica가 보내는 메트릭을 보고 replica 수를 정합니다.

여러 모델을 엮을 때는 `.bind()`로 생성자 인자를 묶은 deployment(bound deployment)를 다른 deployment의 생성자 인자로 넘깁니다. 이 인자는 런타임에 `DeploymentHandle`로 바뀝니다. 그래서 전처리·추론·후처리를 각각 독립된 deployment로 나누고 따로 스케일링할 수 있습니다. 이 글은 영어 텍스트를 요약한 뒤 프랑스어, 독일어, 루마니아어로 번역하는 애플리케이션을 예제로 끝까지 씁니다. 개발 단계에서는 노트북에서 `serve run`으로 띄워 로그를 보며 다듬습니다. 같은 코드를 고치지 않고 여러 머신이나 Kubernetes로 확장할 수 있습니다. 프로덕션에서는 KubeRay의 RayService를 씁니다. RayService는 크래시한 노드와 클러스터를 복구합니다. KubeRay 없이 Ray 클러스터 자체가 실패하면 Serve는 스스로 복구하지 못합니다. Ray Serve는 여러 모델을 Python 코드로 엮고 단계마다 따로 스케일링해야 할 때 잘 맞습니다. 반대로 모델을 더 빠르게 돌리는 모델별 최적화나 모델 수명주기 관리가 필요하다면 Serve만으로는 부족합니다.

## 정적 그래프와 프레임워크 전용 서버로는 풀기 어려운 다중 모델 파이프라인 서빙

여러 모델과 비즈니스 로직을 하나의 온라인 API로 묶으면 단계마다 필요한 자원과 규모가 달라집니다. Ray Serve는 임의의 Python 코드를 Ray 클러스터 전체로 확장하는 프레임워크 비종속 서빙 계층으로 이 문제를 풉니다.

### 단계마다 자원이 다른 추론 파이프라인

실제 서비스는 모델 하나로 끝나는 경우가 드뭅니다. 이미지 처리 애플리케이션은 전처리, 분할, 필터링 같은 여러 단계를 거쳐야 하고, 각 단계의 모델은 아키텍처나 프레임워크가 다를 수 있으며 CPU나 GPU처럼 요구하는 자원도 다릅니다. LLM 애플리케이션도 프롬프트 전처리, 벡터 DB 조회, LLM 호출, 응답 검증을 하나로 엮습니다.

이 글 전체에서 쓰는 Text ML 애플리케이션도 같은 모양입니다. 영어 텍스트를 받아 요약하는 `Summarizer`가 요약 결과를 `Translator`에 넘겨 번역합니다. 다음 그림은 요청 흐름을 개념적으로 그린 예시입니다.

```text
[클라이언트] --POST 영어 텍스트--> [Summarizer] --요약문--> [Translator] --번역문--> [클라이언트]
                                   (요약 모델)                (번역 모델, 기본 프랑스어)
```

예를 들어 긴 문서가 몰리는 시간대에는 요약 단계가 병목이 되고 번역 단계는 놀 수 있습니다. 이때는 두 단계를 한 덩어리로 늘리지 말고 병목이 생긴 단계의 처리 용량만 늘려야 자원을 아낄 수 있습니다.

### 정적 그래프와 프레임워크 전용 서버의 한계

이런 파이프라인을 정의하는 방식은 크게 둘입니다. 많은 도구는 YAML 같은 설정 언어로 정적 그래프를 정의하게 하는데, 이 방식은 제약이 많고 다루기 어렵습니다. TFServing, TorchServe 같은 프레임워크 전용 서빙 솔루션은 커스텀 어댑터나 복잡한 설정을 요구합니다.

Ray Serve의 접근은 다릅니다. 모델 호출을 함수 호출처럼 작성하는 프로그래밍 가능한 API로 여러 모델을 엮고, 각 모델은 서로 다른 자원을 쓰며 클러스터의 다른 머신에서 실행되면서도 코드는 평범한 프로그램처럼 보입니다. Python 네이티브라서 모델 레지스트리와 연결할 때도 복잡한 설정이나 glue 코드가 필요 없고, YAML이나 커스텀 직렬화 형식과 씨름하지 않고 빠르게 반복 개발할 수 있습니다. LLM 애플리케이션이라면 앞서 든 네 단계를 하나의 Python 모듈로 작성하면서, 각 단계는 트래픽에 맞춰 독립적으로 오토스케일링(autoscaling)합니다.

### 프레임워크 비종속 서빙 계층

Ray Serve는 PyTorch, TensorFlow, Keras로 만든 딥러닝 모델부터 Scikit-Learn 모델, 임의의 Python 비즈니스 로직까지 한 도구로 서빙합니다. LLM 서빙에 필요한 응답 스트리밍, 동적 요청 배칭(request batching), 멀티 노드·멀티 GPU 서빙도 지원합니다. 효율 면에서는 replica 수를 조절하는 동적 스케일링, 벡터화 연산을 살리는 요청 배칭, 제한된 하드웨어에서 많은 모델을 서빙하게 하는 유연한 자원 할당 모델을 기본으로 갖췄습니다. Ray 위에 구축되어 여러 머신으로 확장되고, 분할 GPU(fractional GPU) 같은 스케줄링으로 자원을 나눠 씁니다. 임의의 Python 코드를 실행하므로 ONNX, MLflow, FastAPI, gRPC 같은 MLOps 도구와도 어렵지 않게 결합됩니다.

### 노트북에서 클러스터까지 같은 코드

Ray Serve를 쓰는 데 무거운 Kubernetes 개념이나 클라우드 설정 지식은 필요 없습니다. 데이터 과학자는 로컬 머신에서 전체 deployment 그래프를 테스트한 다음 프로덕션 클러스터로 배포합니다. 노트북에서 개발한 코드를 거의 고치지 않고 여러 머신이나 Kubernetes 클러스터로 확장할 수 있고, Kubernetes 배포에는 Kubernetes Operator인 KubeRay를 씁니다. Text ML 애플리케이션도 로컬 실행에서 출발해 config 파일 기반 배포를 거쳐 Kubernetes까지, 같은 Python 코드로 올라갑니다.

## Deployment·replica·application·ingress로 이루어진 Serve의 기본 단위

Serve 코드를 읽으려면 네 단위를 구분해야 합니다. 요청을 처리하는 코드 한 덩어리가 deployment이고, 이것이 Ray actor 프로세스인 replica들로 복제됩니다. `bind`로 묶인 deployment들은 하나의 애플리케이션(application)이 되며, 그 진입점이 ingress deployment입니다.

### 상태를 가진 프로세스, Ray actor

replica의 정체는 Ray의 액터(actor)입니다. Ray 원논문은 애플리케이션 계층을 프로그램을 실행하는 driver, 상태 없이 태스크를 처리하는 worker, 상태를 가진 actor로 나눕니다. 각 메서드는 이전 메서드가 남긴 상태에 의존합니다.

모델 서빙에는 이 성질이 바로 들어맞습니다. 생성자에서 한 번 메모리에 올린 모델 가중치가 프로세스가 살아 있는 동안 상태로 남고, 이후 요청은 그 상태를 재사용합니다.

### replica로 늘어나는 deployment

Deployment는 Ray Serve의 중심 개념입니다. 요청을 처리하는 비즈니스 로직이나 ML 모델을 담고, Ray 클러스터 전체로 확장됩니다. 작성자가 쓴 Python 클래스(단순한 경우에는 함수) 하나를 저장해 요청 처리에 쓰며, `@serve.deployment` 데코레이터가 이 클래스를 `Deployment` 객체로 바꿉니다. 데코레이터의 파라미터로 deployment마다 규모와 설정을 따로 정합니다.

런타임의 deployment는 여러 replica로 구성됩니다. replica는 그 클래스나 함수의 복사본이고, 각각 별도의 Ray Actor 프로세스에서 시작됩니다. replica 수는 요청 부하에 맞춰 늘리거나 줄일 수 있고, 오토스케일링에 맡길 수도 있습니다. 고정 개수는 `num_replicas`로 지정합니다. 요청은 replica 사이에 로드밸런싱되므로 deployment가 수평으로 확장됩니다. 이 파라미터를 코드와 config 중 어디서 정하는지, 오토스케일링 설정과 어떻게 함께 쓰는지는 deployment 파라미터를 다루는 부분에서 정리합니다.

Text ML 애플리케이션의 번역 단계를 deployment로 바꾸면 아래 코드가 됩니다. 일반 클래스와 비교하면 `@serve.deployment` 데코레이터와 `__call__` 메서드만 추가되었습니다.

```python
from starlette.requests import Request
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ray import serve

@serve.deployment(num_replicas=2)
class Translator:
    def __init__(self):
        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

    def translate(self, text: str) -> str:
        # Run inference
        input_ids = self.tokenizer(
            f"translate English to French: {text}", return_tensors="pt"
        ).input_ids
        output_ids = self.model.generate(
            input_ids, num_beams=4, early_stopping=True, max_length=300
        )
        # Post-process output to return only the translation text
        return self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    async def __call__(self, http_request: Request) -> str:
        english_text: str = await http_request.json()
        return self.translate(english_text)

translator_app = Translator.bind()
```

`num_replicas=2`이므로 Ray 위에 `Translator` 프로세스가 둘 뜨고, 프로세스마다 `__init__`이 실행되어 t5-small을 각자 메모리에 올립니다.

### bind가 정의하는 애플리케이션

마지막 줄의 `bind`는 deployment를 생성자에 넘길 인자와 묶습니다. 그 결과가 로컬에서 실행하거나 프로덕션에 배포할 수 있는 Serve 애플리케이션입니다. `Translator`의 생성자는 인자를 받지 않으므로 빈 `bind()`를 호출했습니다. 생성자가 인자를 받는 deployment라면 `bind`에 그 인자를 넘깁니다.

애플리케이션은 Ray Serve 클러스터에서 업그레이드하는 단위이며, deployment 하나 이상으로 구성됩니다. 호출 경로는 둘입니다. 지정된 `route_prefix`로 HTTP 요청을 보낼 수도 있고, Python에서 `DeploymentHandle`로 호출할 수도 있습니다. 다음 표는 네 단위의 관계를 정리한 것입니다.

| 단위 | 정체 | 만드는 방법 |
| --- | --- | --- |
| Deployment | 요청을 처리하는 클래스나 함수 하나 | `@serve.deployment` 데코레이터 |
| Replica | 별도 Ray Actor 프로세스에서 시작된 deployment의 복사본 | `num_replicas` 또는 오토스케일링 |
| Application | deployment 하나 이상의 묶음이자 업그레이드 단위 | `bind` |
| Ingress deployment | 애플리케이션의 모든 인바운드 트래픽을 받는 최상위 deployment | `serve.run`에 넘기는 deployment |

### 진입점이 되는 ingress deployment와 `__call__`

여러 deployment로 이루어진 애플리케이션에서도 `serve.run`에 넘기는 최상위 deployment는 언제나 하나이고, 이것을 ingress deployment라고 부릅니다. 애플리케이션으로 들어오는 모든 트래픽의 진입점이며, 보통 다른 deployment로 요청을 보내거나 `DeploymentHandle` API로 호출한 뒤 결과를 합쳐 사용자에게 돌려줍니다. 핸들을 주고받는 방법은 모델 컴포지션(model composition)을 다룰 때 설명합니다. 예를 들어 완성된 애플리케이션의 구조는 다음과 같은 모양이 됩니다.

```text
Application (업그레이드 단위)
 ├─ Summarizer  ← ingress deployment (HTTP 진입점)
 │    ├─ replica (actor 프로세스)
 │    └─ replica (actor 프로세스)
 └─ Translator  ← Summarizer가 DeploymentHandle로 호출
      ├─ replica (actor 프로세스)
      └─ replica (actor 프로세스)
```

HTTP 처리 로직도 ingress deployment가 정의합니다. 기본적으로 클래스의 `__call__`이 Starlette 요청 객체를 인자로 받아 호출되고, 반환값은 JSON으로 직렬화되어 HTTP 응답 본문이 됩니다. 다른 Starlette 응답 객체를 직접 반환해도 됩니다. 앞의 `Translator.__call__`은 이 규칙대로 요청의 JSON 본문을 읽어 `translate`에 넘기고, 번역문을 반환합니다. 원시 요청을 다루는 대신 FastAPI 통합을 쓰는 방법도 있습니다. 그 사용법은 최소 실행 예제와 함께 다룹니다.

## Controller·프록시·replica actor를 거치는 요청 경로와 컨트롤러 기반 오토스케일링

Serve 인스턴스는 actor 세 종류로 움직입니다. 전역 Controller가 상태를 관리하고, 프록시가 요청을 받아 큐에 넣으며, replica가 그 요청을 실행합니다. 오토스케일링 판단도 Controller 안에서 이루어집니다. Controller 상태는 GCS에 체크포인트되고, Controller에 장애가 나도 요청은 계속 보낼 수 있습니다.

### Serve 인스턴스를 이루는 actor들

Serve는 Ray actor 위에서 동작합니다. 구성 요소는 컨트롤 플레인을 맡는 Controller, HTTP 프록시, gRPC 프록시, 그리고 요청을 실제로 실행하는 replica actor입니다. 각각의 역할은 다음과 같습니다.

- **Controller**: Serve 인스턴스마다 하나만 있는 전역 actor입니다. 다른 actor를 생성하고 갱신하고 제거하며, deployment를 만들거나 조회하는 Serve API 호출은 모두 Controller에 대한 원격 호출로 처리됩니다. 이 actor의 옵션은 `serve.config.ControllerOptions`로 지정합니다.
- **HTTP 프록시**: 기본값에서는 헤드 노드(head node)에 하나 있습니다. Uvicorn HTTP 서버를 띄워 요청을 받고, replica로 전달한 뒤 처리가 끝나면 응답합니다.
- **gRPC 프록시**: 유효한 `port`와 `grpc_servicer_functions`를 주고 Serve를 시작하면 HTTP 프록시와 함께 뜹니다. grpcio 서버를 실행합니다.
- **Replica**: 요청을 받아 코드를 실행하는 actor입니다. 예를 들어 ML 모델 인스턴스를 담을 수 있고, 프록시가 보낸 요청을 하나씩 처리합니다.

Text ML 애플리케이션을 `serve run`으로 띄웠을 때 요청이 지나가는 경로를 그려 보면 다음과 같습니다.

```text
클라이언트 ──POST :8000──▶ [HTTP 프록시 actor] ──큐──▶ [Summarizer replica]
                                 ▲                          │ DeploymentHandle
                                 │ replica 목록               ▼ (핸들 큐)
                          [Controller actor] ◀── 메트릭 ── [Translator replica]
                                 │
                            GCS(헤드 노드)에 체크포인트
```

### 프록시 큐에서 replica 큐까지 이어지는 요청 수명 주기

HTTP나 gRPC 요청이 프록시에 도착하면 세 단계를 거칩니다.

1. 프록시가 요청을 받아 파싱합니다.
2. HTTP URL 경로나 애플리케이션 이름 메타데이터로 해당 deployment를 찾고, 요청을 큐에 넣습니다.
3. 큐의 요청마다 사용 가능한 replica를 찾아 보냅니다. 모든 replica에 `max_ongoing_requests`를 넘는 요청이 처리 중이면, 요청은 replica가 빌 때까지 큐에서 기다립니다.

큐는 replica 쪽에도 하나 더 있습니다. replica는 자기 요청 큐에서 요청을 하나씩 실행하는데, `asyncio`로 여러 요청을 동시에 처리할 수도 있습니다. 핸들러(deployment 함수나 클래스의 `__call__`)를 `async def`로 선언하면 replica는 핸들러가 끝날 때까지 기다리지 않고, 일반 함수로 선언하면 핸들러가 반환할 때까지 막혀 있습니다. 앞서 작성한 `Translator.__call__`은 `async def`이므로 앞의 경우에 해당합니다.

Summarizer가 Translator를 부르는 것처럼 DeploymentHandle로 보낸 요청은 프록시를 거치지 않습니다. 요청은 `DeploymentHandle` 안의 큐에 들어가고 곧바로 3단계로 넘어갑니다. DeploymentHandle은 같은 노드에 있는 라우터의 핸들을 감싼 객체입니다. 그래서 replica끼리 주고받는 요청도 HTTP/gRPC 요청과 같은 데이터 경로를 지나고, deployment 선택과 배칭 절차도 똑같이 적용됩니다. 요청이 실제로 어느 replica에서 실행됐는지 확인하려면 replica 안에서 `serve.get_replica_context`를 호출합니다. 이 함수는 deployment와 replica 태그를 돌려줍니다.

### power of two choices와 `max_ongoing_requests` 상한

3단계에서 replica를 고르는 규칙은 다음과 같습니다. DeploymentHandle은 Controller를 계속 폴링해 새 replica를 찾습니다. 새 replica를 발견하면 버퍼에 쌓인 쿼리와 새 쿼리를 그 replica의 `max_ongoing_requests`에 이를 때까지 보냅니다. 대상 replica는 power of two choices 스케줄링으로 정하며, 이때 어떤 replica도 동시에 `max_ongoing_requests`개를 넘는 요청을 처리하지 않아야 합니다.

power of two choices는 후보 둘을 무작위로 뽑아 덜 붐비는 쪽에 요청을 보내는 부하 분산 방식입니다. 예를 들어 Translator replica가 셋이고 처리 중인 요청이 각각 2, 5, 7개라고 가정합니다. 무작위로 뽑힌 후보가 5개짜리와 7개짜리라면 요청은 5개짜리로 갑니다. 모든 replica의 부하를 매번 조회하지 않고도 가장 붐비는 replica는 피하게 됩니다.

라우팅 방식은 바꿀 수 있습니다. `serve.request_router.RequestRouter`가 요청 라우터의 추상 인터페이스이고, 이를 구현할 때 FIFO 라우팅, 지역성(locality) 라우팅, 멀티플렉스 라우팅용 mixin을 섞어 쓸 수 있습니다. 라우터 설정은 `serve.config.RequestRouterConfig`로 합니다. `max_ongoing_requests`와 큐 길이 상한 값을 정하는 방법은 deployment 파라미터를 설명할 때 다룹니다.

### 크기에 따라 달라지는 요청 전달 방식

Serve는 Ray의 공유 메모리 객체 저장소(object store)와 프로세스 내 메모리 저장소를 함께 씁니다. 작은 요청 객체는 actor 사이 네트워크 호출로 직접 보내고, 100KiB 이상인 큰 객체는 객체 저장소에 쓴 뒤 replica가 zero-copy로 읽습니다. Text ML 애플리케이션에서 짧은 영어 문단은 앞의 경로로 가고, 긴 문서를 통째로 요약 요청하면 뒤의 경로로 넘어가는 식입니다.

### 노드마다 두는 프록시와 요청 배칭

ingress 쪽 확장은 프록시 배치로 해결합니다. `serve.start()`나 config 파일의 `proxy_location` 필드를 쓰면 노드마다 프록시 actor를 하나씩 띄울 수 있습니다. 프록시를 어디서 실행할지는 `serve.config.ProxyLocation`으로 지정합니다. 각 프록시는 같은 포트에 바인딩되므로 어느 노드로 요청을 보내도 모든 모델에 닿고, 그 앞에 별도 로드밸런서를 두어도 됩니다. 이 구조에서는 노드를 늘리면 HTTP/gRPC ingress가 확장되고, deployment의 `num_replicas`를 늘리면 추론 처리량이 확장됩니다.

replica 쪽에서는 `@serve.batch`로 요청을 묶어 처리할 수 있습니다. 이 데코레이터를 붙인 함수는 배치를 비동기로 처리합니다(Python API).

### Controller 안에서 도는 오토스케일러

오토스케일링은 deployment 부하에 따라 replica 수를 자동으로 늘리거나 줄이는 기능입니다. 아래 다이어그램에서는 메트릭이 흐르는 방향을 보면 됩니다. 핸들과 replica가 메트릭을 Controller 쪽 오토스케일러로 보내고, 오토스케일러는 그 결과로 replica 수를 조절합니다.

![핸들과 replica의 메트릭이 Controller의 오토스케일러로 전달되어 replica 수를 조절하는 오토스케일링 흐름도](https://raw.githubusercontent.com/ray-project/images/master/docs/serve/autoscaling.svg)

오토스케일러는 Serve Controller actor 안에서 실행됩니다. 각 `DeploymentHandle`과 replica가 주기적으로 메트릭을 오토스케일러에 푸시하고, 오토스케일러는 deployment마다 핸들 큐와 replica에서 처리 중인 쿼리를 주기적으로 확인해 replica 수를 조정할지 결정합니다. 기본 정책은 단순한 임계값으로 스케일링하는 `serve.autoscaling_policy.replica_queue_length_autoscaling_policy`입니다. 직접 정책을 작성하면 `serve.config.AutoscalingContext`가 판단에 쓸 정보를 넘겨줍니다. 오토스케일러 설정 객체는 `serve.config.AutoscalingConfig`입니다(Python API). 구체적인 설정 값은 deployment 파라미터를 다룰 때 설명합니다.

### 장애 종류마다 다른 복구 주체

애플리케이션 오류와 머신 장애는 다르게 처리됩니다. 모델 평가 코드에서 예외가 나면 Serve가 이를 잡아 traceback과 함께 500 상태 코드로 반환하고, replica는 계속 요청을 처리합니다. actor가 죽는 장애는 아래 표의 주체가 복구합니다.

| 장애 대상 | 복구 주체와 동작 |
| --- | --- |
| Replica actor | Controller가 새 replica로 교체합니다. |
| 프록시 actor | Controller가 다시 시작합니다. |
| Controller actor | Ray가 다시 시작합니다. |
| 노드 또는 클러스터 | KubeRay RayService를 쓰면 KubeRay가 복구하고, KubeRay가 없으면 Serve는 복구하지 못합니다. |

actor가 올라가 있던 머신이 죽으면 그 actor들은 다른 머신에서 자동으로 다시 시작됩니다. 라우팅 정책과 deployment 설정 같은 Controller 데이터는 헤드 노드의 GCS에 체크포인트되어 남습니다. 반면 라우터와 replica가 들고 있던 네트워크 연결이나 내부 요청 큐 같은 일시적 데이터는 사라집니다. 그래서 장애 시점에 큐에서 기다리던 요청은 잃을 수 있다고 보고 클라이언트 재시도를 설계해야 합니다.

> **핵심:** Controller는 요청 경로에 있지 않습니다. Controller가 죽어도 HTTP, gRPC, `DeploymentHandle`로 요청을 계속 보낼 수 있지만, 오토스케일링은 멈춥니다. 다만 Controller가 복구되어 오토스케일링이 재개되면 그전에 모은 메트릭은 모두 사라집니다.

클러스터 크래시는 GCS FT 기능을 써서 피할 수 있습니다. 끝으로 Ray 클러스터의 각 노드는 Serve REST API 서버를 제공하며, Serve를 관리하는 CLI도 따로 있습니다. 둘의 사용법은 Serve config 파일을 다루는 운영 부분에서 설명합니다.

## 설치부터 serve run과 HTTP 질의까지 가는 Text ML 애플리케이션의 최소 실행

모델 클래스에 데코레이터 하나와 `bind` 한 줄을 더하고 `serve run`으로 띄우면 로컬 HTTP 엔드포인트가 생깁니다. FastAPI 통합과 모델 레지스트리 로딩도 같은 틀 위에서 동작합니다.

### 설치와 Translator deployment

설치 명령은 하나입니다.

```bash
pip install "ray[serve]"
```

Text ML 애플리케이션은 영어 텍스트를 요약한 뒤 프랑스어(기본값), 독일어, 루마니아어 중 하나로 번역합니다. 먼저 번역 단계만 deployment로 만듭니다. 앞서 만든 Translator에 자원 예약만 더합니다. replica는 2개이고, 각 replica는 CPU 0.2개와 GPU 0개를 예약합니다. 이 값들을 어떻게 정하는지는 deployment 파라미터에서 따로 설명합니다.

```python
# File name: serve_quickstart.py
from starlette.requests import Request

import ray
from ray import serve

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0.2, "num_gpus": 0})
class Translator:
    def __init__(self):
        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

    def translate(self, text: str) -> str:
        # Run inference
        input_ids = self.tokenizer(
            f"translate English to French: {text}", return_tensors="pt"
        ).input_ids
        output_ids = self.model.generate(
            input_ids, num_beams=4, early_stopping=True, max_length=300
        )

        # Post-process output to return only the translation text
        translation = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return translation

    async def __call__(self, http_request: Request) -> str:
        english_text: str = await http_request.json()
        return self.translate(english_text)


translator_app = Translator.bind()
```

Hugging Face의 `pipeline("sentiment-analysis")`처럼 다른 모델을 쓸 때도 똑같이 `__init__`에서 로드하고 요청 처리 메서드에서 추론합니다.

### serve run으로 띄우고 requests로 질의하기

`serve run`은 `module:application` 형식의 import 경로를 받습니다. 스크립트가 있는 디렉터리에서 실행해야 import가 됩니다.

```bash
serve run serve_quickstart:translator_app
```

이 명령은 터미널을 점유한 채 로그를 콘솔에 계속 출력하고, Ctrl-C를 누르면 애플리케이션을 내립니다. 코드를 고쳤다면 Ctrl-C로 Serve와 모든 애플리케이션을 종료한 뒤 다시 실행합니다. 애플리케이션은 기본적으로 `http://127.0.0.1:8000/`에서 응답합니다. 다른 터미널에서 영어 텍스트를 JSON 본문에 담아 POST로 보내 봅니다.

```python
# File name: model_client.py
import requests

english_text = "Hello world!"

response = requests.post("http://127.0.0.1:8000/", json=english_text)
french_text = response.text

print(french_text)
```

이 클라이언트를 실행하면 `Bonjour monde!`가 출력됩니다.

CLI 대신 Python에서 `serve.run(translator_app)`을 호출해도 같은 애플리케이션이 뜹니다. 이때 `serve.run`은 ingress deployment의 핸들을 반환합니다. 테스트 스크립트에서는 이 핸들로 HTTP를 거치지 않고 바로 호출하면 됩니다. 핸들을 다루는 방법은 모델 컴포지션에서 설명합니다.

### FastAPI ingress로 바꾸기

raw Starlette 요청을 직접 파싱하는 대신 FastAPI에 맡기려면 세 군데를 바꿉니다. FastAPI를 import하고, `@serve.deployment` 아래에 `@serve.ingress(app)`를 붙이고, 이전처럼 `bind`합니다.

```python
import ray
from ray import serve
from fastapi import FastAPI

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = FastAPI()


@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0.2, "num_gpus": 0})
@serve.ingress(app)
class Translator:
    def __init__(self):
        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

    @app.post("/")
    def translate(self, text: str) -> str:
        # Run inference
        input_ids = self.tokenizer(
            f"translate English to French: {text}", return_tensors="pt"
        ).input_ids
        output_ids = self.model.generate(
            input_ids, num_beams=4, early_stopping=True, max_length=300
        )

        # Post-process output to return only the translation text
        translation = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return translation


translator_app = Translator.bind()
```

`__call__`이 없어지고 라우트 메서드가 직접 요청을 받습니다. `text: str` 인자는 FastAPI가 쿼리 파라미터로 해석하므로, 클라이언트도 `params`로 값을 보내고 응답은 `.json()`으로 읽어야 합니다.

```python
import requests

response = requests.post("http://127.0.0.1:8000/", params={"text": "Hello world!"})
french_text = response.json()

print(french_text)
```

`@fastapi_app.get("/{name}")` 같은 경로 파라미터나 FastAPI 응답 클래스도 같은 방식으로 쓸 수 있습니다.

### 모델 레지스트리에서 로드하고 워밍업하기

Serve는 임의의 Python 코드를 실행합니다. 그래서 레지스트리의 표준 Python 클라이언트로 모델을 직접 받아 올 수 있고, 로딩과 추론을 deployment 하나에 함께 둘 수 있습니다. 아래 deployment는 MLflow에 `sk-learn-random-forest-reg-model`로 등록된 모델 가운데 `production` 태그가 붙은 최신 모델을 찾습니다. 해당 모델이 없으면 `ValueError`를 발생시킵니다. 모델은 `__init__`에서 한 번만 로드하고, 더미 입력으로 예측을 한 번 돌려 둡니다. 이렇게 워밍업하면 첫 요청에서 지연이 튀지 않습니다.

```python
from ray import serve
import mlflow.pyfunc
import numpy as np


@serve.deployment
class MLflowModelDeployment:
    def __init__(self):
        # Search for models with production tag
        models = mlflow.search_logged_models(
            filter_string="tags.production='true' AND name='sk-learn-random-forest-reg-model'",
            order_by=[{"field_name": "creation_time", "ascending": False}],
        )
        if models.empty:
            raise ValueError("No model with production tag found")

        # Get the most recent production model
        model_row = models.iloc[0]
        artifact_location = model_row["artifact_location"]

        # Load model once during initialization (warm-start)
        self.model = mlflow.pyfunc.load_model(artifact_location)

        # Pre-warm the model with a dummy prediction
        dummy_input = np.zeros((1, 4))
        _ = self.model.predict(dummy_input)

    async def __call__(self, request):
        data = await request.json()
        features = np.array(data["features"])

        # MLflow validates input against the logged signature automatically
        prediction = self.model.predict(features)
        return {"prediction": prediction.tolist()}


app = MLflowModelDeployment.bind()
```

서빙 시점의 입력 검증은 학습 쪽 기록이 뒷받침해야 합니다. 모델을 기록할 때 `mlflow.models.infer_signature`로 시그니처를 남기면 MLflow가 서빙 중에 스키마를 자동으로 적용합니다. `pip_requirements`로 핵심 라이브러리 버전을 고정하면 학습, 평가, 서빙 환경에서 모델이 똑같이 동작합니다. scikit-learn 모델이라면 전처리를 포함한 `Pipeline` 전체를 기록해야 학습과 서빙의 변환이 어긋나지 않습니다.

## DeploymentHandle로 단계별 독립 스케일링을 얻는 모델 컴포지션

여러 deployment를 하나의 애플리케이션으로 엮을 때는 `bind`와 `DeploymentHandle`을 씁니다. bound deployment를 다른 deployment의 생성자 인자로 넘기면 런타임에 그 인자가 핸들로 바뀌고, 각 단계는 따로 스케일링됩니다.

### 생성자 인자에서 핸들로 바뀌는 Translator

Text ML 애플리케이션의 첫 단계는 요약입니다. 영어 텍스트가 먼저 `Summarizer`에 들어오므로 `__call__`은 `Summarizer`에 두고, `Translator`는 생성자 인자로 받아 요약문을 넘깁니다. `Translator`는 앞서 만든 클래스와 같은 `translate` 메서드를 갖지만, 이번에는 HTTP 요청을 직접 받지 않으므로 `__call__`이 빠집니다. 아래 `text_ml.py`에서 핵심은 생성자의 `translator: DeploymentHandle` 인자와 `__call__` 마지막 줄입니다.

```python
# File name: text_ml.py
from starlette.requests import Request

from ray import serve
from ray.serve.handle import DeploymentHandle

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


@serve.deployment
class Translator:
    def __init__(self):
        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

    def translate(self, text: str) -> str:
        # Run inference
        input_ids = self.tokenizer(
            f"translate English to French: {text}", return_tensors="pt"
        ).input_ids
        output_ids = self.model.generate(
            input_ids, num_beams=4, early_stopping=True, max_length=300
        )

        # Post-process output to return only the translation text
        translation = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return translation


@serve.deployment
class Summarizer:
    def __init__(self, translator: DeploymentHandle):
        self.translator = translator

        # Load model.
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

    def summarize(self, text: str) -> str:
        # Run inference
        input_ids = self.tokenizer(f"summarize: {text}", return_tensors="pt").input_ids
        output_ids = self.model.generate(
            input_ids, num_beams=4, early_stopping=True, max_length=15
        )

        # Post-process output to return only the summary text
        summary = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return summary

    async def __call__(self, http_request: Request) -> str:
        english_text: str = await http_request.json()
        summary = self.summarize(english_text)

        translation = await self.translator.translate.remote(summary)
        return translation


app = Summarizer.bind(Translator.bind())
```

마지막 줄은 `Translator`를 빈 생성자 인자에 묶은 뒤, 그 bound `Translator`를 `Summarizer`의 생성자 인자로 넘깁니다. 코드에서 `Summarizer`가 받는 값은 bound deployment이지만, replica 안에서 `__init__`이 실행될 때는 그 자리에 `DeploymentHandle`이 들어 있습니다. `DeploymentHandle`은 런타임에 deployment로 요청을 보내는 핸들입니다. Ray 2.10부터 예전 `RayServeHandle`과 `RayServeSyncHandle`은 완전히 제거되었으므로, 오래된 예제 코드는 이 API로 옮겨야 합니다.

실행 방법은 단일 deployment 때와 같습니다. `serve run text_ml:app`으로 띄우고, 앞서 만든 클라이언트에서 `english_text`만 『두 도시 이야기』의 첫 문장으로 바꿔 POST하면 `C'était le meilleur des temps, c'était le pire des temps,`가 돌아옵니다. HTTP 요청은 ingress인 `Summarizer`에 도착하고, 요약문은 핸들을 거쳐 `Translator` replica로 전달됩니다.

### 핸들 호출은 결과가 아니라 DeploymentResponse를 돌려줌

`self.translator.translate.remote(summary)`는 네 부분으로 읽습니다. 생성자로 받은 핸들, 호출할 메서드 이름, 다른 deployment로 보내는 호출임을 나타내는 `remote`, 그리고 인자입니다. 인자와 키워드 인자의 개수에는 제한이 없습니다. 이 호출은 결과 자체를 반환하지 않고 결과를 가리키는 `DeploymentResponse`를 반환합니다. 그래서 호출 자체는 비동기로 실행됩니다. 실제 값이 필요한 시점에 `await`하면 원격 호출이 끝나기를 기다렸다가 반환값을 받습니다.

결과를 꺼내는 방법은 `await`와 `.result()` 두 가지입니다. deployment 안에서는 `await`를 써야 합니다. `.result()`는 원격 호출이 끝날 때까지 그 deployment가 다른 코드를 전혀 실행하지 못하게 막습니다. 반면 `await`로 기다리는 동안에는 replica가 다른 요청을 계속 처리합니다. `.result()`가 어울리는 곳은 deployment 바깥입니다. 예를 들어 테스트 스크립트에서 `serve.run(app)`이 돌려준 ingress 핸들로 `handle.remote().result()`를 호출하는 경우입니다.

호출과 대기를 분리하면 하위 deployment 여러 개를 동시에 돌릴 수 있습니다. 두 하위 deployment를 쓰는 `Ingress` 예제가 이 패턴을 보여 줍니다. `self._hello_handle.remote()`와 `self._world_handle.remote()`를 먼저 모두 호출해 응답 객체 두 개를 받아 두고, 마지막에 `(await hello_response) + (await world_response)`로 합칩니다. 두 요청은 첫 `await`보다 먼저 이미 나가 있습니다.

### await 없이 다음 단계로 넘기는 응답 체이닝

`DeploymentResponse`는 기다리지 않은 채 다음 핸들 호출의 인자로 바로 넘길 수 있습니다. 받는 쪽 메서드 안에서는 그 인자가 응답 객체가 아니라 앞 단계의 출력값으로 풀려 있습니다.

```python
from ray import serve
from ray.serve.handle import DeploymentHandle, DeploymentResponse


@serve.deployment
class Adder:
    def __init__(self, increment: int):
        self._increment = increment

    def __call__(self, val: int) -> int:
        return val + self._increment


@serve.deployment
class Multiplier:
    def __init__(self, multiple: int):
        self._multiple = multiple

    def __call__(self, val: int) -> int:
        return val * self._multiple


@serve.deployment
class Ingress:
    def __init__(self, adder: DeploymentHandle, multiplier: DeploymentHandle):
        self._adder = adder
        self._multiplier = multiplier

    async def __call__(self, input: int) -> int:
        adder_response: DeploymentResponse = self._adder.remote(input)
        # Pass the adder response directly into the multiplier (no `await` needed).
        multiplier_response: DeploymentResponse = self._multiplier.remote(
            adder_response
        )
        # `await` the final chained response.
        return await multiplier_response


app = Ingress.bind(
    Adder.bind(increment=1),
    Multiplier.bind(multiple=2),
)

handle = serve.run(app)
response = handle.remote(5)
assert response.result() == 12, "(5 + 1) * 2 = 12"
```

응답 여러 개를 한 호출에 모아도 됩니다. `Adder` 두 개의 응답을 `Combiner.average.remote(...)`에 위치 인자로 함께 넘겨 평균을 내는 예제가 그런 경우입니다. 다만 응답을 넘길 수 있는 자리는 최상위 인자나 키워드 인자로 한정됩니다. 리스트나 딕셔너리 같은 중첩 객체에 담아 넘기는 방식은 완전히 폐기되어 더는 지원되지 않습니다.

Text ML 애플리케이션에 체이닝을 적용하려면 `Summarizer`의 요약 단계도 별도 deployment로 떼어 내야 합니다. 그러면 ingress는 요약 응답을 번역 핸들에 넘기고 최종 응답만 `await`하면 됩니다. 지금 구조에서는 요약이 ingress 안에서 동기적으로 실행되므로 넘길 응답 객체가 없습니다.

### stream=True 핸들로 받는 스트리밍 출력

출력을 여러 번 나눠 돌려주려면 두 조건을 갖춰야 합니다. 호출되는 메서드가 generator여야 하고, 핸들에 `handle.options(stream=True)`를 설정해야 합니다. 이때 핸들 호출은 `DeploymentResponse` 대신 `DeploymentResponseGenerator`를 반환합니다.

```python
from typing import AsyncGenerator, Generator

from ray import serve
from ray.serve.handle import DeploymentHandle, DeploymentResponseGenerator


@serve.deployment
class Streamer:
    def __call__(self, limit: int) -> Generator[int, None, None]:
        for i in range(limit):
            yield i


@serve.deployment
class Caller:
    def __init__(self, streamer: DeploymentHandle):
        self._streamer = streamer.options(
            # Must set `stream=True` on the handle, then the output will be a
            # response generator.
            stream=True,
        )

    async def __call__(self, limit: int) -> AsyncGenerator[int, None]:
        # Response generator can be used in an `async for` block.
        r: DeploymentResponseGenerator = self._streamer.remote(limit)
        async for i in r:
            yield i


app = Caller.bind(Streamer.bind())

handle = serve.run(app).options(
    stream=True,
)

# Response generator can also be used as a regular generator in a sync context.
r = handle.remote(10)
assert list(r) == list(range(10))
```

`DeploymentResponseGenerator`는 동기 generator로도, 비동기 generator로도 소비할 수 있습니다. 하지만 deployment 안에서 동기 generator로 쓰면 그 replica의 다른 요청이 동시에 실행되지 못합니다. `.result()`에서 생기는 문제와 같으므로 위 `Caller`처럼 `async for`를 써야 합니다. 또 `DeploymentResponse`와 달리 `DeploymentResponseGenerator`는 다른 핸들 호출의 인자로 넘길 수 없습니다.

### Ray task로 넘기는 ObjectRef 변환

내부적으로 `DeploymentResponse` 하나는 Ray `ObjectRef` 하나에 대응하고, 스트리밍 호출이라면 `ObjectRefGenerator`에 대응합니다. Serve 바깥의 Ray actor나 task와 결과를 주고받아야 할 때는 개발자 API인 `_to_object_ref`나 `_to_object_ref_sync`로 응답을 `ObjectRef`로 바꿉니다. 관련 예제에서는 deployment 안에서 `await response._to_object_ref()`로 참조를 얻고, 그 참조를 `@ray.remote` 함수에 넘긴 뒤, task가 반환한 `ObjectRef`를 다시 `await`합니다. 앞서 막힌 중첩 객체 전달도 이 방법으로 해결합니다. 중첩 객체에는 응답 대신 `_to_object_ref()`로 얻은 참조를 담으면 됩니다.

핸들 호출이 돌려주는 객체를 정리하면 다음과 같습니다.

| 호출 방식 | 반환 객체 | 다른 핸들 호출에 전달 | deployment 안에서 쓰는 방법 |
|---|---|---|---|
| `handle.method.remote(...)` | `DeploymentResponse` | 최상위 인자·키워드 인자로 가능 | `await` |
| `stream=True` 핸들의 generator 메서드 호출 | `DeploymentResponseGenerator` | 불가 | `async for` |
| `await response._to_object_ref()` | `ray.ObjectRef` | Ray task·actor에 전달 | `await` |

### 단계마다 따로 정하는 replica와 자원

컴포지션을 쓰는 이유는 파이프라인의 각 부분을 별도 deployment로 배포해 하나씩 설정하고 스케일링할 수 있어서입니다. 애플리케이션 안의 deployment마다 `autoscaling_config`, `num_replicas`, `num_cpus`, `num_gpus`를 독립적으로 지정합니다. 예를 들어 `SpanishResponder`는 replica 3개로, `FrenchResponder`는 1개로 두어 워크로드 수요에 맞출 수 있습니다. CPU와 GPU 예약을 포함해 deployment별로 정할 수 있는 다른 설정도 같은 방식으로 나눕니다. 이렇게 하면 단계마다 필요한 자원의 종류와 양이 달라도 한 단계가 애플리케이션 전체의 병목이 되지 않습니다.

Text ML 애플리케이션에 대입해 예를 들어 보겠습니다. 번역 단계의 처리 시간이 요약 단계의 두 배라면 `Translator`의 replica를 `Summarizer`의 두 배로 두는 식으로 단계별로 맞춥니다. 이 값을 데코레이터, `.options()`, Serve config 중 어디에 적고 어느 쪽이 우선하는지는 deployment 파라미터에서 다룹니다.

## 코드와 config 사이 우선순위로 정해지는 deployment 파라미터

replica 수, 자원, 큐 한도, 헬스 체크, 종료 동작 같은 deployment 파라미터는 코드와 Serve config 파일 두 곳에서 지정합니다. 같은 파라미터를 여러 곳에 적었다면 config 파일, 애플리케이션 코드, Serve 기본값 순으로 앞에 있는 값이 적용됩니다.

### 데코레이터와 `.options()`로 코드에 적는 값

코드에서 지정하는 방법은 두 가지입니다. deployment를 처음 정의할 때는 `@serve.deployment` 데코레이터에 인자로 넘기고, 이미 정의된 deployment의 값을 바꿀 때는 `.options()`를 씁니다. `.options()`를 쓰면 deployment 정의를 그대로 재사용하면서 런타임에 파라미터만 다르게 줄 수 있습니다. 아래 코드는 Text ML 애플리케이션의 `Translator`에 두 방법을 모두 적용한 것입니다. 번역 언어를 바꾸는 `reconfigure`도 함께 들어 있습니다.

```python
# excerpt: text_ml.py의 Translator 데코레이터, translate, reconfigure와 애플리케이션 정의 (translate는 self.prefix를 쓰도록 바꿈, __init__과 Summarizer는 앞서 본 정의와 같음)
from typing import Dict

from ray import serve


@serve.deployment(
    name="Translator",
    num_replicas=2,
    ray_actor_options={"num_cpus": 0.2, "num_gpus": 0},
    max_ongoing_requests=100,
    health_check_period_s=10,
    health_check_timeout_s=30,
    graceful_shutdown_timeout_s=20,
    graceful_shutdown_wait_loop_s=2,
)
class Translator:
    def translate(self, text: str) -> str:
        input_ids = self.tokenizer(self.prefix + text, return_tensors="pt").input_ids
        output_ids = self.model.generate(
            input_ids, num_beams=4, early_stopping=True, max_length=300
        )
        return self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def reconfigure(self, config: Dict):
        self.language = config.get("language", "french")

        if self.language.lower() == "french":
            self.prefix = "translate English to French: "
        elif self.language.lower() == "german":
            self.prefix = "translate English to German: "
        elif self.language.lower() == "romanian":
            self.prefix = "translate English to Romanian: "


app = Summarizer.bind(
    Translator.options(ray_actor_options={"num_cpus": 0.2, "num_gpus": 0.0}).bind()
)
```

### 코드보다 우선하는 Serve config 파일

프로덕션에서는 deployment 설정을 Serve config 파일에 두는 방식이 권장됩니다. 이렇게 하면 애플리케이션 코드를 고치지 않고도 값을 바꿀 수 있습니다. config 파일에서는 `applications` 아래 각 애플리케이션의 `deployments` 목록에 deployment 이름을 적고, 그 아래에 데코레이터 인자와 같은 키를 씁니다. 이 파일을 `serve build`로 생성하고 `serve deploy`로 적용하는 과정은 운영 흐름에 속하는 내용이라 여기서는 다루지 않습니다.

우선순위 규칙은 파라미터마다 따로 적용됩니다. 코드에 `@serve.deployment(num_replicas=2, graceful_shutdown_timeout_s=6)`를 적고 config 파일에는 `num_replicas: 5`만 적었다고 해 보겠습니다. 이때 최종 값은 다음 표처럼 정해집니다.

| 파라미터 | 코드 | config 파일 | 적용 값 |
|---|---|---|---|
| `num_replicas` | 2 | 5 | 5 (config) |
| `graceful_shutdown_timeout_s` | 6 | 없음 | 6 (코드) |
| `health_check_period_s` | 없음 | 없음 | 10 (기본값) |

단, 딕셔너리 형태의 파라미터는 키 단위로 합쳐지지 않습니다. `ray_actor_options`는 통째로 하나의 설정으로 취급됩니다. config 파일에 `ray_actor_options`가 있으면 코드의 딕셔너리 전체가 무시되고, 코드에서만 지정한 `num_gpus`나 `runtime_env` 같은 키는 처음부터 설정하지 않은 것으로 간주되어 기본값으로 돌아갑니다. `user_config`와 `autoscaling_config`도 같은 방식으로 덮어써집니다. 그래서 config 파일에 이 딕셔너리를 쓸 때는 필요한 키를 빠짐없이 모두 적어야 합니다.

### 파라미터별 기본값과 제약

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `num_replicas` | 1 | 고정 replica 수인 양의 정수, 또는 기본 설정으로 오토스케일링하는 `auto` |
| `autoscaling_config` | 없음 | 오토스케일링 동작 설정. 지정하면 `num_replicas`에 숫자를 쓸 수 없음 |
| `max_ongoing_requests` | 5 | 응답 없이 한 replica에 보내는 최대 요청 수. Ray 2.32.0에서 100에서 5로 변경 |
| `max_queued_requests` | -1 (무제한) | 호출자(프록시 또는 `DeploymentHandle`)별로 큐에 쌓을 수 있는 최대 요청 수 |
| `ray_actor_options` | 없음 | `accelerator_type`, `memory`, `num_cpus`, `num_gpus`, `object_store_memory`, `resources`, `runtime_env` |
| `health_check_period_s` | 10s | 헬스 체크 호출 간격 |
| `health_check_timeout_s` | 30s | 헬스 체크 응답을 기다렸다가 실패로 판정하기까지의 시간 |
| `graceful_shutdown_wait_loop_s` | 2s | 종료 전에 남은 작업이 끝나기를 기다리는 시간 |
| `graceful_shutdown_timeout_s` | 20s | 정상 종료를 기다렸다가 강제 종료하기까지의 시간 |
| `logging_config` | 없음 | 로그 레벨, 로그 디렉터리, JSON 로그 형식 등 deployment별 로깅 설정 |

`max_ongoing_requests`는 앞서 본 라우팅에서 replica별 동시 요청 상한으로 쓰이는 값입니다. `max_queued_requests` 한도에 도달하면 이후 요청은 핸들 호출의 경우 `BackPressureError`를 일으키고, HTTP 요청의 경우 기본적으로 503 상태 코드를 받습니다. HTTP 응답 코드는 `backpressure_config`로 바꿀 수 있습니다. 기본 헬스 체크는 replica에 아무 일도 하지 않는 actor 호출을 보내는 것입니다. deployment에 상태가 나쁠 때 예외를 던지는 `check_health` 메서드를 정의하면 직접 만든 검사로 대체됩니다.

### replica 재시작 없이 바뀌는 user_config

`user_config`의 값은 deployment의 `reconfigure` 메서드에 전달됩니다. 이 값은 replica를 재시작하지 않고 동적으로 바꿀 수 있으며, 전체가 JSON으로 직렬화 가능해야 합니다. 데코레이터나 config 파일로 deployment를 생성할 때 `user_config`를 지정하면, Serve는 `__init__` 직후에 `reconfigure`를 호출하면서 이 값을 인자로 넘깁니다. 나중에 config 파일의 `user_config`를 수정해 클러스터에 다시 적용해도 `reconfigure`가 호출됩니다. 새 값은 실행 중인 replica와 앞으로 생성될 replica 모두에 적용됩니다. 같은 방식으로 모델 가중치와 버전, 컴포지션 그래프의 트래픽 분할 비율, 피처 플래그와 하이퍼파라미터를 재시작 없이 조정할 수 있습니다.

앞서 정의한 `reconfigure`는 `language` 값에 따라 번역 프롬프트 접두사를 바꿉니다. 값이 없으면 `french`를 기본으로 씁니다. config 파일의 `Translator` 항목을 아래처럼 고쳐 다시 적용하면, replica가 떠 있는 그대로 독일어 번역으로 전환됩니다.

```yaml
# excerpt: serve_config.yaml의 deployments 항목
deployments:
- name: Translator
  num_replicas: 1
  user_config:
    language: german
- name: Summarizer
  num_replicas: 1
```

`Summarizer`도 같은 방식으로 `reconfigure`에서 `min_length`와 `max_length`를 받아 요약 길이를 바꿉니다.

## serve build에서 serve deploy까지 클러스터의 단일 진실 공급원이 되는 Serve config 파일

개발 중에 `serve run`으로 다듬은 애플리케이션은 프로덕션에 올릴 때 하나의 config YAML로 고정하고, 이후 배포와 갱신은 모두 이 파일을 기준으로 합니다. Serve config 파일은 클러스터의 단일 진실 공급원입니다. 시스템 수준 설정과 애플리케이션을 한곳에 지정하고 선언적으로 업데이트합니다. 프로덕션에서 애플리케이션을 배포하고 갱신할 때도 이 방식이 권장됩니다.

### serve build가 만들어 주는 초안과 직접 채워야 하는 runtime_env

`serve run`에 넘기던 import 경로(`text_ml:app`)를 `serve build`에 넘기면, 코드 안의 모든 deployment와 그 파라미터를 담은 config 파일이 `-o`로 지정한 파일에 저장됩니다. 애플리케이션 이름은 기본값으로 붙고, 필요하면 나중에 고칠 수 있습니다.

```bash
serve build text_ml:app -o serve_config.yaml
```

아래는 Text ML 애플리케이션으로 생성한 `serve_config.yaml`이며, 코드에 정의된 deployment와 그 파라미터가 담겨 있습니다. `serve build`가 넣는 HTTP, gRPC, 로깅 기본값도 함께 들어 있습니다.

```yaml
proxy_location: EveryNode

http_options:
  host: 0.0.0.0
  port: 8000

grpc_options:
  port: 9000
  grpc_servicer_functions: []

logging_config:
  encoding: TEXT
  log_level: INFO
  logs_dir: null
  enable_access_log: true

applications:
- name: default
  route_prefix: /
  import_path: text_ml:app
  runtime_env: {}
  deployments:
  - name: Translator
    num_replicas: 1
    user_config:
      language: french
  - name: Summarizer
    num_replicas: 1
```

`serve build`가 생성한 파일의 `runtime_env`는 항상 비어 있으므로 직접 채워야 합니다. Text ML 애플리케이션에는 `torch`와 `transformers`가 필요해서, 두 pip 패키지를 `runtime_env`에 추가합니다.

```yaml
# excerpt: serve_config.yaml의 applications 항목, runtime_env를 채운 뒤
applications:
- name: default
  route_prefix: /
  import_path: text_ml:app
  runtime_env:
    pip:
    - torch
    - transformers
```

`runtime_env`의 `working_dir`와 `py_modules`에는 원격 URI만 쓸 수 있습니다. 로컬 zip 파일이나 디렉터리는 지정할 수 없습니다. 완성한 파일은 클러스터에 올리기 전에 `serve run serve_config.yaml`처럼 `serve run`에 넘겨 로컬에서 먼저 확인할 수 있습니다.

### 다섯 개의 최상위 필드

Serve config의 최상위 필드는 `proxy_location`, `http_options`, `grpc_options`, `logging_config`, `applications` 다섯 개입니다. 각 필드의 역할과 기본값을 표로 정리했습니다.

| 필드 | 주요 키와 기본값 |
|---|---|
| `proxy_location` | `EveryNode`(기본값), `HeadOnly`, `Disabled` 중 하나 |
| `http_options` | `host` 기본 `0.0.0.0`, `port` 기본 `8000` |
| `grpc_options` | `port` 기본 `9000`, `grpc_servicer_functions` 기본 빈 리스트(gRPC 서버를 시작하지 않음) |
| `logging_config` | 컨트롤러, 프록시, replica 로그에 적용되는 전역 설정 |
| `applications` | `name`, `route_prefix`(기본 `/`), `import_path`, `runtime_env`, `deployments` |

HTTP와 gRPC 설정은 Ray 클러스터 전체에 적용되는 전역 설정이며, 런타임 중에는 바꿀 수 없습니다. Kubernetes에서 deployment를 클러스터 밖으로 노출하려면 `host`가 `0.0.0.0`이어야 합니다.

로드밸런서 뒤에 Serve를 둘 때는 `http_options.root_path`에 주의해야 합니다. Ray 2.60부터는 `root_path`가 ASGI 스펙을 따르기 때문에, `/serve/hello`처럼 접두사가 붙은 채 Serve에 도달한 요청은 404를 받습니다. 따라서 ingress나 로드밸런서에서 접두사를 떼어 내거나, `root_path`를 아예 설정하지 않아야 합니다. 값을 설정할 때는 끝에 슬래시를 붙이지 않습니다.

`logging_config`는 전역 기본값 역할만 합니다. 애플리케이션이나 deployment 수준에서 로깅을 따로 설정하면 그쪽이 우선합니다. 애플리케이션 항목은 `import_path` 하나만 있어도 최소 config로 동작합니다. 컨트롤러가 각 프록시의 상태를 확인하는 주기는 환경 변수 `RAY_SERVE_PROXY_HEALTH_CHECK_PERIOD_S`로 조정하며, 기본값은 `10.0`초입니다.

### serve deploy로 반영하는 선언적 배포와 갱신

VM에서는 `ray start --head`로 Ray 클러스터를 띄운 뒤 `serve deploy`로 config를 적용합니다.

```bash
# Start a local Ray cluster.
ray start --head

# Deploy the Text ML application to the local Ray cluster.
serve deploy serve_config.yaml
```

배포 요청이 성공하면 `Sent deploy request successfully!`가 출력됩니다. 이어서 `serve status`를 실행하면 프록시 상태, 애플리케이션 상태, deployment별 상태와 replica 수를 볼 수 있습니다.

```text
$ serve status
proxies:
  cef533a072b0f03bf92a6b98cb4eb9153b7b7c7b7f15954feb2f38ec: HEALTHY
applications:
  default:
    status: RUNNING
    message: ''
    last_deployed_time_s: 1694041157.2211847
    deployments:
      Translator:
        status: HEALTHY
        replica_states:
          RUNNING: 1
        message: ''
      Summarizer:
        status: HEALTHY
        replica_states:
          RUNNING: 1
        message: ''
```

배포된 애플리케이션에는 영어 문장을 JSON 본문으로 보내 질의합니다.

```python
import requests

english_text = (
    "It was the best of times, it was the worst of times, it was the age "
    "of wisdom, it was the age of foolishness, it was the epoch of belief"
)
response = requests.post("http://127.0.0.1:8000/", json=english_text)
french_text = response.text

print(french_text)
# 'C'était le meilleur des temps, c'était le pire des temps,'
```

애플리케이션을 갱신할 때도 같은 방법을 씁니다. config 파일을 고친 뒤 `serve deploy`를 다시 실행하면 됩니다. 앞서 본 user_config 갱신처럼 language만 german으로 고친 뒤 같은 명령을 다시 실행합니다.

```yaml
# excerpt: serve_config.yaml의 Translator 항목, language만 바꾼 뒤
  deployments:
  - name: Translator
    num_replicas: 1
    user_config:
      language: german
```

```bash
serve deploy serve_config.yaml
```

### 같은 config를 받는 REST API와 대시보드

CLI 대신 HTTP로도 config를 적용할 수 있습니다. Serve REST API는 Ray 대시보드와 같은 포트(기본 `8265`)로 노출되며, 이 포트는 `ray start`의 `--dashboard-port`로 바꿉니다. `PUT /api/serve/applications/`는 애플리케이션 목록을 선언적으로 배포합니다. Serve가 이미 실행 중이면 새 config에 없는 애플리케이션을 모두 제거하고, 실행 중이 아니면 Serve를 시작합니다. 따라서 요청에 기존 애플리케이션을 빠뜨리면 그 애플리케이션은 제거됩니다. 아래 요청 본문은 YAML config와 같은 구조를 JSON으로 옮긴 것이며, `working_dir`에 원격 URI를 사용합니다.

```http
PUT /api/serve/applications/ HTTP/1.1
Host: http://localhost:8265/
Accept: application/json
Content-Type: application/json

{
  "applications": [
    {
      "name": "text_app",
      "route_prefix": "/",
      "import_path": "text_ml:app",
      "runtime_env": {
        "working_dir": "https://github.com/ray-project/serve_config_examples/archive/HEAD.zip"
      },
      "deployments": [
        {"name": "Translator", "user_config": {"language": "french"}},
        {"name": "Summarizer"}
      ]
    }
  ]
}
```

이 요청에는 `HTTP/1.1 200 OK` 응답이 돌아옵니다. 배포 이후의 상태는 Ray 대시보드에서 클러스터와 Serve 애플리케이션 단위로 확인합니다. 대시보드는 로컬 테스트 환경과 원격 프로덕션 클러스터 양쪽에서 쓸 수 있습니다. Serve는 기본 메트릭과 로깅을 제공하고, 커스텀 메트릭과 로그를 추가하는 유틸리티도 있습니다. 프로덕션에서는 로그와 메트릭을 사내 관측 플랫폼으로 내보내는 구성이 권장됩니다.

## 독립 엔드포인트와 개별 업그레이드를 주는 다중 애플리케이션 배포

서로 독립적인 모델은 각각 별도 애플리케이션으로 나누고, 이 애플리케이션들을 하나의 다중 앱 config에 담아 `serve deploy`로 한 클러스터에 올립니다. 애플리케이션이 업그레이드 단위이기 때문에, 이렇게 나누면 모델마다 엔드포인트가 따로 생기고 클러스터에서 애플리케이션을 하나씩 추가·삭제·업그레이드할 수 있습니다. 여러 모델을 한 클러스터에 함께 올리므로 하드웨어 활용도도 올라갑니다.

### 컴포지션과 애플리케이션 분할을 가르는 기준

모델 컴포지션과 다중 애플리케이션은 둘 중 하나만 고르는 방식이 아니며, 함께 쓸 수 있습니다. 한 요청을 처리하는 데 함께 실행되고 같은 저장소에 있어서 한꺼번에 업그레이드하는 deployment들은 한 애플리케이션에 두는 것이 권장됩니다. Text ML 애플리케이션의 Summarizer와 Translator가 이 경우에 해당하므로 둘은 계속 한 애플리케이션으로 둡니다. 서로 다른 저장소에 있는 모델 묶음이나, 서로 통신하지 않지만 같은 하드웨어에 함께 올리고 싶은 모델은 애플리케이션을 나눕니다.

### 두 애플리케이션을 담은 config의 생성

예를 들어 이미지 분류기와 독일어 번역기를 독립된 애플리케이션 두 개로 만들어 봅니다. 여기서 쓰는 번역기는 앞서 만든 Text ML 애플리케이션의 Translator와 이름만 같은 별도 예제 클래스로, 독일어 번역만 하고 `text` 필드로 입력을 받으며 `user_config`를 쓰지 않습니다. 첫 번째 파일 `image_classifier.py`는 이미지를 내려받는 `downloader`를 `ImageClassifier`에 주입합니다.

```python
import requests
import starlette

from transformers import pipeline
from io import BytesIO
from PIL import Image

from ray import serve
from ray.serve.handle import DeploymentHandle


@serve.deployment
def downloader(image_url: str):
    image_bytes = requests.get(image_url).content
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return image


@serve.deployment
class ImageClassifier:
    def __init__(self, downloader: DeploymentHandle):
        self.downloader = downloader
        self.model = pipeline(
            "image-classification", model="google/vit-base-patch16-224"
        )

    async def classify(self, image_url: str) -> str:
        image = await self.downloader.remote(image_url)
        results = self.model(image)
        return results[0]["label"]

    async def __call__(self, req: starlette.requests.Request):
        req = await req.json()
        return await self.classify(req["image_url"])


app = ImageClassifier.bind(downloader.bind())
```

두 번째 파일 `text_translator.py`에는 t5-small로 영어를 독일어로 옮기는 `Translator` 하나만 들어 있습니다.

```python
import starlette

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ray import serve


@serve.deployment
class Translator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

    def translate(self, text: str) -> str:
        input_ids = self.tokenizer(
            f"translate English to German: {text}", return_tensors="pt"
        ).input_ids
        output_ids = self.model.generate(
            input_ids, num_beams=4, early_stopping=True, max_length=300
        )
        return self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    async def __call__(self, req: starlette.requests.Request):
        req = await req.json()
        return self.translate(req["text"])


app = Translator.bind()
```

앞서 단일 애플리케이션에 썼던 `serve build`에 import 경로를 여러 개 넘기면, 두 애플리케이션을 모두 담은 config가 하나 생성됩니다.

```bash
serve build image_classifier:app text_translator:app -o config.yaml
```

생성된 파일을 보면 시스템 수준 옵션은 파일 맨 위에 한 번만 있고, `applications` 목록에 애플리케이션마다 `name`, `route_prefix`, `import_path`, `deployments`가 따로 들어갑니다.

```yaml
proxy_location: EveryNode

http_options:
  host: 0.0.0.0
  port: 8000

grpc_options:
  port: 9000
  grpc_servicer_functions: []

logging_config:
  encoding: JSON
  log_level: INFO
  logs_dir: null
  enable_access_log: true

applications:
- name: app1
  route_prefix: /classify
  import_path: image_classifier:app
  runtime_env: {}
  deployments:
  - name: downloader
  - name: ImageClassifier

- name: app2
  route_prefix: /translate
  import_path: text_translator:app
  runtime_env: {}
  deployments:
  - name: Translator
```

애플리케이션 이름은 `app1`, `app2`처럼 자동으로 붙습니다. 다른 이름을 쓰려면 배포 전에 config 파일에서 직접 고칩니다. `route_prefix`는 애플리케이션마다 달라야 하며, 클라이언트는 이 접두사로 엔드포인트를 구분합니다.

이전 형식의 단일 애플리케이션 config를 옮길 때는 기존 config 전체를 `applications` 아래 항목 하나로 복사하고, 애플리케이션에 이름을 붙입니다. 이어서 `route_prefix`를 ingress deployment가 아니라 애플리케이션 수준에 지정하고, `host`와 `port`는 그 항목에서 빼서 `http_options`로 옮깁니다. 개별 애플리케이션 항목에는 `host`, `port` 같은 클러스터 수준 옵션을 둘 수 없습니다.

### 엔드포인트별 배포와 질의

Ray 클러스터를 먼저 띄운 뒤 config를 배포합니다.

```bash
$ ray start --head

$ serve deploy config.yaml
> Sent deploy request successfully!
```

배포가 끝나면 각 애플리케이션을 자기 접두사로 호출합니다.

```python
import requests

bear_url = "https://cdn.britannica.com/41/156441-050-A4424AEC/Grizzly-bear-Jasper-National-Park-Canada-Alberta.jpg"  # noqa
resp = requests.post("http://localhost:8000/classify", json={"image_url": bear_url})
print(resp.text)
# 'brown bear, bruin, Ursus arctos'

text = "Hello, the weather is quite fine today!"
resp = requests.post("http://localhost:8000/translate", json={"text": text})
print(resp.text)
# 'Hallo, das Wetter ist heute ziemlich gut!'
```

로컬에서 반복 개발할 때는 `serve run config.yaml`을 씁니다. 다만 `serve run`이 실행하는 config 파일은 다중 애플리케이션 형식이어야 합니다. import 경로를 직접 넘기면 한 번에 애플리케이션 하나만 실행됩니다.

이후 운영 중에는 `applications` 아래 항목을 추가하거나, 지우거나, 수정한 뒤 다시 배포합니다. 이렇게 해도 클러스터의 다른 애플리케이션에는 영향이 없습니다. 수정된 애플리케이션이 config 변경에 반응하는 방식, 즉 제자리 업데이트 동작은 단일 애플리케이션과 같습니다.

### HTTP를 거치지 않는 앱 간 호출

`serve.get_app_handle`에 앱 이름을 넘기면 그 애플리케이션의 ingress deployment 핸들을 얻습니다. 이 핸들로 클러스터에서 실행 중인 다른 애플리케이션에 HTTP를 거치지 않고 직접 요청을 보낼 수 있습니다. 아래는 `ImageClassifier.__call__`을 고친 예로, 요청에 `should_translate`가 있으면 분류 결과를 `app2`로 넘겨 번역합니다.

```python
# image_classifier.py: downloader 정의 아래의 ImageClassifier와 app을 이 코드로 바꿉니다
import starlette

from transformers import pipeline

from ray import serve
from ray.serve.handle import DeploymentHandle


@serve.deployment
class ImageClassifier:
    def __init__(self, downloader: DeploymentHandle):
        self.downloader = downloader
        self.model = pipeline(
            "image-classification", model="google/vit-base-patch16-224"
        )

    async def classify(self, image_url: str) -> str:
        image = await self.downloader.remote(image_url)
        results = self.model(image)
        return results[0]["label"]

    async def __call__(self, req: starlette.requests.Request):
        req = await req.json()
        result = await self.classify(req["image_url"])

        if req.get("should_translate") is True:
            handle: DeploymentHandle = serve.get_app_handle("app2")
            return await handle.translate.remote(result)

        return result


app = ImageClassifier.bind(downloader.bind())
```

같은 `/classify` 엔드포인트에 `{"image_url": bear_url, "should_translate": True}`를 보내면 `'Braunbär, Bruin, Ursus arctos'`가 돌아옵니다. 애플리케이션 이름이 호출 코드에 문자열로 들어가므로, config에서 이름을 바꾸면 이 호출 코드도 함께 고쳐야 합니다.

### 애플리케이션 단위로 나뉘는 상태 확인

`serve status`는 프록시 상태 아래에 애플리케이션별로 status와 deployment별 `replica_states`를 나열합니다.

```text
$ serve status
proxies:
  2e02a03ad64b3f3810b0dd6c3265c8a00ac36c13b2b0937cbf1ef153: HEALTHY
applications:
  app1:
    status: RUNNING
    message: ''
    last_deployed_time_s: 1693267064.0735464
    deployments:
      downloader:
        status: HEALTHY
        replica_states:
          RUNNING: 1
        message: ''
      ImageClassifier:
        status: HEALTHY
        replica_states:
          RUNNING: 1
        message: ''
  app2:
    status: RUNNING
    message: ''
    last_deployed_time_s: 1693267064.0735464
    deployments:
      Translator:
        status: HEALTHY
        replica_states:
          RUNNING: 1
        message: ''
```

더 자세히 보려면 앞서 소개한 Ray 대시보드에서 `/#/serve` 경로의 Ray Serve 화면을 엽니다. 첫 화면에는 클러스터에 배포된 애플리케이션이 모두 나열됩니다.

![Ray Serve 대시보드의 애플리케이션 목록 화면](https://raw.githubusercontent.com/ray-project/images/master/docs/serve/multi-app/applications-dashboard.png)

대시보드는 애플리케이션마다 그 아래에 속한 deployment 목록도 보여 줍니다. `app1`이라면 `downloader`와 `ImageClassifier`가 나옵니다.

![애플리케이션별 deployment 목록을 보여 주는 대시보드 화면](https://raw.githubusercontent.com/ray-project/images/master/docs/serve/multi-app/deployments-dashboard.png)

deployment마다 replica 목록도 볼 수 있습니다.

![deployment별 replica 목록을 보여 주는 대시보드 화면](https://raw.githubusercontent.com/ray-project/images/master/docs/serve/multi-app/replica-dashboard.png)

## 헬스 체크·복구·무중단 업그레이드를 자동화하는 KubeRay RayService 배포

프로덕션에서 Ray Serve를 실행할 때 권장하는 방법은 Kubernetes에서 KubeRay의 RayService 커스텀 리소스를 쓰는 것입니다. 이 리소스가 헬스 체크, 상태 보고, 장애 복구, 업그레이드를 자동으로 처리합니다. Kubernetes가 아닌 환경에서는 지금까지 써 온 Serve CLI로 Ray 클러스터에 직접 배포해도 됩니다. 앞서 KubeRay 없이는 Ray 클러스터 장애를 복구할 수 없다고 설명했습니다. RayService는 크래시한 노드와 클러스터를 복구합니다.

### Ray 클러스터와 serveConfigV2를 함께 담는 단일 매니페스트

RayService CR 하나에 멀티 노드 Ray 클러스터와 그 위에서 실행되는 Serve 애플리케이션이 함께 들어갑니다. 배포, 업그레이드, 상태 조회는 모두 표준 `kubectl` 명령으로 합니다. 이렇게 하면 Ray Serve의 확장 가능한 컴퓨팅과 Kubernetes의 운영 기능을 함께 쓸 수 있고, 이미 Kubernetes에서 돌고 있는 다른 애플리케이션과도 연동됩니다.

매니페스트에서 `spec.rayClusterConfig`는 Ray 클러스터를, `serveConfigV2`는 그 위에 올릴 Serve 애플리케이션을 정의합니다. 앞 단계에서 생성한 YAML은 Kubernetes 설정에 그대로 복사해 넣습니다. KubeRay 샘플의 `serveConfigV2`는 `applications` 목록 아래에 `fruit_app`과 `math_app` 두 애플리케이션을 두고, 각각 `import_path`, `route_prefix`, `runtime_env`를 지정합니다. 이 샘플에서 `working_dir`는 원격 저장소의 zip URI를 가리킵니다. 배포할 때 설치해야 하는 의존성은 deployment 코드의 `runtime_env`에 넣을 수도 있습니다. 프로덕션이라면 헤드 노드 장애 허용도 설정하는 편이 좋습니다.

### 클러스터 생성, REST 배포, Service 생성 순서로 진행되는 배포

RayService가 만들어지면 KubeRay 컨트롤러는 먼저 주어진 설정으로 Ray 클러스터를 띄웁니다. 클러스터가 실행되면 REST API로 Serve 애플리케이션을 배포하고, 트래픽을 라우팅할 Kubernetes Service도 만듭니다. 

Text ML 예제의 Serve config는 KubeRay 저장소의 샘플 CR에 이미 들어 있습니다. 이 샘플 CR에서 애플리케이션 이름은 `default`가 아니라 `text_ml_app`이며, 뒤에 나오는 `kubectl describe` 출력에도 이 이름으로 표시됩니다. 파일을 내려받아 적용하면 헤드 pod와 워커 pod로 이루어진 Ray 클러스터, 그리고 질의용 서비스가 생깁니다.

```bash
$ curl -o ray-service.text-ml.yaml https://raw.githubusercontent.com/ray-project/kuberay/2ba0dd7bea387ac9df3681666bab3d622e89846c/ray-operator/config/samples/ray-service.text-ml.yaml

$ kubectl apply -f ray-service.text-ml.yaml

$ kubectl get rayservices
NAME SERVICE STATUS NUM SERVE ENDPOINTS
rayservice-sample Running 1

$ kubectl get pods
NAME READY STATUS RESTARTS AGE
rayservice-sample-raycluster-7wlx2-head-hr8mg 1/1 Running 0 XXs
rayservice-sample-raycluster-7wlx2-small-group-worker-tb8nn 1/1 Running 0 XXs

$ kubectl get services
NAME TYPE CLUSTER-IP EXTERNAL-IP PORT(S) AGE
rayservice-sample-head-svc ClusterIP None <none> 10001/TCP,8265/TCP,6379/TCP,8080/TCP,8000/TCP XXs
rayservice-sample-raycluster-7wlx2-head-svc ClusterIP None <none> 10001/TCP,8265/TCP,6379/TCP,8080/TCP,8000/TCP XXs
rayservice-sample-serve-svc ClusterIP 192.168.145.219 <none> 8000/TCP XXs
```

이 샘플은 시연용이라 `numCpus`를 아주 낮게 잡아 두었습니다. 프로덕션에서는 Serve 애플리케이션에 자원을 더 줘야 합니다.

### serve 서비스와 ingress를 거치는 트래픽

질의는 위 출력의 `rayservice-sample-serve-svc`로 보냅니다. 이 서비스는 Serve replica가 떠 있는 모든 워커에 트래픽을 나눠 보냅니다. 클러스터 안에서 시험할 때는 curl pod를 하나 띄우고, 앞서 질의에 쓴 Serve HTTP 포트에 serve 서비스 이름과 애플리케이션의 `route_prefix`를 붙여 POST를 보냅니다. 아래 명령은 KubeRay의 과일 가게·계산기 샘플 `ray-service.sample.yaml`을 쓰는 예입니다. 이 샘플의 RayService 이름도 Text ML과 같은 `rayservice-sample`이므로, 두 리소스를 구분하려고 별도 네임스페이스 `demo`에서 실행합니다.

```bash
curl -O https://raw.githubusercontent.com/ray-project/kuberay/v1.7.0/ray-operator/config/samples/ray-service.sample.yaml
kubectl create namespace demo
kubectl apply -n demo -f ray-service.sample.yaml

kubectl run curl -n demo --image=curlimages/curl:latest -i --tty -- sh
curl -X POST -H 'Content-Type: application/json' rayservice-sample-serve-svc:8000/fruit/ -d '["MANGO", 2]'
# [Expected output]: 6
```

Text ML 애플리케이션이라면 앞서 만든 `requests.post` 클라이언트에서 호스트만 `rayservice-sample-serve-svc:8000`으로 바꿔 클러스터 안의 pod에서 실행하면 됩니다. 클러스터 바깥에 노출하려면 로드 밸런서를 앞에 둔 ingress를 설정합니다. KubeRay에는 ALB ingress 샘플이 있습니다. Serve는 모든 노드에서 HTTP 프록시를 실행하므로 로드 밸런서의 노드 헬스 체크 경로로 `/-/routes`를 쓸 수 있습니다.

### 바뀐 필드에 따라 갈리는 제자리 업데이트와 무중단 업그레이드

업데이트할 때는 매니페스트를 고치고 `kubectl apply`를 다시 실행합니다. 어떤 방식으로 반영될지는 고친 필드가 정합니다.

| 변경한 필드 | KubeRay의 처리 | 비용 |
|---|---|---|
| `serveConfigV2` | 기존 RayCluster에 새 설정을 다시 적용하는 제자리 업데이트입니다 | 새 클러스터를 만들지 않습니다 |
| `spec.rayClusterConfig` (컨테이너 이미지, `rayVersion` 등) | 새 RayCluster를 만들고, 준비되면 트래픽을 옮긴 뒤 이전 클러스터를 지우는 무중단 업그레이드입니다 | 두 클러스터를 동시에 스케줄할 자원이 필요합니다 |
| `replicas`, `minReplicas`, `maxReplicas`, `scaleStrategy.workersToDelete` | RayCluster로 전파하지 않으므로 아무 일도 일어나지 않습니다 | 없습니다 |

제자리 업데이트로는 deployment 규모 조정이나 오토스케일링 파라미터 변경 같은 가벼운 수정을 합니다. 샘플에서 `MangoStand`의 `user_config`를 `price: 3`에서 `price: 4`로 바꿔 다시 적용하면 기존 deployment가 재구성됩니다. `kubectl describe`에서 deployment 상태가 `UPDATING`으로 보이다가 `HEALTHY`로 돌아오고, 그 뒤로 같은 요청 `["MANGO", 2]`의 응답이 `6`에서 `8`로 바뀝니다. Text ML에서 Translator의 `user_config` language를 독일어로 바꾸는 경우도 같은 경로를 탑니다. 

```bash
kubectl apply -n demo -f ray-service.sample.yaml
kubectl describe -n demo rayservices rayservice-sample
```

클러스터 수준 변경은 기본값인 `NewCluster` 전략으로 처리됩니다. RayService가 새 RayCluster를 임시로 만들어 준비될 때까지 기다리고, 자신이 관리하는 head 서비스 `rayservice-sample-head-svc`의 selector를 새 클러스터로 바꿔 트래픽을 넘긴 다음 이전 클러스터를 삭제합니다. 샘플에서 `rayVersion`을 바꾸면 전환 중에 `kubectl get raycluster` 목록에 기존 클러스터와 새 클러스터가 함께 나타납니다. 이전 클러스터가 종료된 뒤에도 같은 serve 서비스로 보낸 요청은 계속 응답합니다. Ray 개발 버전 기준으로 새 클러스터는 먼저 Pending Service Status에 나타나고, 정상이 되면 Active Service Status로 올라갑니다.

워커 그룹의 `replicas`를 1에서 2로 바꾸면 RayService CR에는 2가 기록되지만 RayCluster CR은 1로 남습니다. 따라서 이 값은 실제 워커 수를 바꾸는 수단이 아닙니다. 자원을 두 배로 잡기 어려운 대규모 배포에는 업그레이드 중 자원을 덜 쓰는 `NewClusterWithIncrementalUpgrade` 전략이 따로 있습니다. 이 글에서는 이 전략을 이름으로만 소개하고 내부 동작은 다루지 않습니다.

### Serve와 Kubernetes가 나눠 맡는 오토스케일링

replica 수는 Serve config의 autoscaling 필드로 조절합니다. 앞서 본 `autoscaling_config`가 `serveConfigV2` 안에서 같은 역할을 합니다. replica가 늘어 Ray 노드가 더 필요해지는 상황을 감당하려면 RayCluster 설정에서 `enableInTreeAutoscaling`을 True로 켜야 합니다. 대부분의 경우에는 Kubernetes 클러스터 오토스케일링까지 함께 켜서 클러스터 자원을 충분히 쓰도록 권장합니다. GKE에서는 Autopilot 클러스터를, EKS에서는 Cluster Autoscaler를 씁니다.

### CR 상태에서 오퍼레이터 로그, pod 로그로 좁혀 가는 디버깅

KubeRay 컨트롤러는 실행 중인 RayService를 계속 감시하면서 관련 상태를 CR에 기록합니다. `kubectl describe`로 이 기록을 볼 수 있습니다. 출력에는 클러스터 상태, 헬스 체크 실패나 재시작 같은 이벤트, 그리고 `serve status`가 보고하는 애플리케이션 수준 상태가 들어 있습니다. Text ML 예제의 출력에서 활성 서비스 부분만 추리면 아래와 같습니다.

```text
$ kubectl describe rayservice rayservice-sample
Status:
  Active Service Status:
    Application Statuses:
      text_ml_app:
        Serve Deployment Statuses:
          Summarizer:
            Status: HEALTHY
          Translator:
            Status: HEALTHY
        Status: RUNNING
    Ray Cluster Name: rayservice-sample-raycluster-7wlx2
  Pending Service Status:
Events:
  Normal Running 2m15s (x29791 over 16h) rayservice-controller The Serve applicaton is now running and healthy.
```

과일 가게 샘플의 `kubectl describe` 출력에는 Conditions 항목도 있어 `Ready`와 `UpgradeInProgress`를 볼 수 있습니다. 대기 중인 클러스터가 없을 때 `UpgradeInProgress`의 Reason은 `NoPendingCluster`입니다.

배포 자체가 실패했다면 KubeRay 오퍼레이터 로그를 `kubectl logs -f <kuberay-operator-pod-name>`로 확인합니다. 이 로그에는 Serve 배포 이벤트와 헬스 체크 정보가 담깁니다. 특정 deployment가 왜 실패했는지, 오토스케일링 이벤트가 어떻게 일어났는지는 헤드 pod와 워커 pod 모두의 `/tmp/ray/session_latest/logs/serve/` 아래에 있는 컨트롤러 로그와 deployment 로그에서 찾습니다.

대시보드를 보려면 head 서비스의 대시보드 포트(앞서 본 REST API와 같은 포트)를 로컬로 포워딩한 뒤 Serve 페이지(`/#/serve`)를 엽니다.

```bash
kubectl port-forward svc/rayservice-sample-head-svc 8265:8265
```

이렇게 열면 로컬에서 보던 Serve 페이지가 그대로 나오고, 애플리케이션 상태를 클러스터 바깥에서 확인할 수 있습니다.

![Ray 대시보드의 Serve 페이지 화면](https://docs.ray.io/en/latest/_images/dashboard_serve.png)

장기 모니터링에는 대시보드와 함께 Prometheus·Grafana와 Serve 로그를 씁니다. 처음 RayService를 다뤄 본다면 GPU 없이 로컬 머신에서 돌릴 수 있는 MobileNet 예제로 시작하는 것이 좋습니다. 문제가 생기면 RayService 트러블슈팅 가이드를 참고합니다.

## Python 컴포지션에는 맞고, 모델 최적화와 플랫폼 기능은 바깥에서 채워야 하는 Ray Serve

Ray Serve를 고를 기준은 하나입니다. 여러 모델과 비즈니스 로직을 Python 코드로 엮고, 각 단계를 따로 스케일링해야 하는가입니다. Text ML 애플리케이션처럼 Summarizer와 Translator가 서로 다른 자원과 replica 수로 돌아가야 하는 파이프라인이 전형적인 경우이고, 이런 컴포지션이 Serve가 가장 잘하는 일입니다. 반대로 모델 하나를 최대한 빠르게 돌리는 일, 모델의 수명주기를 관리하는 일은 Serve의 범위 밖입니다.

### 다른 도구와 나누는 역할

Ray Serve와 자주 비교되는 도구군과의 관계를 정리하면 다음과 같습니다.

| 비교 대상 | 그 도구가 주는 것 | Ray Serve의 위치 |
|---|---|---|
| TFServing, TorchServe, ONNXRuntime | 프레임워크 전용 모델 최적화 | 모델별 최적화를 하지 않습니다. 직접 최적화한 모델(PyTorch JIT, ONNXRuntime 컴파일 모델)을 올려 실행합니다 |
| AWS SageMaker, Azure ML, Google Vertex AI | 호스팅형 ML 플랫폼 기능 | 자체 인프라에서 확장성과 안정성을 제공합니다. 모델 수명주기 관리와 성능 시각화 기능은 없습니다 |
| Seldon, KServe, Cortex | Kubernetes 기반 서빙 | K8s 클러스터 없이 노트북에서 시작해 나중에 KubeRay로 옮깁니다 |
| BentoML, Comet.ml, MLflow | 모델 단위 서빙과 확장 | 모델 컴포지션에 집중하며, 모델 패키징·레지스트리 형식에 구애받지 않습니다 |

Translator의 추론 속도가 병목이라면 TorchServe로 갈아탈 필요는 없습니다. 모델을 직접 컴파일한 뒤 같은 deployment의 `__init__`에서 불러오면 됩니다. 호스팅 플랫폼과 견주면 Serve는 완성된 플랫폼이라기보다 그 위에 자체 ML 플랫폼을 쌓기 위한 기본 요소에 가깝습니다. Ray 클러스터 런처로 주요 퍼블릭 클라우드, K8s, 온프레미스 베어메탈에 배포할 수 있습니다. Ray 프로젝트는 다른 도구로 같은 기능을 갖추려면 TensorFlow Serving과 SageMaker 같은 프레임워크 여러 개를 엮거나 마이크로 배칭 컴포넌트를 직접 만들어야 한다고 봅니다.

MLflow는 특히 구분해서 볼 필요가 있습니다. 앞서 MLflow 레지스트리에서 모델을 불러오는 패턴은 그대로 유효합니다. 다만 MLflow 내장 REST 서버는 동시성이 높은 GPU 워크로드에 맞춰 최적화되어 있지 않습니다. 그래서 LLM이나 diffusion 모델처럼 무거운 transformer 계열에는 Hugging Face Hub나 Weights & Biases를 레지스트리로 쓰고, GPU 배칭·오토스케일링·스케줄링은 Ray Serve에 맡기도록 권장합니다. 레지스트리는 바꿔 끼우고 서빙 계층은 Serve로 유지하는 구성입니다.

### 감수해야 할 장애 모드

운영에서 부딪히는 한계는 대부분 앞서 다룬 구조에서 나옵니다.

- **클러스터 장애:** 앞서 본 대로 KubeRay 없이는 복구되지 않으므로 VM 배포라면 클러스터를 다시 띄우는 절차를 따로 마련해야 합니다.
- **컨트롤러 장애:** 앞서 본 오토스케일링 중단과 메트릭 유실을 감수해야 합니다.
- **무제한 큐:** `max_queued_requests`의 기본값은 무제한입니다. 과부하가 걸리면 백프레셔 없이 요청이 계속 쌓입니다.
- **업그레이드 자원:** 앞서 본 클러스터 수준 업그레이드의 이중 자원 요구를 감안해야 합니다.

### 선택 기준

이 글의 판단은 이렇습니다. 전처리, 여러 모델, 후처리를 한 Python 저장소에서 다루며 단계마다 GPU와 replica를 다르게 잡아야 하면 Ray Serve가 기본 선택입니다. 단일 TensorFlow 모델을 최대 처리량으로 서빙하는 것이 목표라면 전용 서버가 더 단순합니다. 실험 추적, 모델 승인, 성능 대시보드가 필요하면 MLflow나 클라우드 ML 플랫폼을 함께 두고, Serve는 서빙 계층으로만 씁니다. 프로덕션에서는 처음부터 KubeRay RayService와 헤드 노드 장애 허용을 함께 설계하고, `max_queued_requests`에 유한한 값을 두는 편이 안전합니다.

- - -

### Sources
* [Ray Serve: Scalable and Programmable Serving — Ray 2.60.0](https://docs.ray.io/en/latest/serve/index.html)
* [Model Registry Integration — Ray 2.60.0](https://docs.ray.io/en/latest/serve/model-registries.html)
* [Getting Started — Ray 2.60.0](https://docs.ray.io/en/latest/serve/getting_started.html)
* [Key Concepts — Ray 2.60.0](https://docs.ray.io/en/latest/serve/key-concepts.html)
* [Architecture — Ray 2.60.0](https://docs.ray.io/en/latest/serve/architecture.html)
* [Deploy Compositions of Models — Ray 2.60.0](https://docs.ray.io/en/latest/serve/model_composition.html)
* [Production Guide — Ray 2.60.0](https://docs.ray.io/en/latest/serve/production-guide/index.html)
* [Develop and Deploy an ML Application — Ray 2.60.0](https://docs.ray.io/en/latest/serve/develop-and-deploy.html)
* [Ray Serve API — Ray 2.60.0](https://docs.ray.io/en/latest/serve/api/index.html)
* [Ray Serve API — Ray 2.60.0](https://docs.ray.io/en/latest/serve/api/)
* [Deploy Multiple Applications — Ray 2.60.0](https://docs.ray.io/en/latest/serve/multi-app.html)
* [Configure Ray Serve deployments — Ray 2.60.0](https://docs.ray.io/en/latest/serve/configure-serve-deployment.html)
* [Serve Config Files — Ray 2.60.0](https://docs.ray.io/en/latest/serve/production-guide/config.html)
* [Deploy on Kubernetes — Ray 2.60.0](https://docs.ray.io/en/latest/serve/production-guide/kubernetes.html)
* [Deploy Ray Serve Applications — Ray 2.60.0](https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/rayservice.html)
* [Ray: A Distributed Framework for Emerging AI Applications](https://arxiv.org/html/1712.05889)
* [Deploy on Kubernetes — Ray 3.0.0.dev0](https://docs.ray.io/en/master/serve/production-guide/kubernetes.html)
