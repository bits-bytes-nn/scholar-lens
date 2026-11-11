---
layout: post
title: "Ray Serve 2.60: Deployments, Request Routing and KubeRay Operations"
date: 2026-11-09 00:10:37
description: "Ray Serve는 여러 모델을 Python 코드로 엮어 서빙하는 방식을 다룹니다. deployment와 레플리카 구조, 파워 오브 투 초이스 라우팅, 오토스케일링, KubeRay RayService 운영 방법을 정리합니다."
categories: ["Tech Guides", "Data-Architecture"]
tags: ["Ray-Serve", "Deployment-Handle", "Power-of-Two-Choices", "KubeRay", "Autoscaling", "Model-Serving", "MLOps"]
cover: /assets/images/data-architecture.jpg
use_math: true
---
## TL;DR

Ray Serve는 온라인 추론 API를 만드는 확장 가능한 모델 서빙 라이브러리이며, 특정 ML 프레임워크에 묶이지 않습니다. 기본 단위는 요청을 처리하는 Python 클래스나 함수인 deployment입니다. deployment는 런타임에 레플리카 여러 개로 실행되고, 레플리카마다 별도의 Ray 액터(프로세스)가 뜨며, 부하에 따라 레플리카 수를 늘리거나 줄일 수 있습니다. deployment 하나 이상을 묶은 애플리케이션은 업그레이드 단위이고, 그중 ingress deployment가 인바운드 트래픽을 모두 받습니다. 여러 모델은 정적 YAML 그래프가 아니라 프로그래밍 가능한 API로 합성하므로, 다른 모델을 호출하는 코드가 함수 호출과 같은 모양이 됩니다. deployment를 서로의 생성자 인자로 엮어 두면 실행 중에는 Serve가 넣어 준 핸들로 서로를 호출하게 됩니다. 이 구조 덕분에 전처리·추론·후처리를 각각 독립 deployment로 나누고 단계마다 따로 스케일할 수 있습니다.

내부에서는 Serve 인스턴스마다 하나뿐인 전역 컨트롤러 액터가 다른 액터를 생성·갱신·삭제합니다. 요청은 레플리카 두 개를 무작위로 골라 처리 중인 요청이 적은 쪽에 보내는 파워 오브 투 초이스(power of two choices) 전략으로 분배됩니다. 이때 레플리카 하나가 동시에 처리하는 요청 수의 상한인 `max_ongoing_requests`를 넘는 레플리카는 없습니다. 오토스케일러는 컨트롤러 안에서 돌아가고, 핸들과 레플리카가 주기적으로 보내는 메트릭을 근거로 레플리카 수를 정합니다. 컨트롤러가 죽어도 요청은 계속 처리되지만, 그동안 오토스케일링은 멈추고 복구되면 이전 메트릭은 모두 사라집니다. Ray 클러스터를 Kubernetes에서 관리하는 오퍼레이터인 KubeRay 없이 운영하면, Ray 클러스터 자체가 실패했을 때 Serve가 복구되지 않습니다. Ray 클러스터와 그 위의 Serve 애플리케이션을 하나로 묶어 실행하는 KubeRay 리소스인 RayService는 노드 장애와 클러스터 장애를 복구합니다. Ray Serve가 잘 맞는 경우는 앞에서 본 합성 API로 여러 모델과 비즈니스 로직을 하나의 Python 애플리케이션으로 엮고, 단계마다 따로 스케일해야 할 때입니다. Kubernetes에서 운영한다면 장애 복구는 RayService에 맡길 수 있습니다.

## 정적 그래프 대신 Python 함수 호출로 엮는 합성 서빙

여러 모델과 로직 단계로 이뤄진 온라인 추론 API는 단계마다 필요한 자원과 트래픽이 달라서, 모델 하나를 엔드포인트 하나로 감싸는 서버만으로는 감당하기 어렵습니다.

### 모델 하나로 끝나지 않는 온라인 추론

ML 서빙 솔루션 상당수는 "tensor-in, tensor-out" 방식에 머뭅니다. 모델을 미리 정의된 구조의 엔드포인트 뒤에 두고, 텐서를 받아 텐서를 돌려주는 데까지만 책임집니다. 실제 서비스에서 모델은 혼자 쓰이지 않습니다. 데이터베이스 조회 같은 비즈니스 로직, 일반적인 웹 요청 처리와 함께 묶여야 쓸모가 생깁니다. LLM 애플리케이션이 전형적인 예입니다. 프롬프트 전처리, 벡터 DB 조회, LLM API 호출, 응답 검증이 한 요청 안에서 차례로 실행됩니다.

단계마다 아키텍처나 프레임워크가 다르고, CPU를 쓰는 단계와 GPU를 쓰는 단계가 섞입니다. 예를 들어 긴 영어 문서를 요약한 뒤 번역해 돌려주는 텍스트 서비스를 생각해 보면, 다음과 같은 흐름이 나옵니다.

```text
HTTP 요청 ──▶ [Summarizer] ──▶ [Translator] ──▶ 응답
               요약 모델          번역 모델
               (자원 A, 복제 n개)  (자원 B, 복제 m개)
```

요약이 병목이면 Summarizer만, 번역이 병목이면 Translator만 늘려야 자원이 낭비되지 않습니다. 두 모델을 한 프로세스에 묶어 통째로 복제하면 한가한 단계까지 같이 늘어납니다.

### 설정 언어 그래프의 한계

이런 다단계 파이프라인을 위해 많은 솔루션은 YAML 같은 설정 언어로 정적 그래프를 정의하게 합니다. 이 방식은 제약이 많고 다루기 어렵습니다.

Ray Serve는 정적 그래프 대신 프로그래밍 가능한 API로 모델 조합(model composition)을 지원합니다. 다른 모델 호출이 함수 호출과 똑같이 보입니다. 각 모델은 서로 다른 자원을 쓰고 클러스터의 여러 머신에 흩어져 실행되지만, 코드는 평범한 프로그램처럼 작성합니다. 앞의 요약·번역 서비스라면 Summarizer 코드 안에서 Translator를 호출하는 한 줄로 두 단계가 이어집니다.

### Ray 위에 놓인 범용 서빙 계층

Ray Serve는 온라인 추론 API를 만드는 확장 가능한 모델 서빙 라이브러리입니다. 프레임워크에 묶이지 않아 PyTorch·TensorFlow·Keras 딥러닝 모델부터 Scikit-Learn 모델, 임의의 Python 비즈니스 로직까지 같은 도구로 서빙합니다. 서빙 단위를 복제한 레플리카(replica)의 개수를 늘리고 줄여 모델별 자원을 동적으로 조정하고, 요청을 묶어 벡터화 연산의 이점을 살리는 배칭과 유연한 자원 할당 모델을 기본으로 제공합니다.

Ray 위에 구축되어 여러 머신으로 쉽게 확장되고, 분할 GPU(fractional GPU) 같은 유연한 스케줄링으로 여러 모델이 자원을 나눠 씁니다. 여러 모델과 비즈니스 로직을 Python 코드 하나로 묶는 합성 서빙과 멀티 모델 서빙에도 특히 잘 맞습니다. LLM 애플리케이션이라면 앞서 든 전처리부터 검증까지의 단계를 Python 모듈 하나로 작성하고, 배포한 뒤에는 단계마다 독립적으로 오토스케일링(autoscaling)해 자원 낭비 없이 트래픽을 받아냅니다. 챗봇형 서비스에 필요한 스트리밍 응답도 지원하며, LLM용으로는 동적 배칭(dynamic batching)과 멀티 노드·멀티 GPU 서빙이 더해집니다.

임의의 Python 코드를 실행하므로 주변 MLOps 도구와도 맞물립니다. ONNX·TVM 같은 모델 최적화기, Seldon Alibi·Arize 같은 모니터링 시스템, MLflow·Weights and Biases 같은 모델 레지스트리, Gradio·Streamlit 같은 데이터 앱 UI, FastAPI·gRPC 같은 웹 API 프레임워크를 함께 씁니다. Ray Serve는 이들을 대체하지 않습니다. 모델 최적화나 수명 주기 관리는 이 도구들에 맡기고, 그 위에서 여러 단계를 조합하고 확장하는 일을 맡습니다.

## Deployment, replica, application: 스케일과 업그레이드의 세 단위

Serve 코드는 세 가지 단위로 읽힙니다. 독립적으로 스케일되는 deployment, 그 deployment를 실제로 실행하는 레플리카, 그리고 deployment들을 묶어 한 번에 배포하고 업그레이드하는 애플리케이션(application)입니다.

### 독립적으로 스케일되는 처리 로직, deployment

deployment는 Ray Serve의 중심 개념입니다. 들어오는 요청을 처리하는 비즈니스 로직이나 ML 모델을 담고, Ray 클러스터 전체로 확장해 실행됩니다. 정의할 때는 Python 클래스에 `@serve.deployment` 데코레이터를 붙입니다. 데코레이터가 클래스를 Ray Serve의 `Deployment` 객체로 바꿉니다. 간단한 경우에는 함수에도 붙일 수 있습니다.

deployment 하나는 사용자가 작성한 함수나 클래스 하나를 담아 그것으로 요청을 처리합니다. deployment마다 데코레이터 파라미터로 따로 스케일하고 설정합니다. 이 글은 영어 텍스트를 프랑스어로 옮기는 Translator로 시작해 나중에 Summarizer를 덧붙이는 텍스트 ML 애플리케이션 하나를 예제로 계속 씁니다. 앞서 본 요약·번역 서비스에서는 이 둘이 각각 deployment가 됩니다.

### 런타임의 실체, Ray 액터로 뜨는 replica

deployment는 설계도에 가깝고, 실제로 요청을 처리하는 쪽은 레플리카입니다. 런타임에 deployment는 여러 레플리카로 이뤄지며, 각 레플리카는 클래스나 함수의 개별 복사본으로 별도의 Ray 액터(프로세스)에서 시작됩니다. 레플리카 수는 들어오는 요청량에 맞춰 늘리거나 줄일 수 있고, 오토스케일링도 됩니다.

Translator를 레플리카 2개로 설정하면 클러스터에는 Translator 클래스를 각자 인스턴스화한 Ray 액터 프로세스 두 개가 뜹니다. 모델 가중치도 프로세스마다 따로 메모리에 올라갑니다. 레플리카 수와 자원 같은 파라미터의 구체적인 지정 방법과 우선순위는 deployment 설정을 다루는 부분에서 정리합니다.

### 업그레이드의 단위, application과 인그레스 deployment

애플리케이션은 Ray Serve 클러스터에서 업그레이드하는 단위입니다. deployment 하나 이상으로 구성되며, 그중 하나가 모든 인바운드 트래픽을 받는 인그레스 deployment(ingress deployment)입니다. `route_prefix`는 애플리케이션이 HTTP 요청을 받는 URL 경로 접두사로, 이 글의 예시에서는 `/`를 씁니다. 외부에서는 지정한 `route_prefix`로 HTTP 호출을 하거나, Python에서 `DeploymentHandle`로 호출해 애플리케이션에 접근합니다.

요약·번역 서비스를 하나의 애플리케이션으로 묶으면 구조는 대략 아래 그림처럼 됩니다. 단위 사이의 관계를 보여 주기 위한 예시입니다.

```text
┌──────────────── application (업그레이드 단위) ────────────────┐
│                                                                │
│  HTTP ──▶ [Summarizer]  ingress deployment                     │
│              ├─ replica (Ray 액터)                             │
│              └─ replica (Ray 액터)                             │
│                    │ DeploymentHandle 호출                     │
│                    ▼                                           │
│           [Translator]  deployment                             │
│              ├─ replica (Ray 액터)                             │
│              └─ replica (Ray 액터)                             │
└────────────────────────────────────────────────────────────────┘
```

스케일 단위는 deployment이고 배포·업그레이드 단위는 애플리케이션입니다. Summarizer와 Translator가 한 애플리케이션 안에 있으면 레플리카 수는 따로 조정되지만 새 버전을 올릴 때는 함께 올라갑니다.

### bind로 정의하고 serve.run으로 띄우는 흐름

deployment를 정의한 뒤에는 `bind`로 생성자에 넘길 인자를 묶습니다. 이 호출이 로컬에서 실행하거나 프로덕션에 배포할 수 있는 애플리케이션을 정의합니다. 그렇게 만든 애플리케이션을 `serve.run`이나 같은 역할의 `serve run` CLI 명령으로 배포합니다. 예제를 돌리려면 Serve와 함께 t5-small 모델이 쓰는 패키지도 설치합니다.

```bash
pip install "ray[serve]" transformers torch
```

Translator deployment를 정의하고 `bind`로 묶어 `serve.run`으로 띄우는 세 단계를 한 파일에 담으면 이렇게 됩니다.

```python
from starlette.requests import Request
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ray import serve
from ray.serve.handle import DeploymentHandle


@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0.2, "num_gpus": 0})
class Translator:
    def __init__(self):
        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

    def translate(self, text: str) -> str:
        input_ids = self.tokenizer(
            f"translate English to French: {text}", return_tensors="pt"
        ).input_ids
        output_ids = self.model.generate(
            input_ids, num_beams=4, early_stopping=True, max_length=300
        )
        return self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    async def __call__(self, http_request: Request) -> str:
        english_text: str = await http_request.json()
        return self.translate(english_text)


translator_app = Translator.bind()
handle: DeploymentHandle = serve.run(translator_app)
```

데코레이터에 넘긴 값은 Translator 하나에만 적용되는 설정입니다.

| 파라미터 | 예시 값 | 의미 |
| --- | --- | --- |
| `num_replicas` | `2` | Ray에서 실행할 deployment 프로세스 복사본 수. 요청은 레플리카들에 로드밸런싱됩니다 |
| `ray_actor_options["num_cpus"]` | `0.2` | 레플리카마다 예약할 논리 CPU 수. 분수로 지정할 수 있습니다 |
| `ray_actor_options["num_gpus"]` | `0` | 레플리카마다 예약할 논리 GPU 수 |

`bind`는 애플리케이션을 정의하고, 실제 배포는 `serve.run`이 맡습니다. `serve.run`이 돌려주는 `DeploymentHandle`은 Python에서 애플리케이션을 호출하는 통로입니다. 이 핸들로 deployment끼리 엮는 방법은 Summarizer→Translator 파이프라인에서 다룹니다.

## 컨트롤러·프록시·레플리카를 거치는 요청 경로와 파워 오브 투 초이스 라우팅

HTTP 요청이든 핸들 호출이든 Serve 안에서는 같은 길을 지납니다. 요청은 deployment의 큐에 들어갔다가 `max_ongoing_requests` 한도에 여유가 있는 레플리카로 보내지고, 레플리카를 고르는 데는 파워 오브 투 초이스(power of two choices) 전략이 쓰입니다.

### Serve 인스턴스를 구성하는 세 종류의 Ray 액터

Serve 인스턴스는 모두 Ray 액터로 이뤄집니다. 그중 컨트롤러(controller)는 인스턴스마다 하나만 있는 전역 액터로, 컨트롤 플레인을 관리하면서 나머지 액터를 생성·갱신·삭제합니다. 외부 요청은 프록시(proxy) 액터가 받습니다. HTTP 프록시는 Uvicorn HTTP 서버를 띄워 요청을 받아 레플리카로 넘기고, 처리가 끝나면 응답합니다. 유효한 `port`와 `grpc_servicer_functions`를 주고 Serve를 시작하면, HTTP 프록시와 함께 grpcio 서버를 돌리는 gRPC 프록시도 뜹니다. 실제 사용자 코드를 실행하는 쪽은 레플리카 액터이며, 앞서 정의한 Translator의 코드도 레플리카에서 실행됩니다.

아래 아키텍처 그림에서는 컨트롤 플레인과 데이터 플레인이 어떻게 갈리는지를 보면 됩니다. 컨트롤러는 다른 액터를 관리할 뿐이고, 요청은 프록시에서 레플리카로 곧장 흘러갑니다.

![Serve 애플리케이션을 이루는 컨트롤러, 프록시, 레플리카 액터의 관계도](https://docs.ray.io/en/latest/_images/architecture-2.0.svg)

### 헤드 노드 하나에서 모든 노드로 넓힐 수 있는 프록시

HTTP 프록시는 기본적으로 헤드 노드(head node)에 하나만 뜹니다. 확장성과 고가용성이 필요하면 `serve.start()`나 config 파일의 `proxy_location` 필드를 설정해 클러스터의 모든 노드에서 프록시를 실행합니다. 이렇게 하면 프록시가 모두 같은 포트에 바인딩되므로 어느 노드로 요청을 보내도 Serve에 닿습니다. 그 위에 별도의 로드밸런서를 둘 수도 있습니다. 이 구조에서 HTTP·gRPC 인그레스는 노드를 추가해 늘리고, 모델 추론은 deployment의 `num_replicas`를 올려 늘립니다. 인그레스 용량과 추론 용량을 서로 다른 손잡이로 조절하는 셈입니다. 로컬에서 `serve run`으로 띄운 Translator라면 노드가 하나뿐이어서 기본값만으로도 충분합니다.

### 파싱·deployment 조회·레플리카 선택의 세 단계

HTTP나 gRPC 요청이 프록시에 도착하면 처리는 세 단계로 진행됩니다.

1. 요청을 받아 파싱합니다.
2. HTTP URL 경로나 애플리케이션 이름 메타데이터로 해당 deployment를 찾아 요청을 큐에 넣습니다.
3. 큐의 요청마다 사용 가능한 레플리카를 찾아 보냅니다. 모든 레플리카가 이미 `max_ongoing_requests`를 넘는 요청을 안고 있으면, 자리가 날 때까지 요청은 큐에서 기다립니다.

DeploymentHandle로 호출하면 요청은 핸들 안의 큐에 들어가고, 두 번째 단계를 건너뛰어 곧바로 레플리카 선택으로 갑니다. Summarizer는 뒤의 모델 합성 섹션에서 추가할 deployment로, DeploymentHandle을 통해 Translator를 호출합니다. 핸들은 같은 노드에 있는 "라우터"의 핸들을 감싸고 있어서, Summarizer 레플리카가 Translator를 호출할 때도 요청은 외부 HTTP·gRPC 요청과 같은 데이터 경로를 탑니다. 그래서 deployment 선택과 배칭도 똑같이 적용됩니다. 텍스트 ML 애플리케이션에 대입해 두 경로를 겹쳐 그리면 아래와 같습니다.

```text
클라이언트 ── POST / ──▶ [HTTP 프록시]  ① 파싱  ② route_prefix로 deployment 조회
                              │
                              ▼  Summarizer 큐
                     ③ 레플리카 선택 ──▶ [Summarizer 레플리카]
                                              │ DeploymentHandle 호출
                                              ▼  (②를 건너뜀) 핸들 큐
                                     ③ 레플리카 선택 ──▶ [Translator 레플리카 A | B]
```

### 무작위 두 후보 중 덜 붐비는 쪽을 고르는 파워 오브 투 초이스

DeploymentHandle은 컨트롤러를 계속 폴링하며 새 레플리카가 생겼는지 확인합니다. 새 레플리카를 발견하면 버퍼에 쌓인 요청과 새 요청을 `max_ongoing_requests`에 이를 때까지 그 레플리카로 보냅니다. 어느 레플리카로 보낼지는 파워 오브 투 초이스 스케줄링으로 정하며, 어떤 레플리카도 동시에 `max_ongoing_requests`를 넘는 요청을 처리하지 않도록 제약합니다. 이 전략은 IEEE에 실린 동명의 논문에서 나왔습니다.

이 알고리즘은 일반적으로 후보 둘을 무작위로 뽑아 그중 덜 붐비는 쪽에 요청을 보냅니다. 숫자를 넣어 따라가 보면 이렇습니다. Translator를 레플리카 4개로 늘렸고, 진행 중 요청 수가 A=3, B=1, C=5, D=2이며, 한도를 가상의 값 5로 잡았다고 가정합니다. 무작위로 A와 C가 뽑히면 C는 한도에 걸려 있으니 A가 받고, B와 D가 뽑히면 B가 받습니다. 매번 전체를 훑어 최솟값을 찾지 않아도 가장 붐비는 레플리카는 대부분의 비교에서 지게 되므로, 부하가 비교적 고르게 퍼집니다. 네 레플리카가 모두 한도에 닿으면 요청은 앞서 본 세 번째 단계처럼 큐에서 기다립니다.

### 같은 노드의 레플리카를 먼저 찾는 요청 라우터

레플리카를 고르는 로직은 요청 라우터(request router)라는 교체 가능한 구성 요소이며, 그 추상 인터페이스는 `serve.request_router.RequestRouter`입니다.

LLM 서빙 구성에서 인그레스가 deployment 핸들로 `LLMServer`를 호출할 때 어느 노드의 레플리카에든 닿을 수 있지만, 기본 요청 라우터는 노드 간 RPC 오버헤드를 줄이려고 같은 노드의 레플리카를 우선합니다. 이 오버헤드는 LLM 서빙에서는 미미해서, 높은 동시성에서도 TTFT(time to first token)에 몇 ms 정도 영향을 줍니다. 그림의 실선은 선호되는 로컬 RPC이고, 점선은 로컬 레플리카가 바쁠 때 일어날 수 있는 노드 간 RPC입니다.

![인그레스에서 LLMServer 레플리카로 가는 로컬 RPC와 노드 간 RPC 경로](https://docs.ray.io/en/latest/_images/llmserver-ingress-rpc.png)

Summarizer와 Translator 레플리카가 여러 노드에 흩어져 있다면, Summarizer의 핸들 호출도 대개 같은 노드의 Translator로 먼저 향하는 그림이 됩니다.

### 레플리카 안의 실행: async 핸들러, 배칭, 예외 처리

레플리카는 자체 요청 큐를 두고 요청을 하나씩 실행하며, `asyncio`로 여러 요청을 동시에 처리할 수도 있습니다. 핸들러(deployment 함수나 클래스의 `__call__`)를 `async def`로 선언하면 레플리카는 핸들러가 끝나기를 기다리지 않습니다. 일반 `def`이면 핸들러가 반환할 때까지 블록됩니다.

레플리카는 `@serve.batch`로 요청을 묶어 처리할 수도 있습니다. `serve.batch`는 함수를 배치 단위로 비동기 처리하는 함수로 바꿉니다. Summarizer가 핸들로 보낸 번역 요청도 앞서 본 공통 데이터 경로를 따라 이 배칭을 거칩니다.

모델 평가 코드에서 난 예외 같은 애플리케이션 오류는 Serve가 잡아서 감쌉니다. 호출자는 traceback이 담긴 500 상태 코드를 받고, 레플리카는 계속 요청을 처리합니다. 액터 자체가 죽는 장애는 이와 달리 복구로 처리됩니다. 컨트롤러 액터가 실패하면 Ray가 다시 시작합니다. 실패한 레플리카 액터는 새 액터로 교체하고, 실패한 프록시 액터는 다시 시작합니다.

### 100KiB를 경계로 갈리는 요청 전달 경로

요청 객체를 넘기는 방식은 크기에 따라 다릅니다. Serve는 Ray의 공유 메모리 객체 저장소(object store)와 프로세스 내 메모리 저장소를 함께 씁니다. 작은 요청 객체는 액터 사이의 네트워크 호출로 직접 전달되고, 100KiB 이상인 큰 요청 객체는 객체 저장소에 기록된 뒤 레플리카가 zero-copy로 읽습니다. Translator에 POST하는 짧은 영어 문장은 첫 번째 경로로 갑니다. 반면 Summarizer가 긴 문서 묶음을 그대로 넘겨 크기가 수백 KiB에 이르면 두 번째 경로를 타게 됩니다.

## 컨트롤러 안의 오토스케일러와 액터별 장애 복구 범위

레플리카 수를 늘리고 줄이는 판단은 컨트롤러 액터 안의 오토스케일러가 내리며, 근거는 DeploymentHandle과 레플리카가 주기적으로 보내 주는 메트릭입니다. 오토스케일링 상태가 컨트롤러에 모여 있으므로 장애가 나면 어느 액터가 죽었는지에 따라 남는 것과 사라지는 것이 달라집니다.

### 핸들 큐와 레플리카의 진행 중 요청으로 내리는 스케일 결정

오토스케일링은 deployment의 부하에 맞춰 레플리카 수를 자동으로 늘리거나 줄이는 기능입니다. 오토스케일러가 메트릭을 직접 긁어 가지는 않습니다. 각 DeploymentHandle과 각 레플리카가 자기 메트릭을 오토스케일러에 주기적으로 푸시합니다. 오토스케일러는 deployment마다 주기적으로 두 신호를 살펴 스케일 여부를 정합니다. 하나는 DeploymentHandle의 큐에 쌓인 요청이고, 다른 하나는 레플리카에서 처리 중인 요청(in-flight query)입니다.

아래 구조도에서 화살표 방향을 보면 됩니다. 메트릭은 핸들과 레플리카에서 컨트롤러 안의 오토스케일러로 올라가고, 스케일 결정은 컨트롤러에서 레플리카 수 조정으로 내려옵니다.

![컨트롤러 안의 오토스케일러가 핸들과 레플리카에서 메트릭을 받아 레플리카 수를 조정하는 구조도](https://raw.githubusercontent.com/ray-project/images/master/docs/serve/autoscaling.svg)

두 신호를 함께 보는 이유는 앞서 본 요청 경로에서 드러납니다. 모든 레플리카가 `max_ongoing_requests`에 닿으면 요청은 큐에서 기다립니다. 레플리카 쪽 진행 중 요청 수만 보면 한도에 막혀 있는 수요가 드러나지 않고, 큐 길이를 함께 봐야 그 수요가 잡힙니다. 텍스트 ML 애플리케이션으로 예를 들어 보겠습니다. Summarizer가 핸들로 Translator를 호출하면 대기 중인 번역 요청은 Summarizer 레플리카 안의 핸들 큐에 쌓이고, 오토스케일러는 이 큐 길이와 Translator 레플리카 두 개의 진행 중 요청 수를 같이 보고 Translator를 늘릴지 정합니다. 새로 뜬 레플리카는 앞서 본 핸들의 컨트롤러 폴링으로 발견되어 바로 트래픽을 받습니다.

### 임계값 기반 기본 정책과 교체 가능한 커스텀 정책

별도로 지정하지 않으면 `serve.autoscaling_policy.replica_queue_length_autoscaling_policy`가 정책으로 쓰입니다. 이 기본 정책은 단순한 임계값을 기준으로 스케일합니다. 정책은 직접 작성한 것으로 바꿀 수 있습니다. 커스텀 오토스케일링 정책에는 `serve.config.AutoscalingContext`라는 컨텍스트 객체가 넘어갑니다. `serve.config.AggregationFunction`은 알파 API여서 안정화 전에 바뀔 수 있습니다. 커스텀 정책을 프로덕션에 넣는다면 이 API에 의존하는 부분은 Ray 버전을 올릴 때마다 다시 확인하는 편이 안전합니다. 정책에 넘기는 목표값과 상하한 같은 파라미터는 뒤의 deployment 설정 우선순위 섹션에서 `autoscaling_config`와 함께 다룹니다.

### 인그레스와 LLMServer를 2:1로 묶어 두는 목표값 계산

여러 deployment가 엮인 애플리케이션에서는 각 deployment가 독립적으로 스케일되므로, 단계 사이의 비율을 맞추는 일이 따로 남습니다. Ray Serve LLM 구성에서는 동시성이 높을 때 인그레스의 이벤트 루프가 병목이 될 수 있습니다. 이때 인그레스 레플리카를 늘리면 CPU 경합이 줄어들며, 인그레스 레플리카 수와 `LLMServer` 레플리카 수의 비율을 최소 2:1로 유지하도록 권장합니다. 오토스케일링 중에도 이 비율을 지키려면 두 deployment의 `target_ongoing_requests`를 비례하게 잡습니다. 계산은 네 단계로 진행됩니다.

1. vLLM 설정을 프로파일링해 최대 동시 요청 수를 찾습니다.
2. 인그레스 대 `LLMServer` 비율을 고릅니다(예: 2:1).
3. `LLMServer`의 `target_ongoing_requests`를 최대 용량의 일정 비율로 둡니다.
4. 인그레스의 `target_ongoing_requests`는 비율이 유지되도록 맞춥니다.

예시 숫자로 따라가면 요청 하나가 인그레스와 `LLMServer`를 한 번씩 지나가므로 두 deployment가 받는 동시 요청 총량은 같습니다. 동시 요청이 480개라면 `LLMServer`는 레플리카당 48개를 목표로 10개, 인그레스는 24개를 목표로 20개가 되어 2:1이 저절로 유지됩니다. Summarizer→Translator 파이프라인에도 같은 계산을 적용할 수 있습니다. 앞단과 뒷단의 목표값 비율을 원하는 레플리카 비율의 역수로 잡으면 됩니다.

### 죽은 컨트롤러가 멈추는 것은 트래픽이 아니라 스케일링

오토스케일러가 컨트롤러 안에서 돌기 때문에, 컨트롤러가 죽으면 트래픽과 오토스케일링이 서로 다른 영향을 받습니다. 컨트롤러가 죽어도 HTTP, gRPC, DeploymentHandle로 보내는 요청은 계속 처리되지만 오토스케일링은 멈춥니다. 컨트롤러가 복구되면 오토스케일링이 재개되지만 그때까지 모은 메트릭은 모두 사라집니다. 앞서 본 대로 컨트롤러 액터는 Ray가 다시 시작합니다.

운영에서는 이 공백을 염두에 둬야 합니다. 트래픽이 몰리는 도중에 컨트롤러가 재시작되면 Translator는 그 순간의 레플리카 수(예를 들어 2개)에 머물고, 복구 직후 오토스케일러는 빈 메트릭에서 다시 시작합니다. 피크 시간대에 레플리카 하한을 넉넉히 잡아 두면 이 공백 동안 늘어나는 지연을 줄일 수 있습니다.

### GCS에 남는 컨트롤러 상태와 사라지는 일시 데이터

액터 단위 장애 허용(fault tolerance)은 액터마다 복구 주체가 다릅니다. 액터가 올라가 있던 머신이 통째로 죽으면 그 액터들은 다른 가용 머신에서 자동으로 다시 시작됩니다. 다시 뜬 액터가 무엇을 되찾는지는 데이터 종류에 따라 갈립니다. 라우팅 정책과 deployment 설정 같은 컨트롤러의 데이터는 모두 헤드 노드의 Ray GCS(Global Control Store)에 체크포인트됩니다. 반면 라우터와 레플리카에 있던 네트워크 연결이나 내부 요청 큐 같은 일시 데이터는 이런 장애에서 사라집니다.

| 장애 대상 | 복구 주체 | 보존되는 것 | 유실되는 것 |
| --- | --- | --- | --- |
| 레플리카 액터 | 컨트롤러가 새 액터로 교체 | 컨트롤러가 가진 deployment 설정 | 레플리카의 내부 요청 큐, 네트워크 연결 |
| 프록시 액터 | 컨트롤러가 재시작 | 컨트롤러가 가진 라우팅 정책 | 라우터의 일시 데이터 |
| 컨트롤러 액터 | Ray가 재시작 | GCS에 체크포인트된 라우팅 정책, deployment 설정 | 수집된 오토스케일링 메트릭 |
| Ray 클러스터 전체 | KubeRay RayService(쓰지 않으면 복구 불가) | KubeRay가 클러스터를 다시 세움 | KubeRay 없이는 Serve 전체 |

머신 장애에서는 내부 요청 큐가 유실 대상이므로, Translator 레플리카가 올라가 있던 머신이 죽으면 그 레플리카의 내부 큐에 있던 번역 요청도 함께 사라집니다.

### KubeRay 없이는 돌아오지 않는 클러스터 장애

클러스터 전체가 무너지는 장애에서 복구가 되는지는 KubeRay를 쓰는지에 달려 있습니다. KubeRay를 쓰지 않는 환경에서 Ray 클러스터가 실패하면 Ray Serve는 복구할 수 없습니다. KubeRay RayService를 쓰면 KubeRay가 죽은 노드나 죽은 클러스터를 복구하고, GCS FT 기능을 켜면 클러스터 크래시 자체를 피할 수 있습니다. 텍스트 ML 애플리케이션을 RayService 매니페스트로 올리는 과정과 GCS 장애 허용 설정은 뒤의 KubeRay 섹션에서 다룹니다.

## FastAPI 인그레스를 단 Translator 첫 배포와 레지스트리 모델 로딩

텍스트 ML 애플리케이션의 첫 단계는 평범한 Python 번역 클래스에 데코레이터 두 개와 `bind` 호출 하나를 더해 로컬에서 HTTP로 질의하는 것입니다. 설치는 명령 한 줄로 끝납니다.

```bash
pip install "ray[serve]"
```

### 데코레이터 두 개로 바뀌는 번역 클래스

t5-small로 영어를 프랑스어로 옮기는 `Translator` 클래스를 FastAPI 기반 Serve 애플리케이션으로 바꾸려면 세 군데를 고칩니다. Ray Serve와 FastAPI 의존성을 가져오고, `@serve.deployment`와 `@serve.ingress(app)` 데코레이터를 붙이고, deployment에 생성자 인자를 넘겨 `bind`합니다(이 예제는 인자가 없습니다). 아래 `model.py`에서는 첫째 변경이 import 줄과 `app = FastAPI()`로, 둘째 변경이 클래스 위의 두 데코레이터로, 셋째 변경이 마지막 줄의 `bind`로 들어갑니다. 둘째 변경에 따라 `translate` 메서드에는 FastAPI 경로 데코레이터 `@app.post("/")`도 붙고, 원래 스크립트 끝에서 인스턴스를 만들어 번역을 출력하던 줄은 빠집니다.

```python
# File name: model.py
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

레플리카 수와 자원은 앞의 Translator와 같습니다. 이 파라미터를 코드 밖에서 바꾸는 방법과 적용 순서는 뒤에서 deployment 파라미터 우선순위를 설명할 때 다룹니다. `translator_app`이 배포할 애플리케이션 객체입니다.

FastAPI는 HTTP 요청을 대신 파싱하고 검증합니다. 그래서 `translate`는 `text: str` 인자를 그대로 받습니다. FastAPI 없이 인그레스 deployment를 쓰면 클래스의 `__call__`이 `Starlette` 요청 객체를 받습니다. 이때 반환값은 JSON으로 직렬화되며, 다른 `Starlette` 응답 객체를 직접 돌려줄 수도 있습니다. 이 방식에서는 `await request.json()`처럼 본문을 직접 읽어야 합니다. 경로와 쿼리 파라미터가 늘어나는 API라면 FastAPI 쪽이 관리하기 편합니다.

### `serve run`으로 띄우고 기본 주소로 질의

`serve run`은 `module:application` 형식의 import 경로를 받습니다. `model.py`가 있는 디렉터리에서 다음 명령을 실행합니다.

```bash
serve run model:translator_app
```

명령은 애플리케이션을 띄운 뒤 콘솔에 로그를 계속 출력하며 블록됩니다. `Ctrl-C`를 누르면 애플리케이션이 내려갑니다. 실행 중인 앱에는 기본 주소 `http://127.0.0.1:8000/`로 접근합니다. 다른 터미널에서 아래 클라이언트를 실행하면 "Hello world!"의 번역을 요청합니다.

```python
# File name: model_client.py
import requests

response = requests.post("http://127.0.0.1:8000/", params={"text": "Hello world!"})
french_text = response.json()

print(french_text)
```

이 요청도 앞서 설명한 프록시와 레플리카 선택 경로를 그대로 거칩니다. 실행 중인 애플리케이션과 deployment의 상태를 확인하는 `serve status`는 운영 도구를 다룰 때 함께 설명합니다.

### `__init__`에서 한 번 끝내는 모델 로딩

`Translator`는 토크나이저와 모델을 `__init__`에서 불러옵니다. 레플리카마다 별도 Ray 액터로 시작되므로, 로딩은 레플리카가 뜰 때 레플리카당 한 번 일어나고 요청 처리 경로에서는 빠집니다. 모델을 초기화 시점에 한 번 로드하는 warm-start는 첫 요청의 지연 급증을 없앱니다.

### 레지스트리 모델을 같은 방식으로 데우는 MLflow deployment

`Translator`가 허브에서 t5-small을 받듯이, 레지스트리에 등록된 모델도 `__init__`에서 한 번 불러와 데울 수 있습니다. Ray Serve는 임의의 Python 코드를 실행하므로 레지스트리의 표준 Python 클라이언트로 모델을 직접 불러오고, 로딩과 추론을 deployment 하나에 함께 둘 수 있습니다. 아래 MLflow 예제는 `production` 태그가 붙은 최신 모델을 찾아 로드한 뒤, 더미 입력으로 예측을 한 번 돌려 모델을 미리 데웁니다.

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

요청 처리부인 `__call__`은 JSON 본문의 `features`로 예측하고 결과를 리스트로 돌려줍니다. 클라이언트는 `{"features": [[0.0, 0.0, 0.0, 0.0]]}`처럼 특징 4개짜리 행을 담은 본문을 보내면 됩니다.

```python
# excerpt: MLflowModelDeployment의 __call__과 애플리케이션 바인딩
    async def __call__(self, request):
        data = await request.json()
        features = np.array(data["features"])

        # MLflow validates input against the logged signature automatically
        prediction = self.model.predict(features)
        return {"prediction": prediction.tolist()}

app = MLflowModelDeployment.bind()
```

모델을 레지스트리에 올릴 때 `mlflow.models.infer_signature`로 시그니처를 기록해 두면 MLflow가 이 입력을 검증할 수 있습니다.

## DeploymentHandle로 엮는 Summarizer→Translator 파이프라인

`.bind()`로 묶은 deployment를 다른 deployment의 생성자 인자로 넘기면 Serve가 런타임에 그 인자를 `DeploymentHandle`로 바꿉니다. 이 핸들을 써서 독립적으로 스케일되는 단계들을 Python 메서드 호출처럼 엮을 수 있습니다. 텍스트 ML 애플리케이션에서는 영어 텍스트를 먼저 요약하고, 그 요약문을 프랑스어로 번역하는 두 단계 파이프라인으로 확장합니다.

### 생성자 인자로 들어오는 Translator 핸들

합성 애플리케이션에서는 요청을 처음 받는 쪽이 `Summarizer`이므로 `Summarizer`가 `__call__`을 가집니다. `Summarizer`는 생성자 인자로 `Translator` 핸들을 받고, 요약한 텍스트를 이 핸들로 `Translator`에 전달합니다. 이제 `Translator`는 인그레스가 아니므로, 앞서 만든 FastAPI 데코레이터 없이 평범한 `@serve.deployment` 클래스로 남고 `translate` 메서드만 외부에 노출합니다. 파일 앞부분은 import와 `Translator`로 시작합니다.

```python
# excerpt: serve_quickstart_composed.py 앞부분 — import와 Translator
# File name: serve_quickstart_composed.py
from starlette.requests import Request

import ray
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
```

같은 파일의 뒷부분에는 `Summarizer`와 애플리케이션 정의가 이어집니다. 요약에도 t5-small을 하나 더 씁니다.

```python
# excerpt: serve_quickstart_composed.py 뒷부분 — Summarizer와 애플리케이션 바인딩
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

마지막 줄에서는 `Translator`를 빈 생성자 인자로 `.bind()`에 묶고, 그 결과를 `Summarizer`의 생성자 인자로 넘깁니다. 코드에서 `translator` 파라미터의 타입이 `DeploymentHandle`로 적혀 있는데, 런타임에 실제로 그 값이 들어오기 때문입니다. 호출은 `핸들.메서드.remote(인자)` 형태로 씁니다. `remote`는 다른 deployment에 보내는 핸들 호출이라는 표시이고, 인자와 키워드 인자는 원하는 만큼 넘길 수 있습니다. 넘긴 인자는 일반 Python 값 그대로 대상 메서드에 전달됩니다.

### 즉시 돌아오는 DeploymentResponse와 `await`

`self.translator.translate.remote(summary)`는 `Translator`의 `translate`를 비동기로 호출하고, 그 즉시 `DeploymentResponse` 객체를 돌려줍니다. 이 응답을 `await`하면 원격 메서드가 끝날 때까지 기다린 뒤 반환값을 받습니다. `DeploymentResponse`는 단일(unary) 핸들 호출의 결과를 감싸는 future 비슷한 객체입니다. 결과 자체가 아니라 결과를 가리키는 참조이므로 호출이 비동기로 진행됩니다.

결과를 꺼내는 방법으로 `response.result()`도 있습니다. 다만 deployment 안에서 이 메서드를 부르면 원격 호출이 끝날 때까지 deployment가 다른 코드를 전혀 실행하지 못합니다. 반면 `await`는 기다리는 동안 다른 요청을 처리하게 해 주므로, deployment 내부에서는 `await`를 씁니다. deployment 바깥에서는 사정이 다릅니다. `serve.run(app)`이 인그레스 deployment의 `DeploymentHandle`을 돌려주고, 드라이버 코드에서 `handle.remote(...).result()`로 결과를 받는 패턴을 그대로 쓸 수 있습니다.

합성 애플리케이션도 같은 `serve run` 명령으로 띄웁니다. 단, 파일이 있는 디렉터리에서 실행해야 합니다.

```bash
serve run serve_quickstart_composed:app
```

앞서 본 FastAPI 클라이언트는 쿼리 파라미터를 보냈습니다. 이번 `Summarizer.__call__`은 `await http_request.json()`으로 본문을 읽으므로, 클라이언트도 영어 텍스트를 JSON 본문에 담아 POST합니다.

```python
# File name: composed_client.py
import requests

english_text = (
    "It was the best of times, it was the worst of times, it was the age "
    "of wisdom, it was the age of foolishness, it was the epoch of belief"
)
response = requests.post("http://127.0.0.1:8000/", json=english_text)
french_text = response.text

print(french_text)
```

애플리케이션이 떠 있는 상태에서 다른 터미널로 클라이언트를 실행하면 요약 후 번역된 문장이 출력됩니다.

```text
$ python composed_client.py

C'était le meilleur des temps, c'était le pire des temps,
```

### `await` 없이 다음 핸들에 넘기는 체이닝

`DeploymentResponse`는 `await`하지 않고 다른 `DeploymentHandle` 호출의 인자로 그대로 넘길 수 있습니다. 이렇게 하면 여러 단계가 하나의 파이프라인으로 이어지고, `await` 처리는 Serve가 내부에서 맡습니다. 첫 호출이 끝나면 Serve는 응답 객체 대신 첫 호출의 출력값을 두 번째 호출에 넘깁니다. 값에 increment를 더하는 `Adder`와 multiple을 곱하는 `Multiplier`를 잇는 인그레스에서는 `__call__`이 이렇게 짧아집니다.

```python
# excerpt: chain.py의 Ingress.__call__과 호출 결과 (Adder·Multiplier 정의는 생략)
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

handle: DeploymentHandle = serve.run(app)
response = handle.remote(5)
assert response.result() == 12, "(5 + 1) * 2 = 12"
```

`Multiplier`는 응답 객체를 받는 것이 아니라 `Adder`의 출력인 6을 받고, 최종 결과는 $$(5+1)\times 2 = 12$$입니다. 응답 여러 개를 한 호출의 인자로 동시에 넘길 수도 있습니다. 두 `Adder` 호출의 응답을 `Combiner.average`에 함께 넘겨 평균을 내는 조합 예제가 이 방식입니다. 텍스트 ML 애플리케이션에서는 요약이 `Summarizer` 레플리카 안의 로컬 메서드라서 원격 호출이 하나뿐입니다. 요약을 별도 deployment로 분리하면 요약 응답을 `Translator` 핸들에 곧바로 넘기는 체인으로 바뀝니다.

### 단계마다 따로 정하는 레플리카 수와 자원

합성 애플리케이션에서도 `autoscaling_config`, `num_replicas`, `num_cpus`, `num_gpus` 같은 파라미터는 deployment마다 따로 설정합니다. 파이프라인의 각 단계는 별도 deployment이므로 개별적으로 구성하고 스케일할 수 있습니다. 합성 버전의 `@serve.deployment`에 옵션이 없는 이유는 이 파라미터들이 모두 선택 사항이라서입니다. 합성 예제는 이 옵션들을 생략했을 뿐이고, 단계마다 다시 붙일 수 있습니다. 텍스트 ML 애플리케이션이라면 `Translator`에만 처음 버전의 설정, 즉 레플리카 2개와 레플리카당 0.2 CPU·0 GPU를 되살리고 `Summarizer`는 평범한 `@serve.deployment`로 둘 수 있습니다. 이렇게 하면 `Translator` 쪽 데코레이터만 다음처럼 바뀝니다.

```python
# excerpt: 합성 버전에서 Translator에만 레플리카 수와 자원을 지정한 데코레이터
@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0.2, "num_gpus": 0})
class Translator:
```

`num_cpus`는 소수로 줄 수 있어서, 레플리카 수보다 CPU가 적은 머신에도 여러 레플리카를 함께 올릴 수 있습니다. 이 값을 코드 대신 config에서 조정하는 방법은 deployment 파라미터 우선순위를 다룰 때 설명합니다.

### `stream=True`로 받는 DeploymentResponseGenerator

출력을 여러 번 내보내는 스트리밍 호출도 핸들로 할 수 있습니다. 조건은 두 가지입니다. 호출되는 메서드가 제너레이터여야 하고, 핸들에 `options(stream=True)`를 설정해야 합니다. 그러면 핸들 호출은 단일 `DeploymentResponse` 대신 `DeploymentResponseGenerator`를 돌려줍니다.

```python
# File name: stream.py
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

handle: DeploymentHandle = serve.run(app).options(
    stream=True,
)

# Response generator can also be used as a regular generator in a sync context.
r: DeploymentResponseGenerator = handle.remote(10)
assert list(r) == list(range(10))
```

`DeploymentResponseGenerator`는 동기 제너레이터로도, 비동기 제너레이터로도 쓸 수 있습니다. 하지만 deployment 안에서 동기 제너레이터로 쓰면, deployment 안의 `.result()`와 마찬가지로 그 레플리카에서 다른 요청이 동시에 실행되지 못합니다. 또한 `DeploymentResponseGenerator`는 다른 핸들 호출의 인자로 넘길 수 없어서, 앞서 본 체이닝에 끼워 넣을 수 없습니다. 위 예제처럼 deployment 안에서는 `async for`로 소비하고, 동기 순회는 드라이버 쪽에서만 씁니다.

### Ray task와 엮는 `_to_object_ref`, 그리고 중첩 객체 제약

내부적으로 각 `DeploymentResponse`는 Ray `ObjectRef`에 대응하고, 스트리밍 호출이면 `ObjectRefGenerator`에 대응합니다. 핸들 호출을 일반 Ray 액터나 task와 조합하려면 개발자 API인 `_to_object_ref`나 `_to_object_ref_sync`로 응답을 `ObjectRef`로 바꿉니다. 아래 발췌에서는 인그레스가 `SayHi` deployment의 응답을 `ObjectRef`로 바꾼 뒤 `@ray.remote` task에 넘깁니다.

```python
# excerpt: response_to_object_ref.py의 Ingress.__call__ (say_hi_task와 SayHi 정의는 생략)
    async def __call__(self):
        # Make a call to the SayHi deployment and pass the result ref to
        # a downstream Ray task.
        response: DeploymentResponse = self._say_hi.remote()
        response_obj_ref: ray.ObjectRef = await response._to_object_ref()
        final_obj_ref: ray.ObjectRef = say_hi_task.remote(response_obj_ref)
        return await final_obj_ref
```

같은 변환은 중첩 객체 안에 응답을 담아 보낼 때도 필요합니다. 리스트나 딕셔너리 같은 중첩 객체 안에 `DeploymentResponse`를 넣어 다음 핸들 호출에 넘기는 방식은 완전히 폐기되어 더는 지원되지 않습니다. 이 경우 `_to_object_ref()`로 얻은 객체 참조를 대신 담습니다. 최상위 인자나 키워드 인자로 `DeploymentResponse`를 넘기는 방식은 계속 지원됩니다.

예전 코드에 남아 있을 수 있는 `RayServeHandle`과 `RayServeSyncHandle`은 Ray 2.10부터 완전히 제거되었으므로 `DeploymentHandle`로 옮겨야 합니다.

## 코드보다 config 파일이 이기는 deployment 파라미터 우선순위

같은 deployment 파라미터를 여러 곳에 적으면 Serve config 파일의 값이 이기고, 그다음이 애플리케이션 코드, 마지막이 Serve 기본값입니다. 그래서 코드에는 개발용 값을 두고, 프로덕션에서는 코드를 고치지 않고 config 파일에서 값을 바꾸는 방식이 권장됩니다.

### 데코레이터와 `.options()`: 코드에서 값을 정하는 두 자리

코드에서 파라미터를 정하는 방법은 두 가지입니다. deployment를 처음 정의할 때는 `@serve.deployment` 데코레이터에 적습니다. 텍스트 ML 애플리케이션의 `Translator`에 주요 파라미터를 모두 적으면 데코레이터가 이렇게 길어집니다.

```python
# excerpt: Translator 데코레이터에 레플리카·큐·헬스 체크·종료 파라미터를 모두 적은 모습
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
```

이미 정의된 deployment의 값을 바꿀 때는 `.options()`를 씁니다. 정의는 그대로 재사용하고 파라미터만 런타임에 정할 수 있습니다.

```python
# excerpt: 정의된 Translator의 자원만 바꿔 바인딩
translator_app = Translator.options(
    ray_actor_options={"num_cpus": 0.2, "num_gpus": 0.0}
).bind()
```

우선순위에서 데코레이터와 `.options()`는 같은 단계, 즉 "애플리케이션 코드"로 묶입니다.

### 파라미터와 기본값

다음 표는 deployment 파라미터와 기본값을 모은 것입니다.

| 파라미터 | 의미 | 기본값 |
| --- | --- | --- |
| `name` | 애플리케이션 안에서 deployment를 구분하는 이름 | 클래스·함수 이름 |
| `num_replicas` | 요청을 처리하는 레플리카 수(양의 정수 또는 `auto`) | 1 |
| `autoscaling_config` | 오토스케일링 동작 파라미터 | 없음 |
| `max_ongoing_requests` | 응답을 받기 전까지 한 레플리카에 보낼 수 있는 최대 요청 수 | 5 |
| `max_queued_requests` | 호출자(프록시 또는 DeploymentHandle)마다 대기시킬 수 있는 최대 요청 수 | -1(무제한) |
| `health_check_period_s` | 헬스 체크 호출 간격 | 10초 |
| `health_check_timeout_s` | 헬스 체크를 실패로 판정하기 전 대기 시간 | 30초 |
| `graceful_shutdown_wait_loop_s` | 종료 전 남은 작업이 없어지기를 기다리는 시간 | 2초 |
| `graceful_shutdown_timeout_s` | 강제 종료 전 정상 종료를 기다리는 시간 | 20초 |
| `ray_actor_options` | Ray 액터 데코레이터에 넘길 자원 요구량 등 | 없음 |
| `user_config` | `reconfigure` 메서드에 넘기는 설정 | 없음 |
| `logging_config` | 로그 레벨, 로그 디렉터리, JSON 로그 형식 등 | 없음 |

`num_replicas`에 양의 정수를 주면 레플리카 수가 고정되고, `auto`를 주면 기본 설정으로 오토스케일링합니다. `autoscaling_config`를 설정했다면 `num_replicas`에 숫자를 줄 수 없습니다. 컨트롤러 안의 오토스케일러가 어떤 메트릭으로 레플리카 수를 정하는지는 앞서 설명한 그대로입니다.

### 백프레셔를 만드는 두 한도

`max_ongoing_requests`와 `max_queued_requests`는 함께 동작합니다. 앞서 본 요청 수명 주기에서 모든 레플리카가 `max_ongoing_requests`만큼 요청을 처리하고 있으면 새 요청은 큐에서 기다렸습니다. 그 큐의 길이를 제한하는 값이 `max_queued_requests`입니다. HTTP 응답은 `backpressure_config`로 바꿀 수 있습니다.

기본값을 그대로 쓰면 결과가 크게 달라집니다. `max_ongoing_requests` 기본값은 Ray 2.32.0에서 100에서 5로 낮아졌습니다. 예를 들어 `Translator` 레플리카 2개에 `max_ongoing_requests=100`이면 동시에 처리 중인 요청은 최대 200개이고, 201번째 요청부터는 호출자 큐에서 기다립니다. 이 값을 생략하면 같은 계산이 레플리카당 5개, 합계 10개로 줄어듭니다. 이 한도는 호출자마다 따로 적용되므로, 프록시가 여러 개면 전체 대기 요청 수는 프록시 수에 한도를 곱한 만큼까지 늘어납니다.

### 헬스 체크와 정상 종료

기본 헬스 체크는 레플리카에 보내는 아무 일도 하지 않는(no-op) 액터 호출입니다. deployment에 `check_health` 메서드를 정의하고 비정상일 때 예외를 던지게 하면 직접 만든 검사로 바꿀 수 있습니다. 이 호출은 `health_check_period_s`마다 실행되고, `health_check_timeout_s` 안에 반환되지 않으면 실패로 판정됩니다.

레플리카를 종료할 때는 두 시간이 관여합니다. 레플리카는 남은 작업이 없어지기를 `graceful_shutdown_wait_loop_s` 동안 기다립니다. `graceful_shutdown_timeout_s`가 지나도 끝나지 않으면 강제로 종료됩니다. 번역처럼 요청 하나가 수 초씩 걸리는 모델이라면, 처리 중인 요청이 강제 종료로 끊기지 않도록 `graceful_shutdown_timeout_s`를 가장 긴 요청보다 넉넉하게 잡아야 합니다.

### 자원, 사용자 설정, 로깅

`ray_actor_options`에 쓸 수 있는 키는 `accelerator_type`, `memory`, `num_cpus`, `num_gpus`, `object_store_memory`, `resources`, `runtime_env`입니다. `user_config`는 deployment의 `reconfigure` 메서드로 전달되며, 레플리카를 재시작하지 않고 값을 갱신할 수 있습니다. 이 값은 전부 JSON으로 직렬화할 수 있어야 합니다. 로그 레벨이나 JSON 형식은 deployment별 `logging_config`로 정합니다. 여러 레플리카를 함께 배치해야 하는 경우에는 Python API에 레플리카의 gang 스케줄링 설정인 `serve.config.GangSchedulingConfig`가 있습니다.

### config 파일이 코드를 덮어쓰는 방식

Serve config 파일에서는 애플리케이션 아래의 `deployments` 목록에 deployment 이름별로 값을 적습니다. 텍스트 ML 애플리케이션의 `Translator`에 코드와 같은 값을 적은 config는 다음과 같습니다.

```yaml
applications:
- name: app1
  import_path: configure_serve:translator_app
  deployments:
  - name: Translator
    num_replicas: 2
    max_ongoing_requests: 100
    graceful_shutdown_wait_loop_s: 2.0
    graceful_shutdown_timeout_s: 20.0
    health_check_period_s: 10.0
    health_check_timeout_s: 30.0
    ray_actor_options:
      num_cpus: 0.2
      num_gpus: 0.0
```

병합은 파라미터 하나 단위로 일어납니다. 어떤 파라미터가 config와 코드에 모두 있으면 config 값을 쓰고, 코드에만 있으면 코드 값을 쓰며, 어디에도 없으면 기본값을 씁니다. 코드에 `@serve.deployment(num_replicas=2, graceful_shutdown_timeout_s=6)`만 적은 `ExampleDeployment`를 `num_replicas: 5`만 담은 config로 배포하면 아래 표와 같이 적용됩니다.

| 파라미터 | 코드 | config | 적용값 |
| --- | --- | --- | --- |
| `num_replicas` | 2 | 5 | 5 |
| `graceful_shutdown_timeout_s` | 6 | 없음 | 6 |
| `health_check_period_s` | 없음 | 없음 | 10(기본값) |

딕셔너리 값은 이 규칙에서 하나의 파라미터로 취급됩니다. config에 `ray_actor_options`가 있으면 그 딕셔너리 전체가 코드의 `ray_actor_options` 전체를 대체합니다. 코드에만 적은 하위 키는 아예 설정하지 않은 것으로 처리되어 기본값으로 돌아갑니다. `user_config`와 `autoscaling_config`도 같은 방식으로 덮어씁니다. 예를 들어 코드에는 `num_cpus: 0.2`와 `num_gpus: 0`을 두고, config의 `ray_actor_options`에는 `num_gpus: 0.0`만 적었다고 해 보겠습니다. config에서 딕셔너리 파라미터를 하나라도 건드린다면 하위 키를 빠짐없이 모두 적어야 합니다.

## serve build·deploy·status와 REST API로 운영하는 다중 애플리케이션 config

개발 중에는 `serve run`으로 실행과 수정을 반복하고, 프로덕션으로 넘어갈 때는 구조화된 config 파일을 만들어 애플리케이션의 단일 진실 공급원(single source of truth)으로 삼습니다. 이 파일 하나에 시스템 수준 설정과 애플리케이션을 함께 적고, 파일을 고쳐 다시 제출하는 방식으로 애플리케이션을 선언적으로 갱신합니다. deployment 파라미터도 모두 이 파일에서 지정할 수 있습니다.

### `serve build`로 뽑아내는 config

`serve build`는 import path를 입력으로 받아, `-o` 플래그로 지정한 파일에 Serve config를 저장합니다. 앞서 만든 Summarizer→Translator 합성 앱을 `text_ml.py`에 저장해 두었다면 아래 명령으로 config를 만듭니다.

```bash
mv serve_quickstart_composed.py text_ml.py
serve build text_ml:app -o serve_config.yaml
```

생성된 파일에는 `import_path`와 `runtime_env`, deployment별 설정 옵션이 들어갑니다. `proxy_location`, `http_options`, `grpc_options` 같은 클러스터 수준 필드도 함께 채워집니다. 애플리케이션 이름은 기본값으로 붙으며, 원하는 이름으로 바꿔도 됩니다. 이 앱은 `torch`와 `transformers` 패키지가 있어야 실행되므로, 생성된 `runtime_env`에 두 pip 패키지를 직접 추가합니다. 손으로 고친 결과는 이런 모양의 파일이 됩니다.

```yaml
proxy_location: EveryNode

http_options:
  host: 0.0.0.0
  port: 8000

applications:
- name: default
  route_prefix: /
  import_path: text_ml:app
  runtime_env:
    pip:
    - torch
    - transformers
  deployments:
  - name: Translator
    num_replicas: 1
    user_config:
      language: french
  - name: Summarizer
    num_replicas: 1
```

import path를 여러 개 넘기면 여러 애플리케이션이 한 파일에 담깁니다.

```bash
serve build image_classifier:app text_translator:app -o config.yaml
```

이렇게 만든 파일에서는 각 애플리케이션 항목이 저마다 `name`, `route_prefix`, `import_path`를 가집니다. 이름은 `app1`, `app2`처럼 자동으로 붙으므로 배포 전에 의미 있는 이름으로 바꾸는 편이 좋습니다. `host`, `port` 같은 클러스터 수준 옵션은 개별 애플리케이션 항목 안에 둘 수 없습니다. 예전 단일 애플리케이션 config를 옮겨 올 때는 이 두 값을 항목에서 빼서 `http_options` 아래로 옮깁니다.

### `serve run`으로 시험하고 `serve deploy`로 올리기

배포 전에 같은 config 파일을 `serve run config.yaml`처럼 넘겨 로컬에서 시험할 수 있습니다. Ctrl-C를 누르면 Serve와 모든 애플리케이션이 함께 종료됩니다. `serve run`에 넘기는 config 파일은 다중 애플리케이션 형식만 지원됩니다. import path를 직접 넘길 때 `serve run`은 한 번에 하나만 실행할 수 있습니다.

실제 배포는 Ray 클러스터를 먼저 띄운 뒤 `serve deploy`로 합니다. 실행 중인 상태는 `serve status`로 확인하며, 출력에는 프록시 상태, 애플리케이션 상태, deployment별 상태와 레플리카 상태가 나옵니다.

```text
# Start a local Ray cluster.
ray start --head

# Deploy the Text ML application to the local Ray cluster.
serve deploy serve_config.yaml
2022-08-16 12:51:22,043 SUCC scripts.py:180 --
Sent deploy request successfully!
* Use `serve status` to check deployments' statuses.
* Use `serve config` to see the running app's config.

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

애플리케이션을 바꾸려면 config 파일을 수정하고 `serve deploy`를 다시 실행합니다. `applications` 아래 항목을 추가하거나 삭제하거나 수정해도 클러스터의 다른 애플리케이션에는 영향이 없습니다. config를 다시 제출했을 때 일어나는 in-place 업데이트(in-place update)는 단일 애플리케이션일 때와 똑같이 동작합니다. Kubernetes에서 이 업데이트를 실제로 거는 과정은 RayService를 다룰 때 이어집니다.

### 대시보드의 Serve 페이지

CLI보다 자세히 들여다보려면 `http://localhost:8265/#/serve`의 Serve 대시보드를 엽니다. 첫 화면에는 클러스터에 배포된 애플리케이션이 모두 나열됩니다.

![Ray 대시보드의 Serve 애플리케이션 목록 화면](https://raw.githubusercontent.com/ray-project/images/master/docs/serve/multi-app/applications-dashboard.png)

애플리케이션 하나를 고르면 그 아래 deployment 목록이 보입니다.

![애플리케이션별 deployment 목록 화면](https://raw.githubusercontent.com/ray-project/images/master/docs/serve/multi-app/deployments-dashboard.png)

deployment를 다시 고르면 그 deployment의 레플리카 목록까지 확인할 수 있습니다.

![deployment별 레플리카 목록 화면](https://raw.githubusercontent.com/ray-project/images/master/docs/serve/multi-app/replica-dashboard.png)

대시보드는 로컬 테스트에서도, 원격 프로덕션 클러스터에서도 쓸 수 있습니다. 다만 프로덕션에서는 로그와 메트릭을 사용 중인 관측 플랫폼으로 내보내는 방식이 권장됩니다.

### 애플리케이션을 나누는 기준과 `get_app_handle`

한 요청을 처리하는 데 필요한 모델과 로직이 한 저장소에 있고 함께 업그레이드된다면, 하나의 애플리케이션에 모아 두는 쪽이 맞습니다. 반대로 서로 통신하지만 저장소가 다른 모델 그룹이 있거나, 통신하지 않는 독립 모델을 하드웨어 활용도를 높이려고 한 클러스터에 함께 올린다면 애플리케이션을 나눕니다. 애플리케이션이 업그레이드 단위이므로, 나눠 두면 각 모델을 서로 다른 엔드포인트 뒤에 두고 독립적으로 추가·삭제·업그레이드할 수 있습니다.

나눈 애플리케이션끼리는 HTTP를 거치지 않고 직접 호출할 수 있습니다. `serve.get_app_handle`로 클러스터에서 실행 중인 애플리케이션의 핸들을 얻으면 됩니다. 이 예제는 텍스트 ML 앱과 별개로, 앞의 다중 애플리케이션 `serve build` 명령에 넘긴 두 파일을 씁니다. 번역 앱 `app2`가 될 `text_translator.py`는 영어를 독일어로 옮기는 `Translator` 하나로 이뤄집니다.

```python
import starlette.requests

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

이미지 분류 앱 `image_classifier.py`에서는 `__call__`이 요청의 `should_translate` 플래그를 보고 `app2`의 `translate`를 직접 부릅니다.

```python
import requests
import starlette.requests
from io import BytesIO
from PIL import Image
from transformers import pipeline

from ray import serve
from ray.serve.handle import DeploymentHandle


@serve.deployment
def downloader(image_url: str):
    image_bytes = requests.get(image_url).content
    return Image.open(BytesIO(image_bytes)).convert("RGB")


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

### REST API로 하는 배포·조회·삭제

CLI와 같은 작업은 HTTP로도 할 수 있습니다. Serve REST API는 Ray 대시보드와 같은 포트로 노출되며, 이 포트의 기본값은 `8265`이고 `ray start`의 `--dashboard-port` 인자로 바꿀 수 있습니다. `/api/serve/applications/` 경로에서는 메서드에 따라 하는 일이 갈립니다.

| 메서드 | 동작 |
| --- | --- |
| `PUT` | 애플리케이션 목록을 선언적으로 배포합니다. Serve가 실행 중이면 새 config에 없는 애플리케이션을 모두 제거하고, 실행 중이 아니면 Serve를 시작합니다. |
| `GET` | 클러스터 수준 정보(컨트롤러, 프록시, HTTP·gRPC 옵션)와 모든 애플리케이션·deployment·레플리카의 상세 상태를 반환합니다. |
| `DELETE` | Serve와 클러스터의 모든 애플리케이션을 종료합니다. Serve가 실행 중이 아니면 아무 효과가 없습니다. |

PUT 요청의 JSON 본문은 다중 애플리케이션 config 스키마를 따릅니다. 텍스트 ML 앱을 `text_app`이라는 이름으로 올리는 요청은 이렇게 생겼습니다.

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

배포 파이프라인에서 GET 응답을 확인 단계로 쓸 때는 `applications` 아래 각 deployment의 `deployment_config`와 `replicas` 상태를 보면, config 파일에 적은 값이 실제로 적용되었는지 대조할 수 있습니다.

## KubeRay RayService로 운영하는 Serve: in-place 업데이트와 무중단 클러스터 업그레이드

프로덕션에서 Ray Serve를 실행할 때 권장하는 방식은 Kubernetes 위에서 KubeRay의 RayService 커스텀 리소스(custom resource)를 쓰는 것입니다. 이 리소스가 헬스 체크, 상태 보고, 장애 복구, 업그레이드를 자동으로 처리합니다. Kubernetes를 쓰지 않는 환경이라면 앞서 본 Serve CLI로 Ray 클러스터에 직접 배포하면 됩니다.

### Pod 하나가 Ray 노드 하나인 KubeRay 오퍼레이터

KubeRay 오퍼레이터는 Kubernetes에서 Ray를 실행하는 권장 방법입니다. 오퍼레이터는 Ray 노드 하나를 Kubernetes Pod 하나로 띄우므로, Ray 클러스터 하나는 head Pod 하나와 worker Pod 여러 개로 구성됩니다.

![head Pod와 worker Pod로 구성된 Ray on Kubernetes 개요도](https://docs.ray.io/en/latest/_images/ray_on_kubernetes.png)

그림에서는 head Pod와 worker Pod들이 함께 Ray 클러스터 하나를 이루는 구조를 보면 됩니다. KubeRay는 GPU를 포함한 이기종 노드를 지원하고, Ray 버전이 서로 다른 클러스터 여러 개를 한 Kubernetes 클러스터에서 함께 운영할 수 있습니다. 오퍼레이터가 도입하는 CRD는 RayCluster, RayJob, RayService 세 가지이며, 쓰임새에 따라 골라 씁니다. Serve 애플리케이션에는 RayService를 씁니다. RayService CR 하나에 멀티 노드 Ray 클러스터와 그 위에서 도는 Serve 애플리케이션이 함께 들어가므로, 배포와 업그레이드, 상태 조회를 모두 표준 `kubectl` 명령으로 처리합니다. 고가용성 구성도 지원합니다.

### `serveConfigV2`와 `rayClusterConfig` 두 필드로 정의하는 매니페스트

RayService CR의 `spec` 아래에는 두 필드를 채웁니다. `serveConfigV2`에는 Ray Serve가 애플리케이션을 배포할 때 쓰는 config를 넣습니다. 앞서 `serve build`로 뽑은 Text ML 앱의 config를 이 필드에 그대로 붙여 넣으면 됩니다. `rayClusterConfig`에는 RayCluster CR YAML의 `spec` 필드 내용을 넣습니다. Serve 쪽 설정과 클러스터 쪽 설정이 매니페스트 하나 안에서 이렇게 갈리며, 아래에서 다룰 두 갈래의 업데이트도 이 구분을 따릅니다.

`serveConfigV2` 안의 각 애플리케이션 항목은 앞서 본 다중 애플리케이션 config와 같은 키로 적습니다. 키 구성을 보기 위해 Text ML 앱이 아닌 별도의 KubeRay 샘플 매니페스트를 살펴보면, 과일 가게 앱 `fruit_app`과 계산기 앱 `math_app`이 들어 있고 두 앱 모두 `import_path`, `route_prefix`(`/fruit`, `/calc`), `runtime_env.working_dir`, `deployments`를 가집니다. 이 과일·계산기 샘플의 `working_dir`는 test_dag 저장소를 가리키며, `fruit_app`의 애플리케이션은 이 저장소의 fruit.py가 `deployment_graph` 변수로 정의합니다. RayService는 런타임에 이 저장소를 내려받아 애플리케이션을 시작합니다. 의존성이 커서 내려받는 데 오래 걸린다면 이미지의 Dockerfile에 미리 넣어 두는 것이 좋습니다. 그러면 Pod가 시작되자마자 의존성을 쓸 수 있어 애플리케이션 안정성이 높아집니다.

RayService를 만들면 KubeRay 컨트롤러는 먼저 주어진 설정으로 Ray 클러스터를 띄웁니다. 클러스터가 실행되면 앞서 본 Serve REST API로 Serve 애플리케이션을 배포하고, 트래픽 라우팅용 Kubernetes Service도 만듭니다. Text ML 예제의 샘플 CR을 내려받아 적용하면 head Pod와 worker Pod, 그리고 서비스들이 생깁니다.

```text
$ curl -o ray-service.text-ml.yaml https://raw.githubusercontent.com/ray-project/kuberay/2ba0dd7bea387ac9df3681666bab3d622e89846c/ray-operator/config/samples/ray-service.text-ml.yaml

$ kubectl apply -f ray-service.text-ml.yaml

$ kubectl get rayservices
NAME SERVICE STATUS NUM SERVE ENDPOINTS
rayservice-sample Running 1

$ kubectl get services
NAME TYPE CLUSTER-IP EXTERNAL-IP PORT(S) AGE
rayservice-sample-head-svc ClusterIP None <none> 10001/TCP,8265/TCP,6379/TCP,8080/TCP,8000/TCP XXs
rayservice-sample-raycluster-7wlx2-head-svc ClusterIP None <none> 10001/TCP,8265/TCP,6379/TCP,8080/TCP,8000/TCP XXs
rayservice-sample-serve-svc ClusterIP 192.168.145.219 <none> 8000/TCP XXs
```

샘플 CR의 `numCpus` 값은 시연용으로 아주 낮게 잡혀 있습니다. 프로덕션에서는 Serve 애플리케이션에 자원을 더 줘야 합니다.

### 쿼리를 받는 `serve-svc`와 `/-/routes` 헬스 체크

출력에 보이는 서비스 가운데 Serve 애플리케이션에 쿼리를 보낼 서비스는 `rayservice-sample-serve-svc`입니다. 이 서비스는 Serve 레플리카가 있는 모든 worker로 트래픽을 분산합니다. 클러스터 안에서는 이 서비스를 바로 호출하면 됩니다. 노트북에서 접근하려면 Kubernetes ingress를 구성하거나, 다음처럼 포트를 포워딩합니다.

```text
$ kubectl port-forward service/rayservice-sample-serve-svc 8000
$ curl -X POST -H "Content-Type: application/json" localhost:8000/summarize_translate -d '"It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief"'
C'était le meilleur des temps, c'était le pire des temps,
```

Summarizer가 요약한 문장을 Translator가 프랑스어로 옮긴 결과가 그대로 돌아옵니다. 로컬에서 `serve run`으로 받던 응답과 같은 앱입니다.

외부 로드밸런서로 노출할 때는 ingress를 둡니다. Ray Serve는 노드마다 HTTP 프록시를 실행하므로 `/-/routes`를 노드 헬스 체크 엔드포인트로 지정할 수 있습니다. ingress가 바라볼 트래픽 포트는 앞서 Serve config의 `http_options`에서 정한 프록시 포트 그대로입니다.

### `kubectl describe`에 모이는 상태와 Pod 안의 Serve 로그

RayService가 실행되는 동안 KubeRay 컨트롤러는 RayService를 계속 감시하면서 상태 변화를 CR에 기록합니다. `kubectl describe`를 실행하면 클러스터 상태, 헬스 체크 실패나 재시작 같은 이벤트, 그리고 `serve status`가 보고하는 애플리케이션 수준 상태가 한 화면에 나옵니다. 아래는 출력 가운데 Text ML 앱 부분만 옮긴 것입니다.

```text
$ kubectl describe rayservice rayservice-sample
    Application Statuses:
      text_ml_app:
        Serve Deployment Statuses:
          Summarizer:
            Status:  HEALTHY
          Translator:
            Status:  HEALTHY
        Status:  RUNNING
Events:
  Type    Reason   Age                     From                   Message
  Normal  Running  2m15s (x29791 over 16h) rayservice-controller  The Serve applicaton is now running and healthy.
```

대시보드는 앞서 REST API와 함께 다룬 대시보드 포트로 열립니다. 그 포트로 head 서비스 `rayservice-sample-head-svc`를 `kubectl port-forward`한 뒤 로컬 브라우저에서 Serve 페이지를 엽니다.

![RayService가 배포한 애플리케이션 상태를 보여 주는 Ray 대시보드 Serve 페이지](https://docs.ray.io/en/latest/_images/dashboard_serve.png)

CLI 없이도 RayService가 올린 애플리케이션과 deployment의 상태를 이 페이지에서 확인할 수 있습니다. 대시보드에 붙일 Prometheus와 Grafana도 따로 구성할 수 있습니다.

배포가 실패했을 때는 먼저 `kubectl logs -f <kuberay-operator-pod-name>`로 KubeRay 오퍼레이터 로그를 봅니다. 이 로그에는 Serve 애플리케이션 배포 이벤트와 헬스 체크 기록이 남습니다. 더 구체적인 원인은 head Pod와 worker Pod 모두의 `/tmp/ray/session_latest/logs/serve/` 아래에 있는 컨트롤러 로그와 deployment 로그에서 찾습니다. 이 로그에는 deployment 실패 사유와 오토스케일링 이벤트가 기록됩니다.

### Serve와 Kubernetes 두 층에서 켜는 오토스케일링

레플리카 수는 Serve config의 autoscaling 필드로 조정하고, 레플리카를 올릴 Pod는 KubeRay가 늘립니다. KubeRay 클러스터에서 오토스케일링을 쓰려면 `enableInTreeAutoscaling`을 True로 설정해야 합니다. 이 설정을 켜면 KubeRay 오퍼레이터가 워크로드에 맞춰 Pod를 추가하고 제거합니다. 대부분의 경우 Kubernetes 클러스터 오토스케일링도 함께 켜서 클러스터 자원을 충분히 활용하라고 권장합니다. GKE에서는 Autopilot 클러스터를, EKS에서는 Cluster Autoscaler를 쓸 수 있습니다.

### 같은 클러스터에서 끝나는 `serveConfigV2` 변경, 새 클러스터로 넘어가는 `rayClusterConfig` 변경

RayService를 갱신할 때는 매니페스트를 고치고 `kubectl apply`를 다시 실행합니다. 다만 어느 필드를 고쳤는지에 따라 클러스터에서 일어나는 일이 크게 달라집니다.

| 변경한 필드 | 반영 방식 | 필요한 자원 |
| --- | --- | --- |
| `serveConfigV2` | 새 RayCluster를 만들지 않고 기존 클러스터에 새 설정을 다시 적용하는 in-place 업데이트입니다. | 기존 클러스터 그대로입니다. |
| `spec.rayClusterConfig` | 새 RayCluster를 띄우고, 준비되면 트래픽을 옮긴 뒤 기존 클러스터를 삭제하는 무중단 업그레이드입니다. | 두 Ray 클러스터를 동시에 스케줄할 만큼 큰 Kubernetes 클러스터가 필요합니다. |

레플리카 수·오토스케일링 파라미터·`user_config`는 `serveConfigV2`에, 컨테이너 이미지는 `rayClusterConfig`에 속합니다.

Text ML 앱으로 in-place 업데이트를 해 보면, `serveConfigV2` 안의 Translator 항목에서 `user_config`를 독일어로 바꾸고 매니페스트를 다시 적용하는 것으로 끝납니다. 적용 직후 deployment는 `UPDATING` 상태가 됩니다.

```yaml
# excerpt: ray-service.text-ml.yaml의 serveConfigV2 중 Translator 항목(수정 후)
serveConfigV2: |
  applications:
    - name: text_ml_app
      route_prefix: /summarize_translate
      deployments:
        - name: Translator
          num_replicas: 1
          user_config:
            language: german
```

```text
$ kubectl apply -f ray-service.text-ml.yaml

$ kubectl describe rayservice rayservice-sample
...
Serve Deployment Statuses:
  text_ml_app_Translator:
    Health Last Update Time: 2023-09-07T18:21:36Z
    Last Update Time: 2023-09-07T18:21:36Z
    Status: UPDATING
...

$ curl -X POST -H "Content-Type: application/json" localhost:8000/summarize_translate -d '"It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief"'
es war die beste Zeit, es war die schlimmste Zeit,
```

앞서 Translator에 추가한 `reconfigure` 경로로 레플리카 재시작 없이 반영되므로, 위 출력처럼 같은 클러스터와 같은 서비스 주소에서 번역 결과만 독일어로 바뀝니다.

`rayClusterConfig`를 고치면 RayService가 새 RayCluster를 임시로 만들고 준비될 때까지 기다립니다. 새 클러스터가 준비되면 RayService가 관리하는 head 서비스 `rayservice-sample-head-svc`의 selector를 새 클러스터 쪽으로 바꿔 트래픽을 전환하고, KubeRay가 기존 클러스터를 삭제합니다. 이 흐름은 기본 업그레이드 전략인 `NewCluster`의 동작입니다. 예를 들어 `rayVersion`은 Autoscaler 사이드카 이미지를 결정하는 필드이므로, 이 값을 바꾸면 무중단 업그레이드가 시작됩니다. 전환이 끝나면 요청은 같은 `serve-svc` 주소로 계속 들어옵니다. 기본 `NewCluster` 전략은 클러스터 전체를 한 벌 더 띄우므로, 대규모 배포처럼 자원을 두 벌 확보하기 어려운 환경에서는 `NewClusterWithIncrementalUpgrade` 전략을 고려합니다. 이 전략은 업그레이드 중에 자원을 덜 씁니다.

예외도 있습니다. `replicas`, `minReplicas`, `maxReplicas`, `scaleStrategy.workersToDelete`는 Ray 오토스케일러가 관리하는 필드라서 바꿔도 무중단 업그레이드가 일어나지 않습니다. KubeRay는 이 변경을 RayService에서 RayCluster로 전파하지 않으므로 실제로는 아무 일도 일어나지 않습니다. worker 그룹의 `replicas`를 1에서 2로 올려 적용하면 RayService CR에는 2가 기록되지만 RayCluster CR은 1로 남습니다. 매니페스트에 적힌 값과 실제 클러스터 상태가 어긋나 보일 수 있으니, 이런 값을 바꾼 뒤에는 RayCluster CR까지 확인해야 합니다.

### 헤드 노드 장애에 대비한 GCS 장애 허용

앞서 본 GCS 장애 허용을 켜 두는 것, 즉 head 노드 장애 허용 구성이 프로덕션 권장 사례입니다.

## Ray Serve가 맞는 경우와 TorchServe·MLflow·BentoML이 나은 경우

여러 모델과 비즈니스 로직을 하나의 Python 애플리케이션으로 엮고 단계마다 따로 스케일해야 한다면 Ray Serve가 맞습니다. 반대로 모델 하나를 더 빠르게 돌리는 최적화나 모델 수명 주기 관리는 Serve의 영역이 아니므로, 그 부분은 다른 도구가 맡아야 합니다.

### 모델 조합에 집중한 설계와 비워 둔 영역

BentoML, Comet.ml, MLflow 같은 도구는 상당수가 모델을 하나씩 독립적으로 서빙하고 스케일하는 데 집중합니다. Serve는 이와 달리 프레임워크에 묶이지 않고 모델 조합에 집중하며, 모델 패키징이나 레지스트리 형식을 가리지 않습니다. 프레임워크 전용 서빙 솔루션은 커스텀 어댑터나 복잡한 설정을 요구합니다. Serve는 임의의 Python 코드를 그대로 실행하므로 YAML 설정이나 커스텀 직렬화 형식과 씨름하지 않고 빠르게 반복 개발할 수 있습니다. Text ML 앱처럼 Summarizer와 Translator를 DeploymentHandle로 잇고, 각각을 config에서 따로 튜닝하는 구성이 Serve가 가장 잘 맞는 형태입니다.

Serve가 제공하는 것은 모델 서빙과, 그 위에 자체 ML 플랫폼을 쌓기 위한 기본 요소입니다.

| 대안 | Ray Serve와 다른 점 | 함께 쓰는 방법 |
| --- | --- | --- |
| TFServing·TorchServe·ONNXRuntime | Serve는 이 프레임워크 전용 솔루션들과 달리 모델별 최적화를 하지 않습니다. | PyTorch JIT나 ONNXRuntime으로 직접 컴파일한 모델을 Serve에서 실행합니다. |
| AWS SageMaker·Azure ML·Google Vertex AI | Serve에는 모델 수명 주기 관리와 성능 시각화 기능이 없습니다. | Ray 클러스터 런처로 주요 퍼블릭 클라우드, K8s, 베어메탈·온프레미스 장비에 Serve를 배포합니다. |
| BentoML·Comet.ml·MLflow | 이 도구들은 상당수가 모델 단위의 독립 서빙에 집중합니다. | 이 도구들의 레지스트리를 모델 저장소로 쓰고, 조합과 스케일링은 Serve가 맡습니다. |

MLflow는 레지스트리로는 잘 맞지만, 내장 REST 서버는 고동시성 GPU 워크로드에 최적화되어 있지 않습니다. 그래서 LLM이나 diffusion 모델에는 Hugging Face Hub나 Weights & Biases를 레지스트리로 쓰고, GPU 배칭·오토스케일링·스케줄링은 Ray Serve에 맡기기를 권장합니다.

### LLM 서빙에서 쓰는 deployment 단위 패턴

LLM에서는 `LLMServer`가 추론 엔진 인스턴스 하나를 관리하는 Serve deployment입니다. 이 deployment의 레플리카는 세 가지 모드로 동작합니다. 각자 독립적으로 수평 확장하는 격리 모드, 한 deployment 안에서 협력하는 데이터 병렬 어텐션, 서로 다른 deployment가 협력하는 prefill-decode 분리입니다. 데이터 병렬 어텐션은 sparse MoE 모델을 높은 처리량으로 서빙할 때 씁니다. prefill-decode 분리는 두 단계를 나눠 각각 독립적으로 스케일합니다. 커스텀 요청 라우팅은 캐시 지역성과 세션 친화성을 살리는 데 씁니다. 이 패턴들은 모두 Text ML 앱에서 쓴 "단계를 deployment로 나눠 따로 스케일한다"는 원리를 LLM에 적용한 것입니다.

### 배포 환경과 실패 지점

클러스터를 직접 관리하고 싶지 않다면 Ray 제작자들이 만든 관리형 플랫폼 Anyscale도 선택지입니다. Anyscale은 EKS, GKE, AKS 또는 자체 호스팅 Kubernetes 같은 기존 인프라에 Ray 클러스터를 배포하는 경로를 제공합니다.

앞서 본 대로 KubeRay 없이는 클러스터 장애에서 복구하지 못하므로, Kubernetes가 아닌 환경에서는 클러스터 수준의 복구 수단을 따로 마련해야 합니다.

### 선택 기준

이 글의 판단은 이렇습니다. 단일 모델을 단일 프레임워크로 서빙하면서 지연 시간 최적화가 핵심이라면 TorchServe 같은 전용 서버가 더 직접적인 답입니다. 다만 그 경우에도 컴파일한 모델을 Serve deployment 안에 넣으면 전처리나 후처리 로직을 같은 애플리케이션에서 함께 스케일할 수 있습니다. 실험 추적, 모델 버전 승인, 성능 대시보드가 필요한 팀은 MLflow 같은 플랫폼을 수명 주기 관리에 쓰고, Serve는 그 아래의 서빙 계층으로 두는 조합이 현실적입니다. 파이프라인 단계가 셋 이상이고 단계마다 필요한 자원(CPU 전처리, GPU 추론)이 다르며 트래픽이 출렁인다면, Serve의 장점이 가장 크게 드러납니다.

- - -

### Sources
* [Ray Serve: Scalable and Programmable Serving — Ray 2.60.0](https://docs.ray.io/en/latest/serve/index.html)
* [Key Concepts — Ray 2.60.0](https://docs.ray.io/en/latest/serve/key-concepts.html)
* [Deploy Compositions of Models — Ray 2.60.0](https://docs.ray.io/en/latest/serve/model_composition.html)
* [Getting Started — Ray 2.60.0](https://docs.ray.io/en/latest/serve/getting_started.html)
* [Architecture — Ray 2.60.0](https://docs.ray.io/en/latest/serve/architecture.html)
* [Deploy on Kubernetes — Ray 2.60.0](https://docs.ray.io/en/latest/serve/production-guide/kubernetes.html)
* [Ray Serve API — Ray 2.60.0](https://docs.ray.io/en/latest/serve/api/)
* [Architecture overview — Ray 2.60.0](https://docs.ray.io/en/latest/serve/llm/architecture/overview.html)
* [Develop and Deploy an ML Application — Ray 2.60.0](https://docs.ray.io/en/latest/serve/develop-and-deploy.html)
* [Model Registry Integration — Ray 2.60.0](https://docs.ray.io/en/latest/serve/model-registries.html)
* [Production Guide — Ray 2.60.0](https://docs.ray.io/en/latest/serve/production-guide/index.html)
* [Configure Ray Serve deployments — Ray 2.60.0](https://docs.ray.io/en/latest/serve/configure-serve-deployment.html)
* [Deploy Multiple Applications — Ray 2.60.0](https://docs.ray.io/en/latest/serve/multi-app.html)
* [Ray Serve API — Ray 2.60.0](https://docs.ray.io/en/latest/serve/api/index.html)
* [Ray on Kubernetes — Ray 2.60.0](https://docs.ray.io/en/latest/cluster/kubernetes/index.html)
* [Deploy Ray Serve Applications — Ray 2.60.0](https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/rayservice.html)
* [Develop and Deploy an ML Application — Ray 3.0.0.dev0](https://docs.ray.io/en/master/serve/develop-and-deploy.html)
