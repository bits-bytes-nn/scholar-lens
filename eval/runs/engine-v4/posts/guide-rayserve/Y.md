---
layout: post
title: "Ray Serve in Practice: Building, Composing, Operating ML Inference"
date: 2026-11-03 05:29:07
description: "Ray Serve를 활용해 온라인 ML 추론 API를 만드는 방법을 다룹니다. 디플로이먼트와 모델 조합, HTTP 처리, 오토스케일링, 설정 파일 배포, KubeRay 운영, 장애 복구, 모니터링까지 구조와 설정을 정리합니다."
categories: ["Tech Guides", "Language-Models"]
tags: ["Ray-Serve", "Model-Composition", "Autoscaling", "KubeRay-RayService", "Prometheus-Metrics", "Model-Serving"]
cover: /assets/images/language-models.jpg
use_math: true
---
## Ray Serve 소개와 설치

Ray Serve는 온라인 추론 API를 만드는 확장형 모델 서빙 라이브러리입니다. 많은 서빙 도구는 모델을 미리 정한 엔드포인트 뒤에 감싸는 "텐서 입력, 텐서 출력" 방식에 집중합니다. 하지만 실제 서비스는 모델 하나로 끝나지 않고, 전처리, 여러 모델, 데이터베이스 조회 같은 비즈니스 로직을 함께 엮어야 합니다. Ray Serve는 이 과정 전체를 하나의 파이썬 프로그램으로 작성해 배포하도록 설계됐고, 특히 모델 조합(model composition)과 다중 모델 서빙에 강합니다.

Ray Serve는 특정 프레임워크에 묶이지 않습니다. PyTorch, TensorFlow, Keras로 만든 딥러닝 모델부터 Scikit-Learn 모델, 임의의 파이썬 비즈니스 로직까지 같은 도구로 서빙합니다. 그 대가로 TFServing이나 TorchServe 같은 프레임워크 전용 도구와 달리 모델별 최적화는 하지 않습니다. 따라서 PyTorch JIT나 ONNXRuntime으로 직접 최적화한 모델을 올려서 씁니다. 또한 Ray 위에서 동작하므로 노트북에서 시작해 여러 머신, Kubernetes, 주요 클라우드, 온프레미스 환경까지 코드를 거의 바꾸지 않고 확장합니다. 분할 GPU 같은 유연한 스케줄링 덕분에 여러 모델이 자원을 나눠 써서 비용도 줄어듭니다. 모델 수명 주기 관리나 성능 시각화 같은 ML 플랫폼 기능은 없습니다. Ray Serve는 서빙과, 그 위에 플랫폼을 쌓는 기본 요소 제공에 집중합니다.

Ray Serve와 의존성은 다음 명령으로 설치합니다.

```bash
pip install "ray[serve]"
```

이 가이드는 Hugging Face의 t5-small 번역 모델을 예제로 사용합니다. 예제를 따라 하려면 관련 패키지도 함께 설치합니다.

```bash
pip install "ray[serve]" transformers requests torch
```

## 서빙 전 모델 준비하기

Ray Serve로 무언가를 서빙하려면 먼저 서빙할 코드가 혼자서도 제대로 동작해야 합니다. 서빙 계층에서 문제가 생겼을 때 원인이 모델 코드인지 서빙 설정인지 가르려면, 순수 파이썬으로 모델을 실행해 보고 결과를 확인해 두는 편이 좋습니다. 예제는 영어 문장을 프랑스어로 번역하는 `Translator` 클래스입니다. Ray Serve와 관련된 코드는 아직 한 줄도 들어 있지 않습니다.

### 번역 모델 코드

다음 코드를 `model.py`로 저장합니다.

```python
# File name: model.py
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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

translator = Translator()

translation = translator.translate("Hello world!")
print(translation)
```

`__init__`은 Hugging Face 허브에서 t5-small 토크나이저와 모델을 불러옵니다. t5-small은 텍스트를 입력받아 텍스트를 내놓는 모델이며, 입력 앞에 작업 설명을 접두어로 붙이면 그 작업을 수행합니다. 번역, 요약 같은 작업마다 모델을 따로 두지 않고 접두어로 작업을 고르는 방식입니다.

`translate`는 입력 문장 앞에 `"translate English to French: "`를 붙여 토큰화하고, `model.generate()`로 출력 토큰을 생성한 뒤 다시 문자열로 디코딩합니다. `generate()`에 넘긴 인자는 디코딩 방식을 정합니다. `num_beams=4`는 후보 시퀀스 네 개를 동시에 추적하는 빔 탐색을 쓰고, `early_stopping=True`는 완성된 후보가 충분히 나오면 빔 탐색을 일찍 멈추며, `max_length=300`은 생성 길이 상한입니다. 디코딩 단계의 `skip_special_tokens=True`는 패딩이나 종료 토큰 같은 특수 토큰을 결과 문자열에서 빼서 번역문만 남깁니다.

### 로컬에서 실행해 보기

스크립트를 그대로 실행하면 `"Hello world!"`를 프랑스어로 번역해 출력합니다.

```bash
$ python model.py

Bonjour Monde!
```

처음 실행할 때는 t5-small 가중치를 내려받으므로 시간이 조금 걸립니다. 이 단계에서 번역 결과가 나오면 `transformers`와 `torch` 설치, 모델 다운로드까지 모두 정상입니다.

### 서빙 관점에서 코드 읽기

이 코드는 서빙으로 옮기기 좋은 형태입니다. 비용이 큰 작업과 요청마다 반복되는 작업이 서로 다른 메서드에 나뉘어 있기 때문입니다. 모델 가중치 로딩은 `__init__`에서 한 번만 일어나고, 추론은 `translate`가 호출될 때마다 일어납니다. Ray Serve에서는 이 클래스가 디플로이먼트(deployment)가 되고, 실행 중에는 클래스의 복사본인 레플리카(replica)가 각자 별도 프로세스에서 뜹니다. 따라서 생성자는 레플리카가 시작될 때 한 번 실행되고, 이후 들어오는 요청은 이미 메모리에 올라간 모델을 재사용합니다. 반대로 요청을 처리하는 메서드 안에서 `from_pretrained`를 호출하면 요청마다 모델을 새로 불러오게 되어 지연 시간이 크게 늘어납니다. 자기 모델을 옮길 때도 로딩은 생성자에, 추론은 메서드에 두는 구조를 먼저 갖춰 두기를 권합니다.

서빙으로 옮길 때 각 부분이 어떻게 바뀌는지 정리하면 다음과 같습니다.

| 코드 부분 | 서빙 시 처리 |
| --- | --- |
| `__init__`의 모델 로딩 | 그대로 유지하며, 레플리카가 시작될 때 실행됩니다. |
| `translate` 추론 로직 | 그대로 유지하며, 요청 처리 경로에서 호출됩니다. |
| HTTP 진입점 | 새로 추가하며, HTTP 요청의 JSON 본문을 읽어 `translate`에 넘깁니다. |
| 파일 끝의 인스턴스 생성과 `print` | 제거하며, 애플리케이션을 정의하는 코드로 바뀝니다. |

`translate`는 파이썬 문자열을 받고 문자열을 돌려줄 뿐 HTTP를 전혀 모릅니다. 그래서 HTTP 요청에서 텍스트를 꺼내 이 메서드로 넘겨 주는 진입점이 따로 필요합니다. 파일 끝의 세 줄은 로컬 확인용 실행 코드라서 서빙 코드에서는 빠집니다. 나머지 모델 로딩과 추론 로직은 한 글자도 바꾸지 않고 그대로 가져갑니다.

t5-small은 예제일 뿐입니다. Ray Serve는 임의의 파이썬 프레임워크로 만든 모델을 다루므로, scikit-learn, PyTorch, TensorFlow 모델도 같은 구조로 준비하면 됩니다. 생성자에서 모델을 불러오고 메서드에서 추론하도록 클래스를 정리한 뒤, 이 예제처럼 로컬에서 먼저 결과를 확인합니다.

## 핵심 개념: 디플로이먼트, 레플리카, 애플리케이션

Ray Serve 코드를 읽고 쓰려면 디플로이먼트, 레플리카, 애플리케이션(application)의 세 추상화를 먼저 구분해야 합니다. 디플로이먼트는 요청을 처리하는 코드의 정의이고, 레플리카는 그 정의를 실제로 실행하는 프로세스이며, 애플리케이션은 하나 이상의 디플로이먼트를 묶어 함께 배포하고 업그레이드하는 단위입니다.

### 디플로이먼트: 클래스에 `@serve.deployment` 붙이기

디플로이먼트는 Ray Serve의 중심 개념입니다. 비즈니스 로직이나 ML 모델을 담아 들어오는 요청을 처리하고, Ray 클러스터 전체로 확장해 실행됩니다. 디플로이먼트를 정의하려면 파이썬 클래스에 `@serve.deployment` 데코레이터를 붙입니다. 간단한 경우에는 함수에도 붙일 수 있습니다.

`Translator`를 디플로이먼트로 바꾼 코드는 다음과 같습니다.

```python
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

모델 코드는 그대로이고, 데코레이터와 `__call__` 메서드만 추가됐습니다. 데코레이터는 평범한 파이썬 클래스 `Translator`를 Ray Serve의 `Deployment` 객체로 바꿉니다. 이 차이가 중요합니다. 데코레이터를 붙인 뒤의 `Translator`라는 이름은 인스턴스를 찍어 내는 클래스가 아니라, 사용자가 작성한 클래스 하나와 그 실행 방식을 함께 담은 정의를 가리킵니다. 그래서 원래 코드처럼 `Translator()`로 인스턴스를 바로 만들지 않고, 파일 끝에서 `Translator.bind()`를 호출합니다. 인스턴스는 이 스크립트를 실행한 프로세스가 아니라 레플리카마다 따로 떠 있는 프로세스 안에 만들어져야 하므로, 코드에서는 "어떤 생성자 인자로 만들지"만 전달하고 생성은 Ray Serve에 맡기는 구조입니다.

데코레이터 인자는 디플로이먼트마다 따로 정합니다. 앞의 코드는 자주 쓰는 `num_replicas`와 `ray_actor_options`를 지정했지만, 모든 인자는 선택 사항이라서 생략해도 됩니다.

```python
@serve.deployment
class Translator:
    ...
```

인자를 디플로이먼트 단위로 받는 데는 이유가 있습니다. 하나의 애플리케이션 안에서도 단계마다 계산 비용과 필요한 자원이 다릅니다. 가벼운 전처리 디플로이먼트는 CPU 조금으로 충분하고, 무거운 모델 디플로이먼트는 GPU와 더 많은 복사본이 필요합니다. 설정을 디플로이먼트마다 두므로 각 단계를 따로 확장하고 구성할 수 있습니다.

### 레플리카: 디플로이먼트를 실행하는 복사본

디플로이먼트는 정의일 뿐이고, 실행 시점에는 여러 레플리카로 이루어집니다. 레플리카는 클래스나 함수의 개별 복사본이며, 각각 별도의 Ray 액터(프로세스)에서 시작됩니다. 프로세스가 다르므로 레플리카마다 모델과 자원 사본이 따로 있습니다. `Translator` 레플리카가 둘이면 t5-small 가중치도 메모리에 두 벌 올라가고, `__init__`도 레플리카마다 한 번씩 실행됩니다.

`num_replicas`는 Ray에서 실행할 디플로이먼트 프로세스의 개수를 정하는 정수입니다. 요청은 이 레플리카들에 부하 분산되므로, 레플리카를 늘리면 디플로이먼트가 수평으로 확장됩니다. 레플리카 수는 들어오는 요청 부하에 맞춰 늘리거나 줄일 수 있고, 오토스케일링(autoscaling)으로 자동 조정할 수도 있습니다.

트레이드오프도 분명합니다. 레플리카를 늘리면 동시에 처리하는 요청이 많아지지만, 모델 사본이 레플리카 수만큼 늘어나 메모리와 자원 예약량도 함께 늘어납니다. 프로세스를 나눈 덕분에 레플리카들이 서로의 상태를 공유하지 않고 각자 요청을 처리하는 대신, 같은 모델을 여러 번 적재하는 비용을 치릅니다. 레플리카 수는 요청 부하에 맞춰 정하고, 레플리카 하나가 차지하는 자원은 `ray_actor_options`로 정합니다.

### `ray_actor_options`로 레플리카 자원 정하기

`ray_actor_options`는 각 레플리카에 적용할 설정을 담은 딕셔너리입니다. 데코레이터 인자 가운데 `num_replicas`는 디플로이먼트 전체의 크기를, `ray_actor_options` 안의 값은 레플리카 하나의 요구량을 정해서 적용 범위가 다릅니다.

| 설정 | 적용 범위 | 값의 의미 |
| --- | --- | --- |
| `num_replicas` | 디플로이먼트 전체에 적용됩니다. | 띄울 레플리카 개수를 나타내는 정수입니다. |
| `ray_actor_options`의 `num_cpus` | 레플리카 하나에 적용됩니다. | 레플리카가 예약할 논리 CPU 수를 나타내는 실수입니다. |
| `ray_actor_options`의 `num_gpus` | 레플리카 하나에 적용됩니다. | 레플리카가 예약할 논리 GPU 수를 나타내는 실수입니다. |
| `ray_actor_options`의 `resources` | 레플리카 하나에 적용됩니다. | HPU나 TPU처럼 GPU가 아닌 가속기 등 그 밖의 자원 요구를 담은 딕셔너리입니다. |

앞의 예제 설정은 레플리카 두 개를 띄우고, 각 레플리카가 CPU 0.2개와 GPU 0개를 예약합니다. `num_cpus`와 `num_gpus`는 실수라서 분수로 지정할 수 있고, 그러면 CPU나 GPU 개수보다 많은 레플리카를 한 머신에 함께 올릴 수 있습니다. 여기서 "논리" 단위라는 말이 핵심입니다. 이 값은 레플리카를 어느 머신에 배치할지 정할 때 쓰는 예약량이며, 작은 모델 여러 개가 CPU나 GPU 하나를 나눠 쓰도록 촘촘히 배치하는 데 쓰입니다.

다음은 분수 예약이 배치에 미치는 영향을 손으로 따라가 본 설명용 계산이며, 실제 측정값이 아닙니다. CPU가 하나뿐인 머신에 `num_cpus=0.2`인 `Translator` 레플리카를 두 개 배치하면 예약량은 다음과 같습니다.

$$
2 \times 0.2 = 0.4 \ \text{CPU}
$$

논리 CPU가 0.6 남으므로, 남은 양이 요구량 이상인 동안 같은 조건의 레플리카를 계속 올려 총 다섯 개까지 이 머신에 들어갑니다. 다섯 번째 레플리카를 올릴 때는 남은 0.2가 요구량 0.2와 같아서 들어가고, 그 뒤에는 남은 양이 없습니다. `num_cpus`를 1로 두었다면 이 머신에는 레플리카가 하나만 들어갑니다. 논리 자원 예약은 배치를 정하는 값이므로, 값은 모델의 실제 부하를 보고 정해야 합니다. 분수 값으로 레플리카를 많이 모을수록 한 머신의 실제 연산 부담도 커지기 때문입니다. GPU 모델이라면 `num_gpus`를 1 대신 0.5처럼 주어 작은 모델 두 개가 GPU 하나를 함께 쓰게 만들 수 있습니다.

### `bind`로 애플리케이션 정의하기

데코레이터를 붙인 것만으로는 아무것도 실행되지 않습니다. 다음 단계는 디플로이먼트를 생성자에 넘길 인자와 `bind`로 묶는 일이며, 이렇게 해서 애플리케이션이 정의됩니다. `Translator`의 생성자는 인자를 받지 않으므로 앞의 코드처럼 `Translator.bind()`를 빈 인자로 호출했습니다.

생성자가 인자를 받는다면 `bind`에 그 값을 넘깁니다.

```python
from ray import serve

@serve.deployment
class MyFirstDeployment:
    # Take the message to return as an argument to the constructor.
    def __init__(self, msg):
        self.msg = msg

    def __call__(self):
        return self.msg

my_first_deployment = MyFirstDeployment.bind("Hello world!")
```

`bind`는 객체를 만들지 않습니다. 어떤 디플로이먼트를 어떤 생성자 인자로 띄울지 기록만 하고, 실제 인스턴스 생성은 애플리케이션을 배포할 때 일어납니다. 배포가 시작되면 각 레플리카가 자기 프로세스에서 `MyFirstDeployment("Hello world!")`에 해당하는 생성을 합니다. 이렇게 정의한 애플리케이션은 로컬에서 실행하거나 프로덕션에 배포할 수 있습니다. 모델 경로처럼 환경마다 달라지는 값은 생성자 인자로 빼 두고 `bind`에서 주입하면 클래스 코드를 바꿀 필요가 없습니다. 키워드 인자도 받으므로 `bind(increment=1)`처럼 써도 됩니다.

정의한 애플리케이션은 `serve.run`이나 같은 역할의 `serve run` CLI 명령으로 배포합니다. `translator_app`처럼 `bind` 결과를 모듈 수준 변수에 담아 두면 CLI가 `module:application` 형식의 경로로 찾아갑니다.

### 애플리케이션과 인그레스 디플로이먼트

애플리케이션은 Ray Serve 클러스터에서 업그레이드하는 단위이며, 하나 이상의 디플로이먼트로 이루어집니다. `Translator.bind()`는 디플로이먼트 하나짜리 애플리케이션이고, 여러 모델과 비즈니스 로직을 엮으면 디플로이먼트 여러 개짜리 애플리케이션이 됩니다. 업그레이드 단위가 애플리케이션이라서 묶는 방식이 운영에 바로 영향을 줍니다. 요약 디플로이먼트와 번역 디플로이먼트를 한 애플리케이션에 묶어 두면, 번역 모델만 고치더라도 갱신 대상은 요약 디플로이먼트까지 포함한 애플리케이션 전체입니다. 그러므로 늘 함께 바뀌고 함께 배포되어야 하는 디플로이먼트는 같은 애플리케이션에 묶고, 서로 독립적으로 갱신할 대상은 별개 애플리케이션으로 나누는 편이 좋습니다.

애플리케이션 안의 디플로이먼트 가운데 하나는 항상 최상위입니다. 이 디플로이먼트를 `serve.run`에 넘기며, 애플리케이션으로 들어오는 모든 트래픽의 진입점이 되므로 인그레스 디플로이먼트(ingress deployment)라고 부릅니다. 인그레스는 요청을 받아 다른 디플로이먼트로 라우팅하거나 호출하고, 결과를 조합해 사용자에게 돌려줍니다. 애플리케이션은 지정한 `route_prefix`의 HTTP로 호출하거나, 파이썬에서 `DeploymentHandle`로 호출합니다.

요약 모델을 디플로이먼트로 만든 `Summarizer`가 있고, 그 생성자가 번역 디플로이먼트를 인자로 받는다고 하면, 요약과 번역을 잇는 애플리케이션은 다음 한 줄로 정의됩니다.

```python
app = Summarizer.bind(Translator.bind())
```

`Translator`를 빈 생성자 인자로 바인딩하고, 그 결과를 `Summarizer`의 생성자 인자로 넘겼습니다. 바깥쪽 `Summarizer`가 `serve.run`에 넘어가는 최상위이므로 인그레스 디플로이먼트이며, 요청을 가장 먼저 받기 때문에 `__call__` 메서드도 `Summarizer`에 둡니다. 생성자 인자로 넘긴 바인딩된 디플로이먼트는 실행 시점에 `DeploymentHandle`로 바뀌어, `Summarizer`가 `Translator`를 호출하는 통로가 됩니다. 이렇게 `bind` 호출의 중첩 구조가 곧 애플리케이션의 디플로이먼트 그래프이고, 각 디플로이먼트는 자기 데코레이터 인자에 따라 따로 확장됩니다.

인그레스 디플로이먼트는 애플리케이션의 HTTP 처리 로직도 맡습니다. 기본 동작에서는 클래스의 `__call__`이 Starlette 요청 객체를 받아 호출되고, 반환값은 JSON으로 직렬화되어 HTTP 응답 본문으로 돌아갑니다. 앞의 `Translator`에 `__call__`을 추가한 이유가 여기에 있습니다. `translate`는 문자열만 다루므로, `__call__`이 요청의 JSON 본문을 읽어 `translate`에 넘기고 번역문을 그대로 반환합니다. 디플로이먼트 하나짜리 애플리케이션에서는 `Translator` 자신이 인그레스입니다.

> **핵심:** 인그레스 디플로이먼트는 별도 종류의 객체가 아니라 `bind` 그래프의 최상위에 놓인 평범한 디플로이먼트입니다. 어느 디플로이먼트가 인그레스인지는 클래스 정의가 아니라 애플리케이션을 조립하는 방식이 정합니다.

네 개념을 코드와 연결해 정리하면 다음과 같습니다.

| 개념 | 코드에서 대응하는 부분 | 실행 시점의 모습 |
| --- | --- | --- |
| 디플로이먼트 | `@serve.deployment`를 붙인 클래스나 함수 | 레플리카 묶음 |
| 레플리카 | `num_replicas`, `ray_actor_options` 설정 | 별도 Ray 액터 프로세스에서 도는 클래스 복사본 |
| 애플리케이션 | `bind` 호출로 만든 객체 | 함께 배포되고 업그레이드되는 디플로이먼트 묶음 |
| 인그레스 디플로이먼트 | `bind` 그래프의 최상위, `serve.run`에 넘기는 디플로이먼트 | 애플리케이션의 모든 트래픽을 받는 진입점 |

## 로컬에서 실행하고 HTTP로 테스트하기

앞서 `@serve.deployment`와 `__call__`을 추가하고 `translator_app = Translator.bind()`로 끝나는 번역 디플로이먼트 코드를 만들었습니다. 이 코드를 `serve_quickstart.py`라는 이름으로 저장하면 바로 로컬에서 띄워 HTTP로 호출할 수 있습니다.

### `serve run`으로 애플리케이션 띄우기

로컬 테스트에는 `serve run` CLI 명령을 씁니다. 이 명령은 애플리케이션의 임포트 경로를 `module:application` 형식으로 받습니다. 콜론 앞은 파이썬 모듈 이름이고, 콜론 뒤는 그 모듈 안에서 `bind` 결과를 담은 변수 이름입니다.

```bash
$ serve run serve_quickstart:translator_app
```

모듈을 임포트해서 애플리케이션을 찾으므로 `serve_quickstart.py`가 있는 디렉터리에서 명령을 실행해야 합니다.

명령을 실행하면 `translator_app`이 실행되고, 터미널은 그대로 붙잡힌 채 로그를 콘솔에 계속 출력합니다. `Ctrl-C`를 누르면 애플리케이션이 내려갑니다. 개발 중에 코드를 고치고 다시 띄우기를 반복할 때 이 명령을 씁니다.

### 클라이언트로 요청 보내기

애플리케이션이 떠 있으면 기본 주소 `http://127.0.0.1:8000/`로 접근합니다. `Translator`의 `__call__`은 요청 본문을 JSON으로 읽어 `translate`에 넘기므로, 클라이언트는 영어 문장을 JSON 본문에 담아 POST 요청을 보냅니다.

```python
# File name: model_client.py
import requests

english_text = "Hello world!"

response = requests.post("http://127.0.0.1:8000/", json=english_text)
french_text = response.text

print(french_text)
```

`serve run`이 돌고 있는 터미널은 그대로 두고, 새 터미널을 열어 클라이언트를 실행합니다.

```bash
$ python model_client.py

Bonjour monde!
```

`json=english_text`는 파이썬 문자열 하나를 그대로 JSON 값으로 직렬화합니다. 그래서 서버 쪽 `await http_request.json()`도 딕셔너리가 아니라 문자열을 돌려주고, 그 값이 곧바로 `translate`의 인자가 됩니다. 요청 스키마를 이렇게 단순하게 두면 예제는 짧아지지만, 입력 필드가 늘어나면 딕셔너리로 바꾸는 편이 낫습니다. 뒤의 scikit-learn 예제가 그런 형태입니다.

### 다른 프레임워크 모델도 같은 순서로

Ray Serve는 프레임워크를 가리지 않으므로 TensorFlow, PyTorch, scikit-learn 모델도 순서가 같습니다. 생성자에서 모델을 불러오고, `__call__`에서 Starlette 요청을 모델 입력으로 바꿔 추론한 뒤, 결과를 JSON으로 직렬화할 수 있는 값으로 반환합니다. 그다음 `bind`로 애플리케이션을 만들고 `serve run`으로 띄웁니다.

scikit-learn 예제는 미리 학습해 파일로 저장해 둔 분류기를 서빙합니다. 다음 코드는 `tutorial_sklearn.py`의 디플로이먼트 부분이며, `MODEL_PATH`와 `LABEL_PATH`는 같은 파일 앞부분에서 Iris 데이터로 `GradientBoostingClassifier`를 학습해 pickle과 JSON으로 저장한 경로입니다.

```python
@serve.deployment
class BoostingModel:
    def __init__(self, model_path: str, label_path: str):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(label_path) as f:
            self.label_list = json.load(f)

    async def __call__(self, starlette_request: Request) -> Dict:
        payload = await starlette_request.json()
        print("Worker: received starlette request with data", payload)

        input_vector = [
            payload["sepal length"],
            payload["sepal width"],
            payload["petal length"],
            payload["petal width"],
        ]
        prediction = self.model.predict([input_vector])[0]
        human_name = self.label_list[prediction]
        return {"result": human_name}

boosting_model = BoostingModel.bind(MODEL_PATH, LABEL_PATH)
```

모델 파일 경로를 생성자 인자로 받아서, 디플로이먼트가 인스턴스화될 때 파일에서 분류기를 불러옵니다. 경로는 `bind`에서 주입하므로 다른 모델 파일로 바꿀 때 클래스 코드를 고치지 않아도 됩니다. 실행과 호출은 다음과 같습니다.

```bash
$ serve run tutorial_sklearn:boosting_model
```

```python
import requests

sample_request_input = {
    "sepal length": 1.2,
    "sepal width": 1.0,
    "petal length": 1.1,
    "petal width": 0.9,
}
response = requests.get("http://localhost:8000/", json=sample_request_input)
print(response.text)
```

결과는 `{"result": "versicolor"}` 같은 형태이며, 학습 데이터를 무작위로 섞기 때문에 예측 라벨은 실행마다 달라질 수 있습니다. 이 클라이언트는 POST가 아니라 GET에 JSON 본문을 실어 보냅니다. 기본적으로 `__call__`이 Starlette 요청 객체를 받아 호출되므로, 이 예제에서는 GET으로 보내도 같은 메서드가 요청을 처리합니다.

PyTorch 예제는 입력이 JSON이 아니라 이미지 바이트입니다. `__call__`에서 `await starlette_request.body()`로 원본 바이트를 읽어 PIL 이미지로 열고, torchvision 전처리를 거쳐 ResNet18에 넣은 뒤 가장 점수가 높은 클래스 인덱스를 반환합니다. 클라이언트도 `json=` 대신 `data=`로 바이트를 보냅니다.

```python
import requests

ray_logo_bytes = requests.get(
    "https://raw.githubusercontent.com/ray-project/"
    "ray/master/doc/source/images/ray_header_logo.png"
).content

resp = requests.post("http://localhost:8000/", data=ray_logo_bytes)
print(resp.json())
```

세 프레임워크 예제를 요청 형식 기준으로 정리하면 다음과 같습니다.

| 예제 | 실행 명령 | 요청 본문 읽는 방법 | 반환값 |
| --- | --- | --- | --- |
| TensorFlow MNIST | `serve run tutorial_tensorflow:mnist_model` | `await starlette_request.json()`의 `"array"` 필드 | 예측값 목록과 모델 파일 경로를 담은 딕셔너리 |
| PyTorch ResNet18 | `serve run tutorial_pytorch:image_model` | `await starlette_request.body()`로 읽은 이미지 바이트 | 클래스 인덱스를 담은 딕셔너리 |
| scikit-learn 분류기 | `serve run tutorial_sklearn:boosting_model` | `await starlette_request.json()`의 꽃받침·꽃잎 필드 | 예측 라벨 이름을 담은 딕셔너리 |

세 예제 모두 반환값이 딕셔너리라서 JSON 응답으로 직렬화되고, 클라이언트는 `resp.json()`이나 `response.text`로 결과를 읽습니다. 자기 모델을 옮길 때는 클라이언트가 보낼 본문 형식을 먼저 정하는 편이 좋습니다. 그 형식에 따라 `json()`과 `body()` 가운데 무엇으로 읽을지와 클라이언트 코드가 함께 정해집니다.

## HTTP 처리와 FastAPI 통합

Ray Serve는 HTTP API를 두 수준으로 제공합니다. 원시 요청 객체를 직접 다뤄도 괜찮다면 Starlette의 `Request` API를 쓰고, 입력 검증과 문서 생성까지 갖춘 API 서버가 필요하면 FastAPI 통합을 씁니다. 어느 쪽이든 HTTP로 노출되는 것은 `serve.run`에 넘긴 인그레스 디플로이먼트 하나뿐이므로, HTTP 처리 코드는 모두 인그레스 디플로이먼트에 둡니다.

### Starlette 요청 객체를 직접 다루기

가장 낮은 수준의 방식은 디플로이먼트에 `__call__`을 정의하는 것입니다. 요청 경로가 `/`이면 Serve HTTP 서버는 디플로이먼트의 `__call__`을 호출하고, Starlette `Request` 객체 하나만 인자로 넘깁니다. 본문뿐 아니라 쿼리 문자열도 이 객체에서 읽습니다.

```python
import starlette.requests
import requests
from ray import serve

@serve.deployment
class Counter:
    def __call__(self, request: starlette.requests.Request):
        return request.query_params

serve.run(Counter.bind())
resp = requests.get("http://localhost:8000?a=b&c=d")
assert resp.json() == {"a": "b", "c": "d"}
```

`__call__`은 JSON으로 직렬화할 수 있는 객체를 반환해도 되고, Starlette `Response` 객체를 반환해도 됩니다. 상태 코드나 헤더를 직접 정해야 한다면 `Response` 객체를 반환합니다. 애플리케이션이 받는 경로 접두사는 기본값이 `/`이며, `serve.run()`이나 Serve 설정 파일(Serve config)에서 `route_prefix`로 바꿉니다.

이 방식은 의존성이 적고 동작이 투명합니다. 다만 메서드 구분, 경로 변수, 입력 타입 검사를 모두 손으로 작성해야 해서, 엔드포인트가 둘 이상으로 늘어나면 `__call__` 안에 분기문이 쌓입니다.

### `@serve.ingress`로 FastAPI 앱 감싸기

HTTP 처리 로직이 복잡해지면 FastAPI 통합을 씁니다. `@serve.ingress` 데코레이터는 FastAPI 앱을 감싸서, 경로 변수, 자동 타입 검증, 의존성 주입 같은 FastAPI 기능을 디플로이먼트 안에서 그대로 쓰게 해 줍니다. 번역 모델을 FastAPI 디플로이먼트로 만들면 다음과 같습니다.

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

필요한 변경은 세 가지입니다. Ray Serve와 FastAPI를 임포트하고, `@serve.deployment`와 `@serve.ingress(app)` 데코레이터를 붙이고, `Translator` 디플로이먼트를 생성자 인자와 함께 `bind`합니다. `translate`에는 `@app.post("/")`를 달았습니다. FastAPI가 `text: str` 시그니처를 보고 요청을 파싱하고 타입을 검증하므로 요청 파싱 코드를 따로 쓰지 않아도 됩니다. 실행은 `serve run model:translator_app`처럼 모듈 이름과 `translator_app`을 넘깁니다. 단순 타입 인자는 FastAPI가 쿼리 파라미터로 받으므로, 클라이언트도 `params`로 값을 보내고 응답을 JSON으로 읽습니다.

```python
# File name: model_client.py
import requests

response = requests.post("http://127.0.0.1:8000/", params={"text": "Hello world!"})
french_text = response.json()

print(french_text)
```

FastAPI 경로는 Serve의 `route_prefix` 위에 덧붙습니다. 애플리케이션의 `route_prefix`가 `/my_app`이고 메서드에 `@app.get("/fetch_data")`를 달았다면 실제 호출 경로는 `/my_app/fetch_data`입니다. 같은 클래스에 HTTP 메서드와 경로가 다른 핸들러를 여러 개 두는 것도 FastAPI 문법 그대로입니다.

```python
import ray
import requests
from fastapi import FastAPI
from ray import serve

app = FastAPI()

@serve.deployment
@serve.ingress(app)
class MyFastAPIDeployment:
    @app.get("/")
    def root(self):
        return "Hello, world!"

    @app.post("/{subpath}")
    def root(self, subpath: str):
        return f"Hello from {subpath}!"

serve.run(MyFastAPIDeployment.bind(), route_prefix="/hello")
resp = requests.post("http://localhost:8000/hello/Serve")
assert resp.json() == "Hello from Serve!"
```

이미 운영 중인 FastAPI 앱이 있다면 코드를 고치지 않고 빈 클래스로 감싸기만 해도 됩니다. 모듈 수준 함수로 등록한 경로도 그대로 동작하며, 기존 미들웨어와 자동 OpenAPI 문서 생성 같은 기능도 그대로 쓸 수 있습니다. 기존 FastAPI 앱을 수정 없이 여러 레플리카로 확장할 때 유용합니다.

```python
import ray
import requests
from fastapi import FastAPI
from ray import serve

app = FastAPI()

@app.get("/")
def f():
    return "Hello from the root!"

@serve.deployment
@serve.ingress(app)
class FastAPIWrapper:
    pass

serve.run(FastAPIWrapper.bind(), route_prefix="/")
resp = requests.get("http://localhost:8000/")
assert resp.json() == "Hello from the root!"
```

두 방식을 비교하면 다음과 같습니다.

| 구분 | Starlette `Request` 직접 처리 | FastAPI 통합 |
| --- | --- | --- |
| 진입 메서드 | `__call__` 하나 | `@app.get`, `@app.post` 등을 붙인 메서드 |
| 입력 파싱 | `await request.json()`, `request.query_params` 등을 직접 호출 | 핸들러 시그니처를 보고 FastAPI가 자동 처리 |
| 경로·메서드 구분 | 직접 분기 | 데코레이터로 선언 |
| 적합한 경우 | 엔드포인트 하나짜리 단순한 모델 API | 검증, 문서 생성, 여러 경로가 필요한 API 서버 |

### 미들웨어 설정

HTTP 미들웨어는 `@serve.ingress`에 넘기는 FastAPI 앱에 등록합니다. `HTTPOptions.middlewares` 필드는 지원 중단(deprecated)됐으므로 새 코드에는 쓰지 않습니다. CORS 설정은 다음과 같습니다.

```python
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ray import serve

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://example.com"],
    allow_methods=["GET", "POST"],
)

@serve.deployment
@serve.ingress(app)
class Ingress:
    @app.get("/")
    def root(self):
        return "ok"

serve.run(Ingress.bind())
resp = requests.get(
    "http://localhost:8000/",
    headers={"Origin": "https://example.com"},
)
assert resp.json() == "ok"
assert resp.headers["access-control-allow-origin"] == "https://example.com"
```

### FastAPI 팩토리 패턴

지금까지 본 방식은 모듈 수준에서 만든 FastAPI 객체를 `@serve.ingress`에 넘깁니다. 이 객체는 레플리카 프로세스로 전달되어야 하므로 cloudpickle로 직렬화할 수 있어야 합니다. 스레드 락처럼 직렬화할 수 없는 구성 요소에 의존하는 라이브러리를 쓰면 여기서 막힙니다. OpenTelemetry의 `FastAPIInstrumentor`가 대표적입니다.

팩토리 패턴은 FastAPI 객체 대신 객체를 만드는 함수를 `@serve.ingress`에 넘깁니다. 앱은 각 레플리카에서 직접 만들어지므로 직렬화가 아예 필요 없습니다.

```python
import requests
from fastapi import FastAPI
from ray import serve
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

@serve.deployment
class ChildDeployment:
    def __call__(self):
        return "Hello from the child deployment!"

def fastapi_factory():
    app = FastAPI()
    FastAPIInstrumentor.instrument_app(app)

    @app.get("/")
    async def root():
        handle = serve.get_deployment_handle("ChildDeployment", app_name="default")
        return {"message": await handle.remote()}

    return app

@serve.deployment
@serve.ingress(fastapi_factory)
class ParentDeployment:
    def __init__(self, child):
        self.child = child

serve.run(ParentDeployment.bind(ChildDeployment.bind()))

resp = requests.get("http://localhost:8000/")
assert resp.json() == {"message": "Hello from the child deployment!"}
```

대가도 있습니다. 팩토리 안에서 정의한 핸들러는 디플로이먼트 인스턴스에 접근하지 못하므로 `self`가 없고, 생성자에서 받은 `self.child`도 쓸 수 없습니다. 그래서 하위 디플로이먼트는 `serve.get_deployment_handle`에 디플로이먼트 이름과 애플리케이션 이름을 넘겨 찾아옵니다. 직렬화 문제가 없는 앱이라면 객체 방식이 더 간단하고, 계측 라이브러리처럼 직렬화가 막히는 경우에만 팩토리 패턴을 고르는 편이 좋습니다.

### WebSocket

WebSocket도 FastAPI를 통해 지원합니다. WebSocket 요청을 처리할 함수에 `@app.websocket`을 붙입니다. 연결은 한 번 맺은 뒤 그 안에서 메시지를 여러 번 주고받으므로, 핸들러는 보통 루프를 돌다가 클라이언트가 연결을 끊으면 `WebSocketDisconnect`를 받아 빠져나옵니다.

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from ray import serve

app = FastAPI()

@serve.deployment
@serve.ingress(app)
class EchoServer:
    @app.websocket("/")
    async def echo(self, ws: WebSocket):
        await ws.accept()

        try:
            while True:
                text = await ws.receive_text()
                await ws.send_text(text)
        except WebSocketDisconnect:
            print("Client disconnected.")

serve_app = serve.run(EchoServer.bind())
```

클라이언트는 `websockets` 패키지(`pip install websockets`)로 접속합니다.

```python
from websockets.sync.client import connect

with connect("ws://localhost:8000") as websocket:
    websocket.send("Eureka!")
    assert websocket.recv() == "Eureka!"

    websocket.send("I've found it!")
    assert websocket.recv() == "I've found it!"
```

### 요청 취소와 `asyncio.CancelledError`

요청 처리 시간이 종단 간 타임아웃을 넘기거나, 클라이언트가 응답을 받기 전에 연결을 끊으면 Serve는 처리 중인 요청을 취소합니다. 취소 방식은 요청이 어디까지 갔느냐에 따라 다릅니다. 프록시(proxy)가 아직 요청을 레플리카에 보내지 않았다면 요청을 그냥 버립니다. 이미 레플리카로 보냈다면 레플리카에서 핸들러를 실행하는 `asyncio.Task`를 취소합니다. 그러면 핸들러가 다음에 `await` 문에 들어가는 순간 `asyncio.CancelledError`가 발생합니다.

이 구조가 중요한 이유는 취소가 `await` 지점에서만 일어나기 때문입니다. 파이썬 asyncio의 작업 취소는 실행 중인 코드를 강제로 멈추지 않고, 작업이 이벤트 루프에 제어권을 돌려주는 순간에 예외를 집어넣습니다. 요청이 끝날 때까지 남은 `await`가 없다면 레플리카는 요청을 끝까지 처리해 응답을 프록시에 보내고, 프록시는 그 응답을 버립니다. 이미 떠난 클라이언트를 위해 자원을 끝까지 쓰게 됩니다. 앞의 번역 핸들러가 바로 이 경우입니다. `model.generate()`는 `await` 없이 끝까지 도는 동기 호출이므로, 클라이언트가 중간에 연결을 끊어도 빔 탐색은 끝까지 진행됩니다. 오래 걸리는 블로킹 작업은 `await`로 기다리는 형태로 작성해 두어야 Serve가 그 작업이 끝나기를 기다리지 않고 요청을 취소합니다.

취소 시점에 정리 작업을 하고 싶다면 `try`/`except`로 예외를 잡습니다.

```python
import asyncio
from ray import serve

@serve.deployment
async def startled():
    try:
        print("Replica received request!")
        await asyncio.sleep(10000)
    except asyncio.CancelledError:
        # Add custom behavior that should run
        # upon cancellation here.
        print("Request got cancelled!")
```

취소는 아래로 전파됩니다. 요청 처리 메서드에서 띄운 하위 디플로이먼트 핸들 호출, 태스크, 액터 호출도 함께 취소되며, 하위 핸들러도 인그레스와 같은 방식으로 `asyncio.CancelledError`를 처리할 수 있습니다. 반대로 요청이 취소되더라도 끝까지 마쳐야 하는 비동기 호출은 `asyncio.shield()`로 감쌉니다.

```python
import asyncio
from ray import serve

@serve.deployment
class SnoringSleeper:
    async def snore(self):
        await asyncio.sleep(1)
        print("ZZZ")

    async def __call__(self):
        try:
            print("SnoringSleeper received request!")

            # Prevent the snore() method from being cancelled
            await asyncio.shield(self.snore())

        except asyncio.CancelledError:
            print("SnoringSleeper's request was cancelled!")

app = SnoringSleeper.bind()
```

요청이 취소되면 `__call__` 안에서는 취소 예외가 발생하지만, `snore()` 안에서는 발생하지 않습니다. 그래서 요청이 취소돼도 `ZZZ`가 출력됩니다. 단, `asyncio.shield`로 `DeploymentHandle` 호출을 감싸도 하위 핸들러의 취소는 막지 못합니다. 하위 디플로이먼트가 취소되면 안 되는 작업을 한다면 그 핸들러에서도 취소 예외를 직접 처리해야 합니다.

다음은 취소 시점을 손으로 따라가 본 설명용 시나리오이며, 측정값이 아닙니다. 핸들러가 전처리, `await`로 기다리는 하위 호출, 동기 후처리의 세 단계로 이루어져 있다고 가정합니다. 클라이언트가 전처리 도중 연결을 끊으면 전처리는 끝까지 실행되고, 하위 호출의 `await`에 들어가는 순간 `CancelledError`가 발생하며 하위 호출에도 취소가 전파됩니다. 연결이 동기 후처리 도중에 끊기면 남은 `await`가 없으므로 후처리가 끝까지 실행되고, 만들어진 응답은 프록시에서 버려집니다. 같은 끊김이라도 어느 단계에서 일어났느냐에 따라 아낄 수 있는 연산량이 달라집니다.

### 스트리밍 응답

LLM 텍스트 생성이나 영상 처리처럼 전체 순전파에 몇 초씩 걸리는 작업은 결과가 나오는 대로 조금씩 돌려주는 편이 사용자 경험이 훨씬 낫습니다. HTTP 스트리밍 응답(streaming response)을 쓰려면 HTTP 핸들러에서 제너레이터를 감싼 Starlette `StreamingResponse`를 반환합니다. `__call__`을 쓰는 기본 인그레스와 FastAPI 통합 모두 지원합니다.

```python
import time
from typing import Generator

import requests
from starlette.responses import StreamingResponse
from starlette.requests import Request

from ray import serve

@serve.deployment
class StreamingResponder:
    def generate_numbers(self, max: int) -> Generator[str, None, None]:
        for i in range(max):
            yield str(i)
            time.sleep(0.1)

    def __call__(self, request: Request) -> StreamingResponse:
        max = request.query_params.get("max", "25")
        gen = self.generate_numbers(int(max))
        return StreamingResponse(gen, status_code=200, media_type="text/plain")

serve.run(StreamingResponder.bind())

r = requests.get("http://localhost:8000?max=10", stream=True)
start = time.time()
r.raise_for_status()
for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
    print(f"Got result {round(time.time()-start, 1)}s after start: '{chunk}'")
```

`__call__`은 제너레이터를 만들어 `StreamingResponse`에 넘기기만 하고 바로 반환합니다. 값은 제너레이터가 `yield`할 때마다 클라이언트로 흘러갑니다. 클라이언트도 `requests`에 `stream=True`를 주고 `iter_content`로 조각을 하나씩 읽어야 합니다. 이 스크립트를 실행하면 숫자가 한꺼번에 오지 않고 약 0.1초 간격으로 하나씩 도착합니다.

클라이언트가 스트림을 끝까지 받기 전에 연결을 끊으면 처리를 멈추고 싶을 때가 많습니다. `StreamingResponse`에 비동기 제너레이터를 넘기면, 클라이언트가 연결을 끊을 때 제너레이터가 취소되면서 `asyncio.CancelledError`가 발생합니다. 요청 취소와 같은 원리이므로 제너레이터 안 어딘가에 반드시 `await`가 있어야 합니다.

```python
import asyncio
import time
from typing import AsyncGenerator

import requests
from starlette.responses import StreamingResponse
from starlette.requests import Request

from ray import serve

@serve.deployment
class StreamingResponder:
    async def generate_forever(self) -> AsyncGenerator[str, None]:
        try:
            i = 0
            while True:
                yield str(i)
                i += 1
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            print("Cancelled! Exiting.")

    def __call__(self, request: Request) -> StreamingResponse:
        gen = self.generate_forever()
        return StreamingResponse(gen, status_code=200, media_type="text/plain")

serve.run(StreamingResponder.bind())

r = requests.get("http://localhost:8000?max=10", stream=True)
start = time.time()
r.raise_for_status()
for i, chunk in enumerate(r.iter_content(chunk_size=None, decode_unicode=True)):
    print(f"Got result {round(time.time()-start, 1)}s after start: '{chunk}'")
    if i == 10:
        print("Client disconnecting")
        break
```

이 제너레이터는 끝나는 조건이 없어서 클라이언트가 끊을 때까지 계속 숫자를 보냅니다. 클라이언트가 열한 번째 조각을 받고 루프를 빠져나가면, 레플리카 로그에 `Cancelled! Exiting.`이 찍히고 제너레이터가 종료됩니다. 토큰을 하나씩 생성하는 LLM 핸들러라면 이 구조로 사용자가 연결을 끊을 때 생성을 멈추게 할 수 있습니다.

### keep-alive 타임아웃

Serve는 내부에서 Uvicorn HTTP 서버로 요청을 처리합니다. Uvicorn은 기본적으로 요청 사이에 HTTP 연결을 5초 동안 유지합니다. 이 값은 Serve 설정 파일의 `http_options` 필드에 `keep_alive_timeout_s`를 설정해 바꿉니다. 이 설정은 Ray 클러스터 전체에 적용되며 실행 중에는 바꿀 수 없으므로, 배포 전에 정해 둡니다.

## 여러 모델 조합하기

실제 추론 서비스가 모델 하나로 끝나는 경우는 드뭅니다. 입력을 정리하는 전처리, 여러 모델의 추론, 결과를 다듬는 후처리가 이어지고, 요청 내용에 따라 어느 모델로 보낼지 고르는 비즈니스 로직도 끼어듭니다. 이 단계를 모두 디플로이먼트 하나에 넣으면, 단계마다 필요한 자원과 처리량이 달라도 함께 늘리고 줄여야 합니다. CPU만 쓰는 전처리까지 GPU 레플리카에 얹히고, 가장 느린 단계가 애플리케이션 전체의 병목이 됩니다. Ray Serve의 모델 조합은 각 단계를 별도 디플로이먼트로 나누고, 디플로이먼트끼리 파이썬 메서드를 호출하듯 서로를 부르게 해서 이 문제를 풉니다. 그러면 단계마다 레플리카 수와 자원 예약을 따로 정할 수 있습니다.

### 바인딩한 디플로이먼트가 DeploymentHandle로 바뀌는 과정

조합의 출발점은 앞서 본 `bind`입니다. 한 디플로이먼트를 `bind`한 결과를 다른 디플로이먼트의 `bind`에 생성자 인자로 넘기면, 실행 시점에 Ray Serve가 그 인자를 `DeploymentHandle`로 바꿔 생성자에 넣어 줍니다. 생성자가 받는 값은 하위 디플로이먼트의 인스턴스가 아니라, 그 디플로이먼트의 레플리카로 요청을 보내는 핸들입니다. 하위 디플로이먼트는 별도 프로세스에서 자기 레플리카 수만큼 떠 있고, 상위 디플로이먼트는 핸들로 그쪽 메서드를 호출합니다.

요청의 언어 필드를 보고 스페인어 또는 프랑스어 응답 디플로이먼트로 요청을 보내는 예제입니다.

```python
# File name: hello.py
from ray import serve
from ray.serve.handle import DeploymentHandle


@serve.deployment
class LanguageClassifer:
    def __init__(
        self, spanish_responder: DeploymentHandle, french_responder: DeploymentHandle
    ):
        self.spanish_responder = spanish_responder
        self.french_responder = french_responder

    async def __call__(self, http_request):
        request = await http_request.json()
        language, name = request["language"], request["name"]

        if language == "spanish":
            response = self.spanish_responder.say_hello.remote(name)
        elif language == "french":
            response = self.french_responder.say_hello.remote(name)
        else:
            return "Please try again."

        return await response


@serve.deployment
class SpanishResponder:
    def say_hello(self, name: str):
        return f"Hola {name}"


@serve.deployment
class FrenchResponder:
    def say_hello(self, name: str):
        return f"Bonjour {name}"


spanish_responder = SpanishResponder.bind()
french_responder = FrenchResponder.bind()
language_classifier = LanguageClassifer.bind(spanish_responder, french_responder)
```

마지막 줄에서 `LanguageClassifer`는 두 응답 디플로이먼트를 생성자 인자로 받고, 실행 중에는 이 인자가 `DeploymentHandle`이 되어 `self.spanish_responder`와 `self.french_responder`에 담깁니다. `self.spanish_responder.say_hello.remote(name)`처럼 핸들 뒤에 메서드 이름을 쓰고 `remote`를 붙이면 `__call__`이 아닌 메서드도 이름으로 부를 수 있으며, 위치 인자와 키워드 인자는 개수 제한 없이 그대로 메서드에 전달됩니다. HTTP 요청을 받는 쪽은 인그레스 디플로이먼트인 `LanguageClassifer`뿐이고, 응답 디플로이먼트는 HTTP를 전혀 모른 채 문자열만 주고받습니다. `hello.py`가 있는 디렉터리에서 실행하고 새 터미널에서 클라이언트를 돌립니다.

```bash
$ serve run hello:language_classifier
```

```python
# File name: hello_client.py
import requests

response = requests.post(
    "http://localhost:8000", json={"language": "spanish", "name": "Dora"}
)
greeting = response.text
print(greeting)
```

```bash
$ python hello_client.py

Hola Dora
```

핸들 호출은 HTTP 요청과 같은 데이터 경로를 타므로 디플로이먼트 선택과 배칭 절차도 똑같이 적용됩니다. 예전에 쓰던 `RayServeHandle`과 `RayServeSyncHandle`은 Ray 2.10에서 완전히 제거됐으므로, 오래된 예제에서 이 이름을 보면 `DeploymentHandle`로 바꿔 읽으면 됩니다.

### `DeploymentResponse`와 `await`

`handle.method.remote(...)`는 결과값이 아니라 `DeploymentResponse` 객체를 곧바로 반환합니다. 이 객체는 아직 나오지 않은 출력을 가리키는 퓨처(future)입니다. 호출은 비동기로 진행되고, 결과가 필요한 시점에 `await response`로 기다리면 원격 메서드가 끝난 뒤 반환값이 나옵니다. 앞의 예제에서는 `return await response`가 이 역할을 합니다.

`response.result()`도 반환값을 돌려주지만, 디플로이먼트 안에서는 쓰지 않아야 합니다. `.result()`는 원격 호출이 끝날 때까지 레플리카를 붙잡아 다른 코드를 전혀 실행하지 못하게 막습니다. `await`를 쓰면 기다리는 동안 같은 레플리카에 들어온 다른 요청을 처리합니다. 하위 디플로이먼트가 모델 추론에 수백 밀리초를 쓰는 동안 상위 레플리카가 아무 일도 못 한다면, 상위 디플로이먼트의 처리량이 하위 호출 지연 시간에 그대로 묶입니다. `.result()`는 `serve.run`이 돌려준 핸들로 드라이버 스크립트나 테스트 코드에서 결과를 확인할 때 씁니다.

`remote()`가 즉시 반환되므로 독립적인 하위 호출 여러 개를 동시에 진행할 수도 있습니다. 호출을 모두 먼저 보내 놓고 나중에 기다리면 됩니다.

```python
from ray import serve
from ray.serve.handle import DeploymentHandle

@serve.deployment
class Hello:
    def __call__(self) -> str:
        return "Hello"

@serve.deployment
class World:
    def __call__(self) -> str:
        return " world!"

@serve.deployment
class Ingress:
    def __init__(self, hello_handle: DeploymentHandle, world_handle: DeploymentHandle):
        self._hello_handle = hello_handle
        self._world_handle = world_handle

    async def __call__(self) -> str:
        hello_response = self._hello_handle.remote()
        world_response = self._world_handle.remote()
        return (await hello_response) + (await world_response)

hello = Hello.bind()
world = World.bind()

# The deployments passed to the Ingress constructor are replaced with handles.
app = Ingress.bind(hello, world)

# Deploys Hello, World, and Ingress.
handle: DeploymentHandle = serve.run(app)

# `DeploymentHandle`s can also be used to call the ingress deployment of an application.
assert handle.remote().result() == "Hello world!"
```

`hello_response`와 `world_response`를 만드는 두 줄은 기다리지 않고 지나가므로, `Hello`와 `World`는 서로 다른 레플리카에서 동시에 실행됩니다. 같은 입력을 여러 모델에 보내고 결과를 합치는 앙상블 구조라면 이 순서가 중요합니다. `await self._hello_handle.remote()`처럼 호출과 대기를 한 줄에 붙이면 두 호출이 순서대로 실행되어 지연 시간이 합산됩니다. 또 `serve.run`은 인그레스 디플로이먼트의 `DeploymentHandle`을 반환하므로, HTTP를 거치지 않고 파이썬에서 애플리케이션을 바로 호출해 테스트할 수 있습니다.

### 요약과 번역을 잇는 파이프라인

번역 모델에 요약 모델을 붙이면 순차 파이프라인이 됩니다. 영어 문장을 받아 요약하고, 요약문을 프랑스어로 번역합니다. 요약 모델도 t5-small을 쓰고, 입력 앞에 `"summarize: "`를 붙이며 출력 길이를 `max_length=15`로 짧게 제한합니다. `Translator`는 앞서 만든 디플로이먼트에서 `__call__`을 뺀 클래스를 같은 파일에 둡니다.

```python
# File name: serve_quickstart_composed.py
from starlette.requests import Request

import ray
from ray import serve
from ray.serve.handle import DeploymentHandle

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Translator: 앞서 만든 Translator에서 __call__을 뺀 @serve.deployment 클래스


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

`Translator`에서 `__call__`이 빠진 이유는 HTTP 요청을 이제 `Summarizer`가 먼저 받고, `Translator`는 핸들로 `translate`만 호출받기 때문입니다. `Summarizer.__call__`은 요약을 직접 계산한 뒤 `self.translator.translate.remote(summary)`로 번역을 맡기고 결과를 `await`합니다.

```bash
$ serve run serve_quickstart_composed:app
```

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

```bash
$ python composed_client.py

C'était le meilleur des temps, c'était le pire des temps,
```

`self.summarize(english_text)`는 `await` 없이 도는 동기 호출이라서, 요약이 끝날 때까지 `Summarizer` 레플리카는 다른 요청을 받지 못합니다. 반면 번역을 기다리는 동안에는 `await` 덕분에 다음 요청의 요약을 시작할 수 있습니다. 파이프라인을 어디서 나눌지 정할 때는 이렇게 레플리카를 붙잡는 구간이 어디인지도 함께 따져 봐야 합니다.

### 응답을 기다리지 않고 넘기는 체이닝

중간 결과를 상위 디플로이먼트가 직접 쓸 일이 없다면 굳이 기다릴 필요가 없습니다. 핸들 호출이 반환한 `DeploymentResponse`를 다른 핸들 호출의 인자로 그대로 넘기면 Ray Serve가 내부에서 대기를 처리하고, 첫 번째 호출이 끝나면 `DeploymentResponse` 객체 대신 그 출력값을 두 번째 호출에 넣습니다.

```python
# File name: chain.py
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

handle: DeploymentHandle = serve.run(app)
response = handle.remote(5)
assert response.result() == 12, "(5 + 1) * 2 = 12"
```

`Ingress`는 `Adder`의 응답을 받자마자 `Multiplier` 호출에 넘기고, 최종 응답 하나만 `await`합니다. `Multiplier.__call__`의 `val`에는 퓨처가 아니라 `Adder`가 계산한 정수 6이 들어오므로, 하위 디플로이먼트 코드는 입력이 어디서 왔는지 신경 쓰지 않습니다. 이 예제처럼 생성자 인자는 `bind`에 키워드로도 넘길 수 있습니다.

요약-번역 파이프라인에도 같은 기법을 적용할 수 있습니다. 요약을 별도 디플로이먼트로 빼고 인그레스가 요약 응답을 번역 호출에 바로 넘기면, 인그레스가 중간 결과를 받아 다시 보내는 왕복이 사라지고, 최종 응답을 `await`하는 동안에도 다른 요청을 처리합니다.

> **핵심:** `DeploymentResponse`는 최상위 위치 인자나 키워드 인자로만 넘겨야 합니다. 리스트나 딕셔너리 같은 중첩 객체 안에 넣어 넘기는 방식은 완전히 지원 중단됐습니다. 중첩 구조가 꼭 필요하다면 `DeploymentResponse._to_object_ref()`로 얻은 Ray `ObjectRef`를 대신 넣습니다.

### 스트리밍 호출

LLM 토큰 생성처럼 결과를 여러 번 나눠 내는 하위 디플로이먼트도 핸들로 호출할 수 있습니다. 호출할 메서드가 제너레이터여야 하고, 핸들에 `handle.options(stream=True)`를 설정해야 합니다. 그러면 핸들 호출은 단일 `DeploymentResponse` 대신 `DeploymentResponseGenerator`를 반환하며, 이 객체를 동기 또는 비동기 제너레이터로 순회합니다.

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

`Caller`는 생성자에서 `options(stream=True)`를 적용한 핸들을 한 번 만들어 두고, `__call__`에서 `async for`로 값을 받는 대로 다시 `yield`합니다. `Caller` 자신도 제너레이터이므로 드라이버 쪽 핸들에도 `stream=True`를 설정했습니다. 디플로이먼트 안에서는 반드시 `async for`로 순회합니다. 동기 제너레이터로 순회하면 `.result()`와 마찬가지로 그 레플리카에서 다른 요청이 동시에 실행되지 못합니다. 또한 `DeploymentResponseGenerator`는 다른 핸들 호출의 인자로 넘길 수 없어서 스트림은 체이닝 대상이 아닙니다.

하위 스트림을 HTTP 클라이언트까지 흘려보내려면, 인그레스에서 이 비동기 순회를 비동기 제너레이터로 감싸 앞서 설명한 `StreamingResponse`에 넘깁니다. `Streamer`를 그대로 두고 인그레스만 바꾸면 다음과 같습니다.

```python
from starlette.requests import Request
from starlette.responses import StreamingResponse

@serve.deployment
class HttpCaller:
    def __init__(self, streamer: DeploymentHandle):
        self._streamer = streamer.options(stream=True)

    async def _relay(self, limit: int):
        async for i in self._streamer.remote(limit):
            yield str(i)

    def __call__(self, request: Request) -> StreamingResponse:
        limit = int(request.query_params.get("limit", "10"))
        return StreamingResponse(self._relay(limit), media_type="text/plain")

app = HttpCaller.bind(Streamer.bind())
```

`Streamer`가 값을 하나 `yield`할 때마다 `_relay`가 이를 문자열로 바꿔 내보내고, `StreamingResponse`가 그 조각을 바로 클라이언트에 보냅니다.

### Ray 태스크·액터와 섞어 쓰기

`DeploymentResponse`는 내부적으로 Ray `ObjectRef` 하나에 대응하고, 스트리밍 호출이라면 `ObjectRefGenerator`에 대응합니다. Serve 디플로이먼트의 출력을 일반 Ray 태스크나 액터에 넘겨야 할 때는 개발자 API인 `_to_object_ref`(비동기)나 `_to_object_ref_sync`(동기)로 응답을 `ObjectRef`로 바꿉니다.

```python
# File name: response_to_object_ref.py
import ray
from ray import serve
from ray.serve.handle import DeploymentHandle, DeploymentResponse

@ray.remote
def say_hi_task(inp: str):
    return f"Ray task got message: '{inp}'"

@serve.deployment
class SayHi:
    def __call__(self) -> str:
        return "Hi from Serve deployment"

@serve.deployment
class Ingress:
    def __init__(self, say_hi: DeploymentHandle):
        self._say_hi = say_hi

    async def __call__(self):
        # Make a call to the SayHi deployment and pass the result ref to
        # a downstream Ray task.
        response: DeploymentResponse = self._say_hi.remote()
        response_obj_ref: ray.ObjectRef = await response._to_object_ref()
        final_obj_ref: ray.ObjectRef = say_hi_task.remote(response_obj_ref)
        return await final_obj_ref

app = Ingress.bind(SayHi.bind())
handle: DeploymentHandle = serve.run(app)
assert handle.remote().result() == "Ray task got message: 'Hi from Serve deployment'"
```

`ObjectRef`를 태스크에 넘기면 Ray가 값을 풀어 `inp`에 넣으므로, 체이닝과 마찬가지로 인그레스가 중간 결과를 직접 받아 다시 보내지 않습니다. 개발자 API이므로, 핸들끼리만 조합한다면 앞의 체이닝으로 충분합니다.

네 가지 호출 방식을 비교하면 표와 같습니다.

| 호출 방식 | 반환 객체 | 결과를 얻는 방법 | 주의할 점 |
| --- | --- | --- | --- |
| `handle.method.remote(...)` | `DeploymentResponse` | 디플로이먼트 안에서는 `await`, 드라이버에서는 `.result()` | 디플로이먼트 안에서 `.result()`를 쓰면 레플리카가 멈춤 |
| 응답을 다른 핸들 호출에 전달 | `DeploymentResponse` | 마지막 응답만 `await` | 최상위 인자나 키워드 인자로만 전달 |
| `handle.options(stream=True)` 후 호출 | `DeploymentResponseGenerator` | `async for`로 순회 | 디플로이먼트 안에서 동기 순회 금지, 다른 핸들 호출에 전달 불가 |
| `response._to_object_ref()` | `ObjectRef` | Ray 태스크·액터에 전달하거나 `await` | 개발자 API |

### 디플로이먼트별 독립적 스케일링

조합으로 단계를 나누는 가장 큰 이유는 각 디플로이먼트를 따로 설정할 수 있어서입니다. 언어 분류 예제에서 요청의 75%가 스페인어이고 25%가 프랑스어라면, `SpanishResponder`는 레플리카 3개, `FrenchResponder`는 1개로 두어 실제 부하 비율에 맞춥니다.

```python
@serve.deployment(num_replicas=3)
class SpanishResponder:
    def say_hello(self, name: str):
        return f"Hola {name}"


@serve.deployment(num_replicas=1)
class FrenchResponder:
    def say_hello(self, name: str):
        return f"Bonjour {name}"
```

자원 예약도 같은 방식으로 나눕니다. 요약-번역 파이프라인이라면 가벼운 라우팅이나 전처리 디플로이먼트는 CPU 레플리카로 두고, 무거운 모델 디플로이먼트에만 `ray_actor_options`로 GPU를 예약합니다. 자원 종류와 양이 다른 단계를 한 디플로이먼트에 묶었을 때 생기는 애플리케이션 수준의 병목을 이렇게 피합니다.

다만 나누는 데도 비용이 듭니다. 핸들 호출은 매번 레플리카 선택과 프로세스 간 데이터 전달을 거치므로, 아주 작은 단계까지 잘게 쪼개면 계산보다 호출 오버헤드가 커질 수 있습니다. 지연 시간이 엄격한 API라면 단일 디플로이먼트로 오버헤드를 줄이는 편이 낫고, 자원 요구량이나 부하가 확연히 다른 단계에서만 디플로이먼트를 나누는 것이 좋은 출발점입니다. 인그레스에는 HTTP 처리와 라우팅처럼 가벼운 일만 맡기고 모델 로딩과 추론은 하위 디플로이먼트에 두면, 두 계층을 각자의 자원 요구량에 맞춰 늘릴 수 있습니다.

## 요청 처리 구조와 아키텍처

`serve run` 한 줄로 애플리케이션이 뜨는 동안, 클러스터 안에서는 역할이 다른 Ray 액터 여러 개가 함께 시작됩니다. Ray Serve는 Ray 위에서 돌아가며, 서빙 인스턴스를 이루는 구성 요소가 모두 Ray 액터입니다. 레플리카 수, `max_ongoing_requests`, 프록시 위치 같은 설정이 무엇을 바꾸는지 이해하려면 이 액터들의 역할과 요청이 지나가는 경로를 알아 두는 편이 좋습니다.

### 세 종류의 액터: 컨트롤러, 프록시, 레플리카



컨트롤러는 다른 액터를 관리하는 제어 평면을 맡고, 요청은 프록시에서 레플리카로 전달됩니다. 제어 평면과 데이터 경로가 여기서 갈라집니다.

| 구성 요소 | 개수와 위치 | 맡는 일 |
| --- | --- | --- |
| 컨트롤러 | Serve 인스턴스마다 하나인 전역 액터 | 제어 평면 관리, 다른 액터의 생성·갱신·삭제 |
| HTTP 프록시 | 기본값은 헤드 노드에 하나 | Uvicorn HTTP 서버로 요청을 받아 레플리카에 전달하고 완료되면 응답 |
| gRPC 프록시 | 유효한 `port`와 `grpc_servicer_functions`로 시작했을 때 HTTP 프록시와 함께 실행 | grpcio 서버로 요청을 받아 레플리카에 전달하고 완료되면 응답 |
| 레플리카 | 디플로이먼트마다 설정한 개수 | 요청에 대응하는 코드 실행, 필요하면 `@serve.batch`로 배칭 |

디플로이먼트를 만들거나 조회하는 Serve API 호출은 모두 컨트롤러에 원격 호출로 전달되고, 컨트롤러가 그에 맞춰 다른 액터를 만들거나 갱신하거나 없앱니다. 요청 데이터는 컨트롤러를 거치지 않으므로, 컨트롤러가 죽어도 HTTP, gRPC, `DeploymentHandle` 요청은 계속 처리됩니다. 다만 그동안 오토스케일링은 멈추고, 컨트롤러가 복구되면 다시 시작하지만 그전까지 모은 메트릭은 사라집니다.

레플리카는 요청을 개별 단위로 받아 처리하며, 동시에 몇 개를 진행하는지는 핸들러 정의에 따라 다릅니다.

### HTTP 요청의 수명 주기

HTTP나 gRPC 요청이 프록시에 도착하면 다음 순서로 처리됩니다.

1. 프록시가 요청을 받아 파싱합니다.
2. HTTP URL 경로나 애플리케이션 이름 메타데이터로 요청을 처리할 디플로이먼트를 찾고, 요청을 큐에 넣습니다.
3. 큐의 요청마다 여유 있는 레플리카를 찾아 보냅니다. 모든 레플리카가 이미 `max_ongoing_requests`보다 많은 요청을 처리 중이면, 요청은 여유가 생길 때까지 큐에서 기다립니다.

두 번째 단계에서 `route_prefix`가 쓰입니다. 애플리케이션은 지정한 `route_prefix`로 HTTP 호출을 받으므로, 프록시는 URL 경로를 보고 어느 인그레스 디플로이먼트로 보낼지 정합니다. 세 번째 단계의 `max_ongoing_requests`는 레플리카 하나가 동시에 떠안는 요청 수의 상한입니다. 이 상한 덕분에 느린 레플리카 하나에 요청이 쌓이지 않고, 넘치는 요청은 큐에서 기다립니다. 보낼 레플리카는 무작위로 고른 두 후보 가운데 더 한가한 쪽을 택하는 '두 개 중 선택(power of two choices)' 방식으로 정하며, 이때도 상한을 넘는 레플리카는 제외합니다.

레플리카 안에서는 자기 요청 큐에서 요청을 하나씩 꺼내 실행합니다. 핸들러(디플로이먼트 함수나 클래스의 `__call__`)가 `async def`이면 레플리카는 핸들러가 끝나기를 기다리지 않고 `asyncio`로 여러 요청을 동시에 진행합니다. 일반 `def`이면 핸들러가 반환할 때까지 레플리카가 멈춥니다.

```python
import requests
from starlette.requests import Request

from ray import serve

@serve.deployment
class MostBasicIngress:
    async def __call__(self, request: Request) -> str:
        name = (await request.json())["name"]
        return f"Hello {name}!"

app = MostBasicIngress.bind()
serve.run(app)
assert (
    requests.get("http://127.0.0.1:8000/", json={"name": "Corey"}).text
    == "Hello Corey!"
)
```

이 핸들러는 `await request.json()`에서 이벤트 루프에 제어권을 돌려주므로, 본문을 읽는 동안 같은 레플리카가 다른 요청을 시작할 수 있습니다. 한 레플리카가 여러 요청을 동시에 진행하려면 핸들러 안에 이렇게 제어권을 양보하는 지점이 있어야 합니다.

### DeploymentHandle 요청이 지나는 경로

`DeploymentHandle` 호출은 프록시를 거치지 않습니다. 요청이 핸들 안의 큐에 들어간 뒤 곧바로 세 번째 단계, 즉 레플리카 선택으로 넘어갑니다. `DeploymentHandle`은 같은 노드에 있는 라우터의 핸들을 감싸고 있고, 이 라우터가 대상 디플로이먼트의 레플리카로 요청을 보냅니다. 이후 경로는 HTTP나 gRPC 요청과 같아서, 디플로이먼트 선택과 배칭도 똑같이 적용됩니다.

라우터가 레플리카 목록을 아는 방식도 제어 평면과 데이터 경로의 분리를 따릅니다. 각 `DeploymentHandle`은 컨트롤러를 계속 폴링해 새 레플리카를 확인하고, 새 레플리카를 발견하면 버퍼에 쌓인 요청이나 새 요청을 `max_ongoing_requests`에 이를 때까지 그쪽으로 보냅니다. 컨트롤러는 목록만 알려 주고, 전달은 핸들 쪽 라우터가 직접 맡습니다.

### 프록시를 노드마다 두기

기본 구성에서는 HTTP 프록시가 헤드 노드에 하나뿐이라 모든 외부 요청이 이 프록시를 지나갑니다. 확장성과 가용성이 필요하면 `serve.start()`나 Serve 설정 파일의 `proxy_location` 필드로 노드마다 프록시를 하나씩 띄웁니다. `serve build`로 생성한 설정 파일에도 `proxy_location: EveryNode`가 들어 있습니다. 노드별 프록시는 모두 같은 포트에 바인딩되고, 어느 노드로 들어와도 모든 모델에 요청을 보낼 수 있습니다.

이 구성에서는 두 계층을 따로 늘립니다. HTTP·gRPC 인그레스 처리량은 노드를 추가해 늘리고, 모델 추론 처리량은 디플로이먼트의 `num_replicas`로 레플리카를 늘려 확보합니다. 클라이언트에 주소 하나만 노출하고 싶다면 노드별 프록시 앞에 직접 운영하는 로드 밸런서를 둘 수 있습니다.

### 대형 요청 처리

Serve는 Ray의 공유 메모리 객체 저장소와 프로세스 내 메모리 저장소를 함께 씁니다. 작은 요청 객체는 네트워크 호출로 액터 사이에 바로 보냅니다. 100KiB 이상인 큰 요청 객체는 객체 저장소에 기록하고, 레플리카는 이를 제로 카피(zero-copy) 읽기로 가져옵니다.

크기에 따라 경로를 나누는 것은 두 방식의 비용이 달라서입니다. 작은 객체는 저장소에 썼다가 다시 읽는 것보다 메시지에 실어 보내는 편이 빠릅니다. 큰 객체는 매번 직렬화해 복사하면 메모리와 CPU를 그만큼 쓰지만, 공유 메모리에 한 번 써 두면 같은 노드의 레플리카는 복사 없이 읽습니다.

## 오토스케일링과 리소스 설정

디플로이먼트는 기본적으로 레플리카가 하나입니다. 모델을 실행하며 요청을 처리하는 워커 프로세스가 하나뿐이라서, 트래픽이 늘면 이 레플리카가 금방 과부하에 걸립니다. 응답 성능을 유지하려면 레플리카를 늘려 디플로이먼트를 수평 확장해야 합니다. 방법은 두 가지입니다. 사람이 레플리카 수를 직접 바꾸는 수동 스케일링이 있고, 트래픽에 따라 Serve가 레플리카 수를 조절하는 오토스케일링이 있습니다.

### 수동 스케일링

오토스케일링은 더 복잡하므로, 수동 스케일링을 먼저 검토할 만합니다. 디플로이먼트 옵션의 `num_replicas` 값을 올리고 제자리 업데이트(in-place update)로 반영하면 레플리카가 늘어납니다. `num_replicas`의 기본값은 1이며, 값을 올리면 늘어난 트래픽에서 지연 시간과 처리량이 모두 나아집니다. Serve 설정 파일에서는 디플로이먼트 항목의 값만 바꾸면 됩니다.

```yaml
# Deploy with a single replica
deployments:
- name: Model
  num_replicas: 1

# Scale up to 10 replicas
deployments:
- name: Model
  num_replicas: 10
```

### `num_replicas="auto"`로 오토스케일링 켜기

고정된 숫자 대신 `num_replicas="auto"`를 설정하면 디플로이먼트가 들어오는 트래픽에 따라 스스로 늘고 줄어듭니다. Serve 오토스케일러는 큐 크기를 지켜보다가 트래픽이 급증하면 레플리카를 추가하고, 줄어들면 제거합니다. 세부 동작은 디플로이먼트 옵션의 `autoscaling_config`로 조정합니다.

```yaml
- name: Model
  num_replicas: auto
```

이 한 줄은 다음 설정과 같습니다.

```yaml
- name: Model
  max_ongoing_requests: 5
  autoscaling_config:
    target_ongoing_requests: 2
    min_replicas: 1
    max_replicas: 100
```

`num_replicas="auto"`를 쓰면 이 기본값이 적용되고, 그중 일부만 `autoscaling_config`에 지정해 덮어쓸 수도 있습니다. 주의할 점이 있습니다. `num_replicas="auto"` 없이 `autoscaling_config`만 직접 설정하면 `max_replicas`의 기본값이 1이라서, 이 값을 명시적으로 올리지 않는 한 오토스케일링이 일어나지 않습니다.

각 파라미터의 의미와 정하는 기준은 다음과 같습니다.

| 파라미터 | 의미 | 정하는 기준 |
| --- | --- | --- |
| `target_ongoing_requests` | 오토스케일러가 유지하려는 레플리카당 평균 진행 중 요청 수 | 요청 처리 시간이 길수록, 목표 지연 시간이 짧을수록 작게 설정 |
| `max_ongoing_requests` | 레플리카 하나에 허용하는 진행 중 요청 수의 상한 | 오토스케일링을 켰다면 `target_ongoing_requests`를 기준으로 함께 설정 |
| `min_replicas` | 디플로이먼트의 최소 레플리카 수 | 트래픽 없는 기간이 길고 확장 중 꼬리 지연을 감수할 수 있으면 0, 아니면 저부하에 필요한 수 |
| `max_replicas` | 디플로이먼트의 최대 레플리카 수 | 피크 트래픽에 필요하다고 보는 수보다 약 20% 많게 설정 |

`max_ongoing_requests`만 `autoscaling_config` 밖에 있습니다. 이 값은 오토스케일링 여부와 관계없이 모든 디플로이먼트에 해당하기 때문입니다. 오토스케일링을 켰다면 `max_ongoing_requests`를 `target_ongoing_requests`에 맞춰 함께 정합니다.

`min_replicas`를 0으로 두면 트래픽이 없을 때 레플리카가 모두 사라지지만, 다음 요청은 레플리카가 새로 뜰 때까지 기다리므로 확장 중 꼬리 지연이 늘어납니다. 표의 기준은 출발점이며, 더 세밀하게 조정하려면 공식 문서의 고급 오토스케일링 가이드(Advanced Ray Serve Autoscaling)를 참고합니다.

### 예제: ResNet50 디플로이먼트 오토스케일링

ResNet50 추론을 동기식으로 실행하는 디플로이먼트에 오토스케일링을 적용한 예제입니다. 레플리카마다 CPU 1개를 예약하고 `num_replicas="auto"`를 지정했습니다.

```python
import requests
from io import BytesIO

from PIL import Image
import starlette.requests
import torch
from torchvision import transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights

from ray import serve


@serve.deployment(
    ray_actor_options={"num_cpus": 1},
    num_replicas="auto",
)
class Model:
    def __init__(self):
        self.resnet50 = (
            models.resnet50(weights=ResNet50_Weights.DEFAULT).eval().to("cpu")
        )
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        resp = requests.get(
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        )
        self.categories = resp.content.decode("utf-8").split("\n")

    async def __call__(self, request: starlette.requests.Request) -> str:
        uri = (await request.json())["uri"]
        image_bytes = requests.get(uri).content
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Batch size is 1
        input_tensor = torch.cat([self.preprocess(image).unsqueeze(0)]).to("cpu")
        with torch.no_grad():
            output = self.resnet50(input_tensor)
            sm_output = torch.nn.functional.softmax(output[0], dim=0)
        ind = torch.argmax(sm_output)
        return self.categories[ind]


app = Model.bind()
```

같은 설정을 Serve 설정 파일로 지정하면 다음과 같습니다.

```yaml
applications:
- name: default
  import_path: resnet:app
  deployments:
  - name: Model
    num_replicas: auto
```

공식 문서의 예제에서는 이 애플리케이션에 Locust로 부하 테스트를 실행했습니다. 각 가상 사용자는 대기 시간 0으로 요청을 하나 보내고, 응답을 받자마자 다음 요청을 보냅니다. 그래서 사용자 수가 곧 동시 요청 수가 됩니다. 결과를 보면 오토스케일된 레플리카 수가 시간에 따라 사용자 수의 대략 절반으로 움직였습니다. Serve가 `target_ongoing_requests=2`를 맞추려 했기 때문입니다.

다음은 이 관계를 손으로 따라가 본 설명용 계산이며, 측정값이 아닙니다. 사용자가 10명이면 진행 중 요청이 10개이고, 레플리카당 평균 2개를 맞추려면 레플리카가 다섯 개 필요합니다.

$$
\text{replicas} \approx \frac{10 \ \text{ongoing requests}}{2 \ \text{per replica}} = 5
$$

사용자와 레플리카가 늘면서 처리량도 함께 늘었고, 지연 시간은 트래픽이 늘어나는 순간 잠깐 튀었다가 그 밖에는 비교적 일정했습니다. 새 레플리카가 뜨는 동안에는 기존 레플리카가 늘어난 요청을 감당해야 하므로, 이렇게 잠깐 튀는 구간이 생깁니다.

### Serve 오토스케일러와 Ray 오토스케일러의 관계

Serve 오토스케일러는 요청 수요를 보고 레플리카 수를 정하고, 레플리카를 올릴 자원 확보는 클러스터 수준의 Ray 오토스케일러가 맡습니다. Serve 오토스케일러는 그 위에서 동작하는 애플리케이션 수준 오토스케일러이며, 두 계층은 다음과 같이 맞물립니다.

1. Serve 오토스케일러가 요청 수요를 보고 Ray에 레플리카 액터를 몇 개 띄우라고 요청합니다.
2. 이 액터를 배치할 CPU, GPU 같은 자원이 모자라면 Ray 오토스케일러가 Ray 노드를 더 요청합니다.
3. 클라우드 제공자가 그 요청에 따라 노드를 추가합니다.
4. 반대로 Serve가 레플리카를 줄일 때는 유휴 노드가 최대한 많이 생기도록 레플리카 액터를 종료하고, Ray 오토스케일러가 그 노드를 제거합니다.

자원이 충분한지는 레플리카 액터를 배치할 CPU, GPU를 기준으로 따지므로, 레플리카마다 `ray_actor_options`로 예약하는 자원량이 두 계층을 잇는 연결 고리입니다.

## Serve 설정 파일로 프로덕션 준비하기

개발 중에는 `serve run`으로 코드를 고치고 다시 띄우는 과정을 반복하지만, 프로덕션에서는 Serve 설정 파일로 배포하고 갱신합니다. Serve 설정 파일은 클러스터의 단일 기준 정보(single source of truth)이며, Serve와 관련된 모든 설정을 YAML 파일 하나에 적습니다. 애플리케이션은 이 파일을 고쳐 다시 적용하는 선언적 방식으로 갱신합니다. 그래서 애플리케이션을 다시 배포하거나 재시작하지 않고도 개별 디플로이먼트의 파라미터만 바꿀 수 있습니다. VM에서는 `serve deploy` CLI 명령에 이 파일을 넘기고, Kubernetes에서는 같은 내용을 RayService 커스텀 리소스에 넣습니다.

### 예제 애플리케이션: `reconfigure`를 추가한 요약·번역 앱

앞서 만든 요약·번역 파이프라인을 조금 고쳐 `text_ml.py`로 저장합니다. 이 앱은 영어 텍스트를 요약한 뒤 프랑스어(기본값), 독일어, 루마니아어 중 하나로 번역합니다. `translate`의 모델 호출, `summarize`의 디코딩, `Summarizer.__call__`, 마지막의 `app = Summarizer.bind(Translator.bind())`는 앞서 만든 파이프라인과 같습니다. 새로 추가하거나 바뀐 부분만 보면 다음과 같습니다.

```python
from typing import Dict

@serve.deployment
class Translator:
    def __init__(self):
        self.language = "french"
        self.prefix = "translate English to French: "
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

    def translate(self, text: str) -> str:
        input_ids = self.tokenizer(
            f"{self.prefix}{text}", return_tensors="pt"
        ).input_ids
        # ... 생성과 디코딩은 앞서 만든 파이프라인과 같습니다.

    def reconfigure(self, config: Dict):
        self.language = config.get("language", "french")

        if self.language.lower() == "french":
            self.prefix = "translate English to French: "
        elif self.language.lower() == "german":
            self.prefix = "translate English to German: "
        elif self.language.lower() == "romanian":
            self.prefix = "translate English to Romanian: "
        else:
            pass

@serve.deployment
class Summarizer:
    def __init__(self, translator: DeploymentHandle):
        # ... 토크나이저·모델 로드와 self.translator는 앞서와 같습니다.
        self.min_length = 5
        self.max_length = 15

    def summarize(self, text: str) -> str:
        input_ids = self.tokenizer(f"summarize: {text}", return_tensors="pt").input_ids
        output_ids = self.model.generate(
            input_ids,
            num_beams=4,
            early_stopping=True,
            min_length=self.min_length,
            max_length=self.max_length,
        )
        # ... 디코딩은 앞서와 같습니다.

    def reconfigure(self, config: Dict):
        self.min_length = config.get("min_length", 5)
        self.max_length = config.get("max_length", 15)
```

### `serve build`로 설정 파일 생성하기

설정 파일을 손으로 처음부터 쓸 필요는 없습니다. `serve build`에 임포트 경로를 넘기면 코드 안의 모든 디플로이먼트와 파라미터를 담은 파일이 만들어집니다.

```bash
$ serve build text_ml:app -o serve_config.yaml
```

`-o`(`--output-path`)를 생략하면 결과를 표준 출력에 씁니다. 임포트 경로는 `-d`(`--app-dir`)로 지정한 디렉터리에서 찾으며, 기본값은 현재 디렉터리 `.`입니다. 생성된 파일은 다음과 같습니다.

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
```

기본 HTTP·gRPC 옵션과 애플리케이션 이름은 필요에 맞게 고쳐도 됩니다. `runtime_env`는 직접 채워야 합니다. `serve build`는 이 필드를 항상 비워 두므로, `torch`와 `transformers`가 클러스터에 전역으로 설치되어 있지 않다면 pip 패키지로 추가합니다.

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

### YAML 구조 읽기

최상위 필드는 다섯 개입니다.

| 필드 | 설정 대상 | 기본값과 주의점 |
| --- | --- | --- |
| `proxy_location` | 프록시를 띄울 위치 | 기본값 `EveryNode`, 그 밖의 값 `HeadOnly`·`Disabled` |
| `http_options` | HTTP 프록시의 `host`, `port`, `request_timeout_s`, `keep_alive_timeout_s` | 기본값 `0.0.0.0`·`8000`, 요청 타임아웃 없음, 실행 중 변경 불가 |
| `grpc_options` | gRPC 프록시의 `port`, `grpc_servicer_functions`, `request_timeout_s` | 기본 포트 `9000`, 서비서 함수 목록이 비면 gRPC 서버 미시작, 실행 중 변경 불가 |
| `logging_config` | 컨트롤러·프록시·레플리카 로그 | 전역 설정, 애플리케이션·디플로이먼트 수준 설정이 우선 |
| `applications` | 애플리케이션 목록 | 애플리케이션별 필드 목록 |

애플리케이션 항목의 주요 필드는 다음과 같습니다.

- `name`: 애플리케이션 이름입니다. `serve build`가 자동으로 붙이며 애플리케이션마다 고유해야 합니다.
- `route_prefix`: HTTP로 호출할 경로 접두사입니다. 기본값은 `/`이고 역시 고유해야 합니다.
- `import_path`: 최상위 디플로이먼트의 경로로, `serve run`에 넘기던 값과 같습니다. 가장 작은 설정 파일은 이 필드 하나로 이루어집니다.
- `runtime_env`: 애플리케이션의 실행 환경입니다. 지정했다면 `import_path`를 그 환경 안에서 임포트할 수 있어야 하고, `working_dir`와 `py_modules`에는 로컬 경로가 아닌 원격 URI만 씁니다.
- `deployments`: 코드의 `@serve.deployment` 설정을 덮어쓰는 목록입니다. 각 항목의 `name`은 코드의 디플로이먼트 이름과 일치해야 합니다.

이 밖에 외부 스케일링 API를 켜는 `external_scaler_enabled`와 애플리케이션 빌더에 인자를 넘기는 `args`가 있습니다.

`deployments`의 항목은 하나하나가 선택 사항입니다. `Summarizer` 항목을 통째로 지워도 파일은 유효하고, `Summarizer`는 데코레이터에 적은 설정 그대로 배포됩니다. 코드에는 합리적인 기본값을 두고, 운영 중에 바꿀 값만 설정 파일에서 덮어쓰면 됩니다.

### `user_config`와 `reconfigure`로 재시작 없이 바꾸기

`num_replicas`는 Serve가 해석하지만, 번역 대상 언어처럼 애플리케이션만 아는 설정도 있습니다. `user_config`에는 JSON으로 직렬화할 수 있는 임의의 객체를 담습니다. Serve는 이 값을 실행 중인 레플리카와 앞으로 뜰 레플리카에 모두 적용하며, 이때 레플리카를 재시작하지 않습니다. 그래서 다음과 같은 값을 실행 중에 바꿀 수 있습니다.

- 모델 가중치와 버전
- 모델 조합 그래프의 트래픽 분할 비율
- 기능 플래그, A/B 테스트, 하이퍼파라미터

이 기능을 쓰려면 JSON 직렬화 가능한 객체 하나만 인자로 받는 `reconfigure` 메서드를 구현합니다. 데코레이터나 설정 파일에 `user_config`가 있으면 Serve는 `__init__` 직후 `reconfigure`를 호출해 그 값을 넘기고, 이후 설정 파일의 `user_config`를 고쳐 다시 적용할 때마다 또 호출합니다. 예제의 `Translator`는 언어 접두사만 바꾸고, 토크나이저와 모델은 `__init__`에서 불러온 것을 그대로 씁니다. 독일어로 바꾸려면 해당 부분만 고칩니다.

```yaml
  deployments:
  - name: Translator
    num_replicas: 1
    user_config:
      language: german
```

상태를 제대로 갱신하는 일은 코드가 맡습니다. `reconfigure`가 건드리지 않은 속성은 이전 값이 남으므로, 관련된 속성이 함께 바뀌어 상태가 일관되게 유지되도록 작성합니다. `Summarizer`는 `config.get(...)`에 기본값을 주어 키가 빠져도 두 속성이 모두 정해집니다. 반면 예제 `Translator`는 지원하지 않는 언어가 오면 `else: pass`로 넘어가서, `self.language`는 새 값으로 바뀌는데 `self.prefix`는 이전 언어 그대로 남습니다. 값을 먼저 검증한 뒤 두 속성을 함께 바꾸면 이런 어긋남이 생기지 않습니다.

```python
PREFIXES = {
    "french": "translate English to French: ",
    "german": "translate English to German: ",
    "romanian": "translate English to Romanian: ",
}

def reconfigure(self, config: Dict):
    language = config.get("language", "french").lower()
    if language not in PREFIXES:
        raise ValueError(f"Unsupported language: {language}")
    self.language = language
    self.prefix = PREFIXES[language]
```

### `serve deploy`와 `serve status`로 배포하고 확인하기

로컬 Ray 클러스터를 시작하고 설정 파일을 배포합니다.

```bash
# Start a local Ray cluster.
ray start --head

# Deploy the Text ML application to the local Ray cluster.
serve deploy serve_config.yaml
```

```text
2022-08-16 12:51:22,043 SUCC scripts.py:180 --
Sent deploy request successfully!
 * Use `serve status` to check deployments' statuses.
 * Use `serve config` to see the running app's config.
```

`serve deploy`는 실행 중인 클러스터의 대시보드 주소에 REST API 요청을 보냅니다. 대시보드 주소의 기본값은 `http://localhost:8265`이고, 원격 클러스터라면 `-a`(`--address`)나 `RAY_DASHBOARD_ADDRESS`로 지정합니다. `serve deploy`는 설정 파일을 클러스터의 목표 상태로 보고 그대로 맞춥니다. Serve가 이미 실행 중이면 새 설정에 없는 애플리케이션은 제거되므로, 여러 애플리케이션을 운영한다면 모두 같은 파일에 적어 둡니다.

배포 상태는 `serve status`로 확인합니다.

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

애플리케이션이 `RUNNING`이고 두 디플로이먼트가 모두 `HEALTHY`이면 요청을 받을 준비가 끝났습니다. 애플리케이션이 여럿이면 `-n`(`--name`)으로 하나만 골라 봅니다. 요청은 앞서 요약·번역 파이프라인을 테스트한 클라이언트로 `http://127.0.0.1:8000/`에 보냅니다.

애플리케이션을 갱신할 때는 설정 파일을 고친 뒤 `serve deploy`를 다시 실행합니다. 같은 파일을 로컬 테스트용 `serve run serve_config.yaml`에 넘겨도 되지만, 이때는 코드가 바뀌지 않은 기존 애플리케이션을 갱신하지 않습니다.

## Kubernetes에서 운영하기: KubeRay RayService

프로덕션에서 Ray Serve를 실행할 때 권장되는 방식은 Kubernetes 위에서 KubeRay의 RayService 커스텀 리소스를 쓰는 것입니다. 앞서 `serve deploy`로 설정 파일을 적용한 방식도 프로덕션에 쓸 수 있습니다. 다만 이 방식은 이미 떠 있는 Ray 클러스터를 전제로 하고, 그 클러스터가 통째로 죽으면 Ray Serve가 스스로 복구하지 못합니다. 레플리카나 컨트롤러가 죽었을 때 클러스터 안에서 다시 띄우는 것은 모두 클러스터가 살아 있을 때의 이야기입니다. 그 바깥에서 노드와 클러스터를 지켜보는 주체가 필요하며, RayService를 쓰면 KubeRay가 그 일을 맡아 죽은 노드나 클러스터를 복구합니다. RayService는 이 밖에도 헬스 체크(health check), 상태 보고, 장애 복구, 업그레이드 같은 프로덕션 요구 사항을 자동으로 처리합니다.

### RayService가 묶는 것: Ray 클러스터와 Serve 애플리케이션

RayService 커스텀 리소스(CR)는 여러 노드로 이루어진 Ray 클러스터와 그 위에서 돌 Serve 애플리케이션을 Kubernetes 매니페스트 하나에 담습니다. 배포, 업그레이드, 상태 조회를 모두 표준 `kubectl` 명령으로 처리하므로, 이미 Kubernetes로 다른 서비스를 운영하는 팀이라면 Ray Serve도 같은 절차와 도구로 관리하게 됩니다.

동작 모델은 앞서 쓴 선언적 방식과 같습니다. 설정 파일을 클러스터의 단일 기준 정보로 두고 파일을 고쳐 다시 적용했듯이, RayService에서는 원하는 상태를 매니페스트에 적고 KubeRay 오퍼레이터가 실제 클러스터와 애플리케이션을 그 상태에 맞춥니다. 차이는 목표 상태의 범위입니다. `serve deploy`는 애플리케이션만 목표 상태로 다루지만, RayService는 클러스터까지 목표 상태에 포함하므로 클러스터가 죽어도 오퍼레이터가 복구합니다. KubeRay는 무중단 업그레이드(zero-downtime upgrade)도 지원하므로 새 버전으로 바꾸는 동안에도 서비스가 계속 요청을 받습니다.

| 구분 | VM에서 `serve deploy` | Kubernetes에서 RayService |
| --- | --- | --- |
| 배포 단위 | 실행 중인 Ray 클러스터에 적용하는 Serve 설정 파일 | Ray 클러스터와 Serve 설정을 함께 담은 커스텀 리소스 |
| 조작 도구 | Serve CLI | `kubectl` |
| 클러스터 장애 | Ray Serve 자체 복구 불가 | KubeRay의 노드·클러스터 복구 |
| 업그레이드 | 설정 파일 수정 후 `serve deploy` 재실행 | 매니페스트 수정 후 재적용, 무중단 업그레이드 지원 |

### 매니페스트의 구성과 설정 파일 옮기기

RayService 매니페스트는 개념상 두 부분으로 나뉩니다. 하나는 Ray 클러스터 정의입니다. 헤드 노드 파드와 워커 노드 파드를 어떻게 구성할지, 파드가 어떤 컨테이너 이미지로 뜰지를 여기에 적습니다. `serve deploy`를 쓸 때는 `ray start --head`로 사람이 클러스터를 먼저 띄웠지만, RayService에서는 이 정의를 보고 오퍼레이터가 파드를 만듭니다. 다른 하나는 Serve 설정입니다. `serve build`로 만든 Serve 설정 파일 YAML을 Kubernetes 설정에 그대로 복사해 넣으면 됩니다. 정확한 필드 이름과 들여쓰기는 KubeRay의 RayService 문서에 있는 샘플 매니페스트를 출발점으로 삼는 편이 안전합니다.

앞서 만든 `serve_config.yaml`의 각 부분이 어디로 옮겨 가는지 정리하면 다음과 같습니다.

| `serve_config.yaml` 부분 | 매니페스트에서 옮겨 갈 위치 | 처리 방법 |
| --- | --- | --- |
| `applications` 목록(`import_path`, `deployments`, `user_config`) | Serve 설정 부분 | 그대로 복사 |
| `proxy_location`, `http_options` | Serve 설정 부분 | 그대로 복사 |
| `runtime_env`의 `pip` 항목 | 커스텀 이미지의 `requirements.txt` | 이미지 빌드 시 미리 설치 |
| 애플리케이션 코드(`text_ml.py`) | 커스텀 이미지의 작업 디렉터리 | Dockerfile에서 복사 |
| 대응 항목 없음 | Ray 클러스터 정의 부분 | 헤드·워커 파드 구성과 이미지 새로 작성 |

Serve 설정을 그대로 복사하므로 `user_config`로 번역 언어를 바꾸는 작업도 Kubernetes에서 똑같이 이루어집니다. 매니페스트 안의 값을 고쳐 다시 적용하면 됩니다. 반대로 코드와 의존성이 이미지로 옮겨 가면 `import_path`가 이미지 안의 작업 디렉터리 기준으로 임포트되어야 합니다. 이미지에 코드를 넣는 위치와 `import_path`를 함께 맞추는 데 주의합니다.

### 커스텀 이미지로 의존성 담기

번역·요약 예제는 `torch`와 `transformers`가 필요해서, 앞서 만든 설정 파일에서는 `runtime_env`의 `pip` 항목에 두 패키지를 적었습니다. Kubernetes에서는 의존성을 컨테이너 이미지에 미리 넣어 두는 방법도 흔히 씁니다. 공식 Ray Docker 이미지는 Docker Hub의 `rayproject` 조직이 관리하며, 이 이미지를 베이스로 애플리케이션 의존성과 코드를 얹은 이미지를 만듭니다.

두 방법은 기동 속도와 유연성 가운데 무엇이 중요한지에 따라 고릅니다. `runtime_env`로 패키지를 지정하면 이미지를 다시 빌드하지 않고 설정만 바꿔 의존성을 갈아 끼울 수 있습니다. 대신 클러스터가 그 환경을 준비하는 동안 기동이 늦어지고, 패키지 저장소에 문제가 생기면 배포도 영향을 받습니다. 의존성을 이미지에 넣어 두면 파드가 뜨는 시점에 필요한 것이 이미 모두 들어 있어서 기동이 예측 가능하고, 어떤 버전이 배포됐는지가 이미지 태그 하나로 정해집니다. 무거운 딥러닝 라이브러리를 쓰는 추론 서비스라면 이미지에 넣는 편이 대체로 다루기 쉽습니다.

다음은 공식 Ray 이미지를 확장해 요구 패키지와 앱 코드를 넣는 Dockerfile 예입니다.

```dockerfile
# pull official base image
FROM rayproject/ray:nightly-py310-cpu

# install requirements.txt
COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt

# set work directory
WORKDIR /serve_app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Copy App folder into workdir
COPY app /serve_app/app
```

`requirements.txt`에 `torch`와 `transformers`를 적어 두면, 이 이미지로 뜬 파드에서는 `runtime_env`에 같은 패키지를 다시 적을 필요가 없습니다. 예제의 베이스 이미지는 nightly 태그를 썼지만, 프로덕션에서는 버전이 고정된 태그를 골라야 같은 이미지를 다시 빌드해도 결과가 같습니다. 로컬 Kubernetes 클러스터로 시험할 때는 이미지를 레지스트리에 올리지 않고 클러스터에 직접 불러와도 됩니다. Docker 컨테이너로 Kubernetes 클러스터를 띄우는 Kind가 이 방식을 지원합니다.

### 배포와 서비스 구성

RayService를 쓰려면 먼저 클러스터에 KubeRay 오퍼레이터와 CRD를 설치해야 하며, 보통 Helm 차트로 설치합니다. 오퍼레이터가 뜬 뒤에는 커스텀 이미지와 Serve 설정을 담은 RayService 매니페스트를 적용하고 상태를 확인합니다. 다음 예에서 매니페스트 파일 이름은 `custom.yaml`, RayService 이름은 `rayservice-sample`입니다.

```bash
# Apply the RayService manifest.
kubectl apply -f custom.yaml

# Check health and readiness of the RayService.
kubectl get rayservice
kubectl describe rayservice rayservice-sample
```

Serve 애플리케이션이 정상 상태가 되면 KubeRay가 Kubernetes 서비스 두 개를 자동으로 만듭니다. 이름은 RayService 이름 뒤에 접미사를 붙인 형태입니다.

| 서비스 | 이름 | 용도 |
| --- | --- | --- |
| 헤드 서비스 | `rayservice-sample-head-svc` | 클러스터 관리와 Ray 대시보드 |
| Serve 서비스 | `rayservice-sample-serve-svc` | 애플리케이션 쿼리와 요청 |

이 서비스들은 애플리케이션이 정상이 된 뒤에야 만들어지므로, 배포가 끝났는지 판단하는 간단한 기준이 됩니다. `kubectl describe` 출력에서 애플리케이션 상태를 확인한 다음 `kubectl get svc`로 두 서비스가 목록에 생겼는지 봅니다. 서비스가 보이지 않으면 아직 애플리케이션이 정상 상태에 이르지 못했으니 `describe` 출력의 상태와 이벤트를 다시 살핍니다.

두 서비스를 나눈 이유는 트래픽 성격이 다르기 때문입니다. 헤드 서비스는 헤드 노드를 가리키는 관리용 창구이고, Serve 서비스는 사용자의 추론 요청을 받는 창구입니다. Serve 서비스는 상태를 반영해 트래픽을 보내므로 업그레이드 중이나 장애 상황에서도 요청이 정상 파드로 갑니다. 앞서 설명한 노드별 프록시 덕분에 Serve 서비스가 어느 파드로 요청을 보내도 모든 모델에 닿습니다. 그래서 외부 요청은 항상 Serve 서비스를 거치게 구성하는 편이 좋습니다.

개발 중에는 포트 포워딩으로 두 서비스에 접근합니다.

```bash
# Forward application requests
kubectl port-forward service/rayservice-sample-serve-svc 8000

# Forward dashboard access
kubectl port-forward svc/rayservice-sample-head-svc --address 0.0.0.0 8265:8265
```

포워딩을 켜 두면 앞서 요약·번역 파이프라인을 테스트한 클라이언트를 고치지 않고 `http://127.0.0.1:8000/`으로 요청을 보낼 수 있습니다. 로컬에서 `serve deploy`로 띄웠을 때처럼 번역된 문장이 돌아오면 RayService 배포가 제대로 된 것입니다. Ray 대시보드는 `http://localhost:8265/#/serve`에서 열립니다.

### EKS로 옮기기와 인그레스

로컬에서 검증한 RayService는 AWS EKS에서도 같은 매니페스트로 배포합니다. 달라지는 부분은 클라우드 자원, 네트워크, 권한입니다. 먼저 EKS 요구 사항에 맞게 퍼블릭·프라이빗 서브넷을 갖춘 VPC를 만들고, EKS 관리형 정책을 붙인 클러스터 IAM 역할을 준비한 뒤 클러스터를 생성합니다. EKS가 사용자를 대신해 AWS 자원을 다루므로 필요한 서비스 역할이 모두 설정되어 있어야 합니다. 이어 로컬 kubeconfig를 EKS 클러스터에 연결하고 `kubectl`이 클러스터에 닿는지 확인합니다.

EKS의 노드 유형은 Fargate와 관리형 노드 두 가지이며, Ray 워크로드에는 제어권과 성능 면에서 관리형 노드 그룹이 낫습니다. GPU 작업은 Ray 헤드 노드에서 돌리지 않는 것이 좋습니다. 헤드 노드에는 컨트롤러의 상태를 체크포인트하는 GCS가 있고 기본 구성의 HTTP 프록시도 떠 있어서, 무거운 추론이 같은 노드의 자원을 놓고 다투면 클러스터 전체가 영향을 받습니다. 추론은 워커 노드 그룹에 맡기고 헤드 노드는 관리 업무에 집중하게 나눕니다.

포트 포워딩은 개발용일 뿐이고, 프로덕션에서는 Kubernetes 인그레스로 외부 접근을 엽니다. EKS에서는 AWS Load Balancer Controller를 설치해 인그레스 리소스가 Application Load Balancer(ALB)로 만들어지게 합니다. 이 컨트롤러는 워커 노드에서 실행되며 ALB·NLB API에 접근할 IAM 권한이 필요합니다. 인그레스를 구성할 때 지켜야 할 요구 사항은 다음과 같습니다.

- 백엔드는 Serve 서비스와 그 HTTP 포트 `8000`으로 지정합니다. 상태를 반영해 트래픽을 나누는 서비스이므로 업그레이드나 장애 중에도 가용성이 유지됩니다.
- ALB에 헬스 체크를 설정합니다. ALB는 헬스 체크를 통과한 파드에만 트래픽을 보냅니다. 헬스 체크 프로토콜, 포트, 경로를 지정하는 어노테이션은 KubeRay 인그레스 문서의 AWS ALB 예제에 나와 있으니 그 예제를 그대로 가져다 쓰는 편이 안전합니다.
- 서브넷은 서로 다른 가용 영역의 퍼블릭 서브넷을 두 개 이상 지정하고, 가용 영역 하나에는 서브넷을 하나만 둡니다. 퍼블릭 서브넷은 퍼블릭 IPv4 주소 자동 할당이 켜져 있어야 합니다.

인그레스를 적용하고 나면 AWS 콘솔의 로드 밸런서 목록에 `k8s-default-`로 시작하는 ALB가 보이고, 외부 클라이언트는 이 ALB의 DNS 이름으로 Ray Serve 애플리케이션을 호출합니다. 요청은 ALB에서 Serve 서비스를 거쳐 각 노드의 프록시로 들어가고, 거기서부터는 로컬에서 본 요청 수명 주기를 그대로 따릅니다.

## 장애 허용성과 복구

온라인 추론 서비스에서는 프로세스, 노드, 클러스터 수준의 장애가 언젠가는 일어납니다. Ray Serve에는 기본 장애 허용성(fault tolerance) 기능이 들어 있습니다. 종단 간 장애 허용성까지 갖추려면 이 기능을 조정하고 Serve를 KubeRay 위에서 실행합니다.

먼저 애플리케이션 오류와 시스템 장애를 구분합니다. 모델 평가 코드에서 던진 예외는 Serve가 잡아서 트레이스백과 함께 500 상태 코드로 돌려주고, 레플리카는 다음 요청을 계속 처리합니다. 액터나 노드가 죽는 시스템 장애는 장애 지점마다 복구 주체가 다릅니다.

| 장애 지점 | 복구 주체 | 복구 중 서비스 상태 |
| --- | --- | --- |
| 레플리카 액터 | 컨트롤러가 새 레플리카로 교체 | 나머지 레플리카가 요청 처리 |
| 프록시 액터 | 컨트롤러가 재시작 | 다른 프록시가 요청 수신 |
| 컨트롤러 액터 | Ray가 재시작 | 요청 처리는 계속, 헬스 체크와 오토스케일링은 중단 |
| 워커 노드 | Serve가 액터를 다른 노드에 다시 띄우고 KubeRay가 노드 교체 | 정상 노드의 레플리카가 요청 처리 |
| 헤드 노드 | KubeRay, GCS 장애 허용성이 있으면 워커 유지 | GCS 장애 허용성이 없으면 전체 워커 재시작으로 중단 |
| Ray 클러스터 전체 | KubeRay만 복구 가능 | KubeRay 없이는 복구 불가 |

컨트롤러의 데이터(라우팅 정책, 디플로이먼트 설정 등)는 헤드 노드의 GCS에 체크포인트되므로 컨트롤러가 다시 떠도 상태가 이어집니다. 반면 액터를 호스팅하던 머신이 죽으면 라우터와 레플리카가 들고 있던 네트워크 연결이나 내부 요청 큐 같은 일시적 데이터는 사라집니다. 또 컨트롤러가 복구되면 오토스케일링은 다시 동작하지만, 이전에 수집한 메트릭은 모두 사라집니다.

### 레플리카 헬스 체크

컨트롤러는 기본적으로 각 레플리카를 주기적으로 헬스 체크하고, 실패하면 재시작합니다. 사용자 코드가 별도 스레드에서 실행되고(`RAY_SERVE_RUN_USER_CODE_IN_SEPARATE_THREAD=1`일 때의 기본 동작) 사용자 정의 `check_health`가 없으면, Serve는 사용자 코드의 이벤트 루프를 감시하는 워치독도 돌립니다. 오래 걸리는 동기 호출이 루프를 막아 탐침이 설정한 횟수만큼 연속해서 시간 초과되면, 그 레플리카는 다음 헬스 체크에서 곧바로 실패하고 재시작됩니다.

| 환경 변수 | 기본값 | 의미 |
| --- | --- | --- |
| `RAY_SERVE_USER_HEALTH_CHECK_PROBE_INTERVAL_S` | `60` | 워치독의 탐침 주기(초) |
| `RAY_SERVE_USER_HEALTH_CHECK_PROBE_TIMEOUT_S` | `300` | 탐침 하나를 실패로 보기까지 기다리는 시간(초) |
| `RAY_SERVE_USER_HEALTH_CHECK_PROBE_MAX_FAIL` | `3` | 헬스 체크 실패로 판정하는 연속 실패 횟수, `0`이면 워치독 비활성화 |

프로세스는 살아 있어도 외부 데이터베이스 연결이 끊기면 그 레플리카는 쓸모가 없습니다. 이런 경우는 애플리케이션 수준의 헬스 체크로 잡습니다. 디플로이먼트 클래스에 인자도 반환값도 없는 `check_health` 메서드를 추가하고, 비정상이면 예외를 던집니다. 실행 주기와 시간 초과는 디플로이먼트 옵션으로 정합니다.

```python
@serve.deployment(health_check_period_s=10, health_check_timeout_s=30)
class MyDeployment:
    def __init__(self, db_addr: str):
        self._my_db_connection = connect_to_db(db_addr)

    def __call__(self, request):
        return self._do_something_cool()

    # Called by Serve to check the replica's health.
    def check_health(self):
        if not self._my_db_connection.is_connected():
            # The specific type of exception is not important.
            raise RuntimeError("uh-oh, DB connection is broken.")
```

`check_health`가 예외를 던지면 컨트롤러가 예외를 로그에 남기고 그 레플리카를 종료한 뒤 다시 띄웁니다. 컨트롤러는 헬스 체크를 레플리카마다 따로 실행합니다. 이 메서드는 컨트롤러용 인터페이스라서 `DeploymentHandle`로 직접 부르면 임의의 레플리카 하나에서만 실행되므로, 그렇게 쓰지 않습니다. 모델 조합에서는 디플로이먼트마다 자기 상태만 책임지므로, 자식 레플리카가 실패해도 부모는 재시작되지 않고 부모의 핸들이 복구된 레플리카로 요청을 보냅니다.

생성자 `__init__`이 실패하면 Serve는 레플리카 생성을 자동으로 다시 시도합니다. 디플로이먼트 단위 상한은 `max_constructor_retry_count`(기본값 `20`)이고, 레플리카 단위 상한은 `RAY_SERVE_MAX_PER_REPLICA_RETRY_COUNT` 환경 변수(기본값 `3`)입니다. 레플리카 수에 레플리카 단위 상한을 곱한 값과 `max_constructor_retry_count` 가운데 작은 쪽이 디플로이먼트 전체의 최대 재시도 횟수입니다.

$$
\min(\text{num\_replicas} \times \text{RAY\_SERVE\_MAX\_PER\_REPLICA\_RETRY\_COUNT},\ \text{max\_constructor\_retry\_count})
$$

```python
# Set max constructor retries for this deployment
@serve.deployment(max_constructor_retry_count=10)
class MyDeployment:
    ...
```

### 헤드 노드 복구: GCS 장애 허용성

기본 구성에서는 헤드 노드가 단일 장애 지점이라, RayService 컨트롤러가 클러스터를 다시 띄우는 동안 서비스가 중단됩니다. KubeRay의 GCS 장애 허용성을 켜면 외부 Redis를 활용해 헤드 노드가 죽어도 클러스터가 유지됩니다. 다만 헤드 노드가 복구되는 동안에는 애플리케이션을 갱신할 수 없고, 액터나 워커 노드가 추가로 죽어도 복구되지 않습니다.

먼저 외부 Redis를 준비합니다. 프로덕션에는 고가용성 Redis를 쓰고, 개발용으로는 같은 Kubernetes 클러스터에 단일 노드 Redis를 띄워도 됩니다. 이어 RayService에 어노테이션과 Redis 주소를 넣습니다.

```yaml
apiVersion: ray.io/v1alpha1
kind: RayService
metadata:
  name: rayservice-sample
  annotations:
    ray.io/ft-enabled: "true"                 # 필수: GCS 장애 허용성 활성화
    ray.io/external-storage-namespace: "my-raycluster-storage-namespace"  # 선택
spec:
  # ...
  rayClusterConfig:
    headGroupSpec:
      # ...
      template:
        spec:
          # ...
          env:
            # ...
            # Redis의 redis:// 주소(호스트와 포트).
            # 여기서 redis는 클러스터 안의 Redis Deployment 이름(호스트), 6379는 포트
            - name: RAY_REDIS_ADDRESS
              value: redis:6379
```

복구 속도에 관여하는 환경 변수도 있으며, Ray를 시작하기 전에 설정합니다.

| 환경 변수 | 기본값 | 효과 |
| --- | --- | --- |
| `RAY_SERVE_KV_TIMEOUT_S` | 없음(시간 초과 없음) | GCS KV 조회·갱신이 멈추지 않고 시간 초과로 빨리 실패해 GCS 장애를 일찍 감지 |
| `LISTEN_FOR_CHANGE_REQUEST_TIMEOUT_S_LOWER_BOUND` / `_UPPER_BOUND` | `30` / `60`초 | 롱 폴링 시간 초과 범위, 줄이면 컨트롤러 재시작 후 반영이 빨라지고 늘리면 폴링 요청 감소 |

```bash
export RAY_SERVE_KV_TIMEOUT_S=5
```

### 레플리카를 노드에 분산하기

노드가 몇 개 죽어도 서비스를 이어 가려면 레플리카가 여러 노드에 흩어져 있어야 합니다. Serve의 기본 분산은 최선 노력이라 고르게 퍼진다는 보장이 없고, 새 노드보다 기존 노드를 선호해서 자원이 충분한 큰 단일 노드 클러스터라면 모든 레플리카가 그 노드에 몰립니다. `max_replicas_per_node`는 노드 하나에 올라갈 레플리카 수에 엄격한 상한을 둡니다.

```python
import ray
from ray import serve

@serve.deployment(max_replicas_per_node=1)
class Deployment1:
    def __call__(self, request):
        return "hello"

@serve.deployment(max_replicas_per_node=2)
class Deployment2:
    def __call__(self, request):
        return "world"
```

레플리카를 두 개씩 띄우면 Serve는 노드가 최소 두 개인 클러스터로 운영하며 `Deployment1`을 노드마다 하나씩 둡니다. `Deployment2`의 두 레플리카는 한 노드에 모여도, 나뉘어도 제약을 만족합니다. 이 옵션은 같은 디플로이먼트의 레플리카끼리만 분산합니다. 값을 1로 두면 레플리카를 노드마다 하나씩 엄격하게 분산합니다.

### 장애 시뮬레이션으로 복구 확인하기

복구 절차는 직접 장애를 일으켜 확인하는 편이 가장 확실합니다. 처리한 레플리카의 PID를 돌려주는 앱을 레플리카 여섯 개, `numCpus: 0`으로 RayService에 배포합니다.

```python
# File name: sleepy_pid.py
from ray import serve

@serve.deployment
class SleepyPid:
    def __init__(self):
        import time
        time.sleep(10)

    def __call__(self) -> int:
        import os
        return os.getpid()

app = SleepyPid.bind()
```

여러 번 요청하면 서로 다른 PID가 돌아오므로 어느 레플리카가 응답했는지 구분됩니다. 다음 순서로 장애를 일으켜 봅니다. 워커 장애는 포트 포워딩하지 않은 워커 파드를 지우고, 헤드 장애는 헤드와 다른 노드의 워커로 포워딩한 뒤 헤드 파드를 지웁니다. 그래야 복구 중에도 응답을 계속 확인합니다. GCS 장애 허용성이 꺼져 있으면 KubeRay가 모든 워커 파드를 재시작하므로 워커가 다시 뜨고 디플로이먼트가 다시 초기화될 때까지 기다려야 합니다. 노드를 지우는 시험은 복구가 더 오래 걸려 파드 수준이 빠릅니다. 액터 장애는 파드에 `exec`해 Ray State API로 액터 목록을 보고, 목록의 `NAME`으로 핸들을 얻어 종료합니다.

```python
import ray

controller_handle = ray.get_actor("SERVE_CONTROLLER_ACTOR", namespace="serve")
ray.kill(controller_handle, no_restart=True)
```

목록을 다시 보면 기존 컨트롤러는 `DEAD`, 새 컨트롤러는 `ALIVE`이고, 그동안에도 요청은 처리됩니다. 레플리카(`no_restart=True`)와 프록시(`no_restart=False`)도 같은 방식으로 이름만 바꿔 종료하며, 복구 중에는 나머지 레플리카와 프록시가 요청을 받습니다.

## 모니터링과 관측성

서빙 중인 애플리케이션을 운영하려면 세 가지를 알아야 합니다. 원하는 상태에 도달했는지, 요청 하나하나에 무슨 일이 있었는지, 처리량과 지연 시간이 시간에 따라 어떻게 변하는지입니다. Ray Serve는 질문마다 다른 도구를 씁니다. 상태는 Ray 대시보드와 `serve status`로, 개별 사건은 로그로, 시간 흐름은 Prometheus 형식 메트릭으로 확인합니다. 셋은 서로 대체하지 못하므로 관측성(observability)을 갖추려면 모두 준비해야 합니다.

### Ray 대시보드의 Serve 화면

대시보드는 대시보드 구성 요소가 포함된 설치 옵션으로 Ray를 설치해야 쓸 수 있습니다.

```bash
pip install "ray[default]"
```

노트북에서 단일 노드 클러스터를 시작하면 Ray가 초기화하면서 출력하는 URL(기본값 `http://localhost:8265`)로 접속하고, `ray.init()`이 돌려주는 컨텍스트 객체로도 주소를 얻습니다.

```python
import ray

context = ray.init()
print(context.dashboard_url)
```

Docker 컨테이너에서는 `ray start --head --dashboard-host=0.0.0.0`처럼 `--dashboard-host`를 반드시 지정합니다. KubeRay 같은 원격 클러스터에서는 대시보드가 헤드 노드에서 뜨지만 대시보드 포트가 외부에 노출되지 않을 수 있으므로, 헤드 노드 밖에서 접근하는 방법은 대시보드 설정 문서를 따릅니다.

Serve 애플리케이션은 대시보드의 Serve 화면에서 봅니다. 전반적인 Serve 설정과 애플리케이션 목록이 나오고, 애플리케이션 이름을 누르면 설정과 메타데이터, 디플로이먼트 목록이 있는 상세 페이지로 들어갑니다. 디플로이먼트를 펼치면 레플리카가 보입니다. 디플로이먼트마다 설정 보기와 Grafana 대시보드 열기가, 레플리카마다 로그 보기와 레플리카 단위 Grafana 대시보드 열기가 있습니다. 레플리카 이름을 누르면 메타데이터와 그 레플리카가 처리한 태스크 이력이 나옵니다.

시계열 그래프는 모두 Prometheus와 Grafana가 있어야 동작합니다. Serve 화면의 상위 수준 메트릭, Grafana 대시보드 링크, 클러스터 전체의 메트릭 화면이 여기에 해당합니다. 대시보드만 설치하면 구조와 상태는 보여도 추이는 보이지 않으므로 둘을 함께 구성합니다. 메트릭 화면의 구성 요소별 CPU·메모리 그래프는 레플리카가 예약한 양보다 실제로 많이 쓰는지 확인할 때 씁니다. 예약값은 스케줄링에 쓰이고 실제 사용량과 늘 일치하지는 않기 때문입니다. 노드와 워커 프로세스 구성은 클러스터 화면에서, 노드별 로그 파일 검색은 로그 화면에서 봅니다.

### `serve config`와 `serve status`로 상태 읽기

프로덕션에서 애플리케이션을 점검하는 CLI 명령은 `serve config`와 `serve status`이며, 원격 클러스터라면 둘 다 `--address`/`-a`로 대상을 지정합니다.

```bash
# 클러스터가 마지막으로 받은 Serve 설정 파일(목표 상태)을 출력합니다.
serve config

# 프록시와 애플리케이션의 현재 상태를 출력합니다.
serve status
```

두 명령은 목표 상태와 현재 상태라는 짝을 이룹니다. `serve config`가 돌려주는 설정 파일이 목표 상태이고, 클러스터는 디플로이먼트를 배포하고 실패한 레플리카를 복구하면서 이 상태를 유지하려고 계속 움직입니다. `serve status`는 그 진행 상황을 보여 줍니다. 배포 후 기대와 다르게 동작하면 먼저 `serve config`로 클러스터가 받은 설정이 의도한 파일과 같은지 보고, 그다음 `serve status`로 어느 단계에서 멈췄는지 봅니다.

`serve status` 출력은 `proxies`와 `applications` 두 부분입니다. 프록시는 실행 중인 노드의 ID로 구분됩니다. 애플리케이션 항목에는 전체 `status`, 설명인 `message`, 마지막 `serve deploy` 요청을 받은 시각, 디플로이먼트별 상태가 있고, 디플로이먼트 항목은 `status`, 상태별 레플리카 수인 `replica_states`, `message`로 이루어집니다.

| 대상 | 상태 값 | 해석 |
| --- | --- | --- |
| 프록시 | `STARTING`, `HEALTHY` | 시작 중이거나 정상적으로 요청을 처리하는 상태입니다. |
| 프록시 | `UNHEALTHY` | 헬스 체크에 실패한 상태이며, 종료된 뒤 같은 노드에 새 프록시가 뜹니다. |
| 프록시 | `DRAINING`, `DRAINED` | 새 요청을 받지 않는 상태이며, 처리 중인 요청이 남아 있으면 `DRAINING`입니다. |
| 애플리케이션 | `NOT_STARTED`, `DEPLOYING`, `RUNNING` | 배포 전, `serve deploy` 요청 처리 중, 정상 상태 유지 중입니다. |
| 애플리케이션 | `DEPLOY_FAILED` | 가장 최근 `serve deploy` 요청이 실패한 상태입니다. |
| 디플로이먼트 | `UPDATING`, `HEALTHY` | 목표 상태로 갱신 중이거나 목표 레플리카 수로 정상 실행 중입니다. |
| 디플로이먼트 | `UPSCALING`, `DOWNSCALING` | 오토스케일링이 켜진 디플로이먼트가 레플리카 수를 늘리거나 줄이는 중입니다. |
| 디플로이먼트 | `UNHEALTHY` | 갱신 뒤 비정상이 된 상태이며, 확장 실패, 헬스 체크 실패, 시스템 오류가 원인일 수 있습니다. |
| 디플로이먼트 | `DEPLOY_FAILED` | 시작이나 갱신에 실패한 상태이며, 대개 생성자 오류가 원인입니다. |
| 레플리카 | `STARTING`, `RUNNING`, `STOPPING` | 시작 중, 정상 처리 중, 종료 중입니다. |
| 레플리카 | `UPDATING`, `RECOVERING` | `reconfigure` 갱신 중이거나 상태를 복구하는 중입니다. |

디플로이먼트의 `UNHEALTHY`와 `DEPLOY_FAILED`는 원인을 찾는 방향이 다릅니다. `DEPLOY_FAILED`라면 처음부터 뜨지 못했으므로 생성자 코드를 먼저 의심하고, `UNHEALTHY`라면 갱신 뒤에 망가졌으므로 레플리카 확장 실패나 헬스 체크 실패, 시스템 오류를 봅니다. 어느 쪽이든 `message` 필드가 첫 단서입니다.

파이썬에서는 `serve.status()`가 같은 정보를 데이터 클래스로 돌려주므로 디플로이먼트나 드라이버 스크립트 안에서 실시간 상태를 읽습니다. 다음 `monitoring_app`은 `RUNNING` 상태인 애플리케이션 이름을 모두 돌려줍니다.

```python
from typing import List, Dict

from ray import serve
from ray.serve.schema import ServeStatus, ApplicationStatusOverview


@serve.deployment
def get_healthy_apps() -> List[str]:
    serve_status: ServeStatus = serve.status()
    app_statuses: Dict[str, ApplicationStatusOverview] = serve_status.applications

    running_apps = []
    for app_name, app_status in app_statuses.items():
        if app_status.status == "RUNNING":
            running_apps.append(app_name)

    return running_apps


monitoring_app = get_healthy_apps.bind()
```

### 로깅과 Loki로 로그 거르기

디플로이먼트 코드에서는 `"ray.serve"` 로거로 로그를 남기고, Serve 로그는 노드의 `/tmp/ray/session_latest/logs/serve/` 아래에 쌓입니다. 레플리카가 여러 노드에 흩어지면 로그도 흩어지므로, 특정 디플로이먼트의 로그만 모아 보려면 로그 수집기가 필요합니다. Grafana Labs의 Loki와 Promtail을 쓰면 Promtail이 로그 파일을 읽어 Loki로 보내고, Grafana에서 LogQL로 질의합니다. Kubernetes에서는 설정이 간단하지만 원리를 익히려면 로컬에서 직접 구성해 봅니다.

Grafana 웹사이트의 안내대로 Loki와 Promtail 실행 파일을 같은 디렉터리에 받고, 다음 파일을 `promtail-local-config.yaml`로 저장합니다.

```yaml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://localhost:3100/loki/api/v1/push

scrape_configs:
- job_name: ray
  static_configs:
    - labels:
        job: ray
        __path__: /tmp/ray/session_latest/logs/serve/*.*
```

Ray Serve와 관련된 부분은 `static_configs`의 `__path__`입니다. `*.*`는 모든 파일에 일치하지만 디렉터리에는 일치하지 않는데, 디렉터리를 넘기면 Promtail이 오류를 내기 때문입니다. Loki를 기본 설정 파일로 실행하고 이 설정 파일로 Promtail을 실행한 뒤, 로그를 남기는 디플로이먼트에 요청을 보냅니다.

```python
from ray import serve
import logging
import requests

logger = logging.getLogger("ray.serve")


@serve.deployment
class Counter:
    def __init__(self):
        self.count = 0

    def __call__(self, request):
        self.count += 1
        logger.info(f"count: {self.count}")
        return {"count": self.count}


counter = Counter.bind()
serve.run(counter)

for i in range(10):
    requests.get("http://127.0.0.1:8000/")
```

Grafana를 실행해 `http://localhost:3000`에 기본 계정(사용자 이름 `admin`, 비밀번호 `admin`)으로 로그인하고 Loki를 데이터 소스로 추가합니다. 왼쪽 패널의 Explore에서 다음 쿼리를 실행합니다.

```text
{job = "ray"} |= "Counter"
```

필터에 걸리는 줄만 남으므로, 노드마다 흩어진 파일을 열지 않고도 `Counter`와 관련된 로그를 한 화면에서 훑어봅니다.

### 내장 메트릭과 Prometheus

Ray Serve는 성공·실패 요청 수 같은 시스템 메트릭을 Ray 메트릭 인프라로 내보내며, 기본적으로 노드마다 Prometheus 형식으로 노출합니다. Prometheus로 이 엔드포인트를 수집하는 설정은 Ray 메트릭 문서(Collecting and monitoring metrics)를 따릅니다. 수집이 되면 Serve 화면에서 메트릭을 보고, Grafana 대시보드에서는 경로, 디플로이먼트, 레플리카 단위로 걸러 봅니다.

먼저 요청 경로에 따라 수집되는 메트릭이 다르다는 점을 알아야 합니다. HTTP·gRPC 프록시로 들어온 요청과 파이썬 `DeploymentHandle` 호출은 잡히는 지표가 다릅니다. 예를 들어 `ray_serve_http_request_latency_ms`는 프록시가 요청을 받은 시점부터 응답을 보낼 때까지를 재므로, 모델 조합에서 인그레스 아래의 하위 호출은 이 지표에 따로 잡히지 않습니다. 메트릭이 많으므로 질문별 시작점을 정해 둡니다.

| 확인할 질문 | 메트릭 | 유형 |
| --- | --- | --- |
| 외부 요청의 종단 간 지연 | `ray_serve_http_request_latency_ms` | Histogram |
| 오류가 난 디플로이먼트 | `ray_serve_num_deployment_http_error_requests_total` | Counter |
| 프록시가 처리 중인 요청 수 | `ray_serve_num_ongoing_http_requests` | Gauge |
| 레플리카 재시작 반복 여부 | `ray_serve_deployment_replica_starts_total` | Counter |
| 콜드 스타트 지연 원인 | `ray_serve_replica_startup_latency_ms`, `ray_serve_replica_initialization_latency_ms` | Histogram |
| 헬스 체크 실패 조짐 | `ray_serve_health_check_failures_total` | Counter |
| 오토스케일러의 목표 추종 여부 | `ray_serve_autoscaling_target_replicas`, `ray_serve_autoscaling_total_requests` | Gauge |
| 디플로이먼트 상태 변화 시점 | `ray_serve_deployment_status` | Gauge |
| 핸들러의 이벤트 루프 블로킹 여부 | `ray_serve_event_loop_scheduling_latency_ms` | Histogram |

몇몇 메트릭은 조합해야 의미가 생깁니다. `ray_serve_replica_startup_latency_ms`는 필요한 경우의 노드 준비, 런타임 환경 구성(pip 설치, 이미지 내려받기), 액터 스케줄링, 생성자 실행을 모두 포함하고, `ray_serve_replica_initialization_latency_ms`는 생성자 실행만 잽니다. 차이가 크면 생성자 밖의 단계, 즉 노드 준비나 런타임 환경 구성, 액터 스케줄링에서 시간이 걸립니다. 오토스케일링에서는 `ray_serve_autoscaling_total_requests`를 `ray_serve_autoscaling_target_ongoing_requests`로 나누면 기본 정책이 계산한 원시 레플리카 수가 나와 정책 회귀를 점검할 수 있고, 확장 지연은 `ray_serve_autoscaling_target_replicas`를 실제 레플리카 수와 비교해 찾습니다. `ray_serve_deployment_status`는 상태를 `0`(UNKNOWN)부터 `6`(HEALTHY)까지 숫자로 내보내므로 상태 변화를 시간축에 놓고 볼 때 씁니다. 이벤트 루프 스케줄링 지연은 루프가 예정된 대기 시간보다 얼마나 더 막혔는지를 잽니다. 10ms 미만은 정상이고, 100ms 이상이면 비동기 핸들러 안의 블로킹 I/O나 CPU 집약 코드를 의심합니다.

메트릭을 직접 보려면 메트릭 내보내기 포트를 지정해 Ray를 시작합니다.

```bash
ray start --head --metrics-export-port=8080
```

그다음 1초씩 잠드는 디플로이먼트에 `Control-C`로 멈출 때까지 요청을 보냅니다.

```python
from ray import serve

import time
import requests


@serve.deployment
def sleeper():
    time.sleep(1)


s = sleeper.bind()

serve.run(s)

while True:
    requests.get("http://localhost:8000/")
```

스크립트가 도는 동안 브라우저에서 `localhost:8080`을 열고 `serve_`로 검색합니다. 기본적으로 10초마다 갱신되므로 새로 고쳐야 새 값이 보입니다. 한동안 실행하면 다음과 같은 줄이 나타납니다.

```text
ray_serve_deployment_processing_latency_ms_count{..., replica="sleeper#jtzqhX"} 48.0
ray_serve_deployment_processing_latency_ms_sum{..., replica="sleeper#jtzqhX"} 48160.6719493866
```

히스토그램은 관측 횟수(`_count`)와 합계(`_sum`)를 함께 내보내므로 평균은 둘을 나눠 구합니다.

$$
\frac{48160.67\ \text{ms}}{48} \approx 1003\ \text{ms}
$$

`time.sleep(1)`을 반영한 1초 남짓한 값입니다. 백분위수는 이렇게 정확히 나오지 않습니다. 값을 미리 정한 버킷(bucket)에 나눠 담으므로 p50, p95, p99는 버킷 경계 사이를 보간한 추정치입니다. 기본 지연 버킷은 다음과 같습니다(밀리초).

```text
[1,2,5,10,20,50,100,200,300,400,500,1000,2000,5000,10000,60000,120000,300000,600000]
```

지연이 600,000ms를 넘으면 모두 `+Inf` 버킷에 들어가고, 실제 지연이 간격이 넓은 두 경계 사이에 몰려 있으면 보간값이 실제와 어긋납니다. 대부분 요청이 10~100ms에 끝나는 서비스라면 그 구간을 촘촘하게 나눠야 합니다. 경계는 `RAY_SERVE_REQUEST_LATENCY_BUCKETS_MS` 같은 환경 변수에 쉼표로 구분해 지정하며, 시작 시점에 고정되므로 바꾼 뒤에는 Serve 액터를 재시작해야 반영됩니다.

```bash
export RAY_SERVE_REQUEST_LATENCY_BUCKETS_MS="10,50,100,500,1000,5000"
```

### 메트릭이 Prometheus에 도착하기까지

메트릭이 늦게 보이는 이유는 버퍼링이 두 단계로 겹치기 때문입니다. 라우터와 레플리카는 요청마다 생기는 오버헤드를 줄이려고 갱신을 프로세스 안에 모았다가 `RAY_SERVE_METRICS_EXPORT_INTERVAL_MS`(기본값 `100`ms)마다 Ray 메트릭 API로 내보냅니다. 값을 늘리면 기록 비용이 줄고 갱신이 드물어지며, `0`이면 모으지 않고 매번 기록합니다.

```bash
export RAY_SERVE_METRICS_EXPORT_INTERVAL_MS=500
```

이어 각 Ray 프로세스는 Ray Core의 `metrics_report_interval_ms`(기본값 `10000`ms)마다 메트릭을 메트릭 에이전트로 보내고, Prometheus는 이 에이전트를 수집합니다. 앞서 `localhost:8080`이 10초마다 갱신된 것도 이 주기 때문입니다. 따라서 Serve 쪽 주기만 줄여서는 Prometheus에 더 빨리 나타나지 않습니다. Ray Core 주기는 시작할 때 시스템 설정으로 바꾸며, 안정된 공개 API가 아니라서 예고 없이 바뀔 수 있습니다.

```bash
ray start --head --system-config='{"metrics_report_interval_ms": 1000}'
```

대규모 배포에서 레이블 값 조합이 만드는 시계열 수(카디널리티)가 부담되면 `RAY_SERVE_CONTROLLER_METRICS_INCLUDE_HIGH_CARDINALITY_TAGS=0`으로 컨트롤러가 내보내는 레플리카 수명 주기 메트릭의 `replica` 같은 출처 식별자를 빼고 `deployment`와 `application`만 남깁니다. 이 설정은 `ray_serve_deployment_replica_starts_total`처럼 레플리카가 내보내는 메트릭에는 영향을 주지 않습니다.

### 사용자 정의 메트릭

모델별 요청 수처럼 애플리케이션만 아는 지표는 `ray.serve.metrics`로 직접 정의하고 디플로이먼트나 레플리카 메타데이터 태그와 함께 내보냅니다.

```python
from ray import serve
from ray.serve import metrics

import time
import requests


@serve.deployment
class MyDeployment:
    def __init__(self):
        self.num_requests = 0
        self.my_counter = metrics.Counter(
            "my_counter",
            description=("The number of odd-numbered requests to this deployment."),
            tag_keys=("model",),
        )
        self.my_counter.set_default_tags({"model": "123"})

    def __call__(self):
        self.num_requests += 1
        if self.num_requests % 2 == 1:
            self.my_counter.inc()


my_deployment = MyDeployment.bind()
serve.run(my_deployment)

while True:
    requests.get("http://localhost:8000/")
    time.sleep(1)
```

내보낸 결과에는 이름 앞에 `ray_`, 카운터 뒤에 `_total`이 붙고, 직접 지정한 `model` 태그 외에 `deployment`와 `replica` 태그가 자동으로 들어갑니다.

```text
# HELP ray_my_counter_total The number of odd-numbered requests to this deployment.
# TYPE ray_my_counter_total counter
ray_my_counter_total{..., deployment="MyDeployment", model="123", replica="MyDeployment#rUVqKh"} 5.0
```

Prometheus 외의 플랫폼으로도 메트릭을 보낼 수 있습니다. 예를 들어 ML 관측 플랫폼 Arize와 연동하려면 Arize 클라이언트 코드를 디플로이먼트 코드에 직접 넣습니다.

## 다음 단계

번역 모델 하나로 익힌 디플로이먼트, 모델 조합, 설정 파일, RayService 흐름은 다른 모델에도 그대로 적용됩니다. 공식 예제 모음은 목적별로 나뉘어 있으므로 비슷한 사례부터 골라 따라 하면 됩니다.

| 분류 | 예제 |
| --- | --- |
| ML 애플리케이션 | [Stable Diffusion](https://docs.ray.io/en/latest/serve/tutorials/stable-diffusion.html), [텍스트 분류](https://docs.ray.io/en/latest/serve/tutorials/text-classification.html), [객체 탐지](https://docs.ray.io/en/latest/serve/tutorials/object-detection.html), [스트리밍 챗봇](https://docs.ray.io/en/latest/serve/tutorials/streaming.html) |
| AI 가속기 | [AWS NeuronCores](https://docs.ray.io/en/latest/serve/tutorials/aws-neuron-core-inference.html), [Intel Gaudi](https://docs.ray.io/en/latest/serve/tutorials/intel-gaudi-inference.html) |
| 통합 | [Gradio](https://docs.ray.io/en/latest/serve/tutorials/gradio-integration.html), [요청 배칭](https://docs.ray.io/en/latest/serve/tutorials/batch.html), [Triton](https://docs.ray.io/en/latest/serve/tutorials/triton-server-integration.html), [MLflow 모델 레지스트리](https://docs.ray.io/en/latest/serve/model-registries.html) |
| 배포 패턴 | [모델 멀티플렉싱](https://docs.ray.io/en/latest/_collections/serve/tutorials/model_multiplexing_forecast/README.html), [추천 시스템 모델 조합](https://docs.ray.io/en/latest/_collections/serve/tutorials/model-composition-recsys/README.html) |

### LLM 서빙

LLM은 `ray.serve.llm`의 Ray Serve LLM으로 서빙합니다. 디플로이먼트 클래스를 직접 정의하는 대신 `LLMConfig`에 모델과 디플로이먼트 설정을 적어 `build_openai_app`에 넘기면 OpenAI 호환 API 애플리케이션이 만들어집니다.

```python
from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app

llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="qwen-0.5b",
        model_source="Qwen/Qwen2.5-0.5B-Instruct",
    ),
    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=2),
    ),
)

app = build_openai_app({"llm_configs": [llm_config]})
serve.run(app, blocking=True)
```

`deployment_config`에는 앞서 다룬 오토스케일링 설정이 그대로 들어갑니다. Ray 2.51부터는 첫 토큰까지 걸리는 시간(TTFT), 출력 토큰당 시간(TPOT), KV 캐시 사용률 같은 vLLM 엔진 메트릭이 기본으로 켜져 있고, Grafana에서 LLM 전용 대시보드로 볼 수 있습니다. 모델 크기별 배포, 비전 모델, 추론 모델, DeepSeek와 gpt-oss 예제는 예제 모음의 [LLM 애플리케이션](https://docs.ray.io/en/latest/serve/examples.html) 항목에 있습니다.

### 더 읽을 문서

- [핵심 개념](https://docs.ray.io/en/latest/serve/key-concepts.html)과 [리소스 할당](https://docs.ray.io/en/latest/serve/resource-allocation.html)
- [오토스케일링 가이드](https://docs.ray.io/en/latest/serve/autoscaling-guide.html)와 [모델 조합 가이드](https://docs.ray.io/en/latest/serve/model_composition.html)
- [개발 워크플로](https://docs.ray.io/en/latest/serve/advanced-guides/dev-workflow.html)와 [디플로이먼트 설정](https://docs.ray.io/en/latest/serve/configure-serve-deployment.html)
- [프로덕션 가이드](https://docs.ray.io/en/latest/serve/production-guide/index.html)와 [모니터링](https://docs.ray.io/en/latest/serve/monitoring.html)

- - -

### Sources
* [Getting Started — Ray 2.59.0](https://docs.ray.io/en/latest/serve/getting_started.html)
* [Ray Serve: Scalable and Programmable Serving — Ray 2.59.0](https://docs.ray.io/en/latest/serve/index.html)
* [Serve ML Models (Tensorflow, PyTorch, Scikit-Learn, others) — Ray 2.59.0](https://docs.ray.io/en/latest/serve/tutorials/serve-ml-models.html)
* [Set Up FastAPI and HTTP — Ray 2.59.0](https://docs.ray.io/en/latest/serve/http-guide.html)
* [Deploy Compositions of Models — Ray 2.59.0](https://docs.ray.io/en/latest/serve/model_composition.html)
* [Key Concepts — Ray 2.59.0](https://docs.ray.io/en/latest/serve/key-concepts.html)
* [Production Guide — Ray 2.59.0](https://docs.ray.io/en/latest/serve/production-guide/index.html)
* [Examples — Ray 2.59.0](https://docs.ray.io/en/latest/serve/examples.html)
* [Develop and Deploy an ML Application — Ray 2.59.0](https://docs.ray.io/en/latest/serve/develop-and-deploy.html)
* [Serve Config Files — Ray 2.59.0](https://docs.ray.io/en/latest/serve/production-guide/config.html)
* [Monitor Your Application — Ray 2.59.0](https://docs.ray.io/en/latest/serve/monitoring.html)
* [Architecture — Ray 2.59.0](https://docs.ray.io/en/latest/serve/architecture.html)
* [Elevate Your LLM Inference: Autoscaling with Ray, ROCm 7.0.0, and SkyPilot — ROCm Blogs](https://rocm.blogs.amd.com/ecosystems-and-partners/ray-rocm7/README.html)
* [Add End-to-End Fault Tolerance — Ray 2.59.0](https://docs.ray.io/en/latest/serve/production-guide/fault-tolerance.html)
* [Observability and monitoring — Ray 2.59.0](https://docs.ray.io/en/latest/serve/llm/user-guides/observability.html)
* [Ray Serve API — Ray 2.59.0](https://docs.ray.io/en/latest/serve/api/index.html)
* [Develop a Ray Serve application &#124; Anyscale Docs](https://docs.anyscale.com/runtime/serve/develop)
* [Ray Serve Autoscaling — Ray 2.59.0](https://docs.ray.io/en/latest/serve/autoscaling-guide.html)
* [Deploying a Distributed Ray Python Server with Kubernetes, EKS &amp; KubeRay &#124; Shav Vimalendiran](https://shav.dev/blog/mlops-deploying-a-distributed-ray-python-server-kubernetes-eks-kuberay)
* [Ray Dashboard — Ray 2.59.0](https://docs.ray.io/en/latest/ray-observability/getting-started.html)
* [Master MLOps: Deploy Ray App with Serve CLI - Step-by-Step Tutorial](https://www.youtube.com/watch?v=n4DjumKhMi8)
* [Collecting and monitoring metrics — Ray 2.59.0](https://docs.ray.io/en/latest/cluster/metrics.html)
* [Configuring Ray — Ray 2.59.0](https://docs.ray.io/en/latest/ray-core/configure.html)
* [Configure Ray — Ray 3.0.0.dev0](https://docs.ray.io/en/master/core/configure.html)
* [Cluster Management CLI — Ray 2.59.0](https://docs.ray.io/en/latest/cluster/cli.html)
