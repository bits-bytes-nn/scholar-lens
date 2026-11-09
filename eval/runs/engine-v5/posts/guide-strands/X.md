---
layout: post
title: "Strands Agents Loop: Reason-Act Cycle and Its Production Controls"
date: 2026-11-08 23:30:56
description: "Strands Agents SDK의 에이전트 루프가 모델 호출과 도구 실행을 반복하는 구조를 설명하고, 호출 한도와 훅, 위험 호출 승인, 세션 저장, OpenTelemetry 트레이싱 같은 운영 제어를 정리합니다."
categories: ["Tech Guides", "Agentic-AI"]
tags: ["Strands-Agents-SDK", "Agent-Loop", "Human-in-the-Loop-Approval", "Structured-Output", "OpenTelemetry-Tracing", "Agentic-AI"]
cover: /assets/images/agentic-ai.jpg
---
## TL;DR

Strands Agents SDK는 에이전트 루프를 대신 돌려 줍니다. 루프는 모델을 호출하고, 모델이 도구를 쓰려 하면 그 도구를 실행한 뒤 결과를 붙여 모델을 다시 부르며, 최종 응답이 나올 때까지 이 과정을 반복합니다. 반복할 때마다 그 내용은 대화 기록에 쌓입니다. 다음에 무엇을 할지는 모델 호출이 돌려주는 정지 이유가 정합니다. `end_turn`은 정상 종료이고, `tool_use`는 도구를 실행한 뒤 다시 호출하라는 뜻입니다. `limit` 계열은 예산이 소진된 경우인데, 이때 기록은 유효한 상태로 남아 다시 호출할 수 있습니다. 도구 본문이 실패해도 루프는 기본적으로 멈추지 않고, 오류 결과를 모델에 돌려줍니다. 텍스트 대신 객체가 필요하면 Python에서는 `structured_output_model`로 Pydantic 스키마를 넘깁니다. Strands는 이 스키마를 도구로 바꾸고, 검증에 실패한 입력은 예외로 던지지 않고 도구 결과로 모델에 돌려보내 모델이 고치게 합니다.

통제 장치도 같은 루프에 겁니다. 훅은 이벤트 객체를 직접 고칩니다. 인터벤션 핸들러는 Proceed, Deny, Guide, Confirm, Transform 같은 타입 지정 액션을 돌려주고, 그 해석은 프레임워크가 맡습니다. 위험 분류기를 켠 `HumanInTheLoop`을 쓰면 위험한 도구 호출일 때만 사람의 승인을 기다립니다. Python에서는 `limits`로 호출당 턴 수와 토큰 수를 제한하고, `cancel_signal`이나 `agent.cancel()`로 실행 중인 호출을 멈춥니다. Python SDK에서는 이미 실행 중인 에이전트를 다시 호출하면 기본값인 `THROW` 모드에서 `ConcurrencyException`이 발생합니다. 대화 기록은 호출이 바뀌어도 유지되고 매 추론마다 모델에 전달됩니다. 길어진 기록은 기본값인 슬라이딩 윈도우 대화 매니저가 컨텍스트 윈도우에 맞게 줄입니다.

## 직접 작성하는 에이전트 루프를 대체하는 모델 주도 오케스트레이션 계층

Strands Agents SDK의 에이전트 루프(agent loop)는 모델이 여러 단계에 걸쳐 추론하고 행동하는 사이클을 관리하는 오케스트레이션 계층입니다. 외부 정보가 필요하거나 실제 부작용을 일으키는 작업을 모델이 풀 수 있게 해 주며, SDK의 다른 기능은 모두 이 루프 위에 올라갑니다. 모델은 학습한 지식만으로 답할 수 없는 요청을 받으면 파일을 읽거나 데이터베이스를 조회하거나 API를 부르거나 코드를 실행해야 합니다. 이런 외부 작업과 모델 호출을 이어 주는 일을 루프가 맡습니다.

고객 지원 티켓 분류 에이전트를 예로 들어 보겠습니다. 메시지 하나를 분류하려면 모델이 먼저 `lookup_customer`로 고객 정보를 가져와야 하고, 조회 결과를 읽은 다음에야 `TicketTriage` 형태의 분류를 내놓을 수 있습니다. 루프를 직접 짜면 모델 응답에서 도구 호출을 파싱하고, 함수를 실행하고, 결과를 메시지 목록에 붙이고, 다시 모델을 부르고, 언제 멈출지 판단하는 코드를 모두 써야 합니다. 운영 단계에 들어가면 여기에 호출 횟수 제한, 중간 취소, 오류 재시도, 실행 기록까지 덧붙여야 합니다.

### ReAct가 보여 준 생각-행동-관찰 궤적

추론이 중요한 과제에서 ReAct는 생각과 행동을 번갈아 생성해, 과제를 푸는 궤적이 생각-행동-관찰 단계를 여러 번 거치게 합니다. 생각은 에이전트의 행동 공간을 언어로 넓힌 것입니다. 외부 환경을 건드리지 않으므로 관찰이 돌아오지 않고, 컨텍스트만 갱신해서 이후의 추론과 행동을 돕습니다. 행동이 많은 의사결정 과제에서는 생각을 드문드문 넣고, 언제 생각할지는 모델이 정합니다. 이 실험은 PaLM-540B에 프롬프팅만 적용한 조건에서 진행했습니다. ReAct는 Wikipedia API와 상호작용하면서 HotpotQA와 Fever에서 체인 오브 소트(chain-of-thought)의 환각과 오류 전파를 줄였습니다. ALFWorld와 WebShop에서는 컨텍스트 내 예시를 한두 개만 주고도 모방학습·강화학습 방법보다 절대 성공률이 각각 34%, 10% 높았습니다.

Strands Agents는 모델 주도(model-driven) 방식으로 AI 에이전트를 만들고 실행하는 SDK입니다.

### 직접 만든 루프에는 없는 기본 제어

루프에는 운영 환경의 에이전트에 필요한 제어가 처음부터 들어 있습니다.

- 호출 한도: 한 번의 호출이 쓸 수 있는 턴과 토큰에 상한을 둡니다.
- 취소: 실행 중인 에이전트를 루프 도중에 멈춥니다.
- 정지 이유(stop reason): 루프가 왜 끝났는지 알려 줍니다.
- 동시 호출: 에이전트 하나로 여러 대화를 동시에 처리합니다.
- 훅(hook)과 재시도 전략: 훅은 루프의 어느 단계든 가로채고, 재시도 전략은 모델 오류와 도구 오류에서 복구합니다.

Strands는 에이전트 루프를 직접 작성해야 할 자리에 쓰도록 설계됐습니다.

### SDK 구성과 위치

Python 패키지는 PyPI의 `strands-agents`, TypeScript 패키지는 npm의 `@strands-agents/sdk`로 배포됩니다. 이 글의 코드는 Python으로 작성했고, TypeScript SDK도 같은 루프 개념을 따릅니다. 티켓 분류 에이전트는 이 가운데 Python SDK의 루프 위에서 동작하며, 분류에 필요한 도구와 스키마만 개발자가 정의합니다.

## 대화 기록에 누적되는 모델 호출-도구 실행 사이클과 정지 이유

한 번의 호출 안에서 루프는 모델 호출과 도구 실행을 번갈아 돌리면서 그 흔적을 모두 대화 기록(conversation history)에 쌓습니다. 다음에 무엇을 할지는 매 모델 호출이 돌려주는 정지 이유를 보고 정합니다.

이 섹션부터 나오는 Python 예제를 실행하려면 SDK를 먼저 설치해 둡니다.

```bash
pip install strands-agents
```

### 네 단계를 반복하는 사이클

루프는 네 단계를 반복합니다. 모델을 호출하고, 모델이 도구를 쓰려는지 확인하고, 그렇다면 도구를 실행한 뒤, 그 결과를 넣어 모델을 다시 호출합니다. 모델이 최종 응답을 내놓을 때까지 이 순서가 이어집니다. 반복할 때마다 대화 기록이 늘어나므로 모델은 원래 요청과 함께 지금까지 호출한 모든 도구와 받은 모든 결과를 봅니다. 여러 단계에 걸친 추론은 이렇게 쌓이는 컨텍스트 덕분에 가능합니다.

아래 그림은 이 구조를 상자와 화살표로 옮긴 예시입니다.

```text
 요청 (프롬프트 + 대화 기록)
          │
          ▼
 ┌──────────────────┐   도구 호출   ┌──────────────┐
 │ 모델: 다음 단계  │ ────────────▶ │  도구 실행   │
 │       결정       │ ◀──────────── │              │
 └──────────────────┘   도구 결과   └──────────────┘
          │ 도구 호출 없음
          ▼
 응답 (최종 메시지)

 ※ 모든 왕복이 대화 기록에 추가됨
```

### 두 역할의 메시지로 이뤄진 작업 메모리

루프를 오가는 메시지의 역할은 `user`와 `assistant` 두 가지뿐입니다. 사용자 메시지에는 사용자가 입력한 텍스트, 앞선 도구 실행의 결과, 파일·이미지·오디오·비디오 같은 미디어가 들어갑니다. 어시스턴트 메시지에는 사용자에게 보여 줄 텍스트 응답, 실행 시스템에 보내는 도구 사용 요청, 모델이 지원할 때의 추론 흔적이 들어갑니다. 도구 결과가 어시스턴트 쪽이 아니라 사용자 메시지에 담기는 데 주의하세요. 대화 기록은 반복마다 이 내용을 모두 누적하며, 그 자체가 모델이 작업하는 동안 쓰는 작업 메모리입니다.

티켓 분류 에이전트로 계산해 보겠습니다. 티켓 메시지 하나가 들어오면 기록에 사용자 텍스트 하나가 생깁니다. 모델이 `lookup_customer` 호출을 요청하면 어시스턴트 메시지가, 조회 결과가 돌아오면 사용자 메시지가 하나씩 더해지고, 두 번째 모델 호출이 분류 결과를 내면 어시스턴트 메시지가 하나 더 붙습니다. 결국 한 번의 호출로 메시지 네 개, 즉 두 쌍이 쌓입니다. 이 기록은 `agent.messages`로 읽습니다. `agent.messages`에는 도구 호출과 도구 결과를 포함해 모든 사용자·어시스턴트 메시지가 들어 있습니다. Strands는 이 기록을 호출과 호출 사이에 유지하고, 추론할 때마다 모델에 보내고, 도구 실행의 컨텍스트로 넘기며, 컨텍스트 윈도우(context window)에 맞게 줄입니다.

이전 대화를 이어 가거나 컨텍스트를 미리 채우려면 기존 메시지를 넘겨 에이전트를 만들면 됩니다. 다음 코드는 티켓 대화의 앞부분을 채운 상태에서 호출을 이어 갑니다.

```python
from strands import Agent

agent = Agent(messages=[
    {"role": "user", "content": [{"text": "고객 C-1042입니다. 결제가 두 번 청구됐어요."}]},
    {"role": "assistant", "content": [{"text": "확인해 보겠습니다. 결제 일시를 알려 주시겠어요?"}]},
])

agent("어제 오후 3시쯤입니다. 이 티켓을 분류해 주세요.")
print(agent.messages)  # 초기 메시지와 이번 호출의 메시지가 모두 담김
```

### 다음 행동을 정하는 정지 이유

각 모델 호출은 정지 이유로 끝나고, 루프는 이 값을 보고 다음 행동을 정합니다. 아래 표의 이름은 Python 표기이며, TypeScript에서는 `endTurn`, `toolUse`처럼 카멜 표기를 씁니다.

| 정지 이유 | 의미 | 루프의 다음 행동 |
|---|---|---|
| `end_turn` | 응답을 마쳤고 더 할 행동이 없음 | 정상 종료, 최종 메시지 반환 |
| `tool_use` | 계속하기 전에 도구 실행이 필요함 | 도구 실행, 결과를 기록에 추가, 모델 재호출 |
| `cancelled` | `agent.cancel()`로 외부에서 중단됨 | 종료 |
| `limit_turns` | 호출당 턴 예산 소진 | 종료, 기록은 유효한 상태로 남음 |
| `limit_total_tokens` | 누적 토큰 예산 소진 | 종료, 기록은 유효한 상태로 남음 |
| `limit_output_tokens` | 출력 토큰 예산 소진 | 종료, 기록은 유효한 상태로 남음 |
| `limit_structured_output_attempts` | 구조화된 출력 생성 실패가 한도에 도달함 | 종료, 기록은 유효한 상태로 남음 |
| `max_tokens` | 토큰 한도에 걸려 응답이 잘림 | 복구 불가, 오류로 종료 |
| `stop_sequence` | 설정한 정지 시퀀스를 만남 | `end_turn`처럼 정상 종료 |
| `content_filtered` | 안전 장치가 응답을 차단함 | 종료 |
| `guardrail_intervened` | 가드레일 정책이 생성을 멈춤 | 종료 |

정상 경로는 `tool_use`를 몇 번 거친 뒤 `end_turn`으로 끝나는 흐름이고, 나머지는 모두 루프를 끝냅니다. 이 가운데 `limit_` 계열은 예산을 다 써서 정상적으로 멈춘 경우입니다. 메시지 기록이 유효한 상태로 남아 있으므로 예산을 늘리거나 프롬프트를 바꿔 다시 호출하면 됩니다. `max_tokens`는 다릅니다. 모델은 잘린 응답에서 이어 쓸 수 없어서 현재 루프 안에서는 복구할 수 없고, 루프는 오류와 함께 끝납니다. `content_filtered`와 `guardrail_intervened`도 루프를 끝내며, 어떻게 처리할지는 애플리케이션 요구 사항에 맞춰 정해야 합니다. 한도와 취소를 실제로 거는 방법과 정지 이유별 대응 코드는 턴·토큰 한도와 취소를 다루는 운영 설정에서 이어서 봅니다.

### 반복마다 커지는 기록과 슬라이딩 윈도우

기록이 반복마다 늘어나므로 도구 호출이 많은 복잡한 작업에서는 기록이 모델의 컨텍스트 윈도우를 넘을 수 있고, 그러면 에이전트는 더 진행하지 못합니다. 증상은 두 가지로 나타납니다. 모델 제공자가 입력 길이 오류를 돌려주거나, 덜 중요한 초기 메시지가 컨텍스트를 채우면서 모델 성능이 떨어집니다. 대응 방법은 넷입니다.

- 도구 출력 줄이기: 전체 데이터 대신 요약이나 필요한 발췌만 돌려줍니다.
- 도구 스키마 단순화: 깊이 중첩된 스키마는 도구 설정과 모델 추론 양쪽에서 토큰을 씁니다.
- 대화 매니저(conversation manager) 구성: 기본 슬라이딩 윈도우 전략으로 충분한 애플리케이션이 많지만, 오래 실행되는 작업에는 요약이나 직접 만든 전략이 필요할 수 있습니다.
- 작업 분해: 큰 작업을 하위 작업으로 나누고 각각을 새 컨텍스트에서 처리합니다.

티켓 분류 에이전트라면 `lookup_customer`가 고객 레코드 전체 대신 분류에 필요한 필드만 돌려주게 만드는 방법이 가장 손이 덜 갑니다.

기본 대화 매니저인 `SlidingWindowConversationManager`는 최근 N개의 메시지 쌍을 유지하고, 윈도우 크기를 넘으면 가장 오래된 메시지부터 지웁니다. 컨텍스트 윈도우 오버플로 예외가 나면 컨텍스트를 줄여서 처리합니다. 따로 지정하지 않아도 이 매니저가 쓰이며, Python에서는 크기를 바꾸려면 `window_size`를 넘깁니다(TypeScript에서는 `windowSize`).

```python
from strands import Agent
from strands.agent.conversation_manager import SlidingWindowConversationManager

conversation_manager = SlidingWindowConversationManager(
    window_size=10,  # 유지할 최대 메시지 쌍 수
)

agent = Agent(conversation_manager=conversation_manager)
```

앞서 계산한 대로 티켓 하나에 두 쌍이 쌓인다면, `window_size=10`일 때 대략 최근 다섯 건의 티켓 처리 흔적이 남습니다.

### 직접 도구 호출도 기록에 남는 기본 동작

모델을 거치지 않고 `agent.tool.<이름>()`으로 도구를 직접 호출해도 그 호출은 기본적으로 대화 기록에 남습니다. Python에서 `record_direct_tool_call=False`를 넘기면 기록하지 않습니다. 아래 코드에서 첫 번째 `notebook` 호출은 기록에 들어가고, 두 번째 호출은 들어가지 않습니다.

```python
from strands import Agent
from strands.vended_tools import notebook

agent = Agent(tools=[notebook])

agent.tool.notebook(mode="create", name="ideas", new_str="# Project ideas")  # 기록됨
agent.tool.notebook(mode="list", record_direct_tool_call=False)            # 기록되지 않음

print(agent.messages)
```

TypeScript에서는 같은 옵션을 `recordDirectToolCall: false`로 씁니다. 디버깅하려고 도구를 직접 불러 보는 호출은 기록에서 빼 두는 편이 낫습니다. 그래야 모델이 다음 티켓을 분류할 때 그 결과를 대화의 일부로 읽지 않습니다.

## 이벤트를 바꾸는 훅과 타입 지정 결정을 돌려주는 인터벤션

루프의 한 지점에 로직을 끼워 넣는 방법은 두 가지입니다. 훅은 이벤트 객체를 직접 고치고, 인터벤션은 "막아라", "다시 시도하게 하라" 같은 결정을 타입으로 돌려주며 그 결정의 해석과 충돌 정리는 프레임워크가 맡습니다.

### 라이프사이클 지점마다 여러 구독자를 받는 콜백

훅 이벤트는 에이전트 라이프사이클의 한 지점이고, 훅 콜백은 그 이벤트가 발생할 때 실행되도록 등록한 함수입니다. 훅은 타입 안전하고 서로 조합할 수 있으며, 이벤트 타입 하나에 구독자를 여럿 붙일 수 있습니다.

구독자가 여럿이면 실행 순서를 정해야 합니다. 기본 규칙에서 After 이벤트 콜백은 정리 작업의 대칭을 맞추려고 등록의 역순으로 실행됩니다. `order` 옵션으로 우선순위를 직접 줄 수 있고, 값이 낮을수록 먼저 실행됩니다. 같은 `order` 그룹 안에서는 Before 이벤트가 등록 순서를, After 이벤트가 그 역순을 따릅니다. SDK 자체 훅의 위치를 나타내는 프리셋으로 `HookOrder.SDK_FIRST`, 아무것도 지정하지 않을 때의 `HookOrder.DEFAULT`, `HookOrder.SDK_LAST`가 있습니다. 이 값들은 강제되는 경계가 아니어서 어떤 숫자든 쓸 수 있으며, `float('-inf')`/`float('inf')`를 주면 절대적인 처음과 끝이 보장됩니다.

예를 들어 티켓 분류 에이전트에 감사 로그 훅 A와 지표 수집 훅 B를 이 순서로, 둘 다 기본 `order`로 등록했다고 해 보겠습니다. 도구 호출 전에는 A, B 순으로 실행되고, 도구 호출 뒤에는 B, A 순으로 실행됩니다. A가 연 자원을 A가 마지막에 닫게 되는 구조입니다. ```python

## excerpt: 감사 로그 훅 A와 지표 훅 B를 기본 order로 등록 (lookup_customer 도구는 별도로 정의)
from strands import Agent
from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent

agent = Agent(tools=[lookup_customer])

def audit_log_hook(event: BeforeToolCallEvent) -> None:
    print("[audit] A before:", event.tool_use["name"])

def metrics_hook(event: BeforeToolCallEvent) -> None:
    print("[metrics] B before:", event.tool_use["name"])

def audit_after(event: AfterToolCallEvent) -> None:
    print("[audit] A after:", event.tool_use["name"])

def metrics_after(event: AfterToolCallEvent) -> None:
    print("[metrics] B after:", event.tool_use["name"])

## A, B 순으로 등록: 도구 호출 전에는 A, B / 호출 뒤에는 B, A
agent.add_hook(audit_log_hook)
agent.add_hook(metrics_hook)
agent.add_hook(audit_after)
agent.add_hook(metrics_after)
```

SDK 훅보다 앞이나 뒤에 확실히 두고 싶은 콜백에는 프리셋을 씁니다.

```python
from strands import Agent
from strands.hooks import BeforeModelCallEvent, HookOrder

agent = Agent()

def early_hook(event: BeforeModelCallEvent) -> None:
    print("I run first")

def late_hook(event: BeforeModelCallEvent) -> None:
    print("I run last")

agent.add_hook(early_hook, order=HookOrder.SDK_FIRST)
agent.add_hook(late_hook, order=HookOrder.SDK_LAST)
```

### 실행 직전에 도구를 바꿔 끼우는 훅

훅은 이벤트 속성을 그 자리에서 바꿉니다. Python에서는 `BeforeToolCallEvent`의 `selected_tool`을 다른 도구로 바꾸고 `tool_use`의 이름도 함께 고치면, 실행 직전에 원래 도구 대신 더 안전한 도구가 돌아갑니다. 아래 발췌는 티켓 분류 에이전트의 `lookup_customer` 호출을 마스킹된 조회 도구로 갈아 끼웁니다. 여기서 쓰는 `HookProvider`는 여러 콜백을 한 객체에 묶어 두고 `register_hooks` 안에서 `registry.add_callback`으로 한꺼번에 등록하는 객체로, 함수 하나를 에이전트에 바로 붙이는 앞의 `add_hook`과 다릅니다.

```python
# excerpt: HookProvider로 도구를 교체하는 부분 (safe_alternative_tool은 별도로 정의)
from strands.hooks import BeforeToolCallEvent, HookProvider, HookRegistry

class ToolInterceptor(HookProvider):
    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeToolCallEvent, self.intercept_tool)

    def intercept_tool(self, event: BeforeToolCallEvent) -> None:
        if event.tool_use["name"] == "lookup_customer":
            # 마스킹된 조회 도구로 교체
            event.selected_tool = self.safe_alternative_tool
            event.tool_use["name"] = "lookup_customer_masked"
```

이벤트 속성을 이렇게 수정할 때는 디버깅과 감사를 위해 변경 내용을 로그로 남겨야 합니다. 호출이 끝난 뒤 새 호출을 이어 붙이는 수정도 가능합니다. `AfterInvocationEvent.resume`에 입력을 넣으면 에이전트가 호출자에게 돌아가지 않고 그 입력으로 다시 호출되며, 이렇게 재호출을 연쇄할 때는 무한 루프에 빠지지 않도록 종료 조건을 넣어야 합니다.

### 결정을 타입으로 돌려주는 인터벤션

훅 방식에는 구조적인 약점이 있습니다. 훅과 플러그인이 `event.cancel = "reason"`처럼 이벤트를 직접 고치면 프레임워크는 왜 취소됐는지 모릅니다. 인가 거부처럼 확정된 차단인지, 다른 방식으로 다시 해 보라는 부드러운 안내인지 구분하지 못합니다. 같은 이벤트를 여러 플러그인이 고치면 마지막에 쓴 쪽이 이기면서 충돌이 조용히 묻힙니다. 인터벤션도 훅 위에서 동작하며, 라이프사이클 메서드마다 훅 콜백이 하나씩 등록됩니다. 다른 점은 핸들러가 이벤트를 고치지 않고 타입 지정 액션을 돌려주고, 그 액션을 프레임워크가 해석한다는 데 있습니다.

액션은 다섯 가지입니다. `Proceed()`는 그대로 진행시키고, `Deny(reason=...)`는 동작을 막으면서 나머지 핸들러를 건너뜁니다. `Guide(feedback=...)`는 동작을 취소하고 모델이 다시 시도할 때 참고할 피드백을 주며, `Confirm(prompt=...)`은 사람의 승인을 기다리고, `Transform(apply=fn)`은 실행을 이어 가기 전에 이벤트 내용을 제자리에서 수정합니다. 클래스 이름은 Python 기준이고, TypeScript에서는 `InterventionActions.deny(reason)` 같은 팩토리를 씁니다.

핸들러가 오버라이드하는 라이프사이클 메서드는 다섯 개이고, 메서드마다 돌려줄 수 있는 액션이 정해져 있습니다. 오버라이드하지 않은 메서드의 기본값은 모두 Proceed입니다. 아래 표의 메서드 이름은 Python 표기이며 TypeScript에서는 `beforeToolCall`처럼 카멜 표기를 씁니다.

| 메서드 | 유효 액션 | 실행 시점 |
|---|---|---|
| `before_invocation` | Proceed, Deny, Guide, Transform | Before the agent loop starts |
| `before_tool_call` | Proceed, Deny, Guide, Confirm, Transform | Before each tool execution |
| `after_tool_call` | Proceed, Transform | After each tool execution |
| `before_model_call` | Proceed, Deny, Guide, Transform | Before each model API call |
| `after_model_call` | Proceed, Guide, Transform | After each model response |

같은 액션이라도 before 이벤트인지 after 이벤트인지에 따라 효과가 달라집니다.

| 액션 | Before 이벤트 | After 이벤트 |
|---|---|---|
| Deny | `event.cancel`을 설정하고 나머지 핸들러를 건너뜀 | 효과 없음(런타임에 경고) |
| Guide | `before_tool_call`/`before_invocation`에서는 누적된 피드백과 함께 취소, `before_model_call`에서는 피드백을 사용자 메시지로 주입 | 피드백을 주입하고 재시도 |
| Confirm | interrupt/resume으로 에이전트를 멈추고 사람 승인 대기, 거부 응답은 `event.cancel` 설정 | 지원하지 않음 |
| Transform | `action.apply(event)` 호출, 이후 핸들러는 수정된 내용을 봄 | `action.apply(event)` 호출 |

### 등록 순서대로 평가되고 Deny에서 끊기는 파이프라인

핸들러는 등록 순서대로 평가되고, 어느 핸들러든 `Deny`를 돌려주면 나머지는 건너뛴 채 동작이 즉시 차단됩니다. 덕분에 인가 같은 빠른 검사를 앞에 두면 LLM 기반 조정처럼 비싼 평가가 불필요하게 돌지 않습니다. 반면 `Guide`가 나오면 모든 핸들러가 계속 실행되고, 각자의 피드백이 누적되어 모델에는 하나로 합쳐진 안내가 전달됩니다. 여러 핸들러의 결정이 엇갈리면 deny > confirm > guide > transform > proceed 순으로 우선합니다.

티켓 분류 에이전트로 계산해 보겠습니다. 호출 횟수 제한(`RateLimiter`)과 고객 이메일 누락 검사(`CustomerEmailValidator`) 핸들러 둘을 이 순서로 등록했다고 하겠습니다. 열한 번째 `lookup_customer` 호출이면 첫 핸들러가 Deny를 내고 두 번째 핸들러는 아예 실행되지 않습니다. 세 번째 호출에 `email`이 빠져 있으면 첫 핸들러는 Proceed, 두 번째는 Guide를 내고, 호출은 그 피드백과 함께 취소되며 모델은 그 안내를 보고 다시 시도합니다. 여기에 Guide를 내는 핸들러가 더 붙으면 각 피드백이 메시지 하나로 합쳐집니다. 아래 발췌는 두 핸들러를 정의하고 등록합니다.

```python
# excerpt: 핸들러 정의와 등록 (lookup_customer 도구는 별도로 정의)
from strands import Agent
from strands.hooks import BeforeToolCallEvent
from strands.interventions import Deny, Guide, InterventionHandler, Proceed

class RateLimiter(InterventionHandler):
    name = "rate-limiter"

    def __init__(self):
        self.call_count = 0

    def before_tool_call(self, event: BeforeToolCallEvent):
        self.call_count += 1
        if self.call_count > 10:
            # Deny가 나오면 뒤에 등록된 핸들러는 건너뜀
            return Deny(reason="Rate limit exceeded")
        return Proceed()

class CustomerEmailValidator(InterventionHandler):
    name = "customer-email-validator"

    def before_tool_call(self, event: BeforeToolCallEvent):
        if event.tool_use["name"] == "lookup_customer":
            tool_input = event.tool_use.get("input", {})
            if not tool_input.get("email"):
                return Guide(feedback="Always pass the customer's email to lookup_customer.")
        return Proceed()

agent = Agent(
    tools=[lookup_customer],
    interventions=[
        RateLimiter(),          # 먼저 평가
        CustomerEmailValidator(),  # RateLimiter가 Deny하면 건너뜀
    ],
)
```

훅으로 같은 일을 하려면 플러그인마다 작업 전에 `event.cancel`을 스스로 확인해야 합니다. 인터벤션에서는 그 확인을 프레임워크가 대신합니다.

### 수렴을 핸들러가 보장해야 하는 after_model_call의 Guide

`after_model_call`에서 `Guide`를 돌려주면 모델이 다시 호출되는데, 프레임워크는 이 재시도에 상한을 두지 않습니다. 응답 어조가 마음에 들지 않을 때마다 Guide를 내는 핸들러는 모델이 끝내 기준을 맞추지 못하면 루프를 계속 돌립니다. after 이벤트의 Deny는 런타임 경고만 낼 뿐 효과가 없고, `after_model_call`의 유효 액션 목록에도 Deny는 없습니다. 따라서 `after_model_call`에서는 핸들러 안에서 Guide 횟수를 세다가 한도에 이르면 Proceed를 돌려줍니다. 루프 전체를 끊는 장치는 턴·토큰 한도를 다루는 운영 설정에서 따로 겁니다.

### 사람 승인을 기다리는 Confirm

`Confirm`은 `before_tool_call`에서만 쓸 수 있고, SDK의 interrupt/resume 시스템에 붙어 도구 실행 전에 사람 승인을 기다립니다. Python에서는 `response` 인자가 동작을 가릅니다. `response`를 주면 그 값이 곧바로 `evaluate` 함수에 넘어가 에이전트가 멈추지 않고, 주지 않으면 에이전트 루프를 빠져나와 외부에서 재개해 주기를 기다립니다. `evaluate`는 응답을 승인으로 볼지 판단하며, 기본 구현은 대소문자 구분 없이 `True`, `"y"`, `"yes"`를 승인으로 받아들입니다. 아래 발췌의 `lookup_customer` 호출은 미리 승인한 형태라 멈추지 않습니다.

```python
# excerpt: 선승인 Confirm (response가 있으면 멈추지 않음)
from strands.hooks import BeforeToolCallEvent
from strands.interventions import Confirm, InterventionHandler, Proceed

class AutoApproveLookup(InterventionHandler):
    name = "auto-approve-lookup"

    def before_tool_call(self, event: BeforeToolCallEvent):
        if event.tool_use["name"] == "lookup_customer":
            return Confirm(
                prompt="Allow lookup_customer?",
                response="yes",
            )
        return Proceed()
```

위험한 호출만 골라 사람에게 묻는 실제 승인 흐름은 `HumanInTheLoop`과 위험 분류기를 다루는 부분에서 이 에이전트에 붙입니다.

### 핸들러마다 고르는 실패 방식

훅에서 던져진 예외는 항상 그대로 전파되지만, 인터벤션 핸들러는 `on_error` 속성으로 자기 실패 방식을 정합니다. 값은 셋입니다. 기본값 `'throw'`는 예외를 다시 던져 호출을 실패시키고, `'proceed'`는 그 핸들러를 건너뛰고 `Proceed()`가 나온 것처럼 진행하며(fail-open), `'deny'`는 오류를 로그로 남기고 `Deny`로 처리합니다(fail-closed). 인가처럼 보안을 맡은 핸들러에는 절대 `'proceed'`를 쓰면 안 됩니다. 핸들러가 예외를 던지는 순간 그 핸들러가 지키던 동작이 검사 없이 실행되기 때문입니다. 티켓 분류 에이전트의 고객 데이터 접근 검사라면 다음처럼 fail-closed로 둡니다.

```python
from strands.hooks import BeforeToolCallEvent
from strands.interventions import Deny, InterventionHandler, OnError, Proceed

class StrictAuth(InterventionHandler):
    name = "strict-auth"

    @property
    def on_error(self) -> OnError:
        return "deny"

    def before_tool_call(self, event: BeforeToolCallEvent):
        # 인가 서비스가 예외를 던지면 호출은 거부됨
        if not self._check_permission(event.tool_use["name"]):
            return Deny(reason="Unauthorized")
        return Proceed()

    def _check_permission(self, tool_name: str) -> bool:
        return tool_name == "lookup_customer"
```

반대로 호출을 기록만 하는 로깅 핸들러는 강제보다 가용성이 중요하므로 `'proceed'`로 두면 로그 서비스가 죽어도 티켓 분류가 멈추지 않습니다.

## 스키마를 도구로 바꿔 검증된 객체를 돌려받는 티켓 분류 에이전트의 첫 실행

텍스트 대신 객체가 필요하면 호출에 Pydantic 스키마 하나만 넘기면 됩니다. 그러면 Strands가 그 스키마를 모델이 마지막에 호출할 도구로 바꾸고, 검증에 실패한 입력은 도구 결과로 모델에 돌려보내 모델이 고치게 합니다.

### 설치와 가장 작은 분류 에이전트

Python SDK는 `strands-agents` 패키지로 설치하고, 스키마를 쓸 때 필요한 Pydantic도 함께 설치됩니다.

```bash
pip install strands-agents
```

PyPI 페이지는 기본 모델 공급자가 Amazon Bedrock이고, 이를 쓰려면 AWS 자격 증명을 먼저 설정해야 한다고 적고 있습니다.

스키마에는 필드마다 이름, 타입, 설명을 적습니다. 스키마를 넘겨 호출하면 에이전트는 텍스트 대신 그 스키마로 검증한 객체를 돌려줍니다. 아래 코드가 티켓 분류 에이전트의 출발점입니다. Python에서는 `structured_output_model`로 스키마를 넘기고 `result.structured_output`에서 객체를 읽습니다. 객체를 받지 못한 경우는 `StructuredOutputException`으로 처리합니다.

```python
from typing import Literal

from pydantic import BaseModel, Field
from strands import Agent
from strands.types.exceptions import StructuredOutputException


class TicketTriage(BaseModel):
    """Routing decision for one customer support message."""

    category: Literal["billing", "bug", "account", "other"] = Field(
        description="The team that should handle the message"
    )
    priority: Literal["low", "medium", "high"] = Field(
        description="high when the customer is blocked or losing money"
    )
    summary: str = Field(description="One sentence describing the problem")


def triage(message: str) -> TicketTriage:
    # A fresh agent per message keeps tickets from sharing conversation history
    agent = Agent(callback_handler=None)
    try:
        result = agent(message, structured_output_model=TicketTriage)
    except StructuredOutputException:
        # Send anything the agent can't classify to a person
        return TicketTriage(category="other", priority="medium", summary=message)
    return result.structured_output


ticket = triage("I was charged twice this month and need one of the charges refunded.")
print(f"{ticket.priority} | {ticket.category} | {ticket.summary}")
```

실행하면 대개 이런 줄이 나옵니다.

```text
high | billing | Customer was charged twice this month and is requesting a refund.
```

`category`와 `priority`는 `Literal`로 고정된 값만 받으므로 에이전트는 미리 정의한 팀과 우선순위 중에서만 고를 수 있습니다. 모델은 필드 설명도 읽습니다. 그래서 `priority`의 설명에는 타입보다 "고객이 막혔거나 돈을 잃고 있을 때 high"처럼 의미를 적어야 합니다. 비동기 경로의 `invoke_async`도 같은 인자를 받습니다.

### 객체를 돌려주기 전에 도구를 부르는 에이전트

스키마를 넘긴 호출에서도 에이전트는 객체를 반환하기 전에 다른 도구를 호출할 수 있습니다. 앞선 훅 예제들이 정의 없이 써 온 `lookup_customer`를 여기서 정의합니다. 이메일로 고객 이름과 요금제를 조회하고, 그 결과로 객체를 채우는 도구입니다.

```python
from typing import Literal

from pydantic import BaseModel
from strands import Agent, tool


@tool
def lookup_customer(email: str) -> dict:
    """Look up a customer's name and plan by email address."""
    customers = {"dana@example.com": {"name": "Dana Ruiz", "plan": "enterprise"}}
    return customers.get(email, {"name": "unknown", "plan": "free"})


class CustomerTicket(BaseModel):
    customer_name: str
    plan: Literal["free", "pro", "enterprise"]
    priority: Literal["low", "medium", "high"]


agent = Agent(tools=[lookup_customer], callback_handler=None)
result = agent(
    "From dana@example.com: our dashboard has been down for an hour.",
    structured_output_model=CustomerTicket,
)
print(result.structured_output)
```

전형적인 출력에서는 조회 결과가 필드에 그대로 들어갑니다.

```text
customer_name='Dana Ruiz' plan='enterprise' priority='high'
```

### 호출 단위로 적용되는 스키마

스키마는 그것을 넘긴 호출에만 적용됩니다. 그래서 같은 에이전트의 이전 턴은 평소처럼 텍스트로 답하고, 스키마를 넘긴 호출에서는 에이전트가 대화 전체를 보고 필드를 채웁니다. 로그인 문제로 몇 턴 대화한 뒤 마지막 호출에만 에스컬레이션 스키마를 넘기면, 앞선 대화 내용이 요약된 객체가 나옵니다.

모든 호출에 같은 스키마를 쓰려면 `Agent(structured_output_model=CustomerInfo, callback_handler=None)`처럼 에이전트를 만들 때 지정합니다. 호출에 넘긴 스키마가 에이전트 스키마보다 우선합니다. 다만 에이전트 스키마는 맥락만 쌓는 턴에도 적용됩니다. 일부 호출이 일반 텍스트로 답해야 한다면 스키마를 호출마다 넘기는 쪽을 택합니다. 예전 API인 `Agent.structured_output()`과 `Agent.structured_output_async()`는 Python에서 deprecated 상태이며 `DeprecationWarning`을 냅니다.

### 마지막에 호출되는 도구로 바뀌는 스키마

Strands는 모델에게 답변 안에 JSON을 쓰라고 요구하지 않습니다. 대신 스키마를 도구로 바꾸고, 모델은 필드 값을 입력으로 넣어 그 도구를 호출하는 방식으로 객체를 돌려줍니다. 이렇게 하면 모델은 채워야 할 타입 있는 입력 스키마를 받고, Strands는 검증 오류를 에이전트 루프로 되돌려 보낼 통로를 얻습니다. Python에서는 클래스 이름이 도구 이름이 되므로 `TicketTriage` 스키마는 `TicketTriage` 도구가 됩니다. 클래스 docstring은 도구 설명이 되고, 각 `Field(description=...)`는 해당 필드를 모델에 설명합니다. TypeScript에서는 Zod 객체 스키마를 쓰고 도구 이름이 `strands_structured_output`으로 고정됩니다. Strands는 도구 설명에 "필요한 다른 도구를 모두 쓴 뒤 이 도구를 마지막에 호출하라"는 지시도 덧붙입니다.

스키마가 있는 호출은 다음 순서로 진행됩니다.

1. Strands가 스키마로 만든 구조화된 출력 도구를 에이전트의 도구 목록에 추가합니다.
2. 모델은 평소처럼 작업하며 그사이 `lookup_customer` 같은 다른 도구를 부를 수 있습니다.
3. 모델이 필드 값을 넣어 구조화된 출력 도구를 호출합니다.
4. Strands가 입력을 검증합니다. 통과하면 호출이 끝나고 객체는 `result.structured_output`에 담깁니다.
5. 호출이 끝나면 구조화된 출력 도구는 제거됩니다.

### 예외 대신 도구 결과로 돌아오는 검증 오류

입력이 검증에 실패해도 Strands는 예외를 던지지 않습니다. 검증 오류를 도구 결과로 모델에 돌려주고, 모델은 값을 고쳐 도구를 다시 호출합니다. 티켓마다 후속 작업을 잡는 스키마로 확인해 보겠습니다. 마감일이 반드시 미래여야 한다는 규칙을 validator에 넣었습니다.

```python
from datetime import date

from pydantic import BaseModel, field_validator
from strands import Agent


class FollowUp(BaseModel):
    """A follow-up task for a support ticket."""

    action: str
    due_date: date

    @field_validator("due_date")
    @classmethod
    def must_be_in_future(cls, value: date) -> date:
        if value <= date.today():
            raise ValueError(f"due_date must be after {date.today().isoformat()}")
        return value


agent = Agent(callback_handler=None)
result = agent(
    "Schedule a follow-up call with the customer next Tuesday.",
    structured_output_model=FollowUp,
)
print(result.structured_output)
```

모델은 오늘 날짜를 모르기 때문에 첫 추측이 검사를 통과하지 못할 수 있습니다. 모델이 처음에 지난 날짜를 `due_date`에 넣어 도구를 호출하면 검증에 실패합니다. 그러면 모델은 오늘 날짜가 담긴 `due_date must be after ...` 오류를 도구 결과로 받고, 그 뒤의 날짜로 다시 호출합니다. 이번에는 검증을 통과해 호출이 객체를 돌려줍니다. 이런 규칙은 프롬프트가 아니라 스키마에 둡니다. 모델은 오류 메시지를 글자 그대로 보므로, 메시지도 모델이 따라 할 수 있는 지시문으로 써야 합니다. 위 메시지에 기준 날짜가 들어 있는 것도 그래서입니다.

### 강제 재시도와 시도 한도

모델이 구조화된 출력 도구를 호출하지 않고 텍스트로 턴을 끝내는 경우도 있습니다. 이때 Strands는 그 도구를 필수로 지정해 한 번 더 시도하므로, 모델은 도구를 호출할 수밖에 없습니다. Python에서는 재시도 직전에 `You must format the previous response as structured output.`라는 사용자 메시지가 대화에 추가됩니다. 이 문구는 `structured_output_prompt`를 에이전트나 단일 호출에 넘겨 바꿀 수 있고, 둘 다 지정하면 호출 쪽 값이 우선합니다. TypeScript는 메시지를 추가하지 않고, 텍스트 응답을 버린 뒤 도구 선택만 바꿔 재시도합니다. 이 재시도에서도 유효한 출력이 나오지 않으면 호출은 `StructuredOutputException`을 던집니다. 맨 처음 `triage` 함수가 이 예외를 잡아 사람이 검토할 `other` 티켓으로 넘긴 이유가 여기에 있습니다.

예외 대신 더 시도하게 하려면 호출에 `limits['structured_output_attempts']`를 지정합니다. 검증에 실패한 도구 호출과 유효한 출력을 내지 못한 강제 재시도는 각각 한 번의 시도로 셉니다. 한도가 설정되어 있으면 Strands는 강제 재시도가 실패해도 예외를 던지지 않고 다시 시도하며, 강제 재시도 전에 부른 다른 도구 호출은 횟수에 들어가지 않습니다. 한도에 도달하면 다음 모델 호출 전에 호출이 멈춥니다. 결과의 정지 이유는 `limit_structured_output_attempts`이고 구조화된 출력은 비어 있습니다. 앞서 정의한 `TicketTriage`에 한도를 걸면 이렇게 됩니다.

```python
from strands import Agent

agent = Agent(callback_handler=None)
result = agent(
    "I was charged twice this month and need one of the charges refunded.",
    structured_output_model=TicketTriage,
    limits={"structured_output_attempts": 3},
)
if result.stop_reason == "limit_structured_output_attempts":
    print("No valid triage after 3 attempts")
else:
    print(result.structured_output)
```

콘텐츠 필터나 가드레일이 강제 재시도를 막으면 SDK마다 처리가 갈립니다. TypeScript는 이를 실패한 시도로 세고 한도 안에서 다시 시도하지만, Python은 `content_filtered` 같은 별도의 정지 이유로 호출을 끝냅니다. 이 한도는 요청 전체에 걸립니다. 같은 요청 안에서 도구로 실행되는 에이전트의 시도도 합산되고, `result.invocation_context`를 다음 호출에 넘기면 횟수가 이어집니다.

### 텍스트는 청크로, 객체는 한 번만 오는 스트리밍

스트리밍에서도 스키마를 넘기는 방법은 같습니다. 텍스트는 모델이 쓰는 대로 청크 단위로 도착하고, 객체는 최종 결과 이벤트에 한 번만 담겨 옵니다. Python에서는 `stream_async`에 `structured_output_model`을 넘기고, `"result"` 이벤트에서 객체를 꺼냅니다. 지연된 환불에 대한 사과 답장 초안을 받는 예입니다.

```python
import asyncio

from pydantic import BaseModel
from strands import Agent


class ReplyDraft(BaseModel):
    subject: str
    body: str


async def main():
    agent = Agent(callback_handler=None)
    async for event in agent.stream_async(
        "Draft a reply apologizing for a delayed refund.",
        structured_output_model=ReplyDraft,
    ):
        if "data" in event:
            print(event["data"], end="", flush=True)
        elif "result" in event:
            draft = event["result"].structured_output
            print(f"\nSubject: {draft.subject}")


asyncio.run(main())
```

화면에는 모델의 텍스트가 먼저 흘러나오고, 마지막에 검증된 `ReplyDraft`에서 꺼낸 제목이 찍힙니다.

## 위험한 도구 호출만 사람에게 묻는 위험 분류기 기반 승인

삭제나 환불처럼 되돌릴 수 없는 호출은 막아야 하지만, 모든 호출마다 승인을 물으면 결정 피로가 쌓여 사람들이 내용을 읽지 않고 승인하게 됩니다. 위험 분류기를 켠 `HumanInTheLoop` 인터벤션 핸들러는 이 문제를 줄입니다. 모델이 도구 호출을 하나씩 평가하고, 안전한 호출은 그대로 실행하며, 위험한 호출을 만났을 때만 사람이 승인하거나 거부할 때까지 에이전트를 멈춥니다.

### 위험한 호출에서만 뜨는 승인 프롬프트

티켓 분류 에이전트에 환불 도구를 붙여 보겠습니다. 고객 조회는 읽기 작업이고 환불은 돈이 실제로 나가는 작업입니다. 핸들러에 세 가지 인자를 켜서 `interventions`에 넘깁니다. `classifier=True`는 위험 분류기를 켜고, `ask="stdio"`는 승인 프롬프트를 터미널에 띄우며, `enable_trust=True`는 trust, 곧 같은 도구의 이후 호출까지 한 번에 승인하는 응답을 다룹니다. `lookup_customer`는 앞서 구조화된 출력 예제에서 정의한 도구를 그대로 씁니다. 이 예제는 승인 흐름만 보여 주므로 `TicketTriage` 스키마를 넘기지 않고, 반환값도 읽지 않습니다.

```python
from strands import Agent, tool
from strands.vended_interventions.hitl import HumanInTheLoop


@tool
def issue_refund(email: str, amount: float) -> str:
    """Refund the given amount to the customer with this email address."""
    return f"Refunded {amount:.2f} to {email}"


agent = Agent(
    tools=[lookup_customer, issue_refund],
    interventions=[
        HumanInTheLoop(
            classifier=True,
            ask="stdio",
            enable_trust=True,
            allowed_tools=["!issue_refund"],
        )
    ],
)
agent("From dana@example.com: I was charged twice this month and need one of the charges refunded.")
```

도구 호출 직전마다 분류기는 도구 이름과 입력을 모델에 보내 이 호출에 승인이 필요한지 묻습니다. 승인이 필요하다고 판단하면 핸들러가 터미널에 승인 프롬프트를 띄우고 에이전트는 답을 기다립니다. 프롬프트에는 도구 이름, 분류기가 든 이유, 호출 입력이 함께 표시됩니다. `ask="stdio"`일 때 받는 응답은 세 종류입니다.

- `y` 또는 `yes`: 이번 호출만 승인합니다.
- `t` 또는 `trust`: 이번 호출을 승인하고, 에이전트가 존재하는 동안 같은 도구의 이후 호출도 모두 승인합니다.
- 그 밖의 입력: 호출을 거부합니다. 에이전트는 거부됐다는 사실을 알게 되므로 사용자에게 알리거나 다른 방법을 시도할 수 있습니다.

상담원이 한 세션에서 환불을 여러 건 처리한다면 `t`로 첫 건만 확인하고 나머지는 넘길 수 있습니다. 반대로 환불처럼 매번 확인해야 하는 도구는 `allowed_tools=["!issue_refund"]`처럼 이름 앞에 `!`를 붙여 부정합니다. 부정된 도구는 항상 프롬프트를 띄우고 trust 대상이 될 수 없습니다.

### 심판 모델이 맡는 위험 판단

기본 분류기는 심판 역할을 하는 LLM이며, 기본적으로 에이전트의 모델에서 실행됩니다. System One 모델은 예/아니오 같은 닫힌 질문에 빠르게 답하도록 만든 작은 모델이며, Jev와 Strands Decider가 그 예입니다. 기본 분류기를 이런 모델로 바꾸거나 직접 작성한 분류기 함수로 바꿀 수도 있습니다. 분류기를 끄면 모든 도구 호출에서 프롬프트가 뜹니다. 분류기의 모델 호출이 실패하면 핸들러는 해당 호출의 승인을 사람에게 요청합니다. 따라서 분류기 장애가 생겨도 위험한 호출이 검사 없이 실행되지는 않습니다. 분류기는 위험한 호출을 안전하다고 잘못 판단할 수 있으며, 그 한계는 마지막 선택 가이드에서 다룹니다. 

### 승인보다 먼저 평가되는 Cedar 경계

일부 사용자만 호출할 수 있는 도구처럼 누가 승인해도 넘으면 안 되는 한계에는 `HumanInTheLoop` 앞에 Cedar Authorization을 둡니다. Cedar는 어떤 정책도 허용하지 않는 호출을 거부하기 때문에, 그런 호출은 분류기에도 사람에게도 도달하지 않습니다. 다음 예에서는 누구나 고객을 조회할 수 있고, 환불은 `alice`만 할 수 있습니다.

```python
from strands import Agent
from strands.vended_interventions.cedar import CedarAuthorization
from strands.vended_interventions.hitl import HumanInTheLoop

cedar = CedarAuthorization(
    policies="""
    permit(principal, action == Action::"lookup_customer", resource);

    permit(
        principal == User::"alice",
        action == Action::"issue_refund",
        resource
    );
    """,
    principal_resolver=lambda state: {"type": "User", "id": state["user_id"]},
)

agent = Agent(
    tools=[lookup_customer, issue_refund],
    interventions=[
        cedar,
        HumanInTheLoop(
            classifier=True,
            ask="stdio",
            enable_trust=True,
            allowed_tools=["!issue_refund"],
        ),
    ],
)

message = "From dana@example.com: I was charged twice this month and need one of the charges refunded."
# alice is asked to approve the refund
agent(message, invocation_state={"user_id": "alice"})
# bob's refund is denied without a prompt
agent(message, invocation_state={"user_id": "bob"})
```

`principal_resolver`는 호출에 넘긴 `invocation_state`에서 사용자 ID를 꺼내 Cedar 주체로 바꿉니다. `invocation_state`는 `agent(..., invocation_state=...)`로 호출마다 넘기는 딕셔너리이고, `principal_resolver` 같은 핸들러가 이 값을 읽으므로 호출한 사용자의 ID를 넣어 두기에 알맞습니다. 리스트 순서가 결과를 좌우합니다. 앞서 본 Deny 단락 평가 규칙 때문에 Cedar를 맨 앞에 둡니다. `bob`의 환불은 Cedar에서 바로 거부되어 분류기 모델 호출도 승인 프롬프트도 생기지 않습니다. `alice`의 환불은 Cedar를 통과한 뒤 분류기 판단에 따라 사람의 승인을 기다립니다.

## 턴·토큰 한도, 취소, 동시성 가드로 묶는 한 번의 호출

프로덕션에서 호출 하나를 통제하는 장치는 네 가지입니다. 작업량은 `limits`로 제한하고, 시간은 취소 신호로 끊고, 결과는 정지 이유로 해석하고, 동시 요청은 에이전트 인스턴스를 나눠서 받습니다. 아무 한도도 넘기지 않으면 에이전트는 모델이 도구를 더 요청하지 않을 때까지 계속 돌고, 매 턴마다 지금까지의 대화를 다시 보냅니다.

### 반복이 시작될 때마다 검사되는 호출 단위 예산

`limits`에는 턴 수(루프 반복 횟수), 출력 토큰, 전체 토큰, 그리고 앞서 다룬 구조화된 출력 시도 횟수의 상한을 줄 수 있습니다. 모두 선택 항목이며 각 값은 양의 정수여야 합니다. `invoke_async`와 `stream_async`도 같은 파라미터를 받습니다. TypeScript에서는 같은 옵션을 `stream`에 쓰고, 값은 양의 유한 수입니다. 티켓 분류 에이전트에 적용하면 다음과 같은 코드가 됩니다.

```python
from strands import Agent

agent = Agent(tools=[lookup_customer], callback_handler=None)
result = agent(
    "From dana@example.com: I was charged twice this month and need one of the charges refunded.",
    structured_output_model=TicketTriage,
    limits={"turns": 5, "output_tokens": 2000, "total_tokens": 10000},
)
if result.stop_reason == "limit_turns":
    print("Hit turn budget")
elif result.stop_reason == "limit_total_tokens":
    print("Hit token budget")
```

한도는 각 루프 반복이 시작될 때만 검사하고, 호출 도중에는 검사하지 않습니다. 그래서 한 턴이 토큰 예산을 넘겨 쓸 수 있습니다. 검사는 다음 턴이 시작되기 전에 이루어지며, 이전 턴이 요청한 도구는 그보다 먼저 끝까지 실행됩니다. 예를 들어 `total_tokens`가 10,000일 때 두 턴 동안 누적 사용량이 6,000이었는데 세 번째 턴 하나가 5,000을 쓰면, 누적 11,000이 되어 예산을 넘긴 뒤에야 네 번째 턴 직전 검사에서 멈춥니다. 여러 한도가 동시에 걸리면 정지 이유는 turns, total tokens, output tokens, 구조화된 출력 시도 순서로 하나만 보고됩니다.

전체 토큰 한도는 Bedrock, Anthropic, OpenAI, Gemini에서 프롬프트 캐시 읽기와 쓰기를 모두 셉니다. 따라서 캐시된 턴도 캐시되지 않은 턴만큼 한도를 소모하고, 이 한도가 제한하는 대상은 비용이 아니라 토큰입니다. 카운터는 호출마다 새로 시작합니다.

### 체크포인트에서만 반영되는 협력적 취소

`agent.cancel()`은 클라이언트 연결 끊김, 타임아웃, 화면의 "중지" 버튼처럼 루프 바깥에서 실행을 멈출 때 씁니다. 에이전트는 다음 네 지점에서 취소 여부를 확인합니다.

| 체크포인트 | 동작 | 비고 |
| --- | --- | --- |
| 모델 응답 스트리밍 중 | 부분 출력을 버립니다 | 모델이 마지막 메타데이터 이벤트를 보내기 전에 스트림이 닫혀 사용량 지표가 부정확할 수 있습니다 |
| 도구 실행 전 | 도구 호출을 건너뛰고, 대화 상태가 유효하도록 오류 결과를 추가합니다 | - |
| MCP 도구 실행 중 | 공유 MCP 세션은 닫지 않고 진행 중인 MCP 요청을 취소합니다 | 원격 취소는 최선 노력 방식이며, 서버가 확인하지 않아도 에이전트는 로컬에서 멈춥니다 |
| 도구 실행 후 | 다음 모델 호출 전에 멈춥니다 | MCP가 아닌 도구는 끝까지 실행된 뒤에 취소가 반영됩니다 |

취소된 호출의 결과는 `stop_reason="cancelled"`입니다. `cancel()`은 스레드 안전하고 멱등입니다. 다만 실행 중인 호출에만 써야 합니다. Python에서 유휴 상태의 에이전트에 `cancel()`을 부르면 다음 호출이 취소됩니다.

멈춰야 할 이유가 요청 하나에 속한다면 호출마다 `cancel_signal`을 넘기는 편이 안전합니다. 아래 코드는 타이머로 만든 데드라인을 `threading.Event`에 연결합니다.

```python
import threading

cancel_signal = threading.Event()
deadline = threading.Timer(30.0, cancel_signal.set)
deadline.start()
try:
    result = agent(
        "From dana@example.com: I was charged twice this month and need one of the charges refunded.",
        cancel_signal=cancel_signal,
    )
finally:
    deadline.cancel()
```

같은 이벤트를 연결 끊김 핸들러나 중지 버튼 핸들러에도 넘길 수 있습니다. 에이전트는 호출자가 소유한 이벤트를 설정하지도 지우지도 않습니다.

도구가 일단 시작되면 취소는 협력적입니다. 일찍 멈출지는 도구가 정합니다. 순차 실행에서는 아직 시작하지 않은 도구를 건너뛸 수 있습니다. 기본 동시 실행기에서는 형제 도구가 이미 실행 중일 수 있습니다. 오래 걸리는 도구는 단계 사이마다 신호를 확인합니다. 티켓 레코드를 일괄 정리하는 도구라면 다음처럼 씁니다.

```python
from strands import tool
from strands.types.tools import ToolContext


@tool(context=True)
def normalize_records(records: list[str], tool_context: ToolContext) -> str:
    """Normalize records while checking for cancellation between items."""
    normalized = []
    for record in records:
        if tool_context.cancel_signal.is_set():
            return f"cancelled after {len(normalized)} records"
        normalized.append(record.strip().upper())
    return "\n".join(normalized)
```

취소는 에이전트를 완전히 멈춥니다. 사람 입력을 기다렸다가 재개하는 interrupt와는 다른 동작입니다.

### 반환된 정지 이유와 발생한 예외는 별개의 결과

대부분의 호출은 `result.stop_reason`으로 다음 행동을 알려 줍니다.

- `end_turn`, `stop_sequence`: 완료된 결과를 반환합니다.
- `limit_*` 계열: 앞서 본 정지 이유 표를 참고합니다.
- `cancelled`: 호출자나 애플리케이션이 중단한 경우입니다. 조립 중이던 불완전한 모델 메시지는 결과와 기록에서 빠집니다. 이미 소비자에게 전달된 스트림 조각은 그대로 보입니다.
- `content_filtered`, `guardrail_intervened`: 앞서 본 정지 이유 표를 참고합니다.
- `interrupt`: 요청된 사람 입력을 받아 에이전트를 재개합니다.

예외는 정지 이유와 다른 경로로 옵니다. 도구 본문이 실패해도 기본적으로 루프는 멈추지 않습니다. 오류가 오류 결과로 모델에 돌아가고, 모델은 이를 보고 방법을 바꿉니다. 반면 라이프사이클 훅 실패, 소진된 모델 재시도, 잘못된 설정 같은 SDK 오류는 애플리케이션까지 전파됩니다. 모델 제공자 오류를 다시 시도하는 일은 재시도 전략이 맡습니다.

앞서 루프 안에서 복구할 수 없다고 한 `max_tokens`는 예외 형태로 나타납니다. 모델 응답 하나가 제공자의 호출당 출력 상한을 넘으면 결과를 반환하지 않고 `MaxTokensReachedException`이 발생합니다. 응답이 지나치게 길거나, 컨텍스트 윈도우가 거의 차 있거나, 도구 결과가 대화를 한도 가까이 밀어 올린 경우에 주로 일어납니다. Python SDK는 복구한 부분 메시지를 기록에 추가한 다음 예외를 던집니다. 그러니 요청을 줄이거나, 제공자의 출력 상한을 올리거나, 작업을 나눈 뒤 같은 에이전트를 다시 호출하면 그 기록에서 이어서 진행합니다.

### 요청마다 새로 만드는 에이전트

에이전트는 호출하는 동안 대화 기록을 바꿉니다. 그래서 기본 모드인 `THROW`에서 이미 실행 중인 에이전트를 다시 호출하면 `ConcurrencyException`이 발생합니다. 호출이 겹치면 메시지가 뒤섞여 기록이 손상되기 때문입니다. 같은 논리 요청이 재시도로 다시 들어오는 경우에는 `idempotency_token`을 넘깁니다. 같은 토큰으로 온 두 번째 호출은 원래 호출이 끝나기를 기다렸다가 같은 `AgentResult`를 받습니다. 토큰이 다르면 여전히 예외가 발생합니다. 원래 호출이 취소되는 등 결과 없이 끝나면, 기다리던 호출은 `IdempotencyAbortedError`를 받고 재시도 여부를 스스로 정합니다. `ConcurrentInvocationMode.UNSAFE_REENTRANT`로 에이전트를 만들면 이 가드가 사라지고 토큰도 무시됩니다. TypeScript에는 설정 가능한 모드도 토큰 중복 제거도 없으며, 겹치는 호출은 `ConcurrentInvocationError`를 던집니다.

동시 요청이 많은 서비스에서는 요청마다 새 에이전트를 만드는 패턴이 확장에 유리합니다. 인스턴스를 공유하는 요청이 없으니 동시성 가드를 신경 쓸 일이 없습니다. 아래 핸들러는 턴과 토큰, 데드라인을 제한하고 정지 이유를 타입 있는 결과로 바꿉니다. 응답 텍스트는 정상 완료일 때만 돌려줍니다.

```python
import threading
from dataclasses import dataclass
from enum import Enum

from strands import Agent


class Outcome(str, Enum):
    COMPLETED = "completed"
    BUDGET_EXCEEDED = "budget_exceeded"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    STOPPED = "stopped"


@dataclass
class Reply:
    outcome: Outcome
    text: str | None
    triage: TicketTriage | None


def classify(stop_reason: str) -> Outcome:
    if stop_reason in ("end_turn", "stop_sequence"):
        return Outcome.COMPLETED
    if stop_reason == "cancelled":
        return Outcome.CANCELLED
    if stop_reason in ("limit_turns", "limit_total_tokens",
                       "limit_output_tokens", "limit_structured_output_attempts"):
        return Outcome.BUDGET_EXCEEDED
    if stop_reason in ("content_filtered", "guardrail_intervened"):
        return Outcome.BLOCKED
    return Outcome.STOPPED


def handle_ticket(message: str, deadline_seconds: float = 20.0) -> Reply:
    agent = Agent(
        tools=[lookup_customer, issue_refund],
        interventions=[HumanInTheLoop(classifier=True, ask="stdio", enable_trust=True)],
        callback_handler=None,
    )
    cancel_signal = threading.Event()
    deadline = threading.Timer(deadline_seconds, cancel_signal.set)
    deadline.start()
    try:
        result = agent(message, structured_output_model=TicketTriage,
                       limits={"turns": 6, "total_tokens": 20_000},
                       cancel_signal=cancel_signal)
    finally:
        deadline.cancel()
    outcome = classify(result.stop_reason)
    return Reply(
        outcome=outcome,
        text=str(result) if outcome == Outcome.COMPLETED else None,
        triage=result.structured_output,
    )
```

예산 소진이나 취소로 멈췄을 때 이 핸들러는 예외를 던지지 않고 결과만 돌려줍니다. 반면 잘린 모델 응답처럼 SDK가 복구하지 못하는 오류는 요청 경계까지 그대로 올라옵니다. 이 무상태 패턴은 interrupt를 재개하지 못합니다. 앞서 붙인 `HumanInTheLoop`의 승인처럼 사람 입력이 필요한 흐름이라면, 에이전트나 세션 상태와 반환된 interrupt ID를 요청이 끝난 뒤에도 보존해야 합니다.

## 스로틀링만 지수 백오프로 재시도하는 기본 전략과 훅 기반 확장

모델 제공자 오류 재시도는 재시도 전략이 맡고, 불안정한 도구 재시도와 실행 횟수 제한은 훅이 맡습니다. 설정이 없으면 재시도 전략은 속도 제한(스로틀링) 오류만 지수 백오프로 다시 시도합니다. 시도 횟수, 대기 시간, 재시도할 오류의 범위는 전략을 바꿔 조정합니다.

### 설정 없이 동작하는 스로틀링 재시도

아무것도 설정하지 않은 에이전트는 스로틀링 오류를 최대 5번 재시도합니다. 처음 호출까지 합치면 6번 시도하며, 첫 대기는 4초입니다. 대기 시간은 4초, 8초, 16초, 32초, 64초 순으로 늘어나고, 6번째 시도마저 실패하면 예외가 발생합니다. 기본 전략이 다루는 예외는 모델 제공자가 속도 제한을 알리며 던지는 `ModelThrottledException`(TypeScript는 `ModelThrottledError`) 하나입니다. 다른 예외는 재시도하지 않고 바로 전파합니다. 앞서 SDK 오류가 애플리케이션까지 올라온다고 할 때 예로 든 "소진된 모델 재시도"는 6번째 시도가 실패한 뒤의 이 예외입니다.

Python의 `ModelRetryStrategy`는 파라미터 세 개를 받습니다.

| 파라미터 | 기본값 | 의미 |
| --- | --- | --- |
| `max_attempts` | `6` | 첫 호출을 포함한 전체 시도 횟수입니다. `1`이면 재시도가 꺼집니다 |
| `initial_delay` | `4` | 첫 재시도 전에 기다리는 초입니다. 이후 재시도마다 두 배가 됩니다 |
| `max_delay` | `128` | 재시도 사이 대기의 상한(초)으로, 지수 증가를 이 값에서 멈춥니다 |

 TypeScript의 `DefaultModelRetryStrategy`는 `maxAttempts`(기본 `6`)와 `backoff`(기본 `new ExponentialBackoff({ baseMs: 4000, maxMs: 240000 })`)를 받으므로 대기 상한의 기본값이 Python과 다릅니다.

### 생성자로 넘기는 전략과 재시도 끄기

전략을 바꾸려면 설정한 인스턴스를 에이전트 생성자에 넘기고, 자동 재시도를 아예 끄려면 `retry_strategy=None`을 넘깁니다. 재시도할 오류의 범위를 넓히거나 좁히려면 `ModelRetryStrategy`를 상속해 `is_retryable`을 오버라이드합니다. `super().is_retryable(exception)`을 호출하면 기본 동작인 스로틀링 재시도가 그대로 남습니다. 티켓 분류 에이전트가 타임아웃까지 재시도하도록 하는 전략은 이렇게 정의합니다.

```python
from strands import ModelRetryStrategy


class TriageRetryStrategy(ModelRetryStrategy):
    def is_retryable(self, exception: Exception) -> bool:
        return super().is_retryable(exception) or isinstance(exception, TimeoutError)
```

재시도 전략 인스턴스는 턴마다 상태를 들고 있으므로 여러 에이전트가 공유하면 안 되고, 에이전트마다 새로 만들어야 합니다. 앞서 만든 `handle_ticket`처럼 요청마다 에이전트를 만드는 구조라면 전략도 그 안에서 함께 만들어야 합니다.

### `AfterModelCallEvent`의 `retry`로 거는 모델 재시도

모델 오류를 확인하고 재시도만 요청하면 될 때는 전략 클래스보다 훅이 간단합니다. `AfterModelCallEvent` 콜백에서 `event.retry`를 참으로 설정하면 됩니다. 이 필드는 예외뿐 아니라 응답 검증이나 내용 품질 검사 같은 기준으로도 재시도를 요청할 수 있습니다. 훅은 대기 시간을 자동으로 넣지 않으므로 백오프는 콜백이 직접 기다려야 합니다. 아래 훅은 `ServiceUnavailable` 오류일 때만 지수 백오프로 재시도하고, 호출이 시작되거나 모델 호출이 성공하면 카운터를 초기화합니다.

```python
import asyncio
import logging

from strands import Agent
from strands.hooks import AfterModelCallEvent, BeforeInvocationEvent, HookProvider, HookRegistry

logger = logging.getLogger(__name__)


class RetryOnServiceUnavailable(HookProvider):
    """Retry model calls when ServiceUnavailable errors occur."""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.retry_count = 0

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeInvocationEvent, self.reset_counts)
        registry.add_callback(AfterModelCallEvent, self.handle_retry)

    def reset_counts(self, event: BeforeInvocationEvent = None) -> None:
        self.retry_count = 0

    async def handle_retry(self, event: AfterModelCallEvent) -> None:
        if event.exception:
            if "ServiceUnavailable" in str(event.exception):
                logger.info("ServiceUnavailable encountered")
                if self.retry_count < self.max_retries:
                    logger.info("Retrying model call")
                    self.retry_count += 1
                    event.retry = True
                    await asyncio.sleep(2 ** self.retry_count)  # Exponential backoff
        else:
            self.reset_counts()
```

TypeScript SDK의 `AfterModelCallEvent`는 1부터 세는 `attemptCount`를 제공합니다. 그래서 TypeScript에서는 위 예제의 `retry_count` 같은 카운터를 훅 바깥에 따로 두지 않고도 재시도 횟수를 제한할 수 있습니다.

### 같은 `tool_use_id`로 다시 실행하는 도구 재시도

앞서 본 대로 도구 본문의 실패는 오류 결과로 모델에 돌아가는데, 같은 도구를 그대로 한 번 더 돌리려면 `AfterToolCallEvent.retry`를 씁니다. 일시적 오류, 타임아웃, 불안정한 외부 서비스가 대상입니다. `retry`를 `True`로 두면 도구 실행기는 현재 결과를 버리고 같은 `tool_use_id`로 도구를 다시 호출합니다. 아래 훅은 `tool_use_id`별로 시도 횟수를 세고, 성공하면 기록을 지웁니다.

```python
import logging

from strands.hooks import AfterToolCallEvent, HookProvider, HookRegistry

logger = logging.getLogger(__name__)


class RetryOnToolError(HookProvider):
    """Retry tool calls that fail with errors."""

    def __init__(self, max_retries: int = 1):
        self.max_retries = max_retries
        self._attempt_counts: dict[str, int] = {}

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(AfterToolCallEvent, self.handle_retry)

    def handle_retry(self, event: AfterToolCallEvent) -> None:
        tool_use_id = str(event.tool_use.get("toolUseId", ""))
        tool_name = event.tool_use.get("name", "unknown")
        attempt = self._attempt_counts.get(tool_use_id, 0) + 1
        self._attempt_counts[tool_use_id] = attempt
        if event.result.get("status") == "error" and attempt <= self.max_retries:
            logger.info(f"Retrying tool '{tool_name}' (attempt {attempt}/{self.max_retries})")
            event.retry = True
        elif event.result.get("status") != "error":
            self._attempt_counts.pop(tool_use_id, None)


agent = Agent(
    tools=[lookup_customer],
    callback_handler=None,
    hooks=[RetryOnServiceUnavailable(max_retries=3), RetryOnToolError(max_retries=1)],
)
```

스트림을 소비하는 쪽은 재시도를 염두에 두어야 합니다. 버려진 시도에서 나온 `ToolStreamEvent`는 이미 호출자에게 전달된 상태입니다. 대화 기록에 들어가는 것은 마지막 시도의 `ToolResultEvent`뿐입니다. `lookup_customer`의 중간 진행 상황을 화면에 보여 주는 경우라면, 버려진 시도의 이벤트도 화면에 나타날 수 있습니다.

## 대화 기록·에이전트 상태·호출 상태의 세 층과 스냅샷 세션 영속화

Strands 에이전트의 상태는 수명이 서로 다른 세 층으로 나뉘고, 재시작 뒤에도 남겨야 하는 부분은 세션 매니저가 스냅샷으로 저장합니다. 세 층은 사용자와 에이전트가 주고받은 메시지 시퀀스인 대화 기록, 대화 컨텍스트 밖에서 요청 사이에 유지되는 키-값 에이전트 상태, 한 번의 호출 동안만 사는 호출 상태입니다. 대화 기록이 호출 사이에 유지되고 매 추론마다 모델에 들어가는 동작은 앞서 `agent.messages`를 다룰 때 이미 설명했습니다. 호출 상태는 한 호출 안에서 루프가 여러 번 돌아도 유지되지만 모델 컨텍스트에는 들어가지 않습니다.

| 구분 | 수명 | 모델이 보는지 여부 | 티켓 분류 에이전트에서의 쓰임 |
| --- | --- | --- | --- |
| 대화 기록 | 호출 사이에 유지 | 매 추론에 포함 | 고객과 주고받은 메시지 |
| 에이전트 상태 | 요청 사이에 유지 | 포함되지 않음 | 분류 횟수, 마지막 분류 결과 |
| 호출 상태 | 한 호출 동안만 | 포함되지 않음 | 요청 ID, 사용자 ID, DB 연결 |

### 요청 사이에 남는 키-값 에이전트 상태

Python에서는 에이전트 상태의 초기값을 생성자의 `state`로 주고, `agent.state`의 `get`·`set`·`delete`로 읽고 씁니다. TypeScript는 같은 역할을 `appState`가 맡습니다. 값은 영속화와 복원을 거쳐야 하므로 JSON 직렬화 검증을 통과해야 합니다. Python에서는 직렬화할 수 없는 값을 넣으면 `ValueError`가 발생합니다.

```python
from strands import Agent

agent = Agent(
    tools=[lookup_customer],
    callback_handler=None,
    state={"triage_count": 0},
)

agent.state.set("triage_count", agent.state.get("triage_count") + 1)
agent.state.set("last_category", "billing")
print(agent.state.get())  # 전체 상태를 dict로 반환
agent.state.delete("last_category")

try:
    agent.state.set("formatter", lambda x: x)  # JSON으로 직렬화할 수 없는 값
except ValueError as e:
    print(f"Error: {e}")
```

도구 안에서는 `tool_context.agent.state`로 같은 상태에 접근하며, 이때 매개변수 이름은 반드시 `tool_context`여야 합니다.

### 한 호출 동안만 사는 호출 상태

호출 상태는 호출할 때 넘기는 `invocation_state` 인자에서 만들어지고, 생략하면 `{}`가 됩니다. 훅과 도구는 이 딕셔너리를 참조로 공유하므로, 앞선 훅이나 도구가 바꾼 값을 같은 호출 안의 뒤쪽 훅과 도구가 그대로 봅니다.

```python
from strands import Agent, ToolContext, tool


@tool(context=True)
def whoami(tool_context: ToolContext) -> str:
    """Return the user ID carried in the invocation state."""
    user_id = tool_context.invocation_state.get("user_id", "unknown")
    return f"Current user: {user_id}"


agent = Agent(tools=[lookup_customer, whoami], callback_handler=None)
result = agent("Who am I?", invocation_state={"request_id": "r-42", "user_id": "u-1"})
```

훅에서는 `event.invocation_state`로 같은 값을 읽습니다. 그래서 DB 클라이언트 같은 커스텀 객체나 세션 ID 같은 요청 컨텍스트를 훅에 건네는 통로로 씁니다. JSON 검증을 받는 에이전트 상태와 달리, 직렬화할 수 없는 연결 객체도 여기에 담을 수 있습니다. 결과 객체에서 돌려받는 범위는 SDK마다 다릅니다. TypeScript는 호출 상태 전체를 `result.invocationState`로 반환하고, Python은 그중 `request_state` 슬롯만 `result.state`로 반환합니다.

### 버전 있는 스냅샷으로 저장하는 세션

앱이 재시작돼도 에이전트 상태와 대화 기록이 남게 하려면 세션 매니저를 붙입니다. Python의 `SnapshotSessionManager`는 버전이 붙은 `Snapshot` JSON blob 하나를 저장합니다. 이 blob에는 메시지, 에이전트 상태, 대화 매니저 상태, 인터럽트 상태, 모델 상태, 시스템 프롬프트가 들어 있습니다. 티켓마다 세션 ID를 하나씩 두면 구성이 이렇게 됩니다.

```python
from strands import Agent
from strands.session import SnapshotSessionManager
from strands.storage import LocalFileStorage

session_manager = SnapshotSessionManager(
    session_id="ticket-1042",
    storage=LocalFileStorage("./sessions/"),
    snapshot_trigger=lambda *, agent_data, **_: len(agent_data.messages) % 4 == 0,
)
agent = Agent(
    tools=[lookup_customer],
    callback_handler=None,
    session_manager=session_manager,
)

result = agent(
    "From dana@example.com: I was charged twice this month and need one of the charges refunded.",
    structured_output_model=TicketTriage,
)
```

세션 매니저는 초기화할 때 `snapshot_latest`를 복원합니다. 저장 시점은 `save_latest_on`이 정하며, 기본값 `"invocation"`이면 호출이 끝날 때 저장하고 `"message"`이면 메시지가 추가될 때마다 저장합니다. 이 시점 밖에서 바꾼 상태는 다음 저장 전까지 영속화되지 않습니다. `snapshot_trigger` 콜백은 따로 동작합니다. 위 코드처럼 메시지가 4개 쌓일 때마다 참을 반환하면 그 시점에 불변 스냅샷이 하나 생깁니다. 운영 환경에서는 `LocalFileStorage` 대신 `S3Storage(bucket=..., prefix=...)`를 넘깁니다. 저장소를 아무것도 지정하지 않으면 Python은 `LocalFileStorage("./.strands/")`를 씁니다.

앞서 만든 `handle_ticket`은 요청마다 에이전트를 새로 만듭니다. 그 안의 에이전트 생성 코드에서 승인 훅·한도·재시도 설정은 그대로 두고, 티켓의 세션 ID로 만든 `session_manager`만 더 넘깁니다. 이렇게 하면 새로 만든 에이전트가 초기화 단계에서 직전 대화를 복원합니다.

### 시간순으로 정렬되는 스냅샷 목록과 복원·삭제

스냅샷 ID는 UUID v7이어서 사전순으로 정렬하면 곧 시간순이 됩니다. `save_snapshot`으로 불변 스냅샷을 강제로 남기고, `list_snapshot_ids`로 목록을 읽고, `restore_snapshot`으로 특정 시점으로 되돌립니다.

```python
import asyncio


async def rewind() -> None:
    snapshot_id = await session_manager.save_snapshot(agent, is_latest=False)
    snapshot_ids = await session_manager.list_snapshot_ids(agent)
    assert snapshot_id in snapshot_ids
    await session_manager.restore_snapshot(agent, snapshot_id=snapshot_id)


asyncio.run(rewind())
```

티켓이 종결되면 `asyncio.run(session_manager.delete_session())`을 호출합니다. 이 호출은 세션 루트 디렉터리 전체, S3라면 세션 prefix 아래의 객체 전부를 지웁니다. 운영 환경에서는 TTL 같은 정리 전략을 함께 두는 편이 좋습니다.

### 세션 ID 범위와 멀티 에이전트 세션

세션 ID는 대화마다 하나씩 줍니다. 여러 에이전트가 한 세션 ID를 공유해도 에이전트 ID만 다르면 세션이 네임스페이스처럼 동작하고, 각 에이전트는 자기 메시지와 상태를 따로 유지합니다. Graph와 Swarm, 그리고 기존 repository 형식의 세션에는 호환 경로인 `FileSessionManager`와 `S3SessionManager`를 씁니다. 이 둘은 `Storage` 백엔드 대신 저장 디렉터리나 S3 버킷을 직접 받습니다. 멀티 에이전트 세션은 기본값 `multi_agent_save_latest_on="node"`에서 노드가 끝날 때마다 저장하고 호출이 끝날 때 한 번 더 저장합니다. `"invocation"`으로 바꾸면 전체 호출이 끝날 때만 저장합니다. 세션 저장소에는 락이 없고 저장소 자체를 신뢰 경계로 다뤄야 하는데, 이 위험은 마지막 선택 기준에서 함께 정리합니다.

## 모델 호출과 도구 호출을 스팬으로 남기는 OpenTelemetry 트레이싱

루프가 실제로 무엇을 했는지 보려면 에이전트를 만들기 전에 exporter를 하나 설정하면 되고, 그러면 그 에이전트의 모든 모델 호출과 도구 호출이 스팬(span)으로 기록됩니다. 관측 데이터는 세 종류입니다. 트레이스는 모델 상호작용, 도구 호출, 토큰 사용을 담고, 메트릭은 실행마다 토큰, 지연, 도구 호출 수, 이벤트 루프 사이클을 집계하며, 로그는 SDK 내부 동작을 드러냅니다.

### 콘솔에서 먼저 확인하는 스팬 출력

콘솔 출력은 collector가 없어도 동작하므로 운영 백엔드에 연결하기 전에 트레이싱이 켜졌는지 확인하기에 가장 빠릅니다. 티켓 분류 에이전트에 붙이면 아래와 같습니다. `setup_console_exporter()`는 `Agent` 생성 앞에 둡니다.

```python
# excerpt: top of the ticket-triage module; handle_ticket is the function assembled in earlier sections
from strands.telemetry import StrandsTelemetry

# Print every span to the console; add setup_otlp_exporter() to ship to a collector
StrandsTelemetry().setup_console_exporter()

triage = handle_ticket(
    "From dana@example.com: I was charged twice this month and need one of the charges refunded."
)
print(triage)
```

TypeScript에서는 `setupTracer({ exporters: { console: true } })`가 같은 역할을 합니다.

### OTLP로 보내는 같은 스팬

콘솔 exporter를 OTLP exporter로 바꾸면 같은 스팬을 Jaeger, Grafana Tempo, AWS X-Ray, Datadog 같은 OpenTelemetry 백엔드로 보냅니다. Python에서는 위 코드의 주석처럼 같은 `StrandsTelemetry()` 객체에서 `setup_otlp_exporter()`를 추가로 호출해 collector로 보냅니다.

```python
from strands.telemetry import StrandsTelemetry

telemetry = StrandsTelemetry()
telemetry.setup_console_exporter()
telemetry.setup_otlp_exporter()
```

New Relic 연동을 다룬 한 서드파티 블로그는 Strands 내장 텔레메트리 헬퍼가 트레이스와 메트릭을 OTEL 환경 변수로 구성한다고 보고합니다. Datadog 블로그는 Strands가 모델 추론, 플래너 단계, 도구 호출, 에이전트 간 핸드오프에 대해 OpenTelemetry 호환 스팬을 낸다고 보고합니다.

### 트레이스 트리에서 드러나는 루프 결함

Datadog에서는 한 번의 실행이 트리 하나로 보입니다. 아래 스크린샷은 복잡한 Strands 에이전트의 실행 예시로, 사용자 프롬프트에서 시작해 플래너, 도구 호출, 서브 에이전트를 거쳐 최종 출력으로 끝납니다. 스팬마다 지연, 토큰, 도구 호출 수, 모델 ID, 응답 상태가 함께 붙어 있습니다.

![Strands 에이전트 실행의 트레이스 트리와 스팬별 지연·토큰 정보](https://web-assets.dd-static.net/42588/1776300811-llm-aws-strands-strands_hero_highres.png?format=auto&fit=bounds&quality=75&disable=upscale&width=1400&dpr=1)

트리에서 같은 도구 스팬이 연달아 나오면 루프 결함을 의심합니다. 지연과 토큰 비용이 계속 늘던 가상의 리서치 어시스턴트라면, Datadog 트레이스에서 같은 요약 도구 스팬이 반복되는 루프가 보이고, 이 로직 오류를 고치면 지연과 토큰 사용이 줄어듭니다. 티켓 분류 에이전트라면 `lookup_customer` 스팬이 반복되는지, 구조화된 출력 도구 호출이 검증 실패로 몇 번 되풀이되는지를 같은 방식으로 살펴봅니다. 다만 트레이스로는 에이전트가 무엇을 했는지만 알 수 있습니다. 그 행동이 올바랐는지는 데이터셋을 기준으로 하는 평가가 따로 측정합니다.

### 직접 메워야 하는 로그 브리지와 백엔드별 속성 조회

앞서 언급한 New Relic 연동 서드파티 블로그는 New Relic에 연결할 때 직접 메워야 하는 지점 두 곳을 보고합니다. 첫째, 이 블로그에 따르면 트레이스·메트릭과 달리 로그는 Python 표준 `logging`에서 OTLP로 수동으로 연결해야 합니다. 둘째, 같은 블로그는 Python SDK에서 Strands가 프롬프트, 응답, 도구 입출력 같은 GenAI 텔레메트리를 OTLP 스팬 이벤트로 기록한다고 설명합니다.

이벤트와 속성을 어떻게 저장하고 조회하는지는 백엔드마다 다릅니다. 따라서 운영 백엔드를 고른 뒤에는 콘솔 출력과 백엔드 화면을 같은 호출로 나란히 비교해, 필요한 필드가 실제로 조회되는지 먼저 확인하는 것이 좋습니다.

## Strands 루프가 맞는 경우와 하네스·분류기·세션 저장소의 한계

에이전트 루프를 직접 조립하고 제어해야 하는 애플리케이션이라면 Strands SDK가 맞습니다. 다만 운영에 들어가기 전에 네 가지 한계를 설계에 반영해야 합니다. 위험 분류기는 보장이 아닙니다. 내장 세션 저장소에는 분산 락이 없고, 세션 저장 디렉터리는 신뢰 경계가 되어 주지 않습니다. 텔레메트리에 대해서는 New Relic 블로그가 도구 입출력 내용을 기본으로 캡처한다고 보고합니다.

### SDK 루프와 하네스 사이의 선택

앞서 본 대로 SDK는 직접 짜던 루프를 대신합니다. 티켓 분류 에이전트처럼 기존 서비스 안에서 요청마다 에이전트를 만들고 `TicketTriage` 객체를 받아 후속 처리로 넘기는 경우에는 SDK 쪽이 자연스럽습니다. 루프의 각 단계를 코드로 붙잡아야 하기 때문입니다. 루프의 단계를 코드로 붙잡을 필요가 없다면 하네스도 함께 검토할 만합니다.

### 판단일 뿐 보장이 아닌 위험 분류기

앞서 붙인 `HumanInTheLoop`의 분류기는 판단을 내릴 뿐이라서, 위험한 호출을 안전하다고 평가하는 경우가 생깁니다. 앞서 본 `!` 부정 목록(`allowed_tools=["!issue_refund"]`)으로 이런 도구는 항상 묻게 합니다. TypeScript에서는 `allowedTools: ['!issue_refund']`로 씁니다. 누구도 승인으로 넘을 수 없어야 하는 경계는 분류기에 맡기지 말고, 앞서 배치한 Cedar Authorization 정책으로 강제합니다.

### 락도 신뢰 경계도 없는 내장 세션 저장소

내장 세션 매니저는 분산 락을 잡지 않습니다. 단일 인스턴스 호출 가드도 프로세스 안에서만 동작합니다. 그래서 세션 매니저도 호출 가드도 다른 프로세스나 호스트에서 도는 두 번째 쓰기 주체를 감지하지 못합니다. 같은 세션 ID와 에이전트 ID를 쓰는 에이전트가 동시에 살아 있으면 문제가 생깁니다. 호출이 겹치면 서로의 턴을 덮어쓰고, 순차 호출이면 두 대화가 한 기록으로 합쳐지며, 어느 경우에도 오류는 나지 않습니다. 기본 ID(`agent_id="default"`, TypeScript는 `id: 'agent'`)를 그대로 쓰면 따로 실행되는 프로세스 사이에서도 이런 충돌이 우연히 일어나기 쉽습니다. 세션 생성도 원자적이지 않습니다. 확인한 뒤 쓰는 방식이라, 새 세션 ID로 동시에 콜드 스타트하면 두 요청이 모두 성공하고 나중에 쓴 쪽이 남습니다.

보안 측면에서는 세션 저장 디렉터리 자체가 신뢰 데이터 저장소입니다. 클론한 저장소에 들어 있는 세션 파일처럼 출처를 신뢰할 수 없는 세션은 로드하지 말아야 하고, 파일 시스템 권한은 에이전트 프로세스만 읽고 쓸 수 있도록 제한해야 합니다. SDK는 저장 디렉터리 안의 심볼릭 링크를 막지 않습니다. 쓰기 권한을 얻은 공격자가 `message_0.json`을 임의의 파일로 연결하면, 그 파일 내용이 대화 기록에 로드될 수 있습니다. 세션 매니저는 스레드 안전하지도 않습니다. 티켓 에이전트를 여러 워커로 확장할 때는 다음 세 가지를 지킵니다.

- 티켓마다 고유한 세션 ID를 씁니다.
- 기본값이 아닌 에이전트 ID를 명시합니다.
- 한 세션은 한 번에 한 워커만 다루도록 애플리케이션 쪽에서 배타 제어를 둡니다.

### 기본으로 민감 데이터를 담는 텔레메트리

New Relic 블로그는 Strands의 GenAI 텔레메트리가 설정과 무관하게 도구 입출력의 메시지 내용을 기본으로 캡처한다고 보고하며, 민감 데이터를 막으려면 트레이서에서 직접 걸러야 한다고 권합니다. 이 보고대로라면 티켓 에이전트에서는 `lookup_customer`가 돌려준 고객 정보와 환불 메시지가 스팬에 실릴 수 있습니다. OTLP 백엔드로 내보내기 전에 콘솔 exporter 출력에서 어떤 필드가 나가는지 먼저 확인하는 편이 안전합니다.

| 한계 | 증상 | 대응 |
|---|---|---|
| 위험 분류기의 오판 | 위험한 호출을 안전하다고 평가함 | `!`로 부정한 도구 지정, Cedar 정책 선배치 |
| 분산 락 부재 | 오류 없이 턴 덮어쓰기 또는 대화 병합 | 고유 세션·에이전트 ID, 애플리케이션 수준 배타 제어 |
| 비원자적 세션 생성 | 동시 콜드 스타트에서 나중 쓰기만 남음 | 세션 생성 경로 직렬화 |
| 신뢰되지 않는 저장 디렉터리 | 심볼릭 링크를 따라 임의 파일 로드 | 권한 제한, 외부 세션 로드 금지 |
| 텔레메트리 내용 캡처(New Relic 블로그 보고) | 도구 입출력이 스팬에 기록된다는 보고 | 트레이서 단계 필터링 |

Strands SDK는 루프 제어가 필요하고 운영 경계를 직접 책임질 수 있는 팀에 잘 맞습니다. 승인은 분류기, 부정 목록, Cedar를 겹겹이 두는 구성으로 다루고, 세션과 텔레메트리는 SDK가 지켜 주지 않는 영역으로 여기고 저장소 락, 권한, 필터링을 애플리케이션이 맡아야 합니다.

- - -

### Sources
* [Agent Loop &#124; Strands Agents](https://strandsagents.com/docs/user-guide/sdk/agents/agent-loop/)
* [Production Lifecycle Controls &#124; Strands Agents](https://strandsagents.com/docs/user-guide/sdk/agents/lifecycle-controls/)
* [Return structured output &#124; Strands Agents](https://strandsagents.com/docs/user-guide/sdk/agents/structured-output/)
* [Structured output &#124; Strands Agents](https://strandsagents.com/docs/user-guide/sdk/agents/structured-output-behavior/)
* [Interventions &#124; Strands Agents](https://strandsagents.com/docs/user-guide/sdk/agents/interventions/)
* [Approve before your agent acts &#124; Strands Agents](https://strandsagents.com/docs/user-guide/sdk/agents/approve-before-acting/)
* [State Management &#124; Strands Agents](https://strandsagents.com/docs/user-guide/sdk/agents/state/)
* [Hooks &#124; Strands Agents](https://strandsagents.com/docs/user-guide/sdk/agents/hooks/)
* [Retry Strategies &#124; Strands Agents](https://strandsagents.com/docs/user-guide/sdk/agents/retry-strategies/)
* [Persist state across sessions &#124; Strands Agents](https://strandsagents.com/docs/user-guide/sdk/agents/session-management/)
* [Observe your agent &#124; Strands Agents](https://strandsagents.com/docs/user-guide/sdk/observability-evaluation/)
* [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/html/2210.03629)
* [strands-agents · PyPI](https://pypi.org/project/strands-agents/)
* [Traces to your AI Agents with OpenTelemetry and Strands SDK &#124; New Relic](https://newrelic.com/blog/observability/traces-for-ai-agents-otel-strand)
* [Gain visibility into Strands Agents workflows with Datadog Agent Observability &#124; Datadog](https://www.datadoghq.com/blog/llm-aws-strands/)
