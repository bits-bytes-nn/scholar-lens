---
layout: post
title: "Strands Agents Loop Controls: Schemas, Approvals, Limits, Sessions"
date: 2026-11-09 14:57:45
description: "Strands Agents에서 에이전트 루프를 제어하는 방법을 다룹니다. 훅과 인터벤션으로 도구 호출을 가로채고, 구조화된 출력, 턴과 토큰 한도, 재시도, 세션 영속화를 티켓 분류 예제로 정리합니다. 승인 정책과 Cedar 권한 검사도 함께 봅니다."
categories: ["Tech Guides", "Agentic-AI"]
tags: ["Lifecycle-Hooks", "Structured-Output", "Human-in-the-Loop", "Retry-Strategy", "Session-Management", "Agentic-AI"]
cover: /assets/images/agentic-ai.jpg
---
## TL;DR

Strands Agents는 모델이 도구 호출 여부와 순서를 정하는 에이전트 루프를 중심에 둔 오픈소스 SDK입니다. 루프 위에 붙은 제어 장치로 반복 횟수, 응답 형식, 도구 실행을 통제합니다. Python은 PyPI의 `strands-agents`, TypeScript는 npm의 `@strands-agents/sdk`로 설치하며, 모든 기능이 이 에이전트 루프를 기반으로 합니다. 루프는 모델을 호출하고, 모델이 도구를 원하면 도구를 실행하고, 그 결과를 넣어 모델을 다시 부릅니다. 모델이 최종 응답을 낼 때까지 이 과정을 되풀이하고, 매 반복마다 대화 기록이 쌓입니다. 프로바이더와 무관하게 설계되어 같은 에이전트 코드가 지원되는 모든 프로바이더에서 그대로 동작합니다.

루프를 다루는 수단은 다섯 갈래입니다. 첫째는 훅입니다. 라이프사이클의 이벤트 지점에 타입 안전한 콜백을 여러 개 등록하는 구조이고, 도구 재실행을 요청하는 `AfterToolCallEvent.retry` 같은 필드도 훅에서 다룹니다. 둘째는 인터벤션입니다. 훅 위에 구축된 장치로, 인터벤션 핸들러는 이벤트를 직접 고치지 않고 Proceed, Deny, Guide, Confirm, Transform 중 하나를 타입이 정해진 액션 객체로 반환하며 프레임워크가 그 액션을 해석합니다. 그중 `Confirm`은 도구 실행 전에 사람 승인을 기다립니다. 셋째인 구조화된 출력은 스키마를 도구로 바꾸어 모델이 그 도구를 호출하게 합니다. 모델이 도구를 호출하지 않고 텍스트로 턴을 끝내면 SDK가 이 도구를 필수로 지정해 한 번 더 재시도합니다. 실패 횟수는 에이전트를 호출할 때 `limits` 인자로 넘기는 딕셔너리의 `structured_output_attempts` 키로 제한합니다(TypeScript는 `limits.structuredOutputAttempts`). 넷째는 한도와 취소입니다. 같은 `limits` 인자에 `turns`, `output_tokens`, `total_tokens` 상한을 걸고 결과의 `stop_reason`으로 어느 한도에 닿았는지 확인합니다. Python에서는 호출에 `cancel_signal`로 `threading.Event`를 넘겨 타이머로 데드라인을 걸 수 있습니다. Python에서 에이전트의 `cancel()`로 취소한 호출은 `stop_reason="cancelled"`인 결과를 반환합니다. 다섯째인 세션 영속화는 대화 기록, 키-값 에이전트 상태, Conversation Manager 같은 상태 정보를 자동으로 저장·복원해 에이전트가 대화를 이어 가게 합니다. 모델이 도구 순서를 정하는 에이전트를 여러 프로바이더에서 같은 코드로 운영하면서, 루프의 각 지점을 훅과 한도로 통제하고 싶을 때 맞는 도구입니다.

## 도구와 목표만 주고 비켜서는 모델 주도 에이전트의 자리

모델에 계획을 맡기면 애플리케이션 코드는 단순해지지만, 그 대가로 반복 횟수, 응답 형식, 도구 실행을 통제할 장치가 따로 있어야 하며, Strands는 이 장치들을 프로바이더와 무관한 하나의 에이전트 루프(agent loop) 위에 얹어 둔 SDK입니다.

언어 모델은 질문에 답할 수 있지만, 에이전트는 일을 처리합니다. 학습만으로 해결할 수 없는 요청이 들어오면 모델은 파일을 읽고, 데이터베이스를 조회하고, API를 부르고, 코드를 실행해야 합니다. 에이전트 루프는 여러 단계와 외부 정보, 실제 부수 효과가 필요한 문제를 풀 수 있도록 이 추론과 행동의 반복을 관리하는 조정 계층입니다. 이 글 전체에서 쓰는 고객 지원 티켓 분류 에이전트가 이런 경우에 해당합니다. "주문이 안 왔어요"라는 티켓 하나를 분류하려 해도 주문 상태를 조회해야 하고, 그 결과를 보고 환불 여부를 판단해야 하며, 마지막에는 정해진 형식의 분류 결과를 돌려줘야 합니다. 프롬프트 한 번으로는 조회 결과를 보고 다음 행동을 고르는 과정을 담을 수 없습니다.

이 반복을 누가 설계하느냐에 따라 프레임워크의 성격이 갈립니다. 개발자가 단계와 분기를 그래프로 미리 그려 두는 방식도 있지만, 한 서드파티 글은 Strands를 모델 주도(model-driven) 철학으로 분류합니다. 모델에 도구와 목표를 주고 비켜서는 방식이며, 저자는 모델의 도구 사용과 다단계 계획 능력이 좋아져서 이 방식이 현실적인 선택이 되었다고 봅니다. 티켓 분류 에이전트에서도 개발자는 `lookup_order`와 `issue_refund` 도구를 제공하고 목표만 설명합니다. 어떤 순서로 도구를 부를지는 모델이 정합니다.

모델 주도 루프의 위험은 이 위임에서 생깁니다. 계획을 모델이 세우므로 루프가 끝없이 돌 수 있고, 마지막 응답이 기대한 형식을 벗어날 수 있으며, 환불처럼 되돌리기 어려운 도구가 확인 없이 실행될 수 있습니다. Strands의 에이전트 루프에는 프로덕션 에이전트에 필요한 제어 장치가 처음부터 들어 있습니다. 위험과 장치를 짝지으면 다음 표와 같습니다.

| 위험 | 루프가 제공하는 장치 |
|---|---|
| 끝없는 반복과 비용 폭주 | 한 번의 호출이 쓸 수 있는 턴과 토큰을 제한하는 호출 한도 |
| 멈춰야 할 때 멈추지 않는 실행 | 실행 중인 루프를 중간에 멈추는 취소 |
| 왜 끝났는지 모르는 결과 | 루프가 끝난 이유를 알려 주는 정지 사유 |
| 여러 대화의 동시 처리 | 한 에이전트로 여러 대화를 동시에 처리하는 동시 호출 |
| 위험한 단계와 일시적 오류 | 모든 단계를 가로채는 훅, 모델·도구 오류에서 복구하는 재시도 전략 |

이 장치들이 모두 루프 위에 쌓여 있어서, Strands에서는 에이전트 루프가 나머지 모든 기능의 기반입니다. 티켓 분류 에이전트는 Pydantic 스키마 `TicketTriage`로 검증된 분류 결과를 받는 데서 시작해 이런 장치를 하나씩 덧붙여 갑니다.

루프가 특정 모델 서비스에 묶여 있지 않은 점도 중요합니다. SDK는 프로바이더에 독립적이어서 같은 에이전트 코드가 지원되는 모든 모델 프로바이더에서 동작하고, 에이전트를 다시 쓰지 않고 프로바이더를 바꾸거나 한 애플리케이션에서 여러 프로바이더를 함께 쓸 수 있습니다. 직접 지원하는 프로바이더는 Amazon Bedrock, Anthropic, OpenAI, Google이고, 커뮤니티 패키지가 더 많은 프로바이더를 추가하며, 그 밖의 서비스는 사용자 정의 프로바이더 인터페이스로 연결합니다. 이 덕분에 티켓 분류 에이전트에 앞으로 붙일 한도·승인·재시도 로직은 모델을 바꿔도 그대로 유지됩니다. SDK는 PyPI에서 `strands-agents`, npm에서 `@strands-agents/sdk`로 배포되며, 이 글의 코드는 Python입니다.

## 재귀로 이어지는 추론-행동-관찰 사이클

요청 하나가 최종 응답이 되는 과정은 "모델 호출 → 멈춘 이유 확인 → 도구 실행"이라는 사이클이 여러 번 이어진 결과입니다. Python SDK에서는 이 사이클이 반복문이 아니라 재귀 호출로 연결되므로, 턴이 하나 늘 때마다 스택 프레임도 하나씩 쌓입니다.

### 대화 기록이 쌓이는 한 바퀴

루프의 원리는 단순합니다. 모델을 호출하고, 모델이 도구를 쓰려 하는지 확인하고, 그렇다면 도구를 실행한 뒤 그 결과를 넣어 모델을 다시 호출합니다. 모델이 최종 응답을 낼 때까지 이 과정을 되풀이합니다. 반복할 때마다 대화 기록이 늘어나므로, 모델은 원래 요청뿐 아니라 지금까지 호출한 모든 도구와 받은 모든 결과를 함께 봅니다. 여러 단계에 걸친 추론은 이렇게 커지는 맥락 덕분에 가능합니다.

티켓 분류 에이전트에 대입하면, 두 번째 모델 호출은 "주문이 안 왔어요"라는 티켓과 함께 `lookup_order`가 돌려준 배송 상태까지 입력으로 받습니다. 모델이 환불할지 판단할 근거는 도구 결과가 대화 기록에 붙은 다음에야 생깁니다.

### 출구가 셋인 사이클 함수

이벤트 루프 모듈이 맡는 일은 네 가지입니다. 대화 메시지를 처리하고, 모델의 요청에 따라 도구를 실행하고, 오류를 복구하고, 재귀 실행 사이클을 관리합니다. 이 내용은 개발 브랜치(main)의 `event_loop.py` 기준입니다. 별도의 스케줄러 없이 함수 하나에 출구가 셋입니다.

아래 그림은 이 글의 설명을 상자와 화살표로 옮긴 예시입니다.

```text
            ┌──────────────────────────────┐
요청 ──────▶│ 사이클 시작 (id 부여)           │
            └──────────────┬───────────────┘
                           │ 최신 메시지에 toolUse가 없으면
                           ▼
                    ┌─────────────┐
                    │  모델 호출    │
                    └──────┬──────┘
                           │ stop_reason
        ┌──────────────────┼──────────────────────┐
        ▼                  ▼                      ▼
   max_tokens          tool_use                 그 밖
   예외 발생        도구 실행 → 다음 사이클을      EventLoopStopEvent
                   재귀 호출                   (최종 응답)
```

세 갈래를 표로 정리하면 다음과 같습니다.

| `stop_reason` | 사이클의 처리 | 결과 |
|---|---|---|
| `max_tokens` | 예외를 던집니다 | 호출이 오류로 끝납니다 |
| `tool_use` | 요청된 도구를 실행하고 다음 사이클로 넘어갑니다 | 도구 결과가 붙은 기록으로 모델을 다시 부릅니다 |
| 그 외 | `EventLoopStopEvent`를 냅니다 | 루프가 최종 응답으로 끝납니다 |

표의 내용은 위 글이 Python 소스를 분석한 결과입니다. 도구를 실행한 사이클 자신은 거기서 끝나고, 모델 재호출은 다음 사이클이 맡습니다. `max_tokens`로 끝난 호출을 프로덕션에서 어떻게 다루는지는 라이프사이클 제어 부분에서 설명합니다.

### 모델 호출 앞뒤의 두 갈림목

세 출구 외에도 모델 호출 직전과 직후에 경로가 바뀌는 지점이 있습니다. 같은 글에 따르면 `BeforeModelCallEvent` 훅(hook)이 `cancel`을 설정하면 모델에 요청을 보내지 않고 `end_turn`으로 사이클이 정상 종료되며, 다른 훅이 이를 재시도로 바꿀 수도 있습니다. 개발 버전(main)의 `event_loop.py`에서는 모델을 호출한 뒤 `AfterModelCallEvent`의 `retry` 플래그를 확인합니다. 플래그가 켜져 있으면 사용량과 메트릭을 갱신하고 `continue`로 모델을 다시 호출하며, 꺼져 있으면 호출 루프를 빠져나옵니다. 예외가 났을 때도 `AfterModelCallEvent`가 `exception`과 함께 발행됩니다. 사이클 사이를 잇는 것은 재귀이지만, 같은 사이클 안에서 모델을 다시 부르는 재시도는 이 내부 반복이 담당합니다. 이 플래그를 훅에서 어떻게 설정하는지는 훅 부분에서 다룹니다.

### while 없는 연쇄와 스택 깊이

사이클을 이어 가는 `while` 문은 없습니다. 다음 사이클은 `recurse_event_loop` 재귀 호출로 진행되므로, 턴 수가 곧 Python 스택 깊이입니다. 앞의 글 저자는 소스를 읽고 직접 실행해 본 결과, 도구 체인이 아주 길면 SDK가 정한 한도보다 Python 재귀 한도에 먼저 닿을 수 있다고 보고합니다.

티켓 한 건으로 깊이를 세어 보면 다음과 같습니다. 숫자는 예시입니다.

| 사이클 | 모델의 판단 | `stop_reason` | 스택에 쌓인 사이클 수 |
|---|---|---|---|
| 1 | 주문 상태를 확인하려고 `lookup_order` 요청 | `tool_use` | 1 |
| 2 | 배송 분실을 확인하고 `issue_refund` 요청 | `tool_use` | 2 |
| 3 | 분류 결과를 담은 최종 응답 | `end_turn` | 3 |

이 티켓은 깊이 3에서 끝나므로 문제가 없습니다. 그런데 모델이 주문 수십 건을 하나씩 조회하는 식으로 도구를 연달아 부르면 그만큼 프레임이 쌓이고, 각 사이클의 대화 기록도 계속 커집니다. 턴 상한을 거는 일은 비용 관리이면서 스택 깊이를 묶어 두는 장치이기도 합니다. 요청마다 상한을 거는 방법은 라이프사이클 제어 부분에서 다룹니다.

티켓 분류 에이전트가 어떤 도구를 왜 불렀는지는 이 결과에 담긴 메시지와 트레이스로 거슬러 올라가 확인할 수 있습니다.

## 이벤트 지점마다 꽂는 강타입 훅 콜백

루프의 특정 지점에 사용자 로직을 끼워 넣는 기본 수단은 훅입니다. 훅 이벤트는 에이전트 라이프사이클의 한 지점이고, 훅 콜백은 그 이벤트가 발생할 때 실행되도록 등록한 함수입니다. 모든 이벤트 타입에 구독자를 여럿 붙일 수 있습니다. 실행과 도구 사용 모니터링, 도구 실행 변경, 검증과 오류 처리, 멀티 에이전트 흐름 추적, 로깅과 메트릭 수집이 모두 이 장치로 구현됩니다. 콜백은 값을 반환하지 않고 이벤트 객체의 속성을 직접 바꿉니다. 도구를 교체하고, 재실행을 요청하고, 호출을 멈추는 일이 모두 이 대입으로 이루어집니다.

### 함수 하나 또는 HookProvider 묶음으로 하는 등록

가장 간단한 방법은 `agent.add_hook()`에 콜백을 넘기는 것입니다. 콜백에 타입 힌트가 있으면 이벤트 타입을 따로 적지 않아도 SDK가 추론합니다. TypeScript에서는 `agent.addHook()`을 쓰며, 이벤트 타입을 첫 인자로 넘깁니다. 아래 블록은 티켓 분류 에이전트에 붙일 로깅 콜백 두 개를 등록합니다. 두 번째 콜백에는 `order`를 지정했습니다.

```python
from strands import Agent
from strands.hooks import BeforeInvocationEvent, BeforeToolCallEvent, HookOrder

agent = Agent()

def log_invocation(event: BeforeInvocationEvent) -> None:
    print("Ticket triage started")

def log_tool(event: BeforeToolCallEvent) -> None:
    print(f"Tool called: {event.tool_use['name']}")

agent.add_hook(log_invocation, BeforeInvocationEvent)
agent.add_hook(log_tool, order=HookOrder.SDK_FIRST)  # 이벤트 타입은 타입 힌트로 추론
```

관련 콜백이 여러 개라면 `HookProvider`를 상속한 클래스의 `register_hooks()`에서 `registry.add_callback()`으로 한꺼번에 등록하고, 그 객체를 `Agent(hooks=[...])`에 넘깁니다. 뒤에 나오는 도구 교체와 재실행 예제가 이 방식을 씁니다.

### SDK 훅을 기준점으로 삼는 실행 순서

같은 이벤트에 구독자가 여럿이면 순서가 중요해집니다. 기본 규칙에서 Before 이벤트는 등록한 순서대로 실행되고, After 이벤트는 정리 작업의 대칭을 맞추려고 등록의 역순으로 실행됩니다. `order` 옵션을 주면 이 규칙 대신 명시적 우선순위가 적용되며, 값이 낮은 콜백이 먼저 실행됩니다. 프리셋은 SDK 자체 훅이 실행되는 위치를 나타냅니다.

| 프리셋 | 값 | 의미 |
|---|---|---|
| `HookOrder.SDK_FIRST` | -100 | SDK의 가장 이른 훅이 실행되는 위치 |
| `HookOrder.DEFAULT` | 0 | `order`를 지정하지 않았을 때의 암묵적 값 |
| `HookOrder.SDK_LAST` | 100 | SDK의 가장 늦은 훅이 실행되는 위치 |

프리셋은 강제되는 경계가 아니어서 어떤 숫자든 쓸 수 있습니다. SDK 훅보다 확실히 먼저 실행해야 하는 가드레일이라면 `HookOrder.SDK_FIRST - 1`을 주고, 절대적인 처음과 끝이 필요하면 `float('-inf')`와 `float('inf')`를 씁니다. 순서 값이 같은 그룹 안에서는 앞의 기본 규칙이 다시 적용됩니다.

### 이벤트 속성 대입으로 바꾸는 도구 선택

`BeforeToolCallEvent`에서 `selected_tool`에 다른 도구를 대입하면, 모델이 요청한 도구 대신 그 도구가 실행됩니다. `tool_use`의 이름도 함께 바꿔야 모델이 교체 사실을 알 수 있습니다. 이벤트 속성을 수정할 때는 디버깅과 감사를 위해 무엇을 바꿨는지 로그로 남기는 편이 좋습니다. 티켓 분류 에이전트에서는 스테이징 환경에서 `issue_refund` 호출을 초안만 만드는 도구로 바꿔치기하는 데 쓸 수 있습니다. agent.py에서 정의한 `lookup_order`, `issue_refund`를 그대로 씁니다. 아래 블록은 agent.py에 이어 붙이며, 새로 정의하는 것은 환불 초안만 기록하는 `draft_refund`와 `ToolInterceptor`뿐입니다.

```python
# excerpt: agent.py에 이어 붙이는 도구 교체 훅 (lookup_order, issue_refund는 agent.py의 정의)
import logging

from strands import Agent, tool
from strands.hooks import BeforeToolCallEvent, HookProvider, HookRegistry

logger = logging.getLogger(__name__)

@tool
def draft_refund(order_id: str, amount: float) -> str:
    """Record a refund draft without moving money."""
    return f"Drafted refund of {amount} for {order_id}"

class ToolInterceptor(HookProvider):
    def __init__(self, safe_tool):
        self.safe_tool = safe_tool

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeToolCallEvent, self.intercept_tool)

    def intercept_tool(self, event: BeforeToolCallEvent) -> None:
        if event.tool_use["name"] == "issue_refund":
            logger.info("Replacing issue_refund with draft_refund")
            event.selected_tool = self.safe_tool
            event.tool_use["name"] = "draft_refund"
```

### 같은 tool_use_id로 다시 도는 도구 재실행

도구 쪽 재시도 신호는 `AfterToolCallEvent.retry`입니다. 일시적 오류, 타임아웃, 불안정한 외부 서비스에 대응하려고 훅이 도구 재실행을 요청할 때 씁니다. 이 값을 `True`로 설정하면 도구 실행기는 현재 결과를 버리고 같은 `tool_use_id`로 도구를 다시 호출합니다. 스트리밍을 소비한다면 주의할 점이 있습니다. 도구가 스트리밍하는 동안 내보내는 중간 이벤트인 `ToolStreamEvent`는 버려진 시도의 것도 이미 호출자에게 전달된 상태이고, 도구의 최종 결과를 담는 이벤트인 `ToolResultEvent`는 마지막 시도의 것 하나만 발행되어 대화 기록에 들어갑니다. 아래 클래스는 앞의 블록과 같은 파일에 이어 붙입니다. 오류 결과일 때 `tool_use_id`별로 시도 횟수를 세어 한도 안에서만 재실행합니다.

```python
from strands.hooks import AfterToolCallEvent

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
    tools=[lookup_order, issue_refund],
    hooks=[ToolInterceptor(draft_refund), RetryOnToolError(max_retries=1)],
)
```

모델 호출을 같은 방식으로 다시 시도하는 `AfterModelCallEvent` 훅은 재시도 전략 부분에서 다룹니다.

### 인터럽트로 멈추고 resume으로 다시 여는 호출

훅은 루프를 멈출 수도 있습니다. `BeforeToolCallEvent`에서 `event.interrupt()`를 호출하면 도구 실행 전에 승인을 요구하며, 이렇게 끝난 호출의 결과는 `stop_reason`이 `"interrupt"`입니다. 반대 방향을 맡는 장치는 `AfterInvocationEvent.resume`입니다. 현재 호출이 끝난 뒤 후속 호출을 일으키므로 자율 반복 패턴을 만들 수 있습니다. 둘을 함께 쓰면 훅이 인터럽트를 직접 처리하고, 호출자는 최종 결과를 바로 받습니다. 인터럽트 상태에서 재개하려면 `interruptResponse` 객체 리스트를 넘겨야 하며, 일반 문자열을 넘기면 `TypeError`가 발생합니다. 환불 도구에 적용한 아래 블록도 앞의 두 블록과 같은 파일에 이어 붙이며, `lookup_order`, `issue_refund`는 agent.py의 정의를, `Agent`, `BeforeToolCallEvent`는 도구 교체 블록의 import를 그대로 씁니다. 자동 승인은 흐름을 보여 주려고 넣은 것입니다.

```python
from strands.hooks import AfterInvocationEvent

def require_refund_approval(event: BeforeToolCallEvent):
    """Interrupt before issuing refunds to require approval."""
    if event.tool_use["name"] == "issue_refund":
        event.interrupt("refund_approval", reason="Approve this refund?")

async def auto_approve(event: AfterInvocationEvent):
    """Automatically approve all interrupted tool calls."""
    if event.result and event.result.stop_reason == "interrupt":
        responses = [
            {"interruptResponse": {"interruptId": intr.id, "response": "approved"}}
            for intr in event.result.interrupts
        ]
        event.resume = responses

approval_agent = Agent(tools=[lookup_order, issue_refund])
approval_agent.add_hook(require_refund_approval)
approval_agent.add_hook(auto_approve)

result = approval_agent("Order A-1001 never arrived. Refund 30 dollars.")
```

`resume`을 매번 설정하면 호출이 끝없이 이어질 수 있습니다. 재호출을 연쇄할 때는 반복 횟수 같은 종료 조건을 반드시 넣어야 합니다.

## 훅 위에 쌓은 타입 결정 체계로서의 인터벤션과 사람 승인

여러 정책이 같은 도구 호출을 판정해야 할 때, 인터벤션은 각 정책이 이벤트를 직접 고치게 두지 않고 판정 결과만 반환하게 해서 충돌을 없앱니다. 인터벤션 핸들러도 훅 위에 구축되며, 라이프사이클 메서드마다 훅 콜백이 하나씩 등록됩니다. 차이는 프레임워크와 소통하는 방식에 있습니다. 앞서 본 훅은 `selected_tool`이나 `retry` 같은 속성에 값을 대입하므로 프레임워크는 그 취소가 권한 거부인지, 다르게 해 보라는 부드러운 유도인지 알 수 없습니다. 플러그인 여럿이 같은 이벤트를 고치면 마지막에 쓴 값이 이기고, 이 충돌은 드러나지 않습니다. 인터벤션은 타입이 정해진 액션을 반환하고 그 해석은 프레임워크가 맡습니다. 이 구조에서 단락, 피드백 누적, 우선순위가 정해진 순서 평가, 핸들러별 오류 정책이 나옵니다.

### 이벤트 대입을 대신하는 다섯 가지 반환 액션

라이프사이클 메서드는 아래 다섯 액션 가운데 하나를 반환합니다. 표의 클래스 이름은 Python 기준이고, TypeScript에서는 `InterventionActions.deny(reason)`처럼 `InterventionActions` 팩토리 함수로 같은 액션을 만듭니다.

| 액션 | Python 클래스 | 의미 |
|---|---|---|
| Proceed | `Proceed()` | 작업을 바꾸지 않고 계속 진행 |
| Deny | `Deny(reason="...")` | 작업을 막고 남은 핸들러를 단락 |
| Guide | `Guide(feedback="...")` | 작업을 취소하고 모델이 다시 시도할 피드백 제공 |
| Confirm | `Confirm(prompt="...")` | 사람 승인을 기다리며 일시 정지 |
| Transform | `Transform(apply=fn)` | 실행이 이어지기 전에 이벤트 내용을 제자리에서 수정 |

같은 액션이라도 효과는 시점마다 다릅니다. Before 이벤트에서 Deny는 남은 핸들러를 단락시키지만, After 이벤트에서는 효과가 없고 런타임 경고만 냅니다. Before 이벤트에서 Deny가 설정하는 값은 `event.cancel`입니다. Guide는 `before_tool_call`과 `before_invocation`에서는 누적된 피드백과 함께 작업을 취소하고, `before_model_call`에서는 피드백을 사용자 메시지로 주입하며, After 이벤트에서는 피드백을 주입한 뒤 재시도합니다. Transform은 어느 시점에서든 `action.apply(event)`를 호출하며, Before 이벤트에서는 뒤의 핸들러가 바뀐 내용을 봅니다. 메서드별로 허용되는 액션도 정해져 있습니다. `after_tool_call`은 Proceed와 Transform만, `after_model_call`은 Proceed·Guide·Transform만 받고, Confirm은 `before_tool_call`에서만 쓸 수 있습니다.

### 등록 순서와 우선순위로 정해지는 판정

핸들러는 언제나 등록한 순서대로 평가되며, 결과가 엇갈리면 deny > confirm > guide > transform > proceed 순으로 우선합니다. 어느 핸들러든 Deny를 반환하면 남은 핸들러는 건너뛰고 작업은 즉시 막힙니다. 권한 검사처럼 빠른 검사를 앞에 두면 LLM 기반 조정처럼 비싼 평가를 실행하지 않고 끝낼 수 있습니다. Guide는 단락하지 않습니다. 모든 핸들러가 계속 실행되고, Guide를 반환한 핸들러들의 피드백은 하나로 합쳐져 모델에 전달됩니다.

티켓 분류 에이전트에 세 정책을 붙여 보겠습니다. 아래 블록은 앞서 정의한 `lookup_order`·`issue_refund` 도구 블록과 같은 파일에 이어 붙입니다. 금액 상한과 주문 번호 형식은 예제를 위해 정한 규칙입니다.

```python
from strands import Agent
from strands.hooks import BeforeToolCallEvent
from strands.interventions import Confirm, Deny, Guide, InterventionHandler, OnError, Proceed

class RefundCap(InterventionHandler):
    name = "refund-cap"

    @property
    def on_error(self) -> OnError:
        return "deny"

    def before_tool_call(self, event: BeforeToolCallEvent):
        if event.tool_use["name"] == "issue_refund":
            tool_input = event.tool_use.get("input", {})
            if tool_input.get("amount", 0) > 500:
                return Deny(reason="Refunds over 500 are not allowed")
        return Proceed()

class OrderIdFormat(InterventionHandler):
    name = "order-id-format"
    def before_tool_call(self, event: BeforeToolCallEvent):
        if event.tool_use["name"] in ("lookup_order", "issue_refund"):
            tool_input = event.tool_use.get("input", {})
            if not str(tool_input.get("order_id", "")).startswith("A-"):
                return Guide(feedback="Order IDs must start with 'A-'.")
        return Proceed()

class RefundApproval(InterventionHandler):
    name = "refund-approval"
    def before_tool_call(self, event: BeforeToolCallEvent):
        if event.tool_use["name"] == "issue_refund":
            tool_input = event.tool_use.get("input", {})
            return Confirm(prompt=f"Approve refund of {tool_input.get('amount')} for {tool_input.get('order_id')}?")
        return Proceed()

agent = Agent(
    tools=[lookup_order, issue_refund],
    interventions=[RefundCap(), OrderIdFormat(), RefundApproval()],
)
```

호출 몇 개가 이 파이프라인을 어떻게 지나는지 손으로 따라가 보겠습니다.

| 모델이 요청한 호출 | `RefundCap` | `OrderIdFormat` | `RefundApproval` | 최종 판정 |
|---|---|---|---|---|
| `issue_refund(order_id="A-1001", amount=800)` | Deny | 실행 안 됨 | 실행 안 됨 | Deny, 단락 |
| `issue_refund(order_id="1001", amount=30)` | Proceed | Guide | Confirm | Confirm |
| `issue_refund(order_id="A-1001", amount=30)` | Proceed | Proceed | Confirm | Confirm |
| `lookup_order(order_id="1001")` | Proceed | Guide | Proceed | Guide |

첫 행에서는 비싼 승인 단계까지 가지 않고 끝납니다. 둘째 행은 우선순위 규칙에 따라 Guide보다 Confirm이 앞서므로 사람 승인 대기로 판정됩니다. 형식 오류를 먼저 모델에 돌려보내려면 `RefundApproval`이 주문 번호가 `A-`로 시작할 때만 Confirm을 반환하게 하십시오.

### 핸들러가 예외를 던질 때의 on_error 정책

앞 블록의 `RefundCap`이 `on_error`를 재정의한 데는 이유가 있습니다. 이 속성은 핸들러가 예외를 던졌을 때의 처리를 정합니다. 기본값 `'throw'`는 오류를 다시 던져 호출 전체를 실패시키고, `'proceed'`는 그 핸들러를 건너뛰어 `Proceed()`를 반환한 것처럼 계속하며(fail-open), `'deny'`는 오류를 로그로 남기고 Deny로 취급합니다(fail-closed). 보안상 중요한 핸들러에는 `'deny'`를 쓰고, `'proceed'`는 가용성이 강제보다 중요한 로깅 같은 핸들러에만 씁니다. 권한 핸들러에 `'proceed'`를 붙이면 그 핸들러가 던지는 순간 지키던 작업이 검사 없이 실행됩니다. 훅에는 이에 해당하는 장치가 없어 던진 오류가 항상 전파됩니다. TypeScript에서는 같은 속성의 이름이 `onError`입니다.

### 상한이 없는 Guide 재시도와 수렴 책임

`after_model_call`에서 Guide를 반환하면 모델 재시도가 일어나는데, 프레임워크는 Guide로 생긴 재시도에 횟수 상한을 두지 않습니다. 수렴은 핸들러가 책임져야 하며, 핸들러가 재시도 횟수를 세어 반복해서 실패하면 Deny로 격상할 수 있습니다. 다만 앞서 본 대로 `after_model_call`에서는 Deny가 유효하지 않으므로, 이 시점에서는 핸들러가 직접 센 횟수를 넘기면 Guide를 멈추고 Proceed를 반환하는 식으로 끊는 편이 확실합니다. 응답 어조를 다듬는 Guide 핸들러를 티켓 분류 에이전트에 붙인다면, 카운터 없이 붙이면 안 됩니다.

### 미리 응답을 주거나 루프를 빠져나오는 Confirm

Confirm은 SDK의 인터럽트/resume 체계와 통합되어 있어, 핸들러가 인터럽트 수명 주기를 직접 관리하지 않아도 도구 실행 전에 사람 승인을 기다립니다. 동작은 `response` 인자 여부로 갈립니다. `response`를 주면 그 값이 곧바로 `evaluate` 함수에 전달되어 에이전트가 멈추지 않고, 주지 않으면 에이전트 루프를 빠져나와 인터럽트 체계를 통한 외부 재개를 기다립니다. `evaluate`는 응답이 승인인지 판정하며, 기본 구현은 `True`, `"y"`, `"yes"`를 대소문자 구분 없이 승인으로 받습니다. 거부 응답이 오면 `event.cancel`이 설정됩니다. 아래 핸들러는 주문 조회 도구를 선제 승인해서 루프를 멈추지 않게 합니다.

```python
from strands.hooks import BeforeToolCallEvent
from strands.interventions import Confirm, InterventionHandler, Proceed

class AutoApprove(InterventionHandler):
    name = "auto-approve"

    def before_tool_call(self, event: BeforeToolCallEvent):
        if event.tool_use["name"] == "lookup_order":
            return Confirm(
                prompt="Allow order lookup?",
                response="yes",
            )
        return Proceed()
```

`response` 없이 멈춘 호출은 앞서 `require_refund_approval`로 멈춘 `approval_agent`를 다시 이어 갈 때와 같은 방식으로 재개합니다.

### 위험한 호출만 사람 앞에 세우는 HumanInTheLoop

모든 환불마다 승인을 묻는 `RefundApproval`은 확실하지만 사람을 지치게 합니다. 기본 제공 핸들러 `HumanInTheLoop`은 각 도구 호출 전에 위험 분류기가 도구 이름과 입력을 모델에 보내 승인이 필요한지 묻습니다. 안전하다고 판단된 호출은 프롬프트 없이 실행되고, 위험한 호출이면 분류기의 사유와 함께 터미널에 승인 프롬프트를 띄운 뒤 에이전트가 답을 기다립니다. 티켓 분류 에이전트에 붙이면 다음과 같은 모양입니다. 도구 블록과 같은 파일에 이어 붙이며, `allowed_tools`의 부정 표기는 뒤에서 설명합니다.

```python
from strands import Agent
from strands.vended_interventions.hitl import HumanInTheLoop

agent = Agent(
    tools=[lookup_order, issue_refund],
    interventions=[
        HumanInTheLoop(
            classifier=True,
            ask="stdio",
            enable_trust=True,
            allowed_tools=["*", "!issue_refund"],
        )
    ],
)
```

`ask="stdio"`에서 프롬프트에 `y`나 `yes`로 답하면 이번 호출만 승인되고, `t`나 `trust`로 답하면 에이전트가 존재하는 동안 같은 도구의 이후 호출까지 승인됩니다. 그 밖의 입력은 호출을 거부하며, 에이전트는 거부 사실을 알게 되어 사용자에게 알리거나 다른 방법을 시도할 수 있습니다.

기본 분류기는 심판 역할을 하는 LLM이며, 에이전트의 모델에서 돌고 보조 모델을 설정했다면 그 모델에서 돕니다. 닫힌 질문에 빠르게 답하도록 만든 작은 모델이나 직접 만든 분류기 함수로 바꿀 수 있고, 분류기가 없으면 모든 도구 호출이 프롬프트를 띄웁니다. 분류기의 모델 호출이 실패하면 핸들러는 해당 호출의 승인을 사람에게 요청하므로, 분류기 장애가 위험한 호출을 검사 없이 통과시키지는 않습니다. 반대 방향의 오류는 남습니다. 분류기는 보장이 아니라 판단이어서 위험한 호출을 안전하다고 평가할 수 있습니다. 그래서 반드시 물어야 하는 도구는 `allowed_tools=["*", "!issue_refund"]`처럼 `"*"`와 함께 부정 표기합니다. `"*"`로 모든 도구를 승인 없이 허용하되 `!issue_refund`는 제외해 환불은 항상 승인을 받습니다. 부정된 도구는 항상 프롬프트를 띄우고 `trust`로 신뢰할 수도 없습니다. TypeScript에서는 같은 옵션의 이름이 `allowedTools`입니다.

### 승인으로도 넘을 수 없는 한도를 맡는 Cedar 선배치

환불 권한이 일부 상담원에게만 있다면, 사람이 승인한다고 넘어가서는 안 되는 한도입니다. 이런 한도는 Cedar Authorization을 `HumanInTheLoop` 앞에 두어 겁니다. Cedar는 어떤 정책도 허용하지 않는 호출을 거부하므로, 그 호출은 분류기에도 사람에게도 도달하지 않습니다. 앞서 본 Deny 단락 덕분에 Cedar를 맨 앞에 두면 거부된 호출이 뒤의 핸들러에 닿지 않습니다.

```python
from strands import Agent
from strands.vended_interventions.cedar import CedarAuthorization
from strands.vended_interventions.hitl import HumanInTheLoop

cedar = CedarAuthorization(
    policies="""
    permit(principal, action == Action::"lookup_order", resource);

    permit(
        principal == User::"alice",
        action == Action::"issue_refund",
        resource
    );
    """,
    principal_resolver=lambda state: {"type": "User", "id": state["user_id"]},
)

agent = Agent(
    tools=[lookup_order, issue_refund],
    interventions=[
        cedar,
        HumanInTheLoop(classifier=True, ask="stdio", enable_trust=True),
    ],
)

# alice의 환불은 분류기 판정에 따라 승인 프롬프트로 이어질 수 있음
agent("Order A-1001 never arrived. Refund 30 dollars.", invocation_state={"user_id": "alice"})

# bob의 환불은 프롬프트 없이 거부됨
agent("Order A-1001 never arrived. Refund 30 dollars.", invocation_state={"user_id": "bob"})
```

`principal_resolver`는 호출마다 넘기는 `invocation_state`에서 주체를 꺼냅니다. 이 호출 상태의 수명과 용도는 상태 영속화 부분에서 다룹니다.

## pip 한 줄에서 도구를 쓰는 첫 티켓 분류 에이전트까지

Python 가상 환경에 `strands-agents`를 설치하고 Bedrock 자격 증명을 갖추면, 함수 두 개를 도구로 넘긴 `Agent` 하나로 티켓 분류 에이전트가 첫 응답을 냅니다.

### 가상 환경 하나와 패키지 하나로 끝나는 설치

Python SDK를 설치하려면 Python 3.10 이상과 활성화된 가상 환경이 있어야 합니다. 설치 명령은 한 줄입니다.

```bash
pip install strands-agents
```

Bedrock 외의 프로바이더는 대부분 선택적 의존성입니다. 쓰려는 프로바이더를 extra로 지정해 설치하고, 전부 필요하면 `all`을 지정합니다.

```bash
pip install 'strands-agents[anthropic]'
pip install 'strands-agents[all]'
```

### 세 경로 중 하나로 맞추는 Bedrock 인증

Python SDK의 기본 프로바이더는 Amazon Bedrock이므로 추가 설치도, 모델 객체도 필요 없습니다. 프로바이더 설치 예시에는 `strands-agents[bedrock]` extra도 나옵니다. 기본 패키지만으로 Bedrock 에이전트를 만들 수 있고, 이 extra는 다른 프로바이더와 같은 형식으로 Bedrock을 명시해 설치하는 방법입니다. PyPI 페이지는 기본 Bedrock 프로바이더를 쓰려면 AWS 자격 증명과 us-west-2 리전의 Claude 4 Sonnet 모델 접근이 필요하다고 안내합니다.

SDK에는 모델 호출 권한이 있는 AWS 자격 증명이 필요하며, 아래 셋 중 하나로 넘깁니다.

- Bedrock API 키: `AWS_BEARER_TOKEN_BEDROCK` 환경 변수에 키를 넣습니다. 로컬 개발에서 가장 빠른 방법입니다.
- AWS 자격 증명: `aws configure`를 쓰거나 `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` 환경 변수를 설정하고, 필요하면 `AWS_SESSION_TOKEN`도 설정합니다.
- IAM 역할: EC2, ECS, Lambda 같은 AWS 서비스 위에서 실행할 때 씁니다.

어느 경로를 고르든 쓰려는 모델에 대한 접근을 Bedrock 콘솔에서 따로 활성화해야 합니다.

### docstring과 타입 힌트로 모델에 설명하는 `@tool` 도구

`@tool` 데코레이터를 붙이면 직접 작성한 Python 함수가 도구가 됩니다. 모델은 함수의 docstring과 타입 힌트를 읽고 언제 이 도구를 호출할지, 어떤 인자를 넘길지 판단합니다. 도구 목록은 `tools` 리스트로 에이전트에 전달합니다. 아래 `agent.py`에서 주문 데이터는 메모리 딕셔너리로 대신합니다.

```python
from strands import Agent, tool

ORDERS = {"A-1001": {"status": "lost in transit", "total": 30.0}}

@tool
def lookup_order(order_id: str) -> str:
    """
    Look up a customer order by its ID.

    Args:
        order_id (str): The order ID, for example "A-1001"

    Returns:
        str: The order status and total amount
    """
    if order_id not in ORDERS:
        raise ValueError(f"Unknown order: {order_id}")
    order = ORDERS[order_id]
    return f"{order_id}: {order['status']}, total {order['total']}"

@tool
def issue_refund(order_id: str, amount: float) -> str:
    """
    Issue a refund for an order.

    Args:
        order_id (str): The order ID to refund
        amount (float): The refund amount in dollars

    Returns:
        str: A confirmation message
    """
    if order_id not in ORDERS:
        raise ValueError(f"Unknown order: {order_id}")
    return f"Refunded {amount} for {order_id}"

agent = Agent(tools=[lookup_order, issue_refund])
result = agent("Order A-1001 never arrived. What happened, and can I get a refund?")
print(result.stop_reason)
print(result.message)
```

`python -u agent.py`로 실행합니다. 호출 하나가 앞서 본 추론-행동-관찰 사이클을 돌고 실행의 메시지, 메트릭, 트레이스를 담은 `AgentResult`를 돌려줍니다. 위 코드는 그 결과에서 멈춘 이유(`stop_reason`)와 에이전트의 마지막 메시지(`message`)를 출력합니다. 파일 편집처럼 흔한 작업에는 SDK가 기본 제공하는 도구도 있으며, 직접 만든 도구와 같은 리스트에 넣으면 됩니다.

### 객체 하나만 바꾸는 모델 프로바이더 교체

모델은 에이전트에 넘기는 객체 하나이므로, 프로바이더를 바꿔도 나머지 코드는 그대로입니다. Anthropic으로 바꾸려면 `strands-agents[anthropic]`를 설치하고 `ANTHROPIC_API_KEY`를 설정한 다음 모델 객체를 만들어 넘깁니다.

```python
from strands import Agent
from strands.models.anthropic import AnthropicModel

# Reads ANTHROPIC_API_KEY from the environment.
model = AnthropicModel(model_id="claude-sonnet-5", max_tokens=4096)
agent = Agent(model=model, tools=[lookup_order, issue_refund])
```

지원 범위는 언어마다 다릅니다. Ollama, LiteLLM, SageMaker 등은 Python에서만, Vercel은 TypeScript에서만 쓸 수 있습니다. 다음에 다룰 구조화된 출력은 Python SDK에서 프로바이더마다 따로 구현되어 있고, 커스텀 프로바이더는 스트리밍 메서드만 필수이고 나머지 기능은 구현에 달려 있습니다. 커스텀 프로바이더를 직접 만든다면 구조화된 출력 지원 여부부터 확인하십시오.

### 부가 호출을 작은 모델로 보내는 `aux_model`

SDK는 메인 루프 밖에서도 컨텍스트 요약, 앞서 본 human-in-the-loop 위험 분류기, LLM으로 에이전트의 행동을 평가해 방향을 잡아 주는 LLM 스티어링 같은 부가 호출을 합니다. 기본으로는 이 호출이 모두 메인 모델에서 돌고, `aux_model`을 설정하면 한꺼번에 더 작고 저렴한 모델로 보냅니다.

```python
from strands import Agent
from strands.models.bedrock import BedrockModel

agent = Agent(
    model=BedrockModel(model_id="global.anthropic.claude-opus-5-20260301-v1:0"),
    aux_model=BedrockModel(model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0"),
    tools=[lookup_order, issue_refund],
)
```

각 부가 호출이 쓸 모델은 컴포넌트 자체의 `model` 옵션, `agent.aux_model`, `agent.model` 순으로 정해집니다. `aux_model`을 설정하지 않으면 동작은 그대로입니다. 요청마다 모델을 고르는 라우터 객체인 `ModelRouter`는 이 자리에 넣을 수 없는데, 부가 호출이 라우터가 붙는 에이전트 루프 밖에서 돌기 때문입니다. 환불 호출마다 분류기를 돌리는 티켓 에이전트라면, 분류기에 따로 `model`을 주지 않는 한 분류기 호출이 Haiku 모델로 옮겨 갑니다.

### 에이전트 실행과는 무관한 문서 MCP 서버

Strands는 IDE의 AI 코딩 어시스턴트가 문서를 검색하고 가져오게 해 주는 MCP 서버를 제공합니다. 개발을 돕는 도구일 뿐 에이전트 실행에는 필요 없습니다. 이 서버는 uv가 있어야 돌아갑니다. Claude Code에는 아래 명령으로 추가하고, 연결은 MCP Inspector로 확인합니다.

```bash
claude mcp add strands uvx strands-agents-mcp-server
npx @modelcontextprotocol/inspector uvx strands-agents-mcp-server
```

## 스키마를 도구로 바꾸는 구조화된 출력과 강제 재시도

티켓 분류 결과를 코드에서 바로 쓰려면 스키마를 호출에 넘기면 됩니다. 그러면 Strands가 그 스키마를 이번 호출에만 쓰는 도구로 붙이고, 모델이 그 도구를 호출하도록 유도합니다. 검증에 실패하면 오류를 모델에 돌려주고, 모델이 텍스트로 끝내면 도구 호출을 강제해 한 번 더 시도합니다.

### 호출마다 붙었다 사라지는 스키마 도구

구조화된 출력(structured output)은 필드 이름, 타입, 설명을 담은 스키마를 받아서 텍스트 대신 그 스키마로 검증한 객체를 돌려줍니다. 스키마는 Python에서 Pydantic 모델로, TypeScript에서 Zod 스키마로 씁니다. 이때 Strands는 모델에 JSON을 쓰라고 지시하지 않습니다. 스키마를 도구로 바꾸고, 모델이 필드 값을 입력으로 넣어 그 도구를 호출하게 합니다.

스키마가 붙은 호출은 아래 순서로 진행됩니다.

1. 스키마로 만든 구조화 출력 도구를 에이전트의 도구 목록에 추가합니다.
2. 모델은 평소처럼 작업하며, 그 과정에서 다른 도구도 호출할 수 있습니다.
3. 모델이 필드 값을 넣어 구조화 출력 도구를 호출합니다.
4. Strands가 그 입력을 검증합니다. 통과하면 호출이 끝나고 객체가 `result.structured_output`에 담깁니다.
5. 호출이 끝나면 구조화 출력 도구를 목록에서 제거합니다.

Python에서는 도구 이름이 클래스 이름을 따릅니다. `TicketTriage` 스키마는 `TicketTriage` 도구가 되고, 클래스 docstring이 도구 설명이 됩니다. Strands는 도구 설명에 "필요한 다른 도구를 모두 쓴 뒤 이 도구를 마지막에 호출하라"는 지시를 덧붙입니다. 그래서 에이전트는 주문 조회 같은 도구를 먼저 쓰고 나서 객체를 반환할 수 있습니다.

### 필드 설명과 고정값으로 좁힌 `TicketTriage` 스키마

모델은 각 필드의 설명을 읽습니다. 그러니 설명에는 타입만 적지 말고 그 필드가 무슨 뜻인지 적어야 합니다. `Literal` 같은 고정값은 모델이 고를 수 있는 선택지를 정해 둔 값으로 제한합니다. 스키마는 `structured_output_model`로 넘기고 결과는 `result.structured_output`으로 읽습니다. `invoke_async`도 같은 인자를 받습니다. 아래 코드는 앞서 만든 `agent.py`에 이어 붙이는 분류 함수입니다. 메시지마다 새 에이전트를 만들어 티켓끼리 대화 기록이 섞이지 않게 했습니다.

```python
# excerpt: agent.py 뒤에 이어 붙이는 분류 함수 (lookup_order, issue_refund, HumanInTheLoop은 앞에서 정의·import)
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
    agent = Agent(
        tools=[lookup_order, issue_refund],
        callback_handler=None,
        interventions=[
            HumanInTheLoop(classifier=True, ask="stdio", allowed_tools=["*", "!issue_refund"])
        ],
    )
    try:
        result = agent(message, structured_output_model=TicketTriage)
    except StructuredOutputException:
        # Send anything the agent can't classify to a person
        return TicketTriage(category="other", priority="medium", summary=message)
    return result.structured_output


ticket = triage("Order A-1001 never arrived. What happened, and can I get a refund?")
print(f"{ticket.priority} | {ticket.category} | {ticket.summary}")
```

Python에서 반환되는 객체는 `TicketTriage` 인스턴스 그 자체입니다. 에이전트가 객체를 끝내 내지 못하면 호출이 `StructuredOutputException`을 던지고, `triage`는 이 예외를 잡아 사람이 검토할 `other` 티켓으로 돌립니다. 이 에이전트는 `issue_refund`도 쓸 수 있으므로 분류 도중 환불이 실행될 수 있습니다. 그래서 앞서 본 `HumanInTheLoop`을 붙여 `issue_refund`만 승인을 거치게 했습니다. 분류 에이전트에 환불 도구를 줄 때는 이 승인 장치를 함께 붙이십시오.

스키마는 두 곳에 지정할 수 있습니다. 에이전트를 만들 때 `Agent(structured_output_model=...)`로 기본 스키마를 정하면 모든 호출에 적용되고, 호출에 넘긴 스키마는 그 호출에만 적용되며 에이전트의 스키마보다 우선합니다. 에이전트 스키마는 맥락만 쌓는 턴에도 적용됩니다. 일부 호출이 일반 텍스트로 답해야 한다면 스키마를 호출마다 넘기십시오. 이렇게 하면 이전 턴은 텍스트를 반환하고, 필드는 대화 전체를 근거로 채워집니다. 고객과 몇 턴 대화한 뒤 마지막 호출에서만 `TicketTriage`를 요청하는 흐름도 이렇게 만들 수 있습니다.

### 예외 대신 도구 결과로 돌아오는 검증 오류

스키마에는 모델이 첫 시도에 맞추지 못할 수도 있는 규칙을 넣을 수 있습니다. 입력이 검증에 실패해도 Strands는 예외를 던지지 않습니다. 대신 검증 오류를 도구 결과로 모델에 돌려주고, 모델은 값을 고쳐 도구를 다시 호출할 수 있습니다. 티켓을 처리한 뒤 고객에게 다시 연락할 후속 작업 스키마에 날짜가 미래인지 확인하는 `field_validator`를 두면 흐름이 잘 드러납니다.

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

모델은 오늘 날짜를 모르기 때문에 첫 추측이 틀릴 수 있습니다. 아래는 그 과정을 순서대로 적은 기록입니다.

```text
1. 모델이 due_date=2025-07-22로 구조화 출력 도구를 호출
2. 검증 실패 → 모델이 "due_date must be after 2026-10-12"를 도구 결과로 받음
3. 모델이 2026-10-13으로 다시 호출
4. 검증 통과 → 호출이 객체를 반환
```

이런 규칙은 프롬프트가 아니라 스키마에 넣어야 합니다. 모델은 오류 메시지를 그대로 받으므로, 메시지를 모델이 읽고 바로 고칠 수 있는 지시문으로 쓰십시오. 한도를 두지 않으면 모델은 필요한 만큼 재시도할 수 있습니다.

### 텍스트로 끝난 턴을 되돌리는 강제 도구 호출

모델이 구조화 출력 도구를 부르지 않고 텍스트 답변으로 턴을 끝내는 경우도 있습니다. 그러면 Strands는 구조화 출력 도구를 필수로 지정하고 한 번 더 시도합니다. 이 재시도에서 모델이 할 수 있는 일은 그 도구를 호출하는 것뿐입니다. 이것이 강제 도구 호출(forced tool call)입니다. Python은 재시도 전에 대화에 `You must format the previous response as structured output.`라는 사용자 메시지를 추가합니다. 이 문구는 에이전트나 개별 호출에 `structured_output_prompt`를 넘겨 바꿀 수 있고, 둘 다 있으면 호출에 넘긴 값이 우선합니다.

강제 재시도에서도 유효한 출력이 나오지 않으면 Python은 `StructuredOutputException`을 던집니다. 앞의 `triage`가 잡는 예외가 바로 이것입니다. 개발 버전(main) 소스에는 처리가 하나 더 있습니다. 강제 재시도에서 모델이 도구 호출을 하지 못하고 JSON 텍스트로 끝나면, 그 텍스트를 구조화 출력으로 검증합니다. 실패하면 시도 한도가 있을 때는 실패 횟수에 넣고, 한도가 없을 때는 예외를 냅니다. 콘텐츠 필터나 가드레일이 강제 재시도를 막으면 Python은 `content_filtered` 같은 별도의 정지 사유로 호출을 끝내고, TypeScript는 이를 실패한 시도로 세어 한도 안에서 다시 시도합니다.

### 실패를 세는 `structured_output_attempts` 한도

예외 대신 다시 시도하게 하려면 호출에 `limits['structured_output_attempts']`를 지정합니다. 모델 응답 하나가 실패할 때마다 시도 하나로 셉니다. 무엇을 세는지는 아래 표와 같습니다.

| 상황 | 시도로 세는지 |
|---|---|
| 검증에 실패한 구조화 출력 도구 호출 | 셉니다 |
| 유효한 출력을 내지 못한 강제 재시도 | 셉니다 |
| 강제 재시도 전에 다른 도구를 호출한 경우 | 세지 않습니다 |

한도를 지정하면 실패한 강제 재시도가 예외로 끝나지 않고 다시 시도됩니다. 한도에 닿으면 호출은 다음 모델 호출 전에 멈춥니다. 이때 정지 사유는 `limit_structured_output_attempts`이고 구조화된 출력은 비어 있습니다. `triage`를 한도 기반으로 바꾸면 예외를 잡는 대신 정지 사유로 분기하게 됩니다. Python에서 필터가 막은 경우에 나오는 정지 사유도 같은 분기에 넣었습니다.

```python
# excerpt: 시도 한도로 실패를 다루는 triage 변형 (TicketTriage와 도구는 위에서 정의)
def triage_limited(message: str) -> TicketTriage:
    agent = Agent(tools=[lookup_order, issue_refund], callback_handler=None)
    result = agent(
        message,
        structured_output_model=TicketTriage,
        limits={"structured_output_attempts": 3},
    )
    if result.stop_reason in ("limit_structured_output_attempts", "content_filtered"):
        return TicketTriage(category="other", priority="medium", summary=message)
    return result.structured_output
```

예를 들어 한도가 3이라고 가정해 봅시다. 모델이 `lookup_order`를 두 번 부르는 동안에는 시도 수가 늘지 않습니다. `priority`에 허용되지 않은 값을 넣은 호출이 두 번 실패하면 시도 수는 2가 되고, 이어서 텍스트로 끝난 턴의 강제 재시도까지 실패하면 3이 되어 다음 모델 호출 전에 멈춥니다. 이 한도는 다른 호출 한도와 마찬가지로 요청 전체에 걸립니다. 같은 요청 안에서 도구로 실행되는 에이전트의 실패도 함께 세고, `result.invocation_context`를 다음 호출에 넘기면 그 횟수가 이어집니다. 나머지 호출 한도는 라이프사이클 제어에서 다룹니다.

### 최종 결과에 한 번만 실리는 스트리밍 객체

스트리밍 호출에도 스키마를 같은 방식으로 넘깁니다. 텍스트는 모델이 쓰는 대로 청크 단위로 도착하고, 객체는 최종 결과에 한 번만 실립니다. 상담원 화면에 진행 상황을 보여 주면서 분류 결과는 마지막에 받으려면 `stream_async`를 이렇게 씁니다.

```python
# excerpt: 분류를 스트리밍으로 받는 비동기 버전 (TicketTriage와 도구는 위에서 정의)
import asyncio


async def triage_stream(message: str) -> TicketTriage:
    agent = Agent(tools=[lookup_order, issue_refund], callback_handler=None)
    async for event in agent.stream_async(message, structured_output_model=TicketTriage):
        if "data" in event:
            print(event["data"], end="", flush=True)
        elif "result" in event:
            return event["result"].structured_output


ticket = asyncio.run(triage_stream("Order A-1001 never arrived. Can I get a refund?"))
```

`data` 이벤트는 화면 출력에만 쓰고, 분기 처리에는 `result` 이벤트의 객체만 쓰십시오. 스키마를 넘기는 방식은 하나로 정리되었습니다. Python의 `Agent.structured_output()`와 `Agent.structured_output_async()`는 deprecated이며 호출하면 `DeprecationWarning`을 냅니다. 스키마는 지금까지 본 것처럼 호출 인자로 넘기면 됩니다.

## 요청마다 턴·토큰·시간 상한을 거는 라이프사이클 제어

동시 요청이 많은 서비스라면 요청마다 새 에이전트를 만들고, 그 호출에 턴·토큰 한도와 데드라인 신호를 함께 걸어야 폭주하는 호출을 안전하게 끊을 수 있습니다. 호출 한도는 반복이 시작될 때만 검사하고, 취소는 정해진 체크포인트에서 협력적으로 일어나기 때문입니다. 끝난 호출은 정지 사유를 타입이 있는 결과로 바꿔 돌려주고, 예외는 그와 다른 경로로 처리합니다.

### 반복의 시작에서만 검사하는 `limits`

한 호출이 하는 일의 양은 `limits`로 제한합니다. 턴(루프 반복) 수, 출력 토큰, 총 토큰, 실패한 구조화 출력 시도를 각각 따로 지정할 수 있고, 모든 항목은 선택입니다. 같은 인자를 `invoke_async`와 `stream_async`에도 넘길 수 있으며, 각 상한은 양의 정수여야 합니다. 한도를 넘기지 않으면 아무 제한도 걸리지 않습니다. 이 경우 모델이 도구 요청을 멈출 때까지 루프가 계속 돌고, 턴마다 그때까지의 대화 전체를 다시 보냅니다. 앞서 본 `structured_output_attempts`도 이 `limits` 묶음에 들어가는 항목 중 하나입니다.

검사 시점은 각 루프 반복이 시작될 때뿐입니다. 호출 도중에는 검사하지 않으므로 한 턴이 토큰 예산을 넘길 수 있습니다. 또 이전 턴이 요청한 도구는 한도를 검사하기 전에 항상 끝까지 실행됩니다. 그러니 한도는 "넘지 않는 선"이 아니라 "다음 턴을 시작하지 않는 선"으로 보고, 한 턴이 쓸 수 있는 최대량만큼 여유를 남겨 두십시오.

토큰 계산에는 프롬프트 캐시도 들어갑니다. Bedrock, Anthropic, OpenAI, Google Gemini에서 총 토큰 한도는 캐시 읽기와 쓰기를 모두 셉니다. 따라서 캐시가 적중한 턴도 적중하지 않은 턴만큼 한도를 소모합니다. 이 한도가 제한하는 대상은 비용이 아니라 토큰입니다. 여러 상한에 동시에 걸리면 정지 사유는 turns, total tokens, output tokens, structured output attempts 순서로 하나만 보고합니다. 카운터는 호출 단위로 초기화되므로, 재사용하는 에이전트도 호출마다 0에서 시작합니다. 한도에 걸려 멈춘 뒤에도 메시지 기록은 다시 호출할 수 있는 유효한 상태이므로, 예산을 높여 같은 에이전트로 재시도할 수 있습니다.

### 체크포인트에서 관찰하는 취소

루프 밖에서 실행을 멈출 때는 취소를 씁니다. 클라이언트 연결이 끊겼을 때, 타임아웃이 났을 때, 상담원이 "중지" 버튼을 눌렀을 때가 그런 경우입니다. Python의 `agent.cancel()`은 내부 신호를 설정합니다. 에이전트는 네 체크포인트에서 이 신호를 확인하고, 확인하면 `stop_reason="cancelled"`인 결과를 돌려줍니다. `cancel()`은 스레드 안전하고 여러 번 불러도 결과가 같습니다. 신호는 호출이 끝나면 자동으로 해제됩니다. 각 체크포인트에서 일어나는 일은 아래와 같습니다.

| 체크포인트 | 동작 | 비고 |
|---|---|---|
| 모델 응답 스트리밍 | 부분 출력을 버립니다 | 모델이 마지막 메타데이터 이벤트를 보내기 전에 스트림을 닫으므로 사용량 메트릭이 부정확할 수 있습니다 |
| 도구 실행 전 | 도구 호출을 건너뛰고 오류 결과를 추가해 대화 상태를 유효하게 유지합니다 | - |
| MCP 도구 실행 중 | 공유 MCP 세션은 닫지 않고 진행 중인 MCP 요청만 취소합니다 | 원격 취소는 최선 노력 방식이며, 서버가 응답하지 않아도 에이전트는 로컬에서 멈춥니다 |
| 도구 실행 후 | 다음 모델 호출 전에 멈춥니다 | MCP가 아닌 도구는 끝까지 실행된 뒤에 취소를 관찰합니다 |

멈춤이 한 요청에만 해당한다면 호출마다 `cancel_signal`로 `threading.Event`를 넘기는 편이 맞습니다. 에이전트는 호출자가 소유한 이 이벤트를 직접 설정하거나 해제하지 않습니다. 반면 `agent.cancel()`은 실행 중인 호출이 확실할 때만 불러야 합니다. 에이전트가 쉬고 있을 때 부르면 다음 호출이 취소되기 때문입니다. 사람 입력을 기다리며 잠시 멈췄다가 재개하는 인터럽트(interrupt)와 달리, 취소는 에이전트를 완전히 중단합니다.

### 도구가 스스로 확인하는 협력적 취소

취소 여부는 도구 실행 전과 안전한 루프 경계에서 확인합니다. 도구를 순차적으로 실행하면 아직 시작하지 않은 도구를 건너뛸 수 있습니다. 하지만 기본 동시 실행기에서는 형제 도구가 이미 실행 중일 수 있습니다. 도구가 일단 시작되면 일찍 멈출지는 도구가 결정합니다. 그래서 오래 걸리는 도구는 단계 사이에 신호를 확인해야 합니다. 티켓 하나에 주문 번호가 여러 개 딸려 오는 경우에 맞춰 일괄 조회 도구를 만들었습니다. 도구는 `context=True`로 받은 `tool_context.cancel_signal`을 주문마다 확인합니다.

```python
# excerpt: agent.py에 추가하는 일괄 조회 도구 (ORDERS는 앞에서 정의)
from strands import tool
from strands.types.tools import ToolContext


@tool(context=True)
def lookup_orders(order_ids: list[str], tool_context: ToolContext) -> str:
    """Look up several orders, stopping early if the request is cancelled."""
    lines = []
    for order_id in order_ids:
        if tool_context.cancel_signal.is_set():
            return f"cancelled after {len(lines)} orders"
        order = ORDERS.get(order_id)
        lines.append(f"{order_id}: {order['status'] if order else 'not found'}")
    return "\n".join(lines)
```

이 검사가 없으면 데드라인이 지나도 도구는 남은 주문을 끝까지 조회하고, 에이전트는 그 뒤에야 "도구 실행 후" 체크포인트에서 멈춥니다.

### 한 인스턴스에 겹치는 호출을 막는 동시성 가드

에이전트는 호출이 진행되는 동안 대화 기록을 고칩니다. 그래서 기본 설정에서는 한 번에 호출 하나만 처리하고 겹치는 호출은 거부합니다. 생성자 파라미터 `concurrent_invocation_mode`는 `ConcurrentInvocationMode`의 `THROW`(기본값)나 `UNSAFE_REENTRANT` 중 하나를 받습니다. `THROW` 모드에서 이미 실행 중인 에이전트를 다시 호출하면 `ConcurrencyException`이 발생합니다. `UNSAFE_REENTRANT`는 이 가드를 아예 없애므로, 그다음부터는 동시에 일어나는 기록 변경을 아무것도 막지 않습니다.

같은 요청을 클라이언트가 재전송한 경우는 오류로 처리하지 말고 중복을 제거하면 됩니다. 호출 메서드(`__call__`, `invoke_async`, `stream_async`)에 `idempotency_token`을 넘기면, 같은 토큰으로 진행 중인 호출이 있을 때 새 호출은 원본이 끝나기를 기다렸다가 같은 `AgentResult`를 받습니다. 토큰이 다르면 여전히 `ConcurrencyException`이 납니다. 원본이 결과 없이 끝나면(예를 들어 도중에 취소되면) 기다리던 호출에서 `IdempotencyAbortedError`가 발생하므로, 그 자리에서 재시도할지 정하면 됩니다. 토큰으로는 티켓 ID 대신, 클라이언트가 요청마다 붙이는 요청 ID(재전송할 때도 같은 값)를 쓰십시오. 티켓 ID를 쓰면 같은 티켓에 서로 다른 메시지 두 개가 겹칠 때 두 번째 메시지가 첫 메시지의 `AgentResult`를 그대로 받습니다.

```python
from strands import Agent
from strands.types.exceptions import ConcurrencyException, IdempotencyAbortedError


def answer(agent: Agent, request_id: str, message: str) -> str | None:
    try:
        return str(agent(message, idempotency_token=request_id))
    except IdempotencyAbortedError:
        # The original call for this request ended without a result
        return None
    except ConcurrencyException:
        # A different request is already running on this agent
        return None
```

TypeScript SDK에서는 실행 중인 에이전트를 호출하면 `ConcurrentInvocationError`가 납니다. 동시성 모드 설정과 토큰 기반 중복 제거는 Python SDK에만 있습니다.

### 요청별 새 에이전트와 타입이 있는 결과

수천 건의 동시 호출을 받는 서비스에서 규모가 커져도 버티는 방식은 요청마다 새 에이전트를 만드는 것입니다. 두 요청이 같은 인스턴스를 공유하지 않으니 동시성 가드에 걸릴 일이 없습니다. 각 요청에는 턴 수, 토큰 예산, 데드라인을 따로 주고, 핸들러는 정지 사유를 호출자가 분기할 수 있는 타입으로 바꿉니다. 먼저 정지 사유를 결과 타입으로 옮기는 부분입니다.

```python
# excerpt: agent.py의 결과 타입 (TicketTriage는 앞에서 정의)
import threading
from dataclasses import dataclass
from enum import Enum

from strands import Agent
from strands.models import BedrockModel

MODEL_ID = "global.anthropic.claude-sonnet-5"


class Outcome(str, Enum):
    COMPLETED = "completed"
    BUDGET_EXCEEDED = "budget_exceeded"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    STOPPED = "stopped"


@dataclass
class Reply:
    outcome: Outcome
    triage: TicketTriage | None


def classify(stop_reason: str) -> Outcome:
    if stop_reason in ("end_turn", "stop_sequence"):
        return Outcome.COMPLETED
    if stop_reason == "cancelled":
        return Outcome.CANCELLED
    if stop_reason in (
        "limit_turns",
        "limit_total_tokens",
        "limit_output_tokens",
        "limit_structured_output_attempts",
    ):
        return Outcome.BUDGET_EXCEEDED
    if stop_reason in ("content_filtered", "guardrail_intervened"):
        return Outcome.BLOCKED
    return Outcome.STOPPED
```

핸들러는 요청마다 만든 새 에이전트를 앞에서 만든 `TicketTriage`를 `structured_output_model`로 넘겨 호출하면서 `limits={"turns": 6, "total_tokens": 20_000, "structured_output_attempts": 3}`를 걸고, `deadline_seconds`의 기본값 `20.0`초가 지나면 `cancel_signal`을 설정하는 타이머를 붙입니다. `finally`에서 타이머를 해제하므로, 일찍 끝난 요청의 타이머가 남아 있다가 신호를 설정하는 일이 없습니다.

```python
# excerpt: agent.py의 요청 핸들러 (lookup_order, issue_refund, lookup_orders, TicketTriage는 위에서 정의)
def handle_ticket(message: str, deadline_seconds: float = 20.0) -> Reply:
    # A fresh agent per request keeps concurrent calls from sharing history
    agent = Agent(
        model=BedrockModel(model_id=MODEL_ID),
        tools=[lookup_order, lookup_orders, issue_refund],
    )

    cancel_signal = threading.Event()
    deadline = threading.Timer(deadline_seconds, cancel_signal.set)
    deadline.start()

    try:
        result = agent(
            message,
            structured_output_model=TicketTriage,
            limits={"turns": 6, "total_tokens": 20_000, "structured_output_attempts": 3},
            cancel_signal=cancel_signal,
        )
    finally:
        deadline.cancel()

    outcome = classify(result.stop_reason)
    triage = result.structured_output if outcome == Outcome.COMPLETED else None
    return Reply(outcome=outcome, triage=triage)
```

예산 소진이나 취소로 멈춰도 이 핸들러는 예외를 던지지 않고 타입이 있는 결과를 돌려줍니다. 결과는 정상 완료일 때만 담으며, 여기서는 검증된 `TicketTriage`가 그 자리에 들어갑니다. 다만 이 무상태 패턴으로는 인터럽트를 재개할 수 없습니다. 환불 전에 `HumanInTheLoop` 승인을 받으려면 에이전트나 세션 상태와 인터럽트 ID를 보존해야 하며, 대화별 세션 영속화에서 그 방법을 다룹니다.

### 정지 사유와 따로 다루는 예외

정지 사유와 예외는 서로 다른 결과로 다뤄야 합니다. 도구 본문에서 난 오류는 기본적으로 루프를 멈추지 않습니다. 오류 결과로 모델에 돌아가고, 모델이 방법을 바꾸거나 다른 길을 찾습니다. 반면 라이프사이클 훅의 실패, 다 써 버린 모델 재시도, 잘못된 설정 같은 SDK 오류는 애플리케이션까지 전파됩니다. 모델 응답 하나가 프로바이더의 호출당 출력 상한을 넘는 경우도 여기에 속합니다. 이것은 `limit_*` 정지 사유와 별개이며, 루프는 결과를 돌려주지 않고 예외를 던집니다. Python은 회수한 부분 메시지를 대화 기록에 추가한 다음 `MaxTokensReachedException`을 발생시킵니다. 다음 요청을 줄이거나, 프로바이더 출력 상한을 높이거나, 작업을 더 작게 나눈 뒤 같은 에이전트를 다시 호출하면 그 기록에서 이어서 진행합니다. TypeScript의 `MaxTokensError`는 `partialMessage`로 미완성 메시지를 보여 주지만 기록에는 추가하지 않습니다. 요청별 에이전트 패턴에서는 이런 예외가 `handle_ticket` 밖의 요청 경계까지 올라오므로, 웹 핸들러 쪽에서 따로 잡아 응답 코드로 바꾸십시오. 속도 제한 같은 프로바이더 오류에 대한 자동 재시도와 그 조정은 바로 이어지는 재시도 전략에서 다룹니다.

## 스로틀링에만 반응하는 기본 지수 백오프와 그 재정의

아무 설정도 하지 않은 에이전트는 모델 프로바이더의 스로틀링(throttling) 오류만 지수 백오프(exponential backoff)로 재시도합니다. 재시도 횟수, 대기 시간, 재시도할 오류 종류는 `ModelRetryStrategy`로 조정하고, 그 범위를 벗어나는 재시도는 훅에서 직접 요청합니다.

### 4초에서 시작해 두 배씩 늘어나는 기본 대기

기본값에서 에이전트는 스로틀링 오류를 최대 5회 재시도합니다. 첫 시도를 합치면 총 6회이고, 대기 시간은 4초에서 시작해 지수적으로 늘어납니다. 시도별 대기 시간을 표로 정리했습니다.

| 시도 | 실패 후 동작 |
|---|---|
| 1 | 4초 대기 |
| 2 | 8초 대기 |
| 3 | 16초 대기 |
| 4 | 32초 대기 |
| 5 | 64초 대기 |
| 6 | 예외 발생 |

재시도 대상은 프로바이더가 속도 제한 때문에 던지는 `ModelThrottledException`(TypeScript는 `ModelThrottledError`)뿐입니다. 다른 예외는 재시도 없이 바로 전파됩니다. 요청마다 짧은 데드라인을 거는 서비스라면 기본 백오프를 그대로 두지 말고 데드라인 안에 들어오도록 줄이십시오.

### 횟수·지연·대상을 바꾸는 `ModelRetryStrategy`

재시도 파라미터는 전략 객체를 만들어 에이전트 생성자에 넘겨서 바꿉니다. `max_attempts`는 첫 시도를 포함한 총 시도 수이고, `initial_delay`는 첫 재시도 전에 기다리는 초 단위 시간, `max_delay`는 백오프 지연의 상한입니다. 어떤 오류를 재시도할지 넓히거나 좁히려면 서브클래스에서 `is_retryable`을 재정의합니다. 기본 구현은 `ModelThrottledException`만 재시도합니다. `super().is_retryable(exception)`를 호출하면 이 기본 판단을 그대로 유지하므로, 재시도 정책의 나머지를 다시 구현하지 않고도 오류 종류를 추가하거나 빼낼 수 있습니다. 아래 코드는 티켓 에이전트의 재시도 대상에 `TimeoutError`를 더하고, 시도 횟수와 지연을 데드라인에 맞게 줄입니다.

```python
# excerpt: agent.py의 에이전트 생성부 (MODEL_ID와 도구들은 앞에서 정의)
from strands import Agent, ModelRetryStrategy
from strands.models import BedrockModel


class TicketRetryStrategy(ModelRetryStrategy):
    def is_retryable(self, exception: Exception) -> bool:
        # Keep the default throttling retries and also retry timeouts
        return super().is_retryable(exception) or isinstance(exception, TimeoutError)


def build_agent() -> Agent:
    return Agent(
        model=BedrockModel(model_id=MODEL_ID),
        tools=[lookup_order, lookup_orders, issue_refund],
        # A fresh strategy instance for every agent
        retry_strategy=TicketRetryStrategy(max_attempts=3, initial_delay=2, max_delay=60),
    )
```

전략 인스턴스는 턴마다 바뀌는 상태를 갖고 있어서 여러 에이전트가 함께 쓰면 안 됩니다. 에이전트마다 새로 만들어야 합니다. 앞서 본 요청별 새 에이전트 패턴이라면 `handle_ticket` 안에서 `build_agent()`를 부르기만 해도 이 조건을 지킬 수 있습니다. TypeScript에서는 `ModelRetryStrategy`가 추상 기반 클래스입니다. 일반적인 설정형 전략은 `DefaultModelRetryStrategy`이며, 재시도 결정을 완전히 직접 통제하려면 추상 클래스를 상속해 `computeRetryDecision`을 구현합니다.

자동 재시도를 아예 끄려면 `retry_strategy=None`을 넘깁니다. TypeScript에서는 `retryStrategy: null`입니다. API 게이트웨이나 큐 소비자처럼 바깥 계층이 이미 재시도한다면, 에이전트 쪽 재시도를 끄는 편이 시도가 중복으로 쌓이지 않아 낫습니다.

### 오류 내용을 보고 직접 거는 훅 재시도

모델 오류를 확인하고 재시도만 요청하면 되는 경우에는 훅이 더 간단합니다. `AfterModelCallEvent` 콜백에서 `event.retry = True`를 설정하면 됩니다. 이 방식에서는 훅이 지연을 자동으로 넣지 않으므로 대기는 직접 구현해야 합니다. 이 플래그는 예외뿐 아니라 응답 검증이나 내용 품질 검사 같은 어떤 기준으로도 설정할 수 있습니다. 플래그가 설정되면 루프가 모델 호출을 다시 하는 흐름은 앞서 사이클 구조에서 본 내용과 같습니다.

훅으로 재시도할 때 주의할 부분은 카운터 관리입니다. 아래 훅은 `ServiceUnavailable` 오류에만 반응합니다. 카운터를 `BeforeInvocationEvent`에서 초기화하고, 모델 호출이 성공했을 때도 다시 초기화합니다. 대기 시간은 `2 ** retry_count`초로 늘어납니다.

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
            # Reset counts on successful call
            self.reset_counts()


agent = Agent(hooks=[RetryOnServiceUnavailable(max_retries=3)])
```

`CustomRetry` 같은 단순한 예제는 `attempts`를 한 번도 초기화하지 않습니다. 이런 훅을 오래 사는 에이전트에 붙이면 첫 몇 번의 실패에서 재시도 예산을 다 써 버리고, 그 뒤로는 재시도가 일어나지 않습니다. 앞서 본 `after_model_call`의 Guide와 마찬가지로 상한은 핸들러가 직접 지켜야 합니다. TypeScript의 `AfterModelCallEvent`는 1부터 세는 `attemptCount`를 제공하므로, 훅 밖에 카운터를 두지 않고도 재시도 횟수를 제한할 수 있습니다.

## 대화 기록·에이전트 상태·호출 상태의 세 층과 세션 영속화

티켓 대화를 재시작 뒤에도, 다른 머신에서도 이어 가려면 세 가지 상태 가운데 대화 기록과 에이전트 상태를 세션 관리자(session manager)에 맡기고, 세션 ID는 티켓마다 따로 주면 됩니다.

### 수명이 서로 다른 세 가지 상태

세 상태가 각각 무엇을 담고 얼마나 유지되는지 표로 정리했습니다.

| 구분 | 내용 | 수명 | 세션 관리자 저장 |
|---|---|---|---|
| `agent.messages` | 도구 호출과 결과를 포함한 모든 사용자·어시스턴트 메시지 | 호출 사이에 유지 | 저장 대상 |
| `agent.state` | JSON으로 직렬화할 수 있는 키-값 | 호출과 도구 실행을 넘어 유지 | 저장 대상 |
| `invocation_state` | 호출할 때 넘기는 딕셔너리 | 한 호출 안의 재귀 사이클 전체 | 세션이 담는 항목(대화 기록·에이전트 상태·대화 관리자) 목록에 없음 |

`agent.messages`는 Strands가 직접 관리합니다. 호출 사이에 유지하고, 추론할 때마다 모델에 보내고, 도구 실행의 컨텍스트로 넘기며, 컨텍스트 윈도우에 들어가도록 잘라 냅니다. 잘라 내는 일은 대화 관리자(conversation manager)가 맡습니다. 따로 지정하지 않으면 `SlidingWindowConversationManager`가 쓰입니다. Python에서는 `Agent(messages=[...])`처럼 기존 메시지를 넣어 에이전트를 만들 수 있어서, 이전 대화를 이어 가거나 컨텍스트를 미리 채울 수 있습니다. 직접 도구 호출은 모델을 거치지 않고 `agent.tool.lookup_order(...)`처럼 코드에서 도구를 바로 부르는 방식이며, 이런 호출도 기본적으로 대화 기록에 남습니다. Python에서는 `agent.tool.lookup_order(..., record_direct_tool_call=False)`처럼 해당 도구 호출의 인자로 `record_direct_tool_call=False`를 넘기면 그 호출은 기록되지 않습니다.

`agent.state`는 영속화와 복원을 염두에 두고 JSON 직렬화 가능 여부를 검증합니다. Python에서 람다처럼 직렬화할 수 없는 값을 `set`하면 `ValueError`가 납니다. 도구 안에서는 `tool_context.agent.state`로 이 상태를 읽고 씁니다. 이때 Python 도구의 매개변수 이름은 반드시 `tool_context`여야 합니다.

`invocation_state`는 생략하면 `{}`입니다. 훅과 도구가 같은 객체를 참조로 공유하므로, 앞에서 바꾼 값은 같은 호출 안의 뒤쪽 훅과 도구에서 그대로 보입니다. 이 상태는 모델 컨텍스트에 들어가지 않습니다. 그래서 요청 ID나 고객 ID처럼 도구에는 필요하지만 모델에는 보여 줄 필요가 없는 값을 담기에 알맞습니다.

```python
from strands import Agent, tool, ToolContext


@tool(context=True)
def current_customer(tool_context: ToolContext) -> str:
    """Return the customer ID carried in the invocation state."""
    user_id = tool_context.invocation_state.get("user_id", "unknown")
    return f"Current customer: {user_id}"


agent = Agent(tools=[current_customer])
result = agent("Who am I?", invocation_state={"request_id": "r-42", "user_id": "u-1"})
```

결과에서 이 상태를 돌려받는 방식은 언어마다 다릅니다. Python은 호출 상태 딕셔너리 안의 `request_state` 키(슬롯)에 담긴 값만 `result.state`로 돌려주고, TypeScript는 상태 전체를 `result.invocationState`로 돌려줍니다.

### 티켓마다 세션을 붙이는 `SnapshotSessionManager`

세션은 대화 기록, 에이전트 상태, 대화 관리자 같은 상태 정보를 담습니다. 내장 영속화 기능이 이 정보를 자동으로 저장하고 복원하므로, 에이전트는 멈춘 지점에서 대화를 이어 갑니다. 새 단일 에이전트 세션이라면 Python에서는 `SnapshotSessionManager`를 씁니다. `FileSessionManager`와 `S3SessionManager`는 Graph, Swarm, 기존 리포지토리 형식 세션을 위한 호환 경로로 남아 있습니다. TypeScript는 `SessionManager` 하나로 두 경우를 모두 처리합니다. 이 관리자는 에이전트용 Plugin으로 붙습니다. Python의 `SnapshotSessionManager`는 에이전트가 초기화될 때 최신 스냅샷을 자동으로 복원하고, 정해진 생명주기 이벤트가 올 때마다 에이전트를 다시 캡처해 최신 스냅샷(`snapshot_latest`)을 덮어씁니다.

티켓 에이전트에서는 티켓 ID로 세션 ID를 만들고, 여러 머신이 같은 세션을 읽을 수 있도록 S3 저장소를 연결합니다.

```python
# excerpt: 티켓별 세션 연결 부분만 표시 (MODEL_ID, lookup_order, issue_refund와 앞 절의 승인·재시도·구조화 출력 설정은 앞에서 정의)
from strands import Agent
from strands.models import BedrockModel
from strands.session import SnapshotSessionManager
from strands.storage import S3Storage


def build_session_agent(ticket_id: str) -> Agent:
    session_manager = SnapshotSessionManager(
        session_id=f"ticket-{ticket_id}",
        storage=S3Storage(bucket="my-agent-sessions", prefix="production/"),
    )
    return Agent(
        model=BedrockModel(model_id=MODEL_ID),
        tools=[lookup_order, issue_refund],
        session_manager=session_manager,
    )


agent = build_session_agent("T-881")
agent("Order A-1001 never arrived. Can I get a refund?")
print(agent.session_id)  # "ticket-T-881"
```

저장소는 관리자에 지정한 것이 에이전트에 지정한 것보다 우선합니다. 어느 쪽에도 없으면 Python은 `LocalFileStorage("./.strands/")`를 쓰고, TypeScript는 초기화 단계에서 실패합니다. 세션 관리자를 붙이지 않은 에이전트의 `session_id`는 에이전트를 만들 때 생성한 임의의 8자리 hex 문자열이며, 재시작하면 사라집니다.

### 시점을 고정하는 스냅샷과 세션 삭제

세션 관리자가 하는 자동 저장과 별개로, 특정 시점을 직접 캡처하고 되돌리려면 스냅샷(snapshot)을 씁니다. 예를 들어 환불을 실행하기 직전에 체크포인트를 남겨 두면 나중에 그 상태로 돌아갈 수 있습니다. 스냅샷 ID는 UUID v7이므로 문자열 순서가 곧 시간 순서입니다. 아래 코드는 `save_snapshot`에 `is_latest=False`를 넘겨 바뀌지 않는(immutable) 체크포인트를 따로 남기고, 그 ID로 다시 복원합니다.

```python
import asyncio

from strands import Agent
from strands.session import SnapshotSessionManager
from strands.storage import LocalFileStorage

session_manager = SnapshotSessionManager(
    session_id="ticket-T-881",
    storage=LocalFileStorage("./sessions/"),
)
agent = Agent(session_manager=session_manager)


async def restore() -> None:
    # Force an immutable checkpoint and read it back later
    snapshot_id = await session_manager.save_snapshot(agent, is_latest=False)
    snapshot_ids = await session_manager.list_snapshot_ids(agent)
    assert snapshot_id in snapshot_ids
    await session_manager.restore_snapshot(agent, snapshot_id=snapshot_id)


asyncio.run(restore())
```

티켓을 닫을 때는 `asyncio.run(session_manager.delete_session())`을 호출합니다. 파일시스템 저장소에서는 세션 루트 디렉터리 전체가, S3에서는 세션 prefix 아래의 모든 객체가 지워집니다.

### 분산 락이 없는 세션에서 지켜야 할 규칙

내장 세션 관리자는 분산 락을 잡지 않습니다. 한 에이전트 인스턴스에 호출이 겹쳐 들어오는 것을 막는 단일 인스턴스 호출 가드도 프로세스 안에서만 동작하므로, 다른 곳에서 돌고 있는 두 번째 writer를 알아채지 못합니다. 세션 ID와 에이전트 ID가 같은 에이전트가 동시에 살아 있으면 두 가지 일이 생길 수 있습니다. 호출이 겹치면 서로의 턴을 덮어쓰고, 호출이 차례로 들어오면 두 대화가 하나의 기록으로 합쳐집니다. 어느 경우에도 오류는 나지 않습니다. 기본 에이전트 ID가 Python에서는 `agent_id="default"`, TypeScript에서는 `id: 'agent'`로 정해져 있어서, 실행마다 에이전트를 따로 만드는 구조에서도 이런 충돌이 쉽게 일어납니다. 세션 생성도 확인한 뒤 쓰는 방식이라 원자적이지 않습니다. 새 세션 ID로 두 요청이 동시에 콜드 스타트하면 둘 다 성공하고, 나중에 쓴 쪽이 남습니다.

이런 문제를 피하려면 대화마다 고유한 세션 ID를 써야 합니다. 에이전트 ID만 다르면 여러 에이전트가 한 세션 ID를 함께 쓸 수 있으며, 이때 세션은 각 에이전트의 메시지와 상태를 따로 담는 네임스페이스 역할을 합니다. Python에서는 에이전트 ID를 `Agent(agent_id="triage", session_manager=session_manager)`처럼 생성자의 `agent_id` 인자로 지정합니다. 앞서 본 요청별 새 에이전트 패턴과 함께 쓴다면, 같은 티켓에 요청이 동시에 두 개 들어오지 않도록 큐나 라우팅 계층에서 티켓 단위로 직렬화하십시오. 세션 관리자는 스레드 안전하지 않습니다. 프로덕션에서는 오래된 세션을 정리할 TTL도 함께 두는 편이 좋습니다.

세션 저장소는 신뢰할 수 있는 데이터 저장소로 다뤄야 합니다. 클론한 리포지토리 안의 세션 파일처럼 출처를 믿을 수 없는 세션은 로드하면 안 됩니다. SDK는 저장 디렉터리 안의 심볼릭 링크를 막지 않고 따라갑니다. 따라서 쓰기 권한을 가진 공격자가 `message_0.json` 같은 링크를 만들면 임의 파일의 내용이 대화 기록에 들어올 수 있습니다. 저장 디렉터리 권한은 에이전트 프로세스만 읽고 쓰도록 제한하십시오.

## AWS 친화적 최소 루프를 고를 때와 피할 때

Strands는 계획을 모델에 맡기는 작은 루프 위에 MCP, OpenTelemetry, Bedrock을 무리 없이 얹고 싶을 때 잘 맞습니다. 대신 성숙도, 배포 형태, 분산 상태에서 생기는 빈틈은 팀이 직접 메워야 합니다.

### 다른 프레임워크와 비교한 Strands의 위치

한 서드파티 비교 글은 Strands를 AWS가 오픈소스로 공개한 최소한의 모델 주도 루프로 분류합니다. 이 글은 LLM이 직접 계획을 세우는 구조를 다른 프레임워크와 반대 방향의 선택으로 보고, 강점으로 OpenTelemetry 관측성과 아주 큰 도구 카탈로그에 대응하는 도구 검색을 꼽습니다. 같은 글은 비교한 여섯 프레임워크가 모두 MCP를 지원하며 Strands는 MCP를 일급으로 다룬다고 보고합니다. MCP 서버로 만든 도구는 어느 프레임워크로 옮겨도 다시 쓸 수 있습니다. 그래서 도구 통합만 보고 프레임워크를 고를 필요는 없습니다.

결정에 더 큰 영향을 주는 쪽은 클라우드입니다. 2026년 3월에 나온 또 다른 서드파티 글은 Strands를 모델에 묶이지 않는 프레임워크로 분류합니다. 그러면서도 Strands는 Bedrock에서, ADK는 Vertex에서 가장 매끄럽게 동작한다고 평가합니다. 같은 글은 Strands가 OpenTelemetry를 거쳐 AWS X-Ray와 CloudWatch에 깔끔하게 연결된다고 덧붙입니다. 티켓 분류 에이전트를 호출할 때마다 돌아오는 `AgentResult`에는 그 실행의 메시지, 메트릭, 트레이스가 담겨 있습니다.

### 감수해야 하는 한계

Bedrock 궁합을 평가한 앞의 서드파티 글은 몇 가지 약점도 보고합니다. Strands는 비교적 새 프레임워크라 커뮤니티가 작고, 일부 기능은 아직 실험 단계입니다. 이 글은 또 Lambda 배포가 스트리밍을 지원하지 않아서 스트리밍에는 AgentCore Runtime이나 Fargate가 필요하다고 보고합니다. `stream_async`로 티켓 응답을 흘려보내는 화면을 계획하고 있다면 배포 대상부터 정해 두는 편이 좋습니다.

상태 영속화는 앞의 7월 비교 글이 QubitTool의 지적을 인용해 대부분 개발자가 직접 관리해야 한다고 평가합니다. 현재 SDK에는 `SnapshotSessionManager`처럼 대화 기록과 에이전트 상태를 저장소에 자동으로 저장하고 복원하는 내장 세션 관리자가 있습니다. 다만 내장 세션 관리자에는 분산 락이 없습니다. 그래서 같은 세션 ID와 같은 에이전트 ID를 쓰는 두 writer는 오류 없이 서로 덮어쓰거나 두 대화를 합쳐 버립니다. 저장과 복원은 SDK가 맡지만, 여러 인스턴스를 조율하는 일은 서비스 쪽 책임입니다.

### 운영에서 다시 만나는 실패 모드

앞에서 다룬 제어 장치에는 각자 경계가 있습니다. 운영에서 문제가 터지는 곳도 대체로 이 경계입니다.

| 실패 모드 | 원인 | 대응 |
|---|---|---|
| 아주 긴 도구 체인의 중단 | 한 저자의 Python SDK 소스 분석에 따르면 다음 사이클이 재귀로 이어져 Python 재귀 한도에 먼저 닿을 수 있음 | 턴 상한을 재귀 한도보다 훨씬 낮게 설정 |
| 토큰 예산 초과 | 한도를 반복이 시작될 때만 검사함 | 예산에 한 턴 분량의 여유를 둠 |
| 끝나지 않는 Guide 재시도 | after_model_call의 Guide에 재시도 상한이 없음 | 핸들러에서 횟수를 세다가 Deny로 격상 |
| 위험한 환불의 자동 승인 | 위험 분류기는 판단일 뿐 보장이 아님 | `issue_refund`를 부정 표기하고 Cedar 정책을 앞에 둠 |
| 대화가 소리 없이 섞임 | 세션에 분산 락이 없음 | 티켓마다 세션 ID를 두고 티켓 단위로 직렬화 |

이 글의 판단은 이렇습니다. AWS 위에서 Bedrock 모델과 MCP 도구로 단일 에이전트 또는 소수 에이전트 서비스를 만들고, 흐름 제어는 모델에 맡기되 한도·승인·재시도를 코드로 묶고 싶다면 Strands가 가장 적은 코드로 그 목적에 닿습니다. 반대로 실행 경로를 그래프로 명시해 검증해야 하는 업무, 다른 클라우드가 중심인 조직, Lambda에서 스트리밍 응답이 꼭 필요한 서비스, 커뮤니티 예제와 서드파티 통합에 크게 기대는 팀이라면 다른 선택지를 먼저 검토하십시오. 실험 기능은 버전을 고정한 뒤에 쓰기를 권합니다.

- - -

### Sources
* [Agent Loop &#124; Strands Agents](https://strandsagents.com/docs/user-guide/sdk/agents/agent-loop/)
* [Model Providers &#124; Strands Agents](https://strandsagents.com/docs/user-guide/sdk/model-providers/)
* [Hooks &#124; Strands Agents](https://strandsagents.com/docs/user-guide/sdk/agents/hooks/)
* [Interventions &#124; Strands Agents](https://strandsagents.com/docs/user-guide/sdk/agents/interventions/)
* [Structured output &#124; Strands Agents](https://strandsagents.com/docs/user-guide/sdk/agents/structured-output-behavior/)
* [Production Lifecycle Controls &#124; Strands Agents](https://strandsagents.com/docs/user-guide/sdk/agents/lifecycle-controls/)
* [Persist state across sessions &#124; Strands Agents](https://strandsagents.com/docs/user-guide/sdk/agents/session-management/)
* [Python Quickstart &#124; Strands Agents](https://strandsagents.com/docs/user-guide/sdk/quickstart/python/)
* [Approve before your agent acts &#124; Strands Agents](https://strandsagents.com/docs/user-guide/sdk/agents/approve-before-acting/)
* [Return structured output &#124; Strands Agents](https://strandsagents.com/docs/user-guide/sdk/agents/structured-output/)
* [Retry Strategies &#124; Strands Agents](https://strandsagents.com/docs/user-guide/sdk/agents/retry-strategies/)
* [State Management &#124; Strands Agents](https://strandsagents.com/docs/user-guide/sdk/agents/state/)
* [Picking an AI Agent Framework in 2026 &#124; AWS Builder Center](https://builder.aws.com/content/3AzsgG6TreTO3uLRqpWNxfEyUhe/picking-an-ai-agent-framework-in-2026)
* [How the Strands Agent Loop Actually Works - DEV Community](https://dev.to/aws-builders/how-the-strands-agent-loop-actually-works-2m17)
* [strands-agents · PyPI](https://pypi.org/project/strands-agents/)
* [LangGraph vs CrewAI vs Claude Agent SDK vs OpenAI Agents SDK: Choosing a Multi-Agent Framework in 2026](https://buildyour.group/blog/best-multi-agent-frameworks-2026)
* [harness-sdk/strands-py/src/strands/event_loop/event_loop.py at main · strands-agents/harness-sdk · GitHub](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/event_loop/event_loop.py)
