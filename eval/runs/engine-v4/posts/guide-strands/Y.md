---
layout: post
title: "Strands Agents SDK Agent Loop and Its Control Surfaces"
date: 2026-11-03 05:33:51
description: "Strands Agents SDK의 에이전트 루프가 추론과 도구 실행을 반복하는 원리를 설명하고, 호출 한도와 취소, 훅과 미들웨어, 재시도 전략, 대화 관리, 구조화된 출력까지 루프를 제어하는 방법을 Python과 TypeScript 예제로 정리합니다."
categories: ["Tech Guides", "Agentic-AI"]
tags: ["Tool-Execution", "Stop-Reasons", "Hooks-and-Middleware", "Conversation-Management", "Structured-Output", "Agentic-AI"]
cover: /assets/images/agentic-ai.jpg
use_math: true
---
## 에이전트 루프의 동작 원리

언어 모델은 질문에 답할 수 있지만, 파일을 읽거나 데이터베이스를 조회하거나 API를 호출하는 일은 모델 혼자 하지 못합니다. 학습된 지식만으로 해결되지 않는 요청을 받으면 모델은 바깥 세계에 손을 뻗어야 합니다. 그 왕복을 조율하는 계층이 에이전트 루프(agent loop)입니다. 에이전트 루프는 Strands Agents SDK(PyPI의 `strands-agents`, npm의 `@strands-agents/sdk`)에서 가장 기본이 되는 실행 단위입니다. 호출 한도, 취소, 훅(hook), 재시도, 대화 관리 같은 제어 장치는 모두 이 루프 위에서 동작합니다. 그래서 루프가 어떤 순서로 무엇을 쌓아 가는지 정확히 알아 두어야 나머지 기능이 언제 개입하는지 예측할 수 있습니다.

### 추론-실행-관찰 사이클

루프의 원리는 단순합니다. 모델을 호출하고, 모델이 도구를 쓰려 하는지 확인하고, 그렇다면 도구를 실행한 뒤 그 결과를 붙여 모델을 다시 호출합니다. 모델이 더 이상 도구를 요청하지 않고 최종 응답을 내놓을 때까지 이 과정을 되풀이합니다.

```text
                 ┌──────── 매 바퀴가 대화 이력에 추가됨 ────────┐
                 │                                              │
┌──────────────────┐      ┌──────────────────┐   도구 호출   ┌──────────────────┐
│ Request          │ ───▶ │ Model            │ ────────────▶ │ Tools            │
│ 프롬프트 + 이력  │      │ 다음 단계 결정   │ ◀──────────── │ 도구 호출 실행   │
└──────────────────┘      └──────────────────┘     결과      └──────────────────┘
                                   │
                                   │ 도구 호출 없음
                                   ▼
                          ┌──────────────────┐
                          │ Response         │
                          │ 최종 메시지      │
                          └──────────────────┘
```

다이어그램에서 볼 곳은 Model과 Tools 사이를 오가는 두 화살표입니다. 도구 호출이 있으면 결과가 다시 모델로 돌아오는 재귀 구조라서, 도구 결과가 다음 추론의 입력이 됩니다. 도구 호출이 없을 때만 흐름이 원을 빠져나가 최종 메시지로 갑니다. 어느 경로로 돌든 매 바퀴가 대화 이력에 무언가를 더합니다.

이 구조에는 설계 철학이 담겨 있습니다. Strands는 작업 흐름을 개발자가 하드코딩하는 대신 모델 주도(model-driven) 방식을 택했습니다. 개발자는 언어 모델, 시스템 프롬프트, 도구 목록 세 가지를 제공하고, 어떤 도구를 어떤 순서로 몇 번 쓸지는 모델이 현재 맥락을 보고 스스로 정합니다. 추론과 도구 사용을 번갈아 하는 ReAct 계열 패턴을 프레임워크의 기본 실행 모델로 삼은 형태입니다. 상태 기계나 분기 로직을 직접 짜지 않아도 다단계 작업이 돌아가는 대신, 루프가 몇 바퀴 돌지 미리 알 수 없습니다. 바로 이 불확실성 때문에 한도·취소·훅 같은 제어 장치가 필요합니다.

루프의 골격을 Python 의사 코드로 적으면 다음과 같습니다. SDK의 실제 API가 아니라 동작 순서를 보여 주는 개념 코드입니다.

```python
# 개념 설명용 의사 코드입니다. Strands SDK의 실제 API가 아닙니다.
def agent_loop(history, model, registry):
    while True:
        assistant_msg = model.invoke(history)       # 추론: 다음 단계 결정
        history.append(assistant_msg)

        tool_uses = [c for c in assistant_msg.content if c.kind == "tool_use"]
        if not tool_uses:
            return assistant_msg                    # 도구 요청이 없으면 최종 응답

        # 실행: 한 메시지 안의 요청 전부를 처리하고, 각 결과를 요청 ID로 짝짓는다
        results = [execute_tool(u, registry) for u in tool_uses]
        # 관찰: 결과 전부를 사용자 메시지 하나에 모아 이력에 넣는다
        history.append({"role": "user", "content": results})


def execute_tool(tool_use, registry):
    try:
        validate(tool_use.input, registry.schema_of(tool_use.name))  # 스키마 검증
        tool = registry.lookup(tool_use.name)                        # 레지스트리 조회
        output = tool(**tool_use.input)
        return tool_result(tool_use.id, status="success", content=output)
    except Exception as e:
        # 예외로 루프를 끝내지 않고, 오류를 결과로 바꿔 모델에 돌려준다
        return tool_result(tool_use.id, status="error", content=str(e))
```

`tool_uses`가 리스트인 데는 이유가 있습니다. 모델은 한 번의 응답에서 도구를 하나 이상 요청할 수 있습니다. 예를 들어 파일 두 개를 동시에 읽겠다고 하면 어시스턴트 메시지 하나에 도구 사용 요청이 둘 들어 있습니다. 개념 코드에서는 각 요청을 실행한 결과에 원래 요청의 `tool_use.id`를 붙이고, 결과 전부를 사용자 메시지 하나에 담아 돌려줍니다. 모델은 이 ID를 보고 어느 결과가 어느 요청의 답인지 구분합니다. 어떤 요청이 실패하더라도 그 자리에 오류 결과가 들어가므로, 요청 하나에 결과 하나라는 짝은 깨지지 않습니다.

실제 SDK에서는 모델 호출이 끝날 때마다 정지 이유(stop reason)가 함께 보고됩니다. 도구 사용(`tool_use`)이면 요청된 도구를 실행하고 결과를 이력에 덧붙인 뒤 모델을 다시 호출하며, 정상 종료(`end_turn`)이면 루프를 빠져나와 모델의 최종 메시지를 반환합니다.

### 구체적인 예: 코드베이스 보안 분석

모델이 기억만으로 할 수 없는 작업을 하나 따라가 보겠습니다. "이 코드베이스에서 보안 취약점을 분석해 달라"는 요청입니다. 아래 표는 루프가 다섯 바퀴 도는 동안 매 바퀴에 모델이 무엇을 결정했고 이력에 무엇이 쌓였는지 손으로 추적한 예시입니다(설명용 예시이며 실제 실행 로그가 아닙니다).

| 반복 | 모델이 본 것과 판단 | 요청한 도구 | 이력에 추가되는 것 |
| --- | --- | --- | --- |
| 1 | 요청만 있음. 구조부터 파악해야 한다고 판단 | 파일 목록 도구(저장소 루트) | 도구 사용 요청, 디렉터리 구조 결과 |
| 2 | 디렉터리 구조를 보고 진입점을 식별 | 파일 읽기 도구(진입점 파일) | 도구 사용 요청, 애플리케이션 코드 |
| 3 | 데이터베이스 쿼리를 발견하고 SQL 인젝션 가능성을 의심 | 파일 읽기 도구(DB 모듈) | 도구 사용 요청, DB 모듈 코드 |
| 4 | 사용자 입력이 SQL에 그대로 이어 붙는 취약점 확인. 영향 범위를 알아야 함 | 코드 검색 도구(취약 함수 호출 지점) | 도구 사용 요청, 호출 지점 12곳 |
| 5 | 필요한 정보를 모두 확보 | 없음 | 취약점·영향 위치·수정 방안을 담은 최종 보고서 |

다섯 바퀴 모두 같은 패턴입니다. 모델은 맥락을 받고, 행동할지 응답할지 결정하고, 그 결정에 따라 루프를 이어 가거나 빠져나옵니다. 반복 3에서 DB 모듈을 열어 보기로 한 판단이나 반복 4에서 호출 지점을 검색하기로 한 판단은 개발자가 미리 짜 둔 흐름이 아니라, 앞선 도구 결과를 보고 모델이 갱신한 이해에서 나왔습니다. 반복 1의 결과가 달랐다면 이후 경로도 전부 달라졌을 것입니다.

### 메시지와 대화 이력의 누적

루프 안을 흐르는 메시지의 역할은 사용자(user)와 어시스턴트(assistant) 둘뿐입니다. 도구 실행 결과에 별도 역할이 없고 사용자 메시지에 실려서 처음에는 낯설 수 있는데, 모델 관점에서 도구 결과는 "바깥 세계에서 들어온 입력"이므로 사용자 쪽에 놓이는 것이 자연스럽습니다.

| 역할 | 담길 수 있는 내용 |
| --- | --- |
| 사용자 메시지 | 사용자의 텍스트 입력, 이전 도구 실행의 결과, 파일·이미지·오디오·비디오 같은 미디어 |
| 어시스턴트 메시지 | 사용자에게 보여 줄 텍스트 응답, 실행 시스템에 보내는 도구 사용 요청, 추론 과정(모델이 지원하는 경우) |

요청이 들어오면 에이전트는 도구를 등록하고 대화 관리자(conversation manager)를 준비한 뒤, 사용자 입력을 대화 이력의 첫 메시지로 넣고 첫 반복을 시작합니다. 그 뒤로는 반복마다 어시스턴트 메시지(도구 사용 요청)와 사용자 메시지(도구 결과)가 한 쌍씩 덧붙습니다. $$k$$번째 반복의 어시스턴트 메시지를 $$a_k$$, 그에 대한 도구 결과 메시지를 $$r_k$$라 하면 이력의 성장은 다음처럼 쓸 수 있습니다.

$$
H_k = H_{k-1} \oplus a_k \oplus r_k, \qquad H_0 = (\text{user request})
$$

도구를 요청하지 않는 마지막 반복에서는 실행할 도구가 없으므로 $$r_k$$ 없이 최종 응답 $$a_k$$만 더해집니다.

이 누적이 에이전트의 작업 기억입니다. 보안 분석 예에서 반복 5의 모델이 보고서를 쓸 수 있었던 것은 원래 요청뿐 아니라 그동안 호출한 모든 도구와 받은 모든 결과를 한꺼번에 보고 있었기 때문입니다. 여러 단계에 걸친 추론은 이렇게 커지는 맥락 위에서만 가능합니다.

같은 성질이 비용과 한계로도 이어집니다. 모델은 매 반복 이력 전체를 입력으로 받으므로, 반복이 늘수록 한 번 호출에 보내는 토큰도 함께 늘어납니다. 도구가 긴 결과를 돌려주면 그 내용이 이후 모든 반복에 실려 다닙니다. 이력은 결국 모델의 컨텍스트 윈도우(context window)를 넘어설 수 있고, 대화 관리자가 여러 전략으로 이력을 컨텍스트 윈도우 안에 맞추면서 가장 관련 있는 정보를 남깁니다. 도구 결과를 원문 그대로가 아니라 필요한 발췌나 요약으로 돌려주도록 설계하면 누적 비용을 처음부터 줄일 수 있습니다.

### 도구 실행 시스템

모델이 도구를 요청하면 실행 시스템이 네 단계를 차례로 처리합니다.

1. 요청 입력을 해당 도구의 스키마에 맞춰 검증합니다.
2. 도구 레지스트리에서 요청된 이름의 도구를 찾습니다.
3. 도구 실행을 오류 처리로 감싸서 실행합니다.
4. 실행 결과를 도구 결과 메시지 형식으로 만들어 이력에 넣습니다.

모델이 어떤 도구를 고를지는 도구의 이름, 설명, 입력 스키마에 달려 있습니다. Python에서는 `@tool` 데코레이터로 일반 함수를 도구로 만들며, 모델은 함수의 독스트링을 읽고 언제 이 도구를 쓸지 판단합니다. 보안 분석 예의 파일 읽기 도구라면 다음처럼 정의할 수 있습니다.

```python
from strands import tool

@tool
def read_file(path: str) -> str:
    """저장소 안의 파일 하나를 읽어 그 내용을 텍스트로 반환합니다.

    Args:
        path: 저장소 루트 기준 파일 경로
    """
    with open(path, encoding="utf-8") as f:
        return f.read()
```

TypeScript에서는 `tool()` 함수에 이름, 설명, Zod 입력 스키마, 콜백을 넘깁니다. 같은 도구를 옮기면 다음과 같습니다.

```typescript
// tool과 z의 import 구문은 생략했습니다.
import { readFile } from 'node:fs/promises'

const readFileTool = tool({
  name: 'read_file',
  description: '저장소 안의 파일 하나를 읽어 그 내용을 텍스트로 반환합니다.',
  inputSchema: z.object({ path: z.string() }),
  callback: async (input) => {
    return readFile(input.path, 'utf-8')
  },
})
```

두 정의를 나란히 놓으면 모델에 전달되는 계약이 같습니다. 이름이 무엇을 하는 도구인지 가리키고, 설명이 언제 써야 하는지 알려 주고, 스키마가 어떤 입력을 받는지 정합니다. Python은 함수 시그니처와 독스트링에서, TypeScript는 명시적인 필드에서 이 정보를 얻을 뿐입니다. 모델이 엉뚱한 도구를 고른다면 대개 이 계약, 특히 설명이 모호한 경우입니다.

두 정의 모두 오류 처리를 따로 하지 않습니다. 모델이 존재하지 않는 경로를 넘기면 파일 읽기가 예외를 던지지만, 그 예외가 에이전트 호출 전체를 무너뜨리지는 않습니다. 실행 시스템이 예외를 붙잡아 오류 결과로 바꾸기 때문입니다.

### 오류 결과 처리: 실패도 관찰의 일부

실행 시스템은 성공한 결과와 실패를 모두 붙잡습니다. 도구가 실패하면 루프를 끝내는 예외를 던지는 대신, 오류 정보를 담은 오류 결과를 모델에 돌려줍니다. 앞의 의사 코드에서 `execute_tool`이 `except` 블록에서 `status="error"`인 결과를 반환하는 부분이 이 동작에 해당합니다.

보안 분석 예의 반복 2에서 모델이 진입점 경로를 잘못 적었다고 가정해 보겠습니다. 이력에는 다음과 같은 한 쌍이 쌓입니다(설명용 개념 예시이며 SDK의 실제 메시지 형식이 아닙니다).

```python
# 설명용 개념 예시입니다. 실제 메시지 형식과 다릅니다.
history += [
    {"role": "assistant", "content": [
        {"kind": "tool_use", "id": "t2", "name": "read_file",
         "input": {"path": "app/main.py"}},
    ]},
    {"role": "user", "content": [
        {"kind": "tool_result", "id": "t2", "status": "error",
         "content": "No such file or directory: 'app/main.py'"},
    ]},
]
```

다음 반복의 모델은 이 오류 결과와 함께 반복 1에서 받은 디렉터리 목록도 이력에서 그대로 봅니다. 목록에 진입점이 `src/app/main.py`로 나와 있었다면, 모델은 경로를 고쳐 `read_file`을 다시 요청하고 루프는 원래 계획대로 이어집니다. 실패한 요청과 오류 결과도 이력에 남으므로, 모델이 같은 실수를 되풀이하지 않게 하는 근거가 됩니다.

이렇게 설계한 이유는 에이전트 루프를 사이클로 보면 분명해집니다. 루프에서 도구 결과는 다음 추론의 입력이고, 실패 역시 모델에 유용한 관찰입니다. 검색 결과가 쓸모없었다면 모델은 질의를 바꿔 다시 검색하거나 다른 도구로 우회할 수 있습니다. 도구 실패가 곧 호출 실패였다면, 사소한 입력 실수 하나로 그때까지 쌓은 맥락이 모두 버려지고 애플리케이션이 처음부터 다시 시작해야 했을 것입니다.

> **핵심:** 도구 오류는 루프를 멈추는 사건이 아니라 이력에 쌓이는 또 하나의 관찰입니다. 그래서 도구의 오류 메시지는 사람이 아니라 모델이 읽는다고 생각하고 작성해야 합니다. "무엇이 왜 실패했고 어떻게 고칠 수 있는지"가 담긴 메시지일수록 모델이 다음 반복에서 스스로 회복할 가능성이 커집니다.

이 방식에도 대가가 있습니다. 모델이 같은 실패를 되풀이하며 반복을 소모할 수 있고, 도구 실패가 예외로 올라오지 않으므로 실패를 별도로 추적하려면 훅으로 관찰하는 방법을 씁니다. 에이전트는 호출 전후, 모델 호출 전후, 도구 실행 전후에 생명주기 이벤트를 내보내므로, 이 이벤트에 훅을 걸면 루프의 핵심 로직은 그대로 둔 채 관찰과 개입을 덧붙일 수 있습니다.

## 루프 생명주기와 정지 이유

에이전트 루프는 진입점과 종료점이 분명합니다. 시작은 단순합니다. 앞서 본 초기화(도구 등록, 대화 관리자 준비)에 더해 지표(metrics) 수집도 준비한 뒤 첫 반복을 시작합니다. 종료 쪽은 조금 더 복잡합니다. 루프가 끝나는 경로가 여러 갈래이고, 애플리케이션이 할 일은 어느 경로로 끝났느냐에 따라 달라집니다. 그 갈래를 알려 주는 값이 정지 이유입니다.

### 정지 이유는 "다음에 무엇을 할지"를 정합니다

모델 호출은 매번 정지 이유를 남기고, 루프는 이 값으로 분기합니다. `tool_use`만 루프를 이어 가고 나머지 값은 대부분 루프를 끝냅니다. 루프가 끝나면 대부분의 경우 마지막 정지 이유가 결과 객체에 실려 애플리케이션으로 돌아옵니다. Python에서는 `result.stop_reason`, TypeScript에서는 `result.stopReason`으로 읽습니다. 값의 이름은 표기법만 다릅니다. Python은 스네이크 케이스(`end_turn`)를, TypeScript는 카멜 케이스(`endTurn`)를 씁니다.

| 정지 이유 Python / TypeScript | 발생 상황 | 루프 동작 | 애플리케이션 조치 |
| --- | --- | --- | --- |
| `end_turn` / `endTurn` | 모델이 응답을 마치고 더 할 행동이 없음 | 정상 종료, 최종 메시지 반환 | 완료된 결과 반환 |
| `stop_sequence` / `stopSequence` | 설정한 정지 시퀀스를 만남 | 정상 종료 | 완료된 결과 반환 |
| `tool_use` / `toolUse` | 모델이 도구 하나 이상을 실행하려 함 | 도구 실행 후 모델 재호출 | 조치 불필요(루프 내부 처리) |
| `limit_turns` / `limitTurns`, `limit_total_tokens` / `limitTotalTokens`, `limit_output_tokens` / `limitOutputTokens`, `limit_structured_output_attempts` / `limitStructuredOutputAttempts` | 반복, 누적 토큰, 출력 토큰 예산 또는 구조화된 출력(structured output) 시도 횟수 소진 | 예산 소진에 따른 정상 종료 | 예산 소진 보고 또는 더 높은 한도로 재호출 |
| `cancelled` | 외부에서 실행을 멈춤 | 종료, 미완성 메시지 제외 | 호출자나 애플리케이션이 시작한 중단으로 처리 |
| `interrupt` | 사람의 입력이 필요함 | 일시 정지 | 요청된 입력을 모은 뒤 에이전트 재개 |
| `content_filtered` / `contentFiltered` | 안전 장치가 응답을 차단 | 종료 | 애플리케이션 정책 적용, 무조건 재시도 금지 |
| `guardrail_intervened` / `guardrailIntervened` | 가드레일 정책이 생성을 중단 | 종료 | 애플리케이션 정책 적용, 무조건 재시도 금지 |
| `max_tokens` / `maxTokens` | 응답이 토큰 한도에 걸려 잘림 | 예외 발생(결과 반환 안 함) | 반환값이 아닌 예외로 처리 |

표에서 먼저 볼 곳은 "루프 동작" 열입니다. 같은 종료라도 성격이 다릅니다. `end_turn`과 `stop_sequence`는 작업이 끝났다는 뜻입니다. 한도 계열은 작업이 끝나지 않았지만 정해 둔 예산 안에서 깔끔하게 멈췄다는 뜻이며, 실패가 아닙니다. 메시지 이력이 유효한 상태로 남아 있으므로 같은 에이전트를 더 높은 예산이나 다른 프롬프트로 다시 호출하면 됩니다.

`cancelled`는 의도된 중단입니다. 조립 중이던 미완성 모델 메시지는 결과와 이력에서 모두 빠집니다. 다만 스트리밍으로 이미 소비자에 전달된 델타는 그대로 남습니다. UI에 일부 텍스트가 먼저 표시되었다면 그 텍스트가 이력에는 없을 수 있으므로, 화면에 남긴 부분 응답과 이력이 어긋나지 않게 처리해야 합니다.

`content_filtered`와 `guardrail_intervened`는 둘 다 루프를 끝냅니다. 이 둘은 기계적으로 재시도하지 말고 애플리케이션 정책에 따라 의도적으로 처리해야 합니다. 사용자에게 안내 메시지를 보여 줄지, 감사 로그에 남길지 같은 규칙을 미리 정해 두는 편이 안전합니다. `interrupt`는 또 다릅니다. 인터럽트(interrupt)는 사람의 입력을 기다리며 루프를 잠시 멈춘 상태이고, 입력을 모아 넘기면 멈춘 지점에서 재개됩니다. 끝난 것이 아니라 보류된 것입니다.

### 반환되는 결과와 던져지는 예외를 구분하기

모든 종료가 정지 이유로 돌아오지는 않습니다. `max_tokens`는 정지 이유 목록에 있지만, 실제로는 루프가 결과를 반환하지 않고 예외를 던집니다(Python `MaxTokensReachedException`, TypeScript `MaxTokensError`). 모델은 잘린 응답에서 이어서 생성할 수 없으므로 현재 루프 안에서는 복구할 방법이 없습니다. 이름이 비슷한 `limit_output_tokens`와 혼동하기 쉽습니다. `limit_output_tokens`는 직접 정한 예산에 따른 정상 종료이고, `max_tokens`는 모델 응답 한 번이 공급자의 호출당 출력 상한을 넘어 잘린 상황입니다.

도구 본문에서 난 오류는 앞서 설명한 대로 오류 결과로 바뀌어 모델에 돌아가고 루프는 계속 돕니다. 반면 훅 실행 실패, 모델 재시도 소진, 잘못된 설정 같은 SDK 오류는 애플리케이션까지 그대로 전파됩니다. 그래서 호출 지점에는 반환된 정지 이유를 분기하는 코드와 던져진 예외를 받는 코드가 따로 있어야 합니다.

정지 이유를 애플리케이션 수준의 결과로 바꾸는 분류 함수는 다음처럼 작성합니다.

```python
from enum import Enum


class Outcome(str, Enum):
    COMPLETED = "completed"
    BUDGET_EXCEEDED = "budget_exceeded"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    STOPPED = "stopped"


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

이 함수는 정지 이유를 다섯 가지 결과로 묶습니다. 호출하는 쪽은 열 개가 넘는 문자열 대신 이 결과만 보고 분기하면 됩니다. 이 함수를 쓰는 요청 핸들러에서는 응답 텍스트를 `COMPLETED`일 때만 사용자에게 노출하고, 예산 소진이나 취소는 예외를 던지지 않고 타입이 정해진 결과로 돌려줍니다. 마지막의 `STOPPED`는 나머지 값을 모두 받아 내는 기본 분기라서, 정지 이유가 새로 추가되어도 함수가 깨지지 않습니다.

`interrupt`도 이 기본 분기로 갑니다. 예상 밖의 값이어서가 아니라, 이 함수가 요청마다 새 에이전트를 만드는 무상태(stateless) 핸들러에서 쓰이기 때문입니다. 그런 핸들러는 인터럽트를 재개하지 않습니다. 사람의 입력이 필요한 애플리케이션이라면 에이전트나 세션 상태와 반환된 인터럽트 ID를 보존하고, `interrupt`를 별도 분기로 떼어 재개 흐름으로 보내야 합니다.

TypeScript에서는 `result.stopReason`을 카멜 케이스 값과 비교하면 됩니다. TypeScript 버전의 같은 분류 함수는 여기에 더해 `refusal`도 `contentFiltered`, `guardrailIntervened`와 함께 차단으로 분류합니다.

## 호출 한도로 작업량 제한하기

모델 주도 루프는 몇 바퀴 돌지 미리 알 수 없고, 매 바퀴가 지금까지의 대화 전체를 다시 보냅니다. 그래서 장시간 실행되거나 자율적으로 움직이는 에이전트에 한도를 걸지 않으면 호출 하나가 예상보다 오래 돌거나 예산보다 많은 토큰을 쓸 수 있습니다. 기본으로 적용되는 한도는 없습니다. 직접 넘기지 않으면 에이전트는 모델이 도구 요청을 멈출 때까지 계속 돕니다. 이런 에이전트에는 반복 한도나 전체 토큰 한도, 또는 둘 다를 거는 것이 좋습니다.

둘을 함께 거는 데는 이유가 있습니다. 반복 한도는 모델이 도구를 끝없이 요청하는 폭주를 막지만, 도구 결과가 길면 몇 바퀴만 돌아도 토큰이 크게 쌓입니다. 반대로 전체 토큰 한도는 이런 누적을 막지만, 짧은 결과를 주고받으며 오래 도는 루프는 한참 뒤에야 잡아냅니다. 두 한도가 막는 실패 양상이 서로 달라서 함께 쓰는 편이 안전합니다.

한도는 호출할 때 `limits` 객체로 넘깁니다. 항목은 넷이고 모두 선택 사항입니다.

| 항목 Python / TypeScript | 제한 대상 | 소진 시 정지 이유 |
| --- | --- | --- |
| `turns` / `turns` | 루프 반복 횟수 | `limit_turns` / `limitTurns` |
| `total_tokens` / `totalTokens` | 모델 호출마다 보고된 전체 토큰의 누적합 | `limit_total_tokens` / `limitTotalTokens` |
| `output_tokens` / `outputTokens` | 출력 토큰 | `limit_output_tokens` / `limitOutputTokens` |
| `structured_output_attempts` / `structuredOutputAttempts` | 구조화된 출력 실패 시도 횟수 | `limit_structured_output_attempts` / `limitStructuredOutputAttempts` |

Python에서는 호출 인자로 딕셔너리를 넘깁니다. 같은 파라미터를 `invoke_async`와 `stream_async`에서도 쓸 수 있고, 각 값은 양의 정수여야 합니다.

```python
from strands import Agent

agent = Agent()

result = agent(
    "Summarize this document",
    limits={
        "turns": 5,
        "output_tokens": 2000,
        "total_tokens": 10000,
    },
)

if result.stop_reason == "limit_turns":
    print("Hit turn budget")
elif result.stop_reason == "limit_total_tokens":
    print("Hit token budget")
```

TypeScript에서는 `invoke`의 옵션 객체에 카멜 케이스 키로 넘깁니다. `stream`에서도 같은 옵션이 동작하며, 각 값은 양의 유한한 수여야 합니다.

```typescript
import { Agent } from '@strands-agents/sdk'

const agent = new Agent()

const result = await agent.invoke('Summarize this document', {
  limits: {
    turns: 5,
    outputTokens: 2000,
    totalTokens: 10000,
  },
})

if (result.stopReason === 'limitTurns') {
  console.log('Hit turn budget')
} else if (result.stopReason === 'limitTotalTokens') {
  console.log('Hit token budget')
}
```

### SDK는 반복이 시작될 때 한도를 검사합니다

한도 검사는 모델 호출 도중이 아니라 각 루프 반복의 맨 앞에서 일어납니다. 그래서 결과가 두 가지 생깁니다.

첫째, 한 번의 반복이 토큰 예산을 넘길 수 있습니다. `total_tokens`를 10,000으로 잡았더라도 누적이 9,000인 상태에서 시작한 반복이 3,000토큰을 쓰면 그 반복은 끝까지 진행되고, 다음 반복을 시작하기 전에야 한도에 걸립니다. 한도는 정확한 상한이 아니라 "이 선을 넘으면 다음 바퀴는 돌지 않는다"는 차단선입니다. 한 반복이 쓰는 양은 이력 길이와 응답 길이에 따라 달라지므로, 비용을 엄격하게 묶어야 한다면 그만큼 넉넉하게 여유를 두고 값을 정하세요.

둘째, 직전 반복이 요청한 도구는 한도 검사 전에 항상 끝까지 실행됩니다. 도구 요청에 결과가 없는 채로 루프가 멈추면 요청과 결과의 짝이 깨지므로, 루프는 결과를 이력에 넣은 뒤에 멈춥니다. 그래서 한도로 멈춘 뒤에도 메시지 이력은 다시 호출할 수 있는 상태로 남습니다. 반대로 오래 걸리는 도구를 한도로 끊을 수는 없습니다. 실행 중인 도구를 멈추려면 취소를 써야 하고, 도구가 실행되는 동안에는 도구 스스로 취소 신호를 확인해야 멈춥니다.

여러 한도가 같은 검사 시점에 동시에 걸리면 보고되는 정지 이유는 반복, 전체 토큰, 출력 토큰, 구조화된 출력 시도 순서로 우선순위를 따릅니다. 반복 한도와 전체 토큰 한도가 한꺼번에 소진되면 결과에는 `limit_turns`만 남습니다. 정지 이유는 원인 하나만 알려 주므로, 어떤 예산이 얼마나 쓰였는지 보려면 사용량 지표를 함께 확인하는 편이 낫습니다.

### 전체 토큰 한도는 비용이 아니라 토큰을 셉니다

전체 토큰 한도는 모델 호출마다 보고된 전체 토큰을 더합니다. Amazon Bedrock, Anthropic, OpenAI, Google Gemini에서는 이 전체 토큰에 프롬프트 캐시 읽기와 쓰기가 온전히 들어가므로, 캐시가 적중한 반복도 캐시되지 않은 반복만큼 한도를 소모합니다. 프롬프트 캐싱으로 재전송 비용은 줄어도 한도 계산에서는 할인이 없습니다. 반복마다 보내는 양 자체를 줄이려면 한도가 아니라 컨텍스트 관리를 손봐야 합니다.

### 적용 범위: 이번 호출, 그리고 SDK마다 다른 경계

한도는 현재 호출에만 적용됩니다. 같은 에이전트를 재사용하면 호출마다 카운터가 새로 시작합니다. 앞서 설명한 대로 한도로 멈춘 뒤에도 이력이 유효하게 남으므로, 더 큰 예산으로 이어 갈 수 있습니다.

```python
from strands import Agent

agent = Agent()

result = agent("Summarize this document", limits={"turns": 5})

if result.stop_reason == "limit_turns":
    # 카운터는 새로 시작하고, 이력은 이전 호출에서 이어집니다.
    result = agent("Continue the summary", limits={"turns": 10})
```

어떤 모델 호출이 한도에 포함되는지는 Python과 TypeScript가 다릅니다. Python은 이 경계를 토큰 한도 기준으로 정하고, TypeScript는 한도 전체가 요청 단위로 동작합니다. 하위 에이전트나 요약 같은 보조 호출을 쓰는 구성이라면 이 차이가 예산 설계에 직접 영향을 줍니다.

| 구분 | Python(토큰 한도) | TypeScript |
| --- | --- | --- |
| 토큰 한도의 기본 범위 | 이 에이전트 자신의 모델 호출만 셉니다 | 요청 전체를 덮습니다 |
| 도구로 넘긴 에이전트 | 한도 밖에서 실행되고 사용량도 더해지지 않습니다 | 요청의 한도와 사용량을 공유합니다 |
| 직접 작성한 도구가 호출하는 에이전트 | 그 호출에 `limits`를 넘겨야 제한됩니다 | 도구가 `context.invocationContext`를 넘길 때만 요청에 합류하고, 아니면 자체 한도로 실행됩니다 |
| 대화 관리자의 요약 호출 | 한도 밖에서 실행되고 사용량도 더해지지 않습니다 | 토큰이 사용량에 더해지지만 한도에 묶이지는 않습니다 |

Python에서 도구로 쓰는 에이전트에 예산을 걸려면, 그 에이전트를 직접 작성한 도구 안에서 호출하고 그 호출에 `limits`를 넘기면 됩니다. 그러면 상위 에이전트의 한도와 별개로 하위 에이전트의 작업량이 묶입니다. TypeScript에서는 하위 에이전트를 요청 전체 예산 안에 넣을지 따로 둘지를 `context.invocationContext` 전달 여부로 고릅니다.

구조화된 출력 시도 한도는 요청 전체를 덮어서, 같은 요청 안에서 도구로 실행되는 에이전트의 시도도 포함합니다. Python에서 도구로 쓰는 에이전트의 토큰 사용량이 한도 밖에 있는 것과 다릅니다.

## 취소: 실행 중인 에이전트 멈추기

호출 한도는 작업량을 미리 정해 두는 장치라서, 호출이 시작된 뒤에 생긴 사정까지는 반영하지 못합니다. 클라이언트가 연결을 끊었거나, 응답 기한이 지났거나, 사용자가 화면의 "중지" 버튼을 눌렀다면 남은 반복을 계속 돌릴 이유가 없습니다. 이럴 때 루프를 바깥에서 멈추는 수단이 취소입니다. 취소된 호출은 예외를 던지지 않고, 정지 이유가 `cancelled`인 결과를 반환합니다(Python `result.stop_reason == "cancelled"`, TypeScript `result.stopReason === 'cancelled'`).

가장 단순한 방법은 `agent.cancel()`을 부르는 것입니다. 이 메서드는 루프를 강제로 중단하지 않고 내부 신호 하나만 세웁니다. 루프는 정해진 체크포인트마다 이 신호를 확인하다가, 신호가 서 있으면 그 지점에서 멈춥니다. 호출이 끝나면 신호는 자동으로 지워지므로 같은 에이전트를 바로 다시 쓸 수 있습니다. `cancel()`은 여러 번 불러도 안전합니다. Python에서는 다른 스레드에서 불러도 안전합니다.

### 체크포인트: 취소가 실제로 관찰되는 지점

신호를 세우는 일과 루프가 멈추는 일 사이에는 간격이 있습니다. 루프는 아무 지점에서나 끊기지 않고, 대화 상태를 유효하게 유지할 수 있는 지점에서만 멈춥니다. 도구 요청에 대응하는 결과가 없으면 요청과 결과의 짝이 깨져 대화 상태가 유효하지 않게 됩니다. 그래서 아직 실행하지 않은 도구 호출도 그냥 버리지 않고, 건너뛰었다는 오류 결과를 넣어 짝을 맞춥니다. 체크포인트 구성은 두 SDK가 조금 다릅니다.

| 체크포인트 | Python | TypeScript | 취소가 관찰되면 |
| --- | --- | --- | --- |
| 각 루프 사이클의 시작 | 없음 | 있음 | 다음 모델 호출 전에 멈춥니다 |
| 모델 응답 스트리밍 중 | 있음 | 있음 | 부분 출력을 버립니다. 마지막 메타데이터 이벤트 전에 스트림이 닫히므로 사용량 지표가 부정확할 수 있습니다 |
| 도구 실행 전 | 있음 | 있음 | 대기 중인 도구 호출을 오류 결과와 함께 건너뜁니다 |
| 순차 도구 실행 사이 | 없음 | 있음 | 남은 도구 호출을 오류 결과와 함께 건너뜁니다 |
| MCP 도구 실행 중 | 있음 | 없음 | 공유 MCP 세션은 닫지 않고 진행 중인 MCP 요청만 취소합니다 |
| 도구 실행 후 | 있음 | 있음 | 다음 모델 호출 전에 멈춥니다 |

표를 보면 "도구 실행 중"에 해당하는 칸이 거의 비어 있습니다. 일반 도구가 실행되는 동안에는 SDK가 끼어들 체크포인트가 없습니다. Python의 MCP 도구만 예외로, 진행 중인 요청을 취소할 수 있습니다. 다만 원격 취소는 최선 노력(best-effort) 방식이라서, 서버가 취소를 확인하지 않아도 에이전트는 로컬에서 멈춥니다. MCP가 아닌 도구는 끝까지 실행된 뒤에야 "도구 실행 후" 체크포인트에서 취소가 관찰됩니다.

도구 실행기(tool executor)에 따라서도 차이가 생깁니다. 순차 실행에서는 아직 시작하지 않은 도구를 건너뛸 수 있습니다. 기본값인 동시 실행기에서는 형제 도구가 이미 함께 돌고 있을 수 있어서, TypeScript에서도 현재 배치의 형제 도구가 모두 끝난 뒤에야 취소가 관찰됩니다. TypeScript의 `stop` 내장 도구는 이 지점을 의도적으로 사용합니다. `cancel({ afterCurrentTools: true })`로 현재 도구들이 끝나기를 기다렸다가 다음 모델 호출 전에 멈춥니다.

취소된 결과에서 조립 중이던 미완성 모델 메시지가 빠지는 방식은 앞서 정지 이유를 다룰 때 설명한 그대로입니다. 스트리밍 중에 취소하면 사용량 지표까지 어긋날 수 있으므로, 취소된 호출의 토큰 수를 비용 집계에 그대로 쓰지 않는 편이 안전합니다.

### `agent.cancel()`과 호출별 취소 신호 고르기

취소를 거는 경로는 둘입니다. 에이전트 내부 신호를 세우는 `agent.cancel()`, 그리고 호출할 때 넘기는 외부 신호입니다. 고르는 기준은 누가 그 신호를 소유하느냐입니다. 취소가 특정 요청 하나에 속한다면 호출별 신호를 쓰세요. 실행 중인 에이전트는 다른 구성 요소가 가지고 있고 그 호출의 신호에는 접근할 수 없다면 `agent.cancel()`을 씁니다.

이 기준이 중요한 이유는 유휴 상태에서 동작이 다르기 때문입니다. Python에서 에이전트가 쉬고 있을 때 `agent.cancel()`을 부르면 다음 호출이 취소됩니다. 실행 중인 호출이 확실히 있을 때만 불러야 합니다. 기한 타이머가 호출이 끝난 직후에 발동하면 상관없는 다음 요청이 시작하자마자 멈추는 사고로 이어집니다. TypeScript에서는 유휴 상태의 `agent.cancel()`이 아무 일도 하지 않습니다. 호출별 신호는 그 호출에만 묶이므로 이런 문제가 생기지 않습니다.

Python에서는 `threading.Event`를 `cancel_signal` 파라미터로 넘깁니다. `__call__()`, `invoke_async()`, `stream_async()` 모두 받습니다. 에이전트는 내부 신호와 외부 이벤트를 함께 보므로 둘 중 어느 쪽이든 독립적으로 취소를 일으킵니다. 에이전트는 호출자의 이벤트를 세우거나 지우지 않습니다.

```python
import threading

from strands import Agent

agent = Agent()

cancel_signal = threading.Event()
deadline = threading.Timer(30.0, cancel_signal.set)
deadline.start()

try:
    result = agent(
        "Analyze this large dataset",
        cancel_signal=cancel_signal,
    )
finally:
    deadline.cancel()  # 호출이 먼저 끝나면 타이머를 정리합니다

if result.stop_reason == "cancelled":
    print("Agent was cancelled due to timeout")
```

같은 이벤트를 연결 끊김 처리기나 "중지" 버튼 처리기에 넘기면 사용자 주도 취소도 같은 경로로 처리됩니다. 이벤트 하나를 여러 에이전트에 공유하면 한 번에 모두 멈출 수도 있습니다.

외부 이벤트 방식에는 알아 둘 비용이 둘 있습니다. 첫째, 이미 세워진 이벤트로 호출하면 첫 체크포인트에서 취소되는데, 그 첫 체크포인트가 모델 스트리밍 안에 있습니다. 그래서 사용자 턴은 이력에 기록되고, 모델 요청이 한 번 나갔다가 중단될 수 있습니다. 둘째, 에이전트는 이벤트를 폴링으로 확인합니다. 지연이 최대 약 50ms이고 `as_tool()` 중첩 단계마다 누적되므로, 폴링 간격 안에 세웠다가 지운 이벤트는 놓칩니다. 이벤트는 호출과 호출 사이에서만 지우세요.

신호는 모델 공급자에도 전달되어, 응답이 계속 흘러가게 두는 대신 진행 중인 요청을 중단하게 합니다. 지원 여부는 공급자마다 다릅니다. Bedrock은 다음 청크 경계에서 스트리밍 응답을 닫지만, 아직 헤더를 돌려받지 못한 요청이나 비스트리밍(`streaming=False`) 호출은 중단하지 못합니다. 신호를 무시하는 공급자라도 SDK의 청크 사이 체크포인트에서는 취소됩니다.

TypeScript에서는 `AbortSignal`을 `invoke()`나 `stream()`의 `cancelSignal` 옵션으로 넘깁니다. 에이전트는 이 신호를 내부 컨트롤러와 `AbortSignal.any()`로 합성하므로, 여기서도 `agent.cancel()`과 외부 신호가 각각 독립적으로 취소를 일으킵니다. 선언적인 기한에는 `AbortSignal.timeout()`이, 직접 제어하려면 `AbortController`가 맞습니다.

```typescript
import { Agent } from '@strands-agents/sdk'

const agent = new Agent()

// 20초 기한을 넘기면 호출을 취소합니다
const timedResult = await agent.invoke('Analyze this large dataset', {
  cancelSignal: AbortSignal.timeout(20_000),
})

if (timedResult.stopReason === 'cancelled') {
  console.log('Agent hit its deadline')
}

// 어디서든 controller.abort()를 부르면 취소됩니다
const controller = new AbortController()
const controllerResult = await agent.invoke('Hello', {
  cancelSignal: controller.signal,
})
```

### 도구 안에서의 협력적 취소

SDK는 도구 호출 전과 사이에서 자동으로 취소를 확인합니다. 도구 본문이 시작되면 취소는 협력적(cooperative)으로 바뀝니다. 실행 중간에 반응할 수 있는 것은 도구 자신뿐입니다. 도구가 아무것도 하지 않으면 끝까지 실행되고, 에이전트는 도구가 반환된 뒤에 취소 처리를 이어 갑니다. 30초 기한을 걸어 두었더라도 10분짜리 일괄 처리 도구가 신호를 보지 않으면 실제로는 10분 뒤에 멈춥니다.

도구가 협력하는 방법은 두 가지입니다. 작업을 여러 단계로 나눠 단계 사이마다 신호를 확인하거나, 신호를 받는 API에 그대로 넘기는 것입니다. Python에서는 `@tool(context=True)`로 선언하고 `tool_context.cancel_signal.is_set()`을 확인합니다.

```python
from strands import tool
from strands.types.tools import ToolContext


@tool(context=True)
def normalize_records(
    records: list[str],
    tool_context: ToolContext,
) -> str:
    """Normalize records while checking for cancellation between items."""
    normalized = []
    for record in records:
        if tool_context.cancel_signal.is_set():
            return f"cancelled after {len(normalized)} records"
        normalized.append(record.strip().upper())
    return "\n".join(normalized)
```

이 예제의 도구는 신호를 확인하면 예외를 던지지 않고 "몇 건까지 처리했는지"를 담은 값을 반환합니다. 그러면 도구 요청에 대응하는 결과가 이력에 들어가고, 이어서 "도구 실행 후" 체크포인트에서 루프가 멈춥니다. 신호를 받는 API로는 `MCPClient.call_tool_async` 같은 것이 있습니다. `as_tool()`로 도구가 된 에이전트는 부모의 신호를 자동으로 받으므로, 부모를 취소하면 위임받은 에이전트도 함께 취소됩니다.

TypeScript에서는 콜백의 두 번째 인자인 `context`에서 `context.cancelSignal`을 꺼냅니다. `fetch`처럼 `AbortSignal`을 받는 API에는 그대로 넘기고, 직접 반복하는 작업이라면 단계 사이에서 `context.cancelSignal.aborted`를 확인합니다.

```typescript
import { tool } from '@strands-agents/sdk'
import { z } from 'zod'

const fetchResource = tool({
  name: 'fetch_resource',
  description: 'Fetch a large resource, respecting cancellation',
  inputSchema: z.object({ url: z.string() }),
  callback: async (input, context) => {
    // 에이전트가 취소되면 요청도 중단되도록 신호를 넘깁니다
    const response = await fetch(input.url, { signal: context?.cancelSignal })
    return response.text()
  },
})
```

> **핵심:** 취소의 응답성은 가장 오래 걸리는 협력하지 않는 도구가 정합니다. 기한을 걸었다면 오래 걸리는 도구가 신호를 확인하거나 전달하는지부터 점검하세요. 그렇지 않은 도구 하나가 있으면 기한은 사실상 그 도구의 실행 시간만큼 늘어납니다.

### 취소와 인터럽트의 차이

두 기능 모두 루프를 도중에 세우지만 목적이 반대입니다. 취소는 에이전트를 완전히 멈추고, 멈춘 지점에서 재개하지 않습니다. 인터럽트는 사람의 입력을 받으려고 잠시 멈추고, 입력이 들어오면 멈춘 지점에서 다시 이어 갑니다. 정지 이유도 `cancelled`와 `interrupt`로 갈립니다.

| 구분 | 취소 | 인터럽트 |
| --- | --- | --- |
| 계기 | 호출자나 애플리케이션의 바깥 신호 | 사람의 입력이 필요한 상황 |
| 정지 이유 | `cancelled` | `interrupt` |
| 이후 흐름 | 재개 없는 종료 | 입력을 모아 넘긴 뒤 멈춘 지점에서 재개 |
| 애플리케이션이 보존할 것 | 없음 | 에이전트나 세션 상태와 인터럽트 ID |

실무에서는 판단 기준이 간단합니다. 작업 결과가 더 이상 필요 없으면 취소를 쓰고, 작업은 계속해야 하는데 사람의 승인이나 정보가 빠졌다면 인터럽트를 씁니다. 취소된 호출도 이력은 유효하게 남으므로 같은 에이전트에 새 프롬프트로 다시 요청할 수는 있습니다. 하지만 이는 새 호출을 시작하는 것이지, 멈춘 지점에서 재개하는 것이 아닙니다.

## 동시 호출과 멱등성

같은 에이전트 인스턴스에 요청이 두 개 이상 동시에 들어오는 일은 생각보다 흔합니다. 클라이언트가 응답을 기다리다 API 호출을 재시도하거나, 사용자가 폼을 두 번 제출하거나, 우연히 같은 에이전트를 공유하는 웹 요청 두 개가 겹치는 경우입니다. 에이전트는 호출이 진행되는 동안 대화 이력을 계속 고쳐 씁니다. 두 호출이 같은 이력에 동시에 메시지를 붙이면 사용자 메시지, 도구 요청, 도구 결과가 뒤섞여 이력이 망가집니다. 그래서 SDK는 기본적으로 한 인스턴스에서 한 번에 한 호출만 처리하고, 겹치는 호출은 거부합니다.

### 단일 호출 가드

Python에서는 생성자 파라미터 `concurrent_invocation_mode`가 이 동작을 정합니다. 값은 `ConcurrentInvocationMode` 열거형이며 `THROW`(기본값)와 `UNSAFE_REENTRANT` 중 하나입니다. 기본 `THROW` 모드에서는 이미 실행 중인 에이전트를 다시 호출하면 `ConcurrencyException`이 발생합니다. TypeScript도 겹치는 호출을 거부하며, 실행 중인 에이전트를 호출하면 `ConcurrentInvocationError`를 던집니다.

대부분의 애플리케이션에는 이 기본 동작이 맞습니다. 에이전트 인스턴스 하나는 대화 하나에 대응하므로, 호출이 겹치도록 허용하면 이력이 망가질 뿐입니다. 가드는 이력을 지켜 주는 대신, 겹친 호출을 호출자가 직접 처리하게 넘깁니다. 이 예외는 정지 이유로 돌아오는 결과가 아니라 던져지는 예외입니다. 따라서 요청 경계에서 따로 잡아 "이미 처리 중"과 같은 응답으로 바꿔 주는 편이 좋습니다.

### 멱등성 토큰으로 재시도 요청 합치기

겹친 호출 가운데 일부는 실제로는 같은 요청입니다. 아직 처리 중인 요청에 재시도가 도착했다면 오류를 돌려주는 것도, 같은 작업을 두 번 실행하는 것도 바람직하지 않습니다. 원래 호출의 결과를 그대로 돌려주는 쪽이 낫습니다. Python에서는 호출 메서드(`__call__`, `invoke_async`, `stream_async`) 어디에나 `idempotency_token`을 넘겨 논리적 요청을 식별합니다. 이 값이 멱등성 토큰(idempotency token)입니다.

동작 규칙은 다음과 같습니다.

- 같은 토큰의 호출이 이미 진행 중이면 새 호출은 원래 호출이 끝나기를 기다렸다가 같은 `AgentResult`를 반환합니다.
- 진행 중인 호출과 토큰이 다르면 새 호출은 `ConcurrencyException`을 냅니다. 중복 제거는 토큰이 일치할 때만 적용됩니다.
- 원래 호출이 결과 없이 끝나면 기다리던 호출들은 `IdempotencyAbortedError`를 냅니다. 실행 도중 취소된 경우가 대표적입니다.

토큰은 동등성으로 비교하므로 안정적인 식별자라면 무엇이든 쓸 수 있습니다. 주문 ID, 요청 UUID, 프롬프트 문자열 자체도 됩니다. 실무에서는 클라이언트가 재시도할 때도 바뀌지 않는 값을 고르는 것이 핵심입니다. 재시도마다 새 UUID를 만드는 클라이언트라면 토큰이 매번 달라져 중복 제거가 일어나지 않고, 원래 호출이 아직 실행 중일 때 들어온 두 번째 호출은 그냥 `ConcurrencyException`으로 끝납니다.

세 가지 결과를 하나의 처리 함수에서 나누면 다음과 같습니다.

```python
from strands import Agent
from strands.types.exceptions import ConcurrencyException, IdempotencyAbortedError

agent = Agent()


def process_order(order_id: str):
    try:
        # 같은 주문의 재시도는 진행 중인 원래 호출의 결과를 함께 받습니다
        return agent(
            f"Process order {order_id}",
            idempotency_token=f"order-{order_id}",
        )
    except IdempotencyAbortedError:
        # 원래 호출이 결과 없이 끝났습니다(예: 취소). 재시도 여부를 여기서 정합니다.
        ...
    except ConcurrencyException:
        # 토큰이 다른 호출이 이 인스턴스에서 실행 중입니다
        ...
```

중복 제거된 호출이 받는 것은 최종 결과뿐입니다. 원래 호출이 스트리밍한 이벤트는 받지 못합니다. 다만 중복 제거된 호출의 `callback_handler`는 최종 결과로 한 번 호출되므로, 출력을 소비하는 쪽이 빈손으로 남지는 않습니다. 스트리밍 UI에 토큰 단위 출력을 보여 주는 구성이라면, 재시도한 클라이언트는 진행 과정 없이 완성된 답만 한 번에 받게 됩니다.

`IdempotencyAbortedError`는 앞서 다룬 취소와 이어져 있습니다. 원래 호출이 취소되면 기다리던 호출에 돌려줄 결과가 없습니다. SDK는 이 예외를 던져 재시도 여부를 호출자가 정하게 합니다.

### 재진입 모드: 가드를 완전히 끄기

한 에이전트에서 여러 호출을 정말로 동시에 돌려야 한다면 생성할 때 모드를 바꿉니다.

```python
from strands import Agent
from strands.types.agent import ConcurrentInvocationMode

agent = Agent(concurrent_invocation_mode=ConcurrentInvocationMode.UNSAFE_REENTRANT)
```

이 모드는 단일 호출 가드를 통째로 없앱니다. 겹친 호출은 예외를 내지도 않고 중복 제거되지도 않으며, 멱등성 토큰도 무시됩니다. 대화 이력을 동시 변경에서 보호하는 장치가 하나도 남지 않으므로, 서로 끼어든 호출들이 이력을 망가뜨릴 수 있습니다. 이름에 `UNSAFE`가 붙은 이유입니다. 각 호출이 격리된 상태에서만 작업하거나 동기화를 직접 처리할 때에만 쓰세요. 독립적인 작업을 병렬로 돌리는 것이 목적이라면 에이전트 하나를 공유하기보다 작업마다 별도 에이전트를 두는 편이 안전합니다.

### TypeScript와의 차이

TypeScript SDK에는 동시성 모드 설정도, 멱등성 토큰을 이용한 재시도 중복 제거도 없습니다. 두 기능 모두 Python SDK에만 있습니다.

| 구분 | Python | TypeScript |
| --- | --- | --- |
| 겹친 호출의 기본 동작 | `ConcurrencyException` 발생 | `ConcurrentInvocationError` 발생 |
| 동시성 모드 설정 | `concurrent_invocation_mode`(`THROW`, `UNSAFE_REENTRANT`) | 없음 |
| 재시도 중복 제거 | `idempotency_token` | 없음 |
| 원래 호출이 결과 없이 끝났을 때 | 대기 중인 호출에 `IdempotencyAbortedError` 발생 | 해당 없음 |

그래서 TypeScript에서 재시도 중복 제거가 필요하면 에이전트 바깥, 즉 애플리케이션 계층에서 요청 식별자를 기준으로 처리해야 합니다. 병렬 처리도 마찬가지로 작업마다 별도 에이전트를 쓰는 것이 권장 방식입니다. Python에서도 같은 원칙이 통합니다. 가드와 멱등성 토큰은 인스턴스를 공유할 수밖에 없는 상황을 위한 안전장치이고, 동시성을 높이는 수단은 아닙니다.

## 훅과 이벤트 모델

루프 안의 특정 지점에서 로그를 남기거나, 도구 입력을 검증하거나, 결과를 고쳐 쓰려면 루프 안으로 들어가는 통로가 따로 있어야 합니다. Strands에서는 훅이 그 통로입니다. 훅은 코어 SDK(PyPI의 `strands-agents`, npm의 `@strands-agents/sdk`)에 포함되어 있고, 사용자 코드뿐 아니라 SDK의 내장 구성 요소도 같은 훅 시스템으로 에이전트 생명주기의 이벤트를 구독합니다.

먼저 용어 두 개를 구분합니다. 훅 이벤트는 생명주기의 한 지점이고, 훅 콜백(callback)은 그 이벤트가 발생할 때 실행되도록 등록한 함수입니다. 콜백은 등록한 이벤트 타입의 이벤트 객체를 하나 받습니다. 타입이 정해진 이 이벤트 객체에는 해당 단계의 데이터가 들어 있습니다. `BeforeInvocationEvent`에는 에이전트와 요청 정보가, `BeforeToolCallEvent`에는 호출될 도구와 파라미터가 있습니다. 이벤트 타입마다 구독자를 여러 개 둘 수 있으므로 로깅 훅과 검증 훅이 같은 이벤트를 서로 모른 채 함께 구독해도 됩니다.

### 훅을 등록하는 방법

Python에서 콜백 하나라면 `agent.add_hook()`에 함수만 넘깁니다. 이벤트 타입은 콜백의 타입 힌트에서 추론됩니다. 타입을 직접 지정하려면 에이전트의 훅 레지스트리인 `agent.hooks`에서 `add_callback(이벤트_타입, 콜백)`을 부릅니다. 첫 인자로는 이벤트 타입 하나, 이벤트 타입 리스트(중복은 무시), 또는 `None`을 받습니다. `None`이면 타입 힌트에서 추론하고, 힌트가 `A | B` 같은 유니언이면 두 이벤트에 모두 등록됩니다.

```python
from strands import Agent
from strands.hooks import (
    AfterModelCallEvent,
    AfterToolCallEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
)

agent = Agent()


# 타입 힌트에서 이벤트 타입을 추론합니다
def on_before_tool(event: BeforeToolCallEvent) -> None:
    print(f"tool requested: {event.tool_use['name']}")


agent.add_hook(on_before_tool)


# 이벤트 타입을 직접 지정합니다
def on_after_tool(event: AfterToolCallEvent) -> None:
    print(f"tool finished: {event.tool_use['name']}")


agent.hooks.add_callback(AfterToolCallEvent, on_after_tool)


# 유니언 타입 힌트: 모델 호출 전후 두 이벤트에 모두 등록됩니다
def on_model(event: BeforeModelCallEvent | AfterModelCallEvent) -> None:
    print(f"model event: {type(event).__name__}")


agent.hooks.add_callback(None, on_model)
```

타입을 추론할 수 없거나 이벤트 타입 리스트가 비어 있으면 `add_callback`이 `ValueError`를 냅니다. 콜백은 동기 함수여도 되고 `async` 함수여도 되지만, 생성자 끝에서 발생하는 `AgentInitializedEvent`에 비동기 콜백을 등록하면 `ValueError`가 납니다. 콜백은 값을 반환하지 않아야 하며, 콜백이 던진 예외는 호출자에게 그대로 전파됩니다. 훅의 버그가 묻히지 않는 대신, 관찰만 하려던 로깅 훅의 예외 하나가 에이전트 호출 전체를 실패시킬 수 있습니다. 관찰용 훅이라면 내부에서 예외를 처리해 두세요.

관련된 콜백 여러 개를 묶으려면 `HookProvider` 프로토콜을 구현합니다. `register_hooks(registry)` 메서드 안에서 `registry.add_callback()`을 부르면 되고, 이 객체는 생성자의 `hooks` 파라미터로 넘기거나 생성 후 `agent.hooks.add_hook()`으로 붙입니다. 모델 호출 전후 이벤트를 짝지어 소요 시간을 재는 훅은 다음과 같습니다.

```python
import time

from strands import Agent
from strands.hooks import (
    AfterModelCallEvent,
    BeforeModelCallEvent,
    HookProvider,
    HookRegistry,
)


class ModelTimer(HookProvider):
    def __init__(self) -> None:
        self._started = 0.0

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeModelCallEvent, self.start)
        registry.add_callback(AfterModelCallEvent, self.stop)

    def start(self, event: BeforeModelCallEvent) -> None:
        self._started = time.perf_counter()

    def stop(self, event: AfterModelCallEvent) -> None:
        elapsed = time.perf_counter() - self._started
        print(f"model call took {elapsed:.2f}s")


agent = Agent(hooks=[ModelTimer()])
```

훅을 설정이나 도구와 함께 묶어 배포하려면 플러그인(plugin)이 더 편합니다. Python에서는 `Plugin`을 상속하고 메서드에 `@hook` 데코레이터를 붙여 `Agent(plugins=[...])`로 넘깁니다. TypeScript SDK에는 `HookProvider` 인터페이스가 없습니다. 콜백 하나는 `agent.addHook(이벤트_타입, 콜백)`으로 등록하고, 여러 훅은 `Plugin`을 구현해 `initAgent(agent)` 안에서 `addHook`을 부르는 방식으로 묶습니다.

```typescript
import { Agent, AfterToolCallEvent, BeforeToolCallEvent } from '@strands-agents/sdk'
import type { LocalAgent, Plugin } from '@strands-agents/sdk'

class ToolLoggingPlugin implements Plugin {
  name = 'tool-logging'

  initAgent(agent: LocalAgent): void {
    agent.addHook(BeforeToolCallEvent, (event) => {
      console.log(`Calling: ${event.toolUse.name}`)
    })
    agent.addHook(AfterToolCallEvent, (event) => {
      console.log(`Completed: ${event.toolUse.name}`)
    })
  }
}

const agent = new Agent({ plugins: [new ToolLoggingPlugin()] })
```

Python의 단일 에이전트 이벤트에는 이벤트를 일으킨 에이전트 인스턴스가 `agent` 속성으로 들어 있습니다. 호출 상태(invocation state)는 호출할 때 넘긴 설정과 문맥 데이터를 루프 전체에 실어 나르며, Python에서는 `event.invocation_state`, TypeScript에서는 `event.invocationState`로 읽습니다. Python에서는 `agent("...", user_id="user123")`처럼 호출 인자로 넘기고, 데이터베이스 연결처럼 JSON으로 직렬화할 수 없는 객체도 넘길 수 있습니다. TypeScript에서는 `invoke()`의 `invocationState` 옵션으로 넘깁니다.

### 이벤트 생명주기

이벤트는 앞서 설명한 추론-실행-관찰 사이클의 경계마다 발생합니다. 도구를 쓰는 전형적인 호출 하나를 따라가면 순서는 다음과 같습니다.

1. 요청이 시작되면 `BeforeInvocationEvent`가 발생하고, 사용자 입력이 이력에 들어가면서 `MessageAddedEvent`가 발생합니다.
2. 각 반복은 `BeforeModelCallEvent`로 시작해 모델 호출이 끝나면 `AfterModelCallEvent`가 이어지고, 모델 메시지가 이력에 추가되면서 다시 `MessageAddedEvent`가 발생합니다.
3. 모델이 도구를 요청했다면 배치 전체를 `BeforeToolsEvent`와 `AfterToolsEvent`가 감싸고, 그 안에서 도구 호출마다 `BeforeToolCallEvent`와 `AfterToolCallEvent`가 짝을 이룹니다. 도구 결과가 이력에 들어가면 `MessageAddedEvent`가 발생하고 루프는 2단계로 돌아갑니다.
4. 도구 요청이 없으면 루프를 빠져나와 `AfterInvocationEvent`가 발생합니다.

여기에 생성자 끝에서 한 번 발생하는 `AgentInitializedEvent`가 더해집니다. 이벤트를 고를 때는 단위를 구분하세요. `BeforeToolsEvent`는 배치당 한 번, `BeforeToolCallEvent`는 도구 호출당 한 번 발생합니다. `MessageAddedEvent`는 대화 이력에 메시지가 추가될 때마다 발생하므로 추가 내역을 감사 로그로 남기기에 알맞습니다. `AfterInvocationEvent`는 요청의 성공·실패와 무관하게 항상 발생하므로 자원 정리를 맡기기 좋습니다.

| 이벤트 | 발생 시점 | 역순 호출 | 수정 가능한 속성 |
| --- | --- | --- | --- |
| `AgentInitializedEvent` | 생성자 끝의 초기화 완료 시점 | 아니요 | 없음 |
| `BeforeInvocationEvent` | 새 호출 요청의 시작 | 아니요 | `cancel`(TypeScript) |
| `MessageAddedEvent` | 대화 이력에 메시지가 추가될 때 | 아니요 | 없음 |
| `BeforeModelCallEvent` | 추론용 모델 호출 직전 | 아니요 | `cancel`(TypeScript) |
| `AfterModelCallEvent` | 모델 호출 완료 직후 | 예 | `retry` |
| `BeforeToolsEvent` | 도구 배치 실행 직전 | 아니요 | `cancel` |
| `BeforeToolCallEvent` | 개별 도구 호출 직전 | 아니요 | Python `cancel_tool`, `selected_tool`, `tool_use` / TypeScript `cancel`, `selectedTool`, `toolUse` |
| `AfterToolCallEvent` | 개별 도구 호출 완료 직후 | 예 | `result`, `retry` |
| `AfterToolsEvent` | 도구 배치 실행 완료 직후 | 예 | Python `end_turn` / TypeScript `endTurn` |
| `AfterInvocationEvent` | 성공·실패와 무관한 요청 종료 시점 | 예 | `resume` |

TypeScript에는 스트리밍 델타, 완성된 콘텐츠 블록, 도구 결과, 최종 결과를 감싸는 이벤트와, 답변되지 않은 인터럽트마다 발생하는 `InterruptEvent`가 더 있습니다. 이 이벤트들을 포함해 TypeScript의 모든 이벤트는 `HookableEvent`를 확장하므로 훅으로 구독할 수도, `agent.stream()`으로 받을 수도 있습니다. 멀티 에이전트 오케스트레이터에는 `MultiAgentInitializedEvent`, `BeforeMultiAgentInvocationEvent`, `AfterMultiAgentInvocationEvent`, 그리고 노드마다 반복되는 `BeforeNodeCallEvent`와 `AfterNodeCallEvent`로 이루어진 별도 생명주기가 있습니다.

### 수정 가능한 속성: 관찰에서 개입으로

대부분의 이벤트 속성은 의도하지 않은 수정을 막도록 읽기 전용입니다. Python 이벤트 클래스는 허용 목록에 없는 속성에 값을 쓰면 `AttributeError`를 냅니다. 표의 "수정 가능한 속성" 열에 있는 속성에 값을 쓰는 순간 훅은 관찰자에서 개입자로 바뀝니다. 개입 방식은 넷입니다.

- 중단: `BeforeToolCallEvent`의 `cancel_tool`(TypeScript `cancel`)은 도구 실행을 메시지와 함께 취소하고, `BeforeToolsEvent`의 `cancel`은 배치 전체를 취소합니다. TypeScript에서는 호출 전체와 모델 호출도 각각 `BeforeInvocationEvent.cancel`, `BeforeModelCallEvent.cancel`로 취소합니다.
- 교체: `selected_tool`은 실행할 도구 자체를, `tool_use`는 실행 전 파라미터를 바꿉니다. TypeScript의 `toolUse`에서는 `name`, `toolUseId`, `input`을 다시 쓸 수 있고, `selectedTool` 없이 `name`만 바꾸면 레지스트리에서 도구를 다시 찾습니다. `AfterToolCallEvent`의 `result`는 모델에 전달되기 전의 도구 결과를 바꿉니다.
- 반복: `retry`는 모델 호출이나 도구 호출을 다시 하도록 요청하고, `resume`은 새 입력으로 후속 호출을 일으킵니다.
- 종료: `AfterToolsEvent`의 `end_turn`(TypeScript `endTurn`)은 도구 배치 뒤에 모델을 다시 부르지 않고 루프를 멈춥니다. `True`면 기본 최종 어시스턴트 메시지가, 문자열이면 그 문자열이, 콘텐츠 블록 리스트면 그 블록들이 최종 메시지가 되며, 결과의 정지 이유는 `end_turn`(TypeScript `endTurn`)입니다.

Python `AfterToolCallEvent`의 `exception`은 읽기 전용으로, 도구가 예외를 던졌다면 원래 예외를, 아니면 `None`을 담습니다. 도구 오류는 오류 결과로 바뀌어 모델에 돌아가므로, 훅은 이 속성으로 원래 예외의 타입을 확인한 뒤 모델에 재시도를 맡길지, 예외를 다시 던질지 정할 수 있습니다. TypeScript SDK에는 아직 없습니다.

쓰기 가능한 속성은 이벤트의 위치와 맞물려 있습니다. 도구를 바꾸거나 취소하는 속성은 실행 직전 이벤트에만, 결과를 고치거나 재시도하는 속성은 실행 직후 이벤트에만 있습니다. 원하는 개입이 있다면 그 개입이 의미를 갖는 가장 이른 이벤트를 먼저 찾으세요.

### 호출 순서 제어

같은 이벤트에 콜백이 여럿이면 순서가 결과를 바꿉니다. 입력을 고치는 훅과 입력을 기록하는 훅이 있다면, 어느 쪽이 먼저 도느냐에 따라 로그에 원래 값이 남을지 고친 값이 남을지가 갈립니다. 기본 규칙은 Before 계열 이벤트는 등록 순서대로, After 계열 이벤트는 등록의 역순으로 콜백을 부르는 것입니다.

역순인 이유는 정리 작업의 대칭성입니다. 중첩된 컨텍스트 관리자처럼 먼저 열린 것이 나중에 닫혀야 짝이 맞습니다. 훅 A가 Before에서 자원을 열고 훅 B가 그 위에 무언가를 얹었다면, After에서는 B가 먼저 걷어내고 A가 마지막에 닫아야 안전합니다.

순서를 직접 정하려면 `order`를 지정합니다. 값이 작을수록 먼저 실행됩니다. Python은 `add_hook`과 `add_callback`의 키워드 인자 `order`로, TypeScript는 `addHook`의 세 번째 인자 `{ order }`로 넘깁니다. SDK는 자체 훅의 위치를 나타내는 프리셋을 제공합니다.

| 프리셋 | 값 | 의미 |
| --- | --- | --- |
| `HookOrder.SDK_FIRST` | -100 | SDK의 가장 이른 훅이 도는 위치입니다 |
| `HookOrder.DEFAULT` | 0 | `order`를 지정하지 않았을 때 쓰이는 값입니다 |
| `HookOrder.SDK_LAST` | 100 | SDK의 가장 늦은 훅이 도는 위치입니다 |

이 값들은 강제되는 경계가 아니라서 어떤 숫자든 쓸 수 있습니다. SDK 훅보다 먼저 돌려야 하는 가드레일이라면 `HookOrder.SDK_FIRST - 1`처럼 프리셋 바깥 값을 쓰고, 무조건 맨 앞이나 맨 뒤에 두려면 Python은 `float('-inf')`와 `float('inf')`, TypeScript는 `-Infinity`와 `Infinity`를 씁니다.

```python
from strands import Agent
from strands.hooks import BeforeToolCallEvent, HookOrder

agent = Agent()


def guardrail(event: BeforeToolCallEvent) -> None:
    print("[guardrail] runs before the SDK's earliest hooks")


def validation(event: BeforeToolCallEvent) -> None:
    print("[validation] validating input")


def logging_hook(event: BeforeToolCallEvent) -> None:
    print(f"[logging] tool called: {event.tool_use['name']}")


agent.add_hook(logging_hook)  # HookOrder.DEFAULT (0)
agent.add_hook(guardrail, order=HookOrder.SDK_FIRST - 1)
agent.add_hook(validation, order=-50)
```

TypeScript도 같은 구성입니다.

```typescript
import { Agent, BeforeToolCallEvent, HookOrder } from '@strands-agents/sdk'

const agent = new Agent()

agent.addHook(BeforeToolCallEvent, (event) => {
  console.log('[logging] Tool called:', event.toolUse.name)
}) // HookOrder.DEFAULT (0)

agent.addHook(
  BeforeToolCallEvent,
  () => console.log('[guardrail] Runs before SDK hooks'),
  { order: HookOrder.SDK_FIRST - 1 },
)

agent.addHook(
  BeforeToolCallEvent,
  () => console.log('[validation] Validating input'),
  { order: -50 },
)
```

`order`와 역순 규칙이 함께 작동하는 방식에는 직관과 어긋나는 부분이 있습니다. 역순은 `order` 전체가 아니라 같은 `order` 그룹 안에서만 적용됩니다. 아래는 순서를 손으로 따라간 예시입니다(설명용 예시이며, 콜백 이름은 임의로 붙였습니다). `audit`(order 0), `guard`(order -101), `metrics`(order 0)을 이 순서대로 Before와 After 이벤트에 모두 등록했다고 가정합니다.

| 단계 | 정렬 기준 | 실행 순서 |
| --- | --- | --- |
| Before 이벤트 | `order` 오름차순, 그룹 안에서는 등록 순서 | `guard`, `audit`, `metrics` |
| After 이벤트 | `order` 오름차순, 그룹 안에서는 등록 역순 | `guard`, `metrics`, `audit` |

> **핵심:** After 이벤트에서도 `order`가 작은 콜백이 먼저 실행됩니다. 역순은 같은 `order` 그룹 안의 등록 순서에만 적용되므로, `order`를 다르게 준 훅 사이에서는 정리 작업의 대칭이 자동으로 생기지 않습니다. Before에서 가장 먼저 열고 After에서 가장 나중에 닫아야 하는 훅이라면, 두 콜백에 서로 반대 방향의 `order`를 따로 지정하세요.

SDK 내장 구성 요소도 같은 레지스트리에서 돌기 때문에, 직접 고친 입력을 SDK 훅이 보게 할지, SDK가 처리한 뒤의 상태를 사용자 훅이 보게 할지는 프리셋을 기준으로 정한 `order` 하나로 결정됩니다.

## 훅 실전 패턴

실전에서 쓰는 훅은 어느 이벤트에 거는지, 어떤 속성을 쓰는지, 상태를 언제 초기화하는지에 따라 아래 표처럼 나뉩니다.

| 패턴 | 이벤트 | 쓰는 속성 | 상태 초기화 시점 |
| --- | --- | --- | --- |
| 도구 가로채기 | `BeforeToolCallEvent` | `selected_tool`, `tool_use` | 상태 없음 |
| 인자 고정 | `BeforeToolCallEvent` | `tool_use`의 입력 | 상태 없음 |
| 결과 수정 | `AfterToolCallEvent` | `result` | 상태 없음 |
| 호출 횟수 제한 | `BeforeInvocationEvent`, `BeforeToolCallEvent` | `cancel_tool` | 호출 시작 |
| 모델 호출 재시도 | `AfterModelCallEvent` | `retry` | 호출 시작·성공 직후(Python), TypeScript는 `attemptCount` 사용 |
| 예외 전파 | `AfterToolCallEvent` | `exception` 읽기 | 상태 없음 |
| 도구 호출 재시도 | `AfterToolCallEvent` | `retry` | 도구 호출 성공 직후 |
| 호출 재개 | `AfterInvocationEvent` | `resume` | 종료 조건으로 직접 관리 |
| 인터럽트 자동 처리 | `BeforeToolCallEvent`, `AfterInvocationEvent` | `interrupt()`, `resume` | 재개 직후 |

### 도구 가로채기와 인자 고정

모델이 위험한 도구를 골랐을 때 실행을 막는 대신 더 안전한 도구로 바꿔 실행하는 것이 도구 가로채기입니다. `BeforeToolCallEvent`에서 `selected_tool`에 다른 도구를 넣으면, 이번 호출에서는 레지스트리가 찾은 도구 대신 그 도구가 실행됩니다. 이때 `tool_use`의 이름도 함께 바꿔야 모델이 교체를 볼 수 있습니다. TypeScript에서는 `event.selectedTool`에 대체 도구를 넣고 `event.toolUse.name`에 그 도구의 `name`을 넣습니다.

```python
from strands.hooks import BeforeToolCallEvent, HookProvider, HookRegistry


class ToolInterceptor(HookProvider):
    def __init__(self, safe_alternative_tool) -> None:
        # safe_alternative_tool은 이미 정의된 도구 객체라고 가정합니다
        self.safe_alternative_tool = safe_alternative_tool

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeToolCallEvent, self.intercept_tool)

    def intercept_tool(self, event: BeforeToolCallEvent) -> None:
        if event.tool_use["name"] == "sensitive_tool":
            # 더 안전한 도구로 교체하고, 이름도 맞춰 모델이 교체를 보게 합니다
            event.selected_tool = self.safe_alternative_tool
            event.tool_use["name"] = "safe_tool"
```

도구는 그대로 두고 파라미터만 시스템 정책으로 덮어쓸 때는 `tool_use`의 입력 딕셔너리를 제자리에서 갱신합니다. 모델이 무엇을 적었든 지정한 값이 이깁니다.

```python
from typing import Any

from strands import Agent
from strands.hooks import BeforeToolCallEvent, HookProvider, HookRegistry


class ConstantToolArguments(HookProvider):
    """특정 도구의 특정 파라미터에 고정값을 씁니다."""

    def __init__(self, fixed_tool_arguments: dict[str, dict[str, Any]]):
        self._tools_to_fix = fixed_tool_arguments

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeToolCallEvent, self._fix_tool_arguments)

    def _fix_tool_arguments(self, event: BeforeToolCallEvent):
        if parameters_to_fix := self._tools_to_fix.get(event.tool_use["name"]):
            tool_input: dict[str, Any] = event.tool_use["input"]
            tool_input.update(parameters_to_fix)


# calculator는 이미 정의된 도구라고 가정합니다. 정밀도를 항상 1자리로 씁니다
fix_parameters = ConstantToolArguments({"calculator": {"precision": 1}})
agent = Agent(tools=[calculator], hooks=[fix_parameters])
result = agent("What is 2 / 3?")
```

TypeScript에서는 `Plugin`의 `initAgent()`에서 같은 이벤트에 걸고, `event.toolUse.input`에 `Object.assign`으로 고정값을 덮어씁니다. 입력을 기록하는 훅이 이 훅보다 먼저 돌면 로그에는 모델이 원래 적은 값이, 나중에 돌면 고정값이 남습니다. 감사 로그에 무엇을 남길지 먼저 정하고 그에 맞춰 `order`를 주세요.

### 결과 수정

`AfterToolCallEvent`의 `result`는 도구 결과가 모델에 전달되기 직전의 값이라서, 형식을 다듬거나 접두어를 붙이는 일을 여기서 합니다. Python에서는 결과 딕셔너리의 콘텐츠를 제자리에서 바꿉니다. 이벤트 속성을 수정할 때는 디버깅과 감사를 위해 변경 내용을 기록하는 것이 권장 방식입니다.

```python
import logging

from strands.hooks import AfterToolCallEvent, HookProvider, HookRegistry

logger = logging.getLogger(__name__)


class ResultProcessor(HookProvider):
    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(AfterToolCallEvent, self.process_result)

    def process_result(self, event: AfterToolCallEvent) -> None:
        if event.tool_use["name"] == "calculator":
            original_content = event.result["content"][0]["text"]
            logger.info(f"Modifying calculator result: {original_content}")
            event.result["content"][0]["text"] = f"Result: {original_content}"
```

TypeScript에서는 결과를 제자리에서 고치지 않고 새 `ToolResultBlock`을 만들어 `event.result`에 넣습니다. `toolUseId`, `status`, 그리고 있다면 `error`를 원래 결과에서 그대로 옮겨야 요청과 결과의 짝이 유지됩니다.

```typescript
import {
  AfterToolCallEvent,
  ToolResultBlock,
  TextBlock,
  type LocalAgent,
  type Plugin,
} from '@strands-agents/sdk'

class ResultProcessor implements Plugin {
  name = 'result-processor'

  initAgent(agent: LocalAgent): void {
    agent.addHook(AfterToolCallEvent, (event) => this.processResult(event))
  }

  private processResult(event: AfterToolCallEvent): void {
    if (event.toolUse.name !== 'calculator') return

    const original = event.result.content.find((block) => block.type === 'textBlock')
    if (original?.type !== 'textBlock') return

    // 수정 전에 기록해 두어야 감사 기록에 두 상태가 모두 남습니다
    console.log(`Modifying calculator result: ${original.text}`)
    event.result = new ToolResultBlock({
      toolUseId: event.result.toolUseId,
      status: event.result.status,
      content: event.result.content.map((block) =>
        block.type === 'textBlock' ? new TextBlock(`Result: ${block.text}`) : block
      ),
      ...(event.result.error !== undefined ? { error: event.result.error } : {}),
    })
  }
}
```

### 호출 횟수 제한: 상태를 가진 훅

앞의 훅들은 상태가 없지만, 도구별 호출 횟수를 제한하는 훅은 몇 번 불렸는지 세고 그 수를 언제 0으로 돌릴지도 정해야 합니다. 폭주하는 도구 사용을 막거나 사용량 할당을 걸 때 씁니다.

```python
from threading import Lock

from strands import Agent
from strands.hooks import (
    BeforeInvocationEvent,
    BeforeToolCallEvent,
    HookProvider,
    HookRegistry,
)


class LimitToolCounts(HookProvider):
    """에이전트 호출 한 번 안에서 도구별 호출 횟수를 제한합니다."""

    def __init__(self, max_tool_counts: dict[str, int]):
        self.max_tool_counts = max_tool_counts
        self.tool_counts = {}
        self._lock = Lock()

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeInvocationEvent, self.reset_counts)
        registry.add_callback(BeforeToolCallEvent, self.intercept_tool)

    def reset_counts(self, event: BeforeInvocationEvent) -> None:
        with self._lock:
            self.tool_counts = {}

    def intercept_tool(self, event: BeforeToolCallEvent) -> None:
        tool_name = event.tool_use["name"]
        with self._lock:
            max_tool_count = self.max_tool_counts.get(tool_name)
            tool_count = self.tool_counts.get(tool_name, 0) + 1
            self.tool_counts[tool_name] = tool_count

        if max_tool_count and tool_count > max_tool_count:
            event.cancel_tool = (
                f"Tool '{tool_name}' has been invoked too many and is now being throttled. "
                f"DO NOT CALL THIS TOOL ANYMORE "
            )


# sleep은 이미 정의된 도구라고 가정합니다
limit_hook = LimitToolCounts(max_tool_counts={"sleep": 3})
agent = Agent(tools=[sleep], hooks=[limit_hook])

agent("Sleep 5 times for 10ms each or until you can't anymore")
agent("Sleep once")  # 호출마다 카운터가 초기화되므로 다시 성공합니다
```

카운터를 `BeforeInvocationEvent`에서 비우므로 할당량은 에이전트 인스턴스 전체가 아니라 호출 한 번에 걸립니다. Python 버전은 동시 실행 환경에서도 카운터가 안전하도록 갱신을 `Lock`으로 감쌉니다. 한도를 넘긴 호출은 예외가 아니라 `cancel_tool` 메시지로 취소되고, 이 메시지는 결과 자리에 들어가 모델이 읽습니다. 그래서 대문자 "DO NOT CALL THIS TOOL ANYMORE"처럼 모델에 주는 지시로 씁니다.

첫 번째 호출의 흐름은 아래 표와 같습니다(설명용 추적이며, 모델이 `sleep`을 하나씩 요청한다고 가정합니다).

| `sleep` 요청 순번 | 카운터 | 판정 | 모델이 받는 결과 |
| --- | --- | --- | --- |
| 1 | 1 | 한도 이내 | 정상 결과 |
| 2 | 2 | 한도 이내 | 정상 결과 |
| 3 | 3 | 한도 이내 | 정상 결과 |
| 4 | 4 | 한도 초과 | 취소 메시지 |

TypeScript 버전은 같은 구조를 `Plugin`으로 작성하고 `event.cancel`에 메시지를 넣습니다. 배치 전체를 한꺼번에 막으려면 `BeforeToolsEvent`의 `cancel`을 씁니다.

### 모델 호출 재시도

`AfterModelCallEvent.retry`로 훅이 원하는 기준에 따라 모델 호출 재시도를 요청합니다. 예외뿐 아니라 응답 검증이나 내용 품질 검사도 기준이 됩니다. 다음 예는 `ServiceUnavailable` 오류가 나면 지수 백오프(backoff)로 재시도합니다.

```python
import asyncio
import logging

from strands import Agent
from strands.hooks import (
    AfterModelCallEvent,
    BeforeInvocationEvent,
    HookProvider,
    HookRegistry,
)

logger = logging.getLogger(__name__)


class RetryOnServiceUnavailable(HookProvider):
    """ServiceUnavailable 오류가 나면 모델 호출을 재시도합니다."""

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
                    await asyncio.sleep(2 ** self.retry_count)  # 지수 백오프
        else:
            # 성공하면 카운터를 비웁니다
            self.reset_counts()


agent = Agent(hooks=[RetryOnServiceUnavailable(max_retries=3)])
result = agent("What is the capital of France?")
```

Python 버전은 카운터를 호출 시작과 모델 호출 성공 직후, 두 곳에서 비웁니다. 성공 시점에 비우지 않으면 앞 반복의 실패로 쌓인 횟수가 뒤 반복의 재시도 예산을 줄입니다. 콜백을 `async`로 두어 `asyncio.sleep`으로 기다리는 동안 이벤트 루프를 막지 않습니다.

TypeScript에서는 수동 카운터가 필요 없습니다. `event.attemptCount`가 방금 실패한 시도를 포함해 1부터 세므로, 오류 확인과 번호 비교만으로 상한이 정해집니다. 오류는 `event.error`에 들어 있습니다.

```typescript
import { AfterModelCallEvent, type LocalAgent, type Plugin } from '@strands-agents/sdk'

class RetryOnServiceUnavailable implements Plugin {
  name = 'retry-on-service-unavailable'

  constructor(private readonly maxRetries = 3) {}

  initAgent(agent: LocalAgent): void {
    agent.addHook(AfterModelCallEvent, (event) => this.handleRetry(event))
  }

  private handleRetry(event: AfterModelCallEvent): void {
    // attemptCount는 1부터 시작하고 방금 실패한 시도를 포함합니다
    if (
      event.error !== undefined &&
      event.error.message.includes('ServiceUnavailable') &&
      event.attemptCount <= this.maxRetries
    ) {
      event.retry = true
    }
  }
}
```

### 예외를 모델에 맡길지, 바로 실패시킬지

도구 예외는 오류 결과로 바뀌어 모델에 돌아가고, 모델은 접근을 바꿔 다시 시도합니다. 검증 실패 같은 예상한 오류에는 맞는 동작입니다. 하지만 단언문 실패, 설정 오류, 버그 같은 예상 밖 오류는 모델이 다시 시도해도 고칠 수 없으므로 즉시 실패시키는 편이 낫습니다. `AfterToolCallEvent.exception`으로 원래 예외의 타입을 보고 허용 목록에 없는 예외만 다시 던지면, 재시도할 필요가 없는 예외만 골라 전파합니다.

```python
from strands import Agent
from strands.hooks import AfterToolCallEvent, HookProvider, HookRegistry


class PropagateUnexpectedExceptions(HookProvider):
    """예상 밖 예외는 모델에 돌려주지 않고 다시 던집니다."""

    def __init__(self, allowed_exceptions: tuple[type[Exception], ...] = (ValueError,)):
        self.allowed_exceptions = allowed_exceptions

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(AfterToolCallEvent, self._check_exception)

    def _check_exception(self, event: AfterToolCallEvent) -> None:
        if event.exception is None:
            return  # 도구가 성공했습니다
        if isinstance(event.exception, self.allowed_exceptions):
            return  # 모델이 재시도하게 둡니다
        raise event.exception  # 예상 밖 오류는 전파합니다


# model, my_tool, ValidationError는 이미 정의·import되어 있다고 가정합니다
agent = Agent(
    model=model,
    tools=[my_tool],
    hooks=[PropagateUnexpectedExceptions(allowed_exceptions=(ValueError, ValidationError))],
)
```

TypeScript SDK에는 이 기능이 아직 없습니다.

### 도구 호출 재시도

일시적 오류나 타임아웃, 불안정한 외부 서비스처럼 모델을 거치지 않고 도구를 그대로 다시 실행하고 싶을 때는 `AfterToolCallEvent.retry`를 `True`로 둡니다. 도구 실행기가 현재 결과를 버리고 같은 `tool_use_id`로 도구를 다시 호출하며, 대화 이력에는 최종 시도의 결과만 들어갑니다.

```python
import logging

from strands import Agent, tool
from strands.hooks import AfterToolCallEvent, HookProvider, HookRegistry

logger = logging.getLogger(__name__)


class RetryOnToolError(HookProvider):
    """오류로 끝난 도구 호출을 재시도합니다."""

    def __init__(self, max_retries: int = 1):
        self.max_retries = max_retries
        self._attempt_counts: dict[str, int] = {}

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(AfterToolCallEvent, self.handle_retry)

    def handle_retry(self, event: AfterToolCallEvent) -> None:
        tool_use_id = str(event.tool_use.get("toolUseId", ""))
        tool_name = event.tool_use.get("name", "unknown")

        # 시도 횟수를 tool_use_id별로 셉니다
        attempt = self._attempt_counts.get(tool_use_id, 0) + 1
        self._attempt_counts[tool_use_id] = attempt

        if event.result.get("status") == "error" and attempt <= self.max_retries:
            logger.info(f"Retrying tool '{tool_name}' (attempt {attempt}/{self.max_retries})")
            event.retry = True
        elif event.result.get("status") != "error":
            # 성공하면 추적을 정리합니다
            self._attempt_counts.pop(tool_use_id, None)


@tool
def flaky_api_call(query: str) -> str:
    """가끔 실패하는 외부 API를 호출합니다.

    Args:
        query: 보낼 질의
    """
    import random
    if random.random() < 0.5:
        raise RuntimeError("Service temporarily unavailable")
    return f"Result for: {query}"


agent = Agent(tools=[flaky_api_call], hooks=[RetryOnToolError(max_retries=1)])
result = agent("Look up the weather")
```

카운터의 키가 도구 이름이 아니라 `tool_use_id`라서, 한 배치에서 같은 도구가 두 번 요청되어도 각자 재시도 예산을 따로 씁니다. TypeScript에서는 `event.result.toolUseId`를 키로 쓰고 `event.error`가 있는지로 실패를 판단합니다.

스트리밍 소비자는 주의할 점이 있습니다. 버려진 시도의 `ToolStreamEvent`는 이미 호출자에게 전달된 뒤이고, 최종 시도의 `ToolResultEvent`만 대화 이력에 들어갑니다. 진행 상황을 화면에 그리는 UI라면 결과 이벤트 전의 진행 이벤트를 확정된 것으로 다루지 마세요.

### 호출 재개(resume)로 후속 호출 잇기

`AfterInvocationEvent.resume`에 유효한 에이전트 입력을 넣으면, 에이전트는 호출자에게 돌아가는 대신 그 입력으로 자신을 다시 호출합니다. 문자열, 콘텐츠 블록 리스트, 메시지 리스트, 인터럽트 응답이 모두 입력이 됩니다. 이전 결과를 바탕으로 이어 가는 자율 루프, 추가 문맥 주입, 한 번의 호출 안에서 끝나는 다단계 작업 흐름에 씁니다.

```python
from strands import Agent
from strands.hooks import AfterInvocationEvent

resume_count = 0


async def summarize_after_tools(event: AfterInvocationEvent):
    """한 번만 재개해 방금 한 일을 요약하게 합니다."""
    global resume_count
    if resume_count == 0 and event.result and event.result.stop_reason == "end_turn":
        resume_count += 1
        event.resume = "Now summarize what you just did in one sentence."


agent = Agent()
agent.add_hook(summarize_after_tools)

# 첫 요청을 처리한 뒤 자동으로 두 번째 호출을 해서 요약을 만듭니다
result = agent("Look up the weather in Seattle")
```

재개는 루프를 몇 바퀴 더 도는 것이 아니라 새 호출 주기를 처음부터 시작하며, `BeforeInvocationEvent`도 다시 발생합니다. 그래서 앞의 상태 있는 훅들도 영향을 받습니다. `LimitToolCounts`와 Python `RetryOnServiceUnavailable`은 이 이벤트에서 카운터를 비우므로, 재개할 때마다 할당량과 재시도 예산이 새로 채워집니다. 재개를 여러 번 잇는 구성에서 사용량 전체를 묶으려면 카운터를 바깥에서 관리하세요.

종료 조건도 SDK가 대신 지켜 주지 않습니다. 위 예제의 `resume_count`처럼 횟수를 세지 않으면 호출이 끝날 때마다 다시 재개되어 무한 루프가 됩니다. 여러 번 다듬는 패턴이라면 상한을 상수로 두고 비교합니다.

```python
from strands import Agent
from strands.hooks import AfterInvocationEvent

MAX_ITERATIONS = 3
iteration = 0


async def iterative_refinement(event: AfterInvocationEvent):
    """MAX_ITERATIONS번까지 다시 호출해 응답을 다듬습니다."""
    global iteration
    if iteration < MAX_ITERATIONS and event.result:
        iteration += 1
        event.resume = (
            f"Review your previous response and improve it. "
            f"Iteration {iteration} of {MAX_ITERATIONS}."
        )


agent = Agent()
agent.add_hook(iterative_refinement)
result = agent("Draft a haiku about programming")
```

예제의 전역 카운터는 0으로 돌아가지 않으므로 같은 에이전트에 두 번째 요청을 보내면 재개가 일어나지 않습니다. 실제로는 카운터를 훅 객체 안에 두고 바깥 요청 단위로 초기화하는 편이 안전합니다. TypeScript에서 `resume`을 설정하면 같은 호출 잠금 아래에서 에이전트 루프에 다시 들어갑니다.

### 인터럽트를 훅 안에서 처리하기

`resume`은 인터럽트 시스템과도 연결됩니다. 호출이 인터럽트로 끝났을 때 훅이 인터럽트 응답으로 재개하면, 인터럽트가 호출자에게 돌아가지 않습니다. 에이전트가 인터럽트 상태일 때는 `interruptResponse` 객체 리스트로 재개해야 하며, 일반 문자열을 넣으면 `TypeError`가 납니다.

```python
from strands import Agent, tool
from strands.hooks import AfterInvocationEvent, BeforeToolCallEvent


@tool
def send_email(to: str, body: str) -> str:
    """이메일을 보냅니다.

    Args:
        to: 받는 사람 주소
        body: 본문
    """
    return f"Email sent to {to}"


def require_approval(event: BeforeToolCallEvent):
    """이메일을 보내기 전에 승인을 요구합니다."""
    if event.tool_use["name"] == "send_email":
        event.interrupt("email_approval", reason="Approve this email?")


async def auto_approve(event: AfterInvocationEvent):
    """인터럽트된 도구 호출을 모두 승인합니다."""
    if event.result and event.result.stop_reason == "interrupt":
        responses = [
            {"interruptResponse": {"interruptId": intr.id, "response": "approved"}}
            for intr in event.result.interrupts
        ]
        event.resume = responses


agent = Agent(tools=[send_email])
agent.add_hook(require_approval)
agent.add_hook(auto_approve)

# 훅이 인터럽트를 처리하므로 호출자는 최종 결과를 바로 받습니다
result = agent("Send an email to alice@example.com saying hello")
```

Python에서는 결과의 `interrupts`에서 인터럽트 ID를 꺼냅니다. TypeScript에서는 `BeforeToolCallEvent`에서 `event.interrupt({ name, reason })`로 인터럽트를 겁니다. `InterruptEvent` 훅이 답변되지 않은 인터럽트를 목록에 모아 두면, `AfterInvocationEvent` 훅이 그 목록으로 응답 리스트를 만들어 `resume`에 넣은 뒤 목록을 비웁니다.

예제는 모든 요청을 무조건 승인하므로 그대로 쓰기보다는 판정 일부를 자동화하는 데 씁니다. 예를 들어 사내 도메인으로 가는 메일만 훅이 승인하고, 나머지는 `resume`을 설정하지 않아 인터럽트가 호출자에게 돌아가게 합니다.

### 멀티 에이전트 훅을 겹쳐 쓰기

Graph나 Swarm 같은 오케스트레이터에도 자체 훅 레지스트리가 있습니다. Python에서는 `orchestrator.hooks.add_callback(BeforeNodeCallEvent, ...)`으로 등록하거나 생성자의 `hooks`에 `HookProvider`를 넘깁니다. TypeScript에서는 `graph.addHook()`을 쓰거나 `MultiAgentPlugin`을 구현해 `initMultiAgent()` 안에서 등록합니다. 이벤트가 담는 정보는 SDK마다 이름이 다릅니다. Python은 오케스트레이터 인스턴스 `source`, 노드 수준 이벤트의 `node_id`, 오케스트레이터 호출을 따라 흐르는 `invocation_state`를 줍니다. TypeScript는 `orchestrator`, `nodeId`, 그리고 현재 호출의 `MultiAgentState`인 `state`를 주며, 이 상태의 `app` 필드에 애플리케이션 자체 상태를 둡니다.

오케스트레이터 훅은 개별 에이전트의 훅을 대체하지 않습니다. 오케스트레이터 안의 에이전트도 자기 훅을 그대로 가지므로 두 층이 함께 돕니다.

```python
from strands import Agent
from strands.hooks import BeforeNodeCallEvent, BeforeToolCallEvent, HookProvider, HookRegistry

# Graph와 도구 tool1, tool2는 이미 import·정의되어 있다고 가정합니다


class AgentLevelHook(HookProvider):
    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeToolCallEvent, self.log_tool_use)

    def log_tool_use(self, event: BeforeToolCallEvent) -> None:
        print(f"Agent tool call: {event.tool_use['name']}")


class OrchestratorLevelHook(HookProvider):
    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeNodeCallEvent, self.log_node_execution)

    def log_node_execution(self, event: BeforeNodeCallEvent) -> None:
        print(f"Orchestrator node execution: {event.node_id}")


agent1 = Agent(tools=[tool1], hooks=[AgentLevelHook()])
agent2 = Agent(tools=[tool2], hooks=[AgentLevelHook()])

orchestrator = Graph(
    agents={"agent1": agent1, "agent2": agent2},
    hooks=[OrchestratorLevelHook()],
)
```

역할은 단위로 나눕니다. 노드 전환, 오케스트레이션 흐름 디버깅처럼 "어느 에이전트가 언제 돌았는가"는 오케스트레이터 훅에 둡니다. 도구 가로채기, 인자 고정, 횟수 제한처럼 도구 호출 하나하나에 걸리는 정책은 에이전트 훅에 둡니다. 같은 정책을 모든 노드에 적용하려면 위 예처럼 같은 훅 클래스를 에이전트마다 붙이되, 상태 있는 훅은 카운터가 섞이지 않도록 에이전트마다 인스턴스를 따로 만드세요.

## 미들웨어: 모델·도구 호출 감싸기

훅은 생명주기 이벤트를 관찰하고 반응하는 데 쓰므로, 한 단계를 실행할지, 몇 번 실행할지 정하는 일에는 맞지 않습니다. 미들웨어(middleware)는 모델 호출이나 도구 실행 같은 단계를 직접 감쌉니다. 단계에 들어가는 입력이나 나오는 결과를 바꾸고, 단계 전체를 감싸서 재시도·캐시·경로 변경·생략을 직접 결정합니다. 생명주기 이벤트를 관찰하고 반응하려면 훅을 쓰고, 스로틀링 재시도나 캐시 응답, 사람의 승인 대기처럼 단계의 흐름을 통제하려면 미들웨어를 씁니다.

### 단계와 세 가지 국면

단계(stage)는 에이전트 실행 안의 가로채기 지점입니다. 단계마다 들어가는 컨텍스트와 돌아오는 결과가 있고, 핸들러는 둘 중 어느 쪽이든 바꿀 수 있습니다. 두 SDK 모두 같은 이름의 단계 두 개를 제공합니다.

| 단계 | 감싸는 대상 | 컨텍스트 | 결과 |
| --- | --- | --- | --- |
| `InvokeModelStage` | 모델 호출 한 번 | `InvokeModelContext` | `InvokeModelResult` |
| `ExecuteToolStage` | 도구 호출 한 번 | `ExecuteToolContext` | `ExecuteToolResult` |

각 단계에는 국면(phase)이 셋 있고, 핸들러를 어떤 순서로 등록했든 국면은 항상 같은 순서로 돕니다.

1. Input: 단계가 실행되기 전에 컨텍스트를 변환합니다.
2. Wrap: 컨텍스트와 `next` 함수를 받아 `next`를 언제, 몇 번 부를지 정합니다. 한 번도 부르지 않으면 단계가 단락(short-circuit)되고, 한 번 부르면 그대로 통과하며, 여러 번 부르면 재시도가 됩니다.
3. Output: 단계가 끝난 뒤 결과를 변환합니다. 핸들러는 단계의 결과 타입을 받아 같은 타입을 반환합니다.

전체 흐름은 Before 훅 → Input → Wrap의 `next` 앞부분 → 단계 실행 → Wrap의 `next` 뒷부분 → Output → After 훅입니다. 훅은 미들웨어 체인 바깥에서 체인 전체를 감쌉니다.

### 핸들러 등록하기

Python은 `agent.add_middleware()`, TypeScript는 `agent.addMiddleware()`로 등록합니다. 순수 변환이면 국면 하위 토큰(`InvokeModelStage.Input`, `InvokeModelStage.Output`)을, Wrap 핸들러면 단계 토큰 자체(`InvokeModelStage`)를 넘깁니다. 컨텍스트는 불변 객체로 다뤄서, Python에서는 `replace()`로, TypeScript에서는 객체 전개로 수정된 사본을 반환합니다.

```python
from strands import Agent
from strands.middleware import InvokeModelStage

agent = Agent()


# Input: 이번 모델 호출에만 다른 시스템 프롬프트를 씁니다
def be_concise(context):
    return context.replace(system_prompt="Be concise.")


# Output: 결과를 관찰한 뒤 그대로 돌려줍니다
def log_stop_reason(result):
    print(f"model stopped with reason {result.result.stop_reason}")
    return result


agent.add_middleware(InvokeModelStage.Input, be_concise)
agent.add_middleware(InvokeModelStage.Output, log_stop_reason)
```

바뀐 프롬프트는 이 모델 호출에만 적용되고 `agent.system_prompt`는 그대로라서, 다음 호출은 원래 프롬프트에서 다시 시작합니다.

Wrap 핸들러는 비동기 제너레이터로, 스트림 이벤트를 흘려보내고 단계 결과를 최종 값으로 넘깁니다. Python의 비동기 제너레이터는 값을 반환하지 못하므로 결과도 스트림 속 이벤트입니다. 모델 호출이면 `ModelStopReason`, 도구 호출이면 `ToolResultEvent`이고, 에이전트가 타입으로 골라내므로 핸들러는 그 앞뒤에 자기 이벤트를 더 내보내도 됩니다. TypeScript에서는 `yield* next(context)`가 돌려준 값을 `return`합니다.

```typescript
import { Agent, InvokeModelStage } from '@strands-agents/sdk'

const agent = new Agent()

const removeTiming = agent.addMiddleware(InvokeModelStage, async function* (context, next) {
  const start = Date.now()
  const result = yield* next(context)
  console.log(`model call took ${Date.now() - start}ms`)
  return result
})

// 등록이 돌려준 정리 함수를 부르면 핸들러가 제거됩니다
removeTiming()
```

### 재시도와 캐시: `next`를 부르는 횟수로 흐름 정하기

`next`를 다시 부르면 재시도입니다. 다음 핸들러는 스로틀링 예외가 나면 지수 백오프로 최대 세 번까지 모델 호출을 시도합니다.

```python
import asyncio

from strands import Agent
from strands.middleware import InvokeModelStage
from strands.types.exceptions import ModelThrottledException

agent = Agent()


async def retry_on_throttle(context, next_fn):
    for attempt in range(3):
        try:
            async for event in next_fn(context):
                yield event
            return
        except ModelThrottledException:
            if attempt == 2:
                raise
            await asyncio.sleep(2 ** attempt)


agent.add_middleware(InvokeModelStage, retry_on_throttle)
```

에이전트의 재시도 전략(retry strategy)은 기본값으로 이미 스로틀링된 모델 호출을 지수 백오프로 재시도합니다. 재시도 미들웨어를 쓴다면 전략을 꺼서 둘이 겹치지 않게 하세요. 전략은 미들웨어 체인 바깥에서 돌기 때문에 전략이 재시도할 때마다 미들웨어도 처음부터 다시 실행되어, 시도 횟수가 곱으로 늘어납니다.

`next`를 부르지 않고 결과를 내면 단락입니다. 응답 캐시는 적중하면 저장해 둔 결과를 바로 내주고, 실패할 때만 모델을 부릅니다. Python에서는 모델 호출의 결과인 `ModelStopReason` 이벤트를 저장했다가 다시 내보냅니다.

```python
import json

from strands import Agent
from strands.middleware import InvokeModelStage, ModelStopReason

agent = Agent()
response_cache = {}


async def cache_responses(context, next_fn):
    key = json.dumps(context.messages, default=str)
    if key in response_cache:
        yield response_cache[key]  # 적중: 모델을 부르지 않습니다
        return

    async for event in next_fn(context):
        if isinstance(event, ModelStopReason):
            response_cache[key] = event
        yield event


agent.add_middleware(InvokeModelStage, cache_responses)
```

TypeScript에서는 `Map`에 `InvokeModelResult`를 저장하고, 적중하면 `next` 없이 그 값을 `return`합니다. 이 예의 키는 메시지 목록만으로 만들기 때문에, 시스템 프롬프트나 모델을 바꾸는 Input 핸들러와 함께 쓴다면 키 구성을 따로 정해야 합니다.

### 호출 하나를 다른 모델로 보내기

컨텍스트의 모델은 호출 단위이고 교체할 수 있습니다. Input 핸들러에서 모델을 바꾸면 에이전트 상태를 건드리지 않고 그 호출만 다른 모델로 갑니다. 다음은 예상 입력 토큰이 큰 요청을 컨텍스트 윈도우가 더 큰 모델로 보내는 예입니다.

```typescript
import { Agent, InvokeModelStage, BedrockModel } from '@strands-agents/sdk'

const agent = new Agent()
const largeContextModel = new BedrockModel({
  modelId: 'global.anthropic.claude-sonnet-4-6',
})

agent.addMiddleware(InvokeModelStage.Input, (context) =>
  (context.projectedInputTokens ?? 0) > 50_000
    ? { ...context, model: largeContextModel }
    : context
)
```

Python에서는 `context.projected_input_tokens`를 확인하고 `context.replace(model=large_context_model)`을 반환합니다.

### 도구 실행을 승인으로 막기

`ExecuteToolStage`의 Wrap 핸들러는 컨텍스트의 `interrupt` 메서드로 실행을 멈추고 사람을 기다립니다. 처음 호출하면 에이전트가 인터럽트로 멈추고, 호출자가 응답과 함께 재개하면 같은 호출이 그 응답을 반환합니다. 거부하면 오류 결과로 단락하고, 승인하면 `next`로 통과시킵니다.

```python
from strands import Agent
from strands.middleware import ExecuteToolStage, ToolResultEvent

agent = Agent()


async def approval_gate(context, next_fn):
    approval = context.interrupt(
        "approve_tool", reason=f"Run {context.tool_use['name']}?"
    )
    if approval.response != "approved":
        yield ToolResultEvent({
            "toolUseId": context.tool_use["toolUseId"],
            "status": "error",
            "content": [{"text": "Denied by reviewer"}],
        })
        return
    async for event in next_fn(context):
        yield event


agent.add_middleware(ExecuteToolStage, approval_gate)
```

거부 결과에는 원래 요청의 `toolUseId`를 넣습니다. 모델은 거부를 오류 결과로 받으므로 다른 접근을 시도할 수 있습니다.

### 훅과 미들웨어의 관계

둘은 같은 단계에서 함께 동작할 수 있지만, 체인 바깥에 있는 훅과 체인 안의 미들웨어는 다르게 동작합니다.

| 상황 | 훅 | 미들웨어 |
| --- | --- | --- |
| 단락 | Before·After 훅이 그대로 발생합니다 | `next`를 부르지 않고 결과를 냅니다 |
| 재시도 | 가장 바깥 호출 기준으로 한 쌍만 봅니다 | 시도 하나하나를 봅니다 |
| 변경 대상 | 에이전트 상태를 바꿉니다 | 이번 호출의 컨텍스트만 바꿉니다 |

그래서 훅으로 호출 횟수를 세면 캐시 적중도 시도로 잡히고, 시도별 관찰이 필요하면 그 로직은 미들웨어에 두어야 합니다. 훅으로 호출 하나의 시스템 프롬프트를 바꾸려면 `agent.system_prompt`를 바꿨다가 되돌려야 하므로, 호출 하나에만 적용할 변경은 미들웨어가 안전합니다.

추적 스팬(trace span)은 미들웨어를 거친 뒤의 상태를 기록합니다. 모델 호출 스팬에는 Input 변환이 모두 반영된, 실제로 쓰인 메시지·시스템 프롬프트·모델이 남습니다. 모델 라우팅을 걸었다면 스팬의 모델 ID도 라우팅된 모델이므로, 추적 값이 에이전트 설정과 달라도 정상입니다.

## 재시도 전략

모델 공급자는 속도 제한, 서비스 일시 중단, 네트워크 타임아웃 같은 오류를 가끔 냅니다. 이런 오류마다 호출 전체를 실패시키면 짧은 장애에도 에이전트 작업이 처음부터 다시 시작됩니다. 그래서 에이전트는 별도 설정 없이도 스로틀링된 모델 응답을 지수 백오프로 자동 재시도합니다. 이 동작은 생성자의 `retry_strategy`(TypeScript `retryStrategy`) 파라미터로 바꿉니다.

### 기본 동작: 다섯 번 재시도, 4초부터 두 배씩

설정하지 않으면 에이전트는 스로틀링 오류를 최대 5번 재시도합니다. 첫 시도까지 합치면 6번입니다. 첫 재시도 전 대기는 4초이고 이후 매번 두 배가 됩니다.

| 시도 | 결과 | 다음 동작 |
| --- | --- | --- |
| 1 | 실패 | 4초 대기 |
| 2 | 실패 | 8초 대기 |
| 3 | 실패 | 16초 대기 |
| 4 | 실패 | 32초 대기 |
| 5 | 실패 | 64초 대기 |
| 6 | 실패 | 예외 발생 |

표의 대기는 지터 없는 기준값입니다. 이 값을 모두 더하면 마지막 예외까지 2분 넘게 걸립니다. TypeScript 기본 백오프는 `'full'` 지터를 쓰므로 실제 대기는 매번 달라집니다. 요청에 기한을 거는 서비스라면 이 대기와 기한을 함께 놓고 시도 횟수를 정하세요. 재시도를 모두 소진하면 오류는 정지 이유로 돌아오지 않고 예외로 애플리케이션까지 전파됩니다.

기본 전략이 재시도하는 대상은 모델 공급자가 속도 제한으로 던지는 스로틀링 오류, 즉 Python의 `ModelThrottledException`과 TypeScript의 `ModelThrottledError`뿐입니다. 다른 예외는 재시도 없이 바로 전파됩니다. 그래서 타임아웃 같은 다른 일시적 오류를 재시도하려면 뒤에 설명할 하위 클래스로 대상을 넓혀야 합니다.

### 파라미터 조정하기

Python에서는 `ModelRetryStrategy`를 설정해 생성자에 넘깁니다.

```python
from strands import Agent, ModelRetryStrategy

agent = Agent(
    retry_strategy=ModelRetryStrategy(
        max_attempts=3,   # 첫 시도를 포함한 전체 시도 횟수
        initial_delay=2,  # 첫 재시도 전 대기(초)
        max_delay=60,     # 백오프 대기의 상한(초)
    )
)
```

| 파라미터 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `max_attempts` | `int` | `6` | 첫 호출을 포함한 전체 시도 횟수입니다. `1`이면 재시도하지 않습니다. |
| `initial_delay` | `float` | `4` | 첫 재시도 전 대기 초입니다. 이후 재시도마다 두 배가 됩니다. |
| `max_delay` | `float` | `128` | 재시도 사이 최대 대기 초입니다. 지수 증가를 이 값에서 자릅니다. |

$$n$$번째 재시도 전 대기 $$d_n$$은 지수로 늘다가 `max_delay`에서 잘립니다.

$$
d_n = \min\left(\text{initial\_delay} \cdot 2^{\,n-1},\ \text{max\_delay}\right)
$$

`max_attempts`는 시도 횟수 전체라서 "재시도 3번"을 원하면 4를 넣어야 합니다. 이름만 보고 재시도 횟수로 읽으면 한 번씩 어긋납니다.

TypeScript에서 `ModelRetryStrategy`는 추상 기반 클래스이고, 설정 가능한 표준 전략은 `DefaultModelRetryStrategy`입니다.

| 파라미터 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `maxAttempts` | `number` | `6` | 첫 호출을 포함한 전체 시도 횟수입니다. `1` 이상이어야 합니다. |
| `backoff` | `BackoffStrategy` | `new ExponentialBackoff({ baseMs: 4000, maxMs: 240000 })` | 재시도 사이 대기를 계산합니다. |

### TypeScript 백오프 전략

TypeScript는 대기 계산을 `backoff`로 받는 `BackoffStrategy`에 맡깁니다. Python에는 별도의 백오프 인터페이스가 없고, 앞의 `initial_delay`와 `max_delay`로만 지수 백오프를 조정합니다. 내장 전략은 셋입니다.

- `ExponentialBackoff`(기본값): 대기가 $$\text{baseMs} \cdot \text{multiplier}^{\,\text{attempt}-1}$$로 늘고 `maxMs`에서 잘립니다.
- `LinearBackoff`: 대기가 $$\text{baseMs} \cdot \text{attempt}$$로 늘고 `maxMs`에서 잘립니다.
- `ConstantBackoff`: 매 재시도에 설정한 대기를 그대로 씁니다.

`ExponentialBackoff`와 `LinearBackoff`는 지터(jitter) 옵션 `jitter`를 받으며 값은 `'none'`, `'full'`, `'equal'`, `'decorrelated'` 중 하나입니다. 기본값 `'full'`은 동시에 실패한 클라이언트들을 대기 구간 전체에 흩어 놓아 같은 순간에 몰리는 재요청을 줄이므로, 바꿀 이유가 없다면 그대로 두세요.

```typescript
import { Agent, DefaultModelRetryStrategy, ExponentialBackoff } from '@strands-agents/sdk'

const agent = new Agent({
  retryStrategy: new DefaultModelRetryStrategy({
    maxAttempts: 4,
    backoff: new ExponentialBackoff({
      baseMs: 2_000,
      maxMs: 60_000,
      multiplier: 2,
      jitter: 'full',
    }),
  }),
})
```

### 재시도 끄기

자동 재시도를 완전히 끄려면 Python에서는 `retry_strategy=None`, TypeScript에서는 `retryStrategy: null`을 넘깁니다. 재시도 미들웨어처럼 다른 계층이 재시도를 맡을 때 끕니다.

```python
from strands import Agent

agent = Agent(retry_strategy=None)
```

```typescript
import { Agent } from '@strands-agents/sdk'

const agent = new Agent({ retryStrategy: null })
```

### 하위 클래스로 재시도 대상 바꾸기

재시도할 오류의 범위를 넓히거나 좁히려면 전략 클래스를 상속해 판정 메서드만 재정의합니다. Python에서는 `ModelRetryStrategy`의 `is_retryable`을 재정의하고, `super().is_retryable(exception)`으로 기본 판정(스로틀링만 재시도)을 살린 채 오류 타입을 더하거나 뺍니다. 나머지 재시도 정책은 다시 구현하지 않아도 됩니다.

```python
from strands import Agent, ModelRetryStrategy


class MyRetryStrategy(ModelRetryStrategy):
    def is_retryable(self, exception: Exception) -> bool:
        # 스로틀링에 더해 타임아웃도 재시도합니다
        return super().is_retryable(exception) or isinstance(exception, TimeoutError)


agent = Agent(
    retry_strategy=MyRetryStrategy(max_attempts=4, initial_delay=2, max_delay=60)
)
```

TypeScript에서는 `DefaultModelRetryStrategy`를 상속해 `isRetryable`을 재정의합니다.

```typescript
import { Agent, DefaultModelRetryStrategy } from '@strands-agents/sdk'

class TransientServiceError extends Error {
  readonly name = 'TransientServiceError'
}

// 기본 대상(스로틀링)에 직접 정의한 일시 오류를 더합니다
class WiderRetryStrategy extends DefaultModelRetryStrategy {
  protected override isRetryable(error: Error): boolean {
    return super.isRetryable(error) || error instanceof TransientServiceError
  }
}

const agent = new Agent({
  retryStrategy: new WiderRetryStrategy({ maxAttempts: 5 }),
})
```

시도 번호에 따라 판단을 바꾸거나 백오프 상태를 직접 들고 있어야 한다면, TypeScript에서는 추상 `ModelRetryStrategy`를 직접 상속해 `computeRetryDecision`을 구현합니다. 단락 처리, 턴 단위 상태 초기화, 시도 사이 대기는 기반 클래스가 맡습니다.

두 SDK 모두 전략 인스턴스가 턴 단위 상태를 들고 있어서 여러 에이전트가 하나를 공유하면 안 됩니다. 에이전트마다 새 인스턴스를 만들고, 요청마다 새 에이전트를 만드는 구성이라면 전략 객체도 그 자리에서 함께 만드세요.

### 전략과 훅 중 고르기

재시도 판정을 직접 하는 방법으로 전략 하위 클래스 외에 `AfterModelCallEvent` 콜백에서 `event.retry = true`를 설정하는 훅 방식도 있습니다. 두 방식은 재시도 정책을 누가 소유하느냐가 다릅니다.

| 구분 | 재시도 전략 하위 클래스 | `AfterModelCallEvent` 훅 |
| --- | --- | --- |
| 대기 | 백오프가 자동으로 적용됩니다. | 자동 대기가 없어 직접 기다려야 합니다. |
| 시도 횟수 | 전략 인스턴스가 턴 단위 상태로 관리합니다. | 직접 세거나 TypeScript `attemptCount`로 제한합니다. |
| 맞는 경우 | 판정 전체를 소유하면서 백오프와 상태 관리를 물려받을 때입니다. | 모델 오류를 한 번 보고 재시도만 요청할 때입니다. |

재시도할 오류 집합만 바꾸면 되는 일이라면 하위 클래스가 적은 코드로 나머지 재시도 정책을 물려받습니다. 모델 오류를 살펴보고 재시도만 요청하면 되는 경우라면 훅이 더 간단합니다.

## 대화 관리

에이전트 루프는 반복마다 모델 응답과 도구 결과를 이력에 더하므로, 대화가 길어지면 언젠가 모델의 토큰 한도에 닿습니다. 대화 관리자는 오래된 메시지를 잘라 내거나 요약해 컨텍스트 윈도우를 한도 안에 유지하고, 아직 필요한 메시지는 남겨 둡니다. 루프 안에서 돌기 때문에 메시지가 쌓이는 동안 별도 호출 없이 관리가 이루어집니다. 모든 관리자는 `ConversationManager` 인터페이스를 구현하고, 이력을 줄이는 전략만 서로 다릅니다.

### 세 가지 내장 관리자 고르기

관리자는 이력을 어떻게 줄이고 싶은지에 따라 고릅니다.

| 관리자 | 이력 처리 방식 | 맞는 경우 |
| --- | --- | --- |
| `SlidingWindowConversationManager` | 최근 메시지를 정해진 개수만큼 남깁니다(기본 관리자) | 최근 맥락만 있으면 되는 대부분의 에이전트입니다 |
| `SummarizingConversationManager` | 오래된 메시지를 요약으로 바꿉니다 | 앞선 턴의 내용을 버리면 안 되는 긴 작업입니다 |
| `NullConversationManager` | 이력을 바꾸지 않습니다 | 짧은 대화, 디버깅, 컨텍스트를 직접 관리할 때입니다 |

기본값을 끄는 방법은 착각하기 쉽습니다. `conversation_manager`(TypeScript `conversationManager`)를 생략하거나 `conversation_manager=None`(TypeScript `undefined`)을 넘기면 트리밍이 꺼지지 않고 기본 슬라이딩 윈도우가 선택됩니다. 이력 전체를 그대로 두려면 `NullConversationManager` 인스턴스를 직접 넘겨야 합니다.

```python
from strands import Agent
from strands.agent.conversation_manager import NullConversationManager

agent = Agent(conversation_manager=NullConversationManager())
```

```typescript
import { Agent, NullConversationManager } from '@strands-agents/sdk'

const agent = new Agent({
  conversationManager: new NullConversationManager(),
})
```

Null 관리자를 쓰면 이력이 한도를 넘었을 때 줄여 줄 장치가 없습니다. 컨텍스트 한도를 넘지 않을 짧은 대화라고 확신할 수 있을 때만 고르세요.

### 슬라이딩 윈도우: 최근 메시지만 남기기

슬라이딩 윈도우(sliding window) 관리자는 메시지 수가 한도를 넘으면 가장 오래된 메시지부터 창 밖으로 밀어냅니다. 메시지 개수만 지키면 끝날 것 같지만, 실제로는 세 가지 일을 더 합니다.

첫째, 미완성 메시지 정리입니다. 앞에서부터 잘라 내다 보면 도구 요청은 사라지고 결과만 남는 식으로 짝이 깨진 메시지가 생길 수 있는데, 관리자가 이런 불완전한 메시지 시퀀스를 제거해 대화 상태를 유효하게 유지합니다. 둘째, 오버플로 트리밍입니다. 메시지 개수와 별개로 요청이 컨텍스트 윈도우를 넘으면, 요청이 들어갈 때까지 가장 오래된 메시지부터 잘라 냅니다. 셋째, 도구 결과 축약입니다. 큰 도구 결과 하나가 윈도우의 상당 부분을 차지할 수 있어서, 메시지가 컨텍스트 윈도우 한도를 넘으면 도구 결과를 줄입니다.

도구 결과 축약은 `should_truncate_results`(TypeScript `shouldTruncateResults`)로 켜고 끄며 기본값은 켜짐입니다. 켜져 있으면 도구 결과가 담긴 메시지 가운데 가장 오래된 것부터 줄이므로 최근 맥락은 최대한 오래 남습니다. 텍스트는 앞부분과 뒷부분을 남기고 그 사이를 `<truncated chars="N"/>` 표시로 바꿉니다. 이미지, 비디오, 바이너리 문서, 지나치게 큰 JSON은 `[image: png, source: bytes, 12345 bytes]` 같은 타입별 자리 표시자로 바뀝니다. 도구 결과의 원래 `status`와 `error` 필드는 그대로 남아서, 모델은 축약된 결과가 실패였는지는 계속 알 수 있습니다. 축약을 끄면 결과 원본은 보존되지만 대신 과거 메시지가 더 많이 지워질 수 있습니다. 결과의 일부와 대화 앞부분 가운데 무엇을 잃을지 골라야 하므로, 오래전 도구 출력의 세부보다 대화 흐름이 중요하다면 기본값을 유지하세요.

```python
from strands import Agent
from strands.agent.conversation_manager import SlidingWindowConversationManager

conversation_manager = SlidingWindowConversationManager(
    window_size=20,               # 남길 최대 메시지 수
    should_truncate_results=True, # 메시지가 너무 크면 도구 결과를 줄입니다
)

agent = Agent(conversation_manager=conversation_manager)
```

```typescript
import { Agent, SlidingWindowConversationManager } from '@strands-agents/sdk'

const conversationManager = new SlidingWindowConversationManager({
  windowSize: 40,
  shouldTruncateResults: true,
})

const agent = new Agent({ conversationManager })
```

기본적으로 슬라이딩 윈도우는 에이전트 루프가 끝난 뒤에만 관리를 적용합니다. 그런데 한 호출 안에서 도구를 여러 번 부르는 에이전트는 호출이 끝나기 전에 이력이 불어납니다. Python에서는 `per_turn`으로 관리 시점을 루프 안으로 당길 수 있습니다. 기본값 `False`는 루프 종료 후에만, `True`는 모든 모델 호출 전에, 0보다 큰 정수 `N`은 모델 호출 `N`번마다 관리를 적용합니다. TypeScript는 이 파라미터를 지원하지 않습니다.

```python
from strands import Agent
from strands.agent.conversation_manager import SlidingWindowConversationManager

# 모델 호출 3번마다 관리를 적용합니다
agent = Agent(conversation_manager=SlidingWindowConversationManager(per_turn=3))
```

### 요약 관리자: 버리지 않고 압축하기

슬라이딩 윈도우는 밀려난 메시지의 내용을 잃습니다. 초반에 사용자가 말한 요구 사항이나 앞서 도구로 확인한 사실이 창 밖으로 나가면 모델이 그것을 다시 묻거나 다시 조회해야 할 수 있습니다. `SummarizingConversationManager`는 오래된 메시지를 요약으로 바꿔서 앞선 턴의 핵심을 컨텍스트에 남깁니다. 대신 요약 자체가 모델 호출이라서 지연과 비용이 더 들고, 요약 과정에서 세부가 빠질 수 있습니다.

동작을 정하는 파라미터는 두 SDK가 공유하는 것과 한쪽에만 있는 것으로 나뉩니다.

| 파라미터 Python / TypeScript | 기본값 | 의미 |
| --- | --- | --- |
| `summary_ratio` / `summaryRatio` | `0.3` | 줄일 때 요약해 대체할 가장 오래된 메시지의 비율입니다. 0.1과 0.8 사이로 제한됩니다. |
| `preserve_recent_messages` / `preserveRecentMessages` | `10` | 항상 남길 최근 메시지의 최소 개수입니다. |
| `summarization_system_prompt` / `summarizationSystemPrompt` | 없음 | 요약용 시스템 프롬프트입니다. 없으면 핵심 주제, 사용한 도구, 기술 정보를 3인칭 글머리표로 정리하는 기본 프롬프트를 씁니다. |
| `summarization_agent` / 없음 | 없음 | Python 전용입니다. 요약을 만들 별도 에이전트이며 `summarization_system_prompt`와 함께 쓸 수 없습니다. |
| 없음 / `model` | `agent.auxModel` | TypeScript 전용입니다. 요약에 쓸 모델을 바꿉니다. |

기본 요약은 Python에서는 `agent.aux_model`, TypeScript에서는 `agent.auxModel`이 맡습니다. 요약에는 더 빠르거나 저렴한 모델을 따로 쓸 수 있습니다. Python에서는 그런 모델로 만든 에이전트를 `summarization_agent`로 넘기는데, 이 에이전트는 자체 시스템 프롬프트, 도구, 모델 설정을 가지고 오므로 요약하면서 도구를 돌리거나 도메인에 맞춘 요약 로직을 적용할 수 있습니다. 시스템 프롬프트가 에이전트 안에 들어 있으니 `summarization_system_prompt`와 동시에 쓸 수 없습니다.

```python
from strands import Agent
from strands.agent.conversation_manager import SummarizingConversationManager
from strands.models import AnthropicModel

summarization_model = AnthropicModel(
    model_id="claude-haiku-4-5-20251001",
    max_tokens=1000,
    params={"temperature": 0.1},  # 일관된 요약을 위해 낮은 온도를 씁니다
)

conversation_manager = SummarizingConversationManager(
    summary_ratio=0.4,
    preserve_recent_messages=8,
    summarization_agent=Agent(model=summarization_model),
)

agent = Agent(conversation_manager=conversation_manager)
```

TypeScript에서는 도구나 훅, 플러그인을 갖춘 `Agent`를 요약용으로 주입하는 기능이 아직 없고, `model`로 모델만 바꿉니다.

```typescript
import { Agent, SummarizingConversationManager } from '@strands-agents/sdk'
import { AnthropicModel } from '@strands-agents/sdk/models/anthropic'

const summarizationModel = new AnthropicModel({
  modelId: 'claude-haiku-4-5-20251001',
  maxTokens: 1000,
  params: { temperature: 0.1 },
})

const agent = new Agent({
  conversationManager: new SummarizingConversationManager({
    model: summarizationModel,
    summaryRatio: 0.4,
    preserveRecentMessages: 8,
  }),
})
```

요약 관리자도 요약하는 동안 도구 요청과 결과 메시지의 짝을 함께 묶어 두고, 요약에 실패해도 안전하게 처리합니다. 코드 작업용 에이전트라면 함수 이름, 파일 경로, 설정 값을 보존하라고 지시하는 도메인 전용 요약 프롬프트를 주는 편이 좋습니다.

### 메시지 고정: 지워지면 안 되는 메시지 지키기

두 전략 모두 오래된 메시지부터 줄이기 때문에, 대화 첫머리에 둔 핵심 지시나 중요한 결정이 가장 먼저 사라집니다. 메시지 고정은 특정 메시지를 축출 대상에서 빼고, 고정된 메시지는 슬라이딩 윈도우 트리밍과 요약 양쪽에서 살아남습니다. 메시지 객체에 `metadata.custom.pinned = true`를 설정하면 고정됩니다.

가장 간단한 방법은 두 관리자가 받는 `pin_first`(TypeScript `pinFirst`)로 처음 N개 메시지를 영구히 보호하는 것입니다. 고정 메타데이터는 첫 번째 컨텍스트 축소 때 기록되고 이후 계속 유지되므로, 그 메시지들은 이후의 모든 축소에서 보호됩니다.

```python
from strands import Agent
from strands.agent.conversation_manager import SlidingWindowConversationManager

agent = Agent(
    conversation_manager=SlidingWindowConversationManager(
        window_size=40,
        pin_first=1,  # 첫 메시지를 항상 남깁니다
    )
)
```

```typescript
import { Agent, SlidingWindowConversationManager } from '@strands-agents/sdk'

const agent = new Agent({
  conversationManager: new SlidingWindowConversationManager({
    windowSize: 40,
    pinFirst: 1,
  }),
})
```

고정한 메시지는 줄일 수 없는 고정 비용이 됩니다. 많이 고정할수록 나머지 메시지가 쓸 자리가 줄어들어 축소가 더 자주, 더 크게 일어나므로 꼭 필요한 메시지만 고정하세요.

### 선제적 압축: 넘치기 전에 줄이기

기본 상태의 대화 관리자는 반응형입니다. 모델이 컨텍스트 윈도우 초과 오류로 요청을 거부한 뒤에야 이력을 줄입니다. 이 방식에는 비용이 둘 있습니다. 거부될 요청을 보내느라 왕복 한 번을 낭비하고, 거부되지 않더라도 입력이 윈도우를 거의 채우면 출력에 쓸 토큰이 모자라게 됩니다. 선제적 압축은 예상 입력 토큰이 모델 컨텍스트 윈도우의 일정 비율을 넘으면 모델 호출 전에 미리 줄입니다.

매 모델 호출 전에 에이전트는 예상 입력 토큰 수를 추정해 `BeforeModelCallEvent`에 붙이고, 관리자는 이 값을 모델의 `contextWindowLimit`과 비교합니다. 조건은 아래 식과 같습니다.

$$
\frac{\text{projectedInputTokens}}{\text{contextWindowLimit}} \ge \text{compressionThreshold} \;\Rightarrow\; \text{reduce()}
$$

내장 관리자 모두에 `proactive_compression`(TypeScript `proactiveCompression`)을 넘겨 켭니다. `True`는 기본 임계값 0.7을 쓰고, 딕셔너리나 객체로 0과 1 사이의 `compression_threshold`(TypeScript `compressionThreshold`)를 줄 수 있습니다. 설정하지 않으면 반응형 오버플로 복구만 동작합니다.

```python
from strands import Agent
from strands.agent.conversation_manager import (
    SlidingWindowConversationManager,
    SummarizingConversationManager,
)

sliding_agent = Agent(
    conversation_manager=SlidingWindowConversationManager(
        window_size=50,
        proactive_compression={"compression_threshold": 0.7},
    ),
)

summarizing_agent = Agent(
    conversation_manager=SummarizingConversationManager(proactive_compression=True),
)
```

```typescript
import { Agent, SummarizingConversationManager } from '@strands-agents/sdk'

const agent = new Agent({
  conversationManager: new SummarizingConversationManager({
    proactiveCompression: { compressionThreshold: 0.7 },
  }),
})
```

`BeforeModelCallEvent`는 도구 사용 사이클 안의 호출까지 포함해 모든 모델 호출 전에 발생하므로, 선제적 압축은 루프 안에서도 작동합니다. 다음은 컨텍스트 윈도우 10만 토큰, 임계값 0.7을 가정한 설명용 추적입니다(토큰 수는 임의로 정한 값입니다).

| 모델 호출 | 예상 입력 토큰 | 사용률 | 동작 |
| --- | --- | --- | --- |
| 1 | 8,000 | 0.08 | 그대로 호출합니다 |
| 2 | 31,000 | 0.31 | 그대로 호출합니다 |
| 3 | 55,000 | 0.55 | 그대로 호출합니다 |
| 4 | 74,000 | 0.74 | 호출 전에 `reduce()`로 줄인 뒤 호출합니다 |

호출 3과 4 사이에 큰 도구 결과가 들어오면서 임계값을 넘었고, 압축은 호출 4 직전에 일어납니다. 반응형 방식이었다면 호출 4는 줄이지 않은 이력으로 그대로 나갔을 것입니다.

선제적 압축은 반응형 복구와 같은 축소 로직을 쓰되 최선 노력 방식입니다. `reduce()`가 예외를 던지거나 `false`를 반환해도 오류는 삼켜지고 모델 호출은 평소대로 진행됩니다. 그러니 압축 실패가 호출을 막을 걱정은 없지만, 압축이 실제로 일어났다고 가정해서도 안 됩니다. 마지막 안전망은 여전히 반응형 오버플로 복구입니다.

### 커스텀 대화 관리자 작성하기

내장 전략이 맞지 않으면 직접 작성합니다. 두 SDK의 인터페이스 모양이 다르므로 나눠서 봅니다.

Python에서는 `ConversationManager` 인터페이스를 구현하며, 구성 요소는 다음과 같습니다.

- `apply_management`: 이벤트 루프 사이클이 끝날 때마다 불려, 도구 결과와 어시스턴트 응답이 더해진 메시지 배열에 관리 전략을 적용합니다.
- `reduce_context`: 컨텍스트 윈도우 초과 예외가 났을 때 불립니다. 재시도 전에 이력을 줄일 기회입니다.
- `removed_message_count`: 관리자가 에이전트 메시지에서 지운 사용자·모델 메시지 수입니다. 세션 관리가 저장소에서 메시지를 효율적으로 불러올 때 이 값을 씁니다. 요약처럼 관리자가 추가한 메시지는 세지 않습니다.
- `register_hooks`(선택): 모델 호출 전 트리밍 같은 선제적 관리를 하려면 재정의합니다. 재정의할 때는 항상 `super().register_hooks`를 불러야 합니다.

세션 관리가 `removed_message_count`로 메시지를 불러오므로, 메시지를 지울 때마다 이 값도 함께 늘려야 합니다.

TypeScript에서는 추상 기반 클래스 `ConversationManager`를 상속하고 `reduce` 하나를 구현합니다. 이 메서드는 두 경우에 불립니다. `ContextWindowOverflowError`가 났을 때 반응형으로 불리면 `options.error`가 설정되어 있고, 임계값을 넘은 모델 호출 전에 선제적으로 불리면 `options.error`가 `undefined`입니다. `agent.messages`를 제자리에서 고치고, 줄였다면 `true`를 반환합니다. 반환값의 의미는 경우마다 다릅니다. `error`가 있을 때 `false`를 반환하면 오류가 잡히지 않고 에이전트 루프 밖으로 전파되고, `error`가 없을 때는 `false`를 반환하거나 예외를 던져도 모델 호출은 그대로 진행됩니다.

```typescript
import {
  Agent,
  ConversationManager,
  type ConversationManagerReduceOptions,
} from '@strands-agents/sdk'

class Last10MessagesManager extends ConversationManager {
  readonly name = 'my:last-10-messages'

  reduce({ agent }: ConversationManagerReduceOptions): boolean {
    if (agent.messages.length <= 10) return false // 반응형 호출이면 오버플로 오류가 전파됩니다
    agent.messages.splice(0, agent.messages.length - 10)
    return true
  }
}

const agent = new Agent({
  conversationManager: new Last10MessagesManager(),
})
```

이 예는 개수만 보고 앞에서 잘라 내므로, 내장 슬라이딩 윈도우가 하는 미완성 메시지 정리나 메시지 고정 존중은 직접 구현해야 합니다. 잘린 경계에서 도구 요청과 결과의 짝이 깨지지 않는지 확인하세요.

호출이 끝날 때마다 트리밍하는 선제적 관리를 더하려면 `initAgent`를 재정의해 훅을 겁니다. 이때 `super.initAgent(agent)`를 반드시 불러야 내장 오버플로 복구와 선제적 압축 훅이 유지됩니다.

```typescript
import {
  ConversationManager,
  AfterInvocationEvent,
  type LocalAgent,
  type ConversationManagerReduceOptions,
} from '@strands-agents/sdk'

class MyManager extends ConversationManager {
  readonly name = 'my:manager'
  private readonly _maxMessages = 5

  reduce({ agent }: ConversationManagerReduceOptions): boolean {
    return this._trim(agent.messages)
  }

  override initAgent(agent: LocalAgent): void {
    super.initAgent(agent) // 오버플로 복구를 유지합니다
    agent.addHook(AfterInvocationEvent, (event) => {
      this._trim(event.agent.messages)
    })
  }

  private _trim(messages: LocalAgent['messages']): boolean {
    if (messages.length <= this._maxMessages) return false
    messages.splice(0, messages.length - this._maxMessages)
    return true
  }
}
```

`reduce`와 훅이 같은 `_trim`을 공유해서, 오버플로 복구와 호출 후 정리가 같은 규칙으로 이력을 줄입니다. 직접 작성할 때는 두 SDK 모두 내장 `SlidingWindowConversationManager` 구현을 참고 예제로 삼아 미완성 메시지 처리 같은 세부를 확인하세요.

## 구조화된 출력

에이전트의 최종 답이 다른 프로그램의 입력이 되어야 할 때가 많습니다. 텍스트 답을 정규식이나 JSON 파서로 해석하면 형식이 조금만 어긋나도 깨집니다. 구조화된 출력을 쓰면 에이전트가 텍스트 대신 검증을 통과한 객체를 돌려줍니다. 스키마를 정의해 에이전트나 호출 하나에 붙이고, 결과에서 객체를 읽으면 됩니다.

### 스키마를 도구로 바꾸는 원리

Strands는 모델에 "답을 JSON으로 써 달라"고 요청하지 않습니다. 스키마를 도구로 바꾸고, 모델은 필드 값을 입력으로 넣어 그 도구를 호출하는 방식으로 객체를 돌려줍니다. 이렇게 하면 모델은 채워 넣을 타입이 정해진 입력 스키마를 받습니다. SDK도 검증 오류를 에이전트 루프로 되돌릴 통로를 얻습니다.

스키마가 붙은 호출 하나는 다음 순서로 진행됩니다.

1. 스키마로 만든 구조화된 출력 도구를 에이전트의 도구 목록에 더합니다.
2. 모델은 평소처럼 작업하고, 그 사이에 다른 도구도 부를 수 있습니다.
3. 모델이 필드 값을 담아 구조화된 출력 도구를 호출합니다.
4. SDK가 입력을 검증합니다. 통과하면 호출이 끝나고 객체가 `result.structured_output`(TypeScript `result.structuredOutput`)에 담깁니다. 스트리밍으로 호출했다면 마지막 결과 이벤트에 들어 있습니다.
5. 호출이 끝나면 구조화된 출력 도구를 목록에서 제거합니다.

도구 정의는 스키마에서 그대로 만들어지므로, 모델은 모든 필드 설명을 읽습니다. Python에서는 Pydantic 모델이 스키마입니다. 도구 이름은 클래스 이름을 따르므로 `TicketTriage` 스키마는 `TicketTriage` 도구가 됩니다. 클래스 독스트링은 도구 설명이 되고, 각 `Field(description=...)`는 해당 필드의 설명이 됩니다. TypeScript에서는 Zod 객체 스키마를 쓰고 도구 이름은 `strands_structured_output`으로 정해집니다. 객체에 붙인 `.describe()`가 도구 설명이 되고, 필드에 붙인 `.describe()`가 필드 설명이 됩니다. SDK는 도구 설명에 "필요한 다른 도구를 다 쓴 뒤 이 도구를 마지막에 호출하라"는 지시도 덧붙입니다.

스키마는 에이전트 생성 시점에 붙일 수도 있고 호출마다 넘길 수도 있습니다. 에이전트에 붙인 스키마는 모든 호출에 적용되고, 호출에 넘긴 스키마는 그 호출에만 적용되며 에이전트 스키마보다 우선합니다. 에이전트 스키마는 맥락만 쌓는 턴에도 적용되므로, 일부 호출은 일반 텍스트로 답해야 한다면 호출마다 스키마를 넘기는 편이 맞습니다. Python의 `Agent.structured_output()`과 `Agent.structured_output_async()`는 폐기 예정이라 `DeprecationWarning`을 내므로, 새 코드에서는 호출 인자로 스키마를 넘기세요.

### 검증 오류를 모델에 돌려주기

스키마에는 모델이 첫 시도에 지키지 못하는 규칙도 넣을 수 있습니다. "기한은 미래 날짜여야 한다" 같은 규칙이 그 예입니다. 입력이 검증에 실패해도 SDK는 예외를 던지지 않습니다. 검증 오류를 도구 결과로 모델에 돌려주고, 모델은 값을 고쳐 도구를 다시 호출할 수 있습니다. 모델은 오늘 날짜를 모르기 때문에 첫 추측이 과거 날짜일 수 있습니다. 그래도 "due_date must be after …"라는 오류를 받으면 다음 호출에서 날짜를 고칩니다.

```python
from datetime import date

from pydantic import BaseModel, Field, field_validator
from strands import Agent


class FollowUp(BaseModel):
    """A follow-up task for a support ticket."""

    action: str = Field(description="What to do in the follow-up")
    due_date: date = Field(description="When the follow-up is due")

    @field_validator("due_date")
    @classmethod
    def must_be_in_future(cls, value: date) -> date:
        if value <= date.today():
            # 모델이 이 문장을 그대로 읽으므로 고칠 방법을 담아 씁니다
            raise ValueError(f"due_date must be after {date.today().isoformat()}")
        return value


agent = Agent(callback_handler=None)
result = agent(
    "Schedule a follow-up call with the customer next Tuesday.",
    structured_output_model=FollowUp,
)
print(result.structured_output)
```

TypeScript에서는 같은 규칙을 Zod의 `refine`으로 표현합니다.

```typescript
import { Agent } from '@strands-agents/sdk'
import { z } from 'zod'

const today = new Date().toISOString().slice(0, 10)

const FollowUpSchema = z
  .object({
    action: z.string(),
    dueDate: z.iso.date().refine((value) => value > today, {
      message: `dueDate must be after ${today}`,
    }),
  })
  .describe('A follow-up task for a support ticket')

const agent = new Agent({ printer: false })
const result = await agent.invoke(
  'Schedule a follow-up call with the customer next Tuesday.',
  { structuredOutputSchema: FollowUpSchema },
)
console.log(result.structuredOutput)
```

이런 규칙은 프롬프트가 아니라 스키마에 두세요. 프롬프트에 적은 규칙은 모델이 지키는지 확인할 방법이 없습니다. 스키마의 규칙은 매번 검사되고, 위반하면 정확한 오류 문장이 모델에 전달됩니다. 오류 메시지는 모델이 읽고 바로 행동할 수 있는 지시문으로 쓰는 것이 좋습니다. 기본 상태에서 모델은 필요한 만큼 몇 번이든 다시 시도하며, 횟수에 상한을 두려면 아래 시도 횟수 제한을 쓰면 됩니다.

### 강제 도구 호출

모델이 구조화된 출력 도구를 한 번도 부르지 않고 텍스트 답으로 턴을 끝내는 경우도 있습니다. 이때 SDK는 구조화된 출력 도구를 필수로 지정해 한 번 더 시도합니다. 모델에 남은 선택지는 그 도구를 호출하는 것뿐입니다. 이 재시도 방식은 두 SDK가 다릅니다.

Python은 재시도 전에 대화에 사용자 메시지 `You must format the previous response as structured output.`을 더합니다. 이 문구를 바꾸려면 에이전트나 호출에 `structured_output_prompt`를 넘기며, 호출에 넘긴 값이 에이전트 값보다 우선합니다. TypeScript는 텍스트 답을 버리고 도구 선택만 바꿔 다시 시도합니다. 메시지를 더하지 않으므로 설정할 프롬프트도 없습니다.

```python
from pydantic import BaseModel
from strands import Agent
from strands.types.exceptions import StructuredOutputException


class Contact(BaseModel):
    name: str
    email: str


agent = Agent(
    structured_output_prompt="Return the contact details you found as a Contact.",
    callback_handler=None,
)

try:
    result = agent("Reach Dana Ruiz at dana@example.com.", structured_output_model=Contact)
    print(result.structured_output)
except StructuredOutputException:
    # 강제 재시도도 유효한 객체를 내지 못했습니다. 대체 경로로 넘어갑니다
    ...
```

강제 재시도에서도 유효한 구조화된 출력이 나오지 않으면 호출은 `StructuredOutputException`(TypeScript `StructuredOutputError`)을 던집니다. 이 예외는 정지 이유로 돌아오지 않으므로, 애플리케이션이 대체 처리를 할 수 있는 위치에서 잡아야 합니다. 예외 대신 다시 시도하게 하려면 시도 한도를 지정합니다.

### 시도 횟수 제한

검증 실패가 계속되면 모델 호출도 계속 이어집니다. 실패 횟수에 상한을 두려면 호출 한도에 `limits['structured_output_attempts']`(TypeScript `limits.structuredOutputAttempts`)를 지정합니다. 구조화된 출력을 내지 못한 모델 응답 하나가 시도 한 번으로 계산됩니다. 검증에 실패한 도구 호출과, 유효한 결과를 내지 못한 강제 재시도가 여기에 해당합니다. 강제 재시도 이전에 다른 도구를 부른 것은 세지 않습니다.

한도를 지정하면 강제 재시도에 실패해도 예외를 던지지 않고 한도 안에서 다시 시도합니다. 한도에 닿으면 다음 모델 호출 전에 호출이 멈춥니다. 결과에는 정지 이유 `limit_structured_output_attempts`(TypeScript `limitStructuredOutputAttempts`)가 담기고 구조화된 출력은 비어 있습니다. 실패가 예외에서 반환값으로 바뀌므로, 다른 예산 소진과 같은 분기에서 처리할 수 있습니다.

```typescript
import { Agent } from '@strands-agents/sdk'
import { z } from 'zod'

const ContactSchema = z.object({
  name: z.string(),
  email: z.string(),
})

const agent = new Agent({ printer: false })
const result = await agent.invoke(
  'Find the contact details in this note: call Dana back tomorrow.',
  { structuredOutputSchema: ContactSchema, limits: { structuredOutputAttempts: 3 } },
)

if (result.stopReason === 'limitStructuredOutputAttempts') {
  console.log('No valid contact after 3 attempts')
} else {
  console.log(result.structuredOutput)
}
```

Python에서는 같은 키를 스네이크 케이스로 `limits={"structured_output_attempts": 3}`처럼 넘기고 `result.stop_reason`을 비교합니다. 예시의 메모에는 이메일이 없어서 모델이 스키마를 채우지 못할 수 있습니다. 이런 입력에는 한도를 두는 편이 안전합니다.

콘텐츠 필터나 가드레일이 강제 재시도를 막은 경우는 두 SDK가 다르게 처리합니다. TypeScript는 이를 실패한 시도로 세고 한도 안에서 다시 시도합니다. Python은 그 재시도에서 호출을 끝내고 `content_filtered` 같은 해당 정지 이유를 돌려줍니다. 시도 한도는 다른 호출 한도처럼 요청 전체에 적용됩니다. 같은 요청 안에서 도구로 실행되는 에이전트의 시도도 포함되며, `result.invocation_context`(TypeScript `result.invocationContext`)를 다음 호출에 넘기면 카운트가 그대로 이어집니다.

## 인터벤션과 휴먼 인 더 루프

훅으로 도구 실행을 막으려면 `event.cancel`에 문자열을 넣습니다. 그런데 이 문자열만으로는 취소의 성격이 드러나지 않습니다. 프레임워크는 그 취소가 권한 위반에 따른 거부인지, 다르게 다시 해 보라는 부드러운 안내인지 구분하지 못합니다. 같은 이벤트를 플러그인 여러 개가 고치면 마지막에 쓴 값이 이기고, 충돌은 조용히 묻힙니다. 앞쪽 훅이 이미 거부했더라도 뒤쪽 훅이 불필요한 작업을 건너뛰려면 `event.cancel`을 직접 확인해야 합니다.

인터벤션(intervention)은 결정에 타입을 붙여 이 문제를 해결합니다. 핸들러는 이벤트를 직접 고치지 않고 `proceed`, `deny`, `guide`, `confirm`, `transform` 가운데 하나를 반환합니다. 그러면 프레임워크가 정해진 의미에 따라 적용하면서 자동 단락, 피드백 누적, 충돌 해소를 맡습니다. 인가(authorization), 가드레일, 스티어링(steering), 콘텐츠 변환처럼 반복해서 나오는 제어 관심사를 이 공통 모델 위에 쌓을 수 있습니다. 인터벤션은 훅 시스템 위에 만들어져 있어서, 핸들러의 생명주기 메서드 하나가 훅 콜백 하나로 등록됩니다.

### 타입 기반 액션 모델

핸들러는 `InterventionHandler`를 상속하고 필요한 생명주기 메서드만 재정의합니다. 재정의하지 않은 메서드는 모두 `Proceed()`(TypeScript `proceed()`)를 반환합니다. 에이전트에는 `interventions` 옵션으로 등록합니다.

```python
from strands import Agent
from strands.interventions import Deny, InterventionHandler, Proceed

# BeforeToolCallEvent import는 생략했습니다


class ToolGuard(InterventionHandler):
    name = "tool-guard"

    def __init__(self, blocked_tools: list[str]):
        self.blocked_tools = blocked_tools

    def before_tool_call(self, event: BeforeToolCallEvent):
        if event.tool_use["name"] in self.blocked_tools:
            name = event.tool_use["name"]
            return Deny(reason=f"Tool '{name}' is not allowed")
        return Proceed()


# search, delete_file은 이미 정의된 도구라고 가정합니다
agent = Agent(
    tools=[search, delete_file],
    interventions=[ToolGuard(blocked_tools=["delete_file"])],
)

# delete_file 호출은 실행 전에 막히고, 모델은 거부 사유를 보고 접근을 바꿉니다
agent("Clean up the temp directory")
```

TypeScript에서는 같은 다섯 가지 결정을 `InterventionActions`의 팩터리 함수로 만듭니다. Python의 `Deny(reason=...)`는 `InterventionActions.deny(reason)`, `Guide(feedback=...)`는 `InterventionActions.guide(feedback)`, `Confirm(prompt=...)`는 `InterventionActions.confirm(prompt)`, `Transform(apply=fn)`은 `InterventionActions.transform(apply)`에 해당합니다.

액션마다 허용되는 위치가 다릅니다. 실행 전 지점에서는 막거나 되돌릴 수 있지만, 실행이 끝난 뒤에는 이미 일어난 일을 거부할 수 없기 때문입니다.

| 생명주기 메서드 Python / TypeScript | 실행 시점 | 허용 액션 |
| --- | --- | --- |
| `before_invocation` / `beforeInvocation` | 에이전트 루프 시작 전 | Proceed, Deny, Guide, Transform |
| `before_tool_call` / `beforeToolCall` | 각 도구 실행 전 | Proceed, Deny, Guide, Confirm, Transform |
| `after_tool_call` / `afterToolCall` | 각 도구 실행 후 | Proceed, Transform |
| `before_model_call` / `beforeModelCall` | 각 모델 API 호출 전 | Proceed, Deny, Guide, Transform |
| `after_model_call` / `afterModelCall` | 각 모델 응답 후 | Proceed, Guide, Transform |

같은 액션이라도 Before 이벤트와 After 이벤트에서 효과가 다릅니다.

| 액션 | Before 이벤트 | After 이벤트 |
| --- | --- | --- |
| Deny | `event.cancel`을 설정하고 나머지 핸들러를 건너뜁니다. | 효과가 없고 런타임 경고만 냅니다. |
| Guide | `before_tool_call`·`before_invocation`에서는 누적된 피드백으로 취소하고, `before_model_call`에서는 피드백을 사용자 메시지로 넣습니다. | 피드백을 넣고 다시 시도합니다. |
| Confirm | 인터럽트·재개로 멈춰 사람의 승인을 기다리고, 거부 응답이면 `event.cancel`을 설정합니다. | 지원하지 않습니다. |
| Transform | `action.apply(event)`를 부르고, 뒤에 평가되는 핸들러는 바뀐 내용을 봅니다. | `action.apply(event)`를 부릅니다. |

`Deny`와 `Guide`의 차이는 모델 쪽에서 보면 분명합니다. `Deny`는 "이 작업은 안 된다"는 단호한 거부이고, `Guide`는 "이렇게 고쳐서 다시 하라"는 교정입니다. 제목 없는 이메일을 막는 검증기에는 `Guide`가 맞고, 운영 데이터베이스 접근을 막는 정책에는 `Deny`가 맞습니다. 실행을 멈추지 않고 입력만 다듬을 때는 `Transform`을 씁니다.

```python
import re

from strands.interventions import Guide, InterventionHandler, Proceed, Transform

# BeforeToolCallEvent import는 생략했습니다


# Guide: 제목 없이 이메일을 보내려 하면 모델을 교정합니다
class EmailValidator(InterventionHandler):
    name = "email-validator"

    def before_tool_call(self, event: BeforeToolCallEvent):
        if event.tool_use["name"] == "send_email":
            tool_input = event.tool_use.get("input", {})
            if not tool_input.get("subject"):
                return Guide(feedback="All emails must include a subject line.")
        return Proceed()


# Transform: 나가는 이메일 본문에서 주민번호 형식의 개인정보를 가립니다
class PiiRedactor(InterventionHandler):
    name = "pii-redactor"

    def before_tool_call(self, event: BeforeToolCallEvent):
        if event.tool_use["name"] == "send_email":

            def redact(e: BeforeToolCallEvent):
                tool_input = e.tool_use.get("input", {})
                body = tool_input.get("body", "")
                tool_input["body"] = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED]", body)

            return Transform(apply=redact)
        return Proceed()
```

### 평가 순서와 단락

핸들러는 등록한 순서대로 평가됩니다. 어느 핸들러든 `Deny`를 반환하면 나머지 핸들러는 건너뛰고 작업은 그 자리에서 막힙니다. 이 규칙을 이용하면 파이프라인 순서를 비용 기준으로 짤 수 있습니다. 인가 검사나 호출 횟수 제한처럼 빠른 검사를 앞에 두면, LLM 기반 스티어링처럼 비싼 평가를 이미 거부된 작업에 낭비하지 않습니다.

```python
from strands import Agent
from strands.interventions import Deny, Guide, InterventionHandler, Proceed

# BeforeToolCallEvent, AfterModelCallEvent import는 생략했습니다


class RateLimiter(InterventionHandler):
    name = "rate-limiter"

    def __init__(self):
        self.call_count = 0

    def before_tool_call(self, event: BeforeToolCallEvent):
        self.call_count += 1
        if self.call_count > 10:
            # 단락: 이 핸들러 뒤에 등록된 핸들러는 실행되지 않습니다
            return Deny(reason="Rate limit exceeded")
        return Proceed()


class ToneSteering(InterventionHandler):
    name = "tone-steering"

    def after_model_call(self, event: AfterModelCallEvent):
        # 거부된 도구 호출에 대해서는 실행되지 않습니다
        return Guide(feedback="Use a more professional tone.")


agent = Agent(
    tools=[search],
    interventions=[
        RateLimiter(),   # 먼저 평가됩니다
        ToneSteering(),  # RateLimiter가 거부하면 건너뜁니다
    ],
)
```

`Guide`는 단락하지 않습니다. 모든 핸들러가 계속 실행되고, 안내를 반환한 핸들러들의 피드백이 하나로 합쳐져 모델에 전달됩니다. 훅이었다면 두 검증기가 `cancel` 메시지를 차례로 덮어써서 앞의 안내가 사라졌겠지만, 인터벤션에서는 "제목을 넣어라"와 "어조를 바꿔라"를 모델이 한 번에 받습니다. 액션 사이의 우선순위는 deny > confirm > guide > transform > proceed로 정해져 있습니다.

`after_model_call`의 `Guide`에는 함정이 있습니다. 이 위치의 안내는 모델 재시도를 일으키는데, 프레임워크는 안내로 시작된 재시도에 상한을 두지 않습니다. 앞 예제의 `ToneSteering`처럼 무조건 `Guide`를 반환하는 핸들러는 모델이 어떤 응답을 내든 재시도를 계속 일으킵니다. 그러므로 수렴은 핸들러가 직접 보장해야 합니다. 재시도 횟수를 세다가 정해 둔 횟수를 넘으면 안내를 멈추는 식입니다. 반복 실패 뒤 `Deny`로 격상하는 방법도 거론되지만, 앞의 표처럼 After 이벤트의 `Deny`는 경고만 내고 효과가 없습니다. 그래서 아래 예제는 예산을 다 쓰면 `Proceed()`를 반환합니다. 카운터는 `before_invocation`에서 비워, 할당량이 호출 한 번 단위로 적용되게 합니다.

```python
from strands.interventions import Guide, InterventionHandler, Proceed

# AfterModelCallEvent import는 생략했습니다


class BoundedToneSteering(InterventionHandler):
    name = "bounded-tone-steering"

    def __init__(self, max_guides: int = 2):
        self.max_guides = max_guides
        self.guides = 0

    def before_invocation(self, event):
        self.guides = 0  # 호출마다 재시도 예산을 새로 시작합니다
        return Proceed()

    def after_model_call(self, event: AfterModelCallEvent):
        if self.guides >= self.max_guides:
            return Proceed()  # 예산을 다 쓰면 더는 재시도시키지 않습니다
        if needs_more_formal_tone(event):  # 판정 함수는 직접 작성한다고 가정합니다
            self.guides += 1
            return Guide(feedback="Use a more professional tone.")
        return Proceed()
```

### 오류 정책: 실패했을 때 열지 닫을지

외부 인가 서비스나 로깅 서비스에 의존하는 핸들러는 그 서비스가 죽으면 예외를 던집니다. 이때 작업을 막을지 통과시킬지는 핸들러의 성격에 따라 달라야 합니다. 핸들러마다 `on_error`(TypeScript `onError`)로 이 정책을 선언합니다.

| 값 | 핸들러가 예외를 던지면 |
| --- | --- |
| `'throw'` | 기본값입니다. 오류를 다시 던져 호출이 실패합니다. |
| `'proceed'` | 핸들러를 건너뛰고 `Proceed()`를 반환한 것처럼 진행합니다(fail-open). |
| `'deny'` | 오류를 기록하고 `Deny`로 처리합니다(fail-closed). |

Python에서는 프로퍼티로, TypeScript에서는 `readonly onError: OnError = 'deny'`처럼 필드로 선언합니다.

```python
from strands.interventions import Deny, InterventionHandler, OnError, Proceed

# BeforeToolCallEvent import는 생략했습니다


class StrictAuth(InterventionHandler):
    name = "strict-auth"

    @property
    def on_error(self) -> OnError:
        return "deny"  # 인가 서비스가 죽으면 작업을 막습니다

    def before_tool_call(self, event: BeforeToolCallEvent):
        if not self._check_permission(event.tool_use["name"]):
            return Deny(reason="Unauthorized")
        return Proceed()

    def _check_permission(self, tool_name: str) -> bool:
        # ... 외부 인가 서비스를 호출합니다
        return True
```

보안에 중요한 핸들러에는 `'deny'`를 쓰고, `'proceed'`는 집행보다 가용성이 중요한 로깅 같은 핸들러에만 쓰세요. 인가 핸들러에 `'proceed'`를 쓰면 핸들러가 예외를 던지는 순간 그 핸들러가 지키던 작업이 검사 없이 실행됩니다. 훅에는 이런 정책이 없어서 던진 오류가 항상 전파됩니다.

### Confirm: 도구 실행 전에 사람에게 묻기

`Confirm`은 `before_tool_call`에서만 쓸 수 있고, SDK의 인터럽트·재개 시스템과 연결되어 도구가 실행되기 전에 사람의 승인을 기다립니다. 핸들러가 인터럽트 생명주기를 직접 관리할 필요는 없습니다. 승인 요청 문구만 반환하면 나머지는 프레임워크가 처리합니다.

Python의 `Confirm`은 `response` 유무에 따라 두 모드로 동작합니다. `response`를 주면 그 값이 바로 `evaluate` 함수로 전달되므로 에이전트가 멈추지 않습니다. `response`가 없으면 에이전트 루프를 빠져나와 인터럽트 시스템으로 외부 재개를 기다립니다. `evaluate`는 응답이 승인인지 판정하며, 기본값은 `True`, `"y"`, `"yes"`(대소문자 무시)를 승인으로 받습니다.

```python
from strands.interventions import Confirm, InterventionHandler, Proceed

# BeforeToolCallEvent import는 생략했습니다


class SensitiveToolApproval(InterventionHandler):
    name = "sensitive-tool-approval"

    def before_tool_call(self, event: BeforeToolCallEvent):
        if event.tool_use["name"] in ("delete_file", "send_email"):
            return Confirm(prompt=f"Allow {event.tool_use['name']}?")
        return Proceed()


# 미리 승인: 응답을 함께 넘기므로 에이전트가 멈추지 않습니다
class AutoApprove(InterventionHandler):
    name = "auto-approve"

    def before_tool_call(self, event: BeforeToolCallEvent):
        if event.tool_use["name"] == "search":
            return Confirm(prompt="Allow search?", response="yes")
        return Proceed()
```

TypeScript에서는 `InterventionActions.confirm(prompt)`가 인터럽트로 루프를 멈추고, 인터럽트가 외부에서 해결되면 재개됩니다.

### HumanInTheLoop로 승인 흐름 구성하기

`Confirm`을 직접 쓰면 어떤 도구를 물을지, 응답을 어디서 받을지를 매번 작성해야 합니다. 휴먼 인 더 루프(human-in-the-loop) 핸들러 `HumanInTheLoop`는 이 흐름을 미리 만들어 둔 인터벤션입니다. 내부적으로 `confirm` 액션을 쓰고, `ask` 옵션을 주면 수동 인터럽트·재개 루프까지 감춰 줍니다. `interventions`에 넘기고 응답 수집 방식만 고르면 됩니다.

| 수집 방식 | 맞는 경우 |
| --- | --- |
| 인터럽트·재개(기본값) | 요청 사이에 상태를 저장하는 무상태 배포입니다 |
| `"stdio"` | 사용자가 터미널에 입력하는 CLI 애플리케이션입니다 |
| 사용자 정의 콜백 | 웹 UI, Slack 봇 같은 비동기 인터페이스입니다 |

`ask`를 생략하면 핸들러가 인터럽트를 일으키고, 에이전트는 정지 이유 `interrupt`로 멈춥니다. 호출자는 인터럽트를 사용자에게 보여 주고, 응답을 받아 에이전트를 다시 호출합니다.

```typescript
import { Agent, tool, InterruptResponseContent } from '@strands-agents/sdk'
import { HumanInTheLoop } from '@strands-agents/sdk/vended-interventions/hitl'
import { z } from 'zod'

const deleteFiles = tool({
  name: 'delete_files',
  description: 'Delete files at the given paths',
  inputSchema: z.object({ paths: z.array(z.string()) }),
  callback: (input) => `Deleted ${input.paths.length} files`,
})

const agent = new Agent({
  tools: [deleteFiles],
  interventions: [new HumanInTheLoop()],
})

let result = await agent.invoke('Delete the temp files')

if (result.stopReason === 'interrupt') {
  // 웹 UI, Slack 등으로 사용자에게 보여 줍니다
  console.log(result.interrupts![0].reason)

  // 사람의 응답으로 재개합니다. 'y', 'yes', true는 승인입니다
  result = await agent.invoke([
    new InterruptResponseContent({
      interruptId: result.interrupts![0].id,
      response: 'yes',
    }),
  ])
}
```

Python에서는 `result.interrupts[0].id`를 담은 `{"interruptResponse": {"interruptId": ..., "response": "yes"}}` 딕셔너리를 리스트로 넘겨 재개합니다. 요청마다 새 에이전트를 만드는 무상태 서비스는 재개할 때 같은 상태가 있어야 하므로, 세션 관리자와 함께 써서 요청 사이에 상태를 보존해야 합니다.

`ask="stdio"`(TypeScript `ask: 'stdio'`)를 주면 그 자리에서 표준 입력으로 묻고, 사용자가 답할 때까지 에이전트가 기다립니다. 호출자 쪽에 인터럽트 처리 코드가 필요 없습니다. 함수를 넘기면 그 함수가 도구 호출을 설명하는 프롬프트 문자열을 받아 사용자 응답을 반환합니다.

```python
# HumanInTheLoop과 delete_files의 import·정의는 생략했습니다
async def ask(prompt: str) -> str:
    # Slack DM, 웹 모달, 푸시 알림 등 직접 만든 UI입니다
    return await ask_user_via_slack(prompt)


agent = Agent(
    tools=[delete_files],
    interventions=[HumanInTheLoop(ask=ask)],
)
agent("Delete the temp files")
```

모든 도구 호출을 사람에게 물으면 승인 피로가 쌓이고, 사람은 결국 습관적으로 승인하게 됩니다. `HumanInTheLoop`는 사람에게 묻기 전에 호출을 먼저 걸러 냅니다. 도구 호출이 허용 목록이나 신뢰 기록에 있으면 바로 실행합니다. 없으면 분류기가 설정된 경우 위험도를 판정해 안전하면 실행하고, 위험하거나 분류기가 없으면 사람에게 묻습니다. 사람이 승인하면 실행하고, 거부하면 취소합니다.

`allowed_tools`(TypeScript `allowedTools`)는 승인 없이 실행할 도구 목록입니다. `"*"`는 모든 도구를 허용해 사실상 핸들러를 끄고, `"!tool_name"`은 부정을 뜻합니다. 예를 들어 `["*", "!delete_files"]`는 `delete_files`만 빼고 모두 허용하며, 부정한 도구는 항상 사람에게 묻습니다. `classifier`를 주면 호출마다 승인 필요 여부를 판정합니다. 허용 목록의 도구는 분류기를 아예 거치지 않으므로, 안전한 읽기 전용 도구를 목록에 넣으면 분류 비용도 아낍니다. `True`는 내장 기본 설정의 LLM 분류기이고, 설정 객체로 시스템 프롬프트나 모델을 바꾸거나 직접 작성한 분류 함수를 넘길 수도 있습니다.

```python
# HumanInTheLoop, read_file, delete_files의 import·정의는 생략했습니다
from strands.vended_interventions.hitl.classifier import ClassifierResult


def my_classifier(event, **kwargs):
    is_dangerous = event.tool_use["name"].startswith("delete")
    return ClassifierResult(
        requires_human_in_the_loop=is_dangerous,
        reason="destructive tool" if is_dangerous else None,
    )


agent = Agent(
    tools=[read_file, delete_files],
    interventions=[
        HumanInTheLoop(
            allowed_tools=["read_file"],  # 분류기를 거치지 않고 실행합니다
            classifier=my_classifier,
            enable_trust=True,
        ),
    ],
)
```

`enable_trust`(TypeScript `enableTrust`, 기본값 꺼짐)를 켜면, 사람이 `'t'`나 `'trust'`로 답했을 때 이번 호출을 승인하고 그 결정을 세션이 끝날 때까지 기억합니다. 이후 그 도구를 부르는 호출은 묻지 않고 실행됩니다. 신뢰는 세 가지 수집 방식에서 모두 동작합니다. 신뢰 상태는 `agent.state`(TypeScript `agent.appState`)에 저장되어 같은 세션의 여러 턴에 걸쳐 유지되고, 에이전트를 다시 만들면 초기화됩니다. 부정한 도구는 신뢰할 수 없습니다. 승인과 신뢰를 판정하는 기준은 각각 `evaluate`와 `evaluate_trust`(TypeScript `evaluateTrust`)로 바꿉니다. 예를 들어 파괴적인 도구라면 `"y"` 대신 `"confirm"`을 정확히 입력해야 승인되도록 `evaluate`를 좁힐 수 있습니다.

허용 목록, 신뢰, 수집 방식만으로 끝나는 도구 단위 승인이라면 `HumanInTheLoop`가 가장 적은 코드로 해결합니다. 인터럽트 형태를 직접 정해야 하거나, 여러 단계에 걸친 대화가 필요하거나, 승인·거부를 넘어서는 흐름이라면 인터럽트를 직접 다루세요.

## 스티어링(Steering)

단계가 30개를 넘는 복잡한 작업을 에이전트에 맡기면 프롬프트가 한계에 부딪힙니다. 지시와 업무 규칙, 운영 지침을 시스템 프롬프트 하나에 모두 넣으면 프롬프트가 비대해집니다. 그러면 에이전트는 지시를 무시하거나, 없는 동작을 지어내거나, 꼭 지켜야 할 절차를 건너뜁니다. 흔한 우회책은 에이전트를 노드와 엣지가 정해진 그래프로 쪼개는 방법입니다. 예측 가능성은 좋아지지만 에이전트의 장점인 적응적 추론이 줄어듭니다. 요구사항이 바뀔 때마다 흐름 로직을 다시 짜야 해서 유지 비용도 큽니다.

스티어링은 모듈형 프롬프팅으로 이 문제를 풉니다. 모든 지시를 처음에 몰아넣지 않고, 맥락을 보고 필요한 순간에만 피드백을 주는 핸들러를 정의합니다. 핸들러는 지킬 업무 규칙과 그 규칙을 검사할 생명주기 지점을 함께 정합니다. 필요할 때만 붙는 메모지처럼 동작하므로 컨텍스트 윈도우를 가볍게 유지하면서 긴 절차도 안정적으로 따르게 합니다.

개념은 두 SDK가 같지만 구현 기반은 다릅니다. Python에서 스티어링 핸들러는 플러그인이고, TypeScript에서는 앞서 다룬 인터벤션 프레임워크와 그 타입 기반 액션 모델 위에 만들어져 있습니다.

### 두 개입 지점: 도구 호출 전과 모델 응답 후

핸들러가 개입하는 지점은 둘입니다. 도구 호출 전에는 실행 직전의 호출을 검사하고, 모델 응답 후에는 응답이 사용자에게 가기 전에 검사합니다.

| 시점 | Python 액션 | TypeScript 액션 | 효과 |
| --- | --- | --- | --- |
| 도구 호출 전 | `Proceed` | `InterventionActions.proceed()` | 도구를 바로 실행합니다. |
| 도구 호출 전 | `Guide` | `InterventionActions.guide(feedback)` | 도구 호출을 취소하고, 에이전트가 피드백을 받아 다시 시도합니다. |
| 도구 호출 전 | `Interrupt` | `InterventionActions.confirm(prompt)` | 도구 실행을 멈추고 사람의 입력이나 승인을 기다립니다. |
| 모델 응답 후 | `Proceed` | `InterventionActions.proceed()` | 응답을 그대로 받아들입니다. |
| 모델 응답 후 | `Guide` | `InterventionActions.guide(feedback)` | 응답을 버리고, 안내를 대화에 넣어 다시 시도합니다. |

도구 호출 전 스티어링은 실행을 막는 데서 그치지 않고 맥락에 맞는 피드백을 줍니다. 에이전트는 이 피드백을 받아 호출을 다시 시도합니다. 모델 응답 후 스티어링은 응답을 검사하거나, 작업을 마치기 전에 필수 도구를 썼는지 확인하거나, 출력에 따라 대화 흐름을 이끄는 데 씁니다.

### 자연어로 규칙 쓰기: `LLMSteeringHandler`

규칙을 명령형 코드가 아니라 평이한 문장으로 표현하고 싶다면 `LLMSteeringHandler`를 씁니다. 이 핸들러는 판단을 LLM에 맡깁니다. LLM은 대기 중인 도구 호출을 시스템 프롬프트와 지금까지 쌓인 스티어링 컨텍스트에 비추어 평가합니다. Python에서는 에이전트의 `plugins`에 핸들러를 넘깁니다.

```python
from strands import Agent, tool
from strands.vended_plugins.steering import LLMSteeringHandler


@tool
def send_email(recipient: str, subject: str, message: str) -> str:
    """Send an email to a recipient."""
    return f"Email sent to {recipient}"


handler = LLMSteeringHandler(
    system_prompt="""
    You are providing guidance to ensure emails maintain a cheerful, positive tone.

    Guidance:
    - Review email content for tone and sentiment
    - Suggest more cheerful phrasing if the message seems negative or neutral
    - Encourage use of positive language and friendly greetings
    """
)

agent = Agent(
    tools=[send_email],
    plugins=[handler],
)

agent(
    "Send a frustrated email to tom@example.com, "
    "a client who keeps rescheduling important meetings at the last minute"
)
```

사용자는 짜증 섞인 이메일을 요청했습니다. 핸들러는 첫 `send_email` 호출을 가로채 어조를 평가하고 취소합니다. 그 결과 `agent.messages`에는 대개 취소된 호출, 안내 메시지, 더 밝은 문장으로 다시 시도한 호출이 차례로 남습니다. TypeScript에서는 같은 핸들러를 `@strands-agents/sdk/vended-interventions/steering`에서 가져와 `interventions` 옵션에 넣습니다.

내부 동작은 TypeScript 문서에 자세히 나와 있습니다. 도구 호출 전마다 핸들러는 등록된 모든 제공자에서 컨텍스트 스냅숏을 모읍니다. 이어서 시스템 프롬프트, 컨텍스트 데이터, 대기 중인 도구 호출을 합쳐 평가 프롬프트를 만듭니다. 그다음 새로 만든 내부 에이전트가 `proceed`, `guide`, `confirm` 가운데 하나를 구조화된 결정으로 돌려주고, 이 결정이 해당 인터벤션 액션으로 바뀝니다. 내부 에이전트를 평가마다 새로 만들기 때문에 핸들러에는 상태가 없습니다. 그래서 여러 부모 에이전트에서 동시에 평가해도 안전합니다. 기본 프롬프트 빌더는 고정 지시와 변하는 컨텍스트 사이에 `CachePointBlock`을 두어 프롬프트 캐싱을 씁니다. 같은 세션에서 평가가 반복될수록 API 비용이 줄어듭니다.

| 옵션 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `systemPrompt` | `string \| SystemContentBlock[]` | 필수 | 평가 LLM에 줄 스티어링 규칙입니다. |
| `model` | `Model` | `agent.auxModel` | 스티어링 평가에 쓰는 모델입니다. |
| `contextProviders` | `SteeringContextProvider[]` | `[new ToolLedgerProvider()]` | 평가 컨텍스트를 공급하는 제공자입니다. `[]`를 넘기면 끕니다. |
| `promptBuilder` | `PromptBuilder` | 캐시 지점이 있는 내장 빌더 | 평가 프롬프트를 만드는 사용자 정의 함수입니다. |
| `name` | `string` | `'strands:llm-steering-handler'` | 핸들러 고유 이름입니다. 핸들러를 여럿 붙이면 서로 달라야 합니다. |

### 컨텍스트 제공자와 도구 원장

핸들러가 판단하려면 근거가 필요합니다. 컨텍스트 제공자는 에이전트 활동을 관찰하고, 그 내용을 구조화된 데이터로 핸들러의 스티어링 컨텍스트에 기록합니다. 내장 제공자인 도구 원장(tool ledger)은 Python에서는 `LedgerProvider`, TypeScript에서는 `ToolLedgerProvider`라는 이름입니다. 도구 원장은 세션 안의 도구 호출마다 이름, 입력 인자, 시작·종료 시각, 실행 상태(`pending`, `success`, `error`), 결과 또는 오류 내용을 남깁니다. 핸들러는 이 기록에서 연속 실패, 과도한 재시도, 도구 사용 순서 같은 패턴을 읽습니다. `LLMSteeringHandler`는 기본으로 도구 원장을 쓰며, 두 SDK 모두 원장을 JSON으로 직렬화할 수 있는 형태로 저장하므로 LLM이 바로 읽을 수 있습니다. TypeScript의 `ToolLedgerProvider`는 `maxEntries`(기본값 `100`, 넘치면 가장 오래된 호출부터 버림)와 `name`(기본값 `'strands:steering:toolLedger'`)을 받습니다.

TypeScript에서 원장이 담지 않는 신호가 필요하면 `SteeringContextProvider` 인터페이스를 직접 구현합니다. 구성 요소는 셋입니다. `name`은 인스턴스 식별자이고, `observeAgent(agent)`는 초기화 때 한 번 불리며 여기서 훅을 겁니다. `context` 게터는 평가 시점의 스냅숏을 돌려줍니다.

```typescript
import { Agent, AfterToolCallEvent } from '@strands-agents/sdk'
import type { LocalAgent } from '@strands-agents/sdk'
import {
  LLMSteeringHandler,
  ToolLedgerProvider,
} from '@strands-agents/sdk/vended-interventions/steering'
import type {
  SteeringContextProvider,
  SteeringContextData,
} from '@strands-agents/sdk/vended-interventions/steering'

// 도구가 끝날 때마다 호출 수를 셉니다
class ToolCallCounter implements SteeringContextProvider {
  readonly name = 'toolCallCounter'
  private _count = 0

  observeAgent(agent: LocalAgent): void {
    agent.addHook(AfterToolCallEvent, () => {
      this._count += 1
    })
  }

  get context(): SteeringContextData {
    return { type: 'toolCallCounter', totalCalls: this._count }
  }
}

const handler = new LLMSteeringHandler({
  systemPrompt: `
    Monitor tool usage. If the agent has made more than 5 tool calls,
    guide it to wrap up and produce a final answer.
  `,
  // 직접 지정하면 기본 원장이 빠지므로 함께 넣습니다
  contextProviders: [new ToolCallCounter(), new ToolLedgerProvider()],
})

// searchWeb은 이미 정의된 도구라고 가정합니다
const agent = new Agent({
  tools: [searchWeb],
  interventions: [handler],
})

await agent.invoke('Research the history of quantum computing')
```

제공자가 `observeAgent`에서 필요한 훅을 스스로 등록하므로, 핸들러는 제공자가 어떤 훅을 쓰는지 알 필요가 없습니다. 이 일은 `SteeringHandler`가 맡습니다. 이 클래스는 `LifecycleObserver`를 구현하고 있어서, 에이전트가 초기화하며 `observeAgent(agent)`를 부르면 그 호출을 등록된 제공자들에 넘깁니다. 역할은 분명히 나뉩니다. 제공자는 관찰하고, 핸들러는 결정합니다.

### TypeScript에서 명령형 스티어링 작성하기

규칙이 코드로 명확히 표현된다면 LLM 평가를 거칠 이유가 없습니다. TypeScript에서는 `SteeringHandler`를 상속합니다. 이 클래스는 `InterventionHandler`를 확장하되 반환 타입을 좁혀 둡니다. `beforeToolCall`은 `Proceed | Guide | Confirm`만, `afterModelCall`은 `Proceed | Guide`만 반환할 수 있습니다. 인터벤션의 `Deny`와 `Transform`은 타입 수준에서 빠지므로, 스티어링의 계약을 벗어난 액션은 컴파일할 때 걸립니다.

```typescript
import { Agent, InterventionActions } from '@strands-agents/sdk'
import type { BeforeToolCallEvent, AfterModelCallEvent } from '@strands-agents/sdk'
import { SteeringHandler } from '@strands-agents/sdk/vended-interventions/steering'

class ToneSteeringHandler extends SteeringHandler {
  override readonly name = 'tone-steering'

  override beforeToolCall(event: BeforeToolCallEvent) {
    if (event.toolUse.name === 'send_email') {
      const input = event.toolUse.input as Record<string, string>
      if (input.message?.includes('URGENT') || input.message?.includes('!!!')) {
        return InterventionActions.guide(
          'Rewrite the email with a calmer, more professional tone. ' +
            'Avoid all-caps words and excessive punctuation.'
        )
      }
    }
    return InterventionActions.proceed()
  }

  override afterModelCall(_event: AfterModelCallEvent) {
    return InterventionActions.proceed()
  }
}

// sendEmail은 이미 정의된 도구라고 가정합니다
const agent = new Agent({
  tools: [sendEmail],
  interventions: [new ToneSteeringHandler()],
})

await agent.invoke('Send an urgent email to the team about the deadline')
```

이 핸들러는 본문에서 "URGENT"나 연속 느낌표를 찾으면 호출을 취소하고 차분한 어조로 다시 쓰라고 안내합니다. 에이전트가 고쳐 쓴 이메일은 다음 시도에서 발송됩니다. 판정 기준이 문자열 검사처럼 결정적이면 명령형 핸들러를 쓰세요. 어조나 업무 맥락처럼 판단이 필요한 규칙에는 `LLMSteeringHandler`가 맞습니다. 스티어링 프롬프트를 잘 쓰는 방법은 Agent Standard Operating Procedures(SOP) 프레임워크(https://github.com/strands-agents/agent-sop)를 참고하세요.

## 자주 겪는 문제와 설계 권장사항

**컨텍스트 소진.** 도구 호출이 많은 작업은 이력이 컨텍스트 윈도우를 넘기기 쉽고, 그러면 에이전트가 더 진행하지 못합니다. 증상은 두 가지입니다. 공급자가 입력 길이 오류를 내거나, 덜 중요한 앞쪽 메시지가 컨텍스트를 채우면서 응답 품질이 떨어집니다. 대응책은 넷입니다. 도구는 전체 데이터 대신 요약이나 발췌를 돌려주게 하고, 깊게 중첩된 스키마는 단순하게 바꿉니다. 긴 작업에는 요약 관리자나 커스텀 관리자를 쓰고, 큰 작업은 새 컨텍스트에서 도는 하위 작업으로 나눕니다.

**도구 선택 오류.** 모델이 계속 엉뚱한 도구를 고르면 원인은 대개 모호한 설명입니다. 두 도구의 설명이 겹치면 모델에는 둘 중 하나를 고를 근거가 없습니다. 모델의 입장에서 설명을 다시 읽어 보세요.

**MaxTokens 예외.** 응답 하나가 공급자의 호출당 출력 상한을 넘으면, 루프는 결과를 반환하지 않고 `MaxTokensReachedException`(TypeScript `MaxTokensError`)을 던집니다. Python은 복구한 부분 메시지를 이력에 넣은 뒤 예외를 던지므로, 같은 에이전트를 다시 호출하면 그 이력에서 이어 갑니다. TypeScript는 `partialMessage`로 부분 메시지를 보여 줄 뿐 이력에 넣지 않습니다. 입력을 줄이거나, 출력 상한을 올리거나, 작업을 나누세요.

**프로덕션 핸들러.** 동시 요청이 많은 서비스에서는 요청마다 새 에이전트를 만드는 방식이 확장성이 좋습니다. 인스턴스를 공유하지 않으니 동시 호출 가드에 걸리지 않습니다. 여기에 반복 한도, 토큰 한도, 기한을 함께 겁니다. 정지 이유는 앞서 작성한 `Outcome`과 `classify`로 분류합니다.

```python
import threading
from dataclasses import dataclass

from strands import Agent
from strands.models import BedrockModel

MODEL_ID = "global.anthropic.claude-sonnet-5"


@dataclass
class Reply:
    outcome: Outcome  # 앞서 정의한 Outcome, classify를 그대로 씁니다
    text: str | None


def handle_request(prompt: str, deadline_seconds: float = 20.0) -> Reply:
    # 요청마다 새 에이전트를 만들어 이력을 공유하지 않습니다
    agent = Agent(model=BedrockModel(model_id=MODEL_ID))

    cancel_signal = threading.Event()
    deadline = threading.Timer(deadline_seconds, cancel_signal.set)
    deadline.start()

    try:
        result = agent(
            prompt,
            limits={"turns": 6, "total_tokens": 20_000},
            cancel_signal=cancel_signal,
        )
    finally:
        deadline.cancel()

    outcome = classify(result.stop_reason)
    text = str(result) if outcome == Outcome.COMPLETED else None
    return Reply(outcome=outcome, text=text)
```

한도나 취소로 멈춘 호출은 예외 없이 타입이 정해진 결과로 돌아오고, 응답 텍스트는 정상 완료일 때만 노출됩니다. 잘린 응답처럼 복구할 수 없는 오류는 요청 경계까지 전파됩니다. 이 무상태 패턴으로는 인터럽트를 재개할 수 없습니다. 사람의 입력이 필요하면 에이전트나 세션 상태와 인터럽트 ID를 따로 보존해야 합니다.

- - -

### Sources
* [Agent Loop &#124; Strands Agents](https://strandsagents.com/docs/user-guide/sdk/agents/agent-loop/)
* [Hooks &#124; Strands Agents](https://strandsagents.com/docs/user-guide/sdk/agents/hooks/)
* [Retry Strategies &#124; Strands Agents](https://strandsagents.com/docs/user-guide/sdk/agents/retry-strategies/)
* [Conversation Management &#124; Strands Agents](https://strandsagents.com/docs/user-guide/sdk/agents/conversation-management/)
* [Production Lifecycle Controls &#124; Strands Agents](https://strandsagents.com/docs/user-guide/sdk/agents/lifecycle-controls/)
* [Structured output &#124; Strands Agents](https://strandsagents.com/docs/user-guide/sdk/agents/structured-output-behavior/)
* [Hook events &#124; Strands Agents](https://strandsagents.com/docs/user-guide/sdk/agents/hooks-events/)
* [Steering &#124; Strands Agents](https://strandsagents.com/docs/user-guide/sdk/agents/interventions/steering/)
* [Interventions &#124; Strands Agents](https://strandsagents.com/docs/user-guide/sdk/agents/interventions/)
* [Middleware &#124; Strands Agents](https://strandsagents.com/docs/user-guide/sdk/agents/middleware/)
* [Human in the loop &#124; Strands Agents](https://strandsagents.com/docs/user-guide/sdk/agents/interventions/human-in-the-loop/)
* [Python Quickstart &#124; Strands Agents](https://strandsagents.com/docs/user-guide/sdk/quickstart/python)
* [strands.hooks.registry &#124; Strands Agents](https://strandsagents.com/docs/api/python/strands.hooks.registry)
* [Strands Agents SDK: A technical deep dive into agent architectures and observability &#124; Artificial Intelligence](https://aws.amazon.com/blogs/machine-learning/strands-agents-sdk-a-technical-deep-dive-into-agent-architectures-and-observability)
* [strands.models.model &#124; Strands Agents](https://strandsagents.com/docs/api/python/strands.models.model)
* [@strands-agents/sdk - npm](https://www.npmjs.com/package/%40strands-agents%2Fsdk)
