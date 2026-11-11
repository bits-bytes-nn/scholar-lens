# 0004. 리뷰·요약은 두 엔진을 함께 둔다

- 상태: 채택
- 대체: ADR 0003의 삭제 기준(`legacy/`와 전용 설정 삭제)
- 관련: `docs/engine.md` 5.1절, 8장 E9, `explanation.engine`, `summary.engine`

## 배경

ADR 0003은 sections 엔진을 근거 장부 엔진이 이길 때까지 남겨 둘 이전 엔진으로 보고, 이기면 지우기로 했습니다. 그런데 두 엔진은 글을 만드는 방식이 다릅니다. sections 엔진은 논문 전체를 보고 절을 계획·집필·검수하고, 근거 장부 엔진은 원문 인용으로 확인한 근거만으로 절을 씁니다. 평가에서는 sections 엔진이 충분성과 길이에서, 근거 장부 엔진이 근거 추적에서 낫습니다. 한쪽을 지울 이유보다 용도에 따라 고를 이유가 큽니다.

## 결정

- 리뷰·요약은 두 엔진을 대등한 선택지로 둡니다. `explanation.engine`, `summary.engine`이 `sections` 또는 `evidence`를 고릅니다.
- 기본값은 ADR 0002 절차의 평가 결과로 정합니다. 현재 기본값은 `sections`입니다.
- sections 엔진은 `scholar_lens/sections_engine/`에 둡니다(이전 이름 `scholar_lens/legacy/`). 유스케이스(`application/use_cases/paper.py`)가 설정에 따라 sections 엔진은 `SectionsWriter` 포트로, 근거 장부 엔진은 `EvidencePaperEngine`으로 부릅니다. sections 엔진을 import하는 곳은 `bootstrap.py`뿐입니다.
- 가이드와 서베이는 근거 장부 엔진만 씁니다. sections 엔진은 논문 전용입니다.

## 결과

- 두 엔진의 설정, 테스트, 문서를 계속 유지합니다.
- 한 엔진에서만 동작하는 설정은 설정 설명에 그 엔진을 밝힙니다.
- 기본값을 바꿀 때는 이 ADR을 고치지 않고 평가 결과(`docs/redesign.md`)와 설정 기본값만 바꿉니다.
