# Scholar-Lens 설계 문서

현재 코드의 동작을 설명하는 기술 참조 문서입니다. 설치와 CLI 사용법은 [README](../README.ko.md)에 있고, 이 문서는 그 뒤에서 무엇이 어떻게 돌아가는지를 다룹니다. 코드 위치는 `파일::심볼` 형식으로 적습니다.

## 목차

1. [개요와 데이터 흐름](#1-개요와-데이터-흐름)
2. [CLI와 작업 계약](#2-cli와-작업-계약)
3. [리뷰 워크플로](#3-리뷰-워크플로)
4. [요약과 기술 가이드 파이프라인](#4-요약과-기술-가이드-파이프라인)
5. [데이터 수집과 파싱](#5-데이터-수집과-파싱)
6. [블로그 규칙과 검증](#6-블로그-규칙과-검증)
7. [발행](#7-발행)
8. [LLM 런타임](#8-llm-런타임)
9. [설정 참조](#9-설정-참조)
10. [인프라](#10-인프라)
11. [관측성](#11-관측성)
12. [테스트와 CI](#12-테스트와-ci)
13. [확장 지점](#13-확장-지점)

## 1. 개요와 데이터 흐름

Scholar-Lens는 세 종류의 글을 만듭니다.

| 종류 | `kind` | 입력 | 파이프라인 |
| --- | --- | --- | --- |
| 논문 리뷰 | `review` | arXiv ID 또는 PDF URL, 선택적 코드 저장소 | `paper_pipeline.py::run_paper_pipeline` → `explainer.py::PaperReviewer` |
| 논문 요약 | `summarize` | arXiv ID 또는 PDF URL | `paper_pipeline.py::run_paper_pipeline` → `summarizer.py::PaperSummarizer` |
| 기술 가이드 | `guide` | 문서 URL 목록, 참고 자료 URL(`--reference`), 선택적 검색어 | `guide_pipeline.py::run_guide_pipeline` → `tech_guide.py::TechGuideGenerator` |

모든 실행은 같은 경로를 탑니다.

```
CLI (로컬)  ─┐
             ├─ JobRequest ─ runtime.py::run_job ─┬─ paper_pipeline / guide_pipeline
Batch worker ┘                                    │     (수집 → 생성 → 포맷)
                                                  ├─ publisher.py::Publisher.publish
                                                  │     정규화 → 이미지 로컬화 → 규칙 검증 → 저장 → S3
                                                  ├─ Publisher.create_pull_request (GitHub API)
                                                  └─ 지표 · SNS 알림 · 결과 매니페스트(S3)
```

패키지 구성은 다음과 같습니다.

| 경로 | 역할 |
| --- | --- |
| `scholar_lens/cli.py`, `__main__.py` | CLI 진입점(`python -m scholar_lens`) |
| `scholar_lens/configs/config.py` | 설정 모델, `config.yaml` 로드 |
| `src/jobs.py` | `JobRequest`·`JobResult` 계약, Batch 제출·조회, 결과 매니페스트 |
| `src/runtime.py` | 실행 컨텍스트, 비밀값 로드, `run_job`, 발행 연결, 알림 |
| `src/paper_pipeline.py`, `src/guide_pipeline.py` | 산출물별 파이프라인 |
| `src/explainer.py`, `src/review_plan.py` | 리뷰 워크플로와 계획 처리 |
| `src/summarizer.py`, `src/tech_guide.py`, `src/web_research.py` | 요약, 가이드, 웹 조사 |
| `src/paper_source.py`, `src/pdf_download.py`, `src/arxiv_handler.py`, `src/parser.py`, `src/content_extractor.py` | 논문 소스, PDF 내려받기, 파싱, 속성·인용 추출 |
| `src/citation_metadata.py`, `src/citation_summarizer.py`, `src/code_retriever.py` | 인용 해석·요약, 코드 검색 |
| `src/formatting.py`, `src/tags.py`, `src/markdown_math.py`, `src/markdown_lint.py` | 글 조립, 태그 정규화, 수식·Markdown 후처리 |
| `src/post_conventions.py`, `src/post_validator.py`, `src/front_matter.py` | 블로그 규칙 로드, 최종 검증, front matter 분리 |
| `src/publisher.py`, `src/blog_sync.py`, `src/publish_types.py` | 이미지 로컬화, 저장, S3 / GitHub 브랜치·PR 동기화 / 발행 요청·오류·슬러그 |
| `src/utils/` | 모델 팩토리, 모델 레지스트리, 재시도, 배치 처리, 공용 파서, 그림 검사·SVG 래스터화(`images.py`) |
| `src/metrics.py` | 토큰·비용 집계, 예산 가드, CloudWatch 지표 |
| `src/url_guard.py`, `src/pinned_transport.py`, `src/rate_limiter.py`, `src/prompt_safety.py` | SSRF 방지, IP 고정 전송, 속도 제한, 프롬프트 펜스 태그 무력화 |
| `src/aws_helpers.py`, `src/constants.py`, `src/logger.py` | S3·SSM·계정 헬퍼, 공용 상수·경로, 로거 |
| `scripts/` | CDK 스택, CI synth 검사, 비밀값 기록, 블로그 규칙 불일치 검사 |

## 2. CLI와 작업 계약

### 2.1 명령

`cli.py::build_parser`가 하위 명령 다섯 개를 정의합니다.

| 명령 | 동작 |
| --- | --- |
| `review SOURCE`, `summarize SOURCE` | 논문 산출물을 로컬에서 실행하거나 `--remote`로 제출 |
| `guide [URL...] [--reference URL]...` | 기술 가이드를 로컬에서 실행하거나 `--remote`로 제출(문서 URL과 참고 자료 URL 중 하나 이상) |
| `status JOB_ID [--wait] [--timeout S]` | Batch 상태와 결과 매니페스트 조회(`jobs.py::describe_job`, `wait_for_job`) |
| `worker --request JSON` | Batch 진입점. `AWS_BATCH_JOB_ID`를 작업 ID로 사용 |

`guide`의 문서 URL은 옵션 앞·사이·뒤 어디에 와도 됩니다(`guide DOC1 --reference R DOC2`). argparse는 하위 명령이 있는 파서에서 위치 인자와 옵션을 섞어 받지 못하므로 `cli.py::parse_args`가 `guide` 하위 파서로 `parse_intermixed_args`를 호출합니다.

stdout에는 JSON 객체 하나만 출력하고 로그는 stderr로 보냅니다. 인자 누락 같은 사용법 오류도 사용법은 stderr로, `{"status": "rejected"}`는 stdout으로 내고 2로 끝납니다. 종료 코드의 정본은 `cli.py` 모듈 docstring이고, `JobResult.status`의 `succeeded`·`failed`·`rejected`가 각각 0·1·2입니다(`_EXIT_BY_STATUS`). 로컬 실행이 설정·자격 증명 문제로 시작하지 못해도 `failed`(1)입니다.

`--remote`는 `SubmittedJob`을 출력하고 0으로 끝납니다. `status`는 결과 매니페스트가 있으면 그 상태의 종료 코드를 돌려줍니다. 없으면 Batch가 `FAILED`일 때 1, 아직 끝나지 않았거나 `--wait`가 시간 초과되면 3, `SUCCEEDED`인데 매니페스트가 없으면 0입니다. `--wait` 중 Batch 조회나 결과 매니페스트 읽기의 요청 제한·5xx·네트워크 오류는 건너뛰고 계속 조회합니다.

`--remote`나 `status`가 실패하면(자격 증명, 설정, Batch·S3 API 오류) `{"status": "error"}`와 종료 코드 4를 냅니다. 작업 실패(1)와 구분하기 위해서이며, 이때 원격 작업의 상태는 알 수 없습니다.

### 2.2 JobRequest와 JobResult

`jobs.py::JobRequest`는 실행 하나를 완전히 기술합니다. `extra="forbid"`이며, 논문 요청은 `source`를, 가이드 요청은 `urls`(문서)나 `reference_urls`(참고 자료) 중 하나 이상을 요구하고 섞어 쓰면 거부합니다.

| 필드 | 대상 | 기본값 |
| --- | --- | --- |
| `kind` | 전체 | 필수(`review`, `summarize`, `guide`) |
| `source` | review, summarize | 필수 |
| `repo_urls`, `parse_pdf` | review, summarize | `[]`, `false` |
| `urls`, `reference_urls`, `search_queries`, `discover_subpages` | guide | `[]`, `[]`(둘 중 하나 이상 필수), `[]`, `true` |
| `create_pr` | 전체 | `true`(GitHub 연동이 켜져 있을 때만 의미) |

로컬 CLI와 Batch worker는 같은 객체를 씁니다. `--remote`는 `JobRequest.model_dump_json(exclude_defaults=True)`를 Batch 파라미터 `request` 하나로 넘기고, 작업 정의의 명령은 `["python3", "-m", "scholar_lens", "worker", "--request", "Ref::request"]`입니다(`scripts/deploy_infra.py::JOB_COMMAND`). 두 경로가 같은 파서를 쓰므로 인자 계약이 어긋나지 않습니다. 기본값인 필드는 보내지 않으므로, 새 필드를 쓰지 않는 요청은 그 필드가 생기기 전에 배포한 worker 이미지(`extra="forbid"`)도 받습니다. 새 필드(예: `reference_urls`)를 쓰는 요청은 `cdk deploy`로 이미지를 갱신한 뒤에만 원격 실행됩니다(`tests/test_jobs.py`가 ddd37c8 필드 집합으로 검증).

`jobs.py::JobResult`에는 `status`, `title`, `markdown_path`, `s3_uri`, `pr_url`, `warnings`, `violations`, `error`, `job_id`, `duration_seconds`, 토큰 수, `estimated_cost_usd`가 들어갑니다. Batch 실행은 이 결과를 `s3://<bucket>/<s3_prefix>/results/<job_id>.json`에 저장하고(`jobs.py::write_result_manifest`), `status`가 이를 읽어 붙입니다.

### 2.3 실행 수명주기

`runtime.py::run_job` 순서는 다음과 같습니다.

1. `Config.load()`와 `build_context`로 boto 세션(기본 리전, Bedrock 리전)과 S3 핸들러를 만듭니다.
2. AWS 안에서는 SSM SecureString에서 비밀값을 환경 변수로 읽습니다(`load_secrets_from_ssm`). 로컬은 `.env`를 씁니다.
3. 파이프라인을 실행합니다. 입력 오류는 `rejected`, `PostValidationError`는 `failed`와 `violations`, 그 밖의 예외는 `failed`로 기록합니다. `SystemExit`·`KeyboardInterrupt`는 정리 후 다시 던집니다. 로컬 실행의 Ctrl-C는 `cli.py::main`이 받아 `INTERRUPTED_ERROR`인 `failed` 결과를 JSON으로 출력하고 1로 끝냅니다.
4. `finally`에서 지표를 보내고, AWS 안에서 `TOPIC_ARN`이 있으면 성공·실패 모두 SNS로 알리며(`format_result_notification`), Batch 실행이면 결과 매니페스트를 저장합니다.

worker는 SIGTERM을 `SystemExit`로 바꿔(`cli.py::_raise_on_sigterm`) Batch 시간 초과나 취소, 호스트 종료 때도 4단계가 실행되게 합니다. 파이프라인은 `asyncio.run` 대신 `runtime.py::run_coroutine`으로 돌려, 중단 시 작업자 스레드에서 진행 중인 Bedrock 호출을 기다리지 않고 바로 정리 단계로 갑니다(Batch는 SIGTERM 약 30초 뒤 SIGKILL). 이때는 매니페스트만 저장하고(`INTERRUPTED_ERROR`) SNS 실패 알림은 보내지 않습니다. 정상 종료는 인터프리터가 남은 작업자 스레드를 기다리므로, worker는 정리 뒤 `os._exit(143)`으로 바로 끝납니다(`cli.py::_work`). 호스트가 사라졌으면 Batch가 재시도하고, 최종 실패면 EventBridge 규칙이 알립니다.

`run_job` 밖의 실패(잘못된 요청 JSON, 런타임 모듈 import 실패 등)는 `cli.py::_work`가 받아 `runtime.py::record_unstarted_run`으로 알림과 매니페스트를 남깁니다. worker는 실패 알림을 보낸 뒤 로컬 CLI와 다른 종료 코드 10(실패)·11(거부)(`APP_REPORTED_EXIT_CODES`)로 끝납니다. 1·2는 worker 코드가 돌기 전의 Python 오류·인자 오류와 겹치기 때문입니다. 실패 알림을 보내지 못하면(`run_job` 안이든 밖이든, `JobResult.notification_failed`) 4로 끝나 EventBridge 규칙이 대신 알립니다.

## 3. 리뷰 워크플로

`explainer.py::PaperReviewer.run`은 계획 → 보강 → 집필·검수 → 편집 → 마무리 다섯 단계입니다. 독자는 일반 DS/MLE로 고정하며(`prompts/base.py`의 `READER_PROFILE`), 모든 단계가 같은 독자 정의를 공유합니다.

### 3.1 계획

- 입력: 논문 전체를 문장 단위로 번호 매긴 텍스트(`review_plan.py::number_sentences`), 그림 목록(`F<n>`과 원문 마커), 참고문헌 목록(`R<n>`), 코드베이스 요약, 번역 지침.
- 모델: `paper_planning_model_id`, effort는 `planning_effort`(기본 `high`).
- 출력: `<plan>` 안의 JSON을 `review_plan.py::ReviewPlan`으로 검증합니다. 해석에 실패하면 오류를 붙여 한 번 더 요청하고, 그래도 실패하면 `PlanError`입니다.

계획의 절(`PlannedSection`)마다 제목, 목적, 원문 문장 범위, 반드시 다룰 항목(`must_cover`), 난이도(`easy`/`medium`/`hard`)와 근거, 중요도(`context`/`supporting`/`core`), 선수 개념(참고문헌 연결 가능), 이 절이 맡는 개념, 그림, 코드 필요 여부가 있습니다. 계획 전체에는 용어집, 서사 흐름, 제외 구간이 붙습니다.

`review_plan.py::normalize_plan`이 계획을 코드로 보정합니다.

- 문장 범위를 논문 길이에 맞게 자릅니다.
- 어느 절에도, 제외 구간에도 속하지 않은 3문장 이상 구간은 가장 가까운 앞 절에 붙입니다. 원문이 조용히 빠지지 않습니다.
- 그림 하나는 한 절에만 배정하고, 모르는 ID는 버립니다.
- 개념 하나는 한 절만 맡습니다. 다른 절은 그 개념을 참조만 합니다.
- `max_sections`를 넘는 절은 마지막 절로 합칩니다.
- 목표 분량은 `review_plan.py::target_words`가 항목 난이도 기본값(쉬움 60, 중간 200, 어려움 450단어 상당)의 합에 중요도 계수(0.8/1.0/1.3)를 곱해 정하며, 절마다 최소 250단어입니다.
- 리뷰 전체 예산은 `7000 × (1 + 0.6 × 어려운 항목 비율)`입니다(`review_word_budget`). 넘치면 쉬움·중간 항목만 한 비율로 줄이고(최대 절반까지), 어려운 항목은 그래도 예산의 1.3배를 넘을 때만 1.3배 쪽으로 줄이되 기본값의 0.8배(항목당 360단어) 아래로는 줄이지 않으므로 어려운 논문은 예산을 넘을 수 있습니다.

### 3.2 보강

LLM 라우팅 호출 없이 계획을 따릅니다(`PaperReviewer._enrich`). 참고문헌에 연결된 선수 개념만 `CitationSummarizer.summarize`로 요약하고, `needs_code`인 절만 `CodeRetriever.search_similar_code`로 코드를 찾습니다. 실패한 조회와 실패 표시가 붙은 요약은 버리며, 보강 실패가 실행을 멈추지 않습니다.

### 3.3 집필과 검수

집필(`_write`)은 절마다 다음을 받습니다.

- 전체 개요(`render_outline`)와 해당 절 지시서(`render_brief`)
- 그 절의 원문(다른 절에 배정된 그림 마커는 제거, `section_source`)
- 보강 자료, 용어집, 다른 절이 맡은 개념(`concepts_owned_elsewhere`), 지금까지 쓴 절 전체

쉬운 항목은 한두 문장으로, 어려운 항목은 직관 → `$$` 수식 → 작은 예제 → 그림 순으로 씁니다. 길이 때문에 끊을 때만 `has_more`로 이어 쓰며 최대 `MAX_CONTINUATIONS`(4)번입니다.

검수(`_check`)는 지시서, 원문, 측정한 분량과 목표 분량을 보고 0–100점과 빠진 항목(`missing_items`), 반드시 고칠 결함(`must_fix`: 반복, 다시 계산하면 틀리는 수치, 근거 없는 주장·그림 설명), 개선 의견을 돌려줍니다. 배점은 누락 없음 30, 깊이 25, 원문 충실도 25, 흐름 20이며, 빠진 항목이 하나라도 있으면 점수 상한은 69입니다. 코드가 찾은 규칙 위반(`markdown_lint.py::writing_issues`: 원 논문 번호 참조, 영어 어간+한국어 어미, 글 자체 지칭, 단일 `$` 수식, 그대로 옮긴 그림 표식, "(미검증)" 같은 독자용 검증 유보 문구)도 `must_fix`에 더합니다. 절은 점수가 `min_quality_score` 이상이고 빠진 항목과 `must_fix`가 없을 때만 통과하며, 그렇지 않으면 지금까지 가장 좋은 초안과 지적을 함께 넘겨 고쳐 쓰게 합니다. 초안 순위는 통과 여부 → 게시 가능 여부(게시를 막는 오류 없음) → 점수 → 막는 결함 수 순이고(`review_plan.py::draft_rank`, 가이드 루프와 공유하되 가이드는 평가자의 차단 문제 수를 점수보다 먼저 봅니다), 기각된 수정본의 문제도 다음 지적에 붙입니다. 초안 수는 `max_synthesis_attempts`가 상한이고, 절은 계획된 `##` 제목으로 시작하게 맞춥니다(`ensure_section_heading`).

### 3.4 전체 편집

절을 모두 이은 글 전체를 한 번 편집합니다(`_edit`, 기획 모델 사용). 입력은 글, 논문 원문, 계획의 핵심 항목 목록, 용어집입니다. 출력은 `<edit><find>…</find><replace>…</replace></edit>` 블록 목록이고(LaTeX와 `&`를 그대로 담기 위해 JSON이 아닌 태그 형식), `review_plan.py::apply_edits`가 다음 조건에서만 적용합니다.

- `find`가 현재 글에 정확히 한 번 나와야 합니다.
- `## ` 헤딩은 지우거나 바꾸거나 강등할 수 없습니다. 중복된 `###`·`####` 소절은 통째로 지울 수 있습니다.
- 그림, 디스플레이 수식, 표의 유일한 사본을 지우는 편집은 거부합니다.
- 편집은 `MAX_EDITS`(40)개까지입니다.

편집자가 누락 항목을 보고하거나 적용되지 못한 편집이 있으면 그 목록을 붙여 한 번 더 편집하며, 횟수는 `max_edit_rounds`(기본 2, 0이면 편집 생략)가 상한입니다.

### 3.5 마무리

`PaperFinalizationPrompt`가 편집된 글에서 세 가지를 만듭니다.

- `subtitle`: 제목의 출력 언어 번역
- `key_takeaways`: `## TL;DR` 아래 `###` 질문 4개, 질문마다 2–4문장 또는 2–4개 글머리표와 핵심 수치
- `closing`: `## 한계와 시사점`(강점, 구체적 한계, 적용 시점, 확실한 후속 연구)

`run()`은 `ReviewText(body, key_takeaways, closing, subtitle)`을 돌려주고, `formatting.py::format_review`가 TL;DR → 본문 → 한계와 시사점 → 참고문헌 순서로 조립합니다.

### 3.6 호출 수와 한계

8개 절 논문에서 계획 1회, 절마다 집필·검수 약 2–3회, 편집 1–2회, 마무리 1회입니다. 계획과 편집은 논문 전체를 입력으로 받고, 컨텍스트를 넘으면 `fit_text`가 뒷부분을 잘라냅니다. 이때도 미배정 구간 보정이 남은 문장을 절에 붙입니다. 모든 단계는 `max_total_tokens` 예산 가드를 거칩니다.

## 4. 요약과 기술 가이드 파이프라인

### 4.1 요약

`summarizer.py::PaperSummarizer.summarize`는 파싱한 논문 전체(컨텍스트에 맞게 자름)와 코드베이스 요약으로 한 번 호출하고, 한국어 요약이 너무 길면 줄이는 호출을 한 번 더 합니다(`_shorten_if_long`, 구조가 깨지면 원문 유지). 출력 태그는 `subtitle`, `summary`, `urls`입니다. 요약은 `> **요약:**` 인용문과 이모지 `##` 절 다섯 개(왜 필요한가, 해결책, 구현, 결과, 의의와 한계)로 이루어지고, 실증 논문이면 결과 절에 무게를 둡니다. `formatting.py::summary_with_tldr`가 앞의 인용문을 `## TL;DR` 절로 바꿔 블로그 규칙에 맞춥니다.

### 4.2 기술 가이드

`tech_guide.py::TechGuideGenerator.generate` 순서입니다.

1. 문서 URL(위치 인자, 0개 이상)과 범위 안 하위 페이지를 수집합니다(`web_research.py::WebResearcher.research`). 참고 자료 URL(`--reference`를 반복해 0개 이상, `JobRequest.reference_urls`: 논문, 설계 문서, 심층 분석 글, 발표 페이지)은 하위 페이지 없이 가져와 참고 자료로 표시합니다(`PageContent.user_reference`). 공식 문서의 호스트와 버전(`OfficialDocs`)은 문서 URL로만 정하므로 참고 자료는 공식 문서가 되지 않습니다. 단, 요청한 문서와 같은 호스트·호환 버전의 페이지는 참고 자료로 넘겨도 공식 문서로 다루고, 문서 URL과 같은 참고 자료 URL은 문서로만 다루고 출처 목록에도 한 번만 넣습니다. 같은 페이지를 여러 표기로 주거나(대소문자, 끝 `/`), 같은 arXiv 논문을 초록·PDF·HTML·버전별 링크로 주어도 한 번만 가져옵니다(`web_research.py::page_key`). 리디렉션으로 이미 가진 페이지에 도착한 참고 자료와 검색 결과도 다시 넣지 않습니다. 읽지 못한 참고 자료는 PR 체크리스트에 올립니다. 문서 URL 없이 참고 자료만 주면 공식 문서가 없는 가이드가 되며, 그 사실을 PR 체크리스트에 올리고 절 제목의 공식 사이트 조회를 하지 않습니다. arXiv 논문 링크(초록·PDF·HTML)는 검색 결과든 입력이든 논문의 HTML 렌더링(`arxiv.org/html/<id>`)으로 읽습니다(`web_research.py::arxiv_paper_urls`). 렌더링이 없거나(404) 변환 실패 안내문뿐이면(읽을 수 있는 글자가 파서의 `MIN_PARSED_TEXT_CHARS` 미만) 초록 페이지를 한 번 읽습니다. 429·5xx·시간 초과·연결 오류는 대체 페이지로 넘어가지 않고 가져오기 실패로 기록하므로, 모든 입력이 이런 이유로 실패하면 `SourceFetchError`(재시도 가능한 실패)가 됩니다. 이 계층은 재시도하지 않습니다. arXiv로 가는 모든 요청은 arXiv API 클라이언트와 같은 프로세스 전역 제한기(`rate_limiter.py::ARXIV_RATE_LIMITER`, 3초에 1회)를 거치고, 429를 받으면 이후 arXiv 요청을 `Retry-After`(없으면 30초)만큼 미룹니다. 논문의 SVG 그림(`<object type="image/svg+xml">`)도 `<img>`처럼 이미지 목록에 넣고, 본문 텍스트의 이미지 마커는 목록에 남은 이미지만 남깁니다. `--reference`는 HTML 페이지나 arXiv 링크여야 하며, arXiv가 아닌 PDF는 읽지 않습니다.
2. 관련성 판정(`TechGuideRelevancePrompt`)을 문서 페이지와 그 하위 페이지만으로(문서 URL이 없으면 참고 자료로) 먼저 합니다. 통과하지 못하면 `NotTechnicalContentError`로 거부하므로 검색 비용이 들지 않습니다. 같은 호출이 문서가 다루는 명령·패키지 이름(`<commands>`)도 돌려주며, 시드 페이지에 단어로 나오는 이름만 씁니다. 하나도 없으면 사이트 호스트 이름(`docs.vllm.ai` → `vllm`)을 씁니다(`web_research.py::product_names`).
3. 검색 제공자가 있고 검색어가 주어지지 않았으면 계획 단계가 문서와 참고 자료를 보고 검색어를 만들고(`auto_research`, `max_research_queries`, 기본 6개), 상위 결과 페이지를 말뭉치에 넣습니다(`fetch_top_results`, 기본 12). 결과는 모든 검색어의 i번째 결과를 i+1번째보다 먼저 고르므로, 기본값에서는 검색어마다 상위 두 결과(이미 가진 페이지는 건너뜀)가 들어옵니다. 계획 프롬프트는 검색어를 사용법(공식 문서), 내부 구조와 설계(논문, 설계 문서, 아키텍처 개요, 메인테이너의 심층 분석 글, 소스 해설), 운영 아키텍처와 연동(배포 패턴, 확장, 관측성, 장애 양상, 주변 시스템과의 인터페이스) 세 의도에 약 3분의 1씩 나누라고 지시하지만, 실제 배분은 모델이 정합니다. 참고 자료가 이미 설명한 내용은 다시 찾지 않습니다. 계획 단계와 관련성 판정이 보는 출처 요약(6,000자)은 참고 자료와 나머지 페이지가 각각 절반 이상을 쓸 수 있게 나눕니다. 계획 단계의 주제는 검색과 개요 입력에 쓰고, 가이드 제목은 개요 호출이 계획한 범위에 맞춰 돌려주는 `<title>`을 씁니다(없으면 계획 주제).
4. 개요를 만들고(최대 `max_sections`개 절, 기본 14, 3–30) 절마다 깊이(deep/standard/brief)를 정합니다. 분량 기준은 가이드 전체 약 35,000–55,000자(한국어 기준, 코드 제외), deep 절 약 5,000자, standard 약 2,500자, brief 1,000자 이하입니다(`prompts/tech_guide.py::LENGTH_BUDGET`). 개요는 출처가 뒷받침하는 범위에서 폭과 깊이를 함께 다룹니다. 도구가 푸는 문제와 핵심 개념, 핵심 메커니즘의 동작 방식과 설계 이유, 주요 기능별 의미(푸는 문제, 동작, 장단점, 쓸 때), 설치와 사용법, 더 큰 시스템에의 통합(위치, 인터페이스, 확장, 관측성, 장애 양상, 권고 형태의 설계 조언)입니다. 쉬운 내용은 짧게, 어려운 메커니즘은 배경부터 깊게 쓰며, 설치는 한 절 이하로 두고 플랫폼별 차이는 표나 공식 페이지 링크로 줄입니다. getting-started 유형도 사용법을 앞에 두되 내부 구조와 통합 절을 뒤에 두며, 어려운 메커니즘에는 deep 절을 줍니다. 설치·사용·설정 절은 공식 문서에, 개념·내부 구조·통합 절은 공식 페이지가 없으면 참고 자료나 서드파티 설명 자료에 기대며, `<section_plan>`에 그 자료를 적습니다. 개요 호출은 절마다 그 절을 뒷받침하는 말뭉치 페이지 URL과 그 절이 맡는 개념도 돌려줍니다(`<section_plan>`). URL은 말뭉치에 있는 것만 남기고, 계획의 번호는 개요의 번호 매긴 줄과 맞춥니다(`guide_authority.py::parse_section_plan`). 계획 줄이 없거나 공식 문서 페이지가 하나도 없는 절은 집필 전에 절 제목으로 공식 사이트를 도메인 제한 검색합니다. 검색 결과로 찾은 페이지만 그 절의 공식 출처로 치며, 검색 백엔드가 없거나 찾지 못하면 집필자에게 출처가 보여 주는 내용만 쓰라고 알립니다. 계획이 그 절에 고른 페이지와 제목 조회로 찾은 페이지는 계획 순서대로 절의 근거 자료 맨 앞에 두고, 각각 근거 자료 상한의 같은 몫을 먼저 받아 긴 논문이 공식 페이지를 밀어내지 않습니다. 남은 자리는 관련도 순으로, 잘린 우선 페이지부터 나눕니다(`ResearchCorpus.relevant_text`). 각 집필 호출에는 다른 절이 맡은 개념 목록(`owned_elsewhere`)을 넘겨 다시 정의하지 않게 합니다.
5. 절마다 작성 → 평가 → 기준 미달 시 수정(`max_revision_attempts`, 기본 1회) 순서로 진행합니다. 평가자는 절의 근거 자료와 전체 목차를 보고 점수(0–100)와 차단 문제(`<blocking_issues>`: 사실 오류, 절 안의 모순, 출처 권한이 요구하는 근거가 없는 플래그·기본값, 어느 출처도 뒷받침하지 않는 내부 구조·아키텍처 주장, 다른 절을 이름이나 번호로 가리키는 표현)를 돌려줍니다. 점수가 기준 이상이고 차단 문제와 코드로 찾은 규칙 위반이 없어야 통과하며, 차단 문제는 수정 지시에 반드시 고칠 항목으로 들어갑니다. 수정본은 원래 초안보다 순위가 높을 때만 남기며, 순위는 통과 여부, 게시 가능 여부, 평가자의 차단 문제 수, 점수, 규칙 위반 수 순서로 비교합니다. 모든 절을 쓴 뒤 사실 확인(`verify_grounding`)을 합니다. 공식 페이지에 없는 CLI 플래그·Python 인자(`gpu_memory_utilization=`)·메트릭 이름은 같은 공식 문서(같은 호스트, 호환되는 버전)의 레퍼런스 페이지로 확인합니다. 집필 전에는 서드파티 페이지에만 나오는 식별자와 공식 출처가 없는 절의 제목을, 사실 확인 전에는 초안에 새로 나온 식별자를 찾으며, 모든 조회를 합쳐 검색 3번, 6쪽까지 씁니다(`WebResearcher.add_official_reference`, 식별자 조회는 검색 뒤에 식별자와 맞는 경로의 링크도 씁니다). 집필 전 조회(서드파티 식별자와 절 제목)는 합쳐서 검색 2번, 4쪽까지만 써서, 사실 확인에 검색 1번과 2쪽 이상이 남습니다. 그중 서드파티 식별자 조회는 절반(검색 1번, 2쪽)까지만 써서 절 제목 조회의 몫을 남깁니다. 그래도 확인되지 않은 식별자는 PR 체크리스트에 올립니다. 출처 목록에는 요청한 페이지(문서와 참고 자료)와 같은 공식 문서(호환되는 버전)의 페이지를 넣고, 그 밖의 페이지는 어느 절의 근거 자료에 실제로 들어갔거나 본문이 링크하거나 이미지를 가져온 경우에만 넣습니다. 레퍼런스 페이지는 본문에 남은 식별자를 낱말 단위로 담은 경우에만 넣고, 이전 버전 문서는 넣지 않습니다(`guide_authority.py::used_source_urls`). 각 절의 호출에는 말뭉치 전체가 아니라 절 제목·설명과 가장 관련 있는 페이지만 약 6만 자까지 넣고(`ResearchCorpus.relevant_text`), 앞선 절은 직전 두 절 전문과 그 이전 절의 요약만 넘깁니다.

출처 권한은 집필·평가·사실 확인 프롬프트가 공유합니다(`prompts/tech_guide.py::SOURCE_AUTHORITY`). 프롬프트의 각 출처는 공식 문서(official documentation), 이전 버전 문서(older docs version), 참고 자료(reference material), 서드파티(third-party)로 표시됩니다(`web_research.py::source_label`). 명령, 플래그, API 이름, 기본값, 설정 키, 환경 변수, 메트릭 이름, 버전·플랫폼 지원은 공식 문서가 정합니다. 참고 자료와 서드파티 페이지(논문, 설계 문서, 메인테이너 글, 엔지니어링 글, 소스 코드)는 개념, 내부 구조, 설계 이유, 장단점, 아키텍처·연동 패턴의 근거가 될 수 있고, 1차 자료가 일반 튜토리얼보다 우선하며 마케팅 글은 근거가 되지 않습니다. 현재 동작을 두고 공식 문서와 어긋나면 공식 문서가 이기고, 이전 버전 문서는 현재 동작을 정하지 않습니다. 근거 자료에 공식 페이지가 있으면 비공식 페이지에만 나오는 명령·플래그·기본값·설정 키는 쓰지 않고, 공식 페이지가 없으면 그 값을 보여 주는 페이지가 있어야 합니다. 평가자는 플래그·기본값에 이 근거를, 개념·내부 구조·아키텍처 주장에는 어느 출처든 근거나 교과서 수준 배경을 요구하며, 근거 없는 내부 구조·아키텍처 주장은 차단 문제입니다. 사실 확인은 출처가 뒷받침하는 개념 설명을 공식 페이지에 없다는 이유로 지우지 않고, 식별자는 계속 공식 문서로 확인합니다. 본문은 사실의 출처를 서술하지 않고, 의견, 벤치마크, 메커니즘이 나온 논문·설계 문서만 한 번 출처를 밝힙니다. 사실 확인이 절을 4분의 1 넘게 줄이면 로그(INFO)에 남깁니다. 참고 자료는 출처 목록에 항상 들어갑니다.

가이드 파이프라인(`guide_pipeline.py`)은 완성된 가이드를 `TechGuideGenerator.classify`로 분류해 블로그 주제 하나, 통제 태그, 세부 태그, `seo_title` 후보를 받고, `formatting.py::build_post_meta`가 spec에 맞게 확정합니다. 분류는 개요가 아니라 완성된 본문의 헤딩과 문단 첫 문장으로 하므로 본문에 없는 내용을 태그로 달지 않습니다. 목차는 블로그 레이아웃이 만들므로 본문에 따로 넣지 않습니다. 문서 사이트의 이미지는 외부 링크로 남기고 검증에서 warning으로 표시합니다.

## 5. 데이터 수집과 파싱

### 5.1 논문 소스

`paper_source.py::resolve_paper_source`가 입력을 판별합니다.

- arXiv ID(신규형, `hep-th/9901001` 같은 구형, `arXiv:` 접두어, 버전 포함)와 arXiv URL은 `ArxivSource`가 됩니다. 메타데이터는 arXiv API에서 받습니다.
- 그 밖의 URL은 `PdfUrlSource`입니다. SSRF 검사 후 content-type과 `%PDF-` 매직 바이트로 실제 PDF인지 확인하며, 아니면 `NotAPdfError`입니다. 제목은 본문에서 추출한 제목으로 대체합니다(`title_is_fallback`).

### 5.2 파싱

- arXiv 소스는 arXiv HTML(`arxiv.org/html`, 실패 시 ar5iv)을 먼저 씁니다(`parser.py::HTMLRichParser`). 실패하거나 내용이 비면 PDF로 넘어갑니다.
- PDF는 Upstage Document Parse(`parser.py::PDFParser`)로 파싱하고, 실패하면 PyMuPDF로 텍스트만 추출하고(`Content.text_only`), 그림과 표가 없다는 메모를 PR 체크리스트에 올립니다. Upstage 호출은 재시도하지 않습니다. 빈 응답은 캐시하지 않습니다.
- 그림은 비전 모델(`figure_analysis_model_id`)로 분석하고, `parser.py::attach_figure_context`가 본문의 이미지 마커에 경로와 분석을 붙입니다. 짝짓기는 순서가 아니라 요소 ID와 파일명으로 합니다.

### 5.3 HTML을 텍스트로

`utils/helpers.py::extract_text_from_html`이 HTML 구조를 유지한 텍스트를 만듭니다.

- 헤딩은 Markdown 헤딩, `<pre>`는 펜스 코드, 블록 요소 사이는 빈 줄입니다. nav, header, footer, aside는 버립니다.
- 수식은 MathML `alttext`(또는 TeX 주석)를 `$$…$$`로 옮기고, 디스플레이 수식은 앞뒤 빈 줄을 둔 별도 줄로 둡니다.
- 표는 `utils/helpers.py::table_to_markdown`이 완성된 GFM 표로 만듭니다. colspan·rowspan을 펼쳐 격자를 만들고, 여러 줄 헤더(thead 또는 숫자 없는 앞쪽 행, 최대 3행)는 열마다 공백으로 합쳐 한 행으로 만들며, 빈 칸은 `-`, 셀 안의 `|`는 `\|`로 씁니다. 캡션은 표 위에 둡니다.

### 5.4 속성과 인용

`content_extractor.py::ContentExtractor`가 두 가지를 추출합니다.

- 속성(`extract_attributes`): 제목, 저자, 발행 조직(`affiliation`), 규칙 파일의 주제 중 하나(`category`), 통제 태그, 세부 키워드, `seo_title`. 키워드 재사용 후보로 블로그 글에 이미 쓰인 태그를 넘깁니다(`runtime.py::RunContext.require_blog_tags`).
- 인용(`extract_citations`, 리뷰만): 참고문헌을 페이지 단위로 추출하며 최대 `MAX_CITATION_PAGES`(20)회입니다.

인용 요약(`citation_summarizer.py::CitationSummarizer`)은 초록을 우선합니다. Crossref, OpenAlex(선택 API 키 `OPENALEX_API_KEY`를 `Authorization: Bearer`로 보냄. 키가 없으면 IP당 하루 약 100회 검색), Semantic Scholar(API 키가 있을 때만), arXiv 순서로 메타데이터를 찾고(`citation_metadata.py`, 속도 제한 포함, `Retry-After`가 60초를 넘으면 그 실행 동안 해당 제공자를 건너뜀), 제목 유사도 0.9 이상이고 버전 표기와 단어가 바뀌지 않은 경우(하이픈·영국식 철자 차이는 같은 단어)만 같은 논문으로 인정합니다(Crossref는 상위 5건 중 제목이 일치하는 레코드 가운데 제목 유사도가 가장 높은 것, OpenAlex는 상위 5건 중 첫 일치 결과). 제목의 내용어를 하나도 담지 않은 초록은 다른 논문의 것으로 보고 버리며, OpenAlex의 arXiv 레코드 초록은 믿지 않고 DOI의 arXiv ID로 arXiv에서 받습니다. 맞지 않는 초록은 요약하지 않으며, 초록도 본문도 찾지 못한 인용은 리뷰 노트(PR 체크리스트)에 올립니다. `prefer_full_text: true`면 arXiv 전문을 받습니다.

### 5.5 코드 저장소

`code_retriever.py::CodeRetriever`는 저장소를 얕게(depth 1, 단일 브랜치, 심볼릭 링크 제외) 시간 제한을 두고 clone합니다. 가상환경·벤더 디렉터리와 저장소 밖을 가리키는 파일은 건너뛰고, Python 파일을 청크로 나눠 Titan 임베딩 FAISS 인덱스를 만듭니다. 인덱스는 저장소 집합별로 구분하고, 코드베이스 요약을 함께 만듭니다. 코드 분석 실패는 "코드 없음"으로 처리합니다.

### 5.6 웹 조사

`web_research.py::WebResearcher`는 시드 경로 접두어(끝 `/` 포함) 안의 하위 페이지를 최대 8개 따라가며(확장자 없는 말단 페이지면 상위 디렉터리를 범위로 씀), 본문 영역의 링크를 우선합니다. 리디렉션 후 최종 URL 기준으로 링크를 해석하고, 5MB를 넘는 페이지는 건너뜁니다. 검색은 Tavily, 없으면 Brave, 둘 다 없으면 생략합니다(`guide_pipeline.py::build_search_provider`). 페이지 본문·제목·URL의 프롬프트 경계 태그는 무력화합니다(`neutralize_prompt_tags`).

### 5.7 SSRF 방지

외부 URL은 `url_guard.py::assert_url_is_public`이 스킴과 해석된 IP를 검사하며, 전역 주소가 아닌 IP(사설, 루프백, 링크 로컬, CGNAT, 메타데이터 대역)는 모두 막습니다. `pinned_transport.py`는 검사한 IP에 연결을 고정해 DNS 재바인딩을 막고, 리디렉션마다 다시 검사합니다.

## 6. 블로그 규칙과 검증

### 6.1 규칙 데이터

정본은 블로그 저장소의 `_data/post_conventions.yml`이고, `assets/post_conventions.yml`은 같은 내용의 사본입니다. `post_conventions.py::PostConventions`가 읽는 항목은 다음과 같습니다.

| 키 | 의미 |
| --- | --- |
| `types` | `categories[0]` 값. `paper: true`인 유형은 논문 글(`paper_author` 필요), `cover`가 있으면 그 표지 |
| `topics` | `categories[1]` 값과 주제별 표지. 표지가 없으면 `/assets/images/default.jpg` |
| `forbidden_keys` | front matter 금지 키(`author`, 블로그 플러그인이 정하는 `last_modified_at`) |
| `tags.controlled` | 통제 태그 목록. 관련 글 연결의 기준 |
| `tags.lowercase_words` | 태그 안에서 소문자로 쓰는 기능어 |
| `tags.aliases` | 퇴역 태그 → 대체 태그 |
| `languages` | front matter `lang`에 쓸 수 있는 언어(기본 `ko`, `en`) |

블로그 파일에 아직 없는 생성기 쪽 한도는 코드 기본값을 씁니다. 세부 태그 검증 상한은 10개(`TagSpec.max_specific`)이고 생성 목표는 5개(`GENERATED_SPECIFIC_TAGS`)입니다. 통제 태그는 생성 시 최대 3개(`max_controlled`)이며, `seo_title` 기준 길이는 70자입니다. 블로그 파일의 `math` 항목(정의 매크로 목록)은 읽지 않습니다. YAML 1.1 파서가 `on`을 불리언으로 읽는 문제는 `TagSpec._unbool`이 되돌립니다.

실행 중에는 GitHub 연동이 켜져 있으면 블로그 `base_branch`의 정본을 읽고(`post_conventions.py::load_conventions`), 읽지 못하면 사본을 쓰며 사본과 다르면 경고합니다. CI의 불일치 검사는 둘이 다르면 실패합니다(12절).

### 6.2 결정적 후처리

검증 전에 코드가 다음을 고칩니다.

- **front matter 조립**(`formatting.py`): `author`는 쓰지 않습니다. 논문 글은 `paper_author`(발행 조직)와 `subtitle`(모델이 쓴 부제 → 본문 첫 H1 → TL;DR 첫 문장 순, 제목과 같으면 다음 후보)을 두고, 논문에 DOI가 있으면 블로그가 인용 수를 조회하는 `paper_doi`를 둡니다. `seo_title`은 제목이 70자를 넘을 때만 넣으며, 모델 후보가 없거나 길면 콜론 앞부분이나 단어 경계에서 자릅니다(`shorten_title`). 카테고리 주제와 표지는 spec에서 정하고(`resolve_topic`, `PostConventions.cover_for`), `use_math`는 `$$`가 있을 때만 켭니다. `date`는 논문 발행일입니다.
- **태그**(`tags.py::normalize_tags`): 공백·`_`는 하이픈, `&`는 `and`, `/`와 따옴표는 삭제, 연속 하이픈 축약, 첫 단어가 아닌 기능어는 소문자로 바꿉니다. 그다음 별칭을 대체 태그로 바꾸고, 대소문자를 무시해 중복을 없애며, 제목 전체·콜론 앞·콜론 뒤와 같은 태그를 지웁니다. 통제 태그는 닫힌 목록만 남기고, 세부 태그 중 통제 태그와 같은 것은 통제 쪽으로 옮깁니다. 발행 순서는 세부 태그 다음 통제 태그입니다.
- **수식**(`markdown_math.py::normalize_math`): `\(…\)`·`\\(…\\)`는 `$$…$$`로, `\[…\]`는 앞뒤 빈 줄을 둔 디스플레이 `$$`로 바꿉니다. `$$` 안의 `\_`는 `\text{}`·`\texttt{}` 안을 빼고 `_`로 되돌리고, 한 줄 전체가 `$$`인 수식 앞뒤에 빈 줄을 넣습니다(표·목록 제외). 확률·정책 인자 안의 조건부 `\|`·`\Vert`는 `\mid`로, 합의 상한으로 쓰인 `\lVert x \rVert`(길이)는 `\lvert x \rvert`로 바꿉니다(노름과 KL 발산의 `\|`는 그대로). 코드와 단일 `$`는 건드리지 않습니다.
- **Markdown**(`markdown_lint.py::lint_markdown`): 헤딩 앞 빈 줄, 생성용 envelope 태그 제거, H1을 H2로 올리고 한 단계 넘게 깊어진 헤딩을 앞 헤딩 바로 아래 단계로 올리기, 네 번째부터의 콜아웃(`> **라벨:** …`)을 일반 문단으로 풀기, 가짜 HTML 태그 이스케이프를 처리합니다. `drop_unknown_images`는 논문 그림이 아닌 이미지를 지우고, `fill_image_alts`는 빈 alt를 그림 캡션으로 채우며, 한국어 글의 영어 alt는 가장 가까운 헤딩으로 바꿉니다. 원 논문 번호 참조 같은 규칙 위반(`writing_issues`)은 게시 직전 본문에서 다시 찾아 PR 체크리스트에 올립니다.

### 6.3 검증 규칙

`post_validator.py::validate_post(markdown, spec, kind)`는 순수 함수이며 `Violation(level, rule, line, message)` 목록을 돌려줍니다. 코드 블록 안은 검사하지 않습니다. 논문 글의 표 행 수치가 원문의 한 줄이나 한 표 안에 함께 나오지 않으면(`unsourced_table_numbers`) PR 체크리스트에 메모로 올립니다.

| 규칙 | 수준 | 조건 |
| --- | --- | --- |
| `front-matter` | error | front matter가 없거나 YAML이 아님 |
| `forbidden-key` | error | `forbidden_keys`의 키가 있음(`author`, 그리고 블로그가 파일명 날짜로 채우는 `last_modified_at`) |
| `date` | error | front matter `date`가 UTC 기준 미래(Jekyll이 글을 건너뜀), 파일명이 `YYYY-MM-DD-`로 시작하지 않거나 `date:`가 파일명 날짜보다 늦음(UTC) |
| `title` | error | 제목이 비어 있음 |
| `categories` | error | `[<유형>, <주제>]`가 아님, 모르는 유형·주제, 산출물 종류와 유형의 `paper` 여부 불일치 |
| `cover` | error | `cover_for(유형, 주제)`와 다름 |
| `paper-author` | error | 논문 글에서 비었거나 `et al.`이 있음, 가이드에 있음 |
| `subtitle` | error | 논문 글에서 비었거나 제목과 같음 |
| `seo-title` | error | 제목이 70자 초과인데 없음, 또는 `seo_title`이 70자 초과 |
| `translation` | error | `lang`이 spec의 `languages`(`ko`/`en`)가 아님, 번역본에 `translation_id`가 없음, `translation_id`가 있는데 `lang`이 없음 |
| `tags` | error | 비어 있음, 정규화되지 않은 태그, 퇴역 태그, 통제 태그 없음, 세부 태그 상한 초과 |
| `tags` | warning | 통제 태그 3개 초과(논문에 없는 태그 구절은 `paper_pipeline.py::_tag_notes`가 PR 메모로 남김) |
| `headings` | error | 본문 H1, 헤딩 레벨 건너뜀, 논문 글이 `## TL;DR`로 시작하지 않음, TL;DR이 비어 있음, 리뷰 TL;DR에 `###` 질문이 없음 |
| `use-math` | error | `use_math`와 `$$` 존재 여부가 다름 |
| `math` | error | kramdown이 글로 보여 주는 줄(인용문 안 포함)에 블로그 `script/lint-posts.rb`와 같은 규칙. 블로그 lint가 코드 줄까지 읽는 차이는 블로그 PR CI가 판단: `\(…\)`·`\\[…\\]` 잔존, 단일 `$` 옆의 TeX(`}$`, `$\명령`), 문단 안에서 짝 없는 `$$`, 닫히지 않은 여러 줄 `$$` 블록, 디스플레이 `$$` 앞뒤 빈 줄 없음, `$$` 안 `\_`(`\text` 밖), 인라인 수식 안 한글, 인라인 수식이 여럿인 줄이나 표 행의 수식 안 맨 `|` |
| `table` | error | 구분선 셀 수가 헤더와 다름, 데이터 행 셀이 헤더보다 많음(적으면 kramdown이 채우는 그룹 라벨 행), 첫 열 외 빈 헤더, 첫 데이터 행이 하위 헤더(첫 셀이 비고 값 셀이 없음), 바로 위 행과 같은 행, 표 행의 `$$` 안 막대 기호(게시 단계가 `\vert`·`\Vert`로 바꿈) |
| `table` | warning | 같은 헤더 라벨이 바로 옆에 반복 |
| `image` | error | arXiv 계열 원격 이미지 잔존, `/assets/images/<파일>`이 아닌 로컬 경로 |
| `image` | warning | 그 밖의 외부 이미지 |
| `description` | warning/error | 첫 문단(검색 설명)에 수식(warning), front matter `description`에 `$$`·강조·백틱·링크·태그·`{{`(error) |
| `envelope` | error | 생성용 태그(`</explanation>` 등) 누출 |
| `image-alt` | warning | 한국어 글의 이미지 alt에 한글이 없음 |
| `list` | warning | 디스플레이 수식 뒤에서 번호 목록이 2 이상으로 다시 시작함(목록이 끊김) |
| `liquid` | error | `{% raw %}` 밖의 `{{ }}`·`{% %}`(코드 블록 포함; 게시 단계가 자동으로 감쌈) |
| `html` | error | 브라우저가 지우는 태그(`<think>`, `<s>`, `<input>`, 속성 없는 `<img>`; 게시 단계가 자동 이스케이프) |
| `code` | error | 닫히지 않은 코드 펜스, 코드 블록 언어 표시에 공백(게시 단계가 첫 단어만 남김) |

`Publisher._validate`가 이 함수를 호출합니다. error가 하나라도 있으면 저장·업로드·PR을 모두 하지 않고, 글과 위반 목록을 작업 폴더의 `<slug>.rejected.md`에 남기며 S3 버킷이 있으면 `<prefix>/rejected/<slug>.rejected.md`에도 올린 뒤 `PostValidationError`를 던집니다. warning은 `PublishRequest.validation_warnings`에 담겨 PR 본문의 `- [ ]` 체크리스트가 됩니다.

## 7. 발행

`publisher.py::Publisher.publish` 순서입니다.

1. 모델이 입력에서 그대로 옮긴 방어용 태그 표식(`prompt_safety.py::strip_defanging`)을 front matter와 본문 모두에서 지우고, 본문에 `markdown_lint.py::publish_body`를 적용합니다. 닫히지 않은 코드 블록 닫기, 콜아웃 3개 제한, `normalize_math`, `lint_markdown` 순서이며, 섹션 루프의 게시 차단 검사도 같은 함수를 씁니다.
2. `ImageLocalizer`가 이미지를 블로그 규칙에 맞게 옮깁니다. arXiv 계열 호스트(`arxiv.org`, `ar5iv.org`, `ar5iv.labs.arxiv.org`와 하위 도메인) 이미지는 URL 검사와 IP 고정 전송을 거쳐 내려받고(30초, 리디렉션 5회, 20MB 상한), PDF에서 뽑은 로컬 그림도 같은 방식으로 처리합니다. PNG·JPEG는 매직 바이트로 확인하고, SVG는 PNG로 래스터화합니다(`utils/images.py::svg_to_png`). SVG는 그림을 만든 쪽이 정하는 외부 입력이라, 외부 참조를 지우고 캔버스 크기를 4096px 안으로 고정한 뒤 시간·메모리를 제한한 자식 프로세스에서 렌더링합니다(동시 2개). 세부 규칙은 `utils/images.py`의 함수 docstring과 상수 주석에 있습니다. 그 밖의 형식은 거부합니다. 저장 이름은 첫 등장 순서대로 `<slug>-<n>.<ext>`이고, 본문 경로는 `/assets/images/<slug>-<n>.<ext>`로 바꿉니다. 파일은 `<work_dir>/blog-images/<slug>/`에 모읍니다. 같은 논문의 요약과 리뷰가 작업 폴더를 같이 쓰므로, 재실행은 자기 슬러그 폴더만 비우고 S3 업로드와 PR에도 자기 이미지만 넣습니다. 실패하면 `PostAssetError`입니다. 그 밖의 외부 이미지는 링크로 남깁니다.
3. 검증합니다(6.3).
4. `<work_dir>/<YYYY-MM-DD>-<slug>.md`로 저장합니다. 파일명 날짜는 게시일(UTC)이고, front matter `date`는 논문 발행 시각입니다(가이드는 생성 시각). Jekyll은 미래 날짜 글을 건너뛰므로 두 날짜 모두 UTC로 씁니다. `slugify`는 ASCII 슬러그를 만들고(80자 안에서 단어 경계로 자름), ASCII 글자가 없는 제목은 `untitled-<8자리 해시>`로, 글자·숫자가 하나도 없는 제목은 `untitled`로 바꿉니다.
5. S3 버킷이 있으면 글은 `<prefix>/posts/`, 이미지는 `<prefix>/assets/images/`에 올립니다. 이미지 하나라도 실패하면 오류입니다. `public_assets`가 켜져 있을 때만 public-read ACL을 씁니다.

`runtime.py::publish_artifact`는 `create_pr`이고 GitHub 연동이 켜져 있으면 `create_pull_request`를 호출합니다. PR 실패(`PublishError`)는 실행을 실패시키지 않고 `warnings`에 남깁니다.

`blog_sync.py::BlogBranchSync`는 clone 없이 GitHub API로 커밋합니다.

- 브랜치는 `<branch_prefix>/<branch_id>`로 고정됩니다(리뷰 `review-<id>`, 요약 `summary-<id>`, 가이드 `guide-<URL 해시>`). 같은 입력으로 다시 실행하면 같은 브랜치와 PR을 갱신합니다. 가이드 브랜치는 정규화한 입력 URL(문서와 참고 자료, 쿼리 포함, 프래그먼트 제외) 집합의 해시라 모델이 고른 제목과 무관합니다. 다만 다시 실행한 가이드의 제목이 바뀌면 열린 PR에 이전 글이 함께 남으므로 직접 지워야 합니다.
- 열린 PR이 있으면 새 커밋을 브랜치 head 위에 쌓아 PR에서 손으로 고친 다른 파일을 지킵니다. 열린 PR이 없으면(병합, squash 병합, 닫힘) 기준 브랜치에서 다시 시작합니다.
- blob SHA를 로컬에서 계산해 바뀐 파일만 올리고, 내용이 같으면 커밋하지 않습니다. 기준 브랜치가 이미 같은 글을 갖고 있으면 PR을 만들지 않습니다. 저장소 트리는 동기화마다 한 번씩 읽고, 목록이 잘렸으면(`truncated`) 실패합니다.
- 이 브랜치에만 있는 예전 글 사본과 그 글들의 번호 이미지 중 이번 실행이 만들지 않은 것은 지웁니다. 예전 사본은 슬러그가 같거나(게시일만 다름), front matter `date`(논문 발행 시각)와 종류가 같은 글(새 arXiv 버전에서 제목이 바뀐 이전 슬러그)입니다. 요약 슬러그는 `-summary`로 끝나므로 같은 논문의 리뷰와 요약은 서로 지우지 않습니다. 같은 종류이고 발행 시각까지 같은 글을 손으로 추가했다면 예전 사본으로 보고 지웁니다.
- 알려진 한계: 병합된 뒤 새 arXiv 버전에서 제목이 바뀌면 기준 브랜치의 예전 글을 슬러그로 찾지 못해 새 URL의 글이 하나 더 생깁니다. 다시 실행한 글의 이미지가 줄어도 기준 브랜치에 있던 남는 번호 이미지는 지우지 않습니다.
- 같은 글이 이미 기준 브랜치에 있으면 그 파일을 갱신하고, PR 체크리스트에 "Replaces the published post" 항목을 넣어 손으로 고친 내용이 사라지지 않는지 확인하게 합니다. 같은 글인지는 종류(요약은 `-summary`)와 정규화한 슬러그로 판단하므로, 예전 `slugify`가 남긴 점·이중 하이픈과 120자 절단 슬러그(`gemini-2.5--pushing-…`)도 같은 글로 봅니다.
- 열린 PR이 있으면 제목과 본문을 갱신하고 URL을 돌려줍니다. 클라이언트는 30초 제한 시간과 `GithubRetry(total=3)`을 쓰고, 오류 메시지에서 토큰을 지웁니다.

PR 본문(`build_pr_body`)에는 논문 ID, PDF URL, 파싱 방식, 글자 수, 저장소, 블로그에 아직 없는 새 태그가 들어가고, 발행 때 검토 메모·규칙 위반·검증 warning 체크리스트가 붙습니다.

## 8. LLM 런타임

### 8.1 모델과 호출

`utils/factories.py::BedrockLanguageModelFactory.get_model`이 모든 채팅 모델을 `ChatBedrockConverse`로 만들고, 리전별 inference profile을 찾아 캐시합니다. 프로필 조회 자체가 실패하면 기본 ID로 대체하지 않고 오류를 냅니다. 모델 특성은 `utils/models.py`의 레지스트리에 있습니다.

| 모델 | 단계 기본값 |
| --- | --- |
| Claude Opus 5.5 | 리뷰 계획·집필·검수·편집·마무리, 요약, 가이드 작성·평가 |
| Claude Sonnet 5.5 | 인용 추출, 가이드 개요 |
| Claude Haiku 5.5 | 그림 분석, 속성 추출, 인용 요약, 코드 분석, 가이드 관련성·분류 |
| Claude Fable 5.1 | 선택 사항 |

`uses_adaptive_thinking` 모델(위 네 모델 모두)에는 설정한 effort(`output_config.effort`)를 보내고, 사고를 켠 호출(`enable_thinking`)에는 `thinking: {type: adaptive}`도 보내며, temperature는 보내지 않습니다. 구형 모델은 `budget_tokens` 방식을 씁니다. 호출당 출력 상한 기본값은 64,000토큰(`DEFAULT_MAX_TOKENS`)입니다.

### 8.2 재시도와 제한 시간

재시도는 한 계층만 소유합니다. botocore는 standard 모드에서 1회만 시도하고(연결 30초, 읽기 3600초: 비스트리밍 호출이 출력 128k 토큰을 끝까지 생성하는 시간), `utils/retry.py::RetryableBase._retry`가 tenacity로 요청 제한, 5xx, 연결·읽기 시간 초과만 지수 백오프와 지터로 재시도합니다. 읽기 시간 초과는 긴 생성 전체를 다시 과금하므로 한 번만 재시도합니다. AccessDenied, ValidationException, ResourceNotFound, 예산 초과, 입력 검증 오류는 바로 실패합니다. `utils/batch.py::BatchProcessor`(코드 분석 일괄 처리)는 실패한 항목만 다시 처리합니다.

### 8.3 예산과 비용

`metrics.py::TokenUsageTracker`가 콜백으로 모든 호출의 토큰을 모델별로 모으고, 레지스트리의 1M 토큰당 단가로 비용을 추정합니다(`estimated_cost_usd`). 단가는 Anthropic 목록가이며 Bedrock 청구액과 다를 수 있습니다. `TokenBudgetGuard`는 단계마다 `max_total_tokens`를 검사해 넘으면 `TokenBudgetExceeded`로 멈춥니다. 가이드의 `max_total_tokens`는 다른 종류와 달리 소프트 상한입니다. 상한에 이르면 남은 절을 쓰되 평가·수정·사실 확인을 건너뛰고, 1.5배를 넘으면 중단합니다. 비용 지표는 실행이 끝날 때 한 번 보내므로, 비용 경보(11절)는 이미 쓴 비용을 알릴 뿐 실행을 멈추지 못합니다. 실행 중 상한은 위 토큰 예산이 맡습니다.

## 9. 설정 참조

`scholar_lens/configs/config.yaml`을 `config.py::Config.load`가 읽습니다. 모든 섹션은 `extra="forbid"`라 모르는 키는 오류입니다. AWS 안에서 파일이 없으면 실패하고, 로컬에서는 기본값을 씁니다. 키별 기본값과 설명은 `config-template.yaml`의 주석이 정본이고, 이 문서에서는 동작에 영향이 큰 것만 짚습니다.

- `resources.s3_bucket_name`: 글·이미지·결과 매니페스트 버킷. 미리 만들어 둔 버킷이어야 하며 배포에 필수입니다. 없으면 `status`는 저장된 결과 없이 Batch 상태만 보여 줍니다.
- `resources.github.*_category`: `categories[0]`이 되므로 블로그 spec의 `types`에 있어야 합니다.
- `explanation.max_synthesis_attempts`·`max_edit_rounds`, `tech_guide.max_revision_attempts`: 호출 수와 비용을 정하는 반복 상한입니다.

### 환경 변수

키 목록과 용도는 `.env.template`이 정본입니다. Batch에서는 같은 비밀값을 SSM에서 읽고(`RUN_SECRETS`), `TOPIC_ARN`은 스택이 주입합니다.

## 10. 인프라

`scripts/deploy_infra.py::PaperReviewStack`(CDK, Python) 구성입니다. `cdk.json`은 `poetry run python scripts/deploy_infra.py`를 실행합니다.

- **네트워크**: 기존 VPC를 쓰거나 NAT 게이트웨이 1개인 VPC를 만듭니다. 새로 만들면 S3 게이트웨이 엔드포인트를 붙입니다. 보안 그룹은 443 아웃바운드만 엽니다.
- **Batch**: 작업 정의 하나(`<project>-<stage>-worker`). 명령은 `JOB_COMMAND`이고 메모리는 8 GiB, 시간 제한은 3시간입니다. 시도는 최대 3회(재시도 2회)지만 재시도는 호스트가 사라진 경우(`Host EC2*` 사유)에만 하며, 나머지는 바로 종료합니다. 컴퓨팅 환경은 온디맨드 하나입니다. 스팟 회수는 작업 전체를 다시 실행해 Bedrock 비용을 반복하므로, 아끼는 EC2 비용보다 손해가 큽니다. 실행 템플릿은 IMDSv2(hop limit 1)를 강제합니다.
- **IAM**: 작업 역할, 실행 역할, 인스턴스 역할을 분리합니다. 실행 역할에는 관리형 정책 없이 이미지 자산의 ECR 저장소 pull과 작업 로그 그룹 쓰기만 붙습니다(CDK가 생성). 작업 역할 권한은 다음과 같습니다.
  - Bedrock 호출과 CountTokens. 대상은 모델 레지스트리에 있는 공급자(`anthropic`, `amazon`, `cohere`)의 foundation model과 계정의 inference profile
  - inference profile 조회(`ListInferenceProfiles`, `GetInferenceProfile`). 리소스 수준 ARN이 없어 `Resource: *`이고 `aws:RequestedRegion`을 기본·Bedrock 리전으로 제한
  - S3 접두어 Get/Put과 버킷 전체 `s3:ListBucket`(조건 없음: 선택 파일인 번역 지침이 없을 때 403이 아니라 404로 보여야 하므로). 그래서 버킷은 이 프로젝트 전용이어야 합니다. `s3:PutObjectAcl`은 `public_assets`일 때만
  - SNS 게시와 토픽 KMS 키 사용
  - `cloudwatch:PutMetricData`(`ScholarLens` 네임스페이스 조건)
  - SSM 비밀값 8개 읽기(`RUN_SECRETS`)
- **SSM**: 스택은 `batch-job-queue`, `batch-job-definition`(String)을 씁니다. 비밀값(SecureString)은 CloudFormation이 만들 수 없으므로 `scripts/put_secrets.py`로 따로 기록합니다(이름을 주면 그 비밀값만). `cdk synth`/`diff`는 비밀값을 건드리지 않습니다. 실행 시 기록되지 않은 비밀값(`ParameterNotFound`)은 조용히 건너뛰고, 권한 오류 등은 warning으로 남깁니다.
- **설정 파일**: 이미지에는 배포하는 체크아웃의 `scholar_lens/configs/config.yaml`이 들어갑니다. 설정을 고치면 `cdk deploy`를 다시 해야 Batch에 반영됩니다. `deploy_infra.py::check_deployable`은 이 파일이나 `s3_bucket_name`이 없으면 배포를 멈춥니다. `cdk` 명령에는 `resources.profile_name`과 같은 `--profile`을 넘깁니다.
- **알림**: KMS 암호화 SNS 토픽(SSL 강제)에 이메일을 구독합니다. 토픽 정책은 이 계정의 CloudWatch 경보(`aws:SourceAccount`·`aws:SourceArn` 조건)와 실패 규칙 하나의 EventBridge 게시만 허용합니다. EventBridge 대상이 자동으로 만드는 허용은 조건이 없으므로, 그 규칙이 아닌 게시를 막는 Deny 문을 따로 둡니다. 토픽 KMS 키의 EventBridge 허용은 CDK가 만든 그대로입니다.

## 11. 관측성

- **로그**: `logger.py`가 stderr로 쓰고, 로컬에서는 `logs/`에도 남깁니다. 모든 실행 끝에 `run_metrics` 한 줄(종류, 성공, 토큰, 호출 수, 비용, 시간)을 남깁니다.
- **지표**: `metrics.py::MetricsEmitter.emit_run`이 `ScholarLens` 네임스페이스에 `InputTokens`, `OutputTokens`, `TotalTokens`, `LLMCalls`, `EstimatedCostUSD`, `DurationSeconds`, `Success`, `Failure`를 `Mode` 차원이 있는 것과 없는 것 두 벌로 보냅니다(AWS 안에서만).
- **경보**:
  - 1시간 추정 비용 합계 50달러 초과, 경보 1개. 지표는 실행이 끝날 때 보내므로 진행 중인 실행을 멈추지 못합니다(8.3절). 지표는 AWS 안에서만 보내므로 이 경보는 Batch 실행만 다루고, 로컬 실행 비용은 `JobResult.estimated_cost_usd`와 `run_metrics` 로그로만 확인합니다
  - EventBridge의 Batch `FAILED` 이벤트를 SNS로 전달. 앱이 직접 알리고 종료 코드 10·11로 끝난 작업은 제외하고, 그 밖의 종료 코드(시간 초과·중단 143, OOM·비정상 종료, worker 실행 전 Python·인자 오류 1·2, 알림 실패 4)와 종료 코드가 없는 경우(호스트 유실)는 알림. 이 규칙의 SNS 전달이 실패하면(`FailedInvocations`) 경보 하나가 같은 토픽으로 알림
  - 실행 결과는 앱 알림과 이 규칙이 다루므로 로그 `ERROR` 경보는 두지 않습니다. 실행을 멈추지 않는 문제는 warning 로그로 남깁니다. PR 생성 실패처럼 결과에 남는 경고는 알림에 모두 적고, 알림·매니페스트 저장 실패는 로그에만 남습니다(실패 실행의 알림 실패는 위의 종료 코드 4로 보고). 성공한 실행의 알림이 실패하면 작업은 0으로 끝나 아무도 알리지 않지만, 결과 매니페스트가 있어 `status`로 결과를 볼 수 있습니다.
- **결과 매니페스트**: 원격 실행의 최종 상태는 S3 매니페스트와 `status` 명령으로 확인합니다.

## 12. 테스트와 CI

테스트는 모두 `tests/`에 있고 AWS·네트워크 없이 돕니다. `conftest.py`가 루프백이 아닌 소켓 연결과 arXiv 속도 제한 대기를 막고, AWS는 moto, HTTP는 `responses`·httpx MockTransport, GitHub는 가짜 저장소 객체로 대신합니다. 파일 이름은 대체로 대상 모듈을 따릅니다(`test_<module>.py`). `test_e2e_pipeline.py`는 실제 모델이 내는 결함을 섞은 가짜 출력이 블로그 규칙을 통과하는 글이 되는지, 고칠 수 없는 글은 막히는지 봅니다. `tests/fixtures/blog_defects/`의 블로그 결함 원본은 error, 수정본은 통과해야 합니다. `test_infra_contract.py`는 가짜 계정으로 CDK를 synth해 작업 명령, 재시도 규칙, 실패 알림 규칙, IAM 범위를 CLI·지표 이름과 대조합니다.

CI는 Action을 커밋 SHA로 고정합니다. `ci.yml`(`main` push, 모든 PR)은 lint(ruff, black `--check`, mypy), Python 3.12·3.13 테스트(커버리지 85% 이상), Batch 이미지 빌드와 이미지 안 import 확인, 잘못된 워커 요청이 JSON과 종료 코드 11로 끝나는지 확인, 브랜치 보호가 요구하는 CDK synth(`scripts/ci_synth_check.py`)를 실행합니다. `post-conventions.yml`은 사본과 블로그 정본을 비교합니다(`scripts/check_post_conventions.py`). 변경 때와 매주 월요일에 돌고, 저장소 변수 `BLOG_REPO`가 없으면 변경 때는 건너뛰지만 일정 실행은 실패합니다. 토큰이 있는데 정본이 없으면(404) 실패하고, 토큰이 없으면 비공개 저장소와 구분할 수 없어 건너뜁니다. `workflow-lint.yml`은 체크섬을 검증한 actionlint로 워크플로를 검사하고, Dependabot은 GitHub Actions, pip, Docker를 매주 확인합니다.

Docker 이미지(`Dockerfile`)는 `python:3.12-slim` 기반 2단계 빌드입니다. 의존성은 `poetry.lock`에서 `--only main`으로 설치하고, 런타임에는 `git`, `tzdata`, `fonts-dejavu-core`(SVG 그림 래스터화용 글꼴)를 더하며, NLTK `punkt_tab`을 빌드 때 받아 `NLTK_DATA`로 지정하고 non-root로 실행합니다. `.dockerignore`는 허용 목록 방식이라 `pyproject.toml`, `poetry.lock`, `README.md`, `assets/post_conventions.yml`, `scholar_lens/`만 빌드 컨텍스트에 들어갑니다.

## 13. 확장 지점

- **새 산출물 종류**: `jobs.py::ArtifactKind`, `post_validator.py::ArtifactKind`, `cli.py::_raw_kind`에 값을 더하고, 파이프라인 모듈을 만들어 `runtime.py::_run_pipeline`에서 분기하며, `post_validator.py`에 그 종류의 규칙을 정합니다.
- **새 모델**: `constants.py::LanguageModelId`와 `utils/models.py`의 레지스트리(컨텍스트, 출력 상한, thinking 방식, 단가)에 추가합니다.
- **새 검색 제공자**: `web_research.py::WebSearchProvider`를 구현하고 `guide_pipeline.py::build_search_provider`에 연결합니다.
- **블로그 규칙 변경**: 블로그의 `_data/post_conventions.yml`을 고치고 `assets/post_conventions.yml`에 같은 내용을 복사합니다. 새 규칙 종류가 필요하면 `post_conventions.py`의 모델과 `post_validator.py`에 함께 추가합니다.
