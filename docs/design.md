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
| 기술 가이드 | `guide` | 문서 URL 목록, 선택적 검색어 | `guide_pipeline.py::run_guide_pipeline` → `tech_guide.py::TechGuideGenerator` |

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
| `src/paper_source.py`, `src/arxiv_handler.py`, `src/parser.py`, `src/content_extractor.py` | 논문 소스, 파싱, 속성·인용 추출 |
| `src/citation_metadata.py`, `src/citation_summarizer.py`, `src/code_retriever.py` | 인용 해석·요약, 코드 검색 |
| `src/formatting.py`, `src/tags.py`, `src/markdown_math.py`, `src/markdown_lint.py` | 글 조립, 태그 정규화, 수식·Markdown 후처리 |
| `src/post_conventions.py`, `src/post_validator.py` | 블로그 규칙 로드, 최종 검증 |
| `src/publisher.py` | 이미지 로컬화, 저장, S3, GitHub PR |
| `src/utils/` | 모델 팩토리, 모델 레지스트리, 재시도, 배치 처리, 공용 파서 |
| `src/metrics.py` | 토큰·비용 집계, 예산 가드, CloudWatch 지표 |
| `src/url_guard.py`, `src/pinned_transport.py`, `src/rate_limiter.py` | SSRF 방지, IP 고정 전송, 속도 제한 |
| `scripts/` | CDK 스택, CI synth 검사, 비밀값 기록, 블로그 규칙 drift 검사 |

## 2. CLI와 작업 계약

### 2.1 명령

`cli.py::build_parser`가 하위 명령 다섯 개를 정의합니다.

| 명령 | 동작 |
| --- | --- |
| `review SOURCE`, `summarize SOURCE` | 논문 산출물을 로컬에서 실행하거나 `--remote`로 제출 |
| `guide URL...` | 기술 가이드를 로컬에서 실행하거나 `--remote`로 제출 |
| `status JOB_ID [--wait] [--timeout S]` | Batch 상태와 결과 매니페스트 조회(`jobs.py::describe_job`, `wait_for_job`) |
| `worker --request JSON` | Batch 진입점. `AWS_BATCH_JOB_ID`를 작업 ID로 사용 |

stdout에는 JSON 객체 하나만 출력하고 로그는 stderr로 보냅니다. 종료 코드는 `cli.py::_EXIT_BY_STATUS`로 정해집니다.

| `JobResult.status` | 의미 | 종료 코드 |
| --- | --- | --- |
| `succeeded` | 글이 저장됨. PR 실패는 `warnings`에 기록 | 0 |
| `failed` | 생성 실패, 블로그 규칙 위반(`violations`), 중단 | 1 |
| `rejected` | 입력 거부: PDF가 아님, 안전하지 않은 URL, 없는 arXiv ID, 기술 문서가 아님 | 2 |

`--remote`는 `SubmittedJob`을 출력하고 0으로 끝납니다. `status`는 결과 manifest가 있으면 그 상태의 종료 코드를 돌려줍니다. 없으면 Batch가 `FAILED`일 때 1, 아직 끝나지 않았거나 `--wait`가 시간 초과되면 3, `SUCCEEDED`인데 manifest가 없으면 0입니다. `--wait` 중 Batch 조회나 결과 manifest 읽기의 throttling·5xx·네트워크 오류는 건너뛰고 계속 조회합니다.

CLI 자체가 실패하면(자격 증명, 설정, Batch·S3 API 오류) `{"status": "error"}`와 종료 코드 4를 냅니다. 작업 실패(1)와 구분하기 위해서이며, 이때 원격 작업의 상태는 알 수 없습니다.

### 2.2 JobRequest와 JobResult

`jobs.py::JobRequest`는 실행 하나를 완전히 기술합니다. `extra="forbid"`이며, 논문 요청은 `source`를, 가이드 요청은 `urls`를 요구하고 섞어 쓰면 거부합니다.

| 필드 | 대상 | 기본값 |
| --- | --- | --- |
| `kind` | 전체 | 필수(`review`, `summarize`, `guide`) |
| `source` | review, summarize | 필수 |
| `repo_urls`, `parse_pdf` | review, summarize | `[]`, `false` |
| `urls`, `search_queries`, `discover_subpages` | guide | 필수, `[]`, `true` |
| `create_pr` | 전체 | `true`(GitHub 연동이 켜져 있을 때만 의미) |

로컬 CLI와 Batch worker는 같은 객체를 씁니다. `--remote`는 `JobRequest.model_dump_json()`을 Batch 파라미터 `request` 하나로 넘기고, 작업 정의의 명령은 `["python3", "-m", "scholar_lens", "worker", "--request", "Ref::request"]`입니다(`scripts/deploy_infra.py::JOB_COMMAND`). 두 경로가 같은 파서를 쓰므로 인자 계약이 어긋나지 않습니다.

`jobs.py::JobResult`에는 `status`, `title`, `markdown_path`, `s3_uri`, `pr_url`, `warnings`, `violations`, `error`, `job_id`, `duration_seconds`, 토큰 수, `estimated_cost_usd`가 들어갑니다. Batch 실행은 이 결과를 `s3://<bucket>/<s3_prefix>/results/<job_id>.json`에 저장하고(`jobs.py::write_result_manifest`), `status`가 이를 읽어 붙입니다.

### 2.3 실행 수명주기

`runtime.py::run_job` 순서는 다음과 같습니다.

1. `Config.load()`와 `build_context`로 boto 세션(기본 리전, Bedrock 리전)과 S3 핸들러를 만듭니다.
2. AWS 안에서는 SSM SecureString에서 비밀값을 환경 변수로 읽습니다(`load_secrets_from_ssm`). 로컬은 `.env`를 씁니다.
3. 파이프라인을 실행합니다. 입력 오류는 `rejected`, `PostValidationError`는 `failed`와 `violations`, 그 밖의 예외는 `failed`로 기록합니다. `SystemExit`·`KeyboardInterrupt`는 정리 후 다시 던집니다.
4. `finally`에서 지표를 보내고, AWS 안에서 `TOPIC_ARN`이 있으면 성공·실패 모두 SNS로 알리며(`format_result_notification`), Batch 실행이면 결과 매니페스트를 저장합니다.

worker는 SIGTERM을 `SystemExit`로 바꿔(`cli.py::_raise_on_sigterm`) Batch 시간 초과나 스팟 회수 때도 4단계가 실행되게 합니다. 파이프라인은 `asyncio.run` 대신 `runtime.py::run_coroutine`으로 돌려, 중단 시 작업자 스레드에서 진행 중인 Bedrock 호출을 기다리지 않고 바로 정리 단계로 갑니다(Batch는 SIGTERM 약 30초 뒤 SIGKILL). 이때는 매니페스트만 저장하고(`INTERRUPTED_ERROR`) SNS 실패 알림은 보내지 않습니다. 정상 종료는 인터프리터가 남은 작업자 스레드를 기다리므로, worker는 정리 뒤 `os._exit(143)`으로 바로 끝납니다(`cli.py::_work`). 스팟 회수면 Batch가 재시도하고, 최종 실패면 EventBridge 규칙이 알립니다.

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
- 리뷰 전체 예산은 `7000 × (1 + 0.6 × 어려운 항목 비율)`입니다(`review_word_budget`). 넘치면 쉬움·중간 항목만 한 비율로 줄이고(최대 절반까지), 어려운 항목은 줄이지 않으므로 어려운 논문은 예산을 넘을 수 있습니다.

### 3.2 보강

LLM 라우팅 호출 없이 계획을 따릅니다(`PaperReviewer._enrich`). 참고문헌에 연결된 선수 개념만 `CitationSummarizer.summarize`로 요약하고, `needs_code`인 절만 `CodeRetriever.search_similar_code`로 코드를 찾습니다. 실패한 조회와 실패 표시가 붙은 요약은 버리며, 보강 실패가 실행을 멈추지 않습니다.

### 3.3 집필과 검수

집필(`_write`)은 절마다 다음을 받습니다.

- 전체 개요(`render_outline`)와 해당 절 지시서(`render_brief`)
- 그 절의 원문(다른 절에 배정된 그림 마커는 제거, `section_source`)
- 보강 자료, 용어집, 다른 절이 맡은 개념(`concepts_owned_elsewhere`), 지금까지 쓴 절 전체

쉬운 항목은 한두 문장으로, 어려운 항목은 직관 → `$$` 수식 → 작은 예제 → 그림 순으로 씁니다. 길이 때문에 끊을 때만 `has_more`로 이어 쓰며 최대 `MAX_CONTINUATIONS`(4)번입니다.

검수(`_check`)는 지시서, 원문, 측정한 분량과 목표 분량을 보고 0–100점과 빠진 항목(`missing_items`), 반드시 고칠 결함(`must_fix`: 반복, 다시 계산하면 틀리는 수치, 근거 없는 주장·그림 설명), 개선 의견을 돌려줍니다. 배점은 누락 없음 30, 깊이 25, 원문 충실도 25, 흐름 20이며, 빠진 항목이 하나라도 있으면 점수 상한은 69입니다. 코드가 찾은 규칙 위반(`markdown_lint.py::writing_issues`: 원 논문 번호 참조, 영어 어간+한국어 어미, 글 자체 지칭, 단일 `$` 수식)도 `must_fix`에 더합니다. 절은 점수가 `min_quality_score` 이상이고 빠진 항목과 `must_fix`가 없을 때만 통과하며, 그렇지 않으면 지금까지 가장 좋은 초안과 지적을 함께 넘겨 고쳐 쓰게 합니다. 초안 순위는 통과 여부 → 막는 결함 수 → 점수 순이고(`review_plan.py::draft_rank`, 가이드 루프와 공유), 기각된 수정본의 문제도 다음 지적에 붙입니다. 초안 수는 `max_synthesis_attempts`가 상한이고, 절은 계획된 `##` 제목으로 시작하게 맞춥니다(`ensure_section_heading`).

### 3.4 전체 편집

절을 모두 이은 글 전체를 한 번 편집합니다(`_edit`, 기획 모델 사용). 입력은 글, 논문 원문, 계획의 핵심 항목 목록, 용어집입니다. 출력은 `<edit><find>…</find><replace>…</replace></edit>` 블록 목록이고(LaTeX와 `&`를 그대로 담기 위해 JSON이 아닌 태그 형식), `review_plan.py::apply_edits`가 다음 조건에서만 적용합니다.

- `find`가 현재 글에 정확히 한 번 나와야 합니다.
- `## ` 헤딩은 지우거나 바꾸거나 강등할 수 없습니다. 중복된 `###`·`####` 소절은 통째로 지울 수 있습니다.
- 그림, 디스플레이 수식, 표의 유일한 사본을 지우는 편집은 거부합니다.
- 편집은 `MAX_EDITS`(40)개까지입니다.

편집자가 누락 항목을 보고하면 그 목록을 붙여 한 번 더 편집하며, 횟수는 `max_edit_rounds`(기본 2, 0이면 편집 생략)가 상한입니다.

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

`summarizer.py::PaperSummarizer.summarize`는 파싱한 논문 전체(컨텍스트에 맞게 자름)와 코드베이스 요약으로 한 번 호출합니다. 출력 태그는 `subtitle`, `summary`, `urls`입니다. 요약은 `> **요약:**` 인용문과 이모지 `##` 절 다섯 개(왜 필요한가, 해결책, 구현, 결과, 의의와 한계)로 이루어지고, 실증 논문이면 결과 절에 무게를 둡니다. `formatting.py::summary_with_tldr`가 앞의 인용문을 `## TL;DR` 절로 바꿔 블로그 규칙에 맞춥니다.

### 4.2 기술 가이드

`tech_guide.py::TechGuideGenerator.generate` 순서입니다.

1. 시드 URL과 범위 안 하위 페이지를 수집합니다(`web_research.py::WebResearcher.research`).
2. 검색 제공자가 있고 검색어가 주어지지 않았으면 계획 단계가 검색어를 만들고(`auto_research`, `max_research_queries`), 상위 결과 페이지를 말뭉치에 넣습니다(`fetch_top_results`).
3. 관련성 판정(`TechGuideRelevancePrompt`)을 통과하지 못하면 `NotTechnicalContentError`로 거부합니다.
4. 개요를 만들고(최대 10개 절) 절마다 영역(CONCEPT/DETAIL/USAGE/APPLICATION)과 깊이(deep/standard/brief)를 정합니다.
5. 절마다 작성 → 평가(0–100) → 기준 미달 시 수정(`max_revision_attempts`) → 사실 확인(`verify_grounding`) 순서로 진행합니다. 각 절의 호출에는 말뭉치 전체가 아니라 절 제목·설명과 가장 관련 있는 페이지만 약 6만 자까지 넣고(`ResearchCorpus.relevant_text`), 앞선 절은 직전 두 절 전문과 그 이전 절의 요약만 넘깁니다.
6. 절이 셋 이상이면 개요에서 "이 가이드에서 다루는 것"을 만듭니다(추가 호출 없음).

가이드 파이프라인(`guide_pipeline.py`)은 완성된 가이드를 `TechGuideGenerator.classify`로 분류해 블로그 주제 하나, 통제 태그, 세부 태그, `seo_title` 후보를 받고, `formatting.py::build_post_meta`가 spec에 맞게 확정합니다. 문서 사이트의 이미지는 외부 링크로 남기고 검증에서 warning으로 표시합니다.

## 5. 데이터 수집과 파싱

### 5.1 논문 소스

`paper_source.py::resolve_paper_source`가 입력을 판별합니다.

- arXiv ID(신규형, `hep-th/9901001` 같은 구형, `arXiv:` 접두어, 버전 포함)와 arXiv URL은 `ArxivSource`가 됩니다. 메타데이터는 arXiv API에서 받습니다.
- 그 밖의 URL은 `PdfUrlSource`입니다. SSRF 검사 후 content-type과 `%PDF-` 매직 바이트로 실제 PDF인지 확인하며, 아니면 `NotAPdfError`입니다. 제목은 본문에서 추출한 제목으로 대체합니다(`title_is_fallback`).

### 5.2 파싱

- arXiv 소스는 arXiv HTML(`arxiv.org/html`, 실패 시 ar5iv)을 먼저 씁니다(`parser.py::HTMLRichParser`). 실패하거나 내용이 비면 PDF로 넘어갑니다.
- PDF는 Upstage Document Parse(`parser.py::PDFParser`)로 파싱하고, 실패하면 PyMuPDF로 텍스트만 추출합니다. 빈 응답은 캐시하지 않습니다.
- 그림은 비전 모델(`figure_analysis_model_id`)로 분석하고, `parser.py::attach_figure_context`가 본문의 이미지 마커에 경로와 분석을 붙입니다. 짝짓기는 순서가 아니라 요소 ID와 파일명으로 합니다.

### 5.3 HTML을 텍스트로

`utils/helpers.py::extract_text_from_html`이 HTML 구조를 유지한 텍스트를 만듭니다.

- 헤딩은 Markdown 헤딩, `<pre>`는 펜스 코드, 블록 요소 사이는 빈 줄입니다. nav, header, footer, aside는 버립니다.
- 수식은 MathML `alttext`(또는 TeX 주석)를 `$$…$$`로 옮기고, 디스플레이 수식은 앞뒤 빈 줄을 둔 별도 줄로 둡니다.
- 표는 `utils/helpers.py::table_to_markdown`이 완성된 GFM 표로 만듭니다. colspan·rowspan을 펼쳐 격자를 만들고, 여러 줄 헤더(thead 또는 숫자 없는 앞쪽 행, 최대 3행)는 열마다 공백으로 합쳐 한 행으로 만들며, 빈 칸은 `-`, 셀 안의 `|`는 `\|`로 씁니다. 캡션은 표 위에 둡니다.

### 5.4 속성과 인용

`content_extractor.py::ContentExtractor`가 두 가지를 추출합니다.

- 속성(`extract_attributes`): 제목, 저자, 발행 조직(`affiliation`), spec 주제 중 하나(`category`), 통제 태그, 세부 키워드, `seo_title`. 키워드 재사용 후보로 블로그 글에 이미 쓰인 태그를 넘깁니다(`runtime.py::RunContext.require_blog_tags`).
- 인용(`extract_citations`, 리뷰만): 참고문헌을 페이지 단위로 추출하며 최대 `MAX_CITATION_PAGES`(20)회입니다.

인용 요약(`citation_summarizer.py::CitationSummarizer`)은 초록을 우선합니다. Crossref와 Semantic Scholar로 메타데이터를 찾고(`citation_metadata.py`, 속도 제한 포함), 제목 유사도 0.9 이상만 같은 논문으로 인정합니다. 맞지 않는 초록은 요약하지 않습니다. `prefer_full_text: true`면 arXiv 전문을 받습니다.

### 5.5 코드 저장소

`code_retriever.py::CodeRetriever`는 저장소를 얕게(depth 1, 단일 브랜치, 심볼릭 링크 제외) 시간 제한을 두고 clone합니다. 가상환경·벤더 디렉터리와 저장소 밖을 가리키는 파일은 건너뛰고, Python 파일을 청크로 나눠 Titan 임베딩 FAISS 인덱스를 만듭니다. 인덱스는 저장소 집합별로 구분하고, 코드베이스 요약을 함께 만듭니다. 코드 분석 실패는 "코드 없음"으로 처리합니다.

### 5.6 웹 조사

`web_research.py::WebResearcher`는 시드 경로 접두어(끝 `/` 포함) 안의 하위 페이지만 최대 8개 따라가며, 본문 영역의 링크를 우선합니다. 리디렉션 후 최종 URL 기준으로 링크를 해석하고, 5MB를 넘는 페이지는 건너뜁니다. 검색은 Tavily, 없으면 Brave, 둘 다 없으면 생략합니다(`guide_pipeline.py::build_search_provider`). 페이지 본문·제목·URL의 프롬프트 경계 태그는 무력화합니다(`neutralize_prompt_tags`).

### 5.7 SSRF 방지

외부 URL은 `url_guard.py::assert_url_is_public`이 스킴과 해석된 IP를 검사하며, 전역 주소가 아닌 IP(사설, 루프백, 링크 로컬, CGNAT, 메타데이터 대역)는 모두 막습니다. `pinned_transport.py`는 검사한 IP에 연결을 고정해 DNS 재바인딩을 막고, 리디렉션마다 다시 검사합니다.

## 6. 블로그 규칙과 검증

### 6.1 규칙 데이터

정본은 블로그 저장소의 `_data/post_conventions.yml`이고, `assets/post_conventions.yml`은 같은 내용의 사본입니다. `post_conventions.py::PostConventions`가 읽는 항목은 다음과 같습니다.

| 키 | 의미 |
| --- | --- |
| `types` | `categories[0]` 값. `paper: true`인 유형은 논문 글(`paper_author` 필요), `cover`가 있으면 그 표지 |
| `topics` | `categories[1]` 값과 주제별 표지. 표지가 없으면 `/assets/images/default.jpg` |
| `forbidden_keys` | front matter 금지 키(`author`) |
| `tags.controlled` | 통제 태그 목록. 관련 글 연결의 기준 |
| `tags.lowercase_words` | 태그 안에서 소문자로 쓰는 기능어 |
| `tags.aliases` | 퇴역 태그 → 대체 태그 |

블로그 파일에 아직 없는 생성기 쪽 한도는 코드 기본값을 씁니다. 세부 태그 검증 상한은 10개(`TagSpec.max_specific`)이고 생성 목표는 5개(`GENERATED_SPECIFIC_TAGS`)입니다. 통제 태그는 생성 시 최대 3개(`max_controlled`)이며, `seo_title` 기준 길이는 70자, 블로그 정의 매크로는 `llbracket`, `rrbracket`, `textsc`입니다. YAML 1.1 파서가 `on`을 불리언으로 읽는 문제는 `TagSpec._unbool`이 되돌립니다.

실행 중에는 GitHub 연동이 켜져 있으면 블로그 `base_branch`의 정본을 읽고(`post_conventions.py::load_conventions`), 읽지 못하면 사본을 쓰며 사본과 다르면 경고합니다. CI의 drift 검사는 둘이 다르면 실패합니다(12절).

### 6.2 결정적 후처리

검증 전에 코드가 다음을 고칩니다.

- **front matter 조립**(`formatting.py`): `author`는 쓰지 않습니다. 논문 글은 `paper_author`(발행 조직)와 `subtitle`(번역 제목, 본문 첫 H1이 있으면 그것)을 둡니다. `seo_title`은 제목이 70자를 넘을 때만 넣으며, 모델 후보가 없거나 길면 콜론 앞부분이나 단어 경계에서 자릅니다(`shorten_title`). 카테고리 주제와 표지는 spec에서 정하고(`resolve_topic`, `PostConventions.cover_for`), `use_math`는 `$$`가 있을 때만 켭니다. `date`는 논문 발행일입니다.
- **태그**(`tags.py::normalize_tags`): 공백·`_`는 하이픈, `&`는 `and`, `/`와 따옴표는 삭제, 연속 하이픈 축약, 첫 단어가 아닌 기능어는 소문자로 바꿉니다. 그다음 별칭을 대체 태그로 바꾸고, 대소문자를 무시해 중복을 없애며, 제목 전체·콜론 앞·콜론 뒤와 같은 태그를 지웁니다. 통제 태그는 닫힌 목록만 남기고, 세부 태그 중 통제 태그와 같은 것은 통제 쪽으로 옮깁니다. 발행 순서는 세부 태그 다음 통제 태그입니다.
- **수식**(`markdown_math.py::normalize_math`): `\(…\)`·`\\(…\\)`는 `$$…$$`로, `\[…\]`는 앞뒤 빈 줄을 둔 디스플레이 `$$`로 바꿉니다. `$$` 안의 `\_`는 `\text{}`·`\texttt{}` 안을 빼고 `_`로 되돌리고, 한 줄 전체가 `$$`인 수식 앞뒤에 빈 줄을 넣습니다(표·목록 제외). 코드와 단일 `$`는 건드리지 않습니다.
- **Markdown**(`markdown_lint.py::lint_markdown`): 헤딩 앞 빈 줄, 생성용 envelope 태그 제거, H1을 H2로 올리고 한 단계 넘게 깊어진 헤딩을 앞 헤딩 바로 아래 단계로 올리기, 네 번째부터의 콜아웃(`> **라벨:** …`)을 일반 문단으로 풀기, 가짜 HTML 태그 이스케이프를 처리합니다. `drop_unknown_images`는 논문 그림이 아닌 이미지를 지우고, `fill_image_alts`는 빈 alt를 그림 캡션으로 채우며, 한국어 글의 영어 alt는 가장 가까운 헤딩으로 바꿉니다. 원 논문 번호 참조 같은 규칙 위반(`writing_issues`)은 게시 직전 본문에서 다시 찾아 PR 체크리스트에 올립니다.

### 6.3 검증 규칙

`post_validator.py::validate_post(markdown, spec, kind)`는 순수 함수이며 `Violation(level, rule, line, message)` 목록을 돌려줍니다. 코드 블록 안은 검사하지 않습니다.

| 규칙 | 수준 | 조건 |
| --- | --- | --- |
| `front-matter` | error | front matter가 없거나 YAML이 아님 |
| `forbidden-key` | error | `forbidden_keys`의 키가 있음 |
| `title` | error | 제목이 비어 있음 |
| `categories` | error | `[<유형>, <주제>]`가 아님, 모르는 유형·주제, 산출물 종류와 유형의 `paper` 여부 불일치 |
| `cover` | error | `cover_for(유형, 주제)`와 다름 |
| `paper-author` | error | 논문 글에서 비었거나 `et al.` 또는 사람 이름 나열, 가이드에 있음 |
| `subtitle` | error | 논문 글에서 비었거나 제목과 같음 |
| `seo-title` | error | 제목이 70자 초과인데 없음, 또는 `seo_title`이 70자 초과 |
| `translation` | error | `lang`이 `ko`/`en`이 아니거나 번역본에 `translation_id`가 없음 |
| `tags` | error | 비어 있음, 정규화되지 않은 태그, 퇴역 태그, 통제 태그 없음, 세부 태그 상한 초과 |
| `tags` | warning | 통제 태그 3개 초과, 태그 구절과 약어가 본문에 없음 |
| `headings` | error | 본문 H1, 헤딩 레벨 건너뜀, 논문 글이 `## TL;DR`로 시작하지 않음 |
| `use-math` | error | `use_math`와 `$$` 존재 여부가 다름 |
| `math` | error | `\(…\)` 잔존, 디스플레이 `$$` 앞뒤 빈 줄 없음, 짝 없는 `$$`(통화를 수식으로 쓴 경우), `$$` 안 `\_`(`\text` 밖), 인라인 수식 안 한글, 인라인 수식이 여럿인 줄이나 표 행의 수식 안 맨 `|` |
| `table` | error | 헤더·구분선·데이터 행 셀 수 불일치, 첫 열 외 빈 헤더, 첫 데이터 행이 하위 헤더(첫 셀이 비고 값 셀이 없음) |
| `table` | warning | 같은 헤더 라벨이 바로 옆에 반복 |
| `image` | error | arXiv 계열 원격 이미지 잔존, `/assets/images/<파일>`이 아닌 로컬 경로 |
| `image` | warning | 그 밖의 외부 이미지 |
| `description` | warning | 첫 문단(검색 설명)에 수식 |
| `envelope` | error | 생성용 태그(`</explanation>` 등) 누출 |

`Publisher._validate`가 이 함수를 호출합니다. error가 하나라도 있으면 저장·업로드·PR을 모두 하지 않고, 글과 위반 목록을 작업 폴더의 `<slug>.rejected.md`에 남긴 뒤 `PostValidationError`를 던집니다. warning은 `PublishRequest.validation_warnings`에 담겨 PR 본문의 `- [ ]` 체크리스트가 됩니다.

## 7. 발행

`publisher.py::Publisher.publish` 순서입니다.

1. 모델이 입력에서 그대로 옮긴 방어용 태그 표식(`prompt_safety.py::strip_defanging`)을 front matter와 본문 모두에서 지우고, 본문에 `normalize_math`와 `lint_markdown`을 적용합니다.
2. `ImageLocalizer`가 이미지를 블로그 규칙에 맞게 옮깁니다. arXiv 계열 호스트(`arxiv.org`, `ar5iv.org`, `ar5iv.labs.arxiv.org`와 하위 도메인) 이미지는 URL 검사와 IP 고정 전송을 거쳐 내려받고(30초, 리디렉션 5회, 20MB 상한), PDF에서 뽑은 로컬 그림도 같은 방식으로 처리합니다. PNG·JPEG는 매직 바이트로 확인하고, SVG는 PNG로 래스터화합니다(`utils/images.py::svg_to_png`). 래스터화 전에 `sanitize_svg`가 `#id`와 래스터 `data:image/…`가 아닌 `href`, 문서 밖을 가리키는 CSS `url(…)`·`@import`를 지웁니다. resvg가 로컬 파일 경로를 그대로 읽기 때문입니다. 내부 DTD(엔티티 선언)가 있거나 XML이 깨진 SVG는 거부합니다. 루트의 `width`·`height`·`viewBox`(단위 포함)로 캔버스 크기를 먼저 계산해 긴 변이 4096px를 넘지 않도록 확대율을 낮춥니다. `viewBox`와 절대 크기가 모두 있지 않으면 resvg가 그려진 내용의 경계로 캔버스를 잡으므로, 계산한 크기를 루트에 절대값으로 적고 `viewBox`를 붙여 브라우저처럼 그 상자로 자릅니다. 크기가 유한하지 않거나 터무니없으면 거부하고, `data:` URL로 넣은 래스터가 5천만 화소를 넘어도(헤더만 읽어 판단) 거부합니다. 캔버스가 작아도 패턴 타일, 필터 영역, `feMorphology` 반경처럼 내부 요소가 메모리를 수 GB 쓰거나 CPU를 몇 시간 쓸 수 있으므로, 렌더링은 자식 프로세스(`python -I -c`)에서 60초 제한으로 실행하고, 동시에 2개까지만 돌려 그림이 많아도 전체 메모리와 CPU가 작업 한도 안에 머물게 합니다. Linux(Batch)에서는 자식에 `RLIMIT_AS` 2GiB를 걸어 할당 실패로 즉시 끝나고, macOS는 이 제한이 걸리지 않아 시간 제한만 적용됩니다. 시간 초과·비정상 종료는 `ValueError`이고, 그림 1개당 추가 비용은 프로세스 기동 약 20~30ms입니다. 그 밖의 형식은 거부합니다. 저장 이름은 첫 등장 순서대로 `<slug>-<n>.<ext>`이고, 본문 경로는 `/assets/images/<slug>-<n>.<ext>`로 바꿉니다. 파일은 `<work_dir>/blog-images/<slug>/`에 모읍니다. 같은 논문의 요약과 리뷰가 작업 폴더를 같이 쓰므로, 재실행은 자기 슬러그 폴더만 비우고 S3 업로드와 PR에도 자기 이미지만 넣습니다. 실패하면 `PostAssetError`입니다. 그 밖의 외부 이미지는 링크로 남깁니다.
3. 검증합니다(6.3).
4. `<work_dir>/<YYYY-MM-DD>-<slug>.md`로 저장합니다. 파일명 날짜는 게시일(UTC)이고, front matter `date`는 논문 발행 시각입니다(가이드는 생성 시각). Jekyll은 미래 날짜 글을 건너뛰므로 두 날짜 모두 UTC로 씁니다. `slugify`는 ASCII 슬러그를 만들고, ASCII 글자가 없는 제목은 `untitled-<8자리 해시>`로 바꿉니다.
5. S3 버킷이 있으면 글은 `<prefix>/posts/`, 이미지는 `<prefix>/assets/images/`에 올립니다. 이미지 하나라도 실패하면 오류입니다. `public_assets`가 켜져 있을 때만 public-read ACL을 씁니다.

`runtime.py::publish_artifact`는 `create_pr`이고 GitHub 연동이 켜져 있으면 `create_pull_request`를 호출합니다. PR 실패(`PublishError`)는 실행을 실패시키지 않고 `warnings`에 남깁니다.

`publisher.py::BlogBranchSync`는 clone 없이 GitHub API로 커밋합니다.

- 브랜치는 `<branch_prefix>/<branch_id>`로 고정됩니다(리뷰 `review-<id>`, 요약 `summary-<id>`, 가이드 `guide-<URL 해시>`). 같은 입력으로 다시 실행하면 같은 브랜치와 PR을 갱신합니다. 가이드 브랜치는 정규화한 입력 URL(쿼리 포함, 프래그먼트 제외) 집합의 해시라 모델이 고른 제목과 무관합니다. 다만 다시 실행한 가이드의 제목이 바뀌면 열린 PR에 이전 글이 함께 남으므로 직접 지워야 합니다.
- 열린 PR이 있으면 새 커밋을 브랜치 head 위에 쌓아 PR에서 손으로 고친 다른 파일을 지킵니다. 열린 PR이 없으면(병합, squash 병합, 닫힘) 기준 브랜치에서 다시 시작합니다.
- blob SHA를 로컬에서 계산해 바뀐 파일만 올리고, 내용이 같으면 커밋하지 않습니다. 기준 브랜치가 이미 같은 글을 갖고 있으면 PR을 만들지 않습니다. 저장소 트리는 동기화마다 한 번씩 읽고, 목록이 잘렸으면(`truncated`) 실패합니다.
- 이 브랜치에만 있는 예전 글 사본과 그 글들의 번호 이미지 중 이번 실행이 만들지 않은 것은 지웁니다. 예전 사본은 슬러그가 같거나(게시일만 다름), front matter `date`(논문 발행 시각)와 종류가 같은 글(새 arXiv 버전에서 제목이 바뀐 이전 슬러그)입니다. 요약 슬러그는 `-summary`로 끝나므로 같은 논문의 리뷰와 요약은 서로 지우지 않습니다. 같은 종류이고 발행 시각까지 같은 글을 손으로 추가했다면 예전 사본으로 보고 지웁니다.
- 알려진 한계: 병합된 뒤 새 arXiv 버전에서 제목이 바뀌면 기준 브랜치의 예전 글을 슬러그로 찾지 못해 새 URL의 글이 하나 더 생깁니다. 다시 실행한 글의 이미지가 줄어도 기준 브랜치에 있던 남는 번호 이미지는 지우지 않습니다.
- 같은 슬러그의 글이 이미 기준 브랜치에 있으면 그 파일을 갱신하고, PR 체크리스트에 "Replaces the published post" 항목을 넣어 손으로 고친 내용이 사라지지 않는지 확인하게 합니다.
- 열린 PR이 있으면 제목과 본문을 갱신하고 URL을 돌려줍니다. 클라이언트는 30초 제한 시간과 `GithubRetry(total=3)`을 쓰고, 오류 메시지에서 토큰을 지웁니다.

PR 본문(`build_pr_body`)에는 논문 ID, PDF URL, 파싱 방식, 글자 수, 저장소, 블로그에 아직 없는 새 태그, 검증 warning 체크리스트가 들어갑니다.

## 8. LLM 런타임

### 8.1 모델과 호출

`utils/factories.py::BedrockLanguageModelFactory.get_model`이 모든 채팅 모델을 `ChatBedrockConverse`로 만들고, 리전별 inference profile을 찾아 캐시합니다. 프로필 조회 자체가 실패하면 기본 ID로 대체하지 않고 오류를 냅니다. 모델 특성은 `utils/models.py`의 레지스트리에 있습니다.

| 모델 | 단계 기본값 | thinking |
| --- | --- | --- |
| Claude Opus 5.5 | 리뷰 계획·집필·검수·편집·마무리, 요약, 가이드 작성·평가 | 끌 수 없음. effort로 조절 |
| Claude Sonnet 5.5 | 인용 추출, 가이드 개요 | 끌 수 없음. effort로 조절 |
| Claude Haiku 5.5 | 그림 분석, 속성 추출, 인용 요약, 코드 분석, 가이드 관련성·분류 | 적응형 |
| Claude Fable 5.1 | 선택 사항 | 항상 켜짐 |

`thinking_always_on` 모델에는 `thinking: disabled`와 temperature를 보내지 않고, 설정한 effort(`output_config.effort`)를 항상 보냅니다. 구형 모델은 `budget_tokens` 방식을 씁니다. 호출당 출력 상한 기본값은 32,000토큰(`DEFAULT_MAX_TOKENS`)입니다.

### 8.2 재시도와 제한 시간

재시도는 한 계층만 소유합니다. botocore는 standard 모드에서 1회만 시도하고(연결 30초, 읽기 3600초: 비스트리밍 호출이 출력 128k 토큰을 끝까지 생성하는 시간), `utils/retry.py::RetryableBase._retry`가 tenacity로 throttling, 5xx, 연결·읽기 시간 초과만 지수 백오프와 지터로 재시도합니다. 읽기 시간 초과는 긴 생성 전체를 다시 과금하므로 한 번만 재시도합니다. AccessDenied, ValidationException, ResourceNotFound, 예산 초과, 입력 검증 오류는 바로 실패합니다. `utils/batch.py::BatchProcessor`(코드 분석 일괄 처리)는 실패한 항목만 다시 처리합니다.

### 8.3 예산과 비용

`metrics.py::TokenUsageTracker`가 콜백으로 모든 호출의 토큰을 모델별로 모으고, 레지스트리의 1M 토큰당 단가로 비용을 추정합니다(`estimated_cost_usd`). 단가는 Anthropic 목록가이며 Bedrock 청구액과 다를 수 있습니다. `TokenBudgetGuard`는 단계마다 `max_total_tokens`를 검사해 넘으면 `TokenBudgetExceeded`로 멈춥니다. 가이드는 상한에 이르면 남은 절을 쓰되 평가·수정·사실 확인을 건너뛰고, 1.5배를 넘으면 중단합니다.

## 9. 설정 참조

`scholar_lens/configs/config.yaml`을 `config.py::Config.load`가 읽습니다. 모든 섹션은 `extra="forbid"`라 모르는 키는 오류입니다. AWS 안에서 파일이 없으면 실패하고, 로컬에서는 기본값을 씁니다. `config-template.yaml`이 모든 키와 기본값을 보여 줍니다.

### `resources`

| 키 | 기본값 | 설명 |
| --- | --- | --- |
| `project_name` | `scholar-lens` | 리소스·SSM 경로 이름 |
| `stage` | `dev` | `dev` 또는 `prod`. prod는 KMS 키와 로그를 보존 |
| `profile_name` | 없음 | 로컬 AWS 프로필 |
| `default_region_name` | `ap-northeast-2` | 스택·S3·SNS 리전 |
| `bedrock_region_name` | `us-west-2` | Bedrock 리전 |
| `s3_bucket_name` | 없음 | 글·이미지·결과 매니페스트 버킷. 미리 만들어 둔 버킷이어야 하며 배포에 필수 |
| `s3_prefix` | `scholar-lens` | S3 키 접두어 |
| `vpc_id`, `subnet_ids` | 없음 | 기존 VPC 사용(없으면 스택이 생성) |
| `email_address` | 없음 | SNS 이메일 구독 |

### `resources.github`

| 키 | 기본값 | 설명 |
| --- | --- | --- |
| `enabled` | `false` | 블로그 PR, 블로그 spec·태그 읽기 |
| `repo_name` | 없음 | `owner/owner.github.io` |
| `base_branch` | `main` | PR 대상 브랜치, spec을 읽는 브랜치 |
| `branch_prefix` | `paper-reviews` | PR 브랜치 접두어 |
| `author_name`, `author_email` | `Scholar Lens Bot`, 없음 | 커밋 작성자 |
| `review_category`, `summary_category`, `tech_guide_category` | `Paper Reviews`, `Paper Summaries`, `Tech Guides` | `categories[0]`. spec `types`에 있어야 함 |
| `public_assets` | `false` | S3 이미지 public-read |

### `paper`, `code`, `citations`

| 키 | 기본값 |
| --- | --- |
| `paper.figure_analysis_model_id` | Haiku 5.5 |
| `paper.citation_extraction_model_id` | Sonnet 5.5 |
| `paper.attributes_extraction_model_id` | Haiku 5.5 |
| `code.code_analysis_model_id`, `code.code_summarization_model_id` | Haiku 5.5 |
| `code.embed_model_id` | `amazon.titan-embed-text-v2:0` |
| `code.chunk_size`, `code.chunk_overlap` | 1024, 256(overlap < size) |
| `citations.citation_summarization_model_id`, `citations.citation_analysis_model_id` | Haiku 5.5 |
| `citations.prefer_full_text` | `false` |

### `explanation`(리뷰)

| 키 | 기본값 | 설명 |
| --- | --- | --- |
| `paper_planning_model_id` | Opus 5.5 | 계획과 전체 편집 |
| `paper_synthesis_model_id` | Opus 5.5 | 절 집필 |
| `paper_evaluation_model_id` | Opus 5.5 | 절 검수 |
| `paper_finalization_model_id` | Opus 5.5 | subtitle, TL;DR, 한계와 시사점 |
| `evaluator_enable_thinking`, `synthesizer_enable_thinking` | `false` | thinking을 켤 수 있는 모델에서만 의미 |
| `thinking_effort` | `medium` | 절 단위 호출 effort |
| `planning_effort` | `high` | 계획·편집 effort |
| `min_quality_score` | 70 | 검수 통과 점수 |
| `max_synthesis_attempts` | 3 | 절당 초안 수(1–6) |
| `max_sections` | 20 | 절 상한(1–60) |
| `max_edit_rounds` | 2 | 전체 편집 횟수(0–3, 0이면 생략) |
| `max_total_tokens` | 8,000,000 | 실행당 토큰 상한(LoRA 길이 논문 약 140만 토큰) |

### `summary`, `tech_guide`, `output_language`

| 키 | 기본값 |
| --- | --- |
| `summary.summary_model_id` | Opus 5.5 |
| `summary.summarizer_enable_thinking`, `summary.thinking_effort`, `summary.max_total_tokens` | `false`, `medium`, 1,000,000 |
| `tech_guide.relevance_model_id` | Haiku 5.5(관련성, 조사 계획, 분류) |
| `tech_guide.synopsis_model_id` | Sonnet 5.5 |
| `tech_guide.writing_model_id`, `tech_guide.evaluation_model_id` | Opus 5.5 |
| `tech_guide.writer_enable_thinking`, `tech_guide.evaluator_enable_thinking` | `false`, 없음(작성자 설정 따름) |
| `tech_guide.thinking_effort` | `medium` |
| `tech_guide.verify_grounding`, `tech_guide.auto_research` | `true`, `true` |
| `tech_guide.max_research_queries`, `tech_guide.fetch_top_results` | 6, 4 |
| `tech_guide.min_quality_score`, `tech_guide.max_revision_attempts` | 75, 2 |
| `tech_guide.max_total_tokens` | 3,000,000 |
| `output_language` | `Korean` |

### 환경 변수

`.env.template`에 있습니다. `GITHUB_TOKEN`(블로그 저장소 contents·pull requests 쓰기), `UPSTAGE_API_KEY`(PDF 파싱), `TAVILY_API_KEY`·`BRAVE_API_KEY`(선택), `LANGCHAIN_*`(선택), `LOG_LEVEL`, `AWS_PROFILE_NAME`(배포·비밀값 기록 시 `profile_name`이 없을 때)입니다. Batch에서는 같은 비밀값을 SSM에서 읽고, `TOPIC_ARN`은 스택이 주입합니다.

## 10. 인프라

`scripts/deploy_infra.py::PaperReviewStack`(CDK, Python) 구성입니다. `cdk.json`은 `poetry run python scripts/deploy_infra.py`를 실행합니다.

- **네트워크**: 기존 VPC를 쓰거나 NAT 게이트웨이 1개인 VPC를 만듭니다. 새로 만들면 S3 게이트웨이 엔드포인트를 붙입니다. 보안 그룹은 443 아웃바운드만 엽니다.
- **Batch**: 작업 정의 하나(`<project>-<stage>-worker`). 명령은 `JOB_COMMAND`이고 메모리는 8 GiB, 시간 제한은 3시간입니다. 재시도는 최대 3회지만 스팟 회수(`SPOT_INSTANCE_RECLAIMED`)와 `Host EC2*` 사유일 때만 하며, 나머지는 바로 종료합니다. 작업 큐는 온디맨드와 스팟 컴퓨팅 환경을 순서대로 씁니다. 실행 템플릿은 IMDSv2(hop limit 1)를 강제합니다.
- **IAM**: 작업 역할, 실행 역할, 인스턴스 역할을 분리합니다. 작업 역할 권한은 다음과 같습니다.
  - Bedrock 호출과 CountTokens
  - S3 접두어 Get/Put. `s3:PutObjectAcl`은 `public_assets`일 때만
  - SNS 게시와 토픽 KMS 키 사용
  - `cloudwatch:PutMetricData`(`ScholarLens` 네임스페이스 조건)
  - SSM 비밀값 5개 읽기(`SECRET_SSM_PARAMS`)
- **SSM**: 스택은 `batch-job-queue`, `batch-job-definition`(String)을 씁니다. 비밀값(SecureString)은 CloudFormation이 만들 수 없으므로 `scripts/put_secrets.py`로 따로 기록합니다. `cdk synth`/`diff`는 비밀값을 건드리지 않습니다. 실행 시 기록되지 않은 비밀값(`ParameterNotFound`)은 조용히 건너뛰고, 권한 오류 등은 warning으로 남깁니다.
- **설정 파일**: 이미지에는 배포하는 체크아웃의 `scholar_lens/configs/config.yaml`이 들어갑니다. 설정을 고치면 `cdk deploy`를 다시 해야 Batch에 반영됩니다. `deploy_infra.py::check_deployable`은 이 파일이나 `s3_bucket_name`이 없으면 배포를 멈춥니다. `cdk` 명령에는 `resources.profile_name`과 같은 `--profile`을 넘깁니다.
- **알림**: KMS 암호화 SNS 토픽(SSL 강제)에 이메일을 구독합니다. 토픽 정책은 EventBridge와 CloudWatch(`aws:SourceAccount` 조건)의 게시를 허용합니다.

## 11. 관측성

- **로그**: `logger.py`가 stderr로 쓰고, 로컬에서는 `logs/`에도 남깁니다. 모든 실행 끝에 `run_metrics` 한 줄(종류, 성공, 토큰, 호출 수, 비용, 시간)을 남깁니다.
- **지표**: `metrics.py::MetricsEmitter.emit_run`이 `ScholarLens` 네임스페이스에 `InputTokens`, `OutputTokens`, `TotalTokens`, `LLMCalls`, `EstimatedCostUSD`, `DurationSeconds`, `Success`, `Failure`를 `Mode` 차원이 있는 것과 없는 것 두 벌로 보냅니다(AWS 안에서만).
- **경보**:
  - 1시간 추정 비용 합계 50달러 이상, 경보 1개
  - EventBridge의 Batch `FAILED` 이벤트를 SNS로 전달. 앱이 직접 알리고 종료 코드 10·11로 끝난 작업은 제외하고, 그 밖의 종료 코드(시간 초과·중단 143, OOM·비정상 종료, worker 실행 전 Python·인자 오류 1·2, 알림 실패 4)와 종료 코드가 없는 경우(호스트 유실)는 알림
  - 실행 결과는 앱 알림과 이 규칙이 다루므로 로그 `ERROR` 경보는 두지 않습니다. 실행을 멈추지 않는 문제는 warning 로그로 남깁니다. PR 생성 실패처럼 결과에 남는 경고는 알림에 모두 적고, 알림·매니페스트 저장 실패는 로그에만 남습니다(실패 실행의 알림 실패는 위의 종료 코드 4로 보고). 성공한 실행의 알림이 실패하면 작업은 0으로 끝나 아무도 알리지 않지만, 결과 매니페스트가 있어 `status`로 결과를 볼 수 있습니다.
- **결과 매니페스트**: 원격 실행의 최종 상태는 S3 매니페스트와 `status` 명령으로 확인합니다.

## 12. 테스트와 CI

테스트는 모두 `tests/`에 있고 AWS·네트워크 없이 돕니다. `conftest.py`가 루프백이 아닌 소켓 연결을 막고, AWS는 moto, HTTP는 `responses`·httpx MockTransport, GitHub는 가짜 저장소 객체로 대신합니다. 주요 묶음은 다음과 같습니다.

| 파일 | 대상 |
| --- | --- |
| `test_jobs.py`, `test_cli.py`, `test_runtime.py`, `test_infra_contract.py` | CLI·작업 계약과 종료 코드, worker 실패 보고, 실행 수명주기, CDK 작업 명령·실패 알림 규칙과 CLI의 일치 |
| `test_review_plan.py`, `test_paper_reviewer.py` | 계획 보정, 리뷰 오케스트레이션(가짜 체인) |
| `test_e2e_pipeline.py` | 요약·리뷰·가이드 전 과정. 실제 모델이 내는 결함을 섞은 가짜 출력이 블로그 규칙을 통과하는 글이 되는지, 고칠 수 없는 글은 막히는지 |
| `test_post_validator.py`, `test_tags.py`, `test_post_conventions.py` | 검증 규칙, 태그 정규화, 블로그 규칙·기존 태그 읽기. `tests/fixtures/blog_defects/`의 블로그 결함 원본은 error, 수정본은 통과 |
| `test_publisher.py` | 이미지 로컬화, 검증 관문, GitHub 브랜치·PR 동기화 |
| `test_markdown_math.py`, `test_markdown_lint.py`, `test_utils_helpers.py`, `test_parser_figures.py` | 후처리, HTML 표·수식 직렬화, 그림 처리 |

CI(`.github/workflows/ci.yml`)는 Action을 커밋 SHA로 고정하고 다음 작업을 실행합니다.

| 작업 | 내용 |
| --- | --- |
| `lint` | ruff, black `--check`, mypy(`scholar_lens`, `scripts`) |
| `test` | Python 3.12·3.13, pytest와 커버리지(`scholar_lens`, `scripts`) |
| `docker-build` | Batch 이미지 빌드, 이미지 안에서 `python3 -m scholar_lens --help`와 모든 파이프라인 모듈 import |
| `cdk-synth` | `scripts/ci_synth_check.py`: 가짜 계정 synth, 작업 정의 1개·명령 일치·재시도 규칙·Lambda 없음 확인 |
| `post-conventions` | `scripts/check_post_conventions.py`: 사본과 블로그 정본 비교. `vars.BLOG_REPO`가 없으면 건너뜀. 토큰이 있는데 정본이 없으면(404) 실패하고, 토큰이 없으면 비공개 저장소와 구분할 수 없어 건너뜀 |

CI는 `main` push, 모든 PR, 매주 월요일 일정(블로그 규칙 변경 감지)에 돕니다. `workflow-lint.yml`은 체크섬을 검증한 actionlint로 워크플로를 검사하고, Dependabot은 GitHub Actions, pip, Docker를 매주 확인합니다.

Docker 이미지(`Dockerfile`)는 `python:3.12-slim` 기반 2단계 빌드입니다. 의존성은 `poetry.lock`에서 `--only main`으로 설치하고, 런타임에는 `git`, `tzdata`, `fonts-dejavu-core`(SVG 그림 래스터화용 글꼴)를 더하며, NLTK `punkt_tab`을 빌드 때 받아 `NLTK_DATA`로 지정하고 non-root로 실행합니다. `.dockerignore`는 허용 목록 방식이라 `pyproject.toml`, `poetry.lock`, `README.md`, `assets/post_conventions.yml`, `scholar_lens/`만 빌드 컨텍스트에 들어갑니다.

## 13. 확장 지점

- **새 산출물 종류**: `jobs.py::ArtifactKind`에 값을 더하고, 파이프라인 모듈을 만들어 `runtime.py::_run_pipeline`에서 분기하며, `post_validator.py`에 그 종류의 규칙을 정합니다.
- **새 모델**: `constants.py::LanguageModelId`와 `utils/models.py`의 레지스트리(컨텍스트, 출력 상한, thinking 방식, 단가)에 추가합니다.
- **새 검색 제공자**: `web_research.py::WebSearchProvider`를 구현하고 `guide_pipeline.py::build_search_provider`에 연결합니다.
- **블로그 규칙 변경**: 블로그의 `_data/post_conventions.yml`을 고치고 `assets/post_conventions.yml`에 같은 내용을 복사합니다. 새 규칙 종류가 필요하면 `post_conventions.py`의 모델과 `post_validator.py`에 함께 추가합니다.
