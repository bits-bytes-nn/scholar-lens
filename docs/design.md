# Scholar-Lens 설계 문서

현재 코드의 동작을 설명하는 기술 참조 문서입니다. 설치와 CLI 사용법은 [README](../README.ko.md)에 있고, 이 문서는 그 뒤에서 무엇이 어떻게 돌아가는지를 다룹니다. 코드 위치는 `파일::심볼` 형식으로 적습니다.

## 목차

1. [개요와 데이터 흐름](#1-개요와-데이터-흐름)
2. [CLI와 작업 계약](#2-cli와-작업-계약)
3. [리뷰 워크플로](#3-리뷰-워크플로)
4. [요약, 기술 가이드, 논문 서베이 파이프라인](#4-요약-기술-가이드-논문-서베이-파이프라인)
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

Scholar-Lens는 네 종류의 글(리뷰, 요약, 기술 가이드, 논문 서베이)을 만듭니다. 서베이는 계획과 쓰기 두 단계입니다(4.3절).

| 종류 | `kind` | 입력 | 파이프라인 |
| --- | --- | --- | --- |
| 논문 리뷰 | `review` | arXiv ID 또는 PDF URL, 선택적 코드 저장소 | `paper_pipeline.py::run_paper_pipeline` → `explainer.py::PaperReviewer` |
| 논문 요약 | `summarize` | arXiv ID 또는 PDF URL | `paper_pipeline.py::run_paper_pipeline` → `summarizer.py::PaperSummarizer` |
| 기술 가이드 | `guide` | 문서 URL 목록, 참고 자료 URL(`--reference`), 선택적 검색어 | `guide_pipeline.py::run_guide_pipeline` → `tech_guide.py::TechGuideGenerator` |
| 논문 서베이 계획 | `survey_plan` | 주제, 필수 논문(`--paper`), 연도 범위 | `survey_pipeline.py::run_survey_plan_pipeline` → 계획 파일(JSON, 글 아님) |
| 논문 서베이 | `survey_write` | 계획 파일(로컬 경로 또는 `s3://`) | `survey_writer.py::run_survey_write_pipeline` → `SurveyWriter` |

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
| `src/survey_pipeline.py`, `src/survey_plan.py`, `src/scholarly_index.py` | 서베이 계획 단계(후보 수집·모델 호출), 결정적 병합·순위·검증·계획 스키마, OpenAlex·DataCite·Semantic Scholar 어댑터 |
| `src/survey_writer.py`, `src/survey_post.py` | 서베이 쓰기 단계(계획 재검증, 카드 캐시, 모델 호출, 발행), 결정적 카드·개요 검증, 길이 목표, 인용 키, 표, 그림 출처, 참고 문헌 |
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

`cli.py::build_parser`가 하위 명령 여섯 개를 정의합니다(`survey`는 `plan`·`write` 두 단계).

| 명령 | 동작 |
| --- | --- |
| `review SOURCE`, `summarize SOURCE` | 논문 산출물을 로컬에서 실행하거나 `--remote`로 제출 |
| `guide [URL...] [--reference URL]...` | 기술 가이드를 로컬에서 실행하거나 `--remote`로 제출(문서 URL과 참고 자료 URL 중 하나 이상) |
| `survey plan TOPIC [--paper ID]... [--reference URL]... [--since Y] [--until Y]` | 서베이 계획 파일을 만들거나 `--remote`로 제출(4.3절). 글을 쓰지 않으므로 `--no-pr`가 없습니다 |
| `survey write --plan PATH_OR_S3_URI [--no-pr]` | 계획 파일로 서베이 글을 쓰거나 `--remote`로 제출(4.3절). `--remote`이면 로컬 계획을 먼저 `<s3_prefix>/surveys/<slug>/plan-<sha256 앞 16자>.json`에 올리고(내용별 키라 덮어쓰지 않음, `survey_writer.py::stage_plan_for_batch`) 그 URI로 제출합니다 |
| `status JOB_ID [--wait] [--timeout S]` | Batch 상태와 결과 매니페스트 조회(`jobs.py::describe_job`, `wait_for_job`) |
| `worker --request JSON` | Batch 진입점. `AWS_BATCH_JOB_ID`를 작업 ID로 사용 |

`guide`의 문서 URL은 옵션 앞·사이·뒤 어디에 와도 됩니다(`guide DOC1 --reference R DOC2`). argparse는 하위 명령이 있는 파서에서 위치 인자와 옵션을 섞어 받지 못하므로 `cli.py::parse_args`가 `guide` 하위 파서로 `parse_intermixed_args`를 호출합니다.

stdout에는 JSON 객체 하나만 출력하고 로그는 stderr로 보냅니다. 인자 누락 같은 사용법 오류도 사용법은 stderr로, `{"status": "rejected"}`는 stdout으로 내고 2로 끝납니다. 종료 코드의 정본은 `cli.py` 모듈 docstring이고, `JobResult.status`의 `succeeded`·`failed`·`rejected`가 각각 0·1·2입니다(`_EXIT_BY_STATUS`). 로컬 실행이 설정·자격 증명 문제로 시작하지 못해도 `failed`(1)입니다.

`--remote`는 `SubmittedJob`을 출력하고 0으로 끝납니다. `status`는 결과 매니페스트가 있으면 그 상태의 종료 코드를 돌려줍니다. 없으면 Batch가 `FAILED`일 때 1, 아직 끝나지 않았거나 `--wait`가 시간 초과되면 3, `SUCCEEDED`인데 매니페스트가 없으면 0입니다. `--wait` 중 Batch 조회나 결과 매니페스트 읽기의 요청 제한·5xx·네트워크 오류는 건너뛰고 계속 조회합니다.

`--remote`나 `status`가 실패하면(자격 증명, 설정, Batch·S3 API 오류) `{"status": "error"}`와 종료 코드 4를 냅니다. 다만 `survey write --remote`가 올릴 계획을 읽을 수 없으면(파일 없음, JSON 아님, 버킷 없음) 입력 문제이므로 `{"status": "rejected"}`와 종료 코드 2입니다. 작업 실패(1)와 구분하기 위해서이며, 이때 원격 작업의 상태는 알 수 없습니다.

### 2.2 JobRequest와 JobResult

`jobs.py::JobRequest`는 실행 하나를 완전히 기술합니다. `extra="forbid"`이며, 논문 요청은 `source`를, 가이드 요청은 `urls`(문서)나 `reference_urls`(참고 자료) 중 하나 이상을 요구하고 섞어 쓰면 거부합니다.

| 필드 | 대상 | 기본값 |
| --- | --- | --- |
| `kind` | 전체 | 필수(`review`, `summarize`, `guide`, `survey_plan`, `survey_write`) |
| `source` | review, summarize | 필수 |
| `repo_urls`, `parse_pdf` | review, summarize | `[]`, `false` |
| `urls`, `reference_urls`, `search_queries`, `discover_subpages` | guide | `[]`, `[]`(둘 중 하나 이상 필수), `[]`, `true` |
| `topic`, `paper_ids`, `reference_urls`, `since_year`, `until_year` | survey_plan | `topic` 필수, `[]`, `[]`, 없음, 없음. 다른 종류에 `topic` 등을 주면 거부하고, `since_year`가 `until_year`보다 늦거나 `paper_ids`에 arXiv ID·URL·DOI가 아닌 값, `reference_urls`에 http(s) URL이 아닌 값이 있어도 거부합니다(`survey_plan.py::parse_paper_id`, Batch 제출 전) |
| `plan` | survey_write | 필수. 로컬 경로 또는 설정한 버킷의 `s3://<bucket>/<key>`. `plan`과 `create_pr` 밖의 필드를 주면 거부하고, 다른 종류에 `plan`을 주면 거부합니다 |
| `create_pr` | 전체 | `true`(GitHub 연동이 켜져 있을 때만 의미, survey_plan은 쓰지 않음) |

서베이 두 종류는 자기 행의 필드와 `create_pr`만 받습니다. 다른 행의 필드에 기본값이 아닌 값(`parse_pdf: true`, `discover_subpages: false` 포함)을 주면 거부합니다.

로컬 CLI와 Batch worker는 같은 객체를 씁니다. `--remote`는 `JobRequest.model_dump_json(exclude_defaults=True)`를 Batch 파라미터 `request` 하나로 넘기고, 작업 정의의 명령은 `["python3", "-m", "scholar_lens", "worker", "--request", "Ref::request"]`입니다(`scripts/deploy_infra.py::JOB_COMMAND`). 두 경로가 같은 파서를 쓰므로 인자 계약이 어긋나지 않습니다. 기본값인 필드는 보내지 않으므로, 새 필드를 쓰지 않는 요청은 그 필드가 생기기 전에 배포한 worker 이미지(`extra="forbid"`)도 받습니다. 새 필드(예: `reference_urls`)를 쓰는 요청은 `cdk deploy`로 이미지를 갱신한 뒤에만 원격 실행됩니다(`tests/test_jobs.py`가 `reference_urls` 이전 worker의 필드 집합 사본으로 검증).

`jobs.py::JobResult`에는 `status`, `title`, `markdown_path`, `plan_path`(로컬 survey_plan 실행의 계획 파일, 이때 `s3_uri`는 그 S3 사본. Batch 실행은 컨테이너와 함께 사라지는 경로라 비우고 `s3_uri`만 남김), `s3_uri`, `pr_url`, `warnings`, `violations`, `error`, `job_id`, `duration_seconds`, 토큰 수, `estimated_cost_usd`가 들어갑니다. Batch 실행은 이 결과를 `s3://<bucket>/<s3_prefix>/results/<job_id>.json`에 저장하고(`jobs.py::write_result_manifest`), `status`가 이를 읽어 붙입니다.

### 2.3 실행 수명주기

`runtime.py::run_job` 순서는 다음과 같습니다.

1. `Config.load()`와 `build_context`로 boto 세션(기본 리전, Bedrock 리전)과 S3 핸들러를 만듭니다.
2. AWS 안에서는 SSM SecureString에서 비밀값을 환경 변수로 읽습니다(`load_secrets_from_ssm`). 로컬은 `.env`를 씁니다.
3. 파이프라인을 실행합니다. 입력 오류는 `rejected`, `PostValidationError`는 `failed`와 `violations`, `PostAssetError`(논문 그림을 저장하지 못함)는 `failed`, 그 밖의 예외는 `failed`로 기록합니다. 앞의 두 발행 오류는 `markdown_path`·`s3_uri`에 거부된 초안(`<slug>.rejected.md`)의 위치를 담습니다. `SystemExit`·`KeyboardInterrupt`는 정리 후 다시 던집니다. 로컬 실행의 Ctrl-C는 `cli.py::main`이 받아 `INTERRUPTED_ERROR`인 `failed` 결과를 JSON으로 출력하고 1로 끝냅니다.
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

용어 규칙(`prompts/base.py::TERMINOLOGY_RULES`)은 계획, 집필, 검수, 편집, 마무리, 요약과 가이드 프롬프트가 함께 씁니다. 번역 지침(`assets/translation_guideline.json`, 로컬에 없으면 S3 사본)에 있는 용어는 지침의 표기를 따르고("어텐션", "트랜스포머"처럼 음차한 표기도 포함), 그 밖의 기술 용어는 자연스러운 번역이 있으면 번역합니다. 고유명사와 제품·API·모델·코드 이름, 자연스러운 번역이 없는 용어는 영어로 둡니다. 용어집에는 표기만 적고 괄호를 붙이지 않으며, 집필자가 글에서 처음 쓸 때 한 번만 영어를 괄호로 덧붙입니다("연속 배칭(continuous batching)"). 영어 단어에 한국어 어미를 붙이지 않으며("prefill합니다" 대신 "prefill을 수행합니다"), 이 규칙은 코드로도 찾습니다(6.2). 번역 지침(`runtime.py::load_translation_guideline`)은 리뷰의 계획·집필, 요약, 가이드의 개요 호출이 받고, 리뷰의 검수·편집·마무리와 가이드의 집필·평가는 지침을 반영한 용어집을 받습니다.

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

`PaperFinalizationPrompt`가 편집된 글에서 네 가지를 만듭니다.

- `subtitle`: 제목의 출력 언어 번역
- `description`: front matter의 검색 설명. 모델·방법 이름으로 시작하는 120–160자 평문, 수치는 TL;DR이나 초록에 있는 것만
- `key_takeaways`: `## TL;DR` 아래 `###` 질문 4개, 질문마다 2–4문장 또는 2–4개 글머리표와 핵심 수치
- `closing`: `## 한계와 시사점`(강점, 구체적 한계, 적용 시점, 확실한 후속 연구)

`_finalization_problems`가 구조와 함께 수치를 코드로 확인합니다. TL;DR의 모든 수치는 논문 본문에, 설명의 모든 수치는 남기는 TL;DR(시도마다 부분별로 남기므로 그 시도의 TL;DR이 아닐 수 있음)이나 초록에 있어야 하고(`post_validator.py::unsourced_numbers`, 6.3), 설명은 `description_problems`(평문, 길이)를 통과해야 합니다. 마무리 프롬프트는 초록도 입력으로 받습니다. 문제가 있으면 한 번 다시 요청하고, 그래도 남으면 PR 메모로 남깁니다. 마크업이 남은 설명과 페이지 `<title>`(논문 제목, 한국어 글은 " 논문 리뷰"를 붙인 길이, `post_validator.py::paper_page_title`)보다 길지 않은 설명은 버려 블로그가 만들게 하고 PR 메모로 남깁니다. `run()`은 `ReviewText(body, key_takeaways, closing, subtitle, description)`을 돌려주고, `formatting.py::format_review`가 TL;DR → 본문 → 한계와 시사점 → 참고문헌 순서로 조립합니다.

### 3.6 호출 수와 한계

8개 절 논문에서 계획 1회, 절마다 집필·검수 약 2–3회, 편집 1–2회, 마무리 1회입니다. 계획과 편집은 논문 전체를 입력으로 받고, 컨텍스트를 넘으면 `fit_text`가 뒷부분을 잘라냅니다. 이때도 미배정 구간 보정이 남은 문장을 절에 붙입니다. 모든 단계는 `max_total_tokens` 예산 가드를 거칩니다.

## 4. 요약, 기술 가이드, 논문 서베이 파이프라인

### 4.1 요약

`summarizer.py::PaperSummarizer.summarize`는 파싱한 논문 전체(컨텍스트에 맞게 자름)와 코드베이스 요약으로 한 번 호출하고, 한국어 요약이 너무 길면 줄이는 호출을 한 번 더 합니다(`_shorten_if_long`, 구조가 깨지면 원문 유지). 출력 태그는 `subtitle`, `description`, `summary`, `urls`입니다. 다시 요청하는 단계가 없으므로 TL;DR(앞 인용문)의 수치가 논문에 없거나 설명이 규칙(3.5)을 어기면 PR 메모로 남기고, 마크업이 있거나 페이지 `<title>`보다 길지 않은 설명은 버립니다. 요약은 `> **요약:**` 인용문과 이모지 `##` 절 다섯 개(왜 필요한가, 해결책, 구현, 결과, 의의와 한계)로 이루어지고, 실증 논문이면 결과 절에 무게를 둡니다. `formatting.py::summary_with_tldr`가 앞의 인용문을 `## TL;DR` 절로 바꿔 블로그 규칙에 맞춥니다.

### 4.2 기술 가이드

`tech_guide.py::TechGuideGenerator.generate` 순서입니다.

1. 문서 URL(위치 인자, 0개 이상)과 범위 안 하위 페이지를 수집합니다(`web_research.py::WebResearcher.research`). 참고 자료 URL(`--reference`를 반복해 0개 이상, `JobRequest.reference_urls`: 논문, 설계 문서, 심층 분석 글, 발표 페이지)은 하위 페이지 없이 가져와 참고 자료로 표시합니다(`PageContent.user_reference`). 공식 문서의 호스트와 버전(`OfficialDocs`)은 문서 URL로만 정하므로 참고 자료는 공식 문서가 되지 않습니다. 단, 요청한 문서와 같은 호스트·호환 버전의 페이지는 참고 자료로 넘겨도 공식 문서로 다루고, 문서 URL과 같은 참고 자료 URL은 문서로만 다루고 출처 목록에도 한 번만 넣습니다. 같은 페이지를 여러 표기로 주거나(대소문자, 끝 `/`), 같은 arXiv 논문을 초록·PDF·HTML·버전별 링크로 주어도 한 번만 가져옵니다(`web_research.py::page_key`). 리디렉션으로 이미 가진 페이지에 도착한 참고 자료와 검색 결과도 다시 넣지 않습니다. 읽지 못한 참고 자료는 PR 체크리스트에 올립니다. 문서 URL 없이 참고 자료만 주면 공식 문서가 없는 가이드가 되며, 그 사실을 PR 체크리스트에 올리고 절 제목의 공식 사이트 조회를 하지 않습니다. arXiv 논문 링크(초록·PDF·HTML)는 검색 결과든 입력이든 논문의 HTML 렌더링(`arxiv.org/html/<id>`)으로 읽습니다(`web_research.py::arxiv_paper_urls`). 렌더링이 없거나(404) 변환 실패 안내문뿐이면(읽을 수 있는 글자가 파서의 `MIN_PARSED_TEXT_CHARS` 미만) 초록 페이지를 한 번 읽습니다. 429·5xx·시간 초과·연결 오류는 대체 페이지로 넘어가지 않고 가져오기 실패로 기록하므로, 모든 입력이 이런 이유로 실패하면 `SourceFetchError`(재시도 가능한 실패)가 됩니다. 이 계층은 재시도하지 않습니다. arXiv로 가는 모든 요청은 arXiv API 클라이언트와 같은 프로세스 전역 제한기(`rate_limiter.py::ARXIV_RATE_LIMITER`, 3초에 1회)를 거치고, 429를 받으면 이후 arXiv 요청을 `Retry-After`(없으면 30초)만큼 미룹니다. 논문의 SVG 그림(`<object type="image/svg+xml">`)도 `<img>`처럼 이미지 목록에 넣고, 본문 텍스트의 이미지 마커는 목록에 남은 이미지만 남깁니다. `--reference`는 HTML 페이지나 arXiv 링크여야 하며, arXiv가 아닌 PDF는 읽지 않습니다.
2. 관련성 판정(`TechGuideRelevancePrompt`)을 문서 페이지와 그 하위 페이지만으로(문서 URL이 없으면 참고 자료로) 먼저 합니다. 통과하지 못하면 `NotTechnicalContentError`로 거부하므로 검색 비용이 들지 않습니다. 같은 호출이 문서가 다루는 명령·패키지 이름(`<commands>`)도 돌려주며, 시드 페이지에 단어로 나오는 이름만 씁니다. 하나도 없으면 사이트 호스트 이름(`docs.vllm.ai` → `vllm`)을 씁니다(`web_research.py::product_names`).
3. 검색 제공자가 있고 검색어가 주어지지 않았으면 계획 단계가 문서와 참고 자료를 보고 검색어를 만들고(`auto_research`, `max_research_queries`, 기본 6개), 상위 결과 페이지를 말뭉치에 넣습니다(`fetch_top_results`, 기본 12). 결과는 모든 검색어의 i번째 결과를 i+1번째보다 먼저 고르므로, 기본값에서는 검색어마다 상위 두 결과(이미 가진 페이지는 건너뜀)가 들어옵니다. 계획 프롬프트는 검색어를 사용법(공식 문서), 내부 구조와 설계(논문, 설계 문서, 아키텍처 개요, 메인테이너의 심층 분석 글, 소스 해설), 운영 아키텍처와 연동(배포 패턴, 확장, 관측성, 장애 양상, 주변 시스템과의 인터페이스) 세 의도에 약 3분의 1씩 나누라고 지시하지만, 실제 배분은 모델이 정합니다. 참고 자료가 이미 설명한 내용은 다시 찾지 않습니다. 계획 단계와 관련성 판정이 보는 출처 요약(6,000자)은 참고 자료와 나머지 페이지가 각각 절반 이상을 쓸 수 있게 나눕니다. 계획 단계의 주제는 검색과 개요 입력에 쓰고, 가이드 제목은 개요 호출이 돌려주는 `<title>`을 씁니다(없으면 계획 주제). 개요·분류 프롬프트는 제목 길이 기준으로 규칙 파일의 `seo_title_max_length`를 받아(`TechGuideGenerator(title_max_length=…)`), 제목을 그 길이 안에서 도구와 가이드의 관점을 담고 절 목록을 나열하지 않게 지시합니다. 게시 단계에서 제목이 그 길이를 넘으면 분류 단계의 `seo_title`(기준 안일 때) 또는 낱말 경계에서 자른 제목(`formatting.py::cut_title`)으로 바꿉니다(`guide_pipeline.py::guide_title`). 콜론 앞부분(`vLLM`)은 도구 이름일 뿐이므로 가이드 제목으로 쓰지 않습니다. 논문 글의 `seo_title`은 콜론 앞의 논문 이름을 그대로 씁니다(`formatting.py::shorten_title`).
4. 개요를 만들고(최대 `max_sections`개 절, 기본 14, 3–30) 절마다 깊이(deep/standard/brief)를 정합니다. 분량 기준은 가이드 전체 약 35,000–55,000자(한국어 기준, 코드 제외), deep 절 약 5,000자, standard 약 2,500자, brief 1,000자 이하입니다(`prompts/tech_guide.py::LENGTH_BUDGET`). 분량은 공백을 하나로 줄인 산문 글자 수이며 코드 블록, 표 행(인용 블록 안의 표와 바깥 `|` 없이 `--- | ---` 구분 행을 둔 표 포함, 표 판별은 `markdown_lint.py`의 `TABLE_ROW`·`TABLE_DELIMITER`를 씀)과 이미지(링크로 감싼 이미지 포함)는 세지 않고, 이미지와 같은 줄의 글은 셉니다(`tech_guide.py::prose_chars`). 집필자 지시, 평가자의 측정 보고와 수정 지시가 같은 함수와 같은 단위(`PROSE_UNIT`)를 씁니다. 개요의 뼈대는 요청한 문서입니다. 절 구성은 문서의 구조와 독자의 진행 순서(`<input_outline>` 순서)를 따르고, 참고 자료와 검색 결과의 내부 구조·설계 이유·연동 내용은 별도 장을 만들지 않고 독자가 관련 기능이나 설정을 만나는 절에 넣습니다(메모리·캐시 설정을 다루는 절에서 메모리 메커니즘을 설명). 메커니즘에 따로 절을 주는 것은 핵심이고 나누면 흩어질 만큼 클 때뿐이며, 참고 자료 하나를 요약하는 절은 만들지 않습니다. 그 뼈대 안에서 출처가 뒷받침하는 범위의 폭과 깊이를 다룹니다. 도구가 푸는 문제와 핵심 개념, 핵심 메커니즘의 동작 방식과 설계 이유, 주요 기능별 의미(푸는 문제, 동작, 장단점, 쓸 때), 설치와 사용법, 더 큰 시스템에의 통합(인터페이스, 확장, 관측성, 장애 양상, 권고 형태의 설계 조언)입니다. 쉬운 내용은 짧게, 어려운 메커니즘을 설명하는 절은 배경부터 깊게 쓰며, 설치는 한 절 이하로 두고 플랫폼별 차이는 표나 공식 페이지 링크로 줄입니다. getting-started 유형은 사용법을 앞에 둡니다. 절은 사용·설정 내용에 공식 문서를, 그 절에 녹인 메커니즘·연동 내용에 참고 자료나 서드파티 설명 자료를 근거로 하며, `<section_plan>`에 그 자료를 적습니다. 집필자는 참고 자료를 따로 떼어 배경 설명으로 두지 않고 그 절의 기능 설명 안에서 가이드의 목소리로(푸는 문제 → 동작 → 설계 이유 → 여기서 쓰는 법) 풀고, 평가자는 참고 자료 하나를 요약한 듯한 절이나 해당 기능과 떨어진 메커니즘 설명을 구조(Structure & flow) 감점으로 봅니다. 개요 호출은 절마다 그 절을 뒷받침하는 말뭉치 페이지 URL과 그 절이 맡는 개념도 돌려줍니다(`<section_plan>`). URL은 말뭉치에 있는 것만 남기고, 계획의 번호는 개요의 번호 매긴 줄과 맞춥니다(`guide_authority.py::parse_section_plan`). 계획 줄이 없거나 공식 문서 페이지가 하나도 없는 절은 집필 전에 절 제목으로 공식 사이트를 도메인 제한 검색합니다. 검색 결과로 찾은 페이지만 그 절의 공식 출처로 치며, 검색 백엔드가 없거나 찾지 못하면 집필자에게 출처가 보여 주는 내용만 쓰라고 알립니다. 계획이 그 절에 고른 페이지와 제목 조회로 찾은 페이지는 계획 순서대로 절의 근거 자료 맨 앞에 두고, 각각 근거 자료 상한의 같은 몫을 먼저 받아 긴 논문이 공식 페이지를 밀어내지 않습니다. 남은 자리는 관련도 순으로, 잘린 우선 페이지부터 나눕니다(`ResearchCorpus.relevant_text`). 자기 몫보다 긴 페이지는 앞부분만 남기지 않고 헤딩 줄에서(그래도 긴 조각은 문단 경계에서, 코드 펜스와 `$$` 수식 안은 자르지 않음) 조각으로 나눕니다. 문단 경계에서 나뉜 절의 뒤쪽 조각을 남기면 그 절의 헤딩이 있는 첫 조각도 함께 남겨 "Deprecated options (removed in v0.10)" 같은 맥락을 잃지 않습니다. 머리말과 본문 사이에 빈 줄이 없거나 조각을 넣을 자리가 없으면 블록을 문단 경계에서 자릅니다. 조각마다 질의 단어를 담았는지 보되, 많은 조각에 나오는 단어(제품 이름, "and")일수록 가중치를 낮추고(역문서 빈도), 점수가 높은 조각을 몫 안에서 골라 페이지 순서대로 두며 빠진 부분은 `…` 줄로 표시합니다(`web_research.py::_excerpt`). 15만 자짜리 CLI 레퍼런스의 뒤쪽에 있는 병렬화·설정 플래그도 해당 절에 들어가게 하기 위해서입니다. 각 집필·평가 호출에는 다른 절이 맡은 개념 목록(`owned_elsewhere`, 그 절도 맡은 개념은 뺌)을 넘겨 다시 정의하지 않게 하고, 평가자는 그 개념을 설명한 것을 중복(Non-duplication)으로 보며 길게 설명하면 차단 문제로 둡니다. 개요 호출은 핵심 용어집(`<glossary>`, 최대 15개, 줄마다 `English term | rendering`)도 돌려줍니다. 표기는 용어 규칙(`prompts/base.py::TERMINOLOGY_RULES`)을 따른 한국어 용어나 고유명사·API의 영어이며 괄호 속 영어는 넣지 않고, 코드가 괄호를 떼어 파싱합니다(`tech_guide.py::parse_glossary`). 집필·평가·사실 확인 호출은 절마다 용어집을 받는데, 앞선 절 본문에 첫 사용 형태인 `표기(English)`(대소문자 무시, 괄호 앞 공백 허용)가 이미 나온 용어는 `[introduced]`, 아니면 `[new]`로 코드가 표시합니다. 표기만 나온 것은 소개로 보지 않습니다(`tech_guide.py::glossary_view`). `[new]` 용어만 그 절의 첫 사용에서 영어를 괄호로 한 번 붙입니다(`prompts/tech_guide.py::GLOSSARY_RULE`). 검색 결과, 사용자 참고 자료와 공식 레퍼런스 조회로 말뭉치에 들어온 페이지는 한 쪽마다 출처 라벨(official documentation, reference material, third-party, older docs version)과 URL을 INFO 로그로 남기고(`Added search page [third-party] https://…`), 집필 전에 절마다 계획된 출처를 한 줄로 남깁니다.
5. 절마다 작성 → 평가 → 기준 미달 시 수정(`max_revision_attempts`, 기본 1회) 순서로 진행합니다. 평가자는 절의 근거 자료와 전체 목차를 보고 점수(0–100)와 차단 문제(`<blocking_issues>`: 사실 오류, 절 안의 모순, 출처 권한이 요구하는 근거가 없는 플래그·기본값, 어느 출처도 뒷받침하지 않는 내부 구조·아키텍처 주장, 다른 절을 이름이나 번호로 가리키는 표현)를 돌려줍니다. 점수가 기준 이상이고 차단 문제와 코드로 찾은 규칙 위반이 없어야 통과하며, 수정 지시는 차단 문제(반드시 고칠 항목), 측정한 분량과 목표, 평가자의 나머지 의견 순서로 씁니다. 수정본은 원래 초안보다 순위가 높을 때만 남기며, 순위는 통과 여부, 게시 가능 여부, 평가자의 차단 문제 수, 점수, 규칙 위반 수 순서로 비교합니다. 모든 절을 쓴 뒤 사실 확인(`verify_grounding`)을 합니다. 수정 뒤에도 남은 평가자의 차단 문제는 그 절의 사실 확인에 넘겨, 근거 자료만으로 바로 그 주장만 고치거나 지우게 합니다(`blocking_issues`, 없으면 프롬프트는 그대로). 사실 확인 프롬프트는 용어 규칙 전체 대신, 절의 용어와 용어집 표기를 그대로 두고 영어 어간에 한국어 어미를 붙이지 말라는 한 줄만 둡니다. 공식 페이지에 없는 CLI 플래그·Python 인자(`gpu_memory_utilization=`)·메트릭 이름은 같은 공식 문서(같은 호스트, 호환되는 버전)의 레퍼런스 페이지로 확인합니다. 집필 전에는 서드파티 페이지에만 나오는 식별자와 공식 출처가 없는 절의 제목을, 사실 확인 전에는 초안에 새로 나온 식별자를 찾으며, 모든 조회를 합쳐 검색 3번, 6쪽까지 씁니다(`WebResearcher.add_official_reference`, 식별자 조회는 검색 뒤에 식별자와 맞는 경로의 링크도 씁니다). 식별자 조회에서는 찾는 식별자 중 하나 이상을 미확인 검사와 같은 표기 규칙(`guide_authority.py::mentions_any`)으로 담은 페이지만 말뭉치에 넣고 쪽수에 셉니다. 담지 않은 페이지(CLI 페이지 목록뿐인 색인 페이지 등)는 넣지 않고, 그 페이지의 레퍼런스 링크(같은 호스트, 호환되는 버전)를 같은 경로 순위로 후보 끝에 붙입니다. 가져오기는 조회마다 쪽수 한도의 2배까지입니다. 절 제목 조회는 검색 결과를 그대로 씁니다. 집필 전 조회(서드파티 식별자와 절 제목)는 합쳐서 검색 2번, 4쪽까지만 써서, 사실 확인에 검색 1번과 2쪽 이상이 남습니다. 그중 서드파티 식별자 조회는 절반(검색 1번, 2쪽)까지만 써서 절 제목 조회의 몫을 남깁니다. 가져온 레퍼런스 페이지는 그 절 본문에 나오는 미확인 식별자가 있는 절에만, 그 식별자와 절 앞부분으로 고른 발췌(2만 자, 긴 페이지는 위와 같이 조각 선택)로 사실 확인 근거 자료 앞에 붙입니다. 그래도 확인되지 않은 식별자는 PR 체크리스트에 올립니다. 출처 목록에는 요청한 페이지(문서와 참고 자료)와 같은 공식 문서(호환되는 버전)의 페이지를 넣고, 그 밖의 페이지는 어느 절의 근거 자료에 실제로 들어갔거나 본문이 링크하거나 이미지를 가져온 경우에만 넣습니다. 레퍼런스 페이지는 본문에 남은 식별자를 낱말 단위로 담은 경우에만 넣고, 이전 버전 문서는 넣지 않습니다(`guide_authority.py::used_source_urls`). 각 절의 호출에는 말뭉치 전체가 아니라 절 제목·설명과 가장 관련 있는 페이지만 약 6만 자까지 넣고(`ResearchCorpus.relevant_text`), 앞선 절은 직전 두 절 전문과 그 이전 절의 요약만 넘깁니다.

출처 권한은 집필·평가·사실 확인 프롬프트가 공유합니다(`prompts/tech_guide.py::SOURCE_AUTHORITY`). 프롬프트의 각 출처는 공식 문서(official documentation), 이전 버전 문서(older docs version), 참고 자료(reference material), 서드파티(third-party)로 표시됩니다(`web_research.py::source_label`). 명령, 플래그, API 이름, 기본값, 설정 키, 환경 변수, 메트릭 이름, 버전·플랫폼 지원은 공식 문서가 정합니다. 참고 자료와 서드파티 페이지(논문, 설계 문서, 메인테이너 글, 엔지니어링 글, 소스 코드)는 개념, 내부 구조, 설계 이유, 장단점, 아키텍처·연동 패턴의 근거가 될 수 있고, 1차 자료가 일반 튜토리얼보다 우선하며 마케팅 글은 근거가 되지 않습니다. 업체의 제품 페이지나 튜토리얼은 그 제품 자체의 사실에만 근거가 되고 다른 제품과의 비교 근거가 되지 않습니다. 논문이나 이전 설계 문서에만 나오는 메커니즘은 현재 공식 페이지가 확인하지 않는 한 현재 선택지가 아니라 그 문서가 설명하는 버전의 설계로(논문 이름이나 그 문서가 밝힌 버전을 적어) 씁니다. 현재 동작을 두고 공식 문서와 어긋나면 공식 문서가 이기고, 이전 버전 문서는 현재 동작을 정하지 않습니다. 근거 자료에 공식 페이지가 있으면 비공식 페이지에만 나오는 명령·플래그·기본값·설정 키는 쓰지 않고, 공식 페이지가 없으면 그 값을 보여 주는 페이지가 있어야 합니다. 평가자는 플래그·기본값에 이 근거를, 개념·내부 구조·아키텍처 주장에는 어느 출처든 근거나 교과서 수준 배경을 요구하며, 근거 없는 내부 구조·아키텍처 주장은 차단 문제입니다. 사실 확인은 출처가 뒷받침하는 개념 설명을 공식 페이지에 없다는 이유로 지우지 않고, 식별자는 계속 공식 문서로 확인합니다. 본문은 사실의 출처를 서술하지 않고, 의견, 벤치마크, 메커니즘이 나온 논문·설계 문서만 한 번 출처를 밝힙니다. 사실 확인이 절을 4분의 1 넘게 줄이면 로그(INFO)에 남깁니다. 참고 자료는 출처 목록에 항상 들어갑니다.

가이드 파이프라인(`guide_pipeline.py`)은 완성된 가이드를 `TechGuideGenerator.classify`로 분류해 블로그 주제 하나, 통제 태그, 세부 태그, `seo_title` 후보를 받고, `formatting.py::build_post_meta`가 spec에 맞게 확정합니다. 분류는 개요가 아니라 완성된 본문의 헤딩과 문단 첫 문장으로 하므로 본문에 없는 내용을 태그로 달지 않습니다. 목차는 블로그 레이아웃이 만들므로 본문에 따로 넣지 않습니다. 문서 사이트 같은 남의 이미지는 블로그로 복사하지 않고 원본 링크로 두며, 검증 warning으로 PR 체크리스트에 올라갑니다(7장).

### 4.3 논문 서베이

서베이는 Lilian Weng의 블로그처럼 한 연구 주제의 주요 논문을 계보(lineage, 같은 접근을 공유하는 논문 묶음)별 시간순으로 엮고, 논문끼리의 관계와 통찰을 정리하는 글입니다. 두 단계로 나눕니다. `survey plan`이 사용자가 고칠 수 있는 계획 파일을 만들고, 사용자가 논문·계보를 고친 뒤 `survey write --plan <file>`이 그 파일로 글을 씁니다. 반드시 넣을 논문은 `--paper`로, 참고할 자료(블로그 글, 기존 서베이·리뷰 글, 논문 페이지)는 `--reference`로 줍니다. 참고 자료는 범위를 잡는 데 쓰이고 거기서 링크한 논문은 후보가 되지만, 필수 논문은 아닙니다. 계획 단계의 모델 호출은 범위와 선정 두 번(각각 최대 한 번의 수정 호출)이고, 사실(서지 정보, 인용 관계, 순위)은 API와 코드가, 의미 판단(계보, 선정, 역할)은 모델이 맡습니다. 쓰기 단계는 4.3.1절입니다.

`survey_pipeline.py::plan_survey` 순서입니다.

1. **필수 논문**(`CandidateFinder.resolve_required`): `--paper`의 arXiv ID·URL은 DataCite에 등록된 arXiv DOI 레코드(`10.48550/arXiv.<id>`)에서, DOI는 OpenAlex에서 찾고, 같은 논문의 OpenAlex 레코드(arXiv DOI, 제목 검색)를 덧붙입니다. ID 형식이 틀리거나 색인이 모르는 논문은 `SurveyInputError`(rejected)이고, 색인 장애는 failed입니다.
2. **참고 자료**(`read_references`, `CandidateFinder.referenced_records`): `--reference` URL마다 한 번(같은 논문의 arXiv abs·pdf 링크와 arXiv DOI 링크, http·https 표기는 하나로) 가이드와 같은 `WebResearcher`로 하위 페이지 없이 읽습니다(SSRF 방지, arXiv 링크는 HTML 전문, 5.6·5.7절). arXiv 링크·doi.org 링크인 자료는 `paper`, 나머지는 `article`입니다(`survey_plan.py::reference_kind`). 자료에서 논문을 코드로 뽑습니다(`cited_paper_ids`): 본문 영역(`<article>`/`<main>` 하나, 없으면 페이지 전체)의 링크(`PageContent.content_links`, 사이트 내비게이션 제외) 중 arXiv abs·pdf·html 링크와 doi.org 링크(arXiv DOI는 arXiv ID로), 본문 글의 `arXiv:<id>`를 계획 ID로 바꾸고 중복과 자료 자신의 논문을 뺍니다. `paper` 자료는 그 논문 자체도 넣습니다(읽지 못해도). arXiv·doi.org 링크만 논문으로 보며, 다른 호스트의 논문 페이지(`huggingface.co/papers/…` 등)는 해석하지 않으므로 후보가 되지 않습니다. 이 ID들은 arXiv ID면 DataCite(50개씩), 그 밖의 DOI면 OpenAlex(`doi:` OR 필터, 50개씩)에서 레코드를 받아 `reference:<자료 URL>`을 붙여 후보에 더합니다. 색인이 모르거나 조회가 실패한 논문은 빠지고(실패는 다른 후보 조회처럼 `notes`), 링크된 논문 수와 찾은 수는 `sources`에 남습니다. 읽지 못한 자료는 실패가 아니라 `notes`에 남기고 계속합니다.
3. **범위**(`SurveyScopePrompt`, `survey.scope_model_id`): 주제, 사용자 연도 범위, 필수 논문 제목, 읽은 참고 자료의 발췌(`reference_excerpts`: 주제와 관련 큰 부분을 자료마다 같은 몫으로 모두 8,000자 안, `(reference material)` 표시. 프롬프트 태그는 모두 무력화, `prompt_safety.py::neutralize_prompt_tags`)를 받아 `<scope>` 안의 JSON 하나로 다듬은 주제 서술, 계보 2–6개(`id`, 이름, 한 줄 설명), 검색어 4–8개, 모델이 기억하는 주요 논문 제목 최대 10개(landmarks)를 돌려줍니다. `survey_plan.py::SurveyScope`가 검증하고, 형식이 틀리면 문제 목록과 함께 한 번 다시 묻습니다. 랜드마크 제목은 검색어로만 쓰며, 색인에서 제목이 일치하는(`citation_metadata.py::titles_match`) 레코드만 후보가 됩니다.
4. **후보 수집**(`CandidateFinder.gather`): 필수 논문과 참고 자료 논문의 레코드에 더해, 검색어마다 OpenAlex `search`(사용자 연도 범위로 필터)와 arXiv API(검색어의 모든 단어를 `all:`로 AND)에서 25건씩 받습니다. 랜드마크는 OpenAlex 제목 검색으로 찾고, arXiv ID가 있는 일치 레코드가 없을 때만 arXiv `ti:` 검색을 합니다. OpenAlex만 돌려준 arXiv ID는 DataCite에서 arXiv 메타데이터(제목, 저자, 초록, v1 제출일)를 50개씩 받아 확인합니다. OpenAlex는 일부 arXiv ID를 다른 논문 제목에 붙여 두기 때문입니다(실측: PPO `1707.06347`, InstructGPT `2203.02155`, Llama 2). 그래서 arXiv 레코드를 얻지 못한 arXiv ID(틀린 ID이거나 그 묶음의 조회 실패)의 레코드는 OpenAlex 메타데이터로 남기지 않고 빼며(필수 논문의 레코드는 arXiv ID만 뗌), 그 ID들을 계획의 `notes`에 적습니다. 조회가 모두 실패하면 `CandidateSearchError`(failed, 다시 실행)입니다. `SEMANTIC_SCHOLAR_API_KEY`가 있으면 마지막에 모든 후보를 배치 요청(500개씩)으로 보내 인용 수, 학회명, 참고문헌을 더합니다.
5. **병합**(`survey_plan.py::merge_records`): arXiv ID, DOI, OpenAlex ID 중 하나가 같거나 제목이 일치하는 레코드는 한 논문이며, 한 레코드가 여러 논문과 맞으면 그 논문들을 합칩니다. 서로 다른 arXiv ID 두 개가 한 논문에 들어가는 병합은 하지 않습니다. 후보 ID는 arXiv ID(버전 없음), 없으면 DOI(소문자)입니다. 둘 다 없는 후보(OpenAlex 저작뿐)는 쓰기 단계가 서지 정보를 찾을 수 없으므로 풀에서 빼고 그 수를 `notes`에 적습니다. 제목·저자·초록·날짜는 arXiv 레코드(arXiv 검색, DataCite)가 우선이고(학회 레코드의 날짜는 `YYYY-01-01`인 경우가 많음), 학회명은 Semantic Scholar, OpenAlex의 arXiv 아닌 위치, `arXiv` 순서입니다. 인용 수는 OpenAlex 레코드 중 최댓값과 Semantic Scholar 값 중 큰 값입니다. OpenAlex의 arXiv 레코드 초록은 쓰지 않습니다(5.4절과 같은 이유).
6. **간선과 순위**(`citation_edges`, `rank_candidates`): 후보 A의 참고문헌(OpenAlex ID, Semantic Scholar의 arXiv ID·DOI)이 후보 B의 키와 같으면 A→B 간선입니다. 여러 후보가 가진 키(병합하지 않은 두 arXiv 논문이 공유한 DOI)는 어느 쪽도 가리키지 않습니다. 사용자 연도 범위 밖이거나 연도를 모르는 후보는 빼고(필수 논문은 남김), 주제 근거(풀 안의 피인용 수 + 그 논문을 찾은 서로 다른 검색어 수) 내림차순, 그 논문이거나 그 논문을 링크한 서로 다른 참고 자료 수 내림차순, 발표 후 연간 인용 수 내림차순, ID 순으로 정렬합니다. 참고 자료는 주제와 무관한 논문(LoRA 같은 기반 기법, 언어 모델 보고서)도 링크하므로 주제 근거가 아니라 같은 주제 근거 사이의 순서만 정합니다. 선정 호출의 후보 목록은 필수 논문과 랜드마크 제목과 일치한 후보를 먼저 넣고 나머지를 순위대로 채운 `candidate_pool`개(기본 60)입니다. 참고 자료 논문은 따로 자리를 받지 않고 순위로만 들어갑니다. 연도 범위는 사용자 범위만 씁니다.
7. **선정**(`SurveySelectionPrompt`, `survey.selection_model_id`, adaptive thinking): 후보마다 ID, 제목, 연도, 학회, 인용 수와 연간 인용 수, 풀 안의 피인용 수, 그 후보를 찾은 검색어 수, 필수(`[REQUIRED]`)·랜드마크(`[LANDMARK]`)·참고 자료(`[REFERENCED by N]`, N은 그 논문이거나 링크한 자료 수) 표시, 초록 앞 400자를 보고 `[min_papers, max_papers]`(기본 6–25) 안에서 논문 수를 주제의 폭과 풀에 맞게 정해 이유를 적고, 논문마다 계보·역할(`core`·`supporting`)·한 줄 이유를, 제외한 주요 후보 최대 10개와 이유를 돌려줍니다. 계보는 합치거나 이름을 바꾸거나 새로 만들 수 있습니다. `check_selection`이 후보 목록에 없는 논문 ID, 중복, 빠진 필수 논문, 범위를 벗어난 논문 수, 핵심(`core`) 논문이 하나도 없는 선정, 모르는 계보, 중복 계보 ID를 모두 찾아 한 번 다시 묻고, 그래도 쓸 수 없으면 `SurveyPlanError`(failed)입니다. 후보 목록이 `min_papers`보다 작으면 하한은 목록 크기입니다. 제외 목록의 후보 목록 밖 ID는 다시 물을 문제가 아니므로 `build_plan`이 빼고 `notes`에 적습니다.
8. **계획 파일**(`build_plan`, `save_plan`): 논문은 날짜순(날짜 없는 논문은 끝)이고, 간선은 선정한 논문 사이의 것만, 계보는 논문이 하나 이상 있는 것만 남깁니다. `papers/survey-<slug>/plan.json`(slug는 사용자 주제로 만들어 같은 요청은 같은 위치. slug가 버리는 한글 같은 비ASCII 글자가 주제에 있으면 주제의 SHA-1 앞 8자를 붙여 `LLM 선호 최적화`와 `LLM 추론 최적화`가 같은 위치를 쓰지 않음)에 저장하며, 내용이 다른 파일이 이미 있으면(손으로 고친 계획일 수 있음) `plan.<UTC 시각>.json`으로 옮겨 둡니다. 버킷이 있으면 `s3://<bucket>/<s3_prefix>/surveys/<slug>/plan.json`에도 올리며(`upload_plan`), 내용이 다른 S3 계획이 이미 있으면 `plan-<sha256 앞 16자>.json`(`survey write --remote`가 올리는 계획과 같은 이름 규칙)으로 남겨 둡니다. 결과의 `title`은 다듬은 주제, `plan_path`(로컬 실행만)·`s3_uri`는 계획 파일, `warnings`는 계획의 `notes`입니다.

API 호출 정책입니다. OpenAlex·DataCite·Semantic Scholar는 `citation_metadata.py::_get_json`(공급자별 속도 제한, 429에 `Retry-After`만큼 기다린 뒤 한 번 재시도, 60초를 넘는 대기는 할당량 소진)을, arXiv는 `ArxivHandler.search`(프로세스 전역 3초 간격, 한 번 시도)를 씁니다. 재시도는 이 한 계층만 합니다. 후보 수집에서 실패한 호출(전송·HTTP 오류, JSON이 아닌 200 응답)은 건너뛰고 계획의 `notes`에 남기며, 할당량이 소진됐거나 연속 두 번 실패한 공급자(arXiv의 429는 몇 분씩 이어짐)는 그 실행 동안 부르지 않습니다. 주제 검색(검색어·랜드마크)이 하나도 답하지 않으면 필수 논문이 있어도 `CandidateSearchError`(failed), 범위 안 후보가 하나도 없으면 `SurveyInputError`(rejected)입니다. 그 밖의 예외(프로그램 오류)는 그대로 올라갑니다. 필수 논문 조회 실패는 건너뛰지 않습니다.

OpenAlex는 arXiv 시대 ML 논문의 인용 수를 크게 적게 세고(실측: DPO의 arXiv 레코드 293회, 학회 레코드 460회) 프리프린트와 NeurIPS·ACL 레코드의 참고문헌이 대부분 비어 있어, Semantic Scholar 키가 없으면 간선이 적고 순위가 검색어 수와 OpenAlex 인용 수에 기댑니다. 이때는 계획의 `notes`에 그 사실을 남깁니다. 후보의 참고문헌과 인용 저작을 한 단계 더 넣는 인용 확장은 하지 않습니다(간선은 검색으로 모은 후보 사이에서만 셉니다). DPO 주제로 실측하면(고정 범위 호출, 검색어 6개, 랜드마크 8개, 필수 논문 DPO) 확장으로 후보는 197개에서 370개, 간선은 22개에서 664개로 늘었지만, 상위 60개 후보 목록의 43자리를 확장으로만 찾은 저작이 차지하고 검색으로 찾은 후보 45개를 밀어냈습니다. 그 43개는 대부분 범용 고피인용 저작(Attention Is All You Need, Adam, GPT-4, LLaMA)이나 다른 분야(염소 이버멕틴 약동학)였고, 주제에 맞는 것은 Constitutional AI, Bradley–Terry 쌍대 비교, TAMER 등 몇 개뿐이었습니다.

토큰 예산은 `survey.max_total_tokens`(기본 30만)의 소프트 상한입니다. 넘으면 수정 호출을 하지 않고, 1.5배를 넘으면 다음 호출 전에 중단합니다.

계획 파일(`survey_plan.py::SurveyPlan`, `extra="forbid"`)의 형식입니다. 내용은 영어이며(사용자가 고치는 작업 문서이고, 글은 쓰기 단계가 출력 언어로 씀), 사용자가 논문을 빼거나 더하거나 계보를 옮기고 이름을 바꿔도 되고, 쓰기 단계가 다시 검증합니다. 파일을 읽을 때의 검증(`SurveyPlan`)은 논문 ID·계보 ID 중복, 모르는 계보, 논문 없는 계보, 계획에 없는 논문을 가리키는 간선, 핵심(`core`) 논문이 없는 계획을 거부합니다.

| 필드 | 내용 |
| --- | --- |
| `schema_version`, `generator`, `created_at` | `3`, `scholar-lens survey-plan/3`, UTC 생성 시각. 버전 2 계획은 `references` 없이, 버전 1 계획은 URL 문자열이던 `references`를 아래 형식으로 바꿔(http(s) URL이 아닌 값은 버림) 읽습니다(`upgrade_plan_data`) |
| `topic`, `scope`, `time_range` | 사용자 주제, 다듬은 주제 서술, `{since, until}`(사용자 범위, 열린 쪽은 `null`) |
| `paper_count_reason` | 논문 수를 그렇게 정한 이유 |
| `lineages[]` | `id`(소문자·숫자·하이픈), `name`, `description` |
| `papers[]` | `id`(arXiv ID 또는 DOI), `title`, `authors`(앞 3명 + `et al.`), `year`, `date`(ISO), `venue`, `citation_count`, `url`(arXiv 초록 또는 doi.org), `lineage`, `role`(`core`·`supporting`), `reason`. 날짜순 |
| `edges[]` | `{"from": 인용한 논문, "to": 인용된 논문}`, 선정한 논문 사이만 |
| `excluded[]` | 제외한 주요 후보 최대 10개: `id`, `title`, `year`, `reason` |
| `references[]` | 사용자 참고 자료: `url`, `title`(읽은 페이지 제목, 못 읽었으면 빈 값), `kind`(`paper`·`article`, 손으로 더할 때 빼면 URL로 정함). `paper` 자료는 계획 단계의 후보였을 뿐이고(선정됐으면 `papers[]`에 있음), `article` 자료는 쓰기 단계가 다시 읽습니다 |
| `sources`, `notes` | 공급자별 요청·실패 수, 품질 메모(인용 수 과소 집계, 실패한 호출, 확인하지 못한 arXiv ID, 뺀 후보 수) |

#### 4.3.1 쓰기 단계

`survey_writer.py::run_survey_write_pipeline` 순서입니다. 블로그 글 규칙(`RunContext.require_conventions`, 6.1)에 `resources.github.survey_category`(기본 `Paper Surveys`) 유형이 없으면 모델을 부르기 전에 실패합니다. 블로그 규칙에서 이 유형은 `paper`가 아니므로 `paper_author`·`paper_doi`·`subtitle`이 없습니다.

1. **계획 재검증**(`load_plan`, `survey_post.py::normalize_plan_data`): 계획 파일(로컬 또는 같은 버킷의 S3)을 JSON으로 읽고, 논문 ID를 계획 ID로 정규화하고(arXiv URL·doi.org 링크 → 버전 없는 arXiv ID·DOI), 빠진 논문을 가리키는 간선과 논문이 없어진 계보는 버리고 메모로 남깁니다. 제목이나 URL이 없는 논문(손으로 더한 논문)은 `id`, `lineage`, `role`만 있으면 되고, arXiv ID는 DataCite, DOI는 OpenAlex(그 논문에 arXiv ID가 있으면 초록은 DataCite의 arXiv 초록)에서 채웁니다(`PaperIndex`, `fill_metadata`). 그다음 `SurveyPlan`으로 검증하고 논문을 날짜순(날짜 없는 논문은 끝)으로 다시 정렬합니다. 색인이 모르는 논문, JSON이 아닌 파일, 스키마 위반(모르는 계보, 잘못된 역할 등)은 `SurveyInputError`(rejected)입니다.
2. **논문 카드**(`gather_cards`, `SurveyCardPrompt`, `survey.card_model_id`): 카드는 `papers/survey-cards/<paper>/v<스키마>-<모델>-<언어>-<variant>.json`과 버킷이 있으면 `<s3_prefix>/surveys/cards/` 아래에 캐시하며(`card_cache_key`, `CardStore`), 같은 논문·카드 스키마·모델·출력 언어·`variant`면 다시 실행하거나 다른 서베이가 재사용합니다(카드 글이 출력 언어로 쓰이므로 언어도 키에 넣음). `variant`는 카드 프롬프트와 `card_source_chars`의 SHA-256 앞 8자(`card_variant`)라 둘 중 하나가 바뀌면 카드를 다시 만듭니다. 캐시가 없는 논문만 원문을 읽습니다. 핵심 논문이 arXiv에 있으면 `web_research.py::WebResearcher`로 arXiv HTML 전문과 그림 목록(이미지 URL과 캡션, `PageContent.image_captions`)을 읽고(arXiv 공용 속도 제한, HTML이 없거나 변환 실패 스텁이면 초록 페이지), 그 밖에는 색인의 초록을 씁니다. 원문이 없는 논문은 빠지고 PR 메모로 남습니다. 카드 호출은 논문당 한 번(동시 4개, 원문 앞 `card_source_chars`자)이며 `<card>` 안의 JSON(이름, 문제, 핵심 아이디어, 방법, 원문에서 그대로 옮긴 핵심 수식과 뜻, 결과(수치와 설정), 한계, 대비하는 선행 연구 이름, 그림 후보 최대 2개)을 `CardAnswer`로 검증합니다. `card_problems`가 원문에 없는 수치(`unsourced_numbers`), 원문 LaTeX에 없는 수식(공백·`$$` 무시), 그림 목록에 없는 ID를 찾으면 한 번 다시 묻고, 그래도 남은 것은 `build_card`가 지웁니다(수치가 틀린 문장·결과·한계, 수식, 그림). 지운 내역은 PR 체크리스트에 남습니다. 초록에서 쓴 핵심 arXiv 논문의 카드는 일시적인 접근 실패일 수 있어 캐시하지 않고, 다른 서베이에서 보조 논문으로 초록 카드가 캐시된 논문이 핵심 논문이면 캐시를 쓰지 않습니다. 읽을 수 없는 캐시도 없는 것으로 봅니다. 프롬프트의 뜻이 바뀌면 `CARD_SCHEMA_VERSION`을 올립니다. 현재 2(선행 연구와의 관계, `<paper id>#E<n>` ID의 핵심 수식 최대 3개, 여러 패널 그림 처리)이며, 이전 버전 카드는 읽지 않습니다.
3. **참고 자료 글**(`write_survey`, `read_references`): 계획의 `article` 자료를 계획 단계처럼 다시 읽습니다(캐시하지 않음, `paper` 자료는 읽지 않음). 읽지 못한 글은 빠지고 PR 메모로 남습니다.
4. **개요**(`SurveyOutlinePrompt`, `survey.outline_model_id`, adaptive thinking): 계보와 논문, 카드 요약(그림·수식 유무), 인용 간선과 카드의 선행 연구를 보고 `<outline>` 안의 JSON으로 영어 제목(블로그 `seo_title_max_length` 이하, 넘으면 단어 경계에서 자름), 절 목록, 용어집(가이드와 같은 `English | rendering` 형식, `parse_glossary`)을 돌려줍니다. 절 종류는 `introduction`(처음), 계보마다 하나의 `lineage`(그 계보의 논문을 계획의 날짜순으로 코드가 배정, 핵심 논문은 `###` 소절), 계보가 서로 얽힐 때만 `cross_cutting`, `comparison`, `insights`, 선택적 `conclusion`(끝)이고, 절마다 깊이, 보여 줄 그림 ID(`<paper id>#F<n>`), 핵심 수식을 보여 줄 논문, 카드로 만들 추가 표, 담당 개념을 정합니다. `check_outline`이 제목 길이, 구조(`structure_problems`: 순서, 계보별 절 하나), 쓸 수 없는 참조(`clean_outline`: 모르는 논문·그림, 그 논문을 다루지 않는 절의 그림·수식, 두 번 쓰는 그림)를 찾아 한 번 다시 묻습니다. 그 뒤에도 구조 문제가 남으면 `SurveyPlanError`(failed)이고, 남은 참조는 코드가 지우고 PR 메모로 남깁니다.
5. **길이**(`section_targets`): `lineage` 절은 논문 몫의 합(`chars_per_core_paper` 3,000, `chars_per_supporting_paper` 800), 다른 절은 깊이별 크기(`DEPTH_CHARS`: deep 5,000, standard 2,500, brief 1,000)이고, 합계가 `[min_total_chars, max_total_chars]`(20,000–70,000) 밖이면 모든 절을 같은 비율로 맞춥니다. 단위는 가이드와 같은 한국어 산문 글자 수(`prose_chars`)이며 다른 언어는 2.5배입니다.
6. **절 집필**(`SurveyWriter.write`): 절마다 그 절 논문의 카드(자기 논문이 없는 도입·비교·통찰·결론 절은 핵심 논문. 카드 전체는 `lineage`·`cross_cutting` 절만 받고, `comparison` 절은 결과와 한계만, 도입·통찰·결론 절은 한 줄 카드만 받음), 다른 논문의 한 줄 카드, 관계 근거(계획의 인용 간선과 카드의 선행 연구), 인용 키 목록, 배정된 그림(URL과 캡션)과 수식, 앞 절 요약과 직전 두 절, 이미 나온 헤딩·수식·표·그림 목록(`article_ledger`), 다른 절의 담당 개념(`owned_elsewhere`), 용어집 상태(`glossary_view`)를 줍니다. 분야를 짜는 절(`REFERENCE_KINDS`: `introduction`, `cross_cutting`, `comparison`, `insights`)은 참고 자료 글의 발췌(`reference_excerpts`, 주제와 절 제목·설명에 관련 큰 부분, 모두 8,000자 안, `(reference material)` 표시)도 받고, 다른 절은 `none`입니다. 작가·평가자·사실 확인이 같은 발췌와 규칙(`REFERENCE_MATERIAL_RULE`)을 봅니다. 참고 자료는 문제 설정, 계보 지도, 장단점, 남은 과제를 풀어 쓰는 설명 자료이고, 서베이 논문의 문제·방법·수식·수치·결과의 근거가 되지 못하며(그것은 카드만), 인용 키나 링크로 인용하지 않습니다. 자료에서 가져온 견해는 그 글의 이름(저자·사이트·제목)으로 한 번 밝힙니다(`NARRATION_RULE`의 유일한 예외). 평가자는 참고 자료만 뒷받침하는 논문 서술을 차단 문제로 봅니다. 가이드의 반복(`tech_guide.py::revise_section`)으로 평가(0–100점, 차단 문제: 카드와 맞지 않는 서술·수치, `connections`와 카드가 뒷받침하지 않는 논문 사이 관계, 구조 참조, 남의 담당 개념 설명) → 차단 문제 우선 피드백으로 한 번 수정 → 더 나은 초안만 채택하고, 카드에 비추어 사실 확인(`checked_grounding`, 규칙을 어기면 한 번 재시도, 둘 다 게시 오류를 늘리면 초안 유지)합니다. 수정 뒤에도 남은 평가자의 차단 문제는 사실 확인에 넘겨, 카드만으로 바로 그 주장만 고치거나 지우게 합니다(`blocking_issues`). 코드 규칙(`rule_issues`)에 인용 규칙을 더합니다. 계획에 없는 인용 키, 손으로 쓴 서베이 밖 논문 링크(arXiv 초록·doi.org; 검증이 거부함), `lineage` 절이 다루면서 인용하지 않은 논문이 차단 문제입니다(`citation_issues`). 평가자와 사실 확인도 다른 논문의 한 줄 카드와 뒤에 붙을 표를 봅니다. 작가가 빈 절을 돌려주면 그 절은 빠지되, 코드의 표는 절 제목과 함께 남습니다. 연표(연도·논문·계보·핵심 아이디어, 모든 논문)는 `introduction` 뒤에, 비교표(논문·계보·설정과 함께 적은 주요 결과(검증된 수치만, 최대 2개)·한계, 핵심 논문. 핵심 아이디어는 연표에 있음)는 `comparison` 뒤에 코드가 붙이며(`timeline_table`, `comparison_table`, 셀 내용은 카드에서. 카드는 용어집보다 먼저 만들어 캐시하므로 셀은 용어집 표기가 아니라 카드의 한국어 용어를 씀), 작가는 표를 소개하고 해석만 합니다. 표 머리글은 출력 언어(한국어, 그 밖에는 영어)입니다.
7. **조립**(`survey_body`): `[@<paper id>]` 키를 `([Rafailov et al., 2023](논문 URL))` 링크로 바꾸고(코드 안은 그대로, 모르는 키는 지우고 메모), 서베이 밖 논문 링크는 링크 글자만 남기고(메모), 작가에게 주지 않은 이미지를 지우고, 그림마다 바로 아래에 출처 줄(`*출처: [저자 et al., 연도](URL)*`)을 넣습니다. 마무리 호출(`SurveyFinalizationPrompt`)이 블로그 주제, 통제 태그, 세부 태그, 검색 설명을 정하고, 설명은 `description_problems`(평문, 120–160자, 본문에 있는 수치만)로 확인해 한 번 다시 묻고, 그래도 검증이 거부할 설명(마크업, 제목보다 짧음)이면 넣지 않아 블로그가 첫 문단에서 만듭니다. front matter는 `categories: ["Paper Surveys", <주제>]`, 표지는 `cover_for`, 태그는 `build_post_meta`가 정하고, 끝에 `- - -`와 `### References`(`Authors (year). Title` 링크, 날짜순)를 붙이고, 읽은 참고 자료 글이 있으면 그 뒤에 `### 참고 자료`(한국어, 그 밖에는 `### Further reading`)와 `* [제목](URL)` 목록을 붙입니다(`survey_post.py::further_reading`, 제목은 계획의 `title`, 없으면 페이지 제목). 논문 그림은 arXiv 그림이므로 발행 단계가 블로그로 복사합니다(7장).
8. **발행**: 검증 종류는 `survey`이고(6.3), 브랜치는 `<branch_prefix>/survey-<주제 slug>`(계획의 주제로 정하므로 같은 계획을 다시 쓰면 같은 브랜치와 PR), 글 파일은 제목 slug입니다. front matter의 날짜는 계획의 `created_at`이라 다시 실행해도 같으므로, 같은 계획을 다시 쓰면 같은 글을 바꿉니다. 개요 호출이 다른 제목을 고르면 글 파일 이름이 달라지지만, 열린 PR에서는 front matter `date`가 같은 이전 파일을 예전 사본으로 지우고(7장), 이미 병합된 글은 슬러그로만 찾으므로 새 글로 올라갑니다(이전 글은 손으로 정리). PR 체크리스트에는 계획 재검증 메모, 지운 카드 내용, 초록만 읽은 핵심 논문, 빠진 논문, 읽지 못한 참고 자료 글, 기준에 못 미친 절, 예산 메모가 들어갑니다.

토큰 예산은 `survey.write_max_total_tokens`(기본 300만)의 소프트 상한입니다. 넘으면 남은 절을 평가·수정·사실 확인 없이 쓰고, 1.5배를 넘으면 다음 호출 전에 중단합니다.

## 5. 데이터 수집과 파싱

### 5.1 논문 소스

`paper_source.py::resolve_paper_source`가 입력을 판별합니다.

- arXiv ID(신규형, `hep-th/9901001` 같은 구형, `arXiv:` 접두어, 버전 포함)와 arXiv URL은 `ArxivSource`가 됩니다. 메타데이터는 arXiv API에서 받습니다.
- 그 밖의 URL은 `PdfUrlSource`입니다. SSRF 검사 후 content-type과 `%PDF-` 매직 바이트로 실제 PDF인지 확인하며, 아니면 `NotAPdfError`입니다. 제목은 본문에서 추출한 제목으로 대체합니다(`title_is_fallback`).

### 5.2 파싱

- arXiv 소스는 arXiv HTML(`arxiv.org/html`)을 먼저 씁니다(`parser.py::HTMLRichParser`). 실패하거나 내용이 비면 PDF로 넘어가고, 그림은 PDF에서 잘라 로컬 파일로 씁니다. ar5iv는 쓰지 않습니다. 잃어버린 그림을 HTTP 200의 "NO IMAGE AVAILABLE" PNG로 돌려주기 때문입니다. 이렇게 얻은 논문 그림(arXiv HTML 그림, arXiv 원본에서 되살린 그림, PDF에서 잘라낸 그림)만 게시 단계에서 블로그로 복사합니다(7장).
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

- **front matter 조립**(`formatting.py`): `author`는 쓰지 않습니다. 논문 글은 `paper_author`(발행 조직)와 `subtitle`(모델이 쓴 부제 → 본문 첫 H1 → TL;DR 첫 문장 순, 제목과 같으면 다음 후보)을 두고, 블로그가 인용 수를 조회하는 `paper_doi`를 둡니다(arXiv 논문은 arXiv DOI `10.48550/arXiv.<id>`, 그 밖에는 논문의 DOI가 있을 때만; 블로그는 Semantic Scholar에서 조회). `seo_title`은 제목이 70자를 넘을 때만 넣으며, 모델 후보가 없거나 길면 콜론 앞부분이나 단어 경계에서 자릅니다(`shorten_title`). 카테고리 주제와 표지는 spec에서 정하고(`resolve_topic`, `PostConventions.cover_for`), `use_math`는 `$$`가 있을 때만 켭니다. `date`는 논문 발행일입니다. 논문 글의 `description`은 마무리 단계(요약은 요약 호출)가 쓴 검색 설명이며, 없으면 넣지 않고 블로그가 첫 문단에서 만듭니다. 글 끝의 참고 목록(논문 글의 `### References`, 가이드의 `### Sources`)은 `* [제목](URL)` 줄이며(`link_list`), 같은 URL(끝 `/` 무시)은 한 번만 넣고 제목이 없거나 자리표시자이거나 URL과 같으면 URL을 링크 글자로 씁니다. 제목은 외부 입력이라 HTML을 이스케이프하고 `[`, `]`, `|`(목록 항목이 표가 됨), `{`, `}`(Liquid)를 엔티티로 바꾸며(`md_link_text`), URL은 공백·괄호·`|` 같은 문자를 퍼센트 인코딩합니다(`md_link_url`, 이미 인코딩된 문자는 유지). 가이드 출처의 제목은 수집한 페이지의 제목입니다.
- **태그**(`tags.py::normalize_tags`): 공백·`_`는 하이픈, `&`는 `and`, `/`와 따옴표는 삭제, 연속 하이픈 축약, 첫 단어가 아닌 기능어는 소문자로 바꿉니다. 그다음 별칭을 대체 태그로 바꾸고, 대소문자를 무시해 중복을 없애며, 제목 전체·콜론 앞·콜론 뒤와 같은 태그를 지웁니다. 통제 태그는 닫힌 목록만 남기고, 세부 태그 중 통제 태그와 같은 것은 통제 쪽으로 옮깁니다. 발행 순서는 세부 태그 다음 통제 태그입니다.
- **수식**(`markdown_math.py::normalize_math`): `\(…\)`·`\\(…\\)`는 `$$…$$`로, `\[…\]`는 앞뒤 빈 줄을 둔 디스플레이 `$$`로 바꿉니다. `$$` 안의 `\_`는 `\text{}`·`\texttt{}` 안을 빼고 `_`로 되돌리고, 한 줄 전체가 `$$`인 수식 앞뒤에 빈 줄을 넣습니다(표·목록 제외). 확률·정책 인자 안의 조건부 `\|`·`\Vert`는 `\mid`로, 합의 상한으로 쓰인 `\lVert x \rVert`(길이)는 `\lvert x \rvert`로 바꿉니다(노름과 KL 발산의 `\|`는 그대로). 코드와 단일 `$`는 건드리지 않습니다.
- **Markdown**(`markdown_lint.py::lint_markdown`): 헤딩 앞 빈 줄, 생성용 envelope 태그 제거, H1을 H2로 올리고 한 단계 넘게 깊어진 헤딩을 앞 헤딩 바로 아래 단계로 올리기, 네 번째부터의 콜아웃(`> **라벨:** …`)을 일반 문단으로 풀기, 가짜 HTML 태그 이스케이프를 처리합니다. `drop_unknown_images`는 알려진 그림(논문 그림, 가이드에 제공한 이미지)이 아닌 이미지를 지우고, `fill_image_alts`는 빈 alt를 그림 캡션으로 채우며, 한국어 글의 영어 alt는 가장 가까운 헤딩으로 바꿉니다. 원 논문 번호 참조, 영어 어간에 붙인 하다·되다 활용형(`prefill합니다`, `prefill하면서`, `swap-in되어`. 대문자 약어(`SVD하면`)와 조사 `한테`는 제외. 3.1의 용어 규칙 중 마지막 규칙) 같은 규칙 위반(`writing_issues`)은 게시 직전 본문에서 다시 찾아 PR 체크리스트에 올립니다.

### 6.3 검증 규칙

`post_validator.py::validate_post(markdown, spec, kind, file_name=, site_posts=)`는 순수 함수이며 `Violation(level, rule, line, message)` 목록을 돌려줍니다. `site_posts`는 블로그 글의 front matter(`post_conventions.py::fetch_blog_posts`가 태그 후보와 함께 한 번 읽음, `RunContext.require_blog_posts`)이고, 이번 글이 대체하는 글(정규화한 슬러그가 같은 글)은 비교에서 뺍니다. 코드 블록 안은 검사하지 않습니다. 논문 글의 표 행 수치가 원문의 한 줄이나 한 표 안에 함께 나오지 않으면(`unsourced_table_numbers`) PR 체크리스트에 메모로 올립니다.

수치 대조(`unsourced_numbers`)는 숫자와 바로 뒤의 단위를 함께 읽습니다. 천 단위 쉼표, 끝자리 0, 부호, `%`는 무시하고, 단위는 닫힌 목록입니다(영문 `K`·`M`·`B`·`T`와 `thousand`·`million`·`billion`·`trillion`, 한국어 `천`·`만`·`백만`·`천만`·`억`·`조`). 단위가 붙은 수는 같은 양일 때만 맞으므로 `70억`은 원문의 `7B`와 맞고 `7억`은 맞지 않습니다. 단위 없는 수는 원문 어디든 같은 값이 있으면 맞습니다. 앞에 영문자가 붙은 수(`H800`, `Qwen3`)는 이름으로 봅니다. `12만 8천`·`1억 2000만` 같은 합성 수는 한 양으로 읽어 `128K`·`120M`과 맞습니다. 숫자 뒤의 `만`은 끝, 한글·공백이 아닌 문자, `의`·`에`·`에서`·`으로`·`로`·`까지`·`부터`, 또는 공백 한 칸 이내의 숫자·한글이 아닌 단어·닫힌 목록의 단위어(`명`, `개`, `토큰` 등)가 뒤따를 때만 단위로 읽습니다. 그래서 `5만 사용`, `10만큼`의 `만`(only)은 단위가 아닙니다.

| 규칙 | 수준 | 조건 |
| --- | --- | --- |
| `front-matter` | error | front matter가 없거나 YAML이 아님 |
| `forbidden-key` | error | `forbidden_keys`의 키가 있음(`author`, 그리고 블로그가 파일명 날짜로 채우는 `last_modified_at`) |
| `date` | error | 파일명이 `YYYY-MM-DD-`로 시작하지 않거나 `date:`(시각 없는 날짜 포함)가 파일명 날짜보다 늦음(UTC). 블로그는 `future: true`로 빌드하므로 미래 날짜 자체는 규칙이 아님 |
| `title` | error | 제목이 비어 있음 |
| `categories` | error | `[<유형>, <주제>]`가 아님, 모르는 유형·주제, 산출물 종류와 유형의 `paper` 여부 불일치 |
| `cover` | error | `cover_for(유형, 주제)`와 다름 |
| `paper-author` | error | 논문 글에서 비었거나 `et al.`이 있음, 가이드·서베이에 있음 |
| `subtitle` | error | 논문 글에서 비었거나 제목과 같음 |
| `seo-title` | error | 제목이 70자 초과인데 없음, 또는 `seo_title`이 70자 초과 |
| `translation` | error | `lang`이 spec의 `languages`(`ko`/`en`)가 아님, 번역본에 `translation_id`가 없음, `translation_id`가 있는데 `lang`이 없음, 파일명(날짜 뒤)이 `<translation_id>`(한국어)·`<translation_id>-en`(영어)이 아님, `-en` 글에 `translation_id`가 없음, 블로그의 다른 글이 이 글을 `translation_id`로 가리키는데 이 글에 `translation_id`가 없음, 같은 `translation_id`의 같은 `lang` 글이 이미 있음 |
| `tags` | error | 비어 있음, 정규화되지 않은 태그, 퇴역 태그, 통제 태그 없음, 세부 태그 상한 초과 |
| `tags` | warning | 통제 태그 3개 초과(논문에 없는 태그 구절은 `paper_pipeline.py::_tag_notes`가 PR 메모로 남김) |
| `headings` | error | 본문 H1, 헤딩 레벨 건너뜀, 논문 글이 `## TL;DR`로 시작하지 않음, TL;DR이 비어 있음, 리뷰 TL;DR에 `###` 질문이 없음 |
| `use-math` | error | `use_math`와 `$$` 존재 여부가 다름 |
| `math` | error | kramdown이 글로 보여 주는 줄(인용문 안 포함)에 블로그 `script/lint-posts.rb`와 같은 규칙. 블로그 lint가 코드 줄까지 읽는 차이는 블로그 PR CI가 판단: `\(…\)`·`\\[…\\]` 잔존, 단일 `$` 옆의 TeX(`}$`, `$\명령`), 문단 안에서 짝 없는 `$$`, 닫히지 않은 여러 줄 `$$` 블록, 디스플레이 `$$` 앞뒤 빈 줄 없음, `$$` 안 `\_`(`\text` 밖), 인라인 수식 안 한글, 인라인 수식이 여럿인 줄이나 표 행의 수식 안 맨 `|` |
| `table` | error | 구분선 셀 수가 헤더와 다름, 데이터 행 셀이 헤더보다 많음(적으면 kramdown이 채우는 그룹 라벨 행), 첫 열 외 빈 헤더, 첫 데이터 행이 하위 헤더(첫 셀이 비고 값 셀이 없음), 바로 위 행과 같은 행, 표 행의 `$$` 안 막대 기호(게시 단계가 `\vert`·`\Vert`로 바꿈) |
| `table` | warning | 같은 헤더 라벨이 바로 옆에 반복 |
| `image` | error | arXiv(`arxiv.org`와 하위 도메인) 원격 이미지 잔존, `data:` 이미지, `/assets/images/<파일>`이 아닌 로컬 경로 |
| `image` | warning | 그 밖의 원격 이미지(`remote image not localized`: 문서 사이트·블로그 등 남의 이미지는 저작권 때문에 복사하지 않고 원본 링크로 두므로, 링크가 살아 있는지와 출처 표기를 사람이 확인) |
| `link` | error | 코드 밖의 ar5iv 링크·이미지(`ar5iv.`; 블로그 lint와 같은 규칙) |
| `lightbox` | error | `<a class="glightbox">`의 `href`가 안쪽 `<img>`의 `src`와 다름(페이지 URL은 빈 iframe으로 열림. 출처는 캡션에 씀; 블로그 lint와 같은 규칙) |
| `description` | error | front matter `description`에 `$$`·강조·백틱·링크·태그·`{{`, 페이지 `<title>`(`head.html`처럼 `paper_author`가 있는 한국어 논문 글은 `seo_title`(없으면 제목)에 " 논문 리뷰"를 붙인 길이, 그 밖의 글은 제목)보다 길지 않음, 블로그의 다른 글과 같음(공백 정규화) |
| `description` | warning | 논문 글에 `description`이 없음(블로그가 첫 문단에서 만듦), 길이가 120–160자 밖(블로그 메타 설명은 160자), `description`이 없는데 첫 문단에 수식 |
| `envelope` | error | 생성용 태그(`</explanation>` 등) 누출 |
| `citations` | error | 서베이 본문에 렌더링되지 않은 인용 키(`[@…]`), 본문이 링크한 논문(arXiv 초록·doi.org)이 `### References`에 없음 |
| `references` | error | 서베이에 `### References`가 없음, 목록의 참고 문헌이 본문 산문에서 한 번도 인용되지 않음(코드가 붙인 연표는 모든 논문을 링크하므로 표 줄의 링크는 세지 않음). 목록은 다음 헤딩에서 끝나므로 그 뒤의 `### 참고 자료`/`### Further reading`(사용자 참고 자료 글)은 인용을 요구하지 않습니다 |
| `image-alt` | warning | 한국어 글의 이미지 alt에 한글이 없음 |
| `list` | warning | 디스플레이 수식 뒤에서 번호 목록이 2 이상으로 다시 시작함(목록이 끊김) |
| `liquid` | error | `{% raw %}` 밖의 `{{ }}`·`{% %}`(코드 블록 포함; 게시 단계가 자동으로 감쌈) |
| `html` | error | 브라우저가 지우는 태그(`<think>`, `<s>`, `<input>`, 속성 없는 `<img>`; 게시 단계가 자동 이스케이프) |
| `code` | error | 닫히지 않은 코드 펜스, 코드 블록 언어 표시에 공백(게시 단계가 첫 단어만 남김) |

`Publisher._validate`가 이 함수를 호출합니다. error가 하나라도 있으면 저장·업로드·PR을 모두 하지 않고, 글과 위반 목록을 작업 폴더의 `<slug>.rejected.md`에 남기며 S3 버킷이 있으면 `<prefix>/rejected/<slug>.rejected.md`에도 올린 뒤 `PostValidationError`를 던집니다. warning은 `PublishRequest.validation_warnings`에 담겨 PR 본문의 `- [ ]` 체크리스트가 됩니다.

## 7. 발행

`publisher.py::Publisher.publish` 순서입니다.

1. 모델이 입력에서 그대로 옮긴 방어용 태그 표식(`prompt_safety.py::strip_defanging`)을 front matter와 본문 모두에서 지우고, 본문에 `markdown_lint.py::publish_body`를 적용합니다. 닫히지 않은 코드 블록 닫기, 콜아웃 3개 제한, `normalize_math`, `lint_markdown` 순서이며, 절 루프의 게시 차단 검사도 같은 함수를 씁니다.
2. `ImageLocalizer`가 논문 자신의 그림만 블로그 규칙에 맞게 옮깁니다. 대상은 arXiv(`arxiv.org`와 하위 도메인) 이미지와 PDF에서 잘라낸 로컬 그림이며, 판정은 `publisher.py::_is_paper_figure` 한 곳에서 합니다. arXiv 링크는 버전이 바뀌면 깨지고 블로그가 크기를 넣지 못하므로 복사합니다. 그 밖의 원격 이미지(문서 사이트, 블로그, 참고 자료)는 저작권이 원작자에게 있으므로 내려받지 않고 원래 URL 그대로 두며, 검증의 `image` warning으로 PR 체크리스트에 올라갑니다(6.3). arXiv 이미지는 URL 검사와 IP 고정 전송을 거쳐 HTTPS로 내려받습니다(`http://`는 `https://`로 바꿈: Batch는 443 포트만 나갑니다. 읽기 30초, 전체 60초, 리디렉션 5회, 20MB 상한, 도구를 밝히는 `User-Agent`). 내려받거나 읽은 그림이 ar5iv의 "NO IMAGE AVAILABLE" PNG(MD5 `ded85833…`, 325×400)이면 arXiv HTML 그림 URL(`arxiv.org/html/<id>v<N>/<경로>`)의 논문 원본(`arxiv.org/e-print/<id>v<N>`, arXiv 속도 제한기 경유, 100MB·180초 상한)에서 같은 경로의 그림 파일을, 없으면 파일 이름(확장자 제외)이 같은 유일한 파일을 찾아 씁니다(`utils/images.py::source_figure`: PNG·JPEG는 그대로, PDF는 PyMuPDF로 렌더링, EPS는 렌더러가 없어 실패). PNG·JPEG·GIF(애니메이션 유지)는 매직 바이트로 확인해 그대로 저장하고, SVG는 PNG로 래스터화합니다(`utils/images.py::svg_to_png`). SVG는 그림을 만든 쪽이 정하는 외부 입력이라, 외부 참조를 지우고 캔버스 크기를 4096px 안으로 고정한 뒤 시간·메모리를 제한한 자식 프로세스에서 렌더링합니다(동시 2개). 세부 규칙은 `utils/images.py`의 함수 docstring과 상수 주석에 있습니다. WebP, AVIF(Pillow에 플러그인이 있을 때), BMP, TIFF는 PNG로 바꾸고(`utils/images.py::raster_to_png`: 5천만 픽셀 초과는 거부, 긴 변 4096px), 그 밖의 형식과 읽을 수 없는 데이터는 실패로 봅니다. 저장할 파일이 5MB를 넘으면 PNG·JPEG는 긴 변 2048px PNG로 줄이고, GIF(줄이면 애니메이션이 사라짐)나 줄여도 넘는 파일은 실패로 봅니다. 저장 이름은 첫 등장 순서대로 `<slug>-<n>.<ext>`이고, 본문 경로는 `/assets/images/<slug>-<n>.<ext>`로 바꿉니다. 파일은 `<work_dir>/blog-images/<slug>/`에 모읍니다. 같은 논문의 요약과 리뷰가 작업 폴더를 같이 쓰므로, 재실행은 자기 슬러그 폴더만 비우고 S3 업로드와 PR에도 자기 이미지만 넣습니다. 그림 하나라도 저장하지 못하면 `PostAssetError`이고, 이 오류도 검증 실패처럼 글을 `<slug>.rejected.md`(S3 버킷이 있으면 `<prefix>/rejected/`에도)로 남긴 뒤 던집니다. `/assets/` 경로와 `data:` URL은 그대로 두며, `data:` 이미지는 검증에서 error입니다.
3. 검증합니다(6.3).
4. `<work_dir>/<YYYY-MM-DD>-<slug>.md`로 저장합니다. 파일명 날짜는 게시일(UTC)이고, front matter `date`는 논문 발행 시각입니다(가이드는 생성 시각). 블로그가 UTC로 빌드하므로 두 날짜 모두 UTC로 씁니다. `slugify`는 ASCII 슬러그를 만들고(80자 안에서 단어 경계로 자름), ASCII 글자가 없는 제목은 `untitled-<8자리 해시>`로, 글자·숫자가 하나도 없는 제목은 `untitled`로 바꿉니다.
5. S3 버킷이 있으면 글은 `<prefix>/posts/`, 이미지는 `<prefix>/assets/images/`에 올립니다. 이미지 하나라도 실패하면 오류입니다. `public_assets`가 켜져 있을 때만 public-read ACL을 씁니다.

`runtime.py::publish_artifact`는 `create_pr`이고 GitHub 연동이 켜져 있으면 `create_pull_request`를 호출합니다. PR 실패(`PublishError`)는 실행을 실패시키지 않고 `warnings`에 남깁니다.

`blog_sync.py::BlogBranchSync`는 clone 없이 GitHub API로 커밋합니다.

- 브랜치는 `<branch_prefix>/<branch_id>`로 고정됩니다(리뷰 `review-<id>`, 요약 `summary-<id>`, 가이드 `guide-<URL 해시>`). 같은 입력으로 다시 실행하면 같은 브랜치와 PR을 갱신합니다. 가이드 브랜치는 정규화한 입력 URL(문서와 참고 자료, 쿼리 포함, 프래그먼트 제외) 집합의 해시라 모델이 고른 제목과 무관합니다. 다만 다시 실행한 가이드의 제목이 바뀌면 열린 PR에 이전 글이 함께 남으므로 직접 지워야 합니다.
- 열린 PR이 있으면 새 커밋을 브랜치 head 위에 쌓아 PR에서 손으로 고친 다른 파일을 지킵니다. 열린 PR이 없으면(병합, squash 병합, 닫힘) 기준 브랜치에서 다시 시작합니다.
- blob SHA를 로컬에서 계산해 바뀐 파일만 올리고, 내용이 같으면 커밋하지 않습니다. 기준 브랜치가 이미 같은 글을 갖고 있으면 PR을 만들지 않습니다. 저장소 트리는 동기화마다 한 번씩 읽고, 목록이 잘렸으면(`truncated`) 실패합니다.
- 이 브랜치에만 있는 예전 글 사본과 그 글들의 번호 이미지 중 이번 실행이 만들지 않은 것은 지웁니다. 단, 그 파일을 건드린 브랜치 커밋이 모두 이 도구의 커밋(작성자와 커미터가 설정한 `author_name`·`author_email`)일 때만 지웁니다. 사람이 추가하거나 고친 파일은 남깁니다. `author_email`이 없으면 커밋이 토큰 계정 이름으로 들어가 사람의 커밋과 구분할 수 없으므로 아무것도 지우지 않습니다. 예전 사본은 슬러그가 같거나(게시일만 다름), front matter `date`(논문 발행 시각)와 종류가 같은 글(새 arXiv 버전에서 제목이 바뀐 이전 슬러그)입니다. 요약 슬러그는 `-summary`로 끝나므로 같은 논문의 리뷰와 요약은 서로 지우지 않습니다. 같은 종류이고 발행 시각까지 같은 글을 손으로 추가했다면 예전 사본으로 보고 지웁니다.
- 알려진 한계: 병합된 뒤 새 arXiv 버전에서 제목이 바뀌면 기준 브랜치의 예전 글을 슬러그로 찾지 못해 새 URL의 글이 하나 더 생깁니다. 다시 실행한 글의 이미지가 줄어도 기준 브랜치에 있던 남는 번호 이미지는 지우지 않습니다.
- 같은 글이 이미 기준 브랜치에 있으면 그 파일(파일명의 날짜와 슬러그)을 갱신하고, PR 체크리스트에 "Replaces the published post" 항목과 함께 그 글이나 글이 보여 주는 `/assets/images/` 파일을 생성 뒤에 고친 커밋을 하나씩(최신순 10개, SHA·제목·날짜·파일·URL) 넣습니다(`_hand_edits`). 생성 커밋은 이 도구의 정체성으로 만들어졌고 `<branch_prefix>/` 브랜치의 PR로 기준 브랜치에 들어온 커밋이며, 경로 이력을 최신부터 읽다가 첫 생성 커밋에서 멈춥니다. 생성 커밋이 없으면(예전 방식으로 올린 글) 모든 커밋을 올립니다. 사람이 같은 정체성으로 생성 PR 브랜치에 직접 push한 커밋은 생성 커밋으로 보입니다. 같은 글인지는 종류(요약은 `-summary`)와 정규화한 슬러그로 판단하므로, 예전 `slugify`가 남긴 점·이중 하이픈과 120자 절단 슬러그(`gemini-2.5--pushing-…`)도 같은 글로 봅니다.
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

`metrics.py::TokenUsageTracker`가 콜백으로 모든 호출의 토큰을 모델별로 모으고, 레지스트리의 1M 토큰당 단가로 비용을 추정합니다(`estimated_cost_usd`). 단가는 Anthropic 목록가이며 Bedrock 청구액과 다를 수 있습니다. `TokenBudgetGuard`는 단계마다 `max_total_tokens`를 검사해 넘으면 `TokenBudgetExceeded`로 멈춥니다. 가이드의 `max_total_tokens`는 다른 종류와 달리 소프트 상한입니다. 상한에 이르면 남은 절을 쓰되 평가·수정·사실 확인을 건너뛰고, 1.5배를 넘으면 중단합니다. 서베이 계획의 `survey.max_total_tokens`와 쓰기의 `survey.write_max_total_tokens`도 소프트 상한입니다(4.3절). 비용 지표는 실행이 끝날 때 한 번 보내므로, 비용 경보(11절)는 이미 쓴 비용을 알릴 뿐 실행을 멈추지 못합니다. 실행 중 상한은 위 토큰 예산이 맡습니다.

## 9. 설정 참조

`scholar_lens/configs/config.yaml`을 `config.py::Config.load`가 읽습니다. 모든 섹션은 `extra="forbid"`라 모르는 키는 오류입니다. AWS 안에서 파일이 없으면 실패하고, 로컬에서는 기본값을 씁니다. 키별 기본값과 설명은 `config-template.yaml`의 주석이 정본이고, 이 문서에서는 동작에 영향이 큰 것만 짚습니다.

- `resources.s3_bucket_name`: 글·이미지·결과 매니페스트 버킷. 미리 만들어 둔 버킷이어야 하며 배포에 필수입니다. 없으면 `status`는 저장된 결과 없이 Batch 상태만 보여 줍니다.
- `resources.github.*_category`: `categories[0]`이 되므로 블로그 spec의 `types`에 있어야 합니다.
- `explanation.max_synthesis_attempts`·`max_edit_rounds`, `tech_guide.max_revision_attempts`: 호출 수와 비용을 정하는 반복 상한입니다.
- `survey.min_papers`·`max_papers`(기본 6·25, `max_papers` ≤ `candidate_pool`), `candidate_pool`(60): 서베이 논문 수 범위와 후보 수집 범위입니다(4.3절).
- `survey.chars_per_core_paper`·`chars_per_supporting_paper`(3,000·800), `min_total_chars`·`max_total_chars`(20,000·70,000, 하한 ≤ 상한), `max_revision_attempts`(1), `verify_grounding`(켬): 서베이 글 길이와 절별 호출 수입니다(4.3.1절). `resources.github.survey_category`는 블로그 규칙의 유형이어야 합니다.

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

- **새 산출물 종류**: `jobs.py::ArtifactKind`, `post_validator.py::ArtifactKind`(글을 발행하는 종류만), `cli.py::_raw_kind`, `runtime.py::_KIND_LABELS`에 값을 더하고, 파이프라인 모듈을 만들어 `runtime.py::_run_pipeline`에서 분기하며, `post_validator.py`에 그 종류의 규칙을 정합니다.
- **새 모델**: `constants.py::LanguageModelId`와 `utils/models.py`의 레지스트리(컨텍스트, 출력 상한, thinking 방식, 단가)에 추가합니다.
- **새 검색 제공자**: `web_research.py::WebSearchProvider`를 구현하고 `guide_pipeline.py::build_search_provider`에 연결합니다.
- **블로그 규칙 변경**: 블로그의 `_data/post_conventions.yml`을 고치고 `assets/post_conventions.yml`에 같은 내용을 복사합니다. 새 규칙 종류가 필요하면 `post_conventions.py`의 모델과 `post_validator.py`에 함께 추가합니다.
