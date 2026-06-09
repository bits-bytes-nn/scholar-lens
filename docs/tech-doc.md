# Scholar-Lens — 기술 레퍼런스

Scholar-Lens 시스템의 심층 기술 문서입니다. 모든 구체적 서술은 현재 코드의
`파일:라인` 참조로 뒷받침되며, 경로는 별도 표기가 없는 한 리포지토리 루트 기준입니다.
(이 문서는 현재 코드 기준으로 생성되었습니다.)

## 목차

- [1. 개요 (Overview)](#1-개요-overview)
- [2. 아키텍처 한눈에 보기 (Architecture Overview)](#2-아키텍처-한눈에-보기-architecture-overview)
- [3. 엔드투엔드 데이터 흐름 (Data Flow)](#3-엔드투엔드-데이터-흐름-data-flow)
- [4. 모듈 레퍼런스 (Module Reference)](#4-모듈-레퍼런스-module-reference)
- [5. 설정 레퍼런스 (Configuration)](#5-설정-레퍼런스-configuration)
- [6. 모델 사용 (Model Usage)](#6-모델-사용-model-usage)
- [7. 프롬프트 & 설명 알고리즘 (Prompt & Explanation Algorithm)](#7-프롬프트-설명-알고리즘-prompt-explanation-algorithm)
- [8. 인용 처리 (Citation Pipeline)](#8-인용-처리-citation-pipeline)
- [9. 인프라 (AWS)](#9-인프라-aws)
- [10. 발행 & 블로그 PR (Publishing)](#10-발행-블로그-pr-publishing)
- [11. Slack 봇 (Paper Bot)](#11-slack-봇-paper-bot)
- [12. 테스트, CI & 확장 (Testing, CI & Extending)](#12-테스트-ci-확장-testing-ci-extending)

---

## 1. 개요 (Overview)

Scholar-Lens는 논문 및 기술 문서를 발행 가능한 콘텐츠로 변환하는 AI 기반 연구 보조 도구입니다. 세 가지 아티팩트 유형을 생성하며, CLI, AWS Batch, 또는 Slack 봇을 통해 구동됩니다.

### 1.1 아티팩트 유형

Scholar-Lens는 다음 세 가지 아티팩트를 생성합니다 (`scholar_lens/main.py:62-68`에서 `Mode` 클래스 정의):

| 아티팩트 | 설명 | 진입점 |
|---------|------|--------|
| **Review** (심층 리뷰) | 다단계 LangGraph 워크플로우를 통한 심층 분석; 핵심 요약 포함 | `scholar_lens/main.py` (기본 모드) |
| **Summary** (간략 요약) | 다섯 섹션 구조의 간결한 요약; 수치/표/수식/코드 포함 | `--mode summarize` |
| **Tech Guide** (기술 가이드/튜토리얼) | 문서 URL을 기반으로 한 자습서; 웹 검색 및 하위 페이지 발견 지원 | `scholar_lens/tech_guide_main.py` |

### 1.2 지원하는 논문 소스

Scholar-Lens는 arXiv ID 또는 임의의 논문 PDF URL을 수용합니다 (`scholar_lens/src/paper_source.py`의 `PaperSource` 인터페이스 기반):

- **arXiv 소스** (`ArxivSource`, `paper_source.py:96-124`): arXiv ID (예: `2312.11805` 또는 `2312.11805v2`) 또는 `arxiv.org` URL (abs/pdf/html 형식) 입력; arXiv API를 통해 풍부한 메타데이터 제공.
- **PDF URL 소스** (`PdfUrlSource`, `paper_source.py:126-315`): 임의의 HTTPS URL이 PDF를 제공해야 함. Content-Type 검증 및 PDF 매직 바이트 확인을 통해 비 PDF URL 거부 (`paper_source.py:256-290`); SSRF 공격 방어를 위한 URL 가드 적용 (`paper_source.py:141-144`, `paper_source.py:268-271`).

메타데이터 해석: arXiv 소스는 제목, 저자, 카테고리 등 완전한 메타데이터를 제공합니다. 임의 PDF URL 소스는 다운로드된 PDF의 임베딩 메타데이터에서 제목을 추출하고 (`paper_source.py:241-254`), 나머지 속성(카테고리, 키워드)은 LLM 속성 추출 단계에서 채워집니다 (`main.py:354-369`의 `ContentExtractor`).

### 1.3 실행 방식

Scholar-Lens는 세 가지 실행 모드를 지원합니다:

#### CLI (Command Line Interface)
개발/로컬 테스트용 (`main.py:732-788`):
```bash
# Review 생성 (기본값: arXiv ID 또는 PDF URL)
python scholar_lens/main.py --source 2312.11805 --parse-pdf True

# Summary 생성
python scholar_lens/main.py --source 2312.11805 --mode summarize

# Tech Guide 생성
python scholar_lens/tech_guide_main.py \
  --urls https://docs.framework.io/start https://docs.framework.io/api \
  --search-queries "best practices"
```

#### AWS Batch
프로덕션 워크로드를 위한 컨테이너화 실행입니다. `main()` 함수와 `tech_guide_main.py:_run()`은 동일한 핵심 로직을 실행하며 (`main.py:80-152`), 선택적으로 SNS 알림 및 Slack 알림을 전송합니다 (`main.py:131-151`).

#### Slack 봇 (Paper Bot)
Socket Mode를 통한 이벤트 기반 디스패칭 (`scholar_lens/slack/bot.py`). 사용자가 Slack에서 봇을 멘션하거나 DM으로 논문 ID/URL 또는 기술 문서 URL을 제공하면:
1. LLM이 사용자의 의도를 파싱 (`scholar_lens/slack/intent.py`의 `ParsedIntent`).
2. AWS Batch 작업으로 디스패치 (`scholar_lens/slack/dispatcher.py`).
3. 완료 후 Slack 채널 또는 스레드로 결과 보고.

**Slack 앱 격리**: Paper Bot은 전용 Slack 앱에서 실행하는 것이 좋습니다. 두 개의 Socket Mode 프로세스가 동일한 앱을 공유하면 Slack이 이벤트를 둘 중 하나로만 무작위로 라우팅하므로 멘션이 무시될 수 있습니다. `SLACK_EXPECTED_APP_ID` 환경 변수로 앱 ID를 검증할 수 있습니다.

### 1.4 데이터 흐름 개요

1. **소스 해석** (`resolve_paper_source()`, `paper_source.py:332-352`): `--source` 인수를 `PaperSource` 구현으로 변환.
2. **논문 준비** (`main.py:314-402`의 `_prepare_paper_data()`):
   - PDF 또는 HTML 파싱 (arXiv의 경우 HTML 선호; 임의 URL은 항상 PDF) 후 콘텐츠 및 그림 추출.
   - LLM 기반 인용, 속성(제목, 저자, 카테고리, 키워드), 목차 추출.
   - 선택적 코드 리포지토리 분석 및 요약.
3. **아티팩트 생성**:
   - **Review**: `ExplainerGraph`를 통한 다단계 분석 (`main.py:219-267`).
   - **Summary**: `PaperSummarizer`를 통한 다섯 섹션 생성 (`main.py:270-287`).
   - **Tech Guide**: `TechGuideGenerator`를 통한 기술 가이드 생성 (`tech_guide_main.py:156-175`).
4. **발행** (`Publisher`, `main.py:206-211`):
   - S3에 마크다운 업로드.
   - GitHub 리포지토리에 PR 생성 (Jekyll 프론트 매터 포함).

모든 아티팩트는 설정 가능한 카테고리 레이블 (`tech_guide_main.py:244`)과 함께 블로그 호환 마크다운으로 생성되며, 출력 언어는 구성 파일에서 설정할 수 있습니다 (`main.py:258`의 `output_language`).

## 2. 아키텍처 한눈에 보기 (Architecture Overview)

세 가지 아티팩트(리뷰 · 요약 · 기술 가이드)는 공통 프런트엔드(소스 해석 → 파싱 → 추출)를 공유한 뒤 각자의 생성 파이프라인으로 분기하고, 다시 공통 `Publisher`로 수렴합니다.

![생성 파이프라인](./diagrams/pipeline.png)

### 상위-레벨 데이터 흐름

Scholar-Lens는 세 가지 주요 단계로 논문을 처리합니다. 전체 흐름은 다음과 같습니다.

```
PaperSource (arXiv / PDF URL)
    ↓
콘텐츠 파싱 및 메타데이터 추출 (Parser / ContentExtractor)
    ↓
생성 에이전트 (ExplainerGraph / PaperSummarizer / TechGuideGenerator)
    ↓
발행 (Publisher) → S3 / GitHub / Slack / SNS
```

**파이프라인 구조** (`scholar_lens/main.py:156–216`):

1. **소스 해석**: `resolve_paper_source()`는 arXiv ID 또는 URL을 받아 `PaperSource` 인터페이스로 정규화합니다 (`scholar_lens/src/paper_source.py`).
   - `ArxivSource`: arXiv API를 통한 풍부한 메타데이터
   - `PdfUrlSource`: 임의 PDF URL, 속성은 LLM 추출

2. **데이터 준비** (`_prepare_paper_data()`):
   - HTML 또는 PDF 파싱 (HTMLRichParser / PDFParser)
   - 인용문, 속성(affiliation, category, keywords), 목차 추출
   - 코드 저장소 처리 (CodeRetriever, 선택사항)
   - `CitationSummarizer` 초기화

3. **생성** (모드별):
   - **review 모드**: `ExplainerGraph` (LangGraph 기반 멀티 에이전트)
   - **summarize 모드**: `PaperSummarizer` (간단한 구조)
   - 참고: 인용문 추출/요약과 목차(TOC) 추출은 **리뷰 전용**입니다. 요약 경로는 이를 사용하지 않으므로 `_prepare_paper_data`가 summarize 모드에서 건너뛰어(속성 추출만 수행) 불필요한 비용/지연을 없앱니다.

4. **발행**: `Publisher`는 마크다운을 S3 업로드, 블로그 PR 생성(생성된 PR URL은 Slack 완료 메시지에 버튼으로 표시), Slack/SNS 통지. PR 본문은 `build_pr_body`로 리뷰/요약/가이드가 동일한 형식(헤더+필드+푸터)을 사용합니다.

### 패키지 레이아웃

```
scholar_lens/
├── main.py                      # CLI 진입점, 파이프라인 오케스트레이션
├── configs/
│   ├── config.py               # Config 로드, 리소스/모델/설정 정의
│   └── __init__.py
├── src/
│   ├── paper_source.py          # PaperSource 추상화 (ArxivSource / PdfUrlSource)
│   ├── parser.py                # HTMLRichParser / PDFParser (콘텐츠 → Content)
│   ├── content_extractor.py     # ContentExtractor (인용/속성/목차 추출)
│   ├── citation_summarizer.py   # CitationSummarizer (인용문 요약, 메타데이터 해석)
│   ├── citation_metadata.py     # RateLimiter 기반 Crossref/Semantic Scholar 조회
│   ├── explainer.py             # ExplainerGraph (멀티 에이전트 LangGraph)
│   ├── summarizer.py            # PaperSummarizer (간단한 구조 생성)
│   ├── code_retriever.py        # 저장소 다운로드, 벡터 인덱싱, 코드 요약
│   ├── publisher.py             # 마크다운 저장, S3 업로드, GitHub PR 생성
│   ├── constants.py             # LanguageModelId, EmbeddingModelId, 경로, 환경변수
│   ├── rate_limiter.py          # 토큰 버킷 기반 정책 (API 속도 제한)
│   ├── url_guard.py             # SSRF 방지: URL 검증, DNS 확인
│   ├── markdown_math.py         # LaTeX 수식 underscores 이스케이프 (kramdown GFM 호환)
│   ├── metrics.py               # TokenUsageTracker (토큰 예산 추적)
│   ├── logger.py                # 통합 로깅
│   ├── aws_helpers.py           # S3Handler, SSM 파라미터, Batch 제출
│   ├── utils/
│   │   ├── factories.py          # BedrockLanguageModelFactory / BedrockEmbeddingModelFactory
│   │   ├── retry.py             # RetryableBase (Tenacity 기반 지수 백오프)
│   │   ├── models.py            # 모델 정보 메타데이터
│   │   ├── helpers.py           # 유틸 (is_placeholder, arg_as_bool, etc.)
│   │   ├── parsers.py           # 출력 파싱 (XMLOutputParser, HTMLTagOutputParser)
│   │   ├── graph.py             # LangGraph 시각화
│   │   └── batch.py             # AWS Batch 제출
│   └── prompts/
│       └── __init__.py          # 프롬프트 클래스 (BasePrompt, ~Prompt)
├── slack/
│   ├── bot.py                   # PaperBot (Slack 디스패처, 의도 파싱)
│   ├── dispatcher.py            # JobDispatcher (AWS Batch 작업 제출)
│   ├── intent.py                # IntentParser (Slack 메시지 분석)
│   └── notifier.py              # SNS/Slack 결과 알림
└── tech_guide_main.py           # 기술 가이드 생성 진입점
```

### 주요 크로스-커팅 유틸

#### 1. 팩토리 (`scholar_lens/src/utils/factories.py`)

`BedrockLanguageModelFactory`와 `BedrockEmbeddingModelFactory`는 Bedrock 모델 인스턴스를 생성합니다:

- 모델별 정보 (최대 토큰, thinking 지원, context-1m 등) 조회
- 크로스 리전 모델 ID 해석 (`global.anthropic.claude-opus-4-8`)
- 온도, thinking 예산, 성능 최적화 설정 동적 적용
- 토크나이저는 lazy-load (네트워크 비용 절감)

**사용**: `_build_publisher()`, `CitationSummarizer`, `ContentExtractor`, `ExplainerGraph` 등 모든 에이전트

#### 2. 재시도 (`scholar_lens/src/utils/retry.py`)

`RetryableBase` 클래스는 Tenacity 기반 지수 백오프를 제공합니다:

- 재시도 전략: 최대 5회, 초기 30초, 최대 120초 대기
- 터미널 에러 (`TokenBudgetExceeded`) 제외 (비용 보호)
- 모든 추출/분석 에이전트(`ContentExtractor`, `CitationSummarizer`) 상속

#### 3. 속도 제한 (`scholar_lens/src/rate_limiter.py`)

`RateLimiter`는 스레드 안전 토큰 버킷 구현입니다:

- arXiv/Crossref/Semantic Scholar API 조절 (HTTP 429 방지)
- 서버 `Retry-After` 헤더 존중 (`penalize()`)
- 동기 호출 차단, 동시 요청자용 락 보호
- `CitationSummarizer`의 아카이브 접근 시 적용

**설정** (`citation_metadata.py`):
```python
_CROSSREF_LIMITER = RateLimiter(rate=5.0, per=1.0, name="crossref")
_SEMANTIC_SCHOLAR_LIMITER = RateLimiter(rate=1.0, per=1.0, name="semantic-scholar")
```

#### 4. SSRF 방지 (`scholar_lens/src/url_guard.py`)

`assert_url_is_public()` / `is_url_public()` 함수는 신뢰할 수 없는 사용자 입력 URL을 검증합니다:

- 스킴: http/https만 허용
- 호스트: 169.254.169.254 (AWS 메타데이터), 127.0.0.1, RFC-1918 범위 차단
- DNS 해석 실행, 모든 해석된 IP 검증 (fail-closed)
- 오프라인 사전 검사: `assert_url_scheme_and_literal()` (구성자용)

**사용처**: `PdfUrlSource.__init__()`, `WebResearcher`, 기술 가이드 크롤링

#### 5. 수식 마크다운 (`scholar_lens/src/markdown_math.py`)

`normalize_math_underscores()`는 LaTeX 수식 내 언더스코어를 이스케이프합니다:

- 목표: kramdown GFM 파이프라인 호환 (마크다운 → MathJax)
- GFM은 `_`를 강조로 해석하므로 인라인 `\(W_0\)` → `\(W\_0\)`로 변환 (디스플레이 `$$...$$`도 동일)
- 인라인 수식은 `\( ... \)`만 인식 — 블로그가 단일 `$`를 제거하므로 `$...$`는 수식으로 취급하지 않음
- 코드 블록(fenced/inline) 내부는 건드리지 않음 (정규식 오더링 중요)

**사용처**: `Publisher.publish()` (마크다운 후처리)

#### 5-1. 마크다운 린트 (`scholar_lens/src/markdown_lint.py`)

`lint_markdown()`은 `normalize_math_underscores` 다음에 `Publisher.publish()`에서 실행되어 블로그(kramdown/MathJax) 렌더링 함정을 처리합니다. `markdown_math`의 검증된 코드/수식 스팬 분리기를 재사용합니다.

- **자동 교정(무손실)**: 앞 줄에 붙은 헤딩 앞에 빈 줄 삽입 (코드 펜스 내부는 제외).
- **경고만(자동 변경 X — 잘못 고치면 새 버그)**: 수식 내 bare `|`(kramdown이 표 구분자로 오인), `\begin{align|equation|gather}`/`\bm`(블로그 MathJax에서 깨짐), 단일 `$...$` 인라인 수식, 블로그 미정의 매크로(stmaryrd 등), http(s)/사이트-상대 경로가 아닌 링크 타깃.

#### 6. 콘텐츠 추출 및 인용문 요약

**ContentExtractor** (`scholar_lens/src/content_extractor.py`):
- 세 병렬 태스크: 인용 추출 (페이지네이션 지원), 속성(affiliation/category/keywords), 목차
- 출력 고정 활성화 (재시도 시 XML 파싱 오류 복구)
- `is_placeholder()` 체크로 "Unknown" / "N/A" 필터링

**CitationSummarizer** (`scholar_lens/src/citation_summarizer.py`):
- 인용문별 메타데이터 체이닝: Crossref (우선) → Semantic Scholar (폴백) → arXiv (마지막)
- `RateLimiter`로 API 조절, 비동기 세마포어로 동시성 제한 (기본 5)
- 캐시 + single-flight: 중복 인용은 한 번만 요약
- `prefer_full_text` 옵션으로 arXiv 풀텍스트 선호/회피

**Citation 타입** (`scholar_lens/src/content_extractor.py`):
```python
class Citation(BaseModel):
    authors: str           # "Smith et al."
    year: int | None       # 2023
    title: str            # 논문 제목
    arxiv_id: str | None  # arXiv 검증됨 (정규식)
```

#### 7. 생성 에이전트

**ExplainerGraph** (`scholar_lens/src/explainer.py`):
- LangGraph 기반 멀티 단계 워크플로우
- 상태: `{ paper, paragraphs, structure, ... }`
- 단계: Paper Analysis → Enrichment → Finalization → Reflection → Synthesis
- 각 단계가 콜백(`TokenUsageTracker`) 업데이트, 출력 고정 재시도

**PaperSummarizer** (`scholar_lens/src/summarizer.py`):
- 단순 체인: 프롬프트 → LLM (thinking 선택) → 구조화 출력 파서
- 다섯 섹션: 동기/해결책/구현/결과/의의
- 마찬가지로 `TokenUsageTracker` 콜백 지원

#### 8. 발행 및 알림

**Publisher** (`scholar_lens/src/publisher.py`):
- 마크다운 저장 (로컬 이미지 경로 재작성)
- S3 업로드 (`S3Handler`)
- GitHub PR 생성 (branch per paper)
- `normalize_math_underscores()` 적용

**Slack 봇** (`scholar_lens/slack/bot.py`):
- App-ID 헤더 검증 (SSL 중단 공격 방지)
- 허용리스트 기반 사용자 인증 (`SLACK_ALLOWED_USER_IDS`)
- 이벤트 중복 제거 (bounded set, 512 용량)
- `IntentParser`가 메시지를 의도로 파싱, `JobDispatcher`가 AWS Batch 제출

**SNS 알림** (`main.py:_send_sns_notification()`):
- 파이프라인 완료 후 SNS 토픽으로 메시지 발행
- 환경 변수 `TOPIC_ARN` (AWS 환경에서만)

### 모델 및 설정

**LanguageModelId** / **EmbeddingModelId** 열거형 (`scholar_lens/src/constants.py`):
- Claude Opus 4.8, Opus 4.6, Sonnet 3.5 등 지원
- 각 모델별 최대 토큰, thinking 지원, context-1m 지원 여부 메타데이터

**Config** (`scholar_lens/configs/config.py`):
- YAML/환경변수 기반 로드
- 섹션: `paper`, `explanation`, `summary`, `citations`, `code`, `resources`, `output_language`
- AWS 리소스 (S3 bucket, Bedrock region), GitHub 저장소 설정

**TokenUsageTracker** (`scholar_lens/src/metrics.py`):
- 모든 Bedrock 호출(token 사용) 집계
- 토큰 예산 초과 시 `TokenBudgetExceeded` 발생 (파이프라인 중단)
- CloudWatch 메트릭 에밋

### 환경 및 AWS 통합

**환경 변수** (`scholar_lens/src/constants.py:EnvVars`):
- `GITHUB_TOKEN`, `SLACK_BOT_TOKEN`: 크레덴셜 (AWS SSM 저장)
- `TOPIC_ARN`: SNS 알림 대상
- `SLACK_ALLOWED_USER_IDS`: 쉼표 분리 허용 목록
- `KMP_DUPLICATE_LIB_OK`: NumPy/MKL 중복 라이브러리 무시

**AWS Batch 통합** (`scholar_lens/slack/dispatcher.py`):
- Slack 메시지 → Lambda → Batch 작업 제출
- 작업 정의, 큐, 컴퓨팅 환경 구성 (IaC)

이 아키텍처는 **모듈식 확장성**(새 PaperSource/Parser/Agent 추가 용이), **견고성**(재시도, SSRF 방지, 속도 제한), **가시성**(로깅, 메트릭, LangGraph 시각화)을 우선합니다.

## 3. 엔드투엔드 데이터 흐름 (Data Flow)

본 섹션에서는 Scholar-Lens의 세 가지 주요 생성 파이프라인(REVIEW, SUMMARIZE, GUIDE)에서 입력 논문이 최종 아티팩트로 변환되는 과정을 단계별로 추적합니다.

### 3.1 REVIEW 파이프라인

REVIEW 모드는 ExplainerGraph를 통해 논문을 섹션 단위로 다단계 분석하여 상세한 리뷰를 생성합니다.

**초기화 및 데이터 준비** (`scholar_lens/main.py:156-217`)  
파이프라인은 `_run_pipeline()`에서 시작되며, 먼저 `_prepare_paper_data()`를 호출하여 논문 콘텐츠를 추출합니다. 이 단계는:

1. **소스 해석** (`paper_source.py:61-94`): 입력이 arXiv ID 또는 임의의 PDF URL인지 판단합니다. ArxivSource는 arXiv API에서 풍부한 메타데이터를, PdfUrlSource는 URL 검증 및 SSRF 방어(url_guard)를 거친 후 PDF 다운로드를 수행합니다.

2. **콘텐츠 파싱** (`main.py:405-447`): `_parse_paper_content()`는 arXiv 소스의 경우 HTMLRichParser로 아카이브의 HTML 렌더링을 파싱하고, HTML 파싱이 실패하거나 PDF-URL 소스인 경우 PDFParser로 대체합니다. 두 경우 모두 figcaption, 테이블, 수식 등이 추출됩니다.

3. **메타데이터 및 인용문 추출** (`main.py:314-402`): 추출된 콘텐츠 텍스트에 대해 ContentExtractor(`content_extractor.py:57-134`)는 병렬로 세 가지 작업을 수행합니다:
   - **인용문 추출**: CitationExtractionPrompt를 사용하여 참고문헌 목록을 파싱하고 arXiv ID 유효성 검증을 거칩니다(`content_extractor.py:174-200`).
   - **속성 추출**: AttributesExtractionPrompt로 카테고리, 키워드, 제목(PDF-URL 소스의 경우), 저자명을 추출합니다(`content_extractor.py:27-35`).
   - **목차 추출**: TableOfContentsPrompt로 섹션 구조를 파싱합니다.

4. **Paper 객체 생성** (`main.py:387-402`): 위의 모든 데이터가 Paper 모델에 수집되어 /tmp/papers/{source_id}/ 디렉터리에 JSON으로 저장됩니다.

**ExplainerGraph 워크플로우 실행** (`main.py:219-267`)  
`_generate_review()`는 ExplainerGraph 인스턴스를 생성하고 `explainer.run()`을 호출합니다.

| 노드 | 목적 | 호출 점 |
|------|------|--------|
| **analyze** | NLTK sent_tokenize로 문장 분할 후 PaperAnalysisPrompt를 통해 구조(섹션 제목, 핵심 포인트, 시작 문장 인덱스)를 추출 | `explainer.py:317-348` |
| **enrich** | 현재 단락에 대해 PaperEnrichmentPrompt를 실행하여 참고문헌 식별자 및 코드 검색 필요 여부를 판단 | `explainer.py:418-475` |
| **synthesize** | 누적된 피드백을 포함하여 PaperSynthesisPrompt로 현재 단락의 설명을 생성하며, `has_more` 플래그로 연속 루프 제어(`max_continuations=8`) | `explainer.py:557-611` |
| **reflect** | PaperReflectionPrompt로 생성된 설명의 품질을 0~100 범위로 평가하고 개선 피드백을 누적 | `explainer.py:495-550` |
| **finalize** | 모든 섹션 설명을 PaperFinalizationPrompt에 전달하여 핵심 요약(key_takeaways) 생성 | `explainer.py:477-487` |

노드 간의 조건부 라우팅:
- **synthesize** → **reflect** (항상)
- **reflect** → **synthesize** 또는 **check_continue** (품질 점수가 `MIN_QUALITY_SCORE(70)` 미만일 경우 재시도, 최대 `MAX_SYNTHESIS_ATTEMPTS(3)`)
- **check_continue** → **update_index** 또는 **finalize** (현재 인덱스 < 단락 수 && 반복 < `MAX_ITERS(20)`)
- **update_index** → **enrich** (다음 단락으로 진행)

**보강 단계 (Enrichment)** (`explainer.py:418-475`)  
enrich 노드에서:
1. 참고문헌 식별자가 추출되면 CitationSummarizer(`citation_summarizer.py:25-100`)로 각 인용문을 요약합니다. ChainedMetadataResolver를 통해 Crossref/DOI/arXiv를 차례로 시도하여 인용문 메타데이터를 획득합니다.
2. `is_affirmative()` 및 `should_search_code` 플래그 기반 코드 검색이 활성화되면, CodeRetriever가 임베딩 기반 유사도 검색으로 최대 10개 코드 스니펫을 반환합니다.
3. 모든 오류는 로깅되며 리뷰를 중단하지 않습니다 (graceful degradation).

**문서 조합 및 발행** (`main.py:206-216`)  
완성된 explanation 및 key_takeaways는 `_format_explanation()`에서 Jekyll 프론트매터, TL;DR 섹션, 설명 본문, 참고자료 링크와 함께 조합되어 마크다운 문서로 변환되고, Publisher를 거쳐 S3 및 GitHub PR로 발행됩니다.

### 3.2 SUMMARIZE 파이프라인

SUMMARIZE 모드는 간결한 5단계 구조화 요약을 생성하는 가벼운 대안입니다.

**준비 단계** (`main.py:270-287`)  
REVIEW와 동일하게 `_prepare_paper_data()`로 Paper 객체를 구성합니다.

**단일 패스 요약 생성** (`summarizer.py:31-113`)  
PaperSummarizer는:
1. `summarize()` 메서드(`summarizer.py:77-98`)에서 paper.content.text를 PaperSummaryPrompt에 전달합니다.
2. 모델은 "동기", "새로운 솔루션", "구현", "결과", "의의"의 다섯 섹션을 반환합니다.
3. HTMLTagOutputParser로 `<summary>`, `<tags>`, `<urls>` 태그를 추출합니다.
4. 반환 결과 딕셔너리는 `{"summary": markdown, "tags": comma-separated, "urls": markdown-links}`입니다.

**문서 조합** (`main.py:645-657`)  
요약 결과는 `_format_summary()`에서 REVIEW와 동일한 프론트매터, 요약 본문, 참고자료 링크로 조합되어 발행됩니다.

### 3.3 GUIDE 파이프라인

GUIDE 모드는 기술 문서 URL 목록으로부터 자체 학습 가이드를 생성합니다(`tech_guide_main.py`, `tech_guide.py`). 단일 문서 페이지를 번역하는 것이 아니라, 딥리서치로 보완 자료를 모으고 → 구조화 플래닝 → 섹션 작성 → 평가·재작성 → 사실 검증을 거치는 다단계 파이프라인입니다(논문 리뷰의 reflect-and-revise 패턴을 가이드에 이식).

**초기화** (`tech_guide_main.py:_run`)  
`_run()` 함수는:
1. WebResearcher를 초기화합니다. Brave API 키가 설정되면 BraveSearchProvider를, 아니면 NullSearchProvider를 사용합니다.
2. TechGuideGenerator를 생성합니다. 여기에는 relevance_chain, research_plan_chain, synopsis_chain, section_chain, evaluation_chain, grounding_chain이 포함됩니다. `auto_research`/`max_research_queries`/`min_quality_score`/`max_revision_attempts` 노브가 config(`TechGuide`)에서 주입됩니다.

**딥리서치 플래닝** (`tech_guide.py:_plan_research`)  
`generate()`는 먼저 시드 URL(및 선택적 하위 페이지)을 **한 번** 크롤링해 ResearchCorpus를 만든 뒤:
1. 명시적 `search_queries`가 없고 `auto_research=True`이면, TechGuideResearchPlanPrompt로 시드 문서에서 토픽과 웹 검색 쿼리(개념/원리·사용법·비교·함정 등 다른 각도)를 자동 생성합니다.
2. 생성된(또는 명시된) 쿼리를 `WebResearcher.run_searches()`로 실행해 코퍼스의 `search_results`를 보강합니다(시드 페이지 재크롤링 없음).
3. 플래닝 실패는 best-effort로 흡수되어 시드 URL만으로 진행합니다.

**관련성 검증** (`tech_guide.py:_assert_relevant`)  
TechGuideRelevancePrompt로 코퍼스가 기술 문서인지 판단합니다. 토픽이 비면 플래너가 추정한 토픽으로 폴백하며, 그래도 없으면 NotTechnicalContentError를 발생시킵니다.

**구조화 플래닝(목차 작성)** (`tech_guide.py:_draft_synopsis`)  
TechGuideSynopsisPrompt는 각 섹션에 **관심 영역**(CONCEPT/DETAIL/USAGE/APPLICATION)과 **깊이**(deep/standard/brief), 시각 요소 힌트를 태깅한 구조화 목차를 생성합니다(최대 `_MAX_SECTIONS=16`). 깊이는 고정 비율이 아니라 "중요·난해한 부분은 깊게, 쉬운·주변 부분은 짧게"라는 원칙을 따릅니다. `_parse_synopsis_sections()`가 이를 `PlannedSection`(title/area/depth)으로 파싱합니다.

**섹션별 작성** (`tech_guide.py:_write_sections`)  
각 섹션에 대해:
1. 섹션 라벨, 이전 섹션 전체 텍스트(중복 방지), 전체 목차, 소스, 사용 가능 이미지, 그리고 깊이별 지시문(`PlannedSection.depth_directive`)을 TechGuideSectionPrompt에 전달합니다. 프롬프트는 공유 `STYLE_RULES`로 리뷰·요약과 톤을 통일하고, 작성 전 명시적 중복 검증을 수행합니다.
2. HTMLTagOutputParser로 `<section_markdown>`을 추출하고, 실패 시 StrOutputParser로 폴백합니다.

**평가 및 재작성** (`tech_guide.py:_evaluate_and_revise`)  
`max_revision_attempts > 0`일 때, TechGuideEvaluationPrompt가 각 섹션을 0–100점(깊이 적합/구조/문체/중복/시각·실용성)으로 채점하고 구체적 피드백을 반환합니다. `min_quality_score` 미만이면 피드백을 깊이 지시문에 덧붙여 재작성합니다(논문 리뷰의 reflect 루프와 동일).

**사실 검증** (`tech_guide.py:_ground_section`)  
`verify_grounding=True`일 경우 TechGuideGroundingPrompt로 작성된 섹션을 재검토하여 비근거 청구를 제거합니다(빈 결과 시 원본 유지 — 검증은 섹션을 개선만 할 뿐 삭제하지 않음).

**발행** (`tech_guide_main.py`)  
완성된 TechGuide는 `_format_guide()`에서 프론트매터 및 소스 링크와 함께 조합되어 Publisher를 거쳐 발행됩니다.

> **다이어그램(향후 과제)**: "시각 요소"는 현재 표·소스 이미지·코드 블록으로 충족합니다. LLM 생성 다이어그램(Mermaid handDrawn→PNG / mingrammer `diagrams`로 실제 AWS 아이콘)은 HTTP 렌더로 기술적으로 가능함을 확인했으나(Chromium 불필요), 별도 과제로 보류되어 있습니다.

**토큰 및 예산 추적**  
세 파이프라인 모두 TokenUsageTracker 콜백(`main.py:173-204`)으로 Bedrock 호출의 입출력 토큰을 추적하며, `max_total_tokens` 설정으로 초과 시 실행을 중단합니다(ExplainerGraph의 `_enforce_token_budget()`, `explainer.py:552-555`).

## 4. 모듈 레퍼런스 (Module Reference)

### `arxiv_handler.py`
**책임**: arXiv API를 통해 논문 메타데이터 및 PDF를 조회·다운로드하며, 속도 제한과 재시도 로직을 관리합니다.

**주요 클래스/함수**:
- `ArxivMetadata` (line 57): 논문 메타데이터를 정규화하고 검증하는 Pydantic 모델. arXiv ID가 존재하면 자동으로 arXiv DOI (`10.48550/arXiv.<id>`)를 생성합니다 (line 95-97).
- `ArxivHandler` (line 114): arXiv 클라이언트를 래핑하여 `download_paper(arxiv_id, papers_dir)` (line 126), `fetch_metadata(arxiv_id)` (line 145), `search_by_title(title, max_results)` (line 226)을 제공합니다.
- `_ARXIV_RATE_LIMITER` (line 26): 프로세스 전체에서 공유하는 토큰 버킷 제한기 (1요청/3초, arXiv 정책 준수).
- `_fetch_single_paper_once()` (line 182): 속도 제한을 획득한 후 API를 호출하며, 일시적 오류 시 Retry-After를 존중하도록 처리합니다 (line 192).

---

### `paper_source.py`
**책임**: 이질적인 논문 출처(arXiv ID, arXiv URL, 임의 PDF URL)를 추상화하여 통합 인터페이스를 제공합니다.

**주요 클래스/함수**:
- `PaperSource` (line 61): 추상 베이스, `source_id`, `pdf_url`, `fetch_metadata()`, `download_pdf()` 계약을 정의합니다.
- `ArxivSource` (line 96): `ArxivHandler`에 위임하는 구현체.
- `PdfUrlSource` (line 126): 임의 URL에서 PDF를 검증하고 다운로드합니다. SSRF 가드 (line 142 `assert_url_scheme_and_literal`, line 269 `assert_url_is_public`)로 악의적 URL을 거부하며, 내장된 PDF 문서 메타데이터에서 제목을 추출합니다 (line 241).
- `resolve_paper_source(source, arxiv_handler)` (line 332): 사용자 입력을 파싱하여 올바른 `PaperSource` 하위 클래스를 선택합니다.

---

### `parser.py`
**책임**: PDF 및 HTML 콘텐츠를 파싱하여 텍스트와 그림을 추출합니다.

**주요 클래스/함수**:
- `Content` (line 54): 텍스트 콘텐츠를 보유하는 Pydantic 모델.
- `Figure` (line 67): 그림 메타데이터(경로, 설명, LLM 분석)를 저장하며, `Figure.from_llm()` (line 82)은 이미지를 base64로 인코딩하고 Bedrock에 보냅니다.
- `HTMLParser` (line 263): arXiv 및 ar5iv 중 하나에서 HTML을 페치하고 BeautifulSoup으로 파싱합니다 (line 268).
- `HTMLRichParser` (line 308): HTML을 파싱하되, 그림을 비동기로 병렬 추출·분석합니다 (line 340-406).
- `PDFParser`: Upstage Document Parse API로 PDF를 파싱하고, 실패 시(예: 대용량 PDF의 413) PyMuPDF(`fitz`)로 텍스트 레이어를 직접 추출하는 경량 폴백을 사용합니다. 폴백은 텍스트 전용이며(그림은 기본 경로에서만 추출), 텍스트 레이어가 없는 스캔 PDF는 명확한 오류로 처리합니다. 그림은 좌표 지역으로 추출합니다.

---

### `content_extractor.py`
**책임**: HTML 콘텐츠에서 인용문, 저자/주제 속성, 목차를 LLM으로 추출합니다.

**주요 클래스/함수**:
- `Attributes` (line 27): 논문의 소속기관, 카테고리, 키워드, 제목/저자(URL 출처용).
- `Citation` (line 37): 인용문 (저자, 연도, 제목, arXiv ID). `validate_arxiv_id()` (line 43)는 형식을 정규화합니다.
- `ContentExtractor` (line 57): `extract_citations(html_content)` (line 174)는 페이지 루프로 누적 인용문을 계속 요청하며 (최대 20페이지, line 60), `extract_attributes(html_content)` (line 240)는 속성을 일회성으로 추출합니다. 기존 키워드 세트는 S3/로컬 파일에서 관리됩니다 (line 76-85).

---

### `citation_metadata.py`
**책임**: Crossref 및 Semantic Scholar를 통해 인용 참고문헌의 제목, 저자, 초록, URL을 느슨하게 해결합니다.

**주요 클래스/함수**:
- `ReferenceMetadata` (line 48): 제목, 저자 목록, 선택적 초록/URL을 저장합니다.
- `MetadataProvider` (line 64): 추상 기본 클래스.
- `CrossrefProvider` (line 71): Crossref REST API 조회 (제한 없음). Crossref 작업 유형을 필터링하여 책 장/백과사전 항목을 제외합니다 (line 108, `_CROSSREF_PRIMARY_TYPES` line 39).
- `SemanticScholarProvider` (line 117): Semantic Scholar 그래프 검색 (CS/ML 추상 적용률 우수).
- `ChainedMetadataResolver` (line 146): 각 공급자를 순서대로 시도하여 첫 번째 "사용 가능한" 결과(초록 포함 선호)를 반환합니다 (line 161).
- 속도 제한: `_CROSSREF_LIMITER` (line 32, 5요청/초), `_SEMANTIC_SCHOLAR_LIMITER` (line 33, 1요청/초).

---

### `citation_summarizer.py`
**책임**: 인용 식별자(제목, arXiv ID)를 해결하고 해당 논문을 요약하여 원문 맥락에 관련된 요약을 생성합니다.

**주요 클래스/함수**:
- `CitationSummarizer` (line 25): `summarize(reference_identifiers, original_content)` (line 85)는 비동기 조회 중복 제거 및 인플라이트 맵(line 55-58)을 사용하여 동일 식별자의 동시 호출을 공유합니다.
- `_process_identifier()` (line 150): arXiv ID → arXiv 직접 조회, 일반 제목 → 메타데이터 공급자(초록 우선) → arXiv 제목 검색 → 순수 제목 분석.
- `_title_matches()` (line 198): 정규화된 제목 유사도 >= 0.72 (line 196, `TITLE_MATCH_THRESHOLD`)인 경우 메타데이터 URL을 신뢰하여 잘못된 논문으로 연결되는 것을 방지합니다.
- `_summarize_from_metadata()` (line 222): 초록에서 요약하여 네트워크 I/O를 절약하고 arXiv API를 우회합니다.

---

### `rate_limiter.py`
**책임**: 토큰 버킷 속도 제한기. 스레드 안전하며 서버 Retry-After를 존중합니다.

**주요 클래스/함수**:
- `RateLimiter` (line 21): `rate` 요청마다 `per` 초 단위로 제한합니다 (line 22-25). `acquire()` (line 41)는 토큰이 가능할 때까지 블록하며, `penalize(retry_after_seconds)` (line 53)는 서버 지시를 존중하기 위해 모든 호출자를 차단합니다 (line 59).
- `_time_until_available()` (line 67): 페널티 창 내에서 허용된 대기 시간을 계산하고 토큰을 비례적으로 보충합니다.

---

### `code_retriever.py`
**책임**: Git 저장소를 다운로드하고, Python 코드를 FAISS 벡터 인덱스로 인덱싱하여 의미적 유사도 검색을 제공합니다.

**주요 클래스/함수**:
- `CodeRetriever` (line 37): `download_repositories(repo_urls)` (line 124)는 얕은 클론(깊이=1)을 수행합니다. `create_or_load_index()` (line 171)는 Python 파일을 청킹하고(선택적으로 증강) FAISS 인덱스를 만듭니다.
- `_create_documents()` (line 207): `.py` 파일을 LangChain `Document` 객체로 읽습니다 (line 209-215). `_augment_documents()` (line 235)는 각 청크에 코드 분석 요약을 선행으로 추가합니다.
- `search_similar_code()` (line 369): 쿼리 텍스트에 대한 상위 k 코드 청크를 반환합니다.
- `generate_codebase_summary()` (line 313): 모든 Python 파일을 읽고(최대 200K 글자, line 38) LLM으로 요약합니다.

---

### `web_research.py`
**책임**: URL 목록을 조사하여 텍스트, 이미지, 링크를 추출하고, 선택적으로 동일 사이트 서브페이지 및 웹 검색 결과를 발견합니다.

**주요 클래스/함수**:
- `WebSearchProvider` (line 73): 추상 기본. `BraveSearchProvider` (line 87)는 Brave Search API 래퍼입니다.
- `WebResearcher` (line 130): `research(urls, discover_subpages, search_queries)` (line 166)은 각 URL을 페치(SSRF 가드 포함, line 209)하고 조사합니다. `_select_subpages()` (line 263)는 동일 호스트·경로 내 링크만 선택합니다.
- `ResearchCorpus` (line 50): 페이지 목록, 검색 결과, 이미지 URL을 보유합니다. `combined_text()` (line 67)는 모든 페이지를 문자 예산 내에서 연결합니다.

---

### `explainer.py`
**책임**: LangGraph 상태 그래프 기반 다단계 리뷰 파이프라인. 절(섹션)-별 분석, 인용 요약, 코드 검색, 합성, 반영 및 품질 점수를 기반으로 한 재시도를 수행합니다.

**주요 클래스/함수**:
- `Paper` (line 67): 완전한 논문 구조(메타데이터, 콘텐츠, 속성, 인용, 목차, 그림, 저장소 URL, 코드베이스 요약).
- `ExplainerGraph` (line 113): `__init__` (line 114)은 LLM 체인을 초기화하고 `_create_workflow()` (line 233)으로 LangGraph를 빌드합니다. 주요 노드는 `analyze_paper()` (line 318, 절 분할), `enrich_paper()` (line 419, 인용·코드 검색), `synthesize_paper()` (line 557, 섹션 설명 생성), `reflect_paper()` (line 496, 품질 점수 매기기). 
- `run()` (line 652)는 상태 그래프를 비동기로 실행합니다 (recursion_limit=200, line 671).
- 최대 반복(20, line 57), 최대 합성 시도(3, line 58), 최소 품질 점수(70, line 59), 최대 연속 수(8, line 64).

---

### `summarizer.py`
**책임**: 경량 5섹션 요약 생성(동기 리뷰 없음).

**주요 클래스/함수**:
- `PaperSummarizer` (line 31): `summarize(paper)` (line 78)은 논문 텍스트에서 `{"summary", "tags", "urls"}`를 추출합니다. 요약은 Markdown 조각(emoji 섹션), 태그는 쉼표로 구분된 키워드, URL은 Markdown 링크입니다.

---

### `tech_guide.py`
**책임**: URL 목록 기반 자학 기술 가이드 생성(관련성 게이트, 개요, 섹션별 작성, 사실 확인).

**주요 클래스/함수**:
- `TechGuideGenerator` (line 63): `generate(urls, discover_subpages, search_queries)` (line 134)은 연구(line 145), 관련성 확인(line 155), 개요 초안(line 156), 섹션 작성(line 157)을 수행합니다.
- `_assert_relevant()` (line 167): 소스가 기술 문서인지 확인하거나 `NotTechnicalContentError`를 발생시킵니다 (line 50).
- `_write_sections()` (line 199): 각 섹션을 작성하고(선택적으로 사실 확인, line 239), 최대 16섹션으로 제한합니다 (line 47).
- `_ground_section()` (line 245): 섹션을 소스 자료에 대해 재작성하여 근거 없는 주장을 제거합니다.

---

### `publisher.py`
**책임**: Markdown 아티팩트를 로컬 파일로 저장하고, S3로 업로드하고, GitHub 저장소에 풀 요청을 엽니다.

**주요 클래스/함수**:
- `PublishRequest` (line 40): 제목, Markdown, 작업 디렉터리, 브랜치/PR 메타데이터를 보유합니다.
- `Publisher` (line 61): `publish(request)` (line 79)은 로컬 경로를 재작성하고(line 87) Markdown을 저장한 후 S3 업로드를 수행합니다 (line 92). 수학 언더스코어를 정규화합니다 (line 85).
- `create_pull_request()` (line 127): 저장소를 클론하고(line 219), 파일을 복사하고(line 231), 커밋하고(line 248), 분기를 푸시하고(line 257) GitHub PR을 엽니다 (line 156).
- `slugify()` (line 55): 제목을 파일 시스템 안전 슬러그로 변환합니다.

---

### `metrics.py`
**책임**: Bedrock 토큰 사용률을 추적하고, 예산을 시행하고, CloudWatch로 메트릭을 방출합니다.

**주요 클래스/함수**:
- `TokenUsageTracker` (line 46): LangChain 콜백. 모든 LLM 호출의 입력·출력 토큰을 누적합니다 (line 58). `estimated_cost_usd()` (line 70)는 모델별 가격 테이블을 사용하여 비용을 추정합니다 (line 34-38, opus/sonnet/haiku).
- `TokenBudgetExceeded` (line 41): 총 토큰 예산을 초과하면 발생합니다.
- `MetricsEmitter` (line 123): CloudWatch에 InputTokens, OutputTokens, EstimatedCostUSD, DurationSeconds, Success를 발행합니다 (line 150-161). 오프라인이거나 boto 불가시 No-op입니다.

---

### `url_guard.py`
**책임**: SSRF 공격을 방지하기 위해 사용자 공급 URL을 검증합니다.

**주요 클래스/함수**:
- `UnsafeUrlError` (line 30): URL이 거부될 때 발생합니다.
- `assert_url_scheme_and_literal()` (line 52): 저렴한 오프라인 사전 확인(스킴, 호스트 유무, IP 리터럴 블로킹, 네트워크 I/O 없음).
- `assert_url_is_public()` (line 75): 전체 확인(DNS 해결, 반환된 모든 주소 검증). 프라이빗/루프백/링크 로컬/멀티캐스트/예약된/미지정/메타데이터 주소를 거부합니다 (line 34-42).
- `is_url_public()` (line 112): 예외를 발생시키지 않는 형태입니다.

---

### `markdown_math.py`
**책임**: LaTeX 수학 식이 kramdown(GFM) + MathJax 파이프라인을 통과할 수 있도록 언더스코어를 이스케이프합니다.

**주요 클래스/함수**:
- `normalize_math_underscores()`: 정규식으로 수학 span(인라인 `\( ... \)`, 디스플레이 `$$...$$`)을 찾고 unescaped 언더스코어를 `\_`로 변환하되, 코드 블록(펜스 또는 인라인)은 건드리지 않습니다. 멱등입니다. 블로그가 단일 `$` 구분자를 (통화 표기 보호를 위해) 제거하므로 `$...$`는 수식으로 취급하지 않습니다 — 인라인 수식은 반드시 `\( ... \)`를 사용합니다.

## 5. 설정 레퍼런스 (Configuration)

### 개요

Scholar-Lens의 동작은 `scholar_lens/configs/config.yaml` 파일과 환경 변수를 통해 제어됩니다. 시스템은 Pydantic 기반의 타입 검증을 수행하므로, YAML 구조와 필드 타입이 엄격하게 강제됩니다 (`scholar_lens/configs/config.py`).

### Resources 블록

AWS 인프라 및 통합 설정을 정의합니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `project_name` | `str` | `"scholar-lens"` | 프로젝트 이름 (필수) |
| `stage` | `"dev"` \| `"prod"` | `"dev"` | 배포 환경 |
| `profile_name` | `str` \| `null` | `null` | AWS 프로필 명; null일 경우 기본 프로필 사용 |
| `default_region_name` | `str` | `"ap-northeast-2"` | 기본 AWS 리전 (S3, SSM 등) |
| `bedrock_region_name` | `str` | `"us-west-2"` | Claude API 호출용 Bedrock 리전 |
| `s3_bucket_name` | `str` \| `null` | `null` | S3 버킷명; null일 경우 로컬 저장소만 사용 |
| `s3_prefix` | `str` | `"scholar-lens"` | S3 객체 접두사 |
| `vpc_id` | `str` \| `null` | `null` | VPC ID (AWS Batch 작업용) |
| `subnet_ids` | `list[str]` \| `null` | `null` | 서브넷 ID 목록 (AWS Batch 작업용) |
| `email_address` | `str` (이메일) \| `null` | `null` | 알림/로깅용 이메일 주소 |

### Resources.Github 블록

GitHub Pages 블로그 배포 설정입니다 (`scholar_lens/configs/config.py:20–61`).

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `false` | GitHub Pages 통합 활성화 여부 |
| `repo_name` | `str` \| `null` | `null` | GitHub 리포지토리명 (예: `bits-bytes-nn/bits-bytes-nn.github.io`) |
| `base_branch` | `str` | `"main"` | 기본 브랜치 (예: PR 베이스) |
| `branch_prefix` | `str` | `"paper-reviews"` | 생성되는 PR 브랜치 접두사 |
| `author_name` | `str` | `"Scholar Lens Bot"` | Git 커밋 작성자 이름 |
| `author_email` | `str` (이메일) \| `null` | `null` | Git 커밋 작성자 이메일 |
| `cover_images` | `dict[str, str]` | `{}` | 카테고리 슬러그 → 커버 이미지 파일명 매핑. 슬러그는 소문자이며 하이픈으로 구분됩니다 (예: `language-models`, `retrieval-augmented-generation`). `review_category`, `summary_category`, `tech_guide_category`를 정규화한 슬러그가 키로 사용됩니다 (`scholar_lens/configs/config.py:52–60`). |
| `default_cover_image` | `str` | `"default.jpg"` | 매칭되는 카테고리가 없을 때의 기본 커버 이미지 파일명 |
| `review_category` | `str` | `"Paper Reviews"` | Paper Review 아티팩트의 블로그 카테고리 레이블 (프론트매터에 기록) |
| `summary_category` | `str` | `"Paper Summaries"` | Paper Summary 아티팩트의 블로그 카테고리 레이블 |
| `tech_guide_category` | `str` | `"Tech Guides"` | Tech Guide 아티팩트의 블로그 카테고리 레이블 |
| `public_assets` | `bool` | `false` | S3 업로드 이미지의 퍼블릭 읽기 권한 설정. 기본값(false)은 보안상 권장. GitHub Pages 리포지토리에서 이미지를 제공하므로 S3 공개 ACL은 불필요합니다. |

**카테고리 슬러그 정규화**: 카테고리명의 대소문자, 공백, 특수문자는 소문자로 변환되고 연속된 영숫자가 아닌 문자는 단일 하이픈으로 압축됩니다. 예를 들어 `"Multimodal Learning"` → `"multimodal-learning"`. 커버 이미지 디렉토리는 `assets/images/` 아래에 위치합니다.

### Paper 블록

PDF 및 arXiv 논문 처리 모델 선택입니다 (`scholar_lens/configs/config.py:77–92`).

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `figure_analysis_model_id` | `LanguageModelId` | `CLAUDE_V3_HAIKU` | 논문 그림 분석용 모델 |
| `citation_extraction_model_id` | `LanguageModelId` | `CLAUDE_V4_5_SONNET` | 인용 문헌 추출용 모델 |
| `attributes_extraction_model_id` | `LanguageModelId` | `CLAUDE_V4_5_HAIKU` | 논문 속성(저자, 출판사 등) 추출용 모델 |
| `table_of_contents_model_id` | `LanguageModelId` | `CLAUDE_V4_5_SONNET` | 목차 생성용 모델 |
| `output_fixing_model_id` | `LanguageModelId` | `CLAUDE_V4_5_SONNET` | 구조화된 출력 포맷 검증 및 재작성용 모델 |

### Code 블록

코드 저장소 처리 및 임베딩 설정입니다 (`scholar_lens/configs/config.py:95–114`).

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `code_analysis_model_id` | `LanguageModelId` | `CLAUDE_V4_5_HAIKU` | 코드 분석용 모델 |
| `code_summarization_model_id` | `LanguageModelId` | `CLAUDE_V4_5_HAIKU` | 코드 요약용 모델 |
| `embed_model_id` | `EmbeddingModelId` | `TITAN_EMBED_V2` | Bedrock 임베딩 모델; 사용 가능한 값: `EMBED_MULTILINGUAL_V3`, `EMBED_ENGLISH_V3`, `TITAN_EMBED_V1`, `TITAN_EMBED_V2` |
| `chunk_size` | `int` | `1024` | RecursiveCharacterTextSplitter 청크 크기 (1 ~ 1,000,000) |
| `chunk_overlap` | `int` | `256` | 청크 오버랩 크기 (0 ~ `chunk_size` - 1). 검증: `chunk_overlap < chunk_size`이어야 함 |

### Citations 블록

인용 논문 메타데이터 수집 설정입니다 (`scholar_lens/configs/config.py:117–128`).

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `citation_summarization_model_id` | `LanguageModelId` | `CLAUDE_V4_5_HAIKU` | 인용 논문 요약용 모델 |
| `citation_analysis_model_id` | `LanguageModelId` | `CLAUDE_V4_5_HAIKU` | 인용 논문 분석용 모델 |
| `prefer_full_text` | `bool` | `false` | 인용 논문의 전문 다운로드 여부. `false` (권장): Crossref/Semantic Scholar/arXiv 메타데이터에서 초록을 이용하므로 호출 수가 적고 arXiv 속도 제한을 회피합니다. `true`: 각 인용 논문의 전체 텍스트를 다운로드하지만 느리고 arXiv 기반 호출이 증가합니다. |

### Explanation 블록

Paper Review 생성 (ExplainerGraph) 모델 선택입니다 (`scholar_lens/configs/config.py:130–150`).

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `paper_analysis_model_id` | `LanguageModelId` | `CLAUDE_V4_5_SONNET` | 논문 초기 분석용 모델 |
| `paper_enrichment_model_id` | `LanguageModelId` | `CLAUDE_V4_5_SONNET` | 정보 보강 단계용 모델 |
| `paper_finalization_model_id` | `LanguageModelId` | (필수) | 누적된 설명을 바탕으로 4개 섹션의 핵심 요약(`key_takeaways`)을 생성하는 모델 (기본값 없음, 반드시 지정 필요) |
| `paper_reflection_model_id` | `LanguageModelId` | `CLAUDE_V4_5_SONNET` | 리뷰 반영 및 개선용 모델 |
| `paper_synthesis_model_id` | `LanguageModelId` | `CLAUDE_V4_5_SONNET` | 단락별 종합 및 마무리용 모델 |
| `reflector_enable_thinking` | `bool` | `false` | Paper Reflection 단계의 확장 사고(thinking) 활성화 여부 |
| `synthesizer_enable_thinking` | `bool` | `false` | Paper Synthesis 단계의 확장 사고(thinking) 활성화 여부 |
| `thinking_effort` | `"low"` \| `"medium"` \| `"high"` | `"medium"` | 확장 사고 모델의 추론 강도. `thinking_effort: low`는 빠르고 저렴, `high`는 더 정교한 추론을 생성합니다. |
| `max_total_tokens` | `int` \| `null` | `null` | 한 번의 Review 실행 시 총 토큰 상한선 (null = 제한 없음). 단락별 합성 루프의 비용 폭발을 방지합니다. 설정 시 반드시 양수여야 함. |

**Thinking 모델**: Opus 4.8 등 적응형 사고 모델은 `thinking_effort`를 존중하지만, 레거시 사고 모델은 고정 토큰 예산을 사용합니다.

### Summary 블록

Paper Summary 생성 모델 선택입니다 (`scholar_lens/configs/config.py:153–157`).

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `summary_model_id` | `LanguageModelId` | `CLAUDE_V4_8_OPUS` | 논문 요약 생성용 모델 |
| `summarizer_enable_thinking` | `bool` | `false` | 요약 생성 시 확장 사고 활성화 여부 |
| `thinking_effort` | `"low"` \| `"medium"` \| `"high"` | `"medium"` | 확장 사고 강도 |

### TechGuide 블록

Tech Guide 생성 모델 선택입니다 (`scholar_lens/configs/config.py:159–171`).

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `relevance_model_id` | `LanguageModelId` | `CLAUDE_V4_5_HAIKU` | 논문의 기술 가이드 관련성 판단용 모델 |
| `synopsis_model_id` | `LanguageModelId` | `CLAUDE_V4_6_SONNET` | 기술 개념 개요 작성용 모델 |
| `writing_model_id` | `LanguageModelId` | `CLAUDE_V4_8_OPUS` | Tech Guide 본문 작성용 모델 |
| `writer_enable_thinking` | `bool` | `false` | 가이드 작성 시 확장 사고 활성화 여부 |
| `thinking_effort` | `"low"` \| `"medium"` \| `"high"` | `"medium"` | 확장 사고 강도 |
| `verify_grounding` | `bool` | `true` | 각 작성된 섹션의 출처 기반 사실 검증 활성화. `true` (권장): 할루시네이션된 API/플래그/개념을 제거하기 위해 섹션당 한 번의 LLM 호출 추가. `false`: 검증 스킵 (비용 절감). |

### Output 블록

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `output_language` | `str` | `"Korean"` | 아티팩트 생성 언어 (현재 시스템에서는 Korean 고정) |

### 환경 변수 (EnvVars)

시스템에서 인식하는 환경 변수는 `scholar_lens/src/constants.py:20–43`에 정의되어 있습니다.

| 환경 변수 | 타입 | 필수 | 설명 |
|-----------|------|------|------|
| `AWS_PROFILE_NAME` | `str` | 선택 | AWS 프로필명 (config.yaml의 `profile_name`과 동일하게 작동) |
| `BRAVE_API_KEY` | `str` | 선택 | Brave Search API 키 |
| `GITHUB_TOKEN` | `str` | 선택 | GitHub API 토큰 (GitHub Pages 배포 시 필수) |
| `LANGCHAIN_API_KEY` | `str` | 선택 | LangChain 추적용 API 키 |
| `LANGCHAIN_TRACING_V2` | `bool` | 선택 | LangChain 추적 활성화 여부 (예: `true`) |
| `LANGCHAIN_ENDPOINT` | `str` | 선택 | LangChain 엔드포인트 URL |
| `LANGCHAIN_PROJECT` | `str` | 선택 | LangChain 프로젝트명 |
| `LOG_LEVEL` | `str` | 선택 | 로깅 레벨 (예: `INFO`, `DEBUG`) |
| `SLACK_BOT_TOKEN` | `str` | 선택 | Slack Bot 토큰 (Socket Mode 통합 필수) |
| `SLACK_APP_TOKEN` | `str` | 선택 | Slack App 토큰 (Socket Mode 통합 필수) |
| `SLACK_EXPECTED_APP_ID` | `str` | 선택 | 허용되는 Slack App ID (예: `A12345XYZ`). 설정 시 시스템은 `SLACK_BOT_TOKEN`이 지정된 앱에 속하는지 검증하고, 다른 앱의 토큰이면 시작 시 실패합니다. Socket Mode 프로세스 충돌 방지 (예: 다른 봇과 토큰 공유) 용도입니다. |
| `SLACK_ALLOWED_USER_IDS` | `str` | 선택 | 쉼표 구분 Slack 사용자 ID 허용 목록 (예: `U123ABC,U456DEF`). 설정 시 허용 목록에 없는 사용자는 작업 트리거 거부. 미설정 시 모든 워크스페이스 멤버가 트리거 가능하지만 기록됩니다. |
| `TOPIC_ARN` | `str` | 선택 | AWS SNS 토픽 ARN (알림/메시징 용도) |
| `UPSTAGE_API_KEY` | `str` | 선택 | Upstage Document AI API 키 (PDF 문서 파싱 시 필수) |

### 모델 ID (LanguageModelId 및 EmbeddingModelId)

`scholar_lens/src/constants.py:46–63` 및 `12–17`에 정의된 모든 사용 가능한 모델입니다.

**Language Models**:
- `CLAUDE_V3_HAIKU`, `CLAUDE_V3_SONNET`, `CLAUDE_V3_OPUS` (Claude 3 시리즈)
- `CLAUDE_V3_5_HAIKU`, `CLAUDE_V3_5_SONNET`, `CLAUDE_V3_5_SONNET_V2` (Claude 3.5 시리즈)
- `CLAUDE_V3_7_SONNET` (Claude 3.7 Sonnet)
- `CLAUDE_V4_SONNET`, `CLAUDE_V4_5_SONNET`, `CLAUDE_V4_6_SONNET` (Sonnet 4 시리즈)
- `CLAUDE_V4_5_HAIKU` (Haiku 4.5)
- `CLAUDE_V4_OPUS`, `CLAUDE_V4_1_OPUS`, `CLAUDE_V4_5_OPUS`, `CLAUDE_V4_6_OPUS`, `CLAUDE_V4_8_OPUS` (Opus 4 시리즈)

**Embedding Models**:
- `EMBED_MULTILINGUAL_V3`, `EMBED_ENGLISH_V3` (Cohere)
- `TITAN_EMBED_V1`, `TITAN_EMBED_V2` (Amazon Titan)

### YAML 설정 파일 예제

```yaml
resources:
  project_name: scholar-lens
  stage: prod
  profile_name: myprofile
  bedrock_region_name: us-west-2
  s3_bucket_name: my-scholar-lens-bucket
  s3_prefix: scholar-lens-prod
  email_address: admin@example.com
  github:
    enabled: true
    repo_name: myorg/myblog.github.io
    base_branch: main
    branch_prefix: paper-reviews
    author_name: Scholar Lens
    author_email: bot@example.com
    cover_images:
      language-models: lm.jpg
      nlp: nlp.jpg
    default_cover_image: default.jpg
    review_category: Research Papers
    summary_category: Paper Summaries
    tech_guide_category: Technical Guides

paper:
  figure_analysis_model_id: anthropic.claude-3-haiku-20240307-v1:0
  citation_extraction_model_id: anthropic.claude-sonnet-4-6
  attributes_extraction_model_id: anthropic.claude-haiku-4-5-20251001-v1:0
  table_of_contents_model_id: anthropic.claude-sonnet-4-6
  output_fixing_model_id: anthropic.claude-sonnet-4-6

code:
  code_analysis_model_id: anthropic.claude-haiku-4-5-20251001-v1:0
  code_summarization_model_id: anthropic.claude-haiku-4-5-20251001-v1:0
  embed_model_id: amazon.titan-embed-text-v2:0
  chunk_size: 2048
  chunk_overlap: 512

citations:
  citation_summarization_model_id: anthropic.claude-haiku-4-5-20251001-v1:0
  citation_analysis_model_id: anthropic.claude-haiku-4-5-20251001-v1:0
  prefer_full_text: false

explanation:
  paper_analysis_model_id: anthropic.claude-sonnet-4-6
  paper_enrichment_model_id: anthropic.claude-sonnet-4-6
  paper_finalization_model_id: anthropic.claude-haiku-4-5-20251001-v1:0
  paper_reflection_model_id: anthropic.claude-sonnet-4-6
  paper_synthesis_model_id: anthropic.claude-opus-4-8
  reflector_enable_thinking: true
  synthesizer_enable_thinking: true
  thinking_effort: medium
  max_total_tokens: 500000

summary:
  summary_model_id: anthropic.claude-opus-4-8
  summarizer_enable_thinking: true
  thinking_effort: medium

tech_guide:
  relevance_model_id: anthropic.claude-haiku-4-5-20251001-v1:0
  synopsis_model_id: anthropic.claude-sonnet-4-6
  writing_model_id: anthropic.claude-opus-4-8
  writer_enable_thinking: true
  thinking_effort: medium
  verify_grounding: true

output_language: Korean
```

### 검증 규칙

- **Code.chunk_overlap < Code.chunk_size**: RecursiveCharacterTextSplitter 안정성 보장 (scholar_lens/configs/config.py:106–114)
- **Explanation.max_total_tokens**: 양수 또는 null만 허용; 0 이하는 거부 (scholar_lens/configs/config.py:150)
- **Resources.project_name**: 1글자 이상 필수
- **이메일 필드**: EmailStr 타입으로 유효한 이메일 형식만 수용
- **LanguageModelId, EmbeddingModelId**: 정의된 열거형 값만 허용; 임의의 문자열 불가

### 로드 순서

1. 환경 변수 로드 (dotenv 통해 `.env` 파일)
2. `scholar_lens/configs/config.yaml` 로드 (존재할 경우)
3. YAML이 없으면 모든 필드 기본값 적용 (scholar_lens/configs/config.py:218–224)
4. `Config.load()` 호출 시 위 순서대로 병합되어 `Config` 인스턴스 반환

## 6. 모델 사용 (Model Usage)

Scholar-Lens는 논문 분석의 각 단계에서 서로 다른 Claude/Titan 모델을 사용하도록 설정합니다. 모든 모델 ID는 Bedrock 런타임을 통해 관리되며, 설정은 YAML 기반 구성(`scholar_lens/configs/config.py:178-209`) 및 모델 정보 사전(`scholar_lens/src/utils/models.py:71-167`)에 정의됩니다.

### 기본 모델 선택

각 처리 단계의 기본 모델 ID는 다음과 같습니다:

| 단계 | 컴포넌트 | 기본 모델 | 모델 ID |
|------|---------|---------|--------|
| **논문 파싱** | 그림 분석 | Claude 3 Haiku | `anthropic.claude-3-haiku-20240307-v1:0` |
| | 인용문 추출 | Claude 4.5 Sonnet | `anthropic.claude-sonnet-4-5-20250929-v1:0` |
| | 속성 추출 | Claude 4.5 Haiku | `anthropic.claude-haiku-4-5-20251001-v1:0` |
| | 목차 추출 | Claude 4.5 Sonnet | `anthropic.claude-sonnet-4-5-20250929-v1:0` |
| | 출력 수정 | Claude 4.5 Sonnet | `anthropic.claude-sonnet-4-5-20250929-v1:0` |
| **코드 분석** | 코드 분석 | Claude 4.5 Haiku | `anthropic.claude-haiku-4-5-20251001-v1:0` |
| | 코드 요약 | Claude 4.5 Haiku | `anthropic.claude-haiku-4-5-20251001-v1:0` |
| | 임베딩 | Titan Embed V2 | `amazon.titan-embed-text-v2:0` |
| **인용 처리** | 인용 요약 | Claude 4.5 Haiku | `anthropic.claude-haiku-4-5-20251001-v1:0` |
| | 인용 분석 | Claude 4.5 Haiku | `anthropic.claude-haiku-4-5-20251001-v1:0` |
| **리뷰 생성** | 논문 분석 | Claude 4.5 Sonnet | `anthropic.claude-sonnet-4-5-20250929-v1:0` |
| | 논문 보강 | Claude 4.5 Sonnet | `anthropic.claude-sonnet-4-5-20250929-v1:0` |
| | 논문 최종화 | Claude 4.5 Haiku | `anthropic.claude-haiku-4-5-20251001-v1:0` |
| | 논문 반영 | Claude 4.5 Sonnet | `anthropic.claude-sonnet-4-5-20250929-v1:0` |
| | 논문 합성 | Claude 4.5 Sonnet | `anthropic.claude-sonnet-4-5-20250929-v1:0` |
| **요약 생성** | 요약 모델 | Claude 4.8 Opus | `anthropic.claude-opus-4-8` |
| **기술 가이드** | 관련성 판단 | Claude 4.5 Haiku | `anthropic.claude-haiku-4-5-20251001-v1:0` |
| | 개요 작성 | Claude 4.6 Sonnet | `anthropic.claude-sonnet-4-6` |
| | 가이드 작성 | Claude 4.8 Opus | `anthropic.claude-opus-4-8` |
| | 근거 검증 (선택) | Claude 4.8 Opus | `anthropic.claude-opus-4-8` |

이 기본값은 각 단계의 복잡성에 맞춰 최적화되었습니다: 빠른 단계(속성/인용 추출)는 Haiku를 사용하고, 복잡한 추론(리뷰 분석, 기술 가이드 작성)은 Sonnet 또는 Opus를 사용합니다.

### 적응형 사고(Adaptive Thinking) — Opus 4.8+

Claude 4.8 Opus와 이후 모델들은 **적응형 사고** API를 사용하여, 쿼리 복잡도에 따라 사고 토큰 할당을 자동으로 조정합니다(`scholar_lens/src/utils/models.py:40-48`). 이전의 고정 예산 방식(`thinking.type='enabled' + budget_tokens`)과는 달리, 적응형 방식(`thinking.type='adaptive' + output_config.effort`)은 세 가지 노력 수준을 제공합니다(`scholar_lens/configs/config.py:14-17`):

- **`low`**: 신속한 답변을 위해 적은 사고 토큰 사용 (빠른 확인/스크리닝용)
- **`medium`** (기본값): 균형 잡힌 추론 (대부분의 생성 작업용)
- **`high`**: 광범위한 추론을 위해 많은 사고 토큰 사용 (복잡한 분석/합성용)

#### 적응형 사고 구성

요약(`PaperSummarizer`)과 기술 가이드(`TechGuideGenerator`)에서 적응형 사고는 선택적으로 활성화됩니다:

```python
# 요약기 생성 (scholar_lens/main.py:276-284)
summarizer = PaperSummarizer(
    summary_model_id=LanguageModelId(context.config.summary.summary_model_id),
    enable_thinking=context.config.summary.summarizer_enable_thinking,  # 기본값: False
    thinking_effort=context.config.summary.thinking_effort,  # 기본값: "medium"
    ...
)

# 기술 가이드 생성 (scholar_lens/tech_guide_main.py:156-168)
generator = TechGuideGenerator(
    writing_model_id=LanguageModelId(context.config.tech_guide.writing_model_id),
    enable_thinking=context.config.tech_guide.writer_enable_thinking,  # 기본값: False
    thinking_effort=context.config.tech_guide.thinking_effort,  # 기본값: "medium"
    verify_grounding=context.config.tech_guide.verify_grounding,  # 기본값: True
    ...
)
```

리뷰 생성(`ExplainerGraph`)의 반영기(Reflector)와 합성기(Synthesizer)도 독립적으로 사고를 제어할 수 있습니다(`scholar_lens/configs/config.py:144-146`):

```python
explainer = ExplainerGraph(
    ...
    reflector_enable_thinking=context.config.explanation.reflector_enable_thinking,
    synthesizer_enable_thinking=context.config.explanation.synthesizer_enable_thinking,
    thinking_effort=context.config.explanation.thinking_effort,
    ...
)
```

#### 온도 및 컨텍스트 윈도우 처리

적응형 사고 모델(Opus 4.8+)에서 온도 매개변수는 더 이상 사용되지 않습니다. 대신 `output_config.effort`로 추론 깊이를 제어합니다(`scholar_lens/src/utils/factories.py:406-413`). 따라서 팩토리는:

- Opus 4.8+에서 온도를 설정하지 않음 (`model_info.uses_adaptive_thinking == True`일 때)
- 이전 모델에서는 온도를 기본값 0.0으로 설정 (결정론적 생성)
- 사고 활성화 시 온도를 1.0으로 강제 설정 (레거시 모델에만)

1M 컨텍스트 윈도우 지원(`supports_1m_context_window=True`)은 분석/요약/기술 가이드 등 전체 본문을 받는 단계에서 명시적으로 활성화되어, 대규모 논문 텍스트와 소스 자료를 처리할 수 있게 합니다(`scholar_lens/src/summarizer.py`, `scholar_lens/src/tech_guide.py`, `scholar_lens/src/explainer.py`).

#### 컨텍스트 윈도우 맞춤(텍스트 피팅)

긴 논문/문서가 모델의 컨텍스트 한도를 초과하지 않도록, 전체 본문을 모델에 보내는 모든 경로는 호출 직전에 텍스트를 모델 예산에 맞게 잘라냅니다. 이 로직은 단일 공유 메서드 `BedrockLanguageModelFactory.fit_text`로 일원화되어 있으며(`scholar_lens/src/utils/factories.py`), content_extractor(인용/속성/목차), explainer(analyze), summarizer, tech_guide, citation_summarizer가 모두 이를 사용합니다.

핵심 원칙은 **토큰 수를 추정(예: chars-per-token, 외부 토크나이저)하지 않고 Bedrock `CountTokens` API로 정확히 측정**하는 것입니다(`count_tokens`). 1M 모델에서는 200K를 초과하는 입력도 측정할 수 있도록 long-context 베타 플래그(`anthropic_beta: ["context-1m-2025-08-07"]`)를 함께 전달하며, 측정 대상 모델 ID는 실제 호출에 쓰이는 크로스 리전 해석 ID를 사용합니다. `CountTokens` 호출에는 **베이스 모델 ID**를 사용합니다 — 크로스 리전 추론 프로파일 ID(`apac.`/`global.`)는 CountTokens가 거부하기 때문입니다(Converse와 다름). 또한 1M 윈도우를 쓰는 모델은 항상 Converse 경로로 호출해 베타 헤더가 실제로 전달되도록 합니다(레거시 InvokeModel 경로에선 누락됨). `fit_text`는 전체 텍스트가 예산을 초과하면 정확한 토큰 수를 기준으로 **이진 탐색**하여 예산 내에 들어가는 가장 긴 접두부를 찾습니다. `CountTokens`가 일시적으로 실패하면(스로틀링 등) 본문을 그대로 두어 콘텐츠가 조용히 잘려나가지 않게 합니다(실제 초과 시에는 모델 호출의 재시도/오류 경로가 처리). 임베딩 모델(Titan/Cohere)은 `CountTokens`가 지원되지 않아 문자 수 상한만 적용합니다.

이 단계의 IAM 권한으로 잡 역할에 `bedrock:CountTokens`가 부여됩니다(`scripts/deploy_infra.py`).

### 모델 정보 조회 및 팩토리 패턴

`BedrockLanguageModelFactory`(`scholar_lens/src/utils/factories.py:276-457`)는 단일 팩토리로부터 모든 언어 모델을 생성합니다. 모델 정보는 정적 사전(`_LANGUAGE_MODEL_INFO`)에 저장되어, 각 모델의 컨텍스트 윈도우 크기, 최대 출력 토큰, 사고/캐싱 지원 여부를 조회할 수 있습니다. 팩토리는:

1. 요청된 모델 ID의 지원 여부 검증
2. 크로스 리전 모델 ID 해석 (필요 시)
3. 사고/성능 최적화/컨텍스트 윈도우 기능 적용
4. 적응형/레거시 사고 API 중 적절한 것 선택

요청할 때마다 새 모델 인스턴스를 생성하므로, 각 파이프라인 단계의 호출자가 필요한 구성(온도, 최대 토큰, 사고 설정)을 팩토리에 명시적으로 전달합니다(`scholar_lens/src/utils/factories.py:293-317`).

## 7. 프롬프트 & 설명 알고리즘 (Prompt & Explanation Algorithm)

### 개요

Scholar-Lens의 핵심 엔진인 ExplainerGraph는 LangGraph 기반의 상태 머신으로 동작하며, 논문 내용을 섹션별로 합성(synthesize)하고 반영(reflect)하며 지속적으로 품질을 개선하는 루프를 실행합니다. 이 알고리즘은 **우선순위 기반 프롬프트 규칙 계층(shared rule blocks)**, 품질 점수 게이팅(quality score gating), 토큰 예산 보호(token budget guard), 그리고 인용 접지(citation grounding) 규칙을 결합하여 정확하고 접근 가능한 기술 리뷰를 생성합니다.

### 워크플로우 아키텍처

ExplainerGraph의 워크플로우는 다섯 단계로 구성됩니다(scholar_lens/src/explainer.py:233-315):

1. **분석(analyze)**: PaperAnalysisPrompt를 사용해 문장을 파싱하고 섹션 경계를 감지합니다.
2. **보강(enrich)**: PaperEnrichmentPrompt로 각 섹션에 필요한 인용(citation_summaries) 및 코드(code) 예제를 선별합니다.
3. **합성(synthesize)**: PaperSynthesisPrompt가 현재 섹션에 대해 반복적으로 설명을 생성합니다.
4. **반영(reflect)**: PaperReflectionPrompt가 생성된 설명의 품질을 평가하고 개선 피드백을 제공합니다.
5. **최종화(finalize)**: PaperFinalizationPrompt가 모든 설명을 통합하여 핵심 정보를 추출합니다.

조건부 엣지는 반영 단계의 품질 점수에 따라 합성을 재시도하거나 다음 섹션으로 진행할지 결정합니다(scholar_lens/src/explainer.py:294-301).

### 합성→반영 루프

**합성 단계** (synthesize_paper, scholar_lens/src/explainer.py:557-611):

- 현재 인덱스에서 단락을 입력받고, 이전 설명(previous_explanation)과 누적된 피드백(accumulated_feedback)을 맥락으로 사용합니다.
- 루프는 max_continuations(기본값 8, ExplainerConfig.MAX_CONTINUATIONS)만큼 반복되며, 각 반복에서 `_synthesize_paper()`를 호출합니다.
- 모델이 반환하는 `<has_more>y</has_more>` 태그를 확인하여 연속성을 결정합니다(scholar_lens/src/explainer.py:596-603).
- 최대 연속 횟수 도달 시 경고를 로깅하고 중단하여, 모델이 무한 루프로 입력 비용을 증가시키는 것을 방지합니다.

**반영 단계** (reflect_paper, scholar_lens/src/explainer.py:495-550):

- PaperReflectionPrompt는 생성된 설명을 검토하고 0~100 범위의 품질 점수를 반환합니다.
- 점수를 파싱할 때 정규표현식 `r"-?\d+"`를 사용하여 "85/100", "N/A" 등의 잘못된 형식을 처리하며, 파싱 실패 시 기본값 0을 사용합니다(scholar_lens/src/explainer.py:542-544).
- 반영 단계는 피드백을 누적된 피드백 리스트에 추가하고 synthesis_attempts를 증가시킵니다(scholar_lens/src/explainer.py:536, 549).

**조건부 재시도** (should_retry_synthesis, scholar_lens/src/explainer.py:248-267):

- 품질 점수가 min_quality_score(기본값 70)을 충족하면 다음 섹션으로 진행합니다.
- 점수가 기준 이하이고 synthesis_attempts < max_synthesis_attempts(기본값 3)이면 합성을 재시도합니다.
- 최대 재시도 횟수 도달 후에도 기준을 충족하지 못하면, 경고를 로깅하고 현재 점수의 설명으로 진행합니다.

이 루프는 **짧은 피드백 주기**를 통해 반복적으로 설명의 정확성, 구조 준수, 기술적 깊이를 개선합니다.

### 토큰 예산 보호

_enforce_token_budget() (scholar_lens/src/explainer.py:552-555)는 각 주요 단계(analyze, enrich, finalize, reflect, synthesize_paper 내부 루프)에서 호출됩니다:

```python
def _enforce_token_budget(self) -> None:
    """Abort the run if the configured total-token budget is exceeded."""
    if self._token_tracker is not None and self.max_total_tokens:
        self._token_tracker.check_budget(self.max_total_tokens)
```

TokenUsageTracker 콜백(callbacks 리스트에서 찾음)은 누적된 입출력 토큰을 추적하고, 예산 초과 시 예외를 발생시켜 전체 실행을 중단합니다. 이는 **긴 논문이나 여러 재시도로 인한 비용 폭발**을 방지합니다.

### 품질 점수 게이팅

PaperReflectionPrompt(scholar_lens/src/prompts/prompts.py:1064-1624)는 0~100점 척도로 다음을 평가합니다:

- **출처 충실도(Source Fidelity, 25점)**: 생성된 설명이 원본 논문과 일치하는지, 인용이 정확한지.
- **목차 정렬(TOC Alignment, 20점)**: 섹션 역할에 맞는 내용 범위와 구조 준수.
- **내용 커버리지(Coverage, 25점)**: 지정된 granularity(STANDARD/DETAILED)와 기술적 깊이(BASIC/INTERMEDIATE/ADVANCED)에 맞는 설명.
- **보조 자료 통합(Supplementary Materials, 20점)**: INTERMEDIATE 또는 ADVANCED 수준에서 인용 및 코드 예제의 적절한 사용.
- **언어 및 스타일(Language & Style, 10점)**: 한국어 품질, 참고 방식, 형식.

**중요한 자동 0점 위반 사항**:
- 섹션 내 90% 이상의 텍스트 내용이 반복되는 경우 (자동 0점).
- 시각 요소 중복 삽입 (그림, 표, 코드).
- 출처에서 문자 단위로 일치해야 하는 이미지 경로 수정.
- 직접 모순되는 기술적 주장.

게이팅 임계값인 MIN_QUALITY_SCORE(70)을 설정함으로써, 낮은 품질의 설명은 개선되거나 최대 시도 횟수가 소진될 때까지 재합성됩니다.

### 공유 프롬프트 규칙 블록 (Shared Rule Constants)

Scholar-Lens는 여러 프롬프트에서 재사용되는 **규칙 상수(rule blocks)**를 scholar_lens/src/prompts/prompts.py의 모듈 수준에서 정의합니다:

#### HEADING_STRUCTURE_RULES (라인 85-97)

- 메인 제목은 # (단일 해시), 섹션은 ##, 하위섹션은 ###, 그 이상은 ####, #####.
- **절대 금지**: 제목에 섹션 번호를 포함("## 1. 서론" 금지, "## 서론" 필수).
- 헤딩 레벨을 건너뛰기 금지(예: ##에서 ####로 직접 이동 불가).

#### CITATION_KEY_RULES (라인 99-120)

- **절대 금지**: citation key 노출(예: smith2023transformer, brown2024attention을 본문에 기록 금지).
- 하이퍼링크 URL은 citation_summaries에 나타나는 URL만 사용(자신의 기억에서 URL 구성 금지 → 잘못된 arXiv ID 연결 위험).
- 저자 정보와 URL이 모두 있을 때: "[Author et al.](url)에서 제안된" 형식.
- URL이 없거나 불명확한 경우: 하이퍼링크 없이 일반 텍스트(예: "Transformer 아키텍처", "BERT 모델").

#### EXCLUDED_CONTENT_RULES (라인 122-138)

절대 제외할 비기술적 행정 콘텐츠:
- Acknowledgments (감사의 말)
- Author Contributions (저자 기여도)
- Funding Information (연구비 지원)
- References/Bibliography 섹션 (in-text citations는 포함).

**침묵하는 제외**: 이들 섹션을 메타 주석 없이 조용히 생략("이 부분은 참고문헌이라 제외합니다" 같은 설명 금지).

#### IMAGE_PATH_RULES (라인 140-161)

- 이미지 경로는 **출처에서 문자 단위로 정확하게 복사**. 수정 금지.
- 로컬 경로 → 로컬 경로로 유지(`/exact/local/path.png`).
- HTTPS URL → HTTPS URL로 유지.
- 상대 경로 → 상대 경로로 유지(`./exact/relative/path.png`).
- **금지된 수정**: 로컬→URL 변환, 슬래시 변경, 확장명 추가/제거, 기타 문자 변경.
- **절대 금지**: URL을 추측하거나 조작하여 대신 논문 방문 페이지나 `/abs/` URL 사용(이들은 웹 페이지이지 이미지가 아님).

#### VISUAL_DUPLICATION_RULES (라인 163-179)

- 각 시각 요소(그림, 표, 코드 블록, 수식)는 **전체 문서에서 한 번만** 삽입.
- 첫 삽입 후 재사용 시: 설명적 참조만 사용 ("위 그림에서 보듯이", "앞서 보여준 표와 같이").
- **절대 금지**: 그림 번호 참조 ("그림 3에서", "Figure 1에서").
- 모든 삽입된 그림은 주변 본문에서 참조되어야 함(캡션만 있는 고아 이미지 금지).

#### TABLE_RENDERING_RULES (라인 181-191)

- 표는 **마크다운 표로 렌더링** (파이프 구분자 사용).
- **절대 금지**: 표를 이미지 링크로 렌더링 (`![표 1: 성능 비교](url)` 형식 금지).

#### STYLE_RULES (라인 196-209, KOREAN_STYLE_RULES와 동일)

- 자연스럽고 흐르는 본문, 전문적이고 접근 가능한 학술 톤.
- 1인칭 회피("우리"/"저" 금지 → "이 방법이 적용되었습니다").
- 명시적 섹션 번호 참조 금지("3.2절에서" 금지 → "앞서 설명한...").
- 기술 용어는 한국어 번역 사용, English-only는 명확한 번역이 없는 proper nouns(BERT, GPT) 또는 기술 용어.
- 한국어 출력: "입니다" 스타일.

#### GRANULARITY_RULES (라인 11-23)

- **STANDARD**: 핵심 아이디어와 주요 개념의 간결한 요약. 필수 세부사항, 방정식, 그림 포함. 소개, 관련 연구, 평가, 결론에 적합.
- **DETAILED**: 모든 기술 내용의 포괄적 설명. 모든 방정식, 알고리즘, 방법론 세부사항 포함. 단계별 설명. 원본의 어떤 것도 생략하지 않음.

#### TECHNICAL_DEPTH_RULES (라인 25-46)

- **BASIC**: 원본 표현/용어/예제/방정식 사용. 핵심 개념 명확하게 설명. 기본 이해와 정확한 표현에 초점.
- **INTERMEDIATE**: 보조 자료 포함. 상세한 개념 설명. 실용 예제, 사용 사례. 참조 논문의 설명 통합. 수학적 도출 확장. **필수: 적당한 보조 자료 사용 (인용 또는 코드)**.
- **ADVANCED**: 가장 상세한 설명. 심화된 이론적 토대와 수학적 증명. 광범위한 예제와 구현. **필수: 광범위한 보조 자료 사용 (인용 AND 코드)**.

### PaperSynthesisPrompt의 우선순위 계층

PaperSynthesisPrompt (scholar_lens/src/prompts/prompts.py:1627-2442)는 명시적인 **우선순위 계층**을 정의합니다:

**우선순위 1 - 출처 충실도 (HIGHEST PRIORITY)**:
- 논문의 핵심 내용을 절대 변경, 오도, 또는 모순되게 해석하지 않음.
- 정확성이 모든 다른 지침(스타일, 간결함, 개선 피드백)보다 우선.
- **코드 충실도**: 논문이 명시하지 않은 상수 추가 금지(예: `* 0.01` 스케일).
- **계산 vs 보도**: 직접 도출한 수치와 논문이 보도한 수치 구분.

**우선순위 2 - 중복 제거 (ZERO DUPLICATION)**:
- previous_explanation의 어떤 내용도 반복 금지.
- 모든 시각 요소(그림, 표, 코드, 수식)는 한 번만 삽입.
- 검증 체크리스트: 이미지 경로, 표 구조, 코드 블록, 수식, 설명 개념 모두 확인.
- 이미 나타난 콘텐츠는 설명적 참조로만 사용.

**우선순위 3 - 개선 피드백 구현**:
- improvement_feedback 지시를 우선순위 1, 2 제약 내에서 구현.
- 피드백이 정확성 또는 중복 규칙과 충돌하면 정확성 우선.

### 인용 접지 규칙

**기술적 인용 관리** (PaperEnrichmentPrompt, scholar_lens/src/prompts/prompts.py:833-950):

- PaperEnrichmentPrompt는 주어진 단락에 대해 **최대 8개**의 필수 인용을 식별합니다(라인 903).
- **포함 기준**: 기초 알고리즘, 핵심 이론적 틀, 기술 스펙, 기본 방법론 패러다임, 평가 메트릭, state-of-the-art 비교.
- **제외 기준**: 지나친 언급, 일반 배경, 부수적 인용, 자기 인용(기초 방법 제외).
- arXiv 논문: 표준 ID 형식(2103.14030), non-arXiv: 완전하고 정확한 제목.

**인용 요약 통합** (enrich_paper, scholar_lens/src/explainer.py:418-475):

- 식별된 인용 ID를 citation_summarizer.summarize()에 전달합니다(라인 448).
- citation_summaries는 dictionary 리스트로, 각 항목이 저자, 제목, URL, 핵심 기여 등을 포함합니다(라인 467).
- 합성 및 반영 단계로 전달되어 보조 자료 선택을 알립니다.

**하이퍼링크 안전성**:
- CITATION_KEY_RULES에 따라, URL은 citation_summaries에 나타나는 것만 사용.
- 자신의 기억으로 URL 생성 금지 → 잘못된 arXiv ID 또는 중복 URL 위험.
- 저자/제목이 명확하고 URL이 있을 때만 하이퍼링크 삽입.

### 토큰 생성 제어

**단락 크기 관리** (scholar_lens/src/explainer.py:576-603):

- synthesize_paper는 ~3000 토큰 주변에서 **자연스러운 단락 경계**에서 중단.
- `<has_more>y</has_more>`를 설정하여 다음 반복에서 연속.
- `</explanation>` 태그는 항상 닫혀야 함.

**최대 연속 제한** (ExplainerConfig.MAX_CONTINUATIONS = 8):
- 모델이 계속 `has_more=y`를 내보내더라도 최대 8회 반복으로 제한.
- 9번째 반복에서 경고를 로깅하고 중단하여, 의도하지 않은 반복 루프 비용 폭발 방지.

---

이 설계는 **정확성 우선 기반 우선순위**, **엄격한 중복 방지**, **누적된 피드백을 통한 반복 개선**, 그리고 **리소스 보호(토큰 및 연속 횟수)**를 결합하여, Scholar-Lens가 신뢰할 수 있고 접근 가능한 기술 리뷰를 생성하도록 합니다.

## 8. 인용 처리 (Citation Pipeline)

### 개요

인용 처리는 원문에서 추출된 참고문헌(reference identifier)을 구조화된 메타데이터로 변환하고, 각 참고문헌의 핵심 내용을 요약한 뒤 원문 본문에 맥락에 맞게 통합하는 단계입니다. 이전 설계는 각 참고문헌마다 arXiv API를 3회 이상 호출했으나, 이는 대량의 동시 요청으로 인한 HTTP 429 폭증을 초래했습니다. 재설계된 파이프라인은 **추상 우선(abstract-first) 메타데이터 해석**을 도입하여 arXiv를 회피하고, 관용적 API(Crossref, Semantic Scholar)를 통해 추상과 출판 정보를 얻은 뒤, 필요시에만 arXiv로 대체합니다.

### 메타데이터 해석 (Metadata Resolution)

#### 제공자 체인 (ChainedMetadataResolver)

메타데이터는 두 외부 API로부터 순차적으로 해석됩니다(`scholar_lens/src/citation_metadata.py:146-183`):

1. **Crossref API** (`CrossrefProvider`, 라인 71-114):
   - 무료 API, 인증키 불필요, 광범위한 학술 출판물 커버리지
   - 쿼리 매개변수: `query.bibliographic` (제목), `rows=1`, `select` 필드(제목, 저자, 초록, URL, 유형)
   - 각 제공자의 속도 제한: 5.0 요청/초 (`_CROSSREF_LIMITER`, 라인 32)

2. **Semantic Scholar API** (`SemanticScholarProvider`, 라인 117-143):
   - 컴퓨터과학/ML 분야 초록 커버리지 우수
   - 쿼리 매개변수: `query`, `limit=1`, `fields` 선택 (제목, 초록, 저자, URL)
   - 각 제공자의 속도 제한: 1.0 요청/초 (`_SEMANTIC_SCHOLAR_LIMITER`, 라인 33)

**선호도 규칙** (라인 161-183):
- 초록 *보유* + URL 보유: 그 결과를 선택 (최선)
- 초록만 보유: 첫 번째 히트 중 초록 있는 결과를 선택
- 모두 실패: 첫 번째 비어있지 않은 결과 반환 (제목/저자/URL 사용 가능)
- 절대로 서로 다른 제공자의 결과를 **혼합하지 않음**. 한 제공자의 URL을 다른 제공자의 메타데이터에 붙일 경우 잘못된 논문을 가리킬 수 있으므로.

#### JATS 제거 및 정규화

Crossref 초록은 JATS XML 태그를 포함합니다. `_strip_jats()` 함수(라인 186-192)는 정규식 `<[^>]+>`로 태그를 제거하고, 공백을 정규화한 뒤 빈 결과는 `None`으로 반환합니다.

### Crossref 작업 유형 필터 (Work-Type Filter)

AI/ML 인용에서 Crossref의 최상위 퍼지 히트는 종종 **원본 논문을 인용하는 책 장, 백과사전 항목, 단행본**입니다. 이들을 링크하면 인용이 잘못 귀속됩니다. `citation_metadata.py`의 `_CROSSREF_PRIMARY_TYPES` (라인 39-45)는 신뢰할 수 있는 작업 유형을 제한합니다:

```python
_CROSSREF_PRIMARY_TYPES = frozenset({
    "journal-article",       # 저널 논문
    "proceedings-article",   # 컨퍼런스 논문
    "posted-content",        # 프리프린트 (arXiv, bioRxiv, …)
})
```

`CrossrefProvider.lookup()` (라인 108)은 반환된 항목의 `type`이 이 집합에 속하는 경우에만 URL을 포함합니다. 다른 유형은 제목/저자/초록은 유지하되 URL을 버립니다.

### 제목 유사도 게이트 (Title-Similarity Link Gate)

메타데이터 제공자는 모호한 쿼리에 대해 잘못된 논문을 반환할 수 있습니다. 요약 단계에서 `_summarize_from_metadata()` (라인 222-253)는 URL을 첨부하기 전에 제목 유사도를 검증합니다:

**`_title_matches()` 로직** (라인 199-219):
- 쿼리 제목과 해석된 제목을 정규화: 소문자, 영숫자/공백만 유지
- 정규화된 제목의 포함 관계 확인: 한 제목이 다른 제목에 완전히 포함되면 `True`
- 포함 없으면 `SequenceMatcher` 비율을 사용: 임계값 0.72 이상 (`TITLE_MATCH_THRESHOLD`, 라인 196)
- 임계값 이하 → URL 버림, 제목을 평문으로 유지

### 단일 비행 캐싱 (Single-Flight Cache)

요약 루프는 단락마다 한 번씩 `summarize()` (라인 85-136)를 호출하므로, 같은 참고문헌이 여러 단락에서 인용될 수 있습니다. 중복 작업을 피하기 위해:

- `_summary_cache: dict[str, str | None]` (라인 57): 완료된 요약을 저장
- `_inflight: dict[str, asyncio.Future]` (라인 58): 진행 중인 요청의 Future를 저장
- **단일 비행 의미**: 같은 identifier에 대해 동시의 호출자들이 하나의 공유 Future를 기다림 (라인 96-104)

**중복 제거**도 호출 내에서 발생 (라인 113): `unique_identifiers = list(dict.fromkeys(reference_identifiers))`로 중복을 제거한 뒤 처리합니다.

### 공유 속도 제한 및 Retry-After

`rate_limiter.py` (라인 1-85)는 토큰 버킷 기반 속도 제한기를 제공합니다:

**arXiv 전역 제한기** (`arxiv_handler.py`, 라인 26):
```python
_ARXIV_RATE_LIMITER = RateLimiter(rate=1.0, per=3.0, name="arxiv")
```
- 1.0 요청 / 3초 (arXiv 권장: ~1 요청/3초)
- **모든 ArxivHandler 인스턴스가 공유**하여 동시 인용 처리가 429 폭증을 일으키지 않음

**Retry-After 처리** (라인 53-65):
- `penalize(retry_after_seconds)` 메서드: 서버의 `Retry-After` 헤더를 존중
- 절대 단조 시간 `_blocked_until` 설정, 해당 시점까지 모든 호출자 대기
- arXiv 429는 `Retry-After`를 생략하는 경우가 많으므로, 폴백 기본값 30초 사용 (`_ARXIV_429_FALLBACK_RETRY_SECONDS`, 라인 21)

**호출 패턴** (라인 87): `_ARXIV_RATE_LIMITER.acquire()`는 스레드 안전 블로킹 호출. arXiv 클라이언트가 `asyncio.to_thread`를 통해 워커 스레드에서 실행되므로 적절합니다.

### arXiv 추상 우선 경로 (Abstract-First Path)

인용 식별자가 명시적 arXiv ID인 경우(`_looks_like_arxiv_id()`, 라인 188-192):
- 정규식 `^\d{4}\.\d{4,5}(v\d+)?$` 또는 `arxiv:` 접두사 인식
- arXiv로 직접 이동

명시적 ID가 아니면 메타데이터 해석 우선 (`_process_identifier()`, 라인 150-185):
1. Crossref/Semantic Scholar로부터 메타데이터 해석 (라인 161)
2. 초록 보유 또는 `prefer_full_text=False` → 초록으로 요약 시작 (라인 162-169)
3. 초록 미보유 *and* `prefer_full_text=True` → 전문 다운로드 (`_extract_paper_content()`, 라인 275-276)
4. 메타데이터 해석 실패 또는 초록 미보유 → arXiv 제목 검색 (라인 172-183)
5. 모두 실패 → 제목 기반 분석만 시도 (`_process_title_item()`, 라인 185)

### LLM 회상 URL 금지 (No LLM-Recalled URLs)

`prompts.py`의 `CITATION_KEY_RULES` (라인 99-120)는 엄격한 제약을 부과합니다:

**핵심 규칙**:
- 초록/메타데이터에서 나온 URL만 복사(verbatim) 허용
- 메모리에서 arXiv ID나 DOI를 "회상"하지 **금지** — arXiv ID 회상은 신뢰도가 낮고 잘못된 논문을 가리킬 수 있음
- `citation_summaries`에 URL이 제공되지 않으면: 제목/모델명/알고리즘명을 평문으로만 언급, 하이퍼링크 없음
- 정확한 포맷 (저자 사용 가능한 경우): `"[Vaswani et al.](url)에서 제안된"` (URL은 `citation_summaries`에서 복사)

### arXiv 디커플링 (PaperSource)

`paper_source.py` (라인 1-352)는 arXiv 의존성을 제거합니다:

**`PaperSource` 인터페이스** (라인 61-93):
- 추상 속성: `source_id` (파일시스템 슬러그), `pdf_url`, `arxiv_html_id` (HTML 렌더링 가능 시 arXiv ID, 아니면 `None`)
- 추상 메서드: `fetch_metadata()`, `download_pdf()`

**`ArxivSource`** (라인 96-123):
- arXiv ID/API로 구현
- `fetch_metadata()`: ArxivHandler 위임
- `download_pdf()`: ArxivHandler PDF 다운로드 위임
- `arxiv_html_id` 반환

**`PdfUrlSource`** (라인 126-314):
- 임의 URL → PDF 다운로드 경로 구현
- **SSRF 보호** (라인 138-144, 268-271):
  - 구성 시: `assert_url_scheme_and_literal()` — 스키마/IP 리터럴 검증 (오프라인)
  - 다운로드 시: `assert_url_is_public()` — DNS 검증 포함 전체 체크
  - `url_guard.py`에서 제공 (내부 IP, 루프백, 비표준 포트 거부)
- **PDF 검증** (라인 256-290):
  - `Content-Type: application/pdf` 확인
  - `Content-Type: text/html` 명시 → 거부
  - 의심스러운 경우 매직 바이트 `%PDF-` 프로브 (라인 292-302)
- **다운로드 제한** (라인 46-50):
  - 타임아웃: 30초 다운로드, 15초 프로브
  - 최대 크기: 100 MB (`_MAX_PDF_BYTES`)
  - `Content-Length` 선언 시 미리 확인 (라인 201-207)
- **PDF 제목 추출** (라인 241-254):
  - PyMuPDF로 임베디드 메타데이터 추출
  - 빈/유명 제목("Untitled", 파일명) 거부

**`resolve_paper_source()` 진입점** (라인 332-352):
- 입력이 arXiv ID인지 확인: `is_arxiv_id()` (라인 317-319)
- arXiv URL (`arxiv.org`) 인지 확인: 내장 arXiv ID 추출 (라인 322-329)
- 둘 다 아님 → `PdfUrlSource`로 처리 (나중에 PDF 검증)

### 요약 형식

모든 경로의 최종 요약 형식 (`_summarize_from_metadata()` 라인 253, `_process_arxiv_item()` 라인 285):

```
Title: [title](url)
Authors: author1, author2, …

[summary text]
```

- `url`은 생략될 수 있음 (제목 불일치, 초록만 제공되는 경우)
- 저자 행이 빠질 수 있음 (저자 정보 없음)

## 9. 인프라 (AWS)

![AWS 아키텍처](./diagrams/aws-architecture.png)

### CDK 스택 구조

Scholar Lens 인프라는 AWS CDK (Cloud Development Kit)를 사용해 정의됩니다. `PaperReviewStack` 클래스 (`scripts/deploy_infra.py:97–686`)는 AWS Batch, VPC, IAM, SNS, CloudWatch, SSM 파라미터 저장소를 단일 스택으로 오케스트레이션합니다.

### VPC 및 네트워크 구성

VPC는 두 가지 모드로 설정합니다 (`scripts/deploy_infra.py:167–195`):

- **기존 VPC 연계**: `vpc_id`와 `subnet_ids`가 제공되면 해당 VPC와 서브넷을 조회해 연결합니다.
- **새 VPC 생성**: 미제공 시 프라이빗 서브넷(NAT Gateway 포함)과 퍼블릭 서브넷 2개를 생성합니다. 각 가용 영역(AZ)의 서브넷은 `/24` CIDR로 구성됩니다.

보안 그룹 (`scripts/deploy_infra.py:197–214`)은 아웃바운드 HTTPS(포트 443)만 허용하며, ECR 이미지 풀, Bedrock 호출, S3 접근, SSM 파라미터 읽기, arXiv/PDF 다운로드를 가능하게 합니다. 모든 인바운드 트래픽과 평문 HTTP는 거부됩니다.

### IAM 역할 및 권한 (최소 권한 원칙)

세 가지 IAM 역할이 정의됩니다:

#### 1. 작업 역할 (Job Role)
`scripts/deploy_infra.py:338–348`에서 정의되며, 컨테이너 내부에서 실행되는 애플리케이션이 가지는 권한입니다. `_create_job_policy_statements()` (`scripts/deploy_infra.py:219–336`)에서 생성되는 정책:

- **Bedrock 호출** (SID: `BedrockInvoke`): `bedrock:InvokeModel`, `bedrock:InvokeModelWithResponseStream` — 기본 리전과 교차 리전(bedrock_region_name) 재단 모델 및 추론 프로필에 제한됩니다.
- **Bedrock 검색** (SID: `BedrockDiscovery`): `bedrock:ListInferenceProfiles`, `bedrock:GetInferenceProfile`, `bedrock:ListFoundationModels` — 리소스 레벨 ARN을 지원하지 않으므로 `aws:RequestedRegion` 조건으로 리전을 제약합니다.
- **S3 접근** (SID: `S3ObjectAccess`, `S3ListBucket`): `s3:GetObject`, `s3:PutObject`, `s3:PutObjectAcl`, `s3:DeleteObject`, `s3:ListBucket`, `s3:GetBucketLocation` — 프로젝트 버킷과 `s3_prefix` 이하로만 제한됩니다 (`scripts/deploy_infra.py:265–291`).
- **SNS 발행** (SID: `SnsPublish`): `sns:Publish` — 이 스택의 SNS 토픽에만 발행 가능합니다.
- **SSM 읽기** (SID: `SsmReadParameters`): `ssm:GetParameter`, `ssm:GetParameters` — `/{project_name}/{stage}/*` 경로의 파라미터만 읽을 수 있습니다 (`scripts/deploy_infra.py:302–313`).
- **AWS Batch** (SID: `BatchSubmit`, `BatchDescribe`): `batch:SubmitJob`은 리소스 레벨 ARN으로 제약 (이 스택의 job-queue와 job-definition). `batch:DescribeJobs`, `batch:ListJobs`는 와일드카드(`"*"`)가 필요합니다 (`scripts/deploy_infra.py:315–335`).

#### 2. 실행 역할 (Execution Role)
`scripts/deploy_infra.py:350–363`에서 정의되며, ECS 에이전트가 Docker 이미지를 ECR에서 풀 하고 CloudWatch Logs에 쓸 권한을 갖습니다. AWS 관리형 정책 `AmazonECSTaskExecutionRolePolicy`를 연결합니다.

#### 3. 인스턴스 역할 (Instance Role)
`scripts/deploy_infra.py:365–379`에서 정의되며, Batch 컴퓨팅 환경의 EC2 인스턴스에 할당됩니다. AWS 관리형 정책 `AmazonEC2ContainerServiceforEC2Role`을 사용합니다.

### AWS Batch 자동 스케일링

#### 컴퓨팅 환경 (Compute Environment)

`_create_job_queue()` (`scripts/deploy_infra.py:510–548`)에서 두 개의 관리형 EC2 컴퓨팅 환경을 생성합니다:

| 환경 | 할당 전략 | 최대 vCPU | Spot 사용 |
|------|---------|---------|---------|
| On-Demand | `BEST_FIT_PROGRESSIVE` | 4 | 아니오 |
| Spot | `SPOT_CAPACITY_OPTIMIZED` | 8 | 예 |

두 환경은 공유 VPC, 보안 그룹, 인스턴스 역할을 사용합니다. On-Demand는 우선순위 1, Spot은 우선순위 2입니다.

#### 작업 정의 (Job Definition)

`_create_job_definitions()` (`scripts/deploy_infra.py:402–468`)에서 두 개의 ECS 작업 정의를 생성합니다:

1. **paper-review**: `scholar_lens.main` 진입점으로 논문 검토/요약을 실행합니다.
2. **tech-guide**: `scholar_lens.tech_guide_main` 진입점으로 기술 가이드를 생성합니다.

두 정의는 동일한 이미지 및 역할을 공유하며, 다음 사양을 가집니다:
- **CPU**: 2 vCPU
- **메모리**: 8192 MiB (8 GiB) — PDF 파싱(PyMuPDF)과 FAISS 코드 임베딩이 메모리를 많이 써서 1 GiB에선 대용량 PDF가 OOM(exit 137)으로 종료됨
- **재시도**: 2회
- **타임아웃**: 3시간
- **로그**: CloudWatch Logs, `/aws/batch/{project_name}-{stage}-{job_name}` 그룹, 1개월 보관 (`scripts/deploy_infra.py:470–508`)

작업 정의는 `Ref::` 매개변수(예: `Ref::source`, `Ref::repo_urls`, `Ref::mode`)를 통해 런타임 동안 선택적 인수를 받습니다 (`scripts/deploy_infra.py:430–446`).

#### 작업 큐 (Job Queue)

단일 `paper-review` 작업 큐가 생성되며, On-Demand 및 Spot 환경에 우선순위 가중치를 할당합니다 (`scripts/deploy_infra.py:510–548`).

### SSM 파라미터 저장소 및 비밀 관리

#### 플레인텍스트 파라미터
Non-secret 파라미터(작업 큐 이름, 작업 정의 이름)는 `_store_ssm_parameters()` (`scripts/deploy_infra.py:550–605`)에서 CloudFormation 스택으로 생성됩니다:

| 파라미터 | 값 |
|---------|---|
| `BATCH_JOB_QUEUE` | paper-review 작업 큐 이름 |
| `BATCH_JOB_DEFINITION` | paper-review 작업 정의 이름 |
| `GUIDE_JOB_QUEUE` | tech-guide 작업 큐 이름 |
| `GUIDE_JOB_DEFINITION` | tech-guide 작업 정의 이름 |

#### 비밀 파라미터 (아웃오브밴드 저장)
비밀 API 키(GitHub, LangChain, Upstage, Brave, Slack 봇 토큰)는 CloudFormation 템플릿에 평문으로 노출되지 않도록 **아웃오브밴드**로 저장됩니다 (`scripts/deploy_infra.py:62–94`):

`put_secure_secrets()` 함수는 배포 후 SSM `put_parameter` API를 호출하여 각 비밀을 암호화된 `SecureString`으로 저장합니다. `SECRET_SSM_PARAMS` 세트 (`scripts/deploy_infra.py:62–68`)에 정의된 비밀은 CDK 스택에서 제외됩니다 (`scripts/deploy_infra.py:587–605`). 이 패턴은 민감한 값이 git 이력이나 CloudFormation 출력에 노출되는 것을 방지합니다.

### Docker 이미지 빌드 및 배포

`DockerImageAsset` (`scripts/deploy_infra.py:411–418`)은 프로젝트 루트의 Dockerfile을 Linux AMD64 플랫폼용으로 빌드합니다. 빌드 캐시 제외 패턴은 `cdk.out`, `.venv`, `.git`, `**/__pycache__`입니다.

이미지는 AWS ECR의 CDK-managed 리포지토리에 푸시되며, 작업 정의에서 참조됩니다.

### 관찰성 및 경보

`_create_observability()` (`scripts/deploy_infra.py:607–686`)는 세 개의 모니터링 규칙을 설정합니다:

#### 1. Batch 작업 실패 → SNS
EventBridge 규칙 (`scripts/deploy_infra.py:613–637`):
- **소스**: `aws.batch`
- **이벤트 유형**: "Batch Job State Change"
- **필터**: `status: FAILED`인 작업, 이 스택의 job-queue
- **액션**: SNS 토픽에 메시지 발행: `"{jobName} (id {jobId}), 사유: {statusReason}"`

#### 2. 로그 에러 메트릭 필터 → 경보 → SNS
각 CloudWatch Logs 그룹에 대해 (`scripts/deploy_infra.py:640–665`):
- **메트릭 필터**: `FilterPattern.any_term("ERROR")`로 "ERROR" 로그 라인 감지
- **메트릭**: `{project_name}/{stage}/LogErrors` 네임스페이스, 메트릭값 1
- **경보**: 5분 윈도우에서 합계 ≥ 1이면 SNS 토픽 발행

#### 3. 예상 비용 보호
커스텀 메트릭 기반 경보 (`scripts/deploy_infra.py:670–685`):
- **네임스페이스**: `ScholarLens`
- **메트릭**: `EstimatedCostUSD` (애플리케이션에서 발행)
- **임계값**: $50 USD
- **윈도우**: 1시간
- **조건**: 1시간 합계 > $50이면 SNS 발행

모든 경보는 `SnsAction`을 통해 `_create_sns_topic()` (`scripts/deploy_infra.py:381–400`)에서 생성된 KMS 암호화 SNS 토픽으로 전달됩니다. 토픽은 이메일 구독(제공된 경우), 강제 SSL, 키 로테이션을 설정합니다.

### 배포 및 런타임 통합

`scholar_lens/src/aws_helpers.py`는 런타임 AWS 상호작용을 지원합니다:

- **`S3Handler`** (`aws_helpers.py:30–141`): 파일 다운로드/업로드, 디렉토리 업로드 (async 지원).
- **`get_ssm_param_value()`** (`aws_helpers.py:152–158`): 암호화된 SecureString 파라미터 읽기 (자동 복호화).
- **`submit_batch_job()`** (`aws_helpers.py:161–182`): Batch 작업 제출.
- **`wait_for_batch_job_completion()`** (`aws_helpers.py:185–218`): 작업 완료 대기 (기본 타임아웃: 3시간, 폴 간격: 30초).

### 배포 명령

```bash
python3 scripts/deploy_infra.py
```

배포는 `Config.load()` (`scholar_lens/configs/config.py:219–224`)에서 `config.yaml` 및 환경 변수를 읽습니다. 필수 환경 변수는 `EnvVars` (`scholar_lens/src/constants.py:20–43`)에 나열됩니다 (GitHubToken, LangChain API Key, Upstage API Key, Brave API Key, Slack Bot Token).

## 10. 발행 & 블로그 PR (Publishing)

생성된 마크다운 문서와 그림은 `Publisher` 클래스를 통해 일관되게 발행됩니다 (scholar_lens/src/publisher.py). 모든 아티팩트 타입 — 논문 리뷰, 요약, 기술 가이드 — 이 동일한 발행 파이프라인을 사용하므로 저장, S3 업로드, 블로그 PR 생성이 표준화됩니다.

### 발행 요청 구성

각 아티팩트는 발행 전에 `PublishRequest` 데이터 클래스로 구성됩니다 (scholar_lens/src/publisher.py:40-52):

- **title**: 문서의 제목. 발행 파일명과 S3 경로 생성 시 사용됩니다.
- **markdown**: 발행할 마크다운 콘텐츠. 수학 정규화, 이미지 경로 재작성, 메타데이터 추가 처리를 거칩니다.
- **work_dir**: 마크다운과 그림이 저장될 작업 디렉터리 (보통 논문별 papers/{paper_id}/ 또는 임시 경로).
- **branch_id**: Git 브랜치 이름의 일부. arXiv ID 또는 가이드 슬러그로 사용됩니다.
- **pr_title, pr_body, commit_message**: GitHub PR 메타데이터.
- **rewrite_local_images** (기본값 true): 로컬 이미지 경로를 블로그 자산 경로로 재작성할지 여부. PDF 파싱 아티팩트는 true, 기술 가이드는 false입니다 (scholar_lens/main.py:697, tech_guide_main.py:274).
- **figures_dirname** (기본값 "figures"): 그림 디렉터리 이름.
- **extra_metadata**: YAML 프론트매터에 추가될 메타데이터 사전.

### 마크다운 정규화 및 저장

발행 시 다음 변환이 순차적으로 적용됩니다 (scholar_lens/src/publisher.py:79-90):

1. **수학 언더스코어 이스케이핑**: `normalize_math_underscores(markdown)` 호출 (scholar_lens/src/markdown_math.py:37-50). 이 함수는 LaTeX 수식 `$...$`와 `$$...$$` 내의 언더스코어를 `\_`로 이스케이프합니다. 이는 kramdown(GFM) 파서가 수식 렌더링 전에 언더스코어를 강조 구분자로 처리하는 것을 방지합니다. 코드 블록 (```...```) 과 인라인 코드 (`` `...` ``) 내 언더스코어는 보존됩니다 (scholar_lens/src/markdown_math.py:21-29의 정규식 순서 중요).

2. **로컬 이미지 경로 재작성** (request.rewrite_local_images가 true일 경우):
   - `_rewrite_local_images(markdown, file_name)` (scholar_lens/src/publisher.py:95-104)
   - 마크다운 이미지 패턴 `![alt](path)` 에서 상대 경로 (http(s):// 또는 /로 시작하지 않음) 는 블로그 자산 레이아웃 경로로 변환됩니다.
   - 대상 경로: `/{S3Paths.ASSETS.value}/{file_name}/{image_filename}` 예: `/assets/2025-01-15-transformer-review/fig1.png`

3. **로컬 파일 저장**: 마크다운을 `{work_dir}/{YYYY-MM-DD}-{slugified_title}.md` 형식으로 저장합니다 (scholar_lens/src/publisher.py:81, 89-90). 슬러그화는 제목의 비알파뉴메릭 문자를 하이픈으로 대체하고 120자 제한을 적용합니다 (scholar_lens/src/publisher.py:55-58의 `slugify` 함수).

### S3 업로드

S3 핸들러가 구성되었을 경우 (scholar_lens/src/publisher.py:106-125):

- **마크다운 포스트**: `posts/` 접두사 아래 업로드.
- **그림 자산**: 로컬 figures_dir이 존재하면, 구성된 파일 확장자 (`.gif, .jpg, .jpeg, .png, .svg, .webp`) 만 필터링해 `assets/{file_name}/` 아래 업로드.
- **S3 반환 경로**: `s3://{bucket}/{posts_prefix}/{file_name}.md` 형식.
- **공개 가독성**: `github_config.public_assets` 설정으로 제어 (configs/config.py:45, publisher.py:123). 기본값은 false (프라이빗). GitHub Pages 자산 서빙 모델에서는 S3 공개 ACL이 불필요하고 퍼블릭 액세스 차단 정책이 이를 거부할 수 있으므로, S3에서 직접 서빙하는 경우에만 true로 설정합니다.

### Git 작업 및 PR 생성

PR이 활성화되었을 경우 (scholar_lens/main.py:210):

1. **저장소 복제** (scholar_lens/src/publisher.py:206-260의 `_git_operations_inner`):
   - 토큰을 포함하는 HTTPS URL로 임시 clone_dir에 저장소 복제: `https://oauth2:{token}@github.com/{repo_name}.git`
   - 클론 디렉터리 기본값: `github_clone/` (LocalPaths.GITHUB_CLONE_DIR, scholar_lens/src/constants.py)

2. **브랜치 생성**:
   - 브랜치 이름: `{branch_prefix}/{branch_id}-{timestamp}` 예: `paper-reviews/2401-06066-20250115-143022`
   - branch_prefix 기본값: `"paper-reviews"` (configs/config.py:24, tech guide는 `"tech-guides"`)
   - 기존 브랜치가 있으면 재사용, 없으면 base_branch에서 새로 생성 (scholar_lens/src/publisher.py:221-227)

3. **파일 배치**:
   - 마크다운: `_posts/` 디렉터리에 복사 (scholar_lens/src/publisher.py:229-231).
   - 그림: `assets/{file_name}/` 디렉터리에 복사 (scholar_lens/src/publisher.py:234-241).

4. **커밋 및 푸시**:
   - 변경 사항이 있을 경우에만 진행 (scholar_lens/src/publisher.py:243-245).
   - 작성자 정보: `{author_name} <{author_email}>` (기본값 "Scholar Lens Bot", configs/config.py:25-26)
   - 강제 푸시 사용 (scholar_lens/src/publisher.py:258).
   - 토큰 누출 방지: 예외 메시지에 GitPython이 포함한 토큰 URL을 `***`로 필터링 (scholar_lens/src/publisher.py:199-204의 `_redact_token`).

5. **PR 생성 실패 처리**:
   - 이미 같은 브랜치의 PR이 존재하면 경고 로깅, 다른 예외는 에러 (scholar_lens/src/publisher.py:156-171).
   - 최종적으로 클론 디렉터리 정리 (scholar_lens/src/publisher.py:175-177).

### Jekyll 프론트매터 생성

Jekyll 블로그 호환성을 위해 각 아티팩트는 YAML 프론트매터를 포함합니다 (scholar_lens/main.py:541-589의 `_build_front_matter`):

```yaml
---
layout: post
title: "{title}"
date: {now}
paper_date: {paper_published_date}
author: "{author}"
categories: [{primary_category}, "{paper_category}"]
tags: [{keywords}]
cover: /assets/images/{cover_image}
use_math: true
---
```

- **categories**: 
  - 첫 번째 (primary_category): 아티팩트 타입별 설정값 (configs/config.py:38-40):
    - 리뷰: `review_category` (기본값 "Paper Reviews")
    - 요약: `summary_category` (기본값 "Paper Summaries")
    - 기술 가이드: `tech_guide_category` (기본값 "Tech Guides")
  - 두 번째: 논문의 추출된 arXiv 카테고리 (하이픈 표준화, HTML 엔티티 언이스케이프 후).
  
- **태그**: 논문에서 추출한 키워드. 각 공백을 하이픈으로 변환하고 따옴표로 감쌈 (scholar_lens/main.py:564-566).

- **커버 이미지**: 논문 카테고리에 따라 매핑 (scholar_lens/main.py:568의 `github_config.cover_image_for(category)`). configs/config.py:27-60의 Github 설정에서:
  - `cover_images`: 카테고리 슬러그 → 이미지 파일명 매핑 (예: `"multimodal-learning": "ml.jpg"`)
  - `default_cover_image` (기본값 "default.jpg"): 매핑되지 않은 카테고리의 폴백.
  - 슬러그화: `re.sub(r"[^a-z0-9]+", "-", category.lower()).strip("-")` (configs/config.py:59).

- **작성자**: 논문의 실제 저자 목록을 렌더링 (scholar_lens/main.py:610-623의 `_format_authors`):
  - 1명: "A"
  - 2명: "A and B"  
  - 3명 이상: "A et al."
  - 플레이스홀더 ("Unknown") 필터링. PDF URL 소스에서 저자가 없을 경우 추출된 affiliation으로 폴백 (scholar_lens/main.py:575-577).

### 메타데이터 백필

비-arXiv 소스 (임의 PDF URL) 는 플레이스홀더 메타데이터를 반환합니다 (예: 제목 "Pdf", 저자 ["Unknown"]). `_backfill_metadata` 함수는 콘텐츠 추출기가 문서 텍스트에서 복구한 속성으로 이들을 대체합니다 (scholar_lens/main.py:592-607):

- 제목이 placeholder이거나 비어있으면 추출된 title로 대체.
- 저자 목록에 유효한 저자가 없으면 추출된 authors로 대체.
- `is_placeholder(a)` 유틸리티는 "Unknown" 같은 고정된 스트링을 판별합니다.

### 문서 형식화

아티팩트별로 프론트매터 다음에 특정 본문 구조가 추가됩니다:

- **리뷰** (scholar_lens/main.py:626-636의 `_format_explanation`):
  ```
  {front_matter}
  ### TL;DR
  {key_takeaways}
  - - -
  {explanation}
  - - -
  ### References
  * [{title_escaped}]({pdf_url})
  ```

- **요약** (scholar_lens/main.py:645-657의 `_format_summary`):
  ```
  {front_matter}
  {summary_body}
  - - -
  ### References
  * [{title_escaped}]({pdf_url})
  {urls_section}
  ```

마크다운 링크 텍스트 중 대괄호 (`[`, `]`) 는 Markdown 문법 충돌을 방지하기 위해 이스케이프됩니다 (scholar_lens/main.py:639-642의 `_md_link_text`).

### 활성화 및 설정

- **S3 업로드**: `config.resources.s3_bucket_name`이 설정되었을 때만 활성화됩니다.
- **GitHub PR**: `config.resources.github.enabled`가 true이고 `config.resources.github.repo_name`이 설정되어야 합니다. GitHub 토큰은 `GITHUB_TOKEN` 환경 변수에서 로드됩니다 (scholar_lens/src/publisher.py:134-139).
- **카테고리 레이블 및 커버 이미지**: 완전히 설정 가능 (configs/config.py의 Github 클래스). 블로그 구조 또는 카테고리 페이지 이름과 일치하도록 커스터마이징할 수 있습니다.

## 11. Slack 봇 (Paper Bot)

Scholar-Lens의 Slack 봇은 Paper Bot이라 불리며, 사용자가 Slack 채널이나 다이렉트 메시지에서 멘션하거나 메시지를 보내면 온디맨드로 작동합니다(`scholar_lens/slack/bot.py`). 봇의 핵심 역할은 자유형식의 채팅 메시지를 구조화된 액션(review, summarize, guide)으로 변환하고, 대응하는 AWS Batch 작업을 비동기로 디스패치하는 것입니다. 파이프라인 자체는 Batch에서 실행되며, 봇은 빠른 응답과 이벤트 라우팅만 담당합니다.

### 아키텍처 및 흐름

**온디맨드 모델**: 사용자의 Slack 멘션이나 DM이 Socket Mode 핸들러에 도착하면(`run_socket_mode()`, bot.py:221~277), 다음 단계를 거칩니다.

1. **이벤트 중복제거** (`_SeenEvents`, bot.py:44~68): Slack은 최소 한 번 이상 같은 이벤트를 재전송할 수 있으므로, 동일한 `client_msg_id` 또는 `event_ts`를 추적하는 512개 크기의 bounded 집합(스레드 안전 락 가드)으로 중복 Batch 작업 생성을 방지합니다.

2. **인증 확인** (`is_user_authorized()`, bot.py:31~41): 환경변수 `SLACK_ALLOWED_USER_IDS`(쉼표로 구분된 사용자 ID 목록)로 허용 목록을 관리합니다. 미설정 시 모든 워크스페이스 멤버가 작업을 트리거할 수 있으므로 (오픈 모드), 설정되면 목록의 사용자만 승인되고 나머지는 거부됩니다.

3. **의도 파싱** (`IntentParser.parse()`, intent.py:57~60): Claude Haiku 4.5 (`CLAUDE_V4_5_HAIKU`, bot.py:28)로 메시지를 `SlackIntentPrompt`(`prompts.py:2914~2958`)를 통해 분류합니다. 프롬프트는 사용자의 **명시적 요청**(동사: "review", "summarize", "guide", "리뷰", "요약", "가이드")을 기반으로 의도를 결정하며, 모호한 경우(예: bare arXiv ID) "unknown"으로 분류하여 봇이 사용자에게 명확히 물을 수 있게 합니다. 파싱 결과는 `ParsedIntent` 객체로 반환되며, intent, sources(arXiv ID 또는 URL), repo_urls(GitHub 링크), reason(분류 설명)을 포함합니다.

4. **작업 디스패치** (`JobDispatcher.dispatch()`, dispatcher.py:64~83): 파싱된 의도가 actionable하면(intent가 UNKNOWN이 아니고 sources가 있으면), 대응하는 Batch 작업을 제출합니다. Review/Summarize는 review job definition을 사용하고 Slack 컨텍스트를 파라미터로 전달합니다. Guide는 별도의 guide job definition(`guide_job_definition`, dispatcher.py:62)을 필수로 요구하며, 미설정 시 에러를 발생시킵니다.

5. **결과 통지**: Batch 작업이 완료되면, 파이프라인은 Slack 컨텍스트(channel, thread_ts)를 통해 `post_slack_result()`(`notifier.py:43~92`)를 호출하여 스레드에 결과(성공/실패, 아티팩트 레이블, S3 URL)를 게시합니다. 봇이 사용자에게 "I'll post the result here when it's ready"라고 약속한 것을 이행합니다.

### 앱 ID 보안 가드 (App-ID Guard)

**문제**: Paper Bot은 전용 Slack 앱에서 실행하는 것이 좋습니다. Socket Mode 핸들러가 같은 토큰으로 여러 봇(예: OmniSummary)을 실행하면 Slack이 이벤트를 무작위로 한 프로세스에만 전달하므로 멘션이 잘못된 봇으로 라우팅될 수 있기 때문입니다.

**솔루션** (`verify_slack_app_identity()`, bot.py:185~218): 환경변수 `SLACK_EXPECTED_APP_ID`를 설정하면 앱 토큰의 형식 `xapp-1-<APP_ID>-...`에서 APP_ID를 추출하여(`_app_id_from_app_token()`, bot.py:172~182) Batch 작업 제출 전에 검증합니다. 토큰의 APP_ID가 기대값과 일치하지 않으면 `SlackAppMismatchError`를 발생시켜 시작을 거부합니다(bot.py:168). 미설정 시 검증을 건너뛰고 경고만 로깅합니다.

### 사용자 승인 (Authorization Allowlist)

`SLACK_ALLOWED_USER_IDS` 환경변수(쉼표로 구분된 Slack 사용자 ID: "U123, U456")로 작업 트리거 권한을 제어합니다. 미설정 시 모든 사용자가 허용되고, 설정되면 명시된 사용자만 가능하며 나머지는 `:lock: Sorry, you're not authorized to run Paper Bot jobs.` 메시지로 거부됩니다(bot.py:248~254). 허용 목록이 없으면 권한 부여는 이루어지지 않으므로 예산 관리가 필요한 환경에서는 반드시 설정해야 합니다.

### 이벤트 중복제거 (_SeenEvents)

`_SeenEvents` 클래스(`bot.py:44~68`)는 Slack의 최소 한 번 이상(at-least-once) 전달 정책에 대비하여 bounded 집합으로 구현됩니다. 용량 512개(기본값, `capacity` 파라미터로 조정 가능)의 dict(순서 유지)를 스레드 락으로 보호하며, 새로운 이벤트는 `false`를 반환하고 중복은 `true`를 반환합니다. 용량 초과 시 가장 오래된 항목(dict 삽입 순서로 첫 번째)을 제거하여 메모리를 유지합니다. Slack의 `client_msg_id`(메시지 고유 ID) 또는 `event_ts`(타임스탬프)를 dedup 키로 사용합니다(bot.py:241).

### Socket Mode 및 전용 앱 권장 사항

Paper Bot은 **Socket Mode** (`SocketModeHandler`, bot.py:276)에서만 작동하며, Slack의 실시간 이벤트 수신을 위해 다음 토큰이 필요합니다.

- **SLACK_BOT_TOKEN**: 봇 계정에 부여된 xoxb- 토큰 (메시지 송수신 권한)
- **SLACK_APP_TOKEN**: xapp-1- 앱 레벨 토큰 (Socket Mode 연결용)
- **전용 Slack 앱**: 다른 Socket Mode 봇과 토큰을 공유하지 않는 것이 좋습니다 (하나의 앱은 하나의 Socket Mode 연결만 지원).

이벤트 핸들러 (`_on_mention()`, `_on_message()`, bot.py:266~273)는 app_mention(멘션) 및 message(DM, `channel_type=="im"` + `bot_id` 없음)를 처리합니다. Batch 작업이 성공해도 실패해도 Slack 컨텍스트가 없으면 `post_slack_result()`는 no-op이므로, Batch 매개변수로 channel/thread_ts가 전달되어야 결과가 스레드에 게시됩니다(dispatcher.py:86~95, 미설정 시 NULL_STRING "null").

### 스택 및 구성

- **IntentParser** (`intent.py:46~71`): Claude Haiku 4.5 + BedrockLanguageModelFactory로 구성된 LangChain chain (SlackIntentPrompt → 모델 → HTMLTagOutputParser)
- **JobDispatcher** (`dispatcher.py:38~155`): review/summarize는 review_job_definition 사용, guide는 guide_job_definition(필수) 사용하여 Batch job 제출
- **PaperBot** (`bot.py:71~117`): handle_message()에서 intent 파싱과 dispatcher를 조합하여 사용자 응답 생성
- **Notifier** (`notifier.py`): Batch 파이프라인에서 호출되며, Slack mrkdwn 제어문자 이스케이프(`_mrkdwn_safe()`, notifier.py:30~40) 후 WebClient로 게시

## 12. 테스트, CI & 확장 (Testing, CI & Extending)

### 테스트 설계

Scholar-Lens는 완전히 격리된 테스트 환경을 유지하며, 실제 AWS 또는 네트워크 호출을 하지 않습니다. `tests/conftest.py`는 모든 테스트에 자동으로 적용되는 전역 고정장치(fixture)를 제공합니다.

**moto와 responses를 이용한 모킹:**
- **AWS 모킹**: `moto`는 boto3 클라이언트를 가로채서 S3, SNS, SSM, Batch, STS 호출을 메모리 내 가짜 서비스로 리다이렉트합니다. 예를 들어 `test_aws_helpers.py`의 `s3_bucket` 픽스처는 `with mock_aws():` 컨텍스트 내에서 가짜 S3 버킷을 생성합니다.
- **HTTP 모킹**: `responses` 라이브러리는 `requests` 라이브러리 호출을 가로챕니다. `test_paper_source.py`에서 PDF 다운로드를 테스트할 때 `@responses.activate` 데코레이터로 arXiv와 Crossref URL 응답을 모킹합니다.

**자동 소켓 가드 (autouse fixture):**
`conftest.py`의 `_block_network` 픽스처는 127.0.0.1/localhost를 제외한 모든 실제 네트워크 연결을 차단합니다. moto와 responses로 모킹되지 않은 누수 시도는 RuntimeError를 발생시키므로 테스트가 결정론적이고 비용이 0입니다 (`tests/conftest.py:36-57`).

**환경 격리:**
모든 테스트는 가짜 AWS 자격증명(`AWS_ACCESS_KEY_ID=testing`, `AWS_DEFAULT_REGION=us-east-1`)으로 실행되어 실제 프로덕션 인증정보가 로드되지 않습니다 (`tests/conftest.py:16-21`).

**주요 테스트 커버리지:**
- `test_arxiv_handler.py`: ArxivMetadata Pydantic 모델, DOI 합성, 추상문 정규화
- `test_paper_source.py`: PaperSource 추상 인터페이스, ArxivSource, PdfUrlSource (SSRF 가드 포함)
- `test_factories_config.py`: BedrockLanguageModelFactory 오프라인 로직, 모델 레지스트리, Opus 4.8 1M 컨텍스트 윈도우 베타 주입
- `test_models_registry.py`: LanguageModelId/EmbeddingModelId 열거형과 메타데이터 일관성
- `test_slack_bot.py`: PaperBot 의도 파싱 및 디스패처 (모킹된 AsyncMock으로)
- `test_citation_redesign.py`: citation_metadata rate_limiter, Crossref 타입 필터, title-match 게이트
- `test_tech_guide.py`: TechGuideGenerator 웹 검색 및 LLM 체인 (moto 및 responses 모킹)

### CI 파이프라인

`.github/workflows/ci.yml`은 모든 푸시 및 PR에 대해 다음 작업을 실행합니다:

**Lint & Format 작업 (Python 3.12):**
```
poetry run ruff check scholar_lens scripts tests    # E, F, I, W, B, UP 규칙
poetry run black --check scholar_lens scripts tests # 코드 포맷 검증
poetry run mypy scholar_lens                        # 타입 체크
```
`pyproject.toml`에서 `ruff`와 `black`는 88자 라인 길이로 설정되고, mypy는 암묵적 Optional을 거부하며 일부 프레임워크 노이즈(call-overload, misc)를 필터링합니다 (`pyproject.toml:70-107`).

**테스트 작업 (Python 3.12 & 3.13):**
```bash
poetry run pytest tests/
  -p no:cacheprovider
  -W ignore::DeprecationWarning
  --cov=scholar_lens
  --cov-report=term-missing
  --cov-report=xml
```
pytest는 `asyncio_mode="auto"`로 설정되어 비동기 테스트를 자동으로 처리합니다 (`pyproject.toml:54-59`). 커버리지는 `scholar_lens` 패키지에서만 수집되고, 테스트와 프롬프트 파일은 제외됩니다.

**CDK 합성 검증 (Python 3.12, Node 20):**
`scripts/ci_synth_check.py`는 실제 AWS 자격증명 없이 가짜 계정/리전에 대해 PaperReviewStack을 합성합니다 (`scripts/ci_synth_check.py:22-50`). 이는 구성 오류를 조기에 감지하고 IAM/보안 배선을 검증합니다.

### 시스템 확장

#### 새로운 모델 추가

1. **열거형 추가:**
   `scholar_lens/src/constants.py`의 `LanguageModelId` 또는 `EmbeddingModelId` 열거형에 새로운 모델을 추가합니다:
   ```python
   class LanguageModelId(str, Enum):
       CLAUDE_V4_8_OPUS = "anthropic.claude-opus-4-8"
       CLAUDE_NEW_MODEL = "anthropic.claude-new-model"  # 추가
   ```

2. **메타데이터 등록:**
   `scholar_lens/src/utils/models.py`의 `_LANGUAGE_MODEL_INFO` 딕셔너리에 모델 기능을 추가합니다:
   ```python
   _LANGUAGE_MODEL_INFO: dict[LanguageModelId, LanguageModelInfo] = {
       LanguageModelId.CLAUDE_V4_8_OPUS: LanguageModelInfo(
           context_window_size=200000,
           max_output_tokens=64000,
           supports_thinking=True,
           uses_adaptive_thinking=True,
           supports_1m_context_window=True,
       ),
       LanguageModelId.CLAUDE_NEW_MODEL: LanguageModelInfo(
           context_window_size=200000,
           max_output_tokens=64000,
           supports_thinking=True,
           uses_adaptive_thinking=True,
       ),  # 추가
   }
   ```

3. **테스트 검증:**
   `tests/test_models_registry.py`는 모든 열거형 멤버가 레지스트리 항목을 가지고 있는지 확인합니다. 새 모델은 이 테스트를 자동으로 통과합니다.

#### 새로운 논문 소스 타입 추가

`PaperSource` 추상 클래스(`scholar_lens/src/paper_source.py:61-94`)는 확장점입니다.

1. **구체적인 구현 작성:**
   ```python
   class OpenReviewSource(PaperSource):
       def __init__(self, openreview_id: str):
           self._id = openreview_id
           
       @property
       def source_id(self) -> str:
           return f"openreview_{self._id}"
       
       @property
       def pdf_url(self) -> HttpUrl:
           return HttpUrl(f"https://openreview.net/pdf?id={self._id}")
       
       def fetch_metadata(self) -> ArxivMetadata:
           # OpenReview API 또는 스크래핑으로 메타데이터 가져오기
           ...
       
       def download_pdf(self, papers_dir: Path) -> Path:
           # PDF 다운로드 로직
           ...
   ```

2. **`resolve_paper_source` 함수 확장:**
   `scholar_lens/src/paper_source.py`의 `resolve_paper_source` 함수에 타입 감지 로직을 추가합니다:
   ```python
   def resolve_paper_source(source: str) -> PaperSource:
       if is_arxiv_id(source.strip()):
           return ArxivSource(source.strip())
       if source.startswith("https://arxiv.org"):
           arxiv_id = extract_arxiv_id_from_url(source)
           return ArxivSource(arxiv_id)
       if source.startswith("https://openreview.net"):
           openreview_id = extract_openreview_id_from_url(source)
           return OpenReviewSource(openreview_id)
       # PDF URL 폴백
       return PdfUrlSource(source)
   ```

3. **테스트 작성:**
   `test_paper_source.py`와 같이 `@responses.activate`로 HTTP 호출을 모킹하고, `PaperSource` 계약(source_id, pdf_url, fetch_metadata, download_pdf)을 테스트합니다.

#### 새로운 에이전트 추가

Scholar-Lens는 LangGraph를 사용하여 `ExplainerGraph`라는 상태 머신 에이전트를 구현합니다 (`scholar_lens/src/explainer.py:113-150`).

1. **새로운 LLM 단계 추가:**
   ExplainerState에 새로운 필드를 추가하고, 그래프에 노드를 추가합니다:
   ```python
   class ExplainerState(TypedDict):
       # 기존 필드...
       code_review: str | None  # 새 필드
   
   class ExplainerGraph(RetryableBase):
       def _build_graph(self) -> CompiledStateGraph:
           graph = StateGraph(ExplainerState)
           # 기존 노드...
           graph.add_node("code_review_node", self._review_code)
           graph.add_edge("code_node", "code_review_node")
           # 엣지 연결...
           return graph.compile(...)
       
       async def _review_code(self, state: ExplainerState) -> dict[str, Any]:
           # 코드 리뷰 LLM 로직
           prompt = ReviewCodePrompt.get_prompt()
           chain = prompt | self.code_review_model
           return {"code_review": await chain.ainvoke(...)}
   ```

2. **모델 팩토리 사용:**
   BedrockLanguageModelFactory를 통해 모델 인스턴스를 생성합니다:
   ```python
   factory = BedrockLanguageModelFactory(boto_session, region_name)
   self.code_review_model = factory.get_model(
       LanguageModelId.CLAUDE_V4_8_OPUS,
       max_tokens=2000,
       enable_thinking=False,
   )
   ```

3. **프롬프트 정의:**
   `scholar_lens/src/prompts/prompts.py`에 새로운 BasePrompt 서브클래스를 추가하고 입출력 변수를 정의합니다:
   ```python
   class ReviewCodePrompt(BasePrompt):
       input_variables = ["code", "language"]
       output_variables = ["review"]
       human_prompt_template = "...code review instructions..."
   ```

4. **테스트:**
   `test_factories_config.py` 스타일로 가짜 boto3 세션으로 팩토리를 테스트하고, `AsyncMock`으로 체인 출력을 스텁합니다:
   ```python
   def test_code_review_node():
       factory = _make_factory()
       model = factory.get_model(LanguageModelId.CLAUDE_V4_8_OPUS)
       chain = MagicMock()
       chain.ainvoke = AsyncMock(return_value={"review": "Good code"})
       state = {...}
       result = await explainer._review_code(state)
       assert result["code_review"] == "Good code"
   ```

### CI에서 테스트 실행

로컬에서 테스트를 실행하려면:
```bash
poetry install --with dev
poetry run pytest tests/ -v --cov=scholar_lens
poetry run ruff check scholar_lens scripts tests
poetry run black --check scholar_lens scripts tests
poetry run mypy scholar_lens
```

모든 검사가 통과하면 CI 파이프라인도 통과합니다. CDK 합성 검증은 별도로 실행하려면:
```bash
export CDK_DEFAULT_ACCOUNT=123456789012
export CDK_DEFAULT_REGION=ap-northeast-2
poetry run python scripts/ci_synth_check.py
```
