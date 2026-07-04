<div align="center">

# 👩‍🏫 Scholar-Lens

**AI/ML 논문과 기술 문서를 발행 가능한 콘텐츠로 — 논문 리뷰, 요약, 기술 가이드.**

CLI · AWS Batch · Slack 봇으로 구동 · Amazon Bedrock (Claude) 기반.

[![CI](https://github.com/bits-bytes-nn/scholar-lens/actions/workflows/ci.yml/badge.svg)](https://github.com/bits-bytes-nn/scholar-lens/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![AWS CDK](https://img.shields.io/badge/IaC-AWS%20CDK-orange)
![Bedrock](https://img.shields.io/badge/LLM-Amazon%20Bedrock%20(Claude)-green)

🇺🇸 [English README](./README.md)

![Sample](./assets/sample.png)

</div>

---

## 무엇을 하나요

Scholar-Lens는 논문 또는 문서 URL 묶음으로부터 세 가지 산출물을 생성합니다.

| 산출물 | 입력 | 출력 |
| --- | --- | --- |
| 📝 **리뷰** | 논문 1개 (arXiv ID 또는 PDF URL) | 멀티 스테이지 LangGraph 워크플로우로 섹션별 심층 해설 (직관 → 형식화 → 코드) |
| 📄 **요약** | 논문 1개 (arXiv ID 또는 PDF URL) | 다섯 섹션 간결 요약 (동기 · 해법 · 구현 · 결과 · 의의) |
| 📚 **기술 가이드** | 문서 URL 1개 이상 | 출처에 근거한 실습형 기술 가이드 / 튜토리얼 |

모든 산출물은 블로그에 바로 올릴 수 있는 깔끔한 Markdown이며, S3에 업로드하고 Jekyll 블로그에 PR로 발행할 수 있습니다.

## 핵심 특징

- **모든 AI/ML 논문** — arXiv ID *또는* 임의의 논문 **PDF URL**을 받습니다(PDF가 아닌 URL은 거부). arXiv 전용이 아닙니다.
- **환각 없이 근거 기반** — 인용은 Crossref / Semantic Scholar로 해석하고 제목 유사도 게이트를 거칩니다. 작성기는 소스에 등장한 URL만 링크할 수 있고, 기술 가이드는 섹션별 사실 검증 패스를 돕니다.
- **최신 Claude** — Amazon Bedrock의 Claude Opus 4.8 / Sonnet 5 / Haiku 4.5 + adaptive thinking, 단계별 설정 가능.
- **코드 인식 리뷰** — GitHub 저장소 분석 + FAISS 시맨틱 검색(선택).
- **온디맨드 Slack 봇** — 논문이나 문서 URL과 함께 멘션하면, LLM이 *명시적* 요청을 파싱하고 Batch 작업을 실행한 뒤 스레드에 결과 링크를 회신합니다. AWS에 서버리스(Slack Events API → Lambda)로 호스팅되어 로컬에서 상주 프로세스를 띄울 필요 없이 항상 동작합니다.
- **확장성과 관찰성** — AWS Batch(온디맨드 + 스팟), CloudWatch 알람, EventBridge→SNS 실패 알림, 암호화 SecureString 시크릿.

## 아키텍처

### 생성 파이프라인

세 산출물은 공통 프런트엔드(해석 → 파싱 → 추출)를 공유한 뒤 분기합니다.

![생성 파이프라인](./docs/diagrams/pipeline.png)

### AWS 인프라

모든 작업은 AWS Batch에서 컨테이너 잡으로 실행되며, 동일한 잡을 CLI · Slack 봇 · `run_batch.py`가 제출합니다.

![AWS 아키텍처](./docs/diagrams/aws-architecture.png)

### 핵심 컴포넌트

| 모듈 | 역할 |
| --- | --- |
| `paper_source.py` | arXiv ID 또는 논문 PDF URL을 `PaperSource`로 해석; URL 소스가 PDF를 제공하는지 검증(SSRF 가드 포함). |
| `parser.py` · `content_extractor.py` | HTML/PDF 파싱 + 그림 추출; 인용 / 속성 / 목차 구조화 추출. |
| `citation_metadata.py` · `citation_summarizer.py` | 추상 우선 참고문헌 해석(Crossref / Semantic Scholar), 속도 제한 + 제목 게이트. |
| `explainer.py` | `ExplainerGraph` — 멀티 스테이지 LangGraph 리뷰 워크플로우. |
| `summarizer.py` | `PaperSummarizer` — 다섯 섹션 요약. |
| `tech_guide.py` · `web_research.py` | 문서 URL 조사(+ 하위 페이지 + 웹 검색) 후 근거 기반 가이드 작성. |
| `code_retriever.py` | 저장소 클론 + 시맨틱 코드 검색(FAISS). |
| `publisher.py` | 산출물 무관 S3 업로드 + Jekyll 블로그 PR. |
| `slack/` | 온디맨드 Paper Bot(의도 파싱 → Batch 디스패치). |

> 📖 데이터 흐름, 모듈 레퍼런스, 프롬프트, 인용 파이프라인, 인프라, Slack Lambda 아키텍처 등 내부 구조 심층 설명은 **[설계 문서](./docs/design.md)**를 참고하세요.

## 기술 스택

Python 3.12+ · AWS CDK · Docker · Amazon Bedrock · LangChain / LangGraph · PyMuPDF · BeautifulSoup4 · FAISS · Pydantic.

## 설정

템플릿을 복사해 편집합니다.

```bash
cp scholar_lens/configs/config-template.yaml scholar_lens/configs/config.yaml
```

```yaml
resources:
  project_name: scholar-lens
  stage: dev
  profile_name: your-profile
  default_region_name: ap-northeast-2
  bedrock_region_name: us-west-2
  s3_bucket_name: your-bucket
  github:
    enabled: true
    repo_name: owner/owner.github.io
    review_category: Paper Reviews     # 블로그 카테고리 탭 라벨(설정 가능)
    summary_category: Paper Summaries
    tech_guide_category: Tech Guides

explanation:
  paper_synthesis_model_id: anthropic.claude-opus-4-8   # 최신 Claude Opus
summary:
  summary_model_id: anthropic.claude-opus-4-8
tech_guide:
  writing_model_id: anthropic.claude-opus-4-8
  verify_grounding: true               # 섹션별 사실 검증 패스

output_language: Korean
```

전체 스키마는 [`scholar_lens/configs/config-template.yaml`](./scholar_lens/configs/config-template.yaml)를 참고하세요.

## 사용법

### 준비

```bash
poetry install --with dev          # 의존성 + 개발/테스트 도구
cp .env.template .env               # 토큰 채우기(.env.template 참고)
python scripts/deploy_infra.py      # AWS 인프라 배포(CDK)
```

### 산출물 생성 (CLI)

```bash
# 리뷰 — --source 는 arXiv ID 또는 논문 PDF URL
python scholar_lens/main.py --source 2312.11805 --parse-pdf True
python scholar_lens/main.py --source https://openreview.net/pdf?id=XXXX

# 요약(다섯 섹션 간결 형식)
python scholar_lens/main.py --source 2312.11805 --mode summarize

# 문서 URL로부터 기술 가이드
python scholar_lens/tech_guide_main.py \
  --urls https://docs.framework.io/start https://docs.framework.io/api \
  --search-queries "framework.io best practices"

# AWS Batch 잡으로 제출
python scripts/run_batch.py --source 2505.09388 --mode review \
  --repo-urls https://github.com/org/repo
```

### Slack 봇

봇은 **AWS에 서버리스로** 실행됩니다 — `python scripts/deploy_infra.py` + `cdk deploy`가
API Gateway HTTP API 뒤의 리시버 Lambda와 워커 Lambda를 프로비저닝합니다. Slack 앱의
**Event Subscriptions** Request URL을 리시버 주소(CDK 출력 `SlackEventsRequestUrl`)로 지정하고,
`app_mention` + `message.im`을 구독한 뒤 `SLACK_BOT_TOKEN` / `SLACK_SIGNING_SECRET`을 설정하면 됩니다.
로컬에 상주 프로세스를 띄울 필요가 없습니다.

원하는 작업과 함께 봇을 멘션하세요 — 예: `review 2401.06066`,
`summarize https://arxiv.org/pdf/2401.06066`, `guide https://docs.framework.io/start`.
결과는 S3에 저장되고 블로그에 발행되며, 봇은 스레드에 상태와 링크를 회신합니다.

```bash
# 로컬 개발용 폴백: Events API 대신 Socket Mode로 봇 실행
# (SLACK_BOT_TOKEN / SLACK_APP_TOKEN 필요, 앱에서 Socket Mode 활성화)
python -m scholar_lens.slack.bot
```

> 💡 봇은 전용 Slack 앱에서 실행하여 같은 워크스페이스의 다른 봇과 토큰을 공유하지 않도록 하세요.

## 테스트 & 품질

```bash
poetry run pytest tests/ --cov=scholar_lens --cov-report=term-missing  # 모든 AWS/네트워크 모킹
poetry run ruff check scholar_lens scripts tests
poetry run black --check scholar_lens scripts tests
poetry run mypy scholar_lens
poetry run python scripts/ci_synth_check.py   # AWS 계정 없이 CDK synth 검증
```

CI(`.github/workflows/ci.yml`)는 푸시·PR마다 lint/format, pytest 매트릭스(Python 3.12 & 3.13),
CDK synth 체크를 실행합니다.
