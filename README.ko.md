<div align="center">

# Scholar-Lens

**AI/ML 논문과 기술 문서를 블로그에 바로 올릴 수 있는 글로 바꿉니다. 심층 리뷰, 간결한 요약, 실습형 기술 가이드를 만듭니다.**

로컬이나 AWS Batch에서 실행하며, 사람과 에이전트(OpenClaw, Claude Code, Codex)가 같은 CLI를 씁니다. Amazon Bedrock의 Claude로 동작합니다.

[![CI](https://github.com/bits-bytes-nn/scholar-lens/actions/workflows/ci.yml/badge.svg)](https://github.com/bits-bytes-nn/scholar-lens/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![AWS CDK](https://img.shields.io/badge/IaC-AWS%20CDK-orange)
![Bedrock](https://img.shields.io/badge/LLM-Amazon%20Bedrock%20(Claude)-green)

[English README](./README.md) · [설계 문서](./docs/design.md)

![Sample](./assets/sample.png)

</div>

## 하는 일

| 산출물 | 입력 | 출력 |
| --- | --- | --- |
| **리뷰** | arXiv ID 또는 논문 PDF URL(코드 저장소 선택) | 일반 DS/MLE 독자를 위한 절별 리뷰입니다. 쉬운 내용은 짧게, 어려운 내용은 직관 → 수식 → 작은 예제 순으로 쌓아 올리며 그림, 표, TL;DR, 한계와 시사점을 포함합니다 |
| **요약** | arXiv ID 또는 논문 PDF URL | TL;DR과 다섯 개의 짧은 절(동기, 아이디어, 구현, 결과, 의의와 한계) |
| **기술 가이드** | 문서 URL 하나 이상 | 하위 페이지와 웹 검색으로 조사한, 출처에 근거한 튜토리얼 |

모든 글은 대상 블로그의 규칙을 따르는 Jekyll Markdown이며, 저장하기 전에 검사합니다. 규칙을 어긴 글은 게시하지 않고 거부합니다. 글은 로컬 파일로 저장되고, 설정에 따라 S3에 올라가며 블로그 저장소에 PR로 제안됩니다.

## 동작 방식

![생성 파이프라인](./docs/diagrams/pipeline.png)

- **리뷰**: 계획 단계가 논문 전체를 읽고 집필 계획을 만듭니다. 계획에는 절 구성, 반드시 다룰 항목, 난이도, 선수 개념, 개념별 담당 절, 그림 배정, 용어집이 들어갑니다. 각 절은 지시서대로 쓴 뒤 지시서에 비추어 검수하고, 빠진 항목이 없을 때까지 고칩니다. 마지막으로 글 전체를 한 번 편집해 용어, 중복, 누락을 바로잡습니다.
- **요약**: 파싱한 논문 전체를 입력으로 한 번 호출합니다.
- **가이드**: 조사 → 관련성 판정 → 개요 → 절별 작성·평가·사실 확인 순서입니다.
- **발행**: 수식과 Markdown 정규화, 그림 로컬화, 블로그 규칙 검사를 거친 뒤 S3에 올리고, 글마다 고정된 브랜치로 GitHub PR을 엽니다.

![AWS 아키텍처](./docs/diagrams/aws-architecture.png)

단계별 동작, 규칙, 설정 키는 [docs/design.md](./docs/design.md)에 있습니다.

## 빠른 시작

사전 조건:

- Python 3.12 이상과 [Poetry](https://python-poetry.org/)
- `bedrock_region_name` 리전에서 설정한 모델(기본값: Claude Opus 5.5, Sonnet 5.5, Haiku 5.5)에 접근할 수 있는 AWS 프로필
- PDF 파싱용 `UPSTAGE_API_KEY`(arXiv가 아닌 소스와 HTML 대체 경로에 필요)
- 선택: 가이드 웹 조사용 `TAVILY_API_KEY` 또는 `BRAVE_API_KEY`, 블로그 PR용 `GITHUB_TOKEN`, 인용 조회 한도를 높이는 `CROSSREF_MAILTO`(Crossref·OpenAlex polite pool)와 `SEMANTIC_SCHOLAR_API_KEY`(키가 있을 때만 Semantic Scholar 조회)

```bash
git clone https://github.com/bits-bytes-nn/scholar-lens.git && cd scholar-lens
poetry install --with dev
cp scholar_lens/configs/config-template.yaml scholar_lens/configs/config.yaml   # 편집
cp .env.template .env                                                           # 키 입력
poetry run python -m scholar_lens summarize 2106.09685 --no-pr
```

## 설정

`scholar_lens/configs/config.yaml`(Git 제외)은 엄격하게 검증하며, 모르는 키가 있으면 오류입니다. 템플릿에 모든 키와 기본값이 있습니다. 파일이 없으면 로컬 실행은 이 기본값(S3 버킷 없음, GitHub 연동 꺼짐)과 기본 AWS 자격 증명으로 동작하므로, Bedrock을 호출하고 비용이 듭니다. 주로 바꾸는 값은 다음과 같습니다.

- `resources`: `profile_name`, `bedrock_region_name`, `s3_bucket_name`(`cdk deploy`와 Batch 실행에 필요. 없으면 `status`는 저장된 결과 없이 Batch 상태만 보여 줌), `email_address`(SNS 알림)
- `resources.github`: `enabled`, `repo_name`(예: `owner/owner.github.io`), `base_branch`, `branch_prefix`, 산출물 유형별 카테고리 이름
- 단계별 `*_model_id`, `thinking_effort`, 품질 기준, `max_total_tokens` 비용 상한(기본값: 리뷰 800만, 요약 100만, 가이드 300만 토큰)
- `output_language`(기본값 `Korean`)

## CLI

```bash
poetry run python -m scholar_lens review 2106.09685 --repo-url "https://github.com/microsoft/LoRA"
poetry run python -m scholar_lens summarize "https://openreview.net/pdf?id=nZeVKeeFYf9" --no-pr
poetry run python -m scholar_lens guide "https://docs.vllm.ai/en/latest/" --search-query "vLLM paged attention"
poetry run python -m scholar_lens review 2106.09685 --remote     # AWS Batch에 제출
poetry run python -m scholar_lens status <job-id> --wait         # 원격 작업 조회
```

| 옵션 | 명령 | 효과 |
| --- | --- | --- |
| `--repo-url URL` | review, summarize | 논문 코드에 근거해 서술합니다(반복 가능) |
| `--parse-pdf` | review, summarize | arXiv HTML이 있어도 PDF를 파싱합니다 |
| `--search-query Q` | guide | 웹 검색어를 추가합니다(반복 가능) |
| `--no-subpages` | guide | 하위 페이지를 따라가지 않습니다 |
| `--no-pr` | review, summarize, guide | 글은 저장(·업로드)하되 PR은 열지 않습니다 |
| `--remote` | review, summarize, guide | AWS Batch에 제출하고 작업 ID를 돌려줍니다 |
| `--wait`, `--timeout S` | status | 작업이 끝나거나 `S`초가 지날 때까지 조회합니다 |

모든 명령은 stdout에 JSON 객체 하나(실행은 `JobResult`, `--remote`는 `job_id`, `status`는 Batch 상태와 저장된 결과)를 출력하고, 로그는 stderr로 보냅니다. 종료 코드는 `0` 성공 또는 제출, `1` 실패, `2` 입력 거부, `3` 진행 중, `4` 명령 자체의 실패입니다([상세](./docs/design.md#21-명령)). URL은 따옴표로 감쌉니다. zsh에서는 따옴표 없는 URL의 `?`가 glob으로 해석되어 명령이 실패합니다.

## 에이전트에서 쓰기(OpenClaw, Claude Code, Codex)

에이전트도 같은 CLI를 씁니다. `~/.openclaw/workspace/skills/scholar-lens/SKILL.md` 스킬이 호출 방법, `--remote`를 쓸 때, 사용자가 게시를 요청하지 않으면 `--no-pr`이 기본이라는 점을 안내합니다. MCP 서버 대신 CLI와 스킬을 택한 이유는 실행이 수 분에서 한 시간까지 걸리기 때문입니다. `--remote`와 `status`로 이미 비동기 계약이 있고, 스킬 디렉터리는 Claude Code와 Codex에도 공유됩니다.

## 블로그 규칙 검사

글을 저장하기 전에 블로그의 `_data/post_conventions.yml`(사본 `assets/post_conventions.yml`) 규칙으로 검사합니다. error가 있으면 게시를 멈추고 글을 `*.rejected.md`로 남기며, warning은 PR 체크리스트가 됩니다([규칙](./docs/design.md#6-블로그-규칙과-검증)).

## AWS 배포

위 조건에 더해 Node.js, Docker, AWS CDK CLI와 `resources.s3_bucket_name`에 적은 기존 S3 버킷이 필요합니다(스택은 버킷을 만들지 않습니다. 작업 역할이 버킷 전체 목록을 볼 수 있으므로 Scholar Lens 전용 버킷을 씁니다). `resources.profile_name`과 같은 프로필을 넘깁니다.

```bash
cdk bootstrap --profile <profile_name>           # 계정·리전당 한 번
cdk deploy --profile <profile_name>              # cdk.json: poetry run python scripts/deploy_infra.py
poetry run python scripts/put_secrets.py         # .env의 비밀값을 SSM SecureString에 기록
poetry run python scripts/put_secrets.py github-token   # 비밀값 하나를 교체한 뒤
```

Batch 작업은 마지막 배포 때 이미지에 넣은 `config.yaml`을 쓰므로, 설정을 고치면 `cdk deploy`를 다시 실행합니다([스택 상세](./docs/design.md#10-인프라)).

## 개발

```bash
poetry run pytest                                # 단위·파이프라인 테스트(AWS·네트워크 없음)
poetry run ruff check scholar_lens scripts tests
poetry run black --check scholar_lens scripts tests
poetry run mypy scholar_lens scripts
poetry run python scripts/ci_synth_check.py      # 가짜 계정으로 CDK synth
```

## 변경 이력

- `citations.citation_analysis_model_id`를 없앴습니다. 기존 `config.yaml`에서 지우지 않으면 설정을 불러올 때 `extra_forbidden` 오류가 납니다.

## 라이선스

[LICENSE](./LICENSE)를 참고하세요.
