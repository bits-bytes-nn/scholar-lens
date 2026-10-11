# AWS Batch job image. Dependencies come from poetry.lock (the set CI tests);
# the job definition supplies the command (`python3 -m scholar_lens worker ...`).
ARG PYTHON_IMAGE=public.ecr.aws/docker/library/python:3.12-slim-bookworm

FROM ${PYTHON_IMAGE} AS builder
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
RUN python -m venv /opt/venv-poetry \
    && /opt/venv-poetry/bin/pip install "poetry==2.2.1" \
    && python -m venv /opt/venv
WORKDIR /build
COPY pyproject.toml poetry.lock README.md ./
RUN /opt/venv-poetry/bin/poetry install --only main --no-root --no-interaction

FROM ${PYTHON_IMAGE}
# git: repository cloning for code-grounded reviews. tzdata: post timestamps.
# fonts-dejavu-core: text in SVG figures rasterized for the blog.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git tzdata fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
ENV TZ=Asia/Seoul \
    LANG=C.UTF-8 \
    NLTK_DATA=/opt/nltk_data \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
COPY --from=builder /opt/venv /opt/venv
# Sentence splitter of the review workflow, fetched at build time so a job never
# depends on downloading it.
RUN python -m nltk.downloader -d /opt/nltk_data punkt_tab
WORKDIR /app
COPY assets/post_conventions.yml ./assets/post_conventions.yml
# Includes the gitignored scholar_lens/configs/config.yaml of the deploying
# checkout: Batch jobs run with the config as of the last `cdk deploy`.
COPY scholar_lens ./scholar_lens
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser
CMD ["python3", "-m", "scholar_lens", "--help"]
