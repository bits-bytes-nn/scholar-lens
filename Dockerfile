FROM public.ecr.aws/ubuntu/ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul \
    LANG=ko_KR.UTF-8 \
    LC_ALL=ko_KR.UTF-8

# Install Python 3.12 (matches pyproject `^3.12` and the CI test matrix) from
# the deadsnakes PPA; Ubuntu 22.04 ships only 3.10.
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common gnupg && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    git \
    curl \
    locales \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && locale-gen ko_KR.UTF-8 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY scholar_lens/requirements.txt .
RUN python3.12 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3.12 -m pip install --no-cache-dir numpy numba && \
    python3.12 -m pip install --no-cache-dir -r requirements.txt

COPY scholar_lens ./scholar_lens

# Run as a non-root user (defense-in-depth: limits blast radius if the process
# is compromised). Done after installs/copies so build steps keep root.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

CMD ["python3", "-m", "scholar_lens.main"]
