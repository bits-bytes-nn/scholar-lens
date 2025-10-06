FROM public.ecr.aws/ubuntu/ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul \
    LANG=ko_KR.UTF-8 \
    LC_ALL=ko_KR.UTF-8

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    git \
    locales \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && locale-gen ko_KR.UTF-8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY scholar_lens/requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY scholar_lens ./scholar_lens

CMD ["python3", "-m", "scholar_lens.main"]
