---
layout: post
title: "vLLM in Practice: PagedAttention Serving from Install to Production"
date: 2026-11-03 06:15:15
description: "vLLM의 PagedAttention 원리부터 설치, 오프라인 추론, vllm serve 온라인 서빙, 프리픽스 캐싱, 분산 실행, Kubernetes 배포까지 운영 흐름을 정리합니다. KV 캐시 메모리 관리와 스케줄러, 벤치마크 관측 방법도 함께 다룹니다."
categories: ["Tech Guides", "Language-Models"]
tags: ["PagedAttention", "Prefix-Caching", "Continuous-Batching", "OpenAI-Compatible-Server", "Kubernetes-Helm-Deployment", "Model-Serving"]
cover: /assets/images/language-models.jpg
use_math: true
---
## vLLM이 해결하는 문제와 핵심 아이디어

LLM 서빙은 연산보다 메모리에서 먼저 막힙니다. 자기회귀 디코딩에서는 입력 토큰마다 만든 키·값 텐서를 GPU 메모리에 남겨 두고, 다음 토큰을 만들 때 다시 씁니다. 이 KV 캐시(KV cache)는 LLaMA-13B에서 시퀀스 하나에 최대 1.7GB를 차지합니다. 크기는 시퀀스 길이에 따라 달라지는데, 이 길이는 편차가 크고 미리 알 수 없습니다. 기존 시스템은 단편화와 과다 예약 때문에 메모리의 60~80%를 낭비했습니다.

PagedAttention은 운영체제의 가상 메모리와 페이징 개념을 빌려 이 문제를 풉니다. 시퀀스의 KV 캐시를 일정 개수의 토큰을 담는 블록으로 나누고, 블록끼리는 메모리에서 연속이 아니어도 됩니다. 물리 블록은 새 토큰이 생성될 때 필요한 만큼만 할당합니다. 그래서 모든 위치에 쓸 공간을 미리 잡아 둘 필요가 없습니다.

![KV 캐시가 비연속 메모리 블록에 나뉘어 저장되는 PagedAttention 동작 애니메이션](https://vllm.ai/blog-assets/figures/annimation0.gif)

애니메이션을 보면 한 시퀀스의 토큰이 서로 떨어진 블록에 채워집니다. 그래도 어텐션 커널은 그 블록들을 찾아 읽습니다. 낭비는 시퀀스의 마지막 블록에서만 생기며 4% 미만입니다. 줄인 메모리만큼 한 배치에 더 많은 시퀀스가 들어가고, 그만큼 GPU 사용률과 처리량이 올라갑니다. vLLM 발표 블로그는 LLaMA-7B(A10G)와 LLaMA-13B(A100 40GB)로 실험했고, 요청의 입출력 길이는 ShareGPT 데이터셋에서 샘플링했습니다. 그 결과 HuggingFace Transformers보다 처리량이 최대 24배 높았다고 보고했습니다.

이 가이드는 설치에서 시작해 `LLM` 클래스로 하는 오프라인 추론과 `vllm serve`로 하는 온라인 서빙을 다룹니다. 이어서 블록 관리, 프리픽스 캐싱(prefix caching), 스케줄러의 동작 원리를 설명합니다. 그다음 메모리 튜닝, 분산 실행, CPU 백엔드, Kubernetes 배포, 관측과 벤치마크까지 이어집니다.

## 사전 요구 사항과 설치 경로 선택

vLLM은 Linux에서 동작하는 Python 라이브러리이며, Python 3.11~3.14를 지원합니다. Windows는 기본 지원 대상이 아닙니다. Windows에서 쓰려면 WSL(Windows Subsystem for Linux)에 호환 배포판을 올리거나 커뮤니티가 관리하는 포크를 사용합니다. macOS에서는 Apple Silicon GPU 가속용 플러그인인 vLLM-Metal을 거쳐 실행합니다.

가장 흔한 NVIDIA 환경에는 연산 능력(compute capability) 7.5 이상 GPU가 필요합니다. T4, RTX20xx, A100, L4, H100, B200 등이 여기에 들어갑니다. 배포 휠에는 CUDA 12.9로 미리 컴파일한 C++·CUDA 바이너리가 들어 있습니다. 성능을 내려고 많은 CUDA 커널을 컴파일해 두었으므로 다른 CUDA 버전이나 PyTorch 빌드와 바이너리가 호환되지 않습니다. PyTorch 버전이 같아도 빌드 설정이 다르면 마찬가지입니다. 따라서 새 Python 환경에 설치하는 방법을 권장합니다. 다른 CUDA 버전을 써야 하거나 기존 PyTorch 설치를 그대로 쓰려면 소스에서 빌드해야 합니다.

### uv로 NVIDIA CUDA 환경에 설치하기

권장 도구는 빠른 Python 환경 관리자인 uv입니다. uv를 설치한 뒤 다음 명령으로 가상 환경을 만들고 vLLM을 설치합니다.

```bash
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
vllm download-kernels
```

`--torch-backend=auto`를 주면 uv가 설치된 CUDA 드라이버 버전을 확인하고, 실행 시점에 맞는 PyTorch 인덱스를 고릅니다. 환경 변수 `UV_TORCH_BACKEND=auto`로도 같은 효과를 냅니다. 특정 백엔드를 고정하려면 `--torch-backend=cu130`이나 `UV_TORCH_BACKEND=cu130`처럼 지정합니다. 자동 선택이 동작하지 않으면 먼저 `uv self update`로 uv를 최신 버전으로 올려 봅니다.

영구 환경 없이 명령만 한 번 실행해 보려면 `uv run`의 `--with` 옵션을 씁니다.

```bash
uv run --with vllm vllm --help
```

conda로 환경을 관리하는 팀이라면 conda 환경 안에 uv를 설치하고, 패키지 설치는 uv로 진행합니다. conda로 설치한 PyTorch는 NCCL 라이브러리를 정적으로 링크하므로, vLLM이 NCCL을 쓸 때 문제가 생길 수 있습니다. PyTorch는 conda가 아니라 vLLM 설치 과정에서 받는 편이 안전합니다.

```bash
conda create -n myenv python=3.12 -y
conda activate myenv
pip install --upgrade uv
uv pip install vllm --torch-backend=auto
```

pip만 써야 한다면 PyTorch 인덱스를 직접 지정합니다.

```bash
# CUDA 12.9용 vLLM 설치
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu129
vllm download-kernels
```

기본 바이너리는 CUDA 12.9와 공개 PyTorch 릴리스로 컴파일되어 있습니다. CUDA 12.8용과 13.0용 바이너리도 함께 배포됩니다. NVIDIA Blackwell GPU(B200, GB200)에는 CUDA 12.8 이상이 필요합니다. 특정 CUDA 버전이 필요하면 릴리스 페이지의 휠을 직접 지정합니다.

```bash
# 특정 CUDA 버전(예: 13.0)용 vLLM 설치
export VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest | jq -r .tag_name | sed 's/^v//')
export CUDA_VERSION=130 # 또는 다른 버전
export CPU_ARCH=$(uname -m) # x86_64 또는 aarch64
uv pip install "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu${CUDA_VERSION}-cp38-abi3-manylinux_2_28_${CPU_ARCH}.whl" --extra-index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION}"
```

### `vllm download-kernels`가 필요한 이유

FlashInfer는 vLLM과 함께 설치되지만, FlashInfer의 사전 컴파일 커널은 함께 설치되지 않습니다. 이 커널이 없으면 vLLM이 시작할 때 커널을 내려받아 컴파일하고 경고를 남깁니다. Hopper 이후 GPU, 특히 Blackwell에서는 이 과정 때문에 기동이 몇 분씩 늦어집니다. `vllm download-kernels`는 설치된 FlashInfer와 CUDA 버전에 맞는 사전 컴파일 커널을 미리 설치해서 이 지연을 없앱니다.

커널은 FlashInfer 버전에 묶여 있습니다. 버전이 맞지 않으면 FlashInfer가 아예 임포트되지 않으므로, vLLM을 업그레이드하면 커널도 다시 받아야 합니다. 이 명령은 vLLM을 설치하거나 업그레이드할 때마다 같은 Python 환경에서 실행합니다. 사용자 정의 컨테이너 이미지를 만들 때도 빌드 단계에 넣어 둡니다. CUDA 12.9보다 오래된 CUDA용 PyTorch를 쓰면 `flashinfer-cubin`만 설치됩니다. 해당 버전용 `flashinfer-jit-cache`를 FlashInfer가 배포하지 않기 때문입니다. CUDA 버전을 직접 지정하거나 야간 커널을 받으려면 FlashInfer CLI를 직접 사용합니다.

> **핵심:** `vllm download-kernels`는 처음 한 번만 실행하는 명령이 아닙니다. vLLM 버전을 바꿀 때마다 다시 실행해야 기동 지연과 FlashInfer 임포트 실패를 막습니다.

### 플랫폼별 설치 변형

NVIDIA 외 가속기는 패키지나 인덱스, 플러그인이 각각 다릅니다. 하드웨어 플러그인은 메인 `vllm` 저장소 밖에서 관리되며, Hardware-Pluggable RFC를 따릅니다.

| 플랫폼 | 설치 방법 | 주요 조건 |
| --- | --- | --- |
| NVIDIA CUDA | `uv pip install vllm --torch-backend=auto` | 연산 능력 7.5 이상 GPU, CUDA 12.9 기본 바이너리 |
| AMD ROCm | `uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/` | Python 3.12 전용 휠, `glibc >= 2.35`, ROCm 10.0 휠 기본 |
| Intel XPU | XPU 백엔드, v0.26.0부터 공식 Docker 이미지 제공 | Python 3.12 필수, `vllm-xpu-kernels` 의존성 |
| Google TPU | `uv pip install vllm-tpu` | 별도 TPU 문서의 설치 절차 |
| Ascend NPU | 커뮤니티 관리 플러그인 vLLM Ascend | NPU 하드웨어와 CANN 버전에 따른 구성 |
| Apple Silicon(Mac) | vLLM-Metal 패키지 | macOS Sonoma 이상, MLX 백엔드, mlx-community 모델 |
| CPU | CPU 전용 설치 경로 | Intel/AMD x86, ARM AArch64, Apple silicon, IBM Z(S390X) 지원 |

AMD 쪽에서 uv를 권장하는 이유는 인덱스 우선순위에 있습니다. uv는 추가 인덱스를 기본 인덱스보다 먼저 봅니다. 반면 pip는 `--extra-index-url`과 기본 인덱스의 패키지를 합친 다음 최신 버전만 고릅니다. ROCm 휠은 MI200s(gfx90a), MI300(gfx942), MI350(gfx950), Radeon RX 7900·9000 시리즈, Ryzen AI MAX / AI 300 시리즈를 대상으로 합니다. ROCm은 6.3 이상이어야 하며, MI350은 ROCm 7.0 이상, Ryzen AI MAX / AI 300 시리즈는 7.0.2 이상이 필요합니다. 또 ROCm 휠은 Python 3.12용으로만 배포됩니다. 다른 Python 버전에서 설치하면 PyPI의 CUDA 휠로 조용히 대체되고, AMD GPU에서 `libcudart.so: cannot open shared object file` 같은 오류를 내며 실패합니다. 설치 전에 버전을 확인하고, 필요하면 3.12 환경을 따로 만듭니다.

```bash
python3 --version
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/ --upgrade
```

ROCm 10.0 휠은 pip로 ROCm SDK(TheRock)를 함께 설치하며, 지원하는 모든 아키텍처의 GPU 커널이 포함됩니다. 시스템 ROCm 설치를 쓰는 기존 ROCm 7.2 스택용 휠은 이전 릴리스에 한해 계속 제공됩니다.

Apple Silicon의 vLLM-Metal은 PyTorch 대신 MLX를 연산 백엔드로 씁니다. 그래서 Hugging Face mlx-community의 MLX 최적화 모델이 필요하며, `mlx-community/Qwen2.5-0.5B-Instruct-4bit`가 그 예입니다. 기능별 하드웨어 지원 범위는 공식 문서의 Feature x Hardware 호환성 표에서 확인합니다.

## 컨테이너 이미지와 소스 빌드

vLLM에는 미리 컴파일된 C++·CUDA 바이너리가 들어 있어서, 운영 환경에서는 의존성을 맞춰 둔 공식 컨테이너 이미지로 시작하는 편이 가장 간단합니다. 이미지가 맞지 않으면 이미지를 확장하고, 그래도 안 되면 소스에서 빌드합니다.

### 공식 Docker 이미지로 서버 띄우기

NVIDIA용 공식 이미지는 Docker Hub의 `vllm/vllm-openai`이며, OpenAI 호환 서버를 실행합니다. 기본 CUDA 릴리스 이미지는 Ubuntu 24.04 기반입니다. `-ubuntu2404` 접미사 태그는 더 이상 배포하지 않으므로 `latest`나 `v<version>`을 씁니다. 아키텍처를 고정하려면 `latest-x86_64`, `latest-aarch64`, `v<version>-x86_64`, `v<version>-aarch64` 태그를 씁니다.

```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen3-0.6B
```

호스트의 Hugging Face 캐시를 마운트하면 컨테이너를 다시 띄워도 모델을 새로 받지 않습니다. 이미지 태그 뒤의 인자는 그대로 엔진 인자로 넘어갑니다. `--ipc=host`(또는 `--shm-size`)는 컨테이너가 호스트의 공유 메모리를 쓰게 합니다. PyTorch가 프로세스 간 데이터를 공유 메모리로 주고받기 때문이며, 특히 텐서 병렬(tensor parallelism) 추론에서 필요합니다. Podman에서는 GPU를 `--device nvidia.com/gpu=all`로 넘기고 이미지 이름을 `docker.io/vllm/vllm-openai:latest`로 씁니다. `podman build`에서 문제가 생기면 `--security-opt label=disable`로 SELinux 레이블링을 꺼야 할 수도 있습니다.

| 플랫폼 | 이미지 | 비고 |
| --- | --- | --- |
| NVIDIA CUDA | `vllm/vllm-openai` | Ubuntu 24.04 기반 기본 릴리스 이미지 |
| AMD ROCm | `vllm/vllm-openai-rocm` | ROCm 10.0 기반, 레이어 풀에 Docker 23 이상 또는 containerd 1.5 이상 필요 |
| Intel XPU | `vllm/vllm-openai-xpu` | v0.26.0부터 안정 릴리스 제공 |
| CPU(x86, Arm) | `vllm/vllm-openai-cpu` | `latest-x86_64`, `latest-arm64` 등 아키텍처별 태그 |

ROCm 컨테이너에는 `--device /dev/kfd`, `--device /dev/dri`, `--group-add=video`를 주고, XPU 컨테이너는 `/dev/dri`를 넘긴 뒤 `--privileged`로 실행합니다.

호스트 드라이버가 이미지의 CUDA 툴킷보다 오래됐다면 실행 명령에 `--env "VLLM_ENABLE_CUDA_COMPATIBILITY=1"`을 더합니다. 그러면 vLLM이 PyTorch를 불러오기 전에 `LD_LIBRARY_PATH`를 이미지에 포함된 CUDA 호환성 라이브러리 경로로 맞춥니다. 일부 전문가용·데이터센터용 NVIDIA GPU에서만 동작합니다. CUDA 13 이미지는 원래 R580 이상 드라이버가 필요하지만, 호환 모드에서는 R535와 R570 드라이버도 지원합니다.

### 공식 이미지를 확장한 사용자 정의 이미지

공식 이미지에는 라이선스 문제 때문에 선택적 의존성이 빠져 있습니다. 오디오 처리 같은 기능이 필요하면 라이선스 조건을 확인한 뒤 설치 레이어를 하나 얹습니다. 설치하는 vLLM 버전은 베이스 이미지와 반드시 같아야 합니다.

```dockerfile
FROM vllm/vllm-openai:v0.11.0

# 예: `audio` 선택적 의존성 설치
# 주의: vLLM 버전을 베이스 이미지와 맞춰야 합니다
RUN uv pip install --system vllm[audio]==0.11.0
```

HF Transformers main 브랜치에만 들어 있는 모델도 있습니다. 이럴 때는 개발 버전 `transformers`를 설치하는 레이어를 추가합니다.

```dockerfile
FROM vllm/vllm-openai:latest

RUN uv pip install --system git+https://github.com/huggingface/transformers.git
```

### 야간 빌드와 특정 커밋 설치

LLM 추론 분야는 변화가 빨라서, 최신 코드에는 아직 릴리스되지 않은 버그 수정과 성능 개선, 새 기능이 들어 있을 수 있습니다. ROCm은 `vllm/vllm-openai-rocm:nightly`, XPU는 `vllm/vllm-openai-xpu:nightly` 태그가 최신 개발 브랜치 빌드입니다. CPU 휠은 야간 인덱스에서 받으며, uv를 씁니다. pip은 `--extra-index-url`과 기본 인덱스의 패키지를 합쳐 최신 버전만 고르지만, uv는 추가 인덱스를 기본 인덱스보다 우선하기 때문입니다.

```bash
uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly/cpu --index-strategy first-index --torch-backend cpu
```

동작 변화나 성능 저하가 생긴 커밋을 이분 탐색할 때는 URL에 전체 커밋 해시를 넣습니다.

```bash
export VLLM_COMMIT=730bd35378bf2a5b56b6d3a45be28b3092d26519 # main 브랜치의 전체 커밋 해시
uv pip install vllm --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT}/cpu --index-strategy first-index --torch-backend cpu
```

Arm CPU용 커밋별 CI 이미지는 테스트 용도라 며칠 뒤 만료되므로 운영에 쓰지 않습니다.

### 소스에서 이미지 빌드하기

저장소의 `docker/Dockerfile`로 이미지를 직접 빌드합니다. `max_jobs`와 `nvcc_threads` 빌드 인자로 컴파일 병렬도를 조절합니다.

```bash
# 선택적으로 지정: --build-arg max_jobs=8 --build-arg nvcc_threads=2
DOCKER_BUILDKIT=1 docker build . \
    --target vllm-openai \
    --tag vllm/vllm-openai \
    --file docker/Dockerfile
```

기본 빌드는 배포 범위를 넓히려고 모든 GPU 종류를 대상으로 컴파일합니다. `--build-arg torch_cuda_arch_list=""`를 주면 아키텍처 선택을 PyTorch에 넘기지만, 빌드 중 GPU가 보여야 효과가 있습니다. 일반 BuildKit 빌드는 GPU를 노출하지 않아 PyTorch가 공통 아키텍처 목록으로 되돌아갑니다.

C++·CUDA 커널을 바꾸지 않았다면 `--build-arg VLLM_USE_PRECOMPILED="1"`로 컴파일을 건너뜁니다. vLLM이 업스트림 `main`과 갈라진 지점(머지 베이스) 커밋을 찾아 야간 빌드의 휠을 가져오므로 빌드 시간이 크게 줄어듭니다. 특정 커밋의 휠은 `--build-arg VLLM_PRECOMPILED_WHEEL_COMMIT=<commit_hash>`로 지정합니다. 빌드한 이미지는 공식 이미지와 같은 방식으로, 이미지 이름 자리에 빌드 태그를 넣어 실행합니다.

ROCm은 Docker Buildx로 `docker/docker-bake-rocm.hcl`을 실행합니다. 배포된 스택 이미지를 재사용하면 vLLM만 컴파일하고, `STACK_IMAGE`와 `RUNTIME`을 빼면 전체 스택을 빌드하느라 몇 시간이 걸립니다. 자기 GPU용으로만 컴파일하려면 `--set vllm-openai.args.ARCH=gfx942`처럼 지정합니다.

```bash
STACK_IMAGE=vllm/vllm-openai-rocm:base-nightly RUNTIME=vllm/vllm-openai-rocm:base-nightly \
    TAG=vllm/vllm-openai-rocm docker buildx bake -f docker/docker-bake-rocm.hcl vllm-openai
```

### Python 전용 빌드와 전체 빌드

Python 코드만 고칠 때는 컴파일 없이 진행하는 Python 전용 빌드를 씁니다. 컴파일된 부분은 사전 빌드 휠에서 가져오고 소스는 편집 가능(editable) 상태로 설치하므로, 해당 플랫폼용 휠이 있어야 합니다. Docker 빌드의 `VLLM_USE_PRECOMPILED`와 같은 원리입니다. CPU 백엔드에서는 다음과 같이 실행합니다.

```bash
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_VARIANT=cpu VLLM_TARGET_DEVICE=cpu uv pip install --editable .
```

C++·CUDA 커널을 고쳤다면 전체 빌드를 합니다. 동시 컴파일이 많으면 시스템에 부하가 걸리므로 `MAX_JOBS`로 제한합니다. WSL은 기본적으로 전체 메모리의 50%만 할당하므로 `MAX_JOBS=1`이 메모리 부족을 막아 주지만 빌드가 훨씬 느려집니다.

```bash
export MAX_JOBS=6
uv pip install -e .
```

빌드가 계속 실패하면 NVIDIA PyTorch Docker 이미지 안에서 빌드하거나, CUDA Toolkit을 설치하고 `CUDA_HOME`을 설정합니다. CPU 전체 빌드는 의존성 설치 후 `VLLM_TARGET_DEVICE=cpu uv pip install . --no-build-isolation`으로 진행하며, CMake가 CUDA를 잡으면 `CMAKE_DISABLE_FIND_PACKAGE_CUDA=ON`을 줍니다. ROCm에서는 `pip install .`이 동작하지 않으므로 `python3 setup.py develop`으로 빌드합니다.

## 오프라인 배치 추론

서버를 띄우지 않고 Python 스크립트 안에서 프롬프트 목록을 한꺼번에 처리하는 방식을 오프라인 배치 추론이라고 합니다. 데이터셋 전체의 응답 생성, 평가용 출력 만들기, 합성 데이터 생성처럼 지연 시간보다 처리량이 중요한 작업에 맞습니다. 핵심 클래스는 둘입니다.

- `LLM`: vLLM 엔진으로 오프라인 추론을 실행하는 주 클래스입니다. `LLM` 객체를 만들 때 엔진을 초기화하고 모델을 불러옵니다.
- `SamplingParams`: 온도, top-p(누적 확률) 샘플링 값 같은 샘플링 과정의 매개변수를 지정합니다.

### 기본 흐름: `LLM`과 `SamplingParams`

공식 예제 `examples/basic/offline_inference/basic.py`의 흐름을 그대로 따르면 다음과 같습니다. 작은 모델인 OPT-125M을 쓰므로 설치 확인용으로도 적당합니다.

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="facebook/opt-125m")

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

`llm.generate`는 입력 프롬프트를 엔진의 대기 큐에 넣은 다음 엔진을 돌려 높은 처리량으로 출력을 만듭니다. 그래서 프롬프트를 반복문으로 한 개씩 넘기기보다 목록 전체를 한 번에 넘겨야 엔진이 여러 시퀀스를 함께 배치로 처리하는 효과를 제대로 냅니다. 반환값은 `RequestOutput` 객체 목록이며, 각 객체에 원래 프롬프트(`output.prompt`)와 생성된 출력 토큰이 모두 들어 있습니다. 생성 텍스트는 `output.outputs[0].text`로 꺼냅니다. `outputs`가 다시 목록인 이유는 요청 하나가 출력 시퀀스를 여러 개 만들 수 있기 때문입니다.

모델은 기본적으로 Hugging Face에서 내려받습니다. ModelScope의 모델을 쓰려면 엔진을 초기화하기 전에 환경 변수를 설정합니다.

```bash
export VLLM_USE_MODELSCOPE=True
```

지원 모델 목록은 공식 문서의 지원 모델 페이지에서 확인합니다.

### `generation_config` 기본 동작

기본적으로 vLLM은 Hugging Face 모델 저장소에 `generation_config.json`이 있으면 그 파일을 적용해서 모델 제작자가 권장한 샘플링 매개변수를 씁니다. `SamplingParams`를 지정하지 않는 경우 대부분은 이 방식이 가장 좋은 결과를 냅니다. 모델마다 알맞은 온도나 top-p가 다르고, 그 정보를 가장 잘 아는 쪽이 제작자이기 때문입니다.

반대로 실험의 재현성을 위해 모델과 상관없이 같은 기본값을 쓰고 싶다면, `LLM`을 만들 때 `generation_config="vllm"`을 지정해 vLLM 자체 기본값을 쓰게 합니다.

```python
from vllm import LLM

llm = LLM(model="facebook/opt-125m", generation_config="vllm")
```

여러 모델을 같은 조건으로 비교 평가할 때 이 차이가 중요합니다. 기본 동작에서는 일부 샘플링 매개변수의 기본값을 모델 제작자의 권장값이 대신하므로, 모델마다 실제로 적용되는 샘플링 설정이 달라질 수 있습니다.

### `generate`와 `chat`의 차이, 그리고 채팅 템플릿

Instruct 모델이나 Chat 모델은 학습 때 본 형식, 즉 역할 구분 토큰과 대화 구조가 들어간 입력을 기대합니다. 이 형식을 정의한 것이 토크나이저에 저장된 채팅 템플릿입니다. 그런데 `llm.generate`는 입력 프롬프트에 모델의 채팅 템플릿을 자동으로 적용하지 않습니다. 문자열을 그대로 모델에 넣으므로, 채팅 모델에 원시 문자열을 주면 기대와 다른 출력이 나오기 쉽습니다.

해결 방법은 두 가지입니다. 첫째는 토크나이저로 템플릿을 직접 적용한 뒤 `generate`에 넘기는 방법이고, 둘째는 `llm.chat`에 메시지 목록을 넘기는 방법입니다. `llm.chat`의 메시지 형식은 OpenAI의 `client.chat.completions`에 넘기는 형식과 같습니다.

OPT-125M은 채팅 모델이 아니므로 다음 코드는 채팅 모델인 Qwen2.5-1.5B-Instruct로 `llm`을 새로 만들고, 앞의 `prompts`와 `sampling_params`를 그대로 씁니다. 이 모델 저장소에 `generation_config.json`이 있으면 앞서 설명한 기본 동작이 채팅 예제에도 그대로 적용됩니다.

```python
from transformers import AutoTokenizer
from vllm import LLM

llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")

# 방법 1: 토크나이저로 채팅 템플릿을 직접 적용한 뒤 generate 호출
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
messages_list = [
    [{"role": "user", "content": prompt}]
    for prompt in prompts
]
texts = tokenizer.apply_chat_template(
    messages_list,
    tokenize=False,
    add_generation_prompt=True,
)

outputs = llm.generate(texts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# 방법 2: chat 인터페이스에 메시지 목록을 그대로 전달
outputs = llm.chat(messages_list, sampling_params)
for idx, output in enumerate(outputs):
    prompt = prompts[idx]
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

방법 1에서 `add_generation_prompt=True`는 템플릿 끝에 어시스턴트 응답 시작 부분을 붙여서 모델이 바로 답을 이어 쓰게 합니다. 또 `output.prompt`에는 템플릿이 적용된 전체 문자열이 들어가므로 실제로 모델에 들어간 입력을 그대로 확인할 수 있습니다. 방법 2의 반복문은 원래 질문을 `prompts[idx]`로 다시 꺼내 출력하는데, 각 출력의 순서가 입력 메시지 목록의 순서와 같아서 이렇게 씁니다.

| 구분 | `llm.generate` | `llm.chat` |
| --- | --- | --- |
| 입력 형식 | 프롬프트 문자열 목록 | OpenAI 채팅 형식의 메시지 목록 |
| 채팅 템플릿 | 자동 적용 없음 | 채팅 템플릿을 적용한 입력 구성 |
| 적합한 모델 | 기본(base) 모델, 또는 템플릿을 직접 적용한 입력 | Instruct 모델, Chat 모델 |

채팅 모델에서 템플릿을 직접 제어하거나 실제로 모델에 들어간 입력 문자열을 기록해야 한다면 방법 1을 씁니다.

## OpenAI 호환 서버로 온라인 서빙

`LLM` 클래스는 스크립트 안에서 끝나는 작업에 맞습니다. 여러 애플리케이션이 네트워크로 모델을 호출해야 한다면 vLLM을 서버로 띄웁니다. vLLM 서버는 OpenAI API 프로토콜을 구현하므로 OpenAI API를 쓰던 애플리케이션에서 그대로 바꿔 끼워 쓸 수 있습니다. OpenAI 클라이언트의 API 키와 접속 주소만 vLLM 서버로 바꾸면 OpenAI API와 같은 형식으로 요청을 보냅니다.

### `vllm serve`로 서버 시작하기

서버는 `vllm serve` 명령으로 띄웁니다. 모델 이름만 주면 기본 주소 `http://localhost:8000`에서 요청을 받습니다.

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

서버가 올라왔는지는 모델 목록을 조회해 확인합니다.

```bash
curl http://localhost:8000/v1/models
```

서버 하나는 한 번에 모델 하나만 올립니다. 서버가 구현하는 주요 엔드포인트는 모델 목록 조회(`/v1/models`), 완성(`/v1/completions`), 채팅 완성(`/v1/chat/completions`)입니다.

주소는 `--host`와 `--port`로 바꿉니다. `--uds`로 Unix 도메인 소켓을 지정해 서빙할 수도 있습니다.

```bash
vllm serve meta-llama/Llama-2-7b-hf --port 8100
vllm serve meta-llama/Llama-2-7b-hf --uds /tmp/vllm.sock
```

서버도 Hugging Face 모델 저장소에 `generation_config.json`이 있으면 기본으로 적용하므로, 일부 샘플링 파라미터의 기본값이 모델 제작자의 권장값으로 바뀝니다. 이 동작은 `--generation-config vllm`으로 끕니다. 채팅 템플릿은 따로 지정하지 않으면 토크나이저에 저장된 템플릿을 씁니다.

옵션이 많으므로 도움말 검색 기능을 익혀 두면 편합니다.

```bash
vllm serve --help=all            # 모든 플래그 나열
vllm serve --help=ModelConfig    # 인자 그룹 하나만 보기
vllm serve --help=max-num-seqs   # 인자 하나만 보기
vllm serve --help=max            # 키워드나 플래그 이름으로 검색
```

정수 인자 상당수는 사람이 읽기 쉬운 접미사를 받습니다. 소문자 `k`, `m`, `g`, `t`는 10진 단위(`1k` = 1,000)이고, 대문자 `K`, `M`, `G`, `T`는 2진 단위(`1K` = 1,024)입니다. 10진 접미사는 `25.6k`처럼 소수도 받지만 2진 접미사는 정수만 받습니다. `--max-model-len`, `--max-num-batched-tokens`, `--kv-cache-memory-bytes` 등이 이 형식을 지원합니다. `32k`와 `32K`는 값이 서로 다르니 주의합니다.

### 완성 엔드포인트

완성(completions) API는 프롬프트 문자열을 받아 이어지는 텍스트를 생성합니다.

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```

요청 본문의 `model`에는 서버를 띄울 때 지정한 모델 이름을 넣습니다. `openai` Python 패키지를 쓸 때는 API 키와 `base_url`을 vLLM 서버에 맞게 바꿉니다.

```python
from openai import OpenAI

# OpenAI 클라이언트의 API 키와 기본 주소를 vLLM 서버로 바꿉니다
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
completion = client.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    prompt="San Francisco is a",
)
print("Completion result:", completion)
```

### 채팅 완성 엔드포인트

채팅 완성(chat completions) API는 역할이 붙은 메시지 목록을 받고, 대화 기록을 메시지로 쌓아 가며 주고받습니다. 맥락이 필요하거나 자세한 설명이 필요한 작업에 유용합니다.

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"}
        ]
    }'
```

```python
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ],
)
print("Chat response:", chat_response)
```

| 구분 | 완성 | 채팅 완성 |
| --- | --- | --- |
| 경로 | `/v1/completions` | `/v1/chat/completions` |
| 입력 | `prompt` 문자열 | 역할이 붙은 `messages` 목록 |
| 오프라인 대응 | `llm.generate` | `llm.chat` |
| SDK 호출 | `client.completions.create` | `client.chat.completions.create` |

간단히 확인할 때는 클라이언트 코드 없이 `vllm` CLI로 실행 중인 서버에 요청을 보냅니다. `vllm chat`과 `vllm complete`는 인자 없이 실행하면 localhost 서버에 접속하고, `--url`로 다른 서버를 지정합니다. `--quick`은 프롬프트 하나로 바로 요청을 보내고, `--stats`는 응답마다 첫 토큰까지 걸린 시간(TTFT)과 처리량을 출력합니다.

```bash
vllm chat --quick "hi"
vllm complete --url http://localhost:8000/v1 --quick "The future of AI is"
```

### API 키로 접근 제한하기

앞의 클라이언트 예제는 `api_key`에 `"EMPTY"`를 넣고도 동작합니다. 키를 지정하지 않은 서버는 헤더의 API 키를 검사하지 않기 때문입니다. 서버를 외부에 노출한다면 `--api-key` 인자나 `VLLM_API_KEY` 환경 변수로 키를 지정합니다. 그러면 서버가 요청 헤더의 API 키를 검사하고, 클라이언트는 `api_key`에 그 키를 넣어야 합니다.

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --api-key old-key new-key
```

`--api-key` 뒤에는 키를 여러 개 줄 수 있고, 서버는 그중 어느 키든 받아들이므로 키를 교체할 때 유용합니다. 옛 키와 새 키를 함께 등록해 두면 클라이언트를 하나씩 새 키로 옮기는 동안 두 키가 모두 통합니다. 다만 명령줄 인자는 프로세스 목록에 그대로 보이므로, 운영 환경에서는 키를 `VLLM_API_KEY` 환경 변수로 넘기는 편이 안전합니다.

### YAML 설정 파일로 인자 관리하기

운영 서버의 명령줄은 금세 길어집니다. 인자를 YAML 파일에 모아 두면 버전 관리와 리뷰가 쉬워집니다. 키 이름에는 CLI 인자의 긴 이름을 씁니다.

```yaml
# config.yaml
model: meta-llama/Llama-3.1-8B-Instruct
host: "127.0.0.1"
port: 6379
uvicorn-log-level: "info"
```

```bash
vllm serve --config config.yaml
```

같은 인자를 명령줄과 설정 파일에 함께 주면 명령줄 값이 우선합니다. 우선순위는 명령줄, 설정 파일, 기본값 순입니다. 예를 들어 `vllm serve SOME_MODEL --config config.yaml`에서는 설정 파일의 `model` 대신 `SOME_MODEL`을 씁니다. 공통 설정은 파일에 두고 환경마다 다른 값만 명령줄로 덮어쓰면 관리가 편합니다.

모델과 하드웨어 조합별 레시피는 vLLM Recipes에 정리되어 있습니다. 저장소의 Recipes 변환 도구는 레시피를 `config.yaml`과 `env.sh`로 바꿔 줍니다. 생성된 환경 변수 파일을 먼저 불러온 뒤 서버를 띄웁니다.

```bash
source env.sh
vllm serve --config config.yaml
```

## 페이징된 KV 캐시 메모리 관리

vLLM 메모리 관리자의 설계는 운영체제의 가상 메모리를 그대로 옮겨 온 형태입니다. 운영체제는 메모리를 고정 크기 페이지로 나누고, 프로그램의 논리 페이지를 물리 페이지에 대응시킵니다. 그래서 연속된 논리 페이지가 실제로는 흩어진 물리 페이지에 놓여도 프로그램은 메모리가 연속된 것처럼 접근합니다. 물리 메모리를 미리 다 잡아 둘 필요도 없어서, 필요해질 때 페이지를 하나씩 내줍니다. PagedAttention 논문은 이 대응을 KV 캐시에 적용합니다. 블록은 페이지, 토큰은 바이트, 시퀀스는 프로세스에 해당합니다. 이 구조를 알면 vLLM이 왜 같은 GPU에 더 많은 시퀀스를 올리는지, 병렬 샘플링이나 빔 서치가 왜 메모리를 덜 쓰는지, 블록 크기가 성능에 왜 영향을 주는지 설명됩니다.

### 논리 블록, 물리 블록, 블록 테이블

요청 하나의 KV 캐시는 논리 KV 블록이 이어진 형태로 표현됩니다. 새 토큰이 생성되어 그 키와 값이 계산될 때마다 블록을 왼쪽부터 채우고, 마지막 블록의 빈 자리는 앞으로 생성될 토큰 몫으로 남겨 둡니다. 논리 블록은 시퀀스 입장에서 본 연속된 주소 공간일 뿐, 실제 데이터는 물리 블록에 들어갑니다.

물리 블록은 GPU 워커의 블록 엔진이 만듭니다. 블록 엔진은 GPU DRAM에서 연속된 큰 영역을 한 번 할당한 다음, 이를 고정 크기 물리 KV 블록으로 잘라 둡니다. 스왑에 대비해 CPU RAM에도 같은 방식으로 물리 블록을 준비합니다. 논리 블록과 물리 블록을 잇는 것이 블록 테이블(block table)입니다. KV 블록 관리자는 요청마다 블록 테이블을 유지하고, 각 항목에는 논리 블록에 대응하는 물리 블록 번호와 그 블록에서 채워진 위치 개수가 기록됩니다.

어텐션 커널은 블록 테이블을 보고 KV 캐시를 읽습니다. 논문 구현은 이런 비연속 접근을 빠르게 하려고 커널을 몇 가지 합쳤습니다. 각 Transformer 층에서 새로 계산한 KV를 블록 단위로 나누고, 블록 읽기에 맞는 메모리 배치로 바꾼 뒤, 블록 테이블이 가리키는 위치에 쓰는 작업을 커널 하나로 처리합니다. 어텐션 쪽도 블록 테이블을 따라 KV를 읽으면서 바로 어텐션을 계산하고, 메모리 접근을 병합(coalesced access)하려고 블록 하나를 GPU 워프 하나가 읽게 했습니다. 이렇게 매핑 계층이 공유 관계를 감추기 때문에 모델과 커널은 시퀀스마다 물리 블록 ID 목록만 받습니다. 샘플링 방식이 서로 다른 요청을 같은 배치에 섞어 넣어도 커널은 그 차이를 알 필요가 없습니다.

예전 구현(`BlockSpaceManagerV1`을 쓰던 버전)의 코드를 보면 이 구조가 그대로 드러납니다. 물리 블록 객체(`PhysicalTokenBlock`)에는 장치(GPU 또는 CPU), 장치 안의 블록 번호, 블록 크기, 참조 횟수(`ref_count`)가 들어 있었습니다. 블록 테이블은 `{seq_id: List[PhysicalTokenBlock]}` 형태의 사전이었고, GPU와 CPU 블록을 각각 다른 할당자가 관리했습니다. 스케줄러는 어떤 블록을 할당하고 복사할지 계획만 세우고, 실제 데이터 이동은 캐시 엔진이 그 계획에 따라 실행했습니다.

### 온디맨드 할당과 메모리 낭비의 상한

논리 블록과 물리 블록을 분리하면 KV 캐시를 모든 위치만큼 미리 예약하지 않아도 됩니다. 물리 블록은 새 토큰이 생성되어 현재 블록이 가득 찼을 때 비로소 하나씩 할당됩니다. 기존 시스템은 시퀀스가 나중에 얼마나 길어질지 모르기 때문에 최대 길이만큼 연속 공간을 잡아 두었고, 그 결과 단편화와 과잉 예약으로 메모리의 60~80%를 낭비했다고 vLLM 저자들은 분석합니다.

블록 크기를 $$B$$, 시퀀스 길이를 $$L$$이라 하면 필요한 물리 블록 수와 시퀀스 하나에서 비는 슬롯 수는 다음과 같습니다.

$$
N_{\text{blocks}} = \left\lceil \frac{L}{B} \right\rceil, \qquad
W = B \cdot N_{\text{blocks}} - L \le B - 1
$$

낭비는 마지막 블록에서만 생기고, 아무리 커도 블록 하나보다 작습니다. vLLM 블로그는 실제 낭비가 4% 미만이라고 보고합니다. 아낀 메모리만큼 더 많은 시퀀스를 한 배치에 넣을 수 있으므로 GPU 활용률과 처리량이 함께 올라갑니다.

디코드 단계에서는 시퀀스마다 토큰이 하나씩 늘어나므로, 예전 구현의 스케줄러는 실행 중인 각 시퀀스에 빈 물리 블록을 최소 하나씩 내줄 수 있는지(`can_append_slot`)부터 확인했습니다. 내줄 수 없으면 일부 요청을 선점(preemption)해 블록을 확보해야 하는데, 그 정책은 스케줄러가 맡습니다.

### 참조 횟수와 copy-on-write

블록 테이블을 쓰면 메모리 공유도 자연스럽게 됩니다. 프로세스들이 물리 페이지를 공유하듯, 서로 다른 시퀀스가 각자의 논리 블록을 같은 물리 블록에 대응시키면 됩니다. 프롬프트 하나에서 출력을 여러 개 뽑는 병렬 샘플링이 대표적인 예입니다. 모든 샘플이 같은 프롬프트를 쓰므로 프리필(prefill) 단계에서 프롬프트 KV 캐시를 한 벌만 만들고 함께 씁니다.

![두 샘플이 프롬프트 블록을 공유하다가 마지막 블록에 쓸 때 copy-on-write로 갈라지는 병렬 샘플링 예시](/assets/images/vllm-in-practice-pagedattention-serving-from-install-to-production-1.png)

이 예시에서 두 샘플 A1과 A2의 논리 블록 0과 1은 모두 물리 블록 7과 1에 대응합니다. 물리 블록 하나를 논리 블록 여러 개가 가리킬 수 있으므로 물리 블록마다 참조 횟수를 두고, 지금은 블록 7과 1의 참조 횟수가 둘 다 2입니다. 생성 단계에 들어가면 두 샘플이 서로 다른 토큰을 뽑으므로 각자 저장할 공간이 필요해집니다. 이때 블록 단위 copy-on-write가 동작합니다.

1. A1이 마지막 논리 블록(논리 블록 1)에 쓰려고 하면, vLLM은 대응하는 물리 블록 1의 참조 횟수가 1보다 크다는 것을 확인합니다.
2. 새 물리 블록 3을 할당하고, 블록 엔진에 물리 블록 1의 내용을 블록 3으로 복사하게 한 뒤, 블록 1의 참조 횟수를 1로 줄입니다.
3. 이어서 A2가 물리 블록 1에 쓸 때는 참조 횟수가 이미 1이므로 복사 없이 바로 씁니다.

가득 찬 프롬프트 블록은 끝까지 공유되고, 복사는 아직 덜 찬 마지막 블록 하나에서만 일어납니다. 프롬프트가 길수록 절약 폭이 커집니다. copy-on-write가 만드는 블록 복사는 흩어진 블록을 대상으로 하므로 `cudaMemcpyAsync`를 블록마다 호출하면 작은 전송이 잔뜩 생깁니다. 논문 구현은 여러 블록의 복사를 커널 한 번 실행으로 묶어 이 비용을 줄였습니다. 예전 구현에서 스케줄러가 넘기던 `blocks_to_copy`, 즉 `{원본 물리 블록: [복사본 물리 블록 목록]}` 사전이 바로 이 묶음 복사의 입력이었습니다.

빔 서치는 공유 패턴이 더 복잡합니다. 프롬프트 블록뿐 아니라 후보끼리 공통으로 거쳐 온 중간 블록도 공유하고, 디코딩이 진행되면서 공유 관계가 계속 바뀝니다. 운영체제에서 fork를 거듭해 만든 프로세스 트리와 비슷합니다. 어떤 후보가 상위 $$k$$개에서 밀려나면 그 후보의 논리 블록을 해제하고 해당 물리 블록의 참조 횟수를 줄이며, 참조 횟수가 0이 된 블록은 바로 반환합니다. 기존 시스템에서는 살아남은 후보가 다른 후보의 KV 캐시 상당 부분을 복사해야 했지만, vLLM에서는 새 토큰이 기존 공유 블록 안에 들어갈 때만 블록 하나를 복사합니다.

![빔 폭 4인 빔 서치에서 후보들이 블록을 공유하고, 탈락한 후보의 블록이 해제되는 과정](/assets/images/vllm-in-practice-pagedattention-serving-from-install-to-production-2.png)

그림에서는 점선 이후 단계에서 탈락한 후보의 블록이 참조 횟수 0이 되어 해제되고, 살아남은 후보들이 앞부분 블록을 여전히 함께 가리키는 모습을 확인하면 됩니다.

저자들은 공유로 아낀 블록 수를 공유하지 않았을 때의 전체 블록 수로 나눠 메모리 절감률을 측정했습니다.

| 데이터셋 | 병렬 샘플링 메모리 절감 | 빔 서치 메모리 절감 |
| --- | --- | --- |
| Alpaca | 6.1% ~ 9.8% | 37.6% ~ 55.2% |
| ShareGPT | 16.2% ~ 30.5% | 44.3% ~ 66.3% |

빔 서치가 공유할 블록이 훨씬 많아서 절감 폭도 큽니다. 같은 실험에서 OPT-13B와 Alpaca 조합의 경우, Orca(Oracle) 대비 vLLM의 개선 폭이 기본 샘플링의 1.3배에서 빔 폭 6인 빔 서치의 2.3배로 커졌습니다.

여러 요청이 같은 시스템 프롬프트로 시작하는 경우에도 같은 장치를 씁니다. 논문은 미리 정한 공통 프리픽스의 KV를 물리 블록에 보관해 두고, 새 요청의 논리 블록을 그 블록에 대응시키되 마지막 블록만 copy-on-write로 표시하는 방식을 설명합니다. 운영체제가 공유 라이브러리를 여러 프로세스에 매핑하는 방식과 같습니다. 예전 구현(`BlockSpaceManagerV1`)에서는 프리픽스 캐싱 방식으로 물리 블록을 관리하는 `CachedBlockAllocator`가 같은 프리픽스를 가진 프롬프트끼리 물리 블록을 공유하게 했습니다.

### 블록 관리자를 직접 따라가 보기

다음 코드는 앞서 설명한 블록 테이블, 온디맨드 할당, 참조 횟수, copy-on-write를 수십 줄로 줄인 장난감 모델입니다. vLLM API가 아니라 동작을 이해하려고 만든 **예시용 시뮬레이션**이며, 블록 크기 4와 물리 블록 8개는 손으로 따라가기 쉽게 고른 값입니다.

```python
BLOCK_SIZE = 4  # 예시용으로 작게 잡은 블록 크기


class ToyBlockManager:
    def __init__(self, num_blocks):
        self.free = list(range(num_blocks))  # 빈 물리 블록 번호
        self.ref = [0] * num_blocks          # 물리 블록별 참조 횟수
        self.tables = {}                     # seq_id -> [[물리 블록 번호, 채워진 개수], ...]
        self.copies = []                     # copy-on-write로 생긴 (원본, 복사본) 쌍

    def _alloc(self):
        pid = self.free.pop(0)
        self.ref[pid] = 1
        return pid

    def add_prompt(self, seq_id, num_tokens):
        table = []
        for start in range(0, num_tokens, BLOCK_SIZE):
            table.append([self._alloc(), min(BLOCK_SIZE, num_tokens - start)])
        self.tables[seq_id] = table

    def fork(self, parent, child):
        # 같은 프롬프트를 공유하는 새 샘플: 블록 테이블만 복사하고 참조 횟수를 올립니다
        self.tables[child] = [list(entry) for entry in self.tables[parent]]
        for pid, _ in self.tables[child]:
            self.ref[pid] += 1

    def append_token(self, seq_id):
        last = self.tables[seq_id][-1]
        if last[1] == BLOCK_SIZE:
            # 마지막 블록이 가득 찼을 때만 새 물리 블록을 할당합니다
            self.tables[seq_id].append([self._alloc(), 1])
            return
        if self.ref[last[0]] > 1:
            # 공유 중인 블록에 쓰려면 먼저 복사합니다 (copy-on-write)
            new = self._alloc()
            self.copies.append((last[0], new))
            self.ref[last[0]] -= 1
            last[0] = new
        last[1] += 1

    def free_seq(self, seq_id):
        for pid, _ in self.tables.pop(seq_id):
            self.ref[pid] -= 1
            if self.ref[pid] == 0:
                self.free.append(pid)


mgr = ToyBlockManager(num_blocks=8)
mgr.add_prompt("A1", num_tokens=7)
mgr.fork("A1", "A2")
mgr.append_token("A1")
mgr.append_token("A2")
mgr.append_token("A1")

print("block tables:", mgr.tables)
print("ref counts:", mgr.ref)
print("copies:", mgr.copies)
```

손으로 따라가면 흐름은 이렇습니다. 프롬프트 7토큰은 물리 블록 0(4칸 채움)과 1(3칸 채움)에 들어가고, `fork` 뒤에는 두 블록의 참조 횟수가 모두 2가 됩니다. A1이 토큰을 추가하면 블록 1이 공유 중이므로 블록 2를 새로 할당하고, 복사 기록에 (1, 2)를 남긴 뒤 블록 1의 참조 횟수를 1로 내립니다. A2는 블록 1을 혼자 쓰게 되었으므로 복사 없이 네 번째 칸을 채웁니다. A1의 다음 토큰은 블록 2가 가득 찼으므로 블록 3을 새로 받습니다. 블록 0은 끝까지 공유되고, 복사는 한 번만 일어났습니다. 논문의 병렬 샘플링 예시와 같은 순서입니다.

### 블록 크기 트레이드오프

블록 크기는 성능에 크게 영향을 줍니다. 블록이 너무 작으면 KV 캐시를 읽고 처리할 때 GPU 병렬성을 충분히 쓰지 못합니다. 반대로 블록이 너무 크면 앞서 본 낭비 상한 $$B-1$$이 커지고, 공유도 줄어듭니다. 블록 단위로만 공유하므로 블록이 클수록 내용이 완전히 같은 블록이 드물어지고, copy-on-write가 복사해야 하는 양도 블록 하나만큼 늘어납니다.

| 블록 크기 | 장점 | 단점 |
| --- | --- | --- |
| 작은 블록 | 내부 단편화와 마지막 블록 낭비 감소, 공유 확률 증가 | KV 캐시 읽기와 처리에서 GPU 병렬성 활용 저하 |
| 큰 블록 | 블록당 처리량이 커서 GPU 활용에 유리 | 내부 단편화 증가, 공유 확률 감소 |

저자들은 고정 요청률에서 기본 샘플링으로 블록 크기를 바꿔 가며 측정했습니다. ShareGPT 트레이스에서는 16에서 128까지가 가장 좋았지만, Alpaca 트레이스에서는 16과 32만 좋았고 그보다 크면 성능이 크게 떨어졌습니다. Alpaca의 시퀀스가 짧아 블록 크기보다 짧아지는 경우가 많고, 그러면 블록 대부분이 빈 채로 남기 때문입니다. 저자들은 16이 대부분의 워크로드에서 GPU를 효율적으로 쓸 만큼 크면서 내부 단편화를 피할 만큼 작다고 판단해 vLLM의 기본 블록 크기를 16으로 정했습니다. 예전 구현의 블록 객체에서도 블록당 슬롯 수의 기본값이 16이었습니다. 블록 크기를 바꿔야 할지 판단할 때는 서비스의 입출력 길이 분포를 먼저 확인합니다. 짧은 요청이 대부분이면 큰 블록은 낭비만 늘리고, 긴 대화가 대부분이면 큰 블록의 단점이 작아집니다.

## 자동 프리픽스 캐싱

서빙 트래픽에는 앞부분이 같은 요청이 많습니다. 모든 요청이 같은 시스템 프롬프트로 시작하거나, 다중 턴 대화에서 매 요청이 이전 대화 기록 전체를 다시 보내는 경우입니다. 공통 부분의 KV 캐시를 매번 새로 계산하면 같은 내용에 프리필 연산과 메모리를 반복해서 씁니다. 앞서 본 공통 프리픽스 공유는 미리 정해 둔 프리픽스에만 적용됩니다. 자동 프리픽스 캐싱은 이 공유를 자동으로 처리합니다. 사용자가 공유할 부분을 지정하지 않아도 새 요청이 이전 요청이 남긴 블록을 재사용합니다. 아래 메커니즘은 vLLM v0.6.2 문서가 설명하는 구현 설계입니다.

### 블록 해시로 공유할 블록 찾기

KV 블록 하나는 블록 안의 토큰과 그 앞의 프리픽스 토큰만 알면 유일하게 식별됩니다. 설계는 여기서 출발합니다. 어텐션 때문에 어떤 토큰의 키와 값은 그 앞의 모든 토큰에 따라 달라지므로, 블록 내용이 같아도 앞 문맥이 다르면 다른 블록입니다. 문장 "A gentle breeze stirred the leaves as children laughed in the distance"가 블록 셋에 나뉘어 들어갔다면, 첫 블록은 "A gentle breeze stirred"만으로 식별됩니다. 세 번째 블록은 블록 토큰 "laughed in the distance"에 프리픽스 "A gentle breeze stirred the leaves as children"을 더해야 식별됩니다.

블록 크기를 $$B$$, 시퀀스 토큰을 $$t_1, t_2, \dots$$라 하면 $$i$$번째 블록의 키는 앞쪽 모든 토큰의 해시입니다.

$$
\text{key}(b_i) = \text{hash}(t_1, t_2, \dots, t_{iB})
$$

vLLM은 이 해시 값과 KV 블록을 일대일로 대응시키고, 블록 관리에 간접 참조를 한 단계 더 둡니다. 원래 각 시퀀스는 논리 블록에서 물리 블록으로 가는 블록 테이블만 유지했습니다. 이 설계에서는 논리 블록을 해시 값에 대응시키고, 모든 물리 블록을 담은 전역 해시 테이블을 따로 둡니다. 두 요청이 같은 프리픽스 블록을 만들면 해시 값이 같으므로 같은 물리 블록을 가리키고 메모리를 함께 씁니다. 각 블록을 쓰는 요청 수는 참조 횟수로 셉니다.

키에 프리픽스 전체가 들어가므로 블록 사이에 트리 구조를 유지할 필요가 없습니다. 블록이 서로 독립적이라 하나씩 따로 할당하고 해제할 수 있고, 그래서 KV 캐시를 운영체제의 일반 캐시처럼 관리할 수 있습니다. 대신 앞쪽 토큰이 하나만 달라도 그 뒤 모든 블록의 해시가 바뀌므로, 공유는 프리픽스가 처음 갈라지는 지점 앞의 블록까지만 일어납니다.

### 캐시가 가득 찼을 때의 축출 정책

블록을 해시 테이블에 남겨 두면 요청이 끝난 뒤에도 KV가 캐시로 남고, 시스템 프롬프트가 같은 새 요청은 그 KV를 재계산 없이 바로 씁니다. 공간은 한정되어 있으므로 빈 블록이 바닥나면 무엇을 내보낼지 정해야 합니다. v0.6.2의 정책은 다음 순서로 적용됩니다.

1. 빈 블록이 없으면 참조 횟수가 0인 블록, 즉 현재 어떤 요청도 쓰지 않는 블록을 축출합니다.
2. 그런 블록이 여럿이면 가장 오래전에 사용된 블록(LRU)을 먼저 축출합니다.
3. 마지막 접근 시각까지 같으면 가장 긴 프리픽스의 끝에 있는 블록, 즉 앞에 블록이 가장 많은 블록을 먼저 축출합니다.

세 번째 규칙 덕분에 공통 프리픽스의 앞부분은 오래 남고, 갈라진 뒤쪽 꼬리부터 정리됩니다. 이 정책을 전체 어텐션(full attention) 모델에 적용하면 SGLang의 RadixAttention과 정확히 같은 정책이 됩니다. RadixAttention은 프리픽스 트리에서 참조 횟수가 0이고 가장 오래전에 사용된 리프 노드부터 축출하는데, vLLM은 트리 없이 해시 테이블만으로 같은 결과를 냅니다.

해시 기반 관리는 확장도 쉽습니다. 다중 LoRA 서빙에서는 블록 해시에 요청이 쓰는 LoRA ID를 넣으면 모든 어댑터에서 캐싱이 되고, 어댑터별 블록을 함께 관리해 구현이 단순해지고 전체 적중률과 효율이 올라갑니다. 멀티모달 입력에는 이미지에 지각 해싱(perceptual hashing)을 쓰는 식으로 모달리티마다 다른 해시 방식을 쓸 수 있습니다.

### 해시 키와 해시 알고리즘

해시 값이 같으면 같은 물리 블록을 쓰므로 해시 함수의 성질이 곧 캐시의 정확성을 정합니다. 내용이 다른 두 프리픽스의 해시가 충돌하면 한 요청이 다른 요청의 KV를 그대로 받아 쓰고, 오류 없이 엉뚱한 출력이 나옵니다. 일반적으로 해시 함수는 계산 비용과 충돌 내성 사이에서 차이가 납니다.

### 캐시 적중을 높이는 요청 구성

해시 키가 프리픽스 전체에 묶여 있으므로 적중률은 요청 구성에서 크게 갈립니다. 요청마다 같은 내용은 앞에, 달라지는 내용은 뒤에 둡니다. 다음 코드는 같은 정보를 담은 두 구성을 비교합니다.

```python
from datetime import datetime, timezone

SYSTEM_PROMPT = (
    "You are a helpful assistant for an internal engineering wiki. "
    "Answer concisely and cite the relevant page title when possible."
)


def build_bad(question):
    # 나쁜 구성: 요청마다 바뀌는 시각이 시스템 프롬프트 맨 앞에 옵니다.
    # 시각 값이 달라지는 블록부터 해시가 바뀌므로, 그 뒤에 오는 시스템 프롬프트 블록은 요청 사이에 공유되지 않습니다.
    now = datetime.now(timezone.utc).isoformat()
    return [
        {"role": "system", "content": f"Current time: {now}\n{SYSTEM_PROMPT}"},
        {"role": "user", "content": question},
    ]


def build_good(question):
    # 좋은 구성: 시스템 프롬프트는 고정하고, 바뀌는 값은 사용자 메시지로 옮깁니다.
    # 시스템 프롬프트가 차지하는 블록은 모든 요청에서 해시가 같습니다.
    now = datetime.now(timezone.utc).isoformat()
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Current time: {now}\n{question}"},
    ]
```

`build_good`이 만든 메시지는 앞서 본 채팅 API에 그대로 넘기면 됩니다. 다중 턴 대화에서도 이전 턴의 메시지를 고치지 않고 그대로 이어 붙여야 앞선 턴의 블록이 적중합니다.

## 연속 배칭 스케줄러와 선점

페이징된 KV 캐시는 시퀀스를 GPU 메모리에 빽빽하게 채웁니다. 그 메모리에 어떤 요청을 언제 올리고, 공간이 모자라면 누구를 내릴지는 스케줄러가 정합니다. vLLM 처리량의 상당 부분은 블록 관리자와 스케줄러가 함께 맞물려 나옵니다. 스케줄러의 동작을 알아 두면 지연 시간이 갑자기 튀는 이유와 정책 플래그를 바꿨을 때 달라지는 점을 설명할 수 있습니다.

### 반복 단위 스케줄링

고전적인 정적 배칭은 요청 여러 개를 묶어 한꺼번에 시작하고, 묶음 안의 모든 요청이 끝나야 다음 묶음을 받습니다. LLM은 출력 길이가 요청마다 크게 다르기 때문에 이 방식은 비효율적입니다. 10토큰 만에 끝난 요청의 자리는 500토큰을 생성하는 요청이 끝날 때까지 비어 있고, 그동안 대기열의 새 요청은 기다리기만 합니다.

연속 배칭은 스케줄링 단위를 "배치"에서 "반복(iteration)", 즉 모델의 순전파 한 번으로 바꿉니다. 매 반복마다 스케줄러가 이번에 실행할 요청 집합을 다시 고릅니다. 종료 토큰을 내거나 최대 길이에 도달한 요청은 그 반복이 끝나면 곧바로 빠지고, 빈자리는 대기 중인 요청이 다음 반복부터 채웁니다. 이 아이디어는 Orca 논문(Yu et al., 2022)의 반복 단위 스케줄링에서 왔습니다. 다만 반복마다 요청을 넣고 빼려면 메모리도 그만큼 유연해야 합니다. 요청마다 최대 길이만큼 연속 공간을 잡아 두는 방식으로는 새 요청을 받을 자리가 금방 사라집니다. 앞서 설명한 온디맨드 블록 할당이 연속 배칭의 전제 조건입니다.

반복 하나를 스케줄러 입장에서 보면 질문이 둘입니다. 이번 반복에 실행할 요청은 무엇인지, 그리고 그 요청들이 새로 만들 KV를 저장할 물리 블록이 충분한지입니다. 두 번째 질문의 답이 "아니요"이면 선점이 일어납니다.

### 프리필과 디코드, 그리고 청크 프리필

요청 하나의 생애는 두 단계로 나뉩니다. 프리필 단계에서는 프롬프트 토큰 전체를 한 번의 순전파로 처리해 모든 위치의 KV를 계산하고 첫 출력 토큰을 냅니다. 토큰 수백~수천 개를 병렬로 처리하므로 연산량이 많고, GPU 연산 자원을 꽉 채웁니다. 디코드 단계에서는 반복마다 토큰을 하나씩 생성합니다. 요청 하나가 반복마다 처리하는 토큰은 하나뿐이지만, 그 토큰을 만들려면 지금까지 쌓인 KV 캐시 전체와 모델 가중치를 읽어야 해서 연산보다 메모리 대역폭에서 막힙니다. 디코드 요청을 많이 묶을수록 가중치를 한 번 읽는 비용을 여러 요청이 나눠 내므로, 연속 배칭이 디코드 처리량을 끌어올립니다.

두 단계가 한 배치에 섞이면 긴장이 생깁니다. 아주 긴 프롬프트의 프리필이 한 반복을 통째로 차지하면, 그 반복 동안 진행 중이던 다른 요청들의 디코드도 함께 늦어집니다. 사용자 입장에서는 토큰이 잘 흘러나오다가 갑자기 멈칫하는 현상으로 나타납니다. 청크 프리필(chunked prefill)은 긴 프롬프트의 프리필을 여러 조각으로 나눠 여러 반복에 걸쳐 처리하고, 각 반복에 디코드 토큰과 프리필 조각을 함께 담는 기법입니다. 반복 하나가 처리하는 토큰 수가 일정한 상한 안에 머물러서 디코드 지연이 고르게 유지됩니다. 대신 긴 프롬프트 하나만 놓고 보면 첫 토큰까지 걸리는 시간이 조금 늘어날 수 있습니다. 반복 하나가 처리하는 토큰 수의 상한은 배치 한도 인자가 정하고, 청크 프리필 관련 플래그 이름과 기본값은 설치한 버전의 도움말 키워드 검색으로 확인합니다.

### FCFS와 우선순위 정책

요청이 시스템 용량보다 많이 들어오면 스케줄러는 일부 요청을 먼저 처리해야 합니다. PagedAttention 논문의 vLLM은 모든 요청에 선착순(FCFS, first-come-first-serve) 정책을 적용합니다. 먼저 도착한 요청을 먼저 처리하고, 선점이 필요할 때는 가장 늦게 도착한 요청부터 내립니다. 공정하고 기아(starvation)가 생기지 않는다는 것이 저자들이 든 이유입니다.

현재 vLLM에서는 `--scheduling-policy`로 정책을 고릅니다. 선택지는 둘이고 기본값은 `fcfs`입니다.

| 정책 | 처리 순서 | 어울리는 상황 |
| --- | --- | --- |
| `fcfs` | 도착 순서대로 처리합니다. | 모든 요청을 동등하게 다루는 일반 서빙입니다. |
| `priority` | 요청별 우선순위 값이 낮은 요청부터 처리하고, 값이 같으면 도착 시각으로 정합니다. | 대화형 트래픽과 배치 작업이 한 서버를 함께 쓰는 경우입니다. |

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --scheduling-policy priority
```

엔진 인자는 `LLM` 클래스의 인자이기도 하므로 오프라인 추론에서도 같은 정책을 씁니다.

```python
from vllm import LLM

llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct", scheduling_policy="priority")
```

`priority`에서는 숫자가 작을수록 먼저 처리된다는 점을 클라이언트 쪽 규칙과 맞춰야 합니다. 흔히 "우선순위 1이 높다"고 말하는 습관과 방향은 같지만, 큰 값을 중요하다고 해석하는 시스템에서 옮겨 오면 정반대로 동작합니다. 또 우선순위를 쓰면 낮은 우선순위 요청은 높은 우선순위 트래픽이 계속 들어오는 동안 밀릴 수 있습니다. FCFS가 보장하던 기아 방지를 일부 포기하는 셈이므로, 배치 작업 쪽 지연을 따로 지켜봐야 합니다.

### 메모리가 바닥났을 때의 선점

LLM 서빙의 어려움은 입력 길이가 제각각이고 출력 길이를 미리 알 수 없다는 데 있습니다. 요청 수와 출력이 늘어나다 보면, 새로 생성한 토큰의 KV를 저장할 물리 블록이 모자라는 순간이 옵니다. 이때 스케줄러는 운영체제의 페이지 교체와 같은 두 질문에 답해야 합니다. 어떤 블록을 내보낼지, 그리고 내보낸 블록이 다시 필요해지면 어떻게 되살릴지입니다.

일반적인 축출 정책은 가장 먼 미래에 접근될 블록을 추측해 내보냅니다. vLLM에는 더 단순한 사실이 있습니다. 한 시퀀스의 블록은 어텐션 계산 때 언제나 함께 읽힙니다. 일부 블록만 남겨 두는 것은 의미가 없으므로 논문은 전부 아니면 전무(all-or-nothing) 축출을 씁니다. 시퀀스 하나의 블록을 모두 내보내거나 하나도 내보내지 않습니다. 빔 서치 후보처럼 한 요청에 속한 여러 시퀀스는 시퀀스 그룹으로 묶여 함께 스케줄링(gang scheduling)됩니다. 앞서 본 것처럼 이들은 블록을 공유할 수 있어서, 그룹 안의 시퀀스는 언제나 함께 선점되고 함께 다시 스케줄링됩니다.

다음은 FCFS 선점이 일어나는 순간을 손으로 따라간 **예시용 시나리오**입니다. 블록 크기 4와 물리 블록 4개는 따라가기 쉽게 고른 값입니다.

- 요청 A가 먼저 도착했고, 프롬프트 5토큰에 3토큰을 생성해 현재 8토큰입니다. 블록 2개가 꽉 차 있습니다.
- 요청 B는 나중에 도착했고, 프롬프트 4토큰에 3토큰을 생성해 현재 7토큰입니다. 블록 2개 중 마지막 블록에 빈칸이 하나 남았습니다.
- 다음 반복에서 B의 8번째 토큰은 남은 칸에 들어가지만, A의 9번째 토큰에는 새 블록이 필요합니다. 빈 블록은 0개입니다.
- 스케줄러는 가장 늦게 도착한 B를 선점하고 B의 블록 2개를 모두 반환합니다. A가 그중 1개를 받고, 1개가 빈 채로 남습니다.
- B는 대기열로 돌아갑니다. 되살리려면 블록 2개가 필요한데 빈 블록은 1개라서, A가 블록을 더 반환하거나 끝날 때까지 기다립니다.

B를 대기열로 돌려보낸 뒤 어떻게 되살리느냐에 따라 방식이 두 가지로 나뉩니다.

### 재계산과 스왑

스왑(swap)은 대부분의 가상 메모리 구현이 쓰는 고전적인 방법입니다. 운영체제가 내보낸 페이지를 디스크의 스왑 공간에 쓰듯, vLLM은 내보낸 블록을 CPU 메모리로 복사합니다. 이를 위해 GPU 블록 할당자와 별도로 CPU RAM의 물리 블록을 관리하는 CPU 블록 할당자를 둡니다. 논문 설계에서는 시퀀스를 선점해 블록을 내보내고 나면, 선점된 시퀀스가 모두 끝날 때까지 새 요청을 받지 않습니다. 실행 중인 요청이 끝나 블록이 풀리면 선점된 시퀀스의 블록을 GPU로 다시 들여와 처리를 이어 갑니다. 이 설계에서는 CPU로 나간 블록 수가 GPU의 전체 물리 블록 수를 넘지 않으므로, CPU 쪽 스왑 공간도 KV 캐시에 할당한 GPU 메모리 크기를 넘지 않습니다.

재계산(recomputation)은 선점된 시퀀스가 다시 스케줄링될 때 KV 캐시를 처음부터 다시 계산합니다. 앞의 시나리오처럼 블록을 버리기만 하므로 CPU 메모리도 복사도 필요 없습니다. 비용이 생각보다 작은 이유는 디코드 단계에서 이미 생성한 토큰을 원래 프롬프트 뒤에 붙여 새 프롬프트로 만들 수 있기 때문입니다. 그러면 모든 위치의 KV를 프리필 반복 한 번에 계산합니다. 토큰을 하나씩 생성하던 원래 경로보다 훨씬 빠르게 같은 상태로 돌아옵니다. 예시의 B라면 원래 프롬프트 4토큰에 생성한 토큰을 이어 붙인 7토큰짜리 프롬프트로 다시 들어오고, 그 프롬프트를 처리하는 첫 프리필에서 KV가 모두 복원됩니다.

| 구분 | 스왑 | 재계산 |
| --- | --- | --- |
| 선점 시 동작 | 시퀀스의 블록을 CPU 메모리로 복사합니다. | 시퀀스의 블록을 버립니다. |
| 복구 시 동작 | CPU의 블록을 GPU로 다시 복사합니다. | 프롬프트와 생성 토큰을 이어 붙여 프리필 한 번으로 KV를 다시 만듭니다. |
| 추가 자원 | KV 캐시용 GPU 메모리 크기를 넘지 않는 CPU 메모리가 필요합니다. | 추가 메모리 없이 GPU 연산을 더 씁니다. |
| 성능을 좌우하는 요인 | CPU RAM과 GPU 메모리 사이 대역폭입니다. | GPU 연산 성능입니다. |

어느 쪽이 빠른지는 하드웨어에 달려 있습니다. 논문도 두 방식의 성능이 CPU-GPU 대역폭과 GPU 연산 성능에 좌우된다고 보고, 블록 크기별로 두 방식의 오버헤드를 따로 측정했습니다. 블록이 작으면 스왑은 작은 전송을 여러 번 해야 하는 반면, 재계산은 블록 크기와 상관없이 프리필 한 번이면 됩니다. 이 차이도 판단에 들어갑니다. 이 두 방식은 PagedAttention 논문이 설명하는 설계이며, 설치한 버전이 어떤 복구 방식을 쓰고 어떤 선택지를 주는지는 그 버전의 도움말로 확인합니다.

운영에서는 선점이 일어난다는 사실 자체보다 얼마나 자주 일어나는지가 중요합니다. 선점이 드물면 메모리를 끝까지 쓰면서 처리량을 지키는 안전장치로 동작합니다. 잦으면 같은 KV를 반복해서 다시 계산하거나 복사하느라 GPU 시간을 버리고, 선점당한 요청의 지연이 크게 늘어납니다. 요청별 선점 횟수는 `request_num_preemptions` 히스토그램 계열로 수집되므로, 이 값이 꾸준히 올라가면 KV 캐시 공간에 비해 동시에 받는 요청이 많다는 신호로 보고 메모리와 배치 한도를 다시 조정합니다.

## 메모리와 배치 설정 튜닝

vLLM은 모델 가중치와 순전파 작업 공간을 잡고 남은 GPU 메모리를 KV 캐시 블록으로 채웁니다. KV 캐시가 작으면 동시에 올릴 수 있는 시퀀스가 줄고 선점이 잦아집니다. 메모리 설정은 KV 캐시 크기를 정하고, 배치 설정은 그 공간을 한 반복에 얼마나 채울지 정합니다. 두 설정 모두 엔진 인자라서 `vllm serve` 플래그로 줄 수도 있고, `LLM` 클래스의 키워드 인자로 줄 수도 있습니다.

### GPU 메모리 예산: `--gpu-memory-utilization`

`--gpu-memory-utilization`은 모델 실행기가 쓸 GPU 메모리 비율이며 0에서 1 사이 값입니다. 0.5를 주면 GPU 메모리의 50%를 씁니다. 이 한도는 인스턴스 단위라서 같은 GPU의 다른 vLLM 인스턴스를 고려하지 않습니다. GPU 하나에 인스턴스 둘을 띄운다면 각각 0.5처럼 직접 나눠 줍니다. GPU가 아닌 장치에서는 해당 장치 메모리 비율을 뜻하며, `--device-memory-utilization`이라는 별칭도 있습니다.

```bash
# 같은 GPU에 두 인스턴스를 띄울 때: 인스턴스마다 절반씩 씁니다
vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000 --gpu-memory-utilization 0.5
vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8001 --gpu-memory-utilization 0.5
```

값을 주지 않으면 확장형 KV 캐시(extensible KV cache)를 쓸 때 기본값이 1.0이고, 쓰지 않을 때는 0.92입니다. 확장형 KV 캐시는 프로파일링 추정치 대신 CUDA 그래프 캡처 뒤 실제로 남은 메모리로 캐시 크기를 정합니다. V2 모델 실행기를 쓰는 CUDA와 ROCm에서 기본으로 켜지고, `--kv-cache-memory-bytes`를 주면 켜지지 않습니다. 드라이버가 VMM을 지원하지 않으면 추정치 기반 크기 결정으로 돌아갑니다.

### KV 캐시 크기를 직접 정하기: `--kv-cache-memory-bytes`

비율로 정하면 GPU 종류가 바뀔 때 절대량도 바뀝니다. 여러 GPU에서 같은 용량을 보장하려면 `--kv-cache-memory-bytes`로 GPU당 KV 캐시 크기를 바이트 단위로 지정합니다. 기본값 `None`에서는 `gpu_memory_utilization`으로 크기를 추론하고, 값을 주면 `gpu_memory_utilization`은 무시됩니다. CLI에서는 `1k`, `2M` 같은 접미사를 받으며 소문자는 1000, 대문자는 1024의 배수입니다.

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --kv-cache-memory-bytes 8G
```

```python
from vllm import LLM

llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct", kv_cache_memory_bytes=8 * 1024**3)
```

### 반복당 배치 한도: `--max-num-seqs`와 `--max-num-batched-tokens`

`--max-num-seqs`는 한 반복에서 처리할 최대 시퀀스 수이고, `--max-num-batched-tokens`는 한 반복에서 처리할 최대 토큰 수입니다. 토큰 한도는 한 반복의 상한이므로 청크 프리필 조각 크기의 상한도 됩니다. 도움말의 기본값은 테스트 편의용이고, 실제 값은 `EngineArgs.create_engine_config`에서 정해집니다.

앞서 본 청크 프리필과 선점 동작에서 다음 관계가 따라 나옵니다.

- `--max-num-batched-tokens`를 낮추면 반복 하나에 들어가는 프리필 토큰이 줄어 디코드 지연이 고르게 유지되지만, 긴 프롬프트의 첫 토큰은 늦어집니다.
- `--max-num-seqs`를 낮추면 KV 캐시를 함께 차지하는 시퀀스가 줄어 선점이 줄지만, 디코드 처리량도 낮아집니다. 효과는 `request_num_preemptions` 히스토그램으로 확인합니다.

```bash
# 예시 값입니다. 하드웨어와 트래픽에 맞춰 조정합니다
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --max-num-seqs 128 \
  --max-num-batched-tokens 8k
```

### KV 캐시 dtype

표준 Transformer에서 토큰 하나의 KV 캐시 크기는 층 수 $$n_{\text{layers}}$$, KV 헤드 수 $$n_{\text{kv}}$$, 헤드 차원 $$d_{\text{head}}$$, 원소당 바이트 수 $$s$$로 정해지며, 앞의 2는 키와 값입니다.

$$
M_{\text{token}} = 2 \cdot n_{\text{layers}} \cdot n_{\text{kv}} \cdot d_{\text{head}} \cdot s
$$

$$s$$를 2바이트(BF16)에서 1바이트(FP8)로 줄이면 같은 예산에 토큰이 두 배 들어갑니다. 대신 정밀도가 낮아지므로 품질 영향은 모델마다 평가합니다.

`--kv-cache-dtype`의 기본값 `auto`는 모델 dtype을 따르고, `float16`이나 `bfloat16`을 명시하면 모델 dtype과 같아야 합니다. CUDA 11.8 이상은 `fp8`(=`fp8_e4m3`)과 `fp8_e5m2`를, ROCm은 `fp8`(=`fp8_e4m3`)을, Intel Gaudi는 `fp8_inc`를 지원합니다. DeepSeekV3.2처럼 기본이 fp8인 모델에 `bfloat16`을 주면 KV 캐시를 bfloat16으로 저장합니다. 기본이 fp8이 아닌 모델에서는 이 지정이 유효하지 않습니다. 정수·NVFP4 계열 등 다른 선택지는 설치한 버전의 도움말로 확인하고, 특정 층을 양자화에서 빼려면 `--kv-cache-dtype-skip-layers`에 층 번호나 `sliding_window` 같은 어텐션 종류 이름을 줍니다.

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --kv-cache-dtype fp8
```

### CPU 오프로딩: `--cpu-offload-gb`

가중치가 GPU에 다 들어가지 않으면 `--cpu-offload-gb`로 GPU당 지정한 GiB만큼을 CPU 메모리로 내립니다. 기본값 0은 오프로딩하지 않습니다. 24GB GPU에 10을 주면 34GB GPU처럼 다뤄서, BF16 가중치로 최소 26GB가 필요한 13B 모델을 올립니다. UVA(Unified Virtual Addressing)로 CPU 메모리를 제로 카피로 읽지만, 순전파마다 모델 일부를 CPU에서 가져오므로 빠른 CPU-GPU 연결이 필요합니다.

```bash
vllm serve meta-llama/Llama-2-13b-hf --cpu-offload-gb 10
```

### 설정별 효과 정리

| 설정 | 조정 대상 | 올리거나 켤 때 얻는 것 | 대가 |
| --- | --- | --- | --- |
| `--gpu-memory-utilization` | 인스턴스가 쓰는 GPU 메모리 비율 | KV 캐시 공간 증가 | 다른 프로세스와 인스턴스의 여유 감소 |
| `--kv-cache-memory-bytes` | GPU당 KV 캐시 절대 크기 | GPU 종류와 무관한 일정 용량 | 비율 설정 무시와 확장형 KV 캐시 비활성화 |
| `--max-num-seqs` | 반복당 시퀀스 수 | 디코드 처리량 증가 | 선점 증가 가능성 |
| `--max-num-batched-tokens` | 반복당 토큰 수 | 프리필 처리 속도 향상 | 디코드 지연 변동 증가 |
| `--kv-cache-dtype` | KV 캐시 원소 정밀도 | 같은 메모리에 담는 토큰 수 증가 | 정밀도 손실 |
| `--cpu-offload-gb` | CPU로 내리는 가중치 크기 | 더 큰 모델 적재 | 순전파마다 CPU-GPU 전송 |

KV 캐시 크기를 먼저 확정한 뒤 배치 한도를 맞추면, 지연이나 선점이 늘었을 때 원인이 메모리인지 스케줄링인지 가르기 쉽습니다.

## 어텐션 백엔드 선택

어텐션 계산을 맡는 커널 구현을 어텐션 백엔드(attention backend)라고 합니다. vLLM은 플랫폼과 가속기 아키텍처마다 여러 백엔드를 지원하고, 시스템과 모델 사양에 호환되는 백엔드 가운데 가장 성능이 좋은 것을 자동으로 고릅니다.

원하는 백엔드로 직접 지정하려면 `--attention-backend` 인자를 씁니다.

```bash
# 온라인 서빙
vllm serve Qwen/Qwen2.5-1.5B-Instruct --attention-backend FLASH_ATTN

# 오프라인 추론
python script.py --attention-backend FLASHINFER
```

선택지는 플랫폼마다 다릅니다. 다음은 그 일부입니다.

| 플랫폼 | 백엔드 선택지 |
| --- | --- |
| NVIDIA CUDA | `FLASH_ATTN`, `FLASHINFER` |
| AMD ROCm | `TRITON_ATTN`, `ROCM_ATTN`, `ROCM_AITER_FA`, `ROCM_AITER_UNIFIED_ATTN`, `TRITON_MLA`, `ROCM_AITER_MLA`, `ROCM_AITER_TRITON_MLA` |
| Intel XPU | `FLASH_ATTN`, `TRITON_ATTN`, `TRITON_MLA`, `XPU_MLA_SPARSE`, `TORCH_SDPA`, `TURBOQUANT` |

FlashInfer는 vLLM과 함께 설치되지만, 미리 컴파일된 커널은 함께 설치되지 않습니다. vLLM을 설치하거나 업그레이드한 뒤 `vllm download-kernels`를 실행해 두면 기동 시점에 커널을 내려받아 컴파일하지 않습니다.

## 다중 GPU 분산 실행

모델 가중치가 GPU 하나에 들어가지 않거나, 들어가더라도 KV 캐시를 둘 자리가 모자라면 GPU를 여러 장 묶어 실행합니다. 묶는 방식은 크게 셋이며, vLLM에서는 모두 엔진 인자로 고릅니다.

### 텐서·파이프라인·데이터 병렬

세 방식은 무엇을 나누느냐가 다릅니다.

- **텐서 병렬**: 각 층의 가중치 행렬을 여러 GPU에 쪼개 둡니다. 층마다 GPU들이 부분 결과를 주고받아 합치므로 통신이 잦고, 그만큼 빠른 연결이 필요합니다. 대신 모든 GPU가 모든 층의 계산에 참여하므로 요청 하나의 지연이 줄어듭니다.
- **파이프라인 병렬(pipeline parallelism)**: 층을 구간별로 나눠 GPU마다 다른 구간을 맡깁니다. 경계에서 활성값만 넘기므로 통신량이 적어 GPU 사이 연결이 느린 환경에 맞습니다. 다만 앞 구간이 끝나야 뒤 구간이 시작되므로, 처리할 요청이 적으면 뒤 구간 GPU가 놀게 됩니다. 그래서 요청 하나의 지연을 줄이는 효과는 텐서 병렬보다 작습니다.
- **데이터 병렬(data parallelism)**: 모델 전체를 복제해 각 복제본이 서로 다른 요청을 처리합니다. 모델은 이미 들어가고 처리량만 늘리고 싶을 때 씁니다.

텐서 병렬 크기와 파이프라인 병렬 크기를 곱한 값이 모델 복제본 하나가 차지하는 GPU 수(world size)입니다. 노드 안 GPU 사이 연결은 대개 빠르므로, 앞의 통신량 차이를 따르면 한 노드에 GPU가 여러 장일 때는 텐서 병렬부터 적용하는 편이 자연스럽습니다. 서버에서는 `-tp`, 오프라인 추론에서는 같은 엔진 인자인 `tensor_parallel_size`를 씁니다.

```bash
# BF16 가중치만 26GB 이상인 13B 모델을 GPU 2장에 나눠 올립니다
vllm serve facebook/opt-13b -tp 2
```

```python
from vllm import LLM

llm = LLM(model="facebook/opt-13b", tensor_parallel_size=2)
```

파이프라인 병렬은 `--pipeline-parallel-size`로 켜며, 텐서 병렬과 함께 쓸 수 있습니다. 다음은 Intel XPU에서의 실행 예로, 파이프라인 2단계 × 텐서 병렬 8로 GPU 16장을 씁니다. XPU는 텐서 병렬 추론과 서빙을 지원하고, 파이프라인 병렬은 온라인 서빙의 베타 기능으로 단일 노드에서 `mp` 백엔드로만 지원합니다.

```bash
vllm serve facebook/opt-13b \
     --dtype=bfloat16 \
     --max_model_len=1024 \
     --distributed-executor-backend=mp \
     --pipeline-parallel-size=2 \
     -tp=8
```

### 실행기 백엔드: `mp`와 `ray`

병렬 실행에서는 여러 워커 프로세스가 뜹니다. 이 워커들을 누가 띄우고 관리할지는 `--distributed-executor-backend`가 정하며, 선택지는 `mp`, `ray`, `uni`, `external_launcher`입니다. 텐서 병렬 크기와 파이프라인 병렬 크기의 곱이 사용 가능한 GPU 수 이하이면 vLLM은 `mp`(멀티프로세싱)를 골라 한 호스트 안에서 처리하고, GPU가 모자라면 오류를 냅니다. `ray`는 자동으로 선택되지 않으므로 직접 지정해야 합니다.

```bash
vllm serve facebook/opt-13b -tp 8 --pipeline-parallel-size 2 \
  --distributed-executor-backend ray
```

Intel XPU에서는 실행 중인 Ray 인스턴스가 없을 때 vLLM이 world size만큼 GPU를 잡은 인스턴스를 자동으로 띄우며, 실행 전에 저장소의 `examples/ray_serving/run_cluster.sh` 도우미 스크립트로 Ray 클러스터를 미리 구성해 두기를 권장합니다. TPU 플랫폼은 분산 추론에 Ray만 지원합니다.

쓸 GPU를 고르는 방법도 다릅니다. `mp`에서는 `--device-ids "2,3,5,7"`처럼 물리 GPU ID나 UUID를 쉼표로 나열합니다. `CUDA_VISIBLE_DEVICES`를 건드리지 않으므로 GPU-NIC 친화성이나 DeepGEMM이 보는 전체 GPU 토폴로지가 그대로 유지됩니다. Ray 실행기에서는 이 인자가 효과가 없으므로 Ray 배치 그룹(placement group)으로 GPU를 고릅니다. 워커를 GPU에 가까운 NUMA 노드의 CPU에 고정하는 `--numa-bind`도 있지만, Ray와 외부 런처는 자동 바인딩에서 빠집니다.

### 데이터 병렬 배포와 부하 분산

복제본 수는 엔진 설정의 `data_parallel_size`로 정하고, 복제본을 띄우는 백엔드는 `--data-parallel-backend`(`-dpb`)로 고릅니다. 기본값은 `mp`이고 `ray`도 쓸 수 있습니다. 기본으로는 vLLM이 데이터 병렬 랭크 사이에서 요청을 직접 나누며, 온라인 서빙에서 `data_parallel_size`가 0보다 크면 부하 분산을 외부와 나누는 모드 두 가지를 더 고를 수 있습니다.

| 모드 | 인자 | 동작 | 적합한 경우 |
| --- | --- | --- | --- |
| 하이브리드 | `--data-parallel-hybrid-lb` (`-dph`) | 노드마다 엔진과 API 서버를 띄워 노드 안 랭크는 vLLM이, 노드 사이는 외부 부하 분산기가 나눔 | 여러 노드에 걸친 데이터 병렬 |
| 외부 | `--data-parallel-external-lb` (`-dpe`) | 랭크마다 독립적으로 띄우고 부하 분산은 모두 외부에 맡김 | Kubernetes에서 랭크마다 파드 하나를 두는 wide-EP 구성 |

하이브리드 모드는 `--data-parallel-start-rank`와 함께 명시적으로 켭니다. 외부 모드는 `vllm serve`에 `--data-parallel-rank`를 명시하면 자동으로 켜집니다. 랭크마다 따로 띄우고 부하 분산을 외부에 맡기는 이 구성은 MoE 모델에서만 지원하므로, MoE가 아닌 모델을 같은 방식으로 운영하려면 `--data-parallel-*` 인자 없이 독립된 vLLM 인스턴스를 여러 개 띄웁니다.

### 컨테이너의 공유 메모리 설정

vLLM이 쓰는 PyTorch는 내부적으로 프로세스 사이 데이터를 공유 메모리로 주고받고, 텐서 병렬 추론에서 특히 이 경로를 많이 씁니다. 그래서 컨테이너에는 `--ipc=host`로 호스트의 공유 메모리를 쓰게 하거나 `--shm-size`로 공유 메모리 크기를 직접 늘려 줍니다. 앞서 본 공식 이미지 실행 명령은 `--ipc=host`를 썼으므로, 호스트 IPC 네임스페이스를 공유하기 어려운 환경이라면 다음처럼 크기를 지정합니다.

```bash
# --shm-size 값은 예시입니다. 워크로드에 맞춰 정합니다
docker run --runtime nvidia --gpus all --shm-size=16g \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    --model facebook/opt-13b \
    -tp 2
```

큰 이미지를 다루는 멀티모달 서빙에서 텐서 병렬을 쓴다면 실험 기능인 `--enable-shm-tensor-arena`도 고려할 만합니다. 엔진이 워커들에 큰 CPU 텐서(예: `pixel_values`)를 보낼 때 읽는 쪽마다 복사본을 보내는 대신 공유 메모리 영역으로 복사 없이 넘깁니다. 기본값은 꺼져 있고, 모든 큐 읽기 쪽이 같은 노드에 있을 때만 동작합니다. 이 기능도 `/dev/shm`에 자리를 예약하므로, 켤 때는 공유 메모리를 넉넉히 잡아 둡니다.

## CPU 백엔드 운영

GPU가 없는 서버에서도 vLLM으로 기본적인 추론과 서빙을 할 수 있습니다. x86 CPU는 FP32, FP16, BF16을 지원합니다. Arm CPU는 NEON이 필요하고 같은 세 가지 dtype을 지원합니다. x86에서는 `avx512f`를 권장하며, `avx2`만 있으면 일부 기능이 제한됩니다. CPU 플래그는 `lscpu`로 확인합니다. 휠로 설치했다면 실행 전에 런타임 라이브러리를 `LD_PRELOAD`에 올립니다. x86 사전 빌드 휠은 Intel OpenMP를, x86 소스 빌드는 TCMalloc과 Intel OpenMP를, Arm은 TCMalloc을 올립니다. CPU 성능은 스레드를 어느 코어에 묶는지에 크게 좌우되므로, KV 캐시 크기와 스레드 배치는 직접 정해 두는 편이 안전합니다.

### CPU KV 캐시 공간

레거시 환경 변수 `VLLM_CPU_KVCACHE_SPACE`는 KV 캐시 크기를 GiB 단위로 정합니다. 40을 주면 KV 캐시에 40GiB를 씁니다. 기본값은 0이고, 0이면 이 변수로 KV 캐시 크기를 덮어쓰지 않습니다. 공간을 늘리면 동시에 처리하는 요청 수와 컨텍스트 길이가 늘어납니다. 새 배포에서는 앞서 설명한 `--kv-cache-memory-bytes`로 크기를 정합니다. 아래 예제는 아직 레거시 변수를 씁니다.

크기를 정할 때는 서버 전체가 아니라 NUMA 노드 하나의 메모리를 기준으로 봅니다. 텐서 병렬 랭크 하나가 쓰는 메모리는 가중치 샤드 크기와 KV 캐시 공간의 합입니다. 이 합이 NUMA 노드 하나의 용량을 넘으면 해당 워커가 메모리 부족으로 `exitcode 9`를 내며 종료됩니다. 그러니 랭크당 예산을 노드 용량 안에서 계산해 둡니다.

### OpenMP 스레드 바인딩

CPU 추론의 행렬 연산은 OpenMP 스레드가 나눠 맡습니다. 스레드가 코어 사이를 옮겨 다니거나 하이퍼스레드 두 개가 물리 코어 하나를 나눠 쓰면 캐시와 메모리 대역폭을 서로 빼앗습니다. 그래서 `VLLM_CPU_OMP_THREADS_BIND`로 스레드를 코어에 고정합니다. 값의 형태는 셋입니다.

- `auto`(기본값): 랭크마다 그 랭크의 NUMA 노드 안 코어에 스레드를 묶습니다. 이상적으로는 스레드 하나가 전용 물리 코어 하나를 씁니다.
- CPU ID 목록: `0-31`은 0~31번 코어에 스레드 32개를 묶습니다. `0-31|32-63`이면 텐서 병렬 프로세스가 둘이 되고, rank0은 0~31번, rank1은 32~63번 코어를 씁니다.
- `nobind`: 개별 코어 바인딩을 끄고 사용자가 정한 OpenMP 변수를 따릅니다. 스레드 수는 `OMP_NUM_THREADS`가 정합니다.

대부분은 `auto`를 씁니다. `VLLM_CPU_NUM_OF_RESERVED_CPU`는 OpenMP 스레드에 주지 않고 남겨 둘 코어 수이며 `auto`일 때만 효과가 있습니다. x86, Arm, RISC-V는 기본으로 `local_world_size`개를 남기고, KV 전송을 켜면 로컬 랭크마다 하나씩 더 남깁니다. 온라인 서빙에서는 프레임워크 프로세스가 추론 스레드와 코어를 다투지 않도록 1~2개 코어를 비워 두기를 권장하며, `world_size`가 1이면 프런트엔드 프로세스용 CPU 하나를 직접 남깁니다. 물리 코어가 32개인 서버라면 다음 둘 중 하나를 씁니다.

```bash
# 방법 1: 31번 코어를 프레임워크용으로 남기고 0~30번에 직접 바인딩
export VLLM_CPU_KVCACHE_SPACE=40
export VLLM_CPU_OMP_THREADS_BIND=0-30
vllm serve facebook/opt-125m --dtype=bfloat16

# 방법 2: 자동 바인딩을 쓰면서 코어 하나를 남김
export VLLM_CPU_KVCACHE_SPACE=40
export VLLM_CPU_NUM_OF_RESERVED_CPU=1
vllm serve facebook/opt-125m --dtype=bfloat16
```

자동 바인딩에서 성능 문제나 예상과 다른 바인딩이 보이면 목록을 직접 지정합니다. 하이퍼스레딩 서버에서는 `lscpu -e`의 `CPU` 열(논리 코어)과 `CORE` 열(물리 코어)을 대조합니다. 0~7번과 8~15번이 같은 물리 코어 8개를 나눠 쓴다면 `0-7`처럼 한쪽만 묶습니다. 바인딩이 실제로 적용됐는지는 벤치마크를 돌리며 `htop`으로 코어 사용률을 보고 확인합니다. GPU용 `--numa-bind`, `--numa-bind-nodes`, `--numa-bind-cpus`는 CPU 워커의 코어 배치에는 영향을 주지 않습니다.

### NUMA와 병렬 구성

멀티 소켓 서버에서 텐서 병렬이나 파이프라인 병렬을 켜면 NUMA 노드 하나가 랭크 하나로 취급됩니다. 원격 메모리 접근을 피하려면 한 랭크의 코어를 같은 NUMA 노드 안에서 고릅니다. `tensor-parallel-size`는 NUMA 노드 수에 맞추기를 권장하며, 현재 릴리스는 6을 지원하지 않습니다.

```bash
lscpu | grep "NUMA node(s):" | awk '{print $3}'
```

`CPU_VISIBLE_MEMORY_NODES`는 `CUDA_VISIBLE_DEVICES`처럼 CPU 워커가 볼 NUMA 메모리 노드를 정합니다. 노드를 가리거나 바인딩 순서를 바꿀 때 쓰며, `auto` 바인딩일 때만 효과가 있습니다. 소켓과 메모리 노드가 충분하면 데이터·텐서·파이프라인 병렬을 함께 쓰기를 권장합니다.

컨테이너에서는 vLLM이 쓰는 NUMA 시스템 호출(`get_mempolicy`, `migrate_pages`)이 기본 seccomp와 권한 설정에 막혀 `get_mempolicy: Operation not permitted` 경고가 뜨기도 합니다. 동작에는 문제가 없지만 NUMA 메모리 바인딩과 이동 최적화가 적용되지 않아 성능이 떨어질 수 있습니다. `--cap-add SYS_NICE`는 `get_mempolicy` 권한 오류를, `--security-opt seccomp=unconfined`는 `migrate_pages` 차단을 풉니다. seccomp를 통째로 풀 수 없다면 Docker 기본 프로파일에 `migrate_pages`만 허용한 프로파일을 쓰고, `--privileged=true`는 권한 범위가 넓어 권장하지 않습니다.

```bash
docker run --rm \
    --security-opt seccomp=unconfined \
    --cap-add SYS_NICE \
    --shm-size=4g \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    -e VLLM_CPU_KVCACHE_SPACE=<KV cache space> \
    -e VLLM_CPU_OMP_THREADS_BIND=<CPU cores for inference> \
    vllm/vllm-openai-cpu:latest-x86_64 \
    meta-llama/Llama-3.2-1B-Instruct \
    --dtype=bfloat16
```

### dtype 선택

CPU 백엔드는 모델의 기본 설정을 dtype으로 씁니다. torch CPU의 float16 지원이 불안정하므로 성능이나 정확도 문제가 보이면 `--dtype=bfloat16`을 명시합니다. AMD Zen CPU에서 `ZenCpuPlatform`이 선택되면 float16에 AVX512-FP16이 필요하고, 이 명령어는 Venice(6세대 EPYC)부터 들어갑니다. Turin(5세대 EPYC) 이하는 bfloat16과 float32만 받으며, float16으로 선언된 모델은 로드할 때 경고와 함께 bfloat16으로 바뀝니다. 지원 목록의 첫 항목이 bfloat16이라 `--dtype auto`도 bfloat16이 됩니다.

### 배치 기본값

CPU에서는 배치 한도 기본값이 `world_size`에 비례합니다. `--block-size`는 32의 배수로 주며 기본값은 128입니다.

| 인자 | 오프라인 추론 기본값 | 온라인 서빙 기본값 | 주로 영향을 주는 지표 |
| --- | --- | --- | --- |
| `--max-num-batched-tokens` | `4096 * world_size` | `2048 * world_size` | 첫 토큰 성능 |
| `--max-num-seqs` | `256 * world_size` | `128 * world_size` | 출력 토큰 성능 |

배치를 키우면 처리량이 오르고 줄이면 지연이 짧아지므로, 기본값에서 출발해 플랫폼에 맞게 조정합니다. 조정은 스레드 바인딩과 KV 캐시 공간이 제대로 적용됐는지 확인한 뒤에 합니다. 바인딩이 어긋나 있으면 배치 값을 바꾼 결과를 해석하기 어렵습니다.

## Kubernetes와 Helm 배포

Helm은 Kubernetes용 패키지 관리자입니다. 변수 값만 덮어쓰면 같은 구성을 여러 네임스페이스에 서로 다른 설정으로 배포할 수 있습니다. vLLM 저장소의 `examples/deployment/chart-helm`에 공식 예제 차트가 있습니다. 클러스터에는 `k8s-device-plugin`이 미리 설치되어 있어야 합니다.

### 공식 차트 설치와 제거

차트 디렉터리에서 릴리스 이름을 `test-vllm`으로 정해 설치합니다. 모델은 기본적으로 S3에서 받으므로 접속 정보를 시크릿 값으로 넘깁니다.

```bash
helm upgrade --install --create-namespace \
  --namespace=ns-vllm test-vllm . \
  -f values.yaml \
  --set secrets.s3endpoint=$ACCESS_POINT \
  --set secrets.s3bucketname=$BUCKET \
  --set secrets.s3accesskeyid=$ACCESS_KEY \
  --set secrets.s3accesskey=$SECRET_KEY
```

`helm upgrade --install`은 릴리스가 없으면 설치하고, 있으면 갱신합니다. 그래서 값 파일을 고친 뒤에도 같은 명령을 다시 실행하면 됩니다. 제거할 때는 `helm uninstall`을 씁니다.

```bash
helm uninstall test-vllm --namespace=ns-vllm
```

이 명령은 차트가 만든 Kubernetes 구성 요소를 **영구 볼륨까지 포함해** 모두 지우고 릴리스를 삭제합니다. PVC에 받아 둔 모델도 함께 사라지므로, 다시 설치하면 모델을 처음부터 받습니다. 큰 모델이라면 제거하기 전에 이 비용을 따져 봅니다.

### 모델 다운로드와 초기화 컨테이너

vLLM 컨테이너가 뜨기 전에 모델 파일이 볼륨에 들어가 있어야 합니다. `extraInit.modelDownload.enabled`의 기본값은 `true`이고, 켜져 있으면 차트가 모델 다운로드 작업과 대기 컨테이너를 만듭니다. 다운로드 이미지 기본값은 `amazon/aws-cli`의 `2.6.4` 태그입니다. S3 안의 모델 위치는 `extraInit.s3modelpath`로, PVC 크기는 `extraInit.pvcStorage`로 정합니다. `pvcStorage` 기본값 `"1Gi"`는 예제 모델용이므로 실제 모델 크기에 맞춰 늘립니다.

서버 실행 명령은 `image.command`에 들어 있습니다. 기본값은 `vllm serve /data/ --served-model-name opt-125m --host 0.0.0.0 --port 8000`입니다. 볼륨의 로컬 경로 `/data/`를 서빙하고, API에는 `--served-model-name`에 준 이름으로 노출합니다. 그래서 클라이언트 요청의 `model` 필드에도 이 이름을 넣습니다. 컨테이너 포트 기본값은 8000이고 Service 포트 기본값은 80입니다. 모델을 바꿀 때는 `s3modelpath`와 `--served-model-name`을 함께 고칩니다.

llm-d처럼 모델은 내려받지 않고 사용자 정의 사이드카만 붙이는 구성도 있습니다. 이때는 `modelDownload.enabled`를 `false`로 두고 `extraInit.initContainers`에 컨테이너를 직접 넣습니다. 다운로드를 켜 둔 상태라면 사용자 정의 초기화 컨테이너가 다운로드 컨테이너 뒤에 붙습니다.

```yaml
extraInit:
  modelDownload:
    enabled: false
  initContainers:
    - name: llm-d-routing-proxy
      image: ghcr.io/llm-d/llm-d-routing-sidecar:v0.2.0
      imagePullPolicy: IfNotPresent
      ports:
        - containerPort: 8080
          name: proxy
      securityContext:
        runAsUser: 1000
      restartPolicy: Always
  pvcStorage: "10Gi"
```

### 프로브와 리소스

활성 프로브(liveness probe)와 준비 프로브(readiness probe)는 둘 다 8000번 포트의 `/health`를 HTTP로 호출합니다. 두 프로브는 같은 엔드포인트를 호출하며, 시작 지연과 주기, 그리고 실패했을 때의 처리가 다릅니다.

| 프로브 | 시작 지연(`initialDelaySeconds`) | 주기(`periodSeconds`) | 실패 허용(`failureThreshold`) | 연속 실패 시 판정 |
| --- | --- | --- | --- | --- |
| `readinessProbe` | 5초 | 5초 | 3회 | 준비되지 않음 |
| `livenessProbe` | 15초 | 10초 | 3회 | 살아 있지 않음 |

준비 프로브가 실패하면 파드가 Service의 트래픽 대상에서 빠집니다. 활성 프로브가 실패하면 kubelet이 컨테이너를 재시작합니다. 큰 모델은 가중치 로딩과 KV 캐시 할당에 시간이 걸립니다. 기본값을 그대로 두면 기동이 끝나기 전에 활성 프로브가 실패해 재시작이 반복될 수 있습니다. 실제 기동 시간을 재 보고 `livenessProbe.initialDelaySeconds`나 `failureThreshold`를 늘리기를 권장합니다.

`resources` 기본값은 요청과 한도 모두 CPU 4개, 메모리 `16Gi`, `nvidia.com/gpu` 1개입니다. 여기서 `memory`는 CPU 메모리이고, GPU 메모리 배분은 vLLM 엔진 인자가 맡습니다. 텐서 병렬을 쓴다면 `nvidia.com/gpu` 개수와 `image.command`의 `-tp` 값을 함께 맞춥니다. 자동 확장(`autoscaling`)은 기본으로 꺼져 있습니다. 켜면 복제본 1~100개 사이에서 CPU 사용률 80%를 목표로 조정합니다. 기준이 CPU 사용률이므로, 이 지표가 GPU 추론 부하를 제대로 반영하는지 확인한 뒤에 켭니다. 다음은 GPU 2장에 텐서 병렬 2로 띄우는 덮어쓰기 예입니다.

```yaml
# values.yaml 덮어쓰기 예시입니다. 지연 시간과 PVC 크기는 모델과 클러스터에 맞춰 정합니다
image:
  command: ["vllm", "serve", "/data/", "--served-model-name", "llama-7b",
            "--host", "0.0.0.0", "--port", "8000", "-tp", "2"]

extraInit:
  modelDownload:
    enabled: true
  pvcStorage: "10Gi"
  s3modelpath: "models/llama-7b"

# GPU 개수를 -tp 값과 맞춥니다. 지정하지 않은 cpu와 memory는 차트 기본값을 따릅니다
resources:
  requests:
    nvidia.com/gpu: 2
  limits:
    nvidia.com/gpu: 2

livenessProbe:
  initialDelaySeconds: 120
```

### 보안 강화 차트 사례

공식 차트는 최소 예제에 가깝습니다. Arktec Quant가 Artifact Hub에 공개한 vLLM 차트를 보면 운영 환경의 보안 요구를 어떻게 구성하는지 알 수 있습니다. 이 차트는 Kubernetes Pod Security Standards의 restricted 프로필을 기본으로 따릅니다. 루트 파일 시스템은 읽기 전용이고, 비루트 사용자로 실행하며, 모든 capability를 뺍니다. seccomp는 `RuntimeDefault`를 쓰고, 서비스 계정 토큰은 자동으로 마운트하지 않습니다.

루트 파일 시스템이 읽기 전용이면 쓰기용 공간을 따로 마련해야 합니다. PyTorch와 transformers가 쓰기 가능한 임시 디렉터리를 요구하기 때문입니다. 이 차트는 PVC를 `/models`에 마운트하고 `HF_HOME`과 `TRANSFORMERS_CACHE`를 그곳으로 돌립니다. `TMPDIR`은 `/models/tmp`로 지정합니다. 그래서 영속성을 켠 상태에서 PVC가 없으면 배포가 실패합니다. 기본으로 켜지는 장치는 다음과 같습니다.

- NetworkPolicy는 모든 트래픽을 막은 상태에서 필요한 것만 엽니다. DNS는 클러스터 DNS 파드로만 질의하게 해서 DNS 터널링을 막습니다.
- PodDisruptionBudget은 자발적 중단 중에도 가용성을 유지합니다.
- AppArmor는 `runtime/default` 프로필을 씁니다.

이미지는 SHA256 다이제스트로 고정할 수 있습니다. 고정하면 레지스트리에서 이미지를 바꿔치기하는 공격을 막습니다. 이 기능은 선택 사항이므로 직접 켜야 합니다.

보안 강화에는 대가가 따릅니다. 이 차트의 이미지는 공격 표면을 줄이려고 컴파일러와 빌드 도구를 모두 뺐습니다. 그래서 `torch.compile` JIT 컴파일과 사용자 정의 CUDA 커널 컴파일은 쓸 수 없고, 런타임 Triton 커널 빌드도 제한됩니다. 지원 범위는 미리 컴파일된 커널로 동작하는 표준 추론입니다. 같은 문서는 기동이 오래 걸린다고도 지적합니다. 큰 모델은 다운로드에만 수십 분이 걸리므로, ArgoCD 같은 GitOps 도구에서는 동기화 제한 시간을 넉넉히 늘리라고 안내합니다. 사내 차트를 만들 때는 이런 기본값을 점검 목록으로 삼을 만합니다.

## 운영 관측과 벤치마크

설정을 바꿨으면 효과를 숫자로 확인해야 합니다. 앞에서 다룬 KV 캐시 크기, 배치 한도, 병렬 구성은 모두 처리량과 지연을 맞바꾸는 설정입니다. 그래서 바꾸기 전과 후에 같은 조건으로 측정하고, 운영 중에는 로그와 지표로 계속 지켜봐야 합니다. vLLM은 측정용으로 `vllm bench` 명령을, 관측용으로 로깅 설정과 Prometheus 지표 옵션을 제공합니다. 과부하에 대비하는 장치로는 요청 큐 상한이 있습니다.

### `vllm bench`로 성능 기준선 잡기

벤치마크 명령은 추가 의존성이 있어야 동작합니다.

```bash
pip install vllm[bench]
```

하위 명령은 넷이며, 측정하는 대상이 각각 다릅니다.

| 하위 명령 | 측정 대상 | 서버 필요 여부 |
| --- | --- | --- |
| `latency` | 요청 한 배치의 지연 | 필요 없음 |
| `throughput` | 오프라인 추론 처리량 | 필요 없음 |
| `serve` | 온라인 서빙 처리량 | 실행 중인 서버 필요 |
| `forward` | 정확히 지정한 배치의 모델 순전파 시간 | 필요 없음(CUDA 전용) |

`latency`와 `throughput`은 엔진을 직접 띄워 측정합니다. 아래 예에서 `--load-format dummy`는 가중치를 무작위 값으로 초기화합니다. 주로 프로파일링에 쓰는 형식이라, 체크포인트를 내려받지 않고도 모델 구조만으로 시간을 잴 수 있습니다. `--enforce-eager`는 공식 예제를 그대로 따른 옵션입니다.

```bash
vllm bench latency \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --input-len 32 \
  --output-len 1 \
  --enforce-eager \
  --load-format dummy

vllm bench throughput \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --input-len 32 \
  --output-len 1 \
  --enforce-eager \
  --load-format dummy
```

운영 판단에 가장 직접 쓰이는 명령은 `serve`입니다. 이미 떠 있는 OpenAI 호환 서버에 요청을 보내므로, 스케줄러와 API 서버까지 포함한 실제 서빙 경로가 측정됩니다. `--host`와 `--port`에는 대상 서버 주소를 넣습니다.

```bash
vllm bench serve \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --host server-host \
  --port server-port \
  --random-input-len 32 \
  --random-output-len 4 \
  --num-prompts 5
```

`forward`는 범위가 훨씬 좁습니다. 각 케이스를 `[N*]Q[@K][c][:D]` 형식으로 적는데, Q는 쿼리 길이, K는 쿼리를 포함한 KV 길이, D는 드래프트 토큰 수입니다. 이렇게 지정한 배치의 순전파 한 번을 일회성 CUDA 그래프로 캡처해 잽니다. 프리필의 프리픽스는 기본적으로 프리픽스 캐시 적중으로 처리되고, `c`를 붙이면 청크 프리필로 바뀝니다. 요청마다 프리픽스 길이는 기본으로 ±50% 흔들리며, `--kv-len-jitter`로 이 폭을 정합니다. 샘플링과 CPU 오버헤드는 측정에서 빠지므로, 커널이나 어텐션 백엔드를 비교할 때 씁니다.

```bash
vllm bench forward \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --case decode=32*1@4096 \
  --case mix=2048,32*1@4096
```

서버를 띄운 뒤 빠르게 감을 잡을 때는 `vllm chat --stats`나 `vllm complete --stats`를 씁니다. 응답이 올 때마다 첫 토큰까지 걸린 시간(TTFT)과 처리량 통계를 출력합니다. 이슈를 보고하거나 다른 환경과 결과를 비교할 때는 `vllm collect-env`로 환경 정보를 모아 함께 남겨 둡니다.

### 로깅 설정

로깅은 Python `dictConfig`로 구성되며, 기본 설정은 포맷을 적용한 스트림 핸들러를 하나 붙입니다. 바꿀 때는 `--logging-config`에 JSON 객체를 통째로 넘기거나, 점 표기 인자로 필드를 하나씩 지정합니다. 로그 수집기가 구조화된 로그를 받는다면 JSON 포매터가 편합니다.

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --logging-config.log_level DEBUG \
  --logging-config.formatter json
```

`--log-level`과 레거시 `--log-config-file`을 주면 각각 대응하는 필드를 덮어씁니다. 애플리케이션이 로깅을 직접 구성한다면 `configure_logging`을 false로 두어 vLLM이 `dictConfig`를 적용하지 않게 합니다.

스케줄러가 실제로 어떻게 돌아가는지 보려면 `--enable-logging-iteration-details`를 켭니다. 기본값은 `False`입니다. 켜면 EngineCore가 반복마다 컨텍스트 요청과 생성 요청의 수, 토큰 수, 그 반복에 걸린 CPU 시간을 기록합니다. 앞서 설명한 반복 단위 스케줄링이 프리필과 디코드를 어떻게 섞는지 직접 확인할 수 있습니다. 로그 양이 많으므로 문제를 재현할 때만 켭니다.

### Prometheus 지표 다루기

서버는 Prometheus 지표를 내보내며, 몇 가지 엔진 인자가 지표의 모양을 정합니다. 먼저 `--served-model-name`에 준 이름이 지표의 `model_name` 태그에도 들어갑니다. 이름을 여러 개 주면 태그에는 첫 번째 이름만 쓰입니다. 대시보드와 알림 규칙이 이 태그로 모델을 구분한다면, 별칭을 추가할 때 목록의 첫 이름은 바꾸지 않습니다.

히스토그램 구간은 `--custom-histogram-buckets`로 바꿉니다. 키는 지표 계열입니다. 정의된 계열은 `request_latency`, `time_to_first_token`, `inter_token_latency`, `iteration_tokens`, `request_params_n`, `request_num_preemptions`, `request_tokens`, `kv_cache_residency`입니다. 값은 엄격히 증가하는 양의 유한한 상한 목록입니다. 지정한 계열은 그 계열의 모든 히스토그램 구간이 이 목록으로 바뀌고, 지정하지 않은 계열은 기본 구간을 그대로 씁니다.

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --custom-histogram-buckets '{"request_latency": [0.01, 0.05, 0.1, 0.5]}'
```

SLO 경계 근처에 구간을 촘촘히 두면 백분위 추정이 정확해집니다. 대신 구간이 하나 늘 때마다 지표와 레이블 조합당 시계열도 하나씩 늘어나므로, 필요한 계열에만 적용합니다. 선점이 의심될 때는 `request_num_preemptions` 계열이 먼저 볼 지표입니다.

버전을 올리다 보면 폐기된 지표가 숨겨지기도 합니다. 이때 `--show-hidden-metrics-for-version`을 주면 지정한 버전 이후 숨겨진 지표를 다시 켭니다. 예를 들어 v0.7.0부터 숨겨진 지표라면 `--show-hidden-metrics-for-version=0.7`을 줍니다. 곧 완전히 제거될 지표이므로, 새 지표로 대시보드를 옮기는 동안만 쓰는 임시 우회로입니다.

추측 디코딩을 쓴다면 집계 지표인 `vllm:spec_decode_*`와 별도로 `--per-request-spec-decode-metrics`로 요청별 수락 지표를 응답 본문의 `metrics.speculative_decoding`에 넣을 수 있습니다. 선택지는 셋입니다. `summary`는 평균 수락 길이, 드래프트 수락률, 드래프트 길이별 히스토그램을 넣고, `detailed`는 여기에 검증 단계별 배열을 더합니다. 기본값 `none`은 끕니다. `n == 1`인 요청에만 보고되며, 이 응답 필드는 실험 기능이라 형태가 바뀔 수 있습니다.

### 커널 수준 트레이스

지표로 병목 위치를 좁혔다면 다음은 GPU 커널 단위로 들여다볼 차례입니다. `vllm bench forward`에 `--profile-dir DIR`을 주면 케이스마다 Perfetto 트레이스와 커널별 GPU 시간 요약을 기록합니다. 요약은 다음 명령으로 출력하며, 파일을 두 개 주면 두 결과를 비교합니다.

```bash
vllm bench forward \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --case decode=32*1@4096 \
  --profile-dir ./profile

python -m vllm.benchmarks.forward.profile A.kernels.json B.kernels.json
```

Nsight Systems를 쓸 때는 `nsys profile --capture-range=cudaProfilerApi --capture-range-end=repeat --cuda-graph-trace=node`로 실행합니다. 그러면 측정용 재생 구간만 기록되고, 리포트가 케이스마다 하나씩 생깁니다. 어텐션 백엔드나 KV 캐시 dtype을 바꾸기 전과 후에 같은 케이스로 트레이스를 남겨 두면, 어느 커널에서 시간이 줄었는지 바로 비교할 수 있습니다.

### 요청 큐 제한: `--max-num-queued-reqs`

`--max-num-seqs`는 한 반복에 넣을 시퀀스 수를 제한할 뿐입니다. 그 뒤에 쌓이는 요청 큐에는 원래 상한이 없습니다. 그래서 트래픽이 용량을 넘으면 대기열이 끝없이 길어집니다. 요청은 거부되지 않지만 지연은 계속 늘어납니다. `--max-num-queued-reqs`는 대기 중이거나 실행 중인 요청 수의 상한입니다. 상한에 이르면 새 요청을 HTTP 503으로 거부해 클라이언트가 다른 인스턴스로 재시도하게 합니다. 값을 주지 않으면 제한이 없습니다. 이 옵션은 세밀한 스케줄링 장치가 아니라 거친 용량 밸브입니다.

`max_num_seqs`는 데이터 병렬 랭크마다 적용되지만, 이 상한은 API 서버 프로세스가 강제합니다. 그래서 API 서버가 요청을 보내는 모든 DP 랭크의 진행 중 요청을 합쳐서 셉니다. 랭크별 수용 한도보다 먼저 걸리지 않게 하려면, 데이터 병렬 크기 $$N_{\text{dp}}$$, 랭크당 `max_num_seqs` $$N_{\text{seqs}}$$, 허용할 대기 깊이 $$D$$로 다음과 같이 잡습니다.

$$
N_{\text{queued}} \approx N_{\text{dp}} \cdot N_{\text{seqs}} + D
$$

```bash
# 예시 값입니다: 2 * 128 + 64 = 320. 대기 깊이는 허용할 대기 시간에 맞춰 정합니다
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --data-parallel-size 2 \
  --max-num-seqs 128 \
  --max-num-queued-reqs 320
```

503 응답은 로드 밸런서나 클라이언트의 재시도 로직이 받아야 의미가 있습니다. 상한을 켤 때는 앞단에서 503을 다른 복제본으로 돌리도록 함께 구성하기를 권장합니다. 상한을 정한 뒤에는 `vllm bench serve`로 요청 수를 늘려 가며 거부가 시작되는 지점과 그 직전의 지연을 확인합니다. 그 결과로 대기 깊이 $$D$$를 조정합니다.

- - -

### Sources
* [Quickstart - vLLM](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
* [Installation - vLLM](https://docs.vllm.ai/en/latest/getting_started/installation/)
* [GPU - vLLM](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
* [CPU - vLLM](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/)
* [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
* [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention &#124; vLLM Blog](https://blog.vllm.ai/2023/06/20/vllm.html)
* [图解大模型计算加速系列：vLLM源码解析3，块管理器BlockManager（上篇）](https://zhuanlan.zhihu.com/p/700780161)
* [vllm 1.0.7 · arktecquant/arktec-quant-helm-charts](https://artifacthub.io/packages/helm/arktec-quant-helm-charts/vllm)
* [vLLM vs SGLang vs TGI: 2026 Inference Engine Benchmark](https://llm-academy.dev/inference/vllm-vs-sglang)
* [vLLM CLI Guide - vLLM](https://docs.vllm.ai/en/latest/cli)
* [Helm - vLLM](https://docs.vllm.ai/en/stable/deployment/frameworks/helm)
* [LLM Inference Engines Compared 2026: vLLM vs SGLang vs TGI vs MAX - DEV Community](https://dev.to/jangwook_kim_e31e7291ad98/llm-inference-engines-compared-2026-vllm-vs-sglang-vs-tgi-vs-max-4fpo)
* [Engine Arguments - vLLM](https://docs.vllm.ai/en/latest/configuration/engine_args/)
* [Server Arguments - vLLM](https://docs.vllm.ai/en/latest/configuration/serve_args/)
* [Architecture Overview - vLLM](https://docs.vllm.ai/en/stable/design/arch_overview)
* [Optimization and Tuning - vLLM](https://docs.vllm.ai/en/stable/configuration/optimization)
* [Data Parallel Deployment - vLLM](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment)
