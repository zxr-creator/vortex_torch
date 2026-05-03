FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

ARG TORCH_CUDA_ARCH_LIST="9.0"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    ninja-build \
    wget \
    libnuma1 \
    libnuma-dev \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"
ENV VIRTUAL_ENV="/opt/venv"

RUN python -m pip install --upgrade pip setuptools wheel

ARG VORTEX_TORCH_REF=v0.3

WORKDIR /workspace

RUN git clone -b "${VORTEX_TORCH_REF}" --recursive \
    https://github.com/Infini-AI-Lab/vortex_torch.git

WORKDIR /workspace/vortex_torch/third_party/sglang

RUN bash install.sh

WORKDIR /workspace/vortex_torch

RUN pip install --no-cache-dir -e .

RUN which python && python --version && which pip && pip --version

# Verify libnuma is visible at runtime
RUN python - <<'PY'
import ctypes
ctypes.CDLL("libnuma.so.1")
print("OK: libnuma.so.1 loaded")
PY


ARG AWS_EFA_INSTALLER_URL=https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz

ENV PATH=/opt/amazon/efa/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu:/opt/amazon/ofi-nccl/lib:/opt/aws-ofi-nccl/lib:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    iproute2 \
    cpio \
    linux-headers-aws \
    linux-aws \
    libhwloc15 \
    tar \
    unzip \
    pciutils environment-modules tcl \
    && rm -rf /var/lib/apt/lists/* \
    || (apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    cpio \
    iproute2 \
    libhwloc15 \
    tar \
    unzip \
    pciutils environment-modules tcl \
    && rm -rf /var/lib/apt/lists/*) \
    || true

RUN curl -fsSL -o aws-efa-installer-latest.tar.gz "${AWS_EFA_INSTALLER_URL}"
RUN tar --no-same-owner -xf aws-efa-installer-latest.tar.gz
RUN apt-get update && \
    apt-get install -y linux-aws linux-headers-aws || true
RUN cd aws-efa-installer && \
    ./efa_installer.sh -y --skip-kmod --disable-ngc --skip-mpi

RUN plugin_path="$(find /opt/amazon /opt/aws-ofi-nccl -name 'libnccl-net.so' -print -quit 2>/dev/null || true)" \
    && test -n "${plugin_path}" \
    && ldconfig \
    && for required_lib in libhwloc.so.15 libfabric.so.1 libefa.so.1 libibverbs.so.1; do \
    ldconfig -p | grep -q "${required_lib}" || exit 1; \
    done \
    && ldd "${plugin_path}" \
    && ! ldd "${plugin_path}" | grep -q "not found" \
    && echo "Found NCCL net plugin at ${plugin_path}"

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip" && \
    unzip -q /tmp/awscliv2.zip -d /tmp && \
    /tmp/aws/install && \
    rm -rf /tmp/awscliv2.zip /tmp/aws

RUN rm -rf /var/lib/apt/lists/*

CMD ["/bin/bash"]