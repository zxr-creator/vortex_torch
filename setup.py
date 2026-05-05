import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def _normalize_single_arch(arch: str):
    """
    Normalize a single architecture string into (major, minor).

    Supported examples:
    - "8.9"
    - "89"
    - "sm_89"
    - "compute_89"
    """
    arch = arch.strip().lower()

    if arch.startswith("sm_"):
        arch = arch[3:]
    elif arch.startswith("compute_"):
        arch = arch[len("compute_"):]

    if "." in arch:
        major, minor = arch.split(".", 1)
        return int(major), int(minor)

    if arch.isdigit():
        if len(arch) == 2:
            return int(arch[0]), int(arch[1])
        elif len(arch) == 3:
            return int(arch[:-1]), int(arch[-1])

    raise ValueError(f"Unsupported CUDA arch format: {arch}")


def _resolve_cuda_capabilities():
    """
    Resolve the set of target CUDA capabilities as (major, minor) tuples.

    Priority:
    1. TORCH_CUDA_ARCH_LIST environment variable
    2. Detect capabilities from visible local GPUs via torch.cuda
    """
    arch_list_env = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    capabilities = set()

    if arch_list_env:
        # Example values:
        # "8.9 9.0"
        # "8.9;9.0"
        # "sm_89 sm_90"
        raw_items = arch_list_env.replace(";", " ").split()
        for item in raw_items:
            major, minor = _normalize_single_arch(item)
            capabilities.add((major, minor))
    else:
        # Import torch lazily so the file can still be imported in some environments.
        import torch

        if torch.cuda.is_available():
            for device_idx in range(torch.cuda.device_count()):
                major, minor = torch.cuda.get_device_capability(device_idx)
                capabilities.add((major, minor))

    return capabilities


def get_cuda_gencode_flags(include_ptx: bool = False):
    """
    Generate NVCC -gencode flags automatically.

    Args:
        include_ptx: Whether to also emit PTX fallback flags.

    Returns:
        A list of NVCC flags such as:
        [
            "-gencode=arch=compute_89,code=sm_89",
            "-gencode=arch=compute_90,code=sm_90",
        ]
    """
    flags = []
    for major, minor in sorted(_resolve_cuda_capabilities()):
        compute = f"compute_{major}{minor}"
        sm = f"sm_{major}{minor}"
        flags.append(f"-gencode=arch={compute},code={sm}")

        if include_ptx:
            flags.append(f"-gencode=arch={compute},code={compute}")

    return flags


_cuda_caps = _resolve_cuda_capabilities()

nvcc_flags = [
    "-O3",
] + get_cuda_gencode_flags(include_ptx=False)

# Enable FP8 (e4m3fn, e5m2) code paths only when at least one target arch is sm_90+.
if any(cap >= (9, 0) for cap in _cuda_caps):
    nvcc_flags.append("-DVORTEX_ENABLE_FP8")


setup(
    name="vortex_torch",
    version="0.3.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.7",
        "lighteval[math]==0.12.2",
        "huggingface_hub==0.36.2",
        "inspect-ai==0.3.207",
        "transformers==4.57.6",
        "s3fs==2025.9.0",
        "wonderwords",
        "langdetect",
        "jsonlines",
        "immutabledict",
        "datasets",
        "beautifulsoup4",
        "accelerate",
        "unitxt",
    ],
    ext_modules=[
        CUDAExtension(
            name="vortex_torch_C",
            sources=[
                "csrc/register.cc",
                "csrc/utils_sglang.cu",
                "csrc/utils_sglang_v2.cu",
                "csrc/topk.cu",
                "csrc/topk_v2.cu",
                "csrc/topk_v2_remap.cu",
                "csrc/approx_topk.cu",
                "csrc/approx_topk_remap.cu",
                "csrc/sglang_topk.cu",
            ],
            include_dirs=["csrc"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": nvcc_flags,
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
