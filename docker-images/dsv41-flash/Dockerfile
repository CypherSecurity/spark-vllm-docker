# DeepSeek-V4.1-Flash — DGX Spark (arm64, sm_121a), clone-at-pin.
#
# One Dockerfile for tonyd2wild's overlay1/3/5 chain (MIT; see versions.env
# header + patches/README note). No fallbacks: every gate fails loud.
#
#   base    pinned vllm/vllm-openai nightly (arm64 digest)
#   vLLM    vllm-project/vllm @ dsv41-feat commit+tree (day-0 DSV4.1 support)
#   ext     _C_stable_libtorch rebuilt for sm_121a (stable-only cmake build)
#   FI      FlashInfer 0.7.0rc1 from pinned source (0.6.18 lacks the SM120
#           sparse-MLA decode kernel for page_block_size=32)
#   warm    sparse-MLA + mxfp8 JIT kernels prebuilt in the image runtime env
#   patches the 7 boot-10 file replacements, baked over dist-packages,
#           md5-gated against the published manifest
#
# Model: deepseek-ai/DeepSeek-V4.1-Flash (510 GB; Engram tables stay on disk
# via the baked engram.py patch — DSV41_ENGRAM_DISK=1 at serve time).

ARG VLLM_BASE_IMAGE
FROM ${VLLM_BASE_IMAGE}

ARG VLLM_REPO
ARG VLLM_REF
ARG VLLM_REF_TREE
ARG VLLM_PRETEND_VERSION
ARG CUTLASS_TAG
ARG CUTLASS_URL
ARG FI_SHA
ARG FI_CUTLASS_SHA
ARG FI_CCCL_SHA
ARG FI_SPDLOG_SHA

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MAX_JOBS=4

# ---- vLLM python tree @ pin (clone-at-pin: commit + tree hash gates) ----
RUN set -eux; \
    apt-get update -qq && apt-get install -y --no-install-recommends git ca-certificates >/dev/null && rm -rf /var/lib/apt/lists/*; \
    git config --global protocol.version 0; \
    git init -q /src/vllm; \
    git -C /src/vllm fetch --depth 1 ${VLLM_REPO} ${VLLM_REF}; \
    git -C /src/vllm checkout -q ${VLLM_REF}; \
    test "$(git -C /src/vllm rev-parse HEAD)" = "${VLLM_REF}"; \
    test "$(git -C /src/vllm rev-parse 'HEAD^{tree}')" = "${VLLM_REF_TREE}"; \
    echo "vLLM pinned at ${VLLM_REF} / tree ${VLLM_REF_TREE}"

# ---- _C_stable_libtorch rebuilt for sm_121a (their build_stable_ext.sh) ----
# stable-only build: external_projects includes commented out, cutlass fetched
# at the pinned tag, Release build of the single C extension target.
RUN set -eux; \
    apt-get update -qq && apt-get install -y --no-install-recommends cmake ninja-build g++ >/dev/null && rm -rf /var/lib/apt/lists/*; \
    cd /src/vllm; \
    mkdir -p build/_deps/cutlass-src; \
    curl -sL -m 600 ${CUTLASS_URL} | tar xz -C build/_deps/cutlass-src --strip-components=1; \
    ls build/_deps/cutlass-src/include/cutlass/cutlass.h >/dev/null; \
    cp CMakeLists.txt CMakeLists.txt.orig; \
    sed -i -E "s|^(\s*)include\(cmake/external_projects/|\1# STABLE-ONLY: include(cmake/external_projects/|" CMakeLists.txt; \
    PYPATH=$(python3 -c "import sys;print(':'.join(p for p in sys.path if p))"); \
    TORCH_PREFIX=$(python3 -c "import torch;print(torch.utils.cmake_prefix_path)"); \
    NVRTC=$(ls /usr/local/cuda/lib64/libnvrtc.so /usr/local/cuda/lib64/libnvrtc.so.* /usr/local/lib/python3.12/dist-packages/nvidia/*/lib/libnvrtc.so* /usr/lib/aarch64-linux-gnu/libnvrtc.so* 2>/dev/null | head -1); \
    test -n "${NVRTC}"; \
    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_TARGET_DEVICE=cuda \
        -DVLLM_PYTHON_EXECUTABLE=$(which python3) -DVLLM_PYTHON_PATH="$PYPATH" \
        -DFETCHCONTENT_BASE_DIR=build/_deps -DFETCHCONTENT_SOURCE_DIR_CUTLASS=build/_deps/cutlass-src \
        -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" -DNVCC_THREADS=2 -DCUDA_nvrtc_LIBRARY="$NVRTC" -DTORCH_CUDA_ARCH_LIST=12.1a; \
    cmake --build build --target _C_stable_libtorch -j 20; \
    find build vllm -maxdepth 2 -name "_C_stable_libtorch*.so" | tee /tmp/stable_ext.so; \
    # place the built extension in the package dir (vLLM imports vllm._C_stable_libtorch)
    cp -v $(find build vllm -maxdepth 2 -name "_C_stable_libtorch*.so" | head -1) vllm/

# ---- merge the branch python tree (+ the sm121 .so) over the base install ----
RUN set -eux; \
    SITE=/usr/local/lib/python3.12/dist-packages/vllm; \
    cp -a /src/vllm/vllm/. ${SITE}/; \
    find ${SITE} -name "__pycache__" -type d -prune -exec rm -rf {} +; \
    python3 -c "import vllm; print('vllm', vllm.__version__)"; \
    # existence check only: dlopen of the CUDA extension needs libcuda.so.1
    # (GPU driver), which the build container does not have — the real import
    # happens at serve time on the GPU node
    ls ${SITE}/_C_stable_libtorch*.so

# ---- FlashInfer 0.7.0rc1 from pinned source (0.6.18 has no SM120 sparse-MLA) ----
RUN set -eux; \
    pip uninstall -y -q flashinfer-jit-cache flashinfer-cubin flashinfer-python || true; \
    rm -rf /root/.cache/flashinfer; \
    mkdir -p /opt/fi-src; \
    curl -sL -m 900 "https://codeload.github.com/flashinfer-ai/flashinfer/tar.gz/${FI_SHA}" | tar xz -C /opt/fi-src --strip-components=1; \
    mkdir -p /opt/fi-src/3rdparty/cutlass /opt/fi-src/3rdparty/cccl /opt/fi-src/3rdparty/spdlog; \
    curl -sL -m 900 "https://codeload.github.com/NVIDIA/cutlass/tar.gz/${FI_CUTLASS_SHA}" | tar xz -C /opt/fi-src/3rdparty/cutlass --strip-components=1; \
    curl -sL -m 900 "https://codeload.github.com/NVIDIA/cccl/tar.gz/${FI_CCCL_SHA}" | tar xz -C /opt/fi-src/3rdparty/cccl --strip-components=1; \
    curl -sL -m 900 "https://codeload.github.com/gabime/spdlog/tar.gz/${FI_SPDLOG_SHA}" | tar xz -C /opt/fi-src/3rdparty/spdlog --strip-components=1; \
    ls /opt/fi-src/3rdparty/cutlass/include/cutlass/cutlass.h /opt/fi-src/3rdparty/cccl/README.md /opt/fi-src/3rdparty/spdlog/include/spdlog/spdlog.h; \
    cd /opt/fi-src && BUILD_NVEP=0 FLASHINFER_BUILD_NO_PIP=1 pip install --no-deps --no-build-isolation -q .; \
    rm -rf /opt/fi-src/build; \
    python3 -c "import flashinfer; assert '0.7.0rc1' in flashinfer.__version__, flashinfer.__version__; print('flashinfer', flashinfer.__version__)"

ENV VLLM_HAS_FLASHINFER_CUBIN=1

# ---- prebuild the SM120 sparse-MLA + mxfp8 GEMM JIT kernels ----
# Exact runtime env (no verbose/debug nvcc flags — those change the cache key).
# The compiled modules persist in /root/.cache/flashinfer inside the image.
COPY prewarm5.py /tmp/prewarm5.py
RUN set -eux; \
    FLASHINFER_CUDA_ARCH_LIST=12.1a TORCH_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_DISABLE_VERSION_CHECK=1 VLLM_HAS_FLASHINFER_CUBIN=1 \
    FLASHINFER_NVCC_THREADS=1 MAX_JOBS=4 \
    python3 /tmp/prewarm5.py | tee /tmp/prewarm.log; \
    grep -q "SPARSE-BUILT+LOADED" /tmp/prewarm.log || { echo "sparse MLA prewarm FAILED"; exit 1; }

# ---- bake the 7 boot-10 patch files (md5-gated against their manifest) ----
COPY patches/ /tmp/patches/
RUN set -eux; \
    python3 - <<'PY'
import hashlib, os, shutil, sys
SITE = "/usr/local/lib/python3.12/dist-packages/vllm"
# md5 prefixes from tonyd2wild patch/README.md (boot-10 manifest)
MD5 = {
    "engram.py": "c0329107", "model_state.py": "0a14bee6",
    "weight_utils.py": "7e1027f1", "attention.py": "da9ef196",
    "flashinfer_sparse.py": "af0f8447", "sparse_swa.py": "cc419353",
    "sparse_attn_indexer.py": "a9b73756",
}
bad = []
for line in open("/tmp/patches/mounts.txt"):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    f, rel = line.split()
    src = f"/tmp/patches/{f}"
    m = hashlib.md5(open(src, "rb").read()).hexdigest()[:8]
    if m != MD5.get(f):
        bad.append(f"{f}: {m} != {MD5.get(f)}")
        continue
    dst = os.path.join(SITE, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"baked {f} -> {rel} (md5 {m})")
sys.exit("\n".join(bad) if bad else None)
PY

# ---- final gates ----
RUN set -eux; \
    python3 -c "import vllm; print('vllm', vllm.__version__)"; \
    ls /usr/local/lib/python3.12/dist-packages/vllm/_C_stable_libtorch*.so; \
    python3 -c "import flashinfer; print('flashinfer', flashinfer.__version__)"; \
    grep -q "DSV41_ENGRAM_DISK" /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/common/engram.py; \
    echo "=== image gates passed ==="
