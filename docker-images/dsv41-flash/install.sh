#!/usr/bin/env bash
# Build dsv41-flash:local. Run from this folder, or set GIST_ID and run from
# anywhere to clone the published gist into /tmp first.
# Usage: curl -fsSL <raw-url>/install.sh | bash   (after publishing as a gist)
set -euo pipefail

GIST_ID='1ba224bfe28779c571d096cf954fff31'  # filled in by `sm121 publish --gist`
DIR=/tmp/dsv41-flash-4x-build
IMAGE='dsv41-flash:local'

HERE="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$HERE/Dockerfile" ]; then
  SRC="$HERE"
else
  rm -rf "$DIR"; git clone -q "https://gist.github.com/${GIST_ID}.git" "$DIR"; SRC="$DIR"
fi

docker build -t "$IMAGE" \
  --build-arg 'CUTLASS_TAG'='v4.7.1' \
  --build-arg 'CUTLASS_URL'='https://codeload.github.com/NVIDIA/cutlass/tar.gz/refs/tags/v4.7.1' \
  --build-arg 'FI_CCCL_SHA'='16bd510c9b712e82b0ab6cbb630d8e29ba1f7116' \
  --build-arg 'FI_CUTLASS_SHA'='b46b16d003484063bca4ed365e44095c4c6ed633' \
  --build-arg 'FI_SHA'='07869c61ba581e6d6b8ad8d142f4a6c89b707cc1' \
  --build-arg 'FI_SPDLOG_SHA'='c3aed4b68373955e1cc94307683d44dca1515d2b' \
  --build-arg 'VLLM_BASE_IMAGE'='docker.io/vllm/vllm-openai@sha256:a551e05307cd2e0092139d84db32af9c97e67d2eeeff072d21e429131d8c23f0' \
  --build-arg 'VLLM_PRETEND_VERSION'='0.1.dev0+dsv41.e47aa780' \
  --build-arg 'VLLM_REF'='e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba' \
  --build-arg 'VLLM_REF_TREE'='8bf2b4034505699b829f9fcc1a8cc8e041c49384' \
  --build-arg 'VLLM_REPO'='https://github.com/vllm-project/vllm.git' \
"$SRC"
echo "built: $IMAGE — see run.sh"
