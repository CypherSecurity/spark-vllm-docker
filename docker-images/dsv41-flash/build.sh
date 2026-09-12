#!/usr/bin/env bash
set -euo pipefail; cd "$(dirname "$0")"
TAG="${TAG:?set TAG=<recipe docker_image>}"
DOCKER_BUILDKIT=1 docker build -t "$TAG" -f Dockerfile . 2>&1
echo "=== Done: ${TAG} ==="
