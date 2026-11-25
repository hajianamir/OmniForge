#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

for target in amd64 arm64; do
    case "$target" in
        amd64) platform=x86_64-manylinux_2_28 ;;
        arm64) platform=aarch64-manylinux_2_28 ;;
    esac
    uv pip compile pyproject.toml --extra api --extra ui --extra reporting --extra data --extra forecasting \
        --python-version 3.12 --python-platform "$platform" --only-binary :all: --generate-hashes \
        --no-header --no-annotate -o "docker/requirements-${target}.lock"
done

python3 -c 'import tomllib; print(*tomllib.load(open("pyproject.toml", "rb"))["build-system"]["requires"], sep="\n")' |
    uv pip compile - --python-version 3.12 --only-binary :all: --generate-hashes \
        --no-header --no-annotate -o docker/requirements-build.lock
