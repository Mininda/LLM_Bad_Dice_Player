#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"
python -m py_compile \
  src/common.py \
  src/generate_samples.py \
  src/generate_downstream.py \
  src/verify_release.py
python src/verify_release.py
printf 'validate_release.sh: all checks passed\n'
