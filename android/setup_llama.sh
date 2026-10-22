#!/usr/bin/env bash
# Fetch llama.cpp at the commit the training export uses (keep LLAMA_CPP_COMMIT in sync
# with training/export_gguf.py). Run once before opening the project in Android Studio.
# On Windows, run it from Git Bash.
set -euo pipefail
COMMIT=0c1e57098bba43ac29e6e3b677cdceebdd22334f
DIR="$(cd "$(dirname "$0")" && pwd)/app/src/main/cpp/llama.cpp"
if [ -f "$DIR/CMakeLists.txt" ]; then echo "llama.cpp already present: $DIR"; exit 0; fi
mkdir -p "$DIR" && cd "$DIR"
git init -q
git fetch -q --depth 1 https://github.com/ggml-org/llama.cpp "$COMMIT"
git checkout -q FETCH_HEAD
echo "llama.cpp $COMMIT ready in $DIR"
