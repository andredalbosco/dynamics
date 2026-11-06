#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ $# != 1 ]] || [[ ! $1 =~ ^[a-z_]+$ ]] || [[ ! -f scenes/$1.cpp ]]; then
    echo "Usage: ./watch.sh <scene> (spring, pendulum, double_pendulum, cloth, falling_cloth, mpm)" >&2
    exit 2
fi

if ! command -v watchexec >/dev/null; then
    echo "Install watchexec to use this script." >&2
    exit 1
fi

exec watchexec --restart \
    --watch scenes --watch core --watch utils --watch-non-recursive . \
    --filter "**/scenes/$1.cpp" \
    --filter '**/core/**' --filter '**/utils/**' --filter '**/CMakeLists.txt' \
    --shell=none -- bash -c '
        cmake -S . -B build -DCMAKE_BUILD_TYPE=Release &&
        cmake --build build --target "$1" --parallel 8 &&
        exec "./build/$1"
    ' _ "$1"
