#!/usr/bin/env bash
# Compile only native CPU ATen; never invoke Python or link torch_python.
set -euo pipefail
test "$#" -eq 2
torch_root=$(realpath "$1")
output=$2
source_dir=$(cd "$(dirname "$0")" && pwd)
test ! -e "$output"
test -f "$torch_root/include/ATen/ATen.h"
test -f "$torch_root/lib/libtorch_cpu.so"
test "$(sha256sum "$torch_root/lib/libtorch_cpu.so" | cut -d' ' -f1)" = 0341b39fc44ec6533c42861e8dd602dfa4182fdba1ceef02a2b9187bb5001112
test "$(sha256sum "$torch_root/lib/libc10.so" | cut -d' ' -f1)" = 63b97230854de82cd0730dab4b874372a1fbca5e9082ab0e72bdedaf3337ada1
rg -q '(^| )avx2( |$)' /proc/cpuinfo
mkdir -p "$(dirname "$output")"
g++ -std=c++20 -O2 -fno-fast-math -ffp-contract=off -Wall -Wextra \
  -isystem "$torch_root/include" -isystem "$torch_root/include/torch/csrc/api/include" \
  "$source_dir/reference.cpp" -L "$torch_root/lib" -Wl,-rpath,"$torch_root/lib" \
  -ltorch_cpu -lc10 -o "$output"
sha256sum "$output" "$torch_root/lib/libtorch_cpu.so" "$torch_root/lib/libc10.so" \
  "$source_dir/reference.cpp" > "$output.sha256"
g++ --version > "$output.compiler.txt"
ldd "$output" > "$output.linked-libraries.txt"
if rg -i 'libpython|torch_python' "$output.linked-libraries.txt"; then
  printf '%s\n' 'Unexpected Python runtime linkage' >&2
  exit 1
fi
