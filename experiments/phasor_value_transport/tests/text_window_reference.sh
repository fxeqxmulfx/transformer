#!/usr/bin/env bash
set -euo pipefail
: "${LIBTORCH:?set LIBTORCH to a native LibTorch distribution}"
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT
"${CXX:-c++}" -std=c++20 -O2 -isystem "$LIBTORCH/include" \
    -isystem "$LIBTORCH/include/torch/csrc/api/include" -L"$LIBTORCH/lib" \
    tests/reference/text_window_torch.cpp -ltorch_cpu -lc10 -Wl,-rpath,"$LIBTORCH/lib" -o "$tmp/source"
for split in train validation test; do
    hash=$("$tmp/source" ../../.lake/reproduction/gpt/data/tinyshakespeare.txt "$split" | sha256sum)
    printf '%s\t%s\n' "$split" "${hash%% *}"
done > "$tmp/hashes.tsv"
diff -u tests/reference/text_window_sha256.tsv "$tmp/hashes.tsv"
cargo +stable test --release --locked --offline --features fp32-affine-reference \
    --bin phasor-value-transport text_window
