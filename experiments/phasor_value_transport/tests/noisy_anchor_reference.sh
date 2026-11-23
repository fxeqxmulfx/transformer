#!/usr/bin/env bash
set -euo pipefail
: "${NUMPY_SOURCE:?set NUMPY_SOURCE to the numpy directory in NumPy 2.5.4}"
: "${NUMPY_RANDOM_ARCHIVE:?set NUMPY_RANDOM_ARCHIVE to the matching libnpyrandom.a}"
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT
pcg="$NUMPY_SOURCE/random/src/pcg64"
cc -std=c11 -O2 -c "$pcg/pcg64.c" -o "$tmp/pcg.o"
c++ -std=c++20 -O2 -Wall -Wextra -Werror -I"$pcg" -I"$NUMPY_SOURCE/_core/include" \
    tests/reference/noisy_anchor_numpy.cpp "$tmp/pcg.o" "$NUMPY_RANDOM_ARCHIVE" -lm -o "$tmp/source"
while read -r population percent count digest; do
    file="$tmp/$population-$percent.bin"
    "$tmp/source" "$population" "$count" > "$file"
    echo "$digest  $file" | sha256sum -c -
done < tests/reference/noisy_anchor_draws.tsv
cargo +stable test --release --locked --offline --features fp32-affine-reference \
    --bin phasor-value-transport stress::noisy_anchors::tests
