#!/usr/bin/env bash
# Freeze this data-only producer and every literal local include before generation.
set -euo pipefail
test "$#" -eq 2
freeze=$1
binary=$(realpath "$2")
test ! -e "$freeze"
test -x "$binary"
test -z "$(git status --porcelain)"
mkdir -p "$freeze/source"
git rev-parse HEAD > "$freeze/producer.commit"
git ls-files -- experiments/convex_stress_arithmetic_data \
    experiments/convex_stress/src/vendor/arithmetic_data.rs experiments/convex_stress/suite.json \
    stress_tests_for_gpt.md AGENTS.md > "$freeze/source-paths.txt"
while IFS= read -r source; do
    mkdir -p "$freeze/source/$(dirname "$source")"
    cp "$source" "$freeze/source/$source"
    sha256sum "$source"
done < "$freeze/source-paths.txt" > "$freeze/source.sha256"
cargo metadata --offline --locked --manifest-path experiments/convex_stress_arithmetic_data/Cargo.toml \
    --format-version 1 > "$freeze/cargo-metadata.json"
test -f "$binary.d"
cp "$binary.d" "$freeze/producer.d"
work=$(git rev-parse --show-toplevel)
read -r -a dependencies < "$binary.d"
for dependency in "${dependencies[@]:1}"; do
    included=$(realpath -e "$dependency")
    if [[ "$included" == "$work/"* ]]; then
        relative=${included#"$work/"}
        rg -F -x -- "$relative" "$freeze/source-paths.txt" > /dev/null
    fi
done
cp "$binary" "$freeze/convex-stress-arithmetic-data"
chmod 755 "$freeze/convex-stress-arithmetic-data"
sha256sum "$freeze/convex-stress-arithmetic-data" > "$freeze/binary.sha256"
rustc --version > "$freeze/rustc.version"
cargo --version > "$freeze/cargo.version"
cd "$freeze/source"
sha256sum -c "$freeze/source.sha256" > "$freeze/source-check.log"
