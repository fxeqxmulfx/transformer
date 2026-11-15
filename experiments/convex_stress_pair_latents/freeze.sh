#!/usr/bin/env bash
# Preserve the exact native source and full local Lean import closure.
set -euo pipefail
test "$#" -eq 2
freeze=$1
binary=$2
test ! -e "$freeze"
test -x "$binary"
test -f "$binary.d"
test -z "$(git status --porcelain)"
cd "$(git rev-parse --show-toplevel)"
mkdir -p "$freeze/source"
git rev-parse HEAD > "$freeze/producer.commit"
git ls-files -- experiments/convex_stress_bounded_latent experiments/convex_stress_pair_latents \
    src scripts/BoundedLatentAxioms.lean scripts/Axioms.lean scripts/native_index \
    lakefile.toml lake-manifest.json lean-toolchain INDEX.md AGENTS.md \
    EXPERIMENT_PLAN.md experiments/README.md review/flat_spectrum.md > "$freeze/source-paths.txt"
while IFS= read -r path; do
    mkdir -p "$freeze/source/$(dirname "$path")"
    cp "$path" "$freeze/source/$path"
    sha256sum "$path"
done < "$freeze/source-paths.txt" > "$freeze/source.sha256"
cp "$binary" "$freeze/convex-stress-pair-latents"
cp "$binary.d" "$freeze/producer.d"
sha256sum "$freeze/convex-stress-pair-latents" > "$freeze/binary.sha256"
rustc --version > "$freeze/rustc.version"
cargo --version > "$freeze/cargo.version"
cd "$freeze/source"
sha256sum -c "$freeze/source.sha256" > "$freeze/source-check.log"
