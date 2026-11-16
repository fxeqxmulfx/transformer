#!/usr/bin/env bash
# Preserve the complete native source and Lean closure before original fitting.
set -euo pipefail
test "$#" -eq 2
freeze=$1
binary=$(realpath "$2")
test ! -e "$freeze"
test -x "$binary"
test -f "$binary.d"
test -z "$(git status --porcelain)"
cd "$(git rev-parse --show-toplevel)"
jq -e '.production_ready == true and .required_modes == ["forward", "mini"] and
    (.references | all(.[]; type == "string" and test("^[0-9a-f]{64}$")))' \
    experiments/convex_stress_compact_compare/protocol.json > /dev/null
mkdir -p "$freeze/source"
git rev-parse HEAD > "$freeze/producer.commit"
git ls-files -- experiments/convex_stress_bounded_latent experiments/convex_stress_pair_latents \
    experiments/convex_stress_reduced_latent experiments/convex_stress_wide_latents experiments/convex_stress_composed_latents experiments/convex_stress_reduced_compare \
    experiments/convex_stress_composed_compare experiments/convex_stress_compact_latent \
    experiments/convex_stress_compact_compare src scripts/BoundedLatentAxioms.lean \
    scripts/ObservedRowAxioms.lean scripts/ObservedProjectionAxioms.lean scripts/FamilyExtensionAxioms.lean \
    scripts/PackedRecordAxioms.lean scripts/UniformRowAxioms.lean scripts/RepeatedRowAxioms.lean \
    scripts/Axioms.lean scripts/native_index lakefile.toml lake-manifest.json \
    lean-toolchain INDEX.md AGENTS.md EXPERIMENT_PLAN.md experiments/README.md \
    review/flat_spectrum.md > "$freeze/source-paths.txt"
while IFS= read -r path; do
    mkdir -p "$freeze/source/$(dirname "$path")"
    cp "$path" "$freeze/source/$path"
    sha256sum "$path"
done < "$freeze/source-paths.txt" > "$freeze/source.sha256"
cp "$binary" "$freeze/convex-stress-compact-compare"
cp "$binary.d" "$freeze/producer.d"
sha256sum "$freeze/convex-stress-compact-compare" > "$freeze/binary.sha256"
rustc --version > "$freeze/rustc.version"
cargo --version > "$freeze/cargo.version"
cd "$freeze/source"
sha256sum --check "$freeze/source.sha256" > "$freeze/source-check.log"
chmod -R a-w "$freeze/source"
chmod a-w "$freeze/convex-stress-compact-compare"
