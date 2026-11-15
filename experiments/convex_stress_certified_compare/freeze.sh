#!/usr/bin/env bash
# Adapt task347's freeze closure at1371ae4f for the full CM63 comparison.
# Freeze tracked local dependencies and the closure of literal Rust includes.
set -euo pipefail
test "$#" -eq 2
freeze=$1
binary=$2
test ! -e "$freeze"
test -x "$binary"
test -z "$(git status --porcelain)"
jq -e '.baseline347_verified_sha256 | strings | test("^[0-9a-f]{64}$")' \
    experiments/convex_stress_certified_compare/protocol.json > /dev/null
work=$(git rev-parse --show-toplevel)
cd "$work"
mkdir -p "$freeze/source"
git rev-parse HEAD > "$freeze/producer.commit"
cargo metadata --offline --locked --manifest-path experiments/convex_stress_certified_compare/Cargo.toml \
    --format-version 1 > "$freeze/cargo-metadata.json"
jq -r '.packages[] | select(.source == null) | .manifest_path | sub("/Cargo.toml$"; "")' \
    "$freeze/cargo-metadata.json" > "$freeze/package-paths.txt"
while IFS= read -r package; do
    git ls-files -- "${package#"$work/"}"
    if [[ -f "$package/reference-sources.sha256" ]]; then
        sha256sum -c "$package/reference-sources.sha256" >> "$freeze/reference-source-check.log"
        while read -r digest source; do
            test "${#digest}" -eq 64
            git ls-files --error-unmatch "$source"
        done < "$package/reference-sources.sha256"
    fi
done < "$freeze/package-paths.txt" > "$freeze/source-paths.unsorted"
git ls-files -- experiments/convex_stress experiments/convex_stress_catalog \
    AGENTS.md EXPERIMENT_PLAN.md experiments/README.md review/flat_spectrum.md \
    >> "$freeze/source-paths.unsorted"
git ls-files -- src/Transformer/Grokking/FlatSpectrum/ConvexDepth/LatentPrograms \
    src/Transformer/Grokking/FlatSpectrum/ConvexDepth/LatentPrograms.lean \
    src/Transformer/Grokking/FlatSpectrum/ConvexCausal/Generation.lean \
    scripts/LatentGenerationAxioms.lean scripts/PrefixCountingGenerationAxioms.lean >> "$freeze/source-paths.unsorted"
# Preserve the complete local Lean import closure for CM57 and its audit.
git ls-files -- src lakefile.toml lake-manifest.json lean-toolchain INDEX.md \
    scripts/Axioms.lean scripts/UnseenClassGeometryAxioms.lean scripts/CoordinateProgramAxioms.lean scripts/ObservedPrefixAxioms.lean scripts/StreamingSearchAxioms.lean scripts/ExactScoringAxioms.lean scripts/native_index \
    >> "$freeze/source-paths.unsorted"
# Cargo's dependency file additionally covers the exact compiled include graph.
test -f "$binary.d"
cp "$binary.d" "$freeze/producer.d"
read -r -a dependencies < "$binary.d"
for dependency in "${dependencies[@]:1}"; do
    included=$(realpath -e "$dependency")
    if [[ "$included" == "$work/"* ]]; then
        if [[ -d "$included" ]]; then
            git ls-files -- "${included#"$work/"}" >> "$freeze/source-paths.unsorted"
        else
            git ls-files --error-unmatch "${included#"$work/"}" > /dev/null
            printf '%s\n' "${included#"$work/"}" >> "$freeze/source-paths.unsorted"
        fi
    fi
done
sort -u "$freeze/source-paths.unsorted" > "$freeze/source-paths.txt"
while :; do
    cp "$freeze/source-paths.txt" "$freeze/previous-paths.txt"
    while IFS= read -r source; do
        if [[ "$source" != *.rs ]]; then continue; fi
        while IFS= read -r literal; do
            relative=${literal#*\"}
            relative=${relative%\"}
            included=$(realpath -e "$(dirname "$source")/$relative")
            [[ "$included" == "$work/"* ]]
            git ls-files --error-unmatch "${included#"$work/"}" > /dev/null
            printf '%s\n' "${included#"$work/"}" >> "$freeze/source-paths.unsorted"
        done < <(rg --only-matching '(include_(str|bytes)!\(\s*|#\[path\s*=\s*)"[^"]+"' "$source" || test "$?" -eq 1)
    done < "$freeze/previous-paths.txt"
    sort -u "$freeze/source-paths.unsorted" > "$freeze/source-paths.txt"
    if cmp -s "$freeze/previous-paths.txt" "$freeze/source-paths.txt"; then break; fi
done
while IFS= read -r source; do
    mkdir -p "$freeze/source/$(dirname "$source")"
    cp "$source" "$freeze/source/$source"
done < "$freeze/source-paths.txt"
while IFS= read -r source; do sha256sum "$source"; done < "$freeze/source-paths.txt" > "$freeze/source.sha256"
name=$(basename "$binary")
cp "$binary" "$freeze/$name"
chmod 755 "$freeze/$name"
printf '%s\n' "$name" > "$freeze/binary-name.txt"
sha256sum "$freeze/$name" > "$freeze/binary.sha256"
rustc --version > "$freeze/rustc.version"
cargo --version > "$freeze/cargo.version"
rm "$freeze/previous-paths.txt" "$freeze/source-paths.unsorted"
cd "$freeze/source"
sha256sum -c "$freeze/source.sha256" > "$freeze/source-check.log"
