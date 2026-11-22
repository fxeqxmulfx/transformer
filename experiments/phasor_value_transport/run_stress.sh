#!/usr/bin/env bash
# Train, replay or score source test splits for NV11, the GPT tiers and additions.
set -euo pipefail

nv11=(arithmetic-cubic-product)
if [[ ${1:-} == --gpt-only ]]; then
    nv11=()
    shift
fi
if [[ $# -lt 2 || $# -gt 3 || ( $1 != cell && $1 != replay && $1 != probes ) ]]; then
    echo 'usage: run_stress.sh [--gpt-only] {cell|replay|probes} RUN_DIR [extended|EXTENSIONS.tsv]' >&2
    exit 2
fi
dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bin=${PHASOR_BIN:-"$dir/target/release/phasor-value-transport"}
routes=$("$bin" routes)
if [[ ${3:-} == extended ]]; then
    routes+=$'\n'"$("$bin" extensions)"
elif (( $# == 3 )); then
    routes+=$'\n'"$("$bin" extensions "$3")"
fi
mapfile -t names < <(awk -F '\t' 'NF {print $1}' <<< "$routes")
for name in "${nv11[@]}" "${names[@]}"; do
    for seed in 0 1 2; do
        "$bin" "$1" "$name" "$seed" "$2"
    done
done
