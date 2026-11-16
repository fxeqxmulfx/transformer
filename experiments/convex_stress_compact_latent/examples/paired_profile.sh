#!/usr/bin/env bash
# Alternate two already built fixed-fixture profilers on one allowed CPU.
set -euo pipefail
if test "$#" -ne 4; then
    printf '%s\n' 'usage: paired_profile.sh BEFORE_BINARY AFTER_BINARY NEW_OUTPUT CPU' >&2
    exit 2
fi
before=$1
after=$2
output=$3
cpu=$4
test -x "$before"
test -x "$after"
test -n "${SHARED_COLUMNS_WORKSPACE:?set workspace path}"
test -n "${SHARED_COLUMNS_STORAGE:?set storage path}"
analysis=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/compare_phases.jq
mkdir -- "$output"
trap 'printf "%s\n" "$?" > "$output/exit"' EXIT
sha256sum -- "$before" "$after" > "$output/binaries.sha256"
printf '%s\n' "$cpu" > "$output/cpu.txt"
for round in 1 2 3 4 5 6 7; do
    modes=(before after)
    if test "$((round % 2))" -eq 0; then modes=(after before); fi
    for mode in "${modes[@]}"; do
        binary=$before
        if test "$mode" = after; then binary=$after; fi
        /usr/bin/time -v taskset -c "$cpu" prlimit --as=4294967296:4294967296 -- "$binary" \
            > "$output/$mode-$round.jsonl" 2> "$output/$mode-$round.resources"
    done
    jq -n --slurpfile before "$output/before-$round.jsonl" \
        --slurpfile after "$output/after-$round.jsonl" -f "$analysis" > "$output/pair-$round.json"
done
# Include every predeclared pair. Report median ratios and the full observed range.
jq -s 'flatten | group_by([.programs,.classes]) | map(
  . as $rows | {programs:.[0].programs,classes:.[0].classes,pairs:length,
    all_words_and_work_equal:all(.complete_endpoint_words_and_work_equal),
    speedups:(["head_speedup","known_prediction_speedup","exact_prediction_speedup",
      "forward_total_speedup","minibatch_total_speedup"] | map(. as $key |
      ($rows|map(.[$key])|sort) as $values |
      {phase:$key,median:$values[3],minimum:$values[0],maximum:$values[6]}))})' \
    "$output"/pair-*.json > "$output/summary.json"
sha256sum --check "$output/binaries.sha256" > "$output/binaries-after.txt"
