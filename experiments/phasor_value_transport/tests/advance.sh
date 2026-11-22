#!/usr/bin/env bash
# Compare original-budget cells with interrupted optimizer trajectories.
set -euo pipefail
if (( $# != 2 )); then
    echo 'usage: advance.sh GRAPH_WITNESS_BINARY OUTPUT_DIRECTORY' >&2
    exit 2
fi
bin=$(realpath -- "$1")
mkdir -p -- "$2"
root=$(realpath -- "$2")
run=$(mktemp -d "$root/advance-XXXXXXXX")
echo "$run" >&2
for row in 'arithmetic-abs-diff-blind 19 57' 'gpt-probe-linear-transport 32 96' 'suite-boolean-and-shift 4 12'; do
    read -r task lower upper <<< "$row"
    dir="$run/window/$task/seed0"
    for limit in "$lower" "$upper"; do
        "$bin" advance "$task" 0 "$run/window" "$limit" > "$run/$task-$limit.json"
        jq -e --argjson limit "$limit" '.status == "screen cutoff" and .observation.step == $limit' "$run/$task-$limit.json" >/dev/null
    done
    cp "$dir/progress.bin" "$run/$task-saved.bin"
    for limit in "$lower" "$upper"; do
        "$bin" advance "$task" 0 "$run/window" "$limit" > "$run/repeated.json"
        cmp "$run/$task-$limit.json" "$run/repeated.json"
    done
    cmp "$dir/progress.bin" "$run/$task-saved.bin"
    if [[ $task == arithmetic-abs-diff-blind ]]; then
        for mode in wrong-seed truncated; do
            seed=0
            if [[ $mode == wrong-seed ]]; then seed=1; fi
            dest="$run/$mode/$task/seed$seed"
            mkdir -p "$dest"
            cp "$dir/progress.bin" "$dest/progress.bin"
            if [[ $mode == truncated ]]; then truncate -s -1 "$dest/progress.bin"; fi
            if "$bin" advance "$task" "$seed" "$run/$mode" 76 > "$run/$mode.log" 2>&1; then
                echo "accepted invalid $mode progress" >&2
                exit 1
            fi
        done
        rg -q 'different model, training protocol or shape' "$run/wrong-seed.log"
        rg -q 'checksum' "$run/truncated.log"
    fi
    for mode in window straight; do
        "$bin" cell "$task" 0 "$run/$mode" > "$run/$task-$mode.json"
        "$bin" probes "$task" 0 "$run/$mode" > "$run/$task-$mode-replay.json"
    done
    for file in weights.f32 observations.jsonl; do
        cmp "$dir/$file" "$run/straight/$task/seed0/$file"
    done
    cmp "$run/$task-window-replay.json" "$run/$task-straight-replay.json"
    test ! -e "$dir/progress.bin"
    cp "$run/$task-saved.bin" "$dir/progress.bin"
    "$bin" cell "$task" 0 "$run/window" > "$run/$task-completed-replay.json"
    test ! -e "$dir/progress.bin"
    jq -c '.observation | {case,step,correct,total}' "$run/$task-window.json"
done
