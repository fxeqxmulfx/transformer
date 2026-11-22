#!/usr/bin/env bash
# Time numerically identical builds on fresh cells, alternating their order.
set -euo pipefail

if (( $# < 4 )); then
    echo 'usage: time_cells.sh BEFORE_BIN AFTER_BIN RUN_DIR CASE...' >&2
    exit 2
fi
before=$(realpath -- "$1")
after=$(realpath -- "$2")
mkdir -p -- "$3"
out=$(realpath -- "$3")
run=$(mktemp -d "$out/paired-XXXXXXXX")
lock=${PHASOR_TIMING_LOCK:-"$out/device.lock"}
shift 3
cases=("$@")
echo "$run" >&2

for seed in 0 1 2; do
    for index in "${!cases[@]}"; do
        task=${cases[index]}
        order=(before after)
        if (( (seed + index) % 2 == 1 )); then order=(after before); fi
        (
            flock 9
            for arm in "${order[@]}"; do
                bin=$before
                if [[ $arm == after ]]; then bin=$after; fi
                "$bin" cell "$task" "$seed" "$run/$arm" > "$run/$arm-cell.json"
            done
        ) 9>"$lock"
        for artifact in weights.f32 observations.jsonl; do
            cmp "$run/before/$task/seed$seed/$artifact" "$run/after/$task/seed$seed/$artifact"
        done
        jq -nce --slurpfile a "$run/before-cell.json" --slurpfile b "$run/after-cell.json" '
            $a[0] as $a | $b[0] as $b |
            if $a.observation == $b.observation and $a.suite_sha256 == $b.suite_sha256
               and $a.fp32_affines == $b.fp32_affines and $a.gpu == $b.gpu
               and $a.elapsed_seconds > 0 and $b.elapsed_seconds > 0
            then {case:$a.observation.case,seed:$a.observation.seed,step:$a.observation.step,
                  before_seconds:$a.elapsed_seconds,after_seconds:$b.elapsed_seconds,
                  ratio:($b.elapsed_seconds / $a.elapsed_seconds),weights_sha256:$a.weights_sha256}
            else error("cell inputs, observations, precision, device or timings differ unexpectedly") end
        ' | tee -a "$run/pairs.jsonl"
    done
done
