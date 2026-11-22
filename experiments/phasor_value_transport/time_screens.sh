#!/usr/bin/env bash
# Time equal scheduled work on two architectures. Every seed runs in both orders.
set -euo pipefail
if (( $# < 5 )); then
    echo 'usage: time_screens.sh CONTROL_BIN CANDIDATE_BIN RUN_DIR UPDATES CASE...' >&2
    exit 2
fi
control=$(realpath -- "$1")
candidate=$(realpath -- "$2")
mkdir -p -- "$3"
out=$(realpath -- "$3")
steps=$4
[[ $steps =~ ^[1-9][0-9]*$ ]] || { echo 'UPDATES must be a positive integer' >&2; exit 2; }
shift 4
cases=("$@")
declare -A seen=()
for task in "${cases[@]}"; do
    if [[ ! $task =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ || -n ${seen[$task]:-} ]]; then
        echo 'supply each source case name once' >&2
        exit 2
    fi
    seen[$task]=1
done
run=$(mktemp -d "$out/fixed-XXXXXXXX")
sha256sum -- "$control" "$candidate" > "$run/binaries.sha256"
lock=${PHASOR_TIMING_LOCK:-"$out/device.lock"}
echo "$run" >&2

for round in 0 1 2 3 4 5; do
    seed=$(( round % 3 ))
    for i in "${!cases[@]}"; do
        task=${cases[i]}
        pair="$run/round$round/$task"
        mkdir -p -- "$pair"
        order=(control candidate)
        if (( (round + i) % 2 )); then order=(candidate control); fi
        (
            flock 9
            for arm in "${order[@]}"; do
                bin=$control
                if [[ $arm == candidate ]]; then bin=$candidate; fi
                LC_ALL=C /usr/bin/time -f '%e' -o "$pair/$arm.seconds" \
                    "$bin" screen "$task" "$seed" "$pair/$arm" "$steps" > "$pair/$arm.json"
            done
        ) 9>"$lock"
        # Compare the entire schedule, including examples consumed in short final batches.
        for arm in control candidate; do
            jq -ce '{case,tier,seed,step,examples_seen,total}' \
                "$pair/$arm/$task/seed$seed/observations.jsonl" > "$pair/$arm.schedule"
        done
        cmp "$pair/control.schedule" "$pair/candidate.schedule"
        jq -nec --arg task "$task" --arg first "${order[0]}" --argjson seed "$seed" \
            --argjson steps "$steps" --argjson round "$round" \
            --slurpfile a "$pair/control.json" --slurpfile b "$pair/candidate.json" \
            --rawfile ta "$pair/control.seconds" --rawfile tb "$pair/candidate.seconds" '
            def environment: {gpu,kernite_version,fp32_affines,bf16,tf32_backward,suite_sha256};
            def known: (.gpu|type=="string" and length>0) and
                (.kernite_version|type=="string" and length>0) and
                (.suite_sha256|type=="string" and test("^[0-9a-f]{64}$")) and
                ([.fp32_affines,.bf16,.tf32_backward]|all(.[];type=="boolean"));
            ($ta|rtrimstr("\n")|tonumber) as $x | ($tb|rtrimstr("\n")|tonumber) as $y |
            if ([$a[0],$b[0]] | all(.[]; .status == "screen cutoff" and
                .observation.case == $task and .observation.seed == $seed and
                .observation.step == $steps and .observation.pass == false))
                and ($a[0]|known) and ($a[0]|environment) == ($b[0]|environment)
                and $x > 0 and $y > 0
            then {case:$task,seed:$seed,round:$round,first:$first,step:$steps,
                control:$a[0].model,candidate:$b[0].model,environment:($a[0]|environment),
                control_seconds:$x,candidate_seconds:$y,ratio:($y/$x)}
            else error("check the timing environments and choose a cutoff below both first crossings") end
        ' | tee -a "$run/pairs.jsonl"
    done
done
jq -s 'group_by(.case) | map({case:.[0].case,pairs:length,
    candidate_faster:map(select(.ratio < 1))|length,
    geometric_mean_time_ratio:(map(.ratio|log)|add / length|exp)})' "$run/pairs.jsonl"
