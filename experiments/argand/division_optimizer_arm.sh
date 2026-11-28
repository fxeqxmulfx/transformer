# Shared by both frozen comparisons. The caller supplies the protocol, output,
# binary, WSD schedule and task before running a registered arm.
run_arm() {
    local arm=$1 seed=$2 index=$3 expected rule scale decay first proposal radius cell root archive output interrupted
    local -a args
    root="$out/$arm"; output="$out/$index-$arm.json"
    expected=$(jq -c --arg arm "$arm" '.[$arm].identity' "$out/protocol.json")
    rule=$(jq -r .optimizer <<< "$expected")
    scale=$(jq -r '(.adamw//.softsign_lion//.equal_moment//.uncorrected_adam//.muon//.rmsprop).peak*1000' <<< "$expected")
    decay=$(jq -r '(.adamw//.softsign_lion//.equal_moment//.uncorrected_adam//.muon//.rmsprop).weight_decay' <<< "$expected")
    cell="$root/$task/seed$seed"
    case "$rule" in
      adamw-scaled-wsd) args=(adamw-scaled-until-perfect) ;;
      equal-moment-wsd) args=(equal-moment-until-perfect) ;;
      uncorrected-adam-wsd) args=(uncorrected-adam-until-perfect) ;;
      softsign-lion-wsd) args=(softsign-until-perfect) ;;
      complex-muon-wsd) args=(muon-until-perfect) ;;
      softsign-lion-wsd-checkpoint-proposals|row-clipped-softsign-lion-wsd-checkpoint-proposals) args=(softsign-checkpoint-proposals) ;;
      rmsprop-wsd-checkpoint-proposals) args=(rmsprop-checkpoint-proposals) ;;
      equal-moment-wsd-checkpoint-proposals) args=(equal-moment-checkpoint-proposals) ;;
      *) echo "unsupported paired optimizer $rule" >&2; return 1 ;;
    esac
    if [[ $rule == *-checkpoint-proposals ]]; then
        first=$(jq -er '.checkpoint_proposals.first_eligible_update' <<< "$expected")
        proposal=$(jq -er '.checkpoint_proposals.seed' <<< "$expected")
        args+=("$task" "$seed" "$proposal" "$root" "$cap" "$first" "$scale" "$wsd" "$decay")
        radius=$(jq -r '.softsign_lion.max_projection_row_norm//empty' <<< "$expected")
        if [[ -n $radius ]]; then args+=("$radius"); fi
        cell="$cell-proposal$proposal-from$first"
    else
        args+=("$task" "$seed" "$root" "$cap" "$scale" "$wsd")
    fi
    if jq -e '.gradient_reduction=="ordered-f32-v1"' <<< "$expected" >/dev/null; then args+=(--ordered-gradients); fi
    mkdir -p -- "$cell"
    (
        flock -x 8
        [[ ! -s "$cell/result.json" && ( -e "$cell/observations.jsonl" || -e "$cell/gpu-start.csv" ) ]] || exit 0
        mkdir -p -- "$cell/interrupted"
        archive=$(mktemp -d "$cell/interrupted/attempt.XXXXXXXX")
        for f in "$cell"/{observations.jsonl,weights.f32,gpu-start.csv,gpu-end.csv} "$cell"/*.pending; do
            [[ ! -f $f ]] || mv -- "$f" "$archive/"
        done
    ) 8>>"$cell/cell.lock"
    if [[ ! -s "$cell/replay.json" ]]; then
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$cell/gpu-start.csv"
        "$bin" "${args[@]}" > /dev/null
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$cell/gpu-end.csv"
    fi
    if [[ ! -s "$cell/observed-certificate.json" ]]; then "$bin" division-certificate "$cell" terminal >/dev/null; fi
    interrupted=false; if [[ -d "$cell/interrupted" ]]; then interrupted=true; fi
    jq -en --arg arm "$arm" --argjson seed "$seed" --argjson expected "$expected" --argjson cap "$cap" \
        --argjson interrupted "$interrupted" --slurpfile r "$cell/result.json" --slurpfile v "$cell/replay.json" \
        --slurpfile c "$cell/observed-certificate.json" -f "$dir/verify_optimizer_pair_arm.jq" > "$output.pending"
    mv -- "$output.pending" "$output"
    jq -c '{arm,seed,step,success,first_attempt}' "$output"
}
