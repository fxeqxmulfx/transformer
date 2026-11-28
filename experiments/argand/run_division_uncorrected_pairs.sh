#!/usr/bin/env bash
# Compare uncorrected Adam against stock AdamW with identical moment decays.
# Six untouched controls are shared with the higher-rate validation.
set -euo pipefail
if (( $# < 3 || $# > 4 )) || [[ ${4:---prepare-only} != --prepare-only ]]; then
    echo 'usage: bash run_division_uncorrected_pairs.sh HIGHER_RATE_COMPARISON CANDIDATE_DISCOVERY_CELL OUTPUT [--prepare-only]' >&2
    exit 2
fi
dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
shared=$(realpath -e -- "$1")
pilot=$(realpath -e -- "$2")
mkdir -p -- "$3"
out=$(cd -- "$3" && pwd -P)
candidate_bin=$(realpath -e -- "${ARGAND_BIN:-$dir/target/release/argand}")
control_bin=$(realpath -e -- "${ARGAND_CONTROL_BIN:-$candidate_bin}")
reg="$dir/division-uncorrected-paired-registration.json"
task=grok-division-p97
cap=$(jq -er .first_budget "$reg")
wsd=$(jq -er .schedule_budget "$reg")
exec 9>"$out/comparison.lock"
flock -x 9
jq -nSe --slurpfile reg "$reg" --slurpfile high "$shared/protocol.json" \
    --slurpfile r "$pilot/result.json" --slurpfile v "$pilot/replay.json" --slurpfile c "$pilot/observed-certificate.json" \
    --arg candidate "$(sha256sum -- "$candidate_bin" | cut -d' ' -f1)" \
    --arg control "$(sha256sum -- "$control_bin" | cut -d' ' -f1)" '
    $reg[0] as $p | $high[0] as $h | $r[0] as $r | $v[0] as $v | $c[0] as $c |
    $h.higher_rate_control as $ctrl | ($r.identity|del(.seed,.budget,.order_seed)) as $id |
    if $candidate==$p.candidate.binary_sha256 and $control==$p.control.binary_sha256 and $control==$h.binary_sha256 and
      ($p.seeds|length)==11 and ($p.seeds|unique|length)==11 and ($p.seeds|index(0))==null and
      $p.seeds[:6]==[$p.shared_control_indices[] as $i|$h.seeds[$i]] and
      all($p.seeds[6:][]; . as $s|($h.seeds|index($s))==null) and
      $r.identity.seed==0 and $r.identity.order_seed==10000 and $r.step==$p.candidate.discovery_step and
      $r.weights_sha256==$p.candidate.discovery_weights_sha256 and $r.strict.all_strict and
      $v.identity==$r.identity and $v.step==$r.step and $v.weights_sha256==$r.weights_sha256 and $r.strict==$v.strict and
      $c.weights_sha256==$r.weights_sha256 and $c.step==$r.step and $c.hidden.strict_correct==4656 and
      $c.hidden.minimum_gap==$r.strict.min_gap and $ctrl.pilot_step==$p.control.discovery_step and
      $ctrl.identity.optimizer==$p.control.optimizer and $id.optimizer==$p.candidate.optimizer and
      ($id|del(.uncorrected_adam,.optimizer))==($ctrl.identity|del(.adamw,.optimizer)) and
      ($id.uncorrected_adam|del(.bias_correction))==$ctrl.identity.adamw and
      ($id.uncorrected_adam | .budget==$p.schedule_budget and .peak==$p.peak and .weight_decay==$p.weight_decay and
        .beta1==$p.beta1 and .beta2==$p.beta2 and .bias_correction==$p.candidate.bias_correction)
    then {registration:$p,candidate_binary_sha256:$candidate,control_binary_sha256:$control,control:$ctrl,
      candidate:{identity:$id,pilot_step:$r.step,pilot_weights:$r.weights_sha256}}
    else error("repair registration, binaries, paired schedules or discovery replay disagree") end' > "$out/protocol.pending"
if [[ -s "$out/protocol.json" ]]; then cmp "$out/protocol.pending" "$out/protocol.json"; fi
mv -- "$out/protocol.pending" "$out/protocol.json"
if (( $#==4 )); then exit 0; fi

# shellcheck source=division_optimizer_arm.sh
source "$dir/division_optimizer_arm.sh"
reuse_control() {
    local index=$1 seed=$2 old=$3 record cell interrupted
    record="$shared/$old-higher_rate_control.json"
    until [[ -s "$record" ]]; do sleep 10; done
    cell="$shared/higher_rate_control/$task/seed$seed"
    interrupted=$(jq -er '.first_attempt' "$record")
    jq -en --arg arm control --argjson seed "$seed" --argjson cap "$cap" \
        --argjson interrupted "$(if [[ $interrupted == complete ]]; then echo false; else echo true; fi)" \
        --argjson expected "$(jq -c .control.identity "$out/protocol.json")" \
        --slurpfile r "$cell/result.json" --slurpfile v "$cell/replay.json" --slurpfile c "$cell/observed-certificate.json" \
        -f "$dir/verify_optimizer_pair_arm.jq" > "$out/$index-control.pending"
    jq -es '.[0]==(.[1]|.arm="control")' "$out/$index-control.pending" "$record" >/dev/null
    mv -- "$out/$index-control.pending" "$out/$index-control.json"
}
mapfile -t indices < <(jq -r '.shared_control_indices[]' "$reg")
mapfile -t seeds < <(jq -r '.seeds[]' "$reg")
for i in "${!seeds[@]}"; do
    seed=${seeds[i]}
    if (( i<${#indices[@]} )); then
        reuse_control "$i" "$seed" "${indices[i]}"
        bin="$candidate_bin" run_arm candidate "$seed" "$i"
    else
        arms=(control candidate)
        if (( (i-${#indices[@]})%2 )); then arms=(candidate control); fi
        for arm in "${arms[@]}"; do
            if [[ $arm == control ]]; then bin=$control_bin; else bin=$candidate_bin; fi
            run_arm "$arm" "$seed" "$i"
        done
    fi
    bash "$dir/report_division_optimizer_pairs.sh" "$out" > "$out/summary.pending"
    mv -- "$out/summary.pending" "$out/summary.json"
    bash "$dir/plot_division_optimizer_pairs.sh" "$out"
done
