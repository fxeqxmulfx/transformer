#!/usr/bin/env bash
# Compare recursive normalization with the stock controls of the registered cohort.
set -euo pipefail
if (( $# < 4 || $# > 5 )) || [[ $# == 5 && $5 != --prepare-only && $5 != --collect-only ]]; then
    echo 'usage: bash run_division_normalized_pairs.sh REPAIR_COMPARISON HIGHER_RATE_COMPARISON DISCOVERY_CELL OUTPUT [--prepare-only|--collect-only]' >&2
    exit 2
fi
dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source_root=$(realpath -e -- "$1")
high=$(realpath -e -- "$2")
pilot=$(realpath -e -- "$3")
mkdir -p -- "$4"
out=$(realpath -e -- "$4")
bin=$(realpath -e -- "${ARGAND_BIN:-$dir/target/release/argand}")
reg="$dir/division-normalized-validation.json"
source_reg="$dir/division-uncorrected-paired-registration.json"
task=grok-division-p97
cap=$(jq -er .first_budget "$reg")
wsd=$(jq -er .schedule_budget "$reg")
exec 9>"$out/comparison.lock"
flock -x 9
jq -nSe --slurpfile p "$reg" --slurpfile original "$source_reg" --slurpfile src "$source_root/protocol.json" \
    --slurpfile h "$high/protocol.json" --slurpfile r "$pilot/result.json" --slurpfile v "$pilot/replay.json" \
    --slurpfile c "$pilot/observed-certificate.json" --arg digest "$(sha256sum -- "$bin" | cut -d' ' -f1)" \
    --arg registration "$(sha256sum -- "$source_reg" | cut -d' ' -f1)" '
    $p[0] as $p | $src[0] as $src | $r[0] as $r | $v[0] as $v | $c[0] as $c |
    ($r.identity|del(.seed,.budget,.order_seed)) as $id | $src.control as $ctrl |
    if $p.binary_sha256==$digest and $p.source_registration_sha256==$registration and
      $src.registration==$original[0] and $original[0].seeds==$p.seeds and
      $h[0].higher_rate_control==$ctrl and $h[0].binary_sha256==$src.control_binary_sha256 and
      ($p.seeds|length)==11 and ($p.seeds|unique|length)==11 and
      $r.identity.seed==$p.discovery_seed and ($p.seeds|index($p.discovery_seed))==null and
      $r.step==$p.discovery_step and $r.weights_sha256==$p.discovery_weights_sha256 and $r.strict.all_strict and
      $v.identity==$r.identity and $v.step==$r.step and $v.weights_sha256==$r.weights_sha256 and $v.strict==$r.strict and
      $c.step==$r.step and $c.weights_sha256==$r.weights_sha256 and $c.hidden.minimum_gap==$r.strict.min_gap and
      $id.optimizer==$p.optimizer and ($id|del(.normalized_adamw,.optimizer))==($ctrl.identity|del(.adamw,.optimizer)) and
      ($id.normalized_adamw|del(.normalization,.bias_correction))==$ctrl.identity.adamw and
      ($id.normalized_adamw|.normalization==$p.normalization and .bias_correction==true and
       .peak==$p.peak and .weight_decay==$p.weight_decay and .beta1==$p.beta1 and .beta2==$p.beta2 and .budget==$p.schedule_budget)
    then {registration:$p,candidate_binary_sha256:$digest,control_binary_sha256:$src.control_binary_sha256,
      control:$ctrl,candidate:{identity:$id,pilot_step:$r.step,pilot_weights:$r.weights_sha256}}
    else error("normalized moment registration, binary, discovery or control mismatch") end' > "$out/protocol.pending"
if [[ -s "$out/protocol.json" ]]; then cmp -- "$out/protocol.pending" "$out/protocol.json"; fi
mv -- "$out/protocol.pending" "$out/protocol.json"
mode=${5:-train}
if [[ $mode == --prepare-only ]]; then exit 0; fi

source "$dir/division_optimizer_arm.sh"
mapfile -t seeds < <(jq -r '.seeds[]' "$reg")
mapfile -t shared_indices < <(jq -r '.shared_control_indices[]' "$source_reg")
expected=$(jq -c '.control.identity' "$out/protocol.json")
for i in "${!seeds[@]}"; do
    seed=${seeds[i]}
    record="$source_root/$i-control.json"
    if [[ $mode == --collect-only && ! -s $record ]]; then continue; fi
    until [[ -s "$record" ]]; do sleep 8; done
    if (( i<${#shared_indices[@]} )); then cell="$high/higher_rate_control/$task/seed$seed"
    else cell="$source_root/control/$task/seed$seed"; fi
    (cd -- "$cell" && jq -r '"\(.weights_sha256)  weights.f32"' "$record" | sha256sum --check --status)
    interrupted=$(jq -c '.first_attempt!="complete"' "$record")
    jq -en --arg arm control --argjson seed "$seed" --argjson cap "$cap" --argjson expected "$expected" \
        --argjson interrupted "$interrupted" --slurpfile r "$cell/result.json" --slurpfile v "$cell/replay.json" \
        --slurpfile c "$cell/observed-certificate.json" -f "$dir/verify_optimizer_pair_arm.jq" > "$out/$i-control.pending"
    cmp -- "$out/$i-control.pending" "$record"
    mv -- "$out/$i-control.pending" "$out/$i-control.json"
    if [[ $mode == --collect-only && ( ! -s "$out/candidate/$task/seed$seed/replay.json" || ! -s "$out/candidate/$task/seed$seed/observed-certificate.json" ) ]]; then continue; fi
    run_arm candidate "$seed" "$i"
    bash "$dir/report_division_optimizer_pairs.sh" "$out" > "$out/summary.pending"
    mv -- "$out/summary.pending" "$out/summary.json"
    bash "$dir/plot_division_optimizer_pairs.sh" "$out"
done
if [[ $mode == --collect-only ]]; then
    bash "$dir/report_division_optimizer_pairs.sh" "$out" > "$out/summary.pending"
    mv -- "$out/summary.pending" "$out/summary.json"
    if [[ -s "$out/0-control.json" ]]; then bash "$dir/plot_division_optimizer_pairs.sh" "$out"; fi
fi
