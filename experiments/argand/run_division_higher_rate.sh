#!/usr/bin/env bash
# Add the preregistered 0.003 AdamW control. Each trained candidate is shared
# with the primary comparison, so no candidate is retrained on these seeds.
set -euo pipefail
if (( $# != 3 )); then echo 'usage: bash run_division_higher_rate.sh PRIMARY_COMPARISON ADAMW_003_DISCOVERY_CELL OUTPUT' >&2; exit 2; fi
dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
bin=$(realpath -e -- "${ARGAND_BIN:-$dir/target/release/argand}")
primary=$(realpath -e -- "$1")
cell=$(realpath -e -- "$2")
mkdir -p -- "$3"
out=$(cd -- "$3" && pwd -P)
exec 9>"$out/comparison.lock"
flock -x 9
reg="$dir/division-optimizer-higher-rate-registration.json"
original="$dir/division-optimizer-paired-registration.json"
digest=$(sha256sum -- "$original" | cut -d' ' -f1)
binary=$(sha256sum -- "$bin" | cut -d' ' -f1)
task=grok-division-p97
cap=$(jq -er '.first_budget' "$reg")
wsd=$(jq -er '.schedule_budget' "$reg")
jq -nSe --slurpfile reg "$reg" --slurpfile orig "$original" --slurpfile primary "$primary/protocol.json" \
  --slurpfile r "$cell/result.json" --slurpfile v "$cell/replay.json" --slurpfile c "$cell/observed-certificate.json" \
  --arg digest "$digest" --arg binary "$binary" '
  $primary[0] as $p | $reg[0] as $reg | $reg.control as $ctrl | $r[0] as $r | $r.identity as $id |
  if $p.registration==$orig[0] and $reg.primary_registration_sha256==$digest and $p.binary_sha256==$binary and
    $r.strict.all_strict and $r.strict==$v[0].strict and $r.identity==$v[0].identity and
    $r.weights_sha256==$v[0].weights_sha256 and $r.weights_sha256==$c[0].weights_sha256 and
    $c[0].hidden.total==4656 and $c[0].hidden.strict_correct==4656 and $c[0].hidden.minimum_gap==$r.strict.min_gap and
    $id.case==$reg.case and $id.seed==0 and $id.optimizer==$ctrl.optimizer and $id.adamw.budget==$reg.schedule_budget and
    $id.adamw.beta1==$ctrl.beta1 and $id.adamw.beta2==$ctrl.beta2 and $id.adamw.weight_decay==$ctrl.weight_decay and
    $id.adamw.peak==($ctrl.rate_multiplier*0.001) and $id.observe==$reg.evaluation_interval and
    ($p.control.identity|del(.adamw))==($id|del(.adamw,.seed,.order_seed,.budget))
  then {registration:$reg,seeds:$p.registration.seeds,candidate:$p.candidate,binary_sha256:$binary,
        higher_rate_control:{identity:($id|del(.seed,.order_seed,.budget)),pilot_step:$r.step,pilot_weights:$r.weights_sha256}}
  else error("higher-rate registration, training binary or discovery evidence disagrees") end' > "$out/protocol.pending"
if [[ -s "$out/protocol.json" ]]; then cmp -- "$out/protocol.pending" "$out/protocol.json"; fi
mv -- "$out/protocol.pending" "$out/protocol.json"
# shellcheck source=division_optimizer_arm.sh
source "$dir/division_optimizer_arm.sh"
mapfile -t seeds < <(jq -r '.seeds[]' "$out/protocol.json")
for i in "${!seeds[@]}"; do run_arm higher_rate_control "${seeds[i]}" "$i"; done
