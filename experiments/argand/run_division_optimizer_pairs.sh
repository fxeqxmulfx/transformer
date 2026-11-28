#!/usr/bin/env bash
# Lock the candidate using only discovery seed 0, then alternate which arm
# trains first on each of the eleven previously registered model seeds.
set -euo pipefail
if (( $# != 3 )); then echo 'usage: bash run_division_optimizer_pairs.sh ADAMW_DISCOVERY_CELL CANDIDATE_DISCOVERY_CELL OUTPUT' >&2; exit 2; fi
dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
bin=$(realpath -e -- "${ARGAND_BIN:-$dir/target/release/argand}")
control=$(realpath -e -- "$1")
candidate=$(realpath -e -- "$2")
mkdir -p -- "$3"
out=$(cd -- "$3" && pwd -P)
task=grok-division-p97
exec 9>"$out/comparison.lock"
flock -x 9
reg="$dir/division-optimizer-paired-registration.json"
cap=$(jq -er '.first_budget' "$reg")
wsd=$(jq -er '.schedule_budget' "$reg")
for cell in "$control" "$candidate"; do
    jq -es '.[0] as $r | .[1] as $v | .[2] as $c |
      $r.identity.seed==0 and $c.seed==0 and $r.identity.order_seed==10000 and
      $r.identity==$v.identity and $r.strict==$v.strict and $r.strict.all_strict and
      $r.step==$v.step and $r.step==$c.step and $r.weights_sha256==$c.weights_sha256 and
      $r.weights_sha256==$v.weights_sha256 and $c.hidden.total==4656 and
      $c.hidden.strict_correct==4656 and $c.hidden.minimum_gap==$r.strict.min_gap and $c.hidden.minimum_gap>0' \
      "$cell/result.json" "$cell/replay.json" "$cell/observed-certificate.json" >/dev/null
done
jq -nSe --slurpfile reg "$reg" --slurpfile a "$control/result.json" --slurpfile c "$candidate/result.json" \
  --arg binary "$(sha256sum -- "$bin" | cut -d' ' -f1)" '
  def settings: .adamw // .equal_moment // .rmsprop // .softsign_lion // .muon;
  def arm: {identity:(.identity|del(.seed,.order_seed,.budget)),pilot_step:.step,pilot_weights:.weights_sha256};
  def shared: .identity | del(.optimizer,.budget,.adamw,.equal_moment,.rmsprop,.softsign_lion,.muon,.checkpoint_proposals);
  $reg[0] as $p | $a[0] as $a | $c[0] as $c |
  if ($p.seeds|length)==11 and ($p.seeds|unique|length)==11 and ($p.seeds|index(0))==null and
    $a.identity.optimizer==$p.baseline.optimizer and $a.identity.adamw.weight_decay==$p.baseline.weight_decay and
    $a.identity.adamw.peak==(0.001*$p.baseline.rate_multiplier) and $a.identity.case==$p.case and
    $a.identity.observe==$p.evaluation_interval and ($a|shared)==($c|shared) and
    ($a.identity|settings|.budget)==$p.schedule_budget and ($c.identity|settings|.budget)==$p.schedule_budget and
    $c.step<=$a.step and $c.identity.optimizer!=$a.identity.optimizer
  then {registration:$p,binary_sha256:$binary,control:($a|arm),candidate:($c|arm)}
  else error("discovery pair did not qualify under the registered design") end' > "$out/protocol.pending"
if [[ -s "$out/protocol.json" ]]; then cmp -- "$out/protocol.pending" "$out/protocol.json"; fi
mv -- "$out/protocol.pending" "$out/protocol.json"

# shellcheck source=division_optimizer_arm.sh
source "$dir/division_optimizer_arm.sh"

mapfile -t seeds < <(jq -r '.registration.seeds[]' "$out/protocol.json")
for i in "${!seeds[@]}"; do
    arms=(control candidate)
    if (( i % 2 )); then arms=(candidate control); fi
    for arm in "${arms[@]}"; do run_arm "$arm" "${seeds[i]}" "$i"; done
    jq -s 'def counts: {evaluated:length,first_attempt_hits:(map(select(.first_attempt=="complete" and .success))|length),
        misses:(map(select(.first_attempt=="complete" and (.success|not)))|length),
        interrupted:(map(select(.first_attempt=="interrupted"))|length)};
      {registered:11,control:(map(select(.arm=="control"))|counts),candidate:(map(select(.arm=="candidate"))|counts),
       pairs:(group_by(.seed)|map(select(length==2)| (map({key:.arm,value:.})|from_entries) |
         {seed:.control.seed,control:.control.step,candidate:.candidate.step,
          difference:(if all(.[]; .success and .first_attempt=="complete") then .candidate.step-.control.step else null end)}))}' \
      "$out"/[0-9]*-{control,candidate}.json > "$out/summary.pending"
    mv -- "$out/summary.pending" "$out/summary.json"
done
