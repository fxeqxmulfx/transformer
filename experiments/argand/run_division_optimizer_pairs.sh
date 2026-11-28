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

run_arm() {
    local arm=$1 seed=$2 index=$3 expected rule scale decay first proposal radius cell root archive output interrupted
    local -a args
    root="$out/$arm"; output="$out/$index-$arm.json"
    expected=$(jq -c --arg arm "$arm" '.[$arm].identity' "$out/protocol.json")
    rule=$(jq -r .optimizer <<< "$expected")
    scale=$(jq -r '(.adamw//.softsign_lion//.equal_moment//.muon//.rmsprop).peak*1000' <<< "$expected")
    decay=$(jq -r '(.adamw//.softsign_lion//.equal_moment//.muon//.rmsprop).weight_decay' <<< "$expected")
    cell="$root/$task/seed$seed"
    case "$rule" in
      adamw-scaled-wsd) args=(adamw-scaled-until-perfect) ;;
      equal-moment-wsd) args=(equal-moment-until-perfect) ;;
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
