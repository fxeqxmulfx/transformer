#!/usr/bin/env bash
# Train each registered seed once to the fixed horizon and score strict hidden
# predictions at every observation, including interruptions as failures.
set -euo pipefail
out=${1:?usage: bash run_division_stock_persistence.sh OUTPUT}
dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
reg="$dir/division-stock-persistence-protocol.json"
mkdir -p -- "$out"
exec 9>"$out/lock"
flock -x 9
if [[ ! -f $out/argand ]]; then
    cp -- "${ARGAND_BIN:-$dir/target/release/argand}" "$out/argand.pending"
    mv -- "$out/argand.pending" "$out/argand"
fi
expected=$(jq -er .binary_sha256 "$reg")
actual=$(sha256sum -- "$out/argand" | cut -d' ' -f1)
[[ $actual == "$expected" ]]
if [[ -f $out/protocol.json ]]; then cmp -- "$reg" "$out/protocol.json"; else cp -- "$reg" "$out/protocol.json"; fi
task=$(jq -er .case "$reg")
horizon=$(jq -er .horizon "$reg")
wsd=$(jq -er .wsd "$reg")
mapfile -t seeds < <(jq -r '.seeds[]' "$reg")
records=()
for i in "${!seeds[@]}"; do
    seed=${seeds[i]}
    trial="$out/trial-$i"
    mkdir -p -- "$trial"
    cell="$trial/$task/seed$seed"
    rec="$trial/verified.json"
    records+=("$rec")
    if [[ -f $rec ]]; then continue; fi
    if [[ ! -e $trial/attempted ]]; then
        touch "$trial/attempted"
        "$out/argand" adamw-scaled-run "$task" "$seed" "$trial" "$horizon" 3 "$wsd" --ordered-gradients \
            >"$trial/stdout" 2>"$trial/stderr" || true
    fi
    if [[ ! -s $cell/replay.json ]]; then
        jq -n --argjson seed "$seed" '{seed:$seed,interrupted:true,learned:false,no_relapse:false}' >"$rec.pending"
    else
        "$out/argand" division-certificate "$cell" terminal >"$trial/certificate-log"
        first='{}'
        if [[ -f $cell/first-strict.json ]]; then
            "$out/argand" division-certificate "$cell" first-strict >"$trial/first-certificate-log"
            first=$(cat "$cell/first-strict-certificate.json")
        fi
        jq -ne --argjson seed "$seed" --slurpfile p "$reg" --slurpfile o "$cell/observations.jsonl" \
            --slurpfile res "$cell/result.json" --slurpfile replay "$cell/replay.json" \
            --slurpfile cert "$cell/observed-certificate.json" --argjson f "$first" '
          $res[0] as $r | $p[0] as $p | $replay[0] as $v | $cert[0] as $c | $r.identity as $id |
          ($p.latest_entry / $p.observe | floor * $p.observe) as $entry |
          ($r.persistence.first_strict_observation) as $first |
          [$o[] | select(.strict.all_strict != true) | .step] as $misses |
          if $id.seed!=$seed or $id.case!=$p.case or $id.budget!=$p.horizon or $id.observe!=$p.observe or
            $id.adamw.budget!=$p.wsd or $id.optimizer!="adamw-scaled-wsd" or $id.fixed_horizon!=true or
            $id.strict_every_observation!=true or $id.gradient_reduction!=$p.gradient_reduction or
            $id.adamw.peak!=$p.peak or $id.adamw.weight_decay!=$p.weight_decay or $id.adamw.beta1!=$p.beta1 or
            $id.adamw.beta2!=$p.beta2 or ([$o[].step] != [range(0; $p.horizon+1; $p.observe)]) or
            any($o[]; .identity!=$id or .strict.total!=4656 or .total!=4656) or
            $o[-1].step!=$r.step or $o[-1].strict!=$r.strict or $r.step!=$p.horizon or
            $v.identity!=$id or $v.step!=$r.step or $v.strict!=$r.strict or $v.weights_sha256!=$r.weights_sha256 or
            $c.weights_sha256!=$r.weights_sha256 or $c.output_tokens_compared!=239 or $c.hidden.total!=4656 or
            $c.observed.total!=4656 or $c.hidden.strict_correct!=$r.strict.strict_correct or
            ($first!=null and $first>0 and ($f.step!=$first or $f.hidden.strict_correct!=4656 or $f.output_tokens_compared!=239))
          then error("unmatched persistence trial")
          else {seed:$seed,interrupted:false,initial_correct:$o[0].correct,initial_all_strict:$o[0].strict.all_strict,
            initial_gap:$o[0].strict.min_gap,first_strict:$first,
            non_strict_after_first:[$misses[] | select($first!=null and .>$first)],
            last_non_strict:($misses | max),least_gap_from_first:$r.persistence.least_gap_from_first,
            terminal_hidden_strict:$c.hidden.strict_correct,terminal_gap:$c.hidden.minimum_gap,
            terminal_observed_strict:$c.observed.strict_correct,terminal_observed_risk_bound:$c.observed_rule_risk.upper_bound,
            first_gap:$f.hidden.minimum_gap,weights_sha256:$r.weights_sha256,
            learned:($o[0].correct<4656 and all($o[] | select(.step>=$entry); .strict.all_strict==true) and
              $c.hidden.strict_correct==4656 and $c.hidden.minimum_gap>0),
            no_relapse:($o[0].correct<4656 and $first!=null and $r.persistence.persistent_through_horizon and
              $c.hidden.minimum_gap>0)} end' >"$rec.pending"
    fi
    mv -- "$rec.pending" "$rec"
done
jq -s '{trials:.,passed:(map(select(.learned))|length),no_relapse:(map(select(.no_relapse))|length),
  interrupted:(map(select(.interrupted))|length),total:length}' "${records[@]}" >"$out/summary.pending"
mv -- "$out/summary.pending" "$out/summary.json"
