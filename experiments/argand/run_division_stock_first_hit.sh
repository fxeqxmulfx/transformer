#!/usr/bin/env bash
# Count ordinary first hits once per registered execution, including interruptions.
set -euo pipefail
out=${1:?usage: bash run_division_stock_first_hit.sh OUTPUT}
dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
reg="$dir/division-stock-first-hit-protocol.json"
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
cap=$(jq -er .cap "$reg")
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
        "$out/argand" adamw-scaled-until-perfect "$task" "$seed" "$trial" "$cap" 3 "$wsd" >"$trial/stdout" 2>"$trial/stderr" || true
    fi
    if [[ ! -s $cell/replay.json ]]; then
        jq -n --argjson seed "$seed" '{seed:$seed,interrupted:true,learned:false}' >"$rec.pending"
    else
        "$out/argand" division-certificate "$cell" terminal >"$trial/certificate-log"
        head -n1 "$cell/observations.jsonl" >"$trial/initial.json"
        jq -ne --argjson seed "$seed" --slurpfile p "$reg" --slurpfile init "$trial/initial.json" \
            --slurpfile res "$cell/result.json" --slurpfile replay "$cell/replay.json" --slurpfile cert "$cell/observed-certificate.json" '
          $res[0] as $r | $p[0] as $p | $init[0] as $i | $replay[0] as $v | $cert[0] as $c | $r.identity as $id |
          if $id.seed!=$seed or $id.case!=$p.case or $id.budget!=$p.cap or $id.observe!=$p.observe or $id.adamw.budget!=$p.wsd or
            $id.optimizer!="adamw-scaled-wsd" or $id.stop_on!="first-strict-100-percent" or $id.adamw.peak!=$p.peak or
            $id.adamw.weight_decay!=$p.weight_decay or $id.adamw.beta1!=$p.beta1 or $id.adamw.beta2!=$p.beta2 or
            $i.step!=0 or $i.total!=4656 or $i.identity!=$id or $v.identity!=$id or $v.step!=$r.step or
            $r.step>$p.cap or $v.strict!=$r.strict or $c.output_tokens_compared!=239 or $c.hidden.total!=4656 or $c.observed.total!=4656 or
            $v.weights_sha256!=$r.weights_sha256 or $c.weights_sha256!=$r.weights_sha256 or
            $c.hidden.strict_correct!=$r.strict.strict_correct then error("unmatched stock trial")
          else {seed:$seed,interrupted:false,initial_correct:$i.correct,step:$r.step,hidden_strict:$c.hidden.strict_correct,
            observed_strict:$c.observed.strict_correct,gap:$c.hidden.minimum_gap,weights_sha256:$r.weights_sha256,
            learned:($i.correct<4656 and $r.first_strict_step==$r.step and $r.strict.all_strict and $c.hidden.minimum_gap>0)} end' >"$rec.pending"
    fi
    mv -- "$rec.pending" "$rec"
done
jq -s '{trials:.,passed:(map(select(.learned))|length),interrupted:(map(select(.interrupted))|length),total:length}' "${records[@]}" >"$out/summary.pending"
mv -- "$out/summary.pending" "$out/summary.json"
