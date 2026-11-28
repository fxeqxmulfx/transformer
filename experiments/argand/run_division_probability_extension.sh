#!/usr/bin/env bash
# Fixed-100k all-success test. Start the next seed only while the test
# can still reject; a longer diagnostic cannot earn credit for this test.
set -euo pipefail
if (( $# != 2 )); then echo 'usage: bash run_division_probability_extension.sh PRIMARY OUTPUT' >&2; exit 2; fi
dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
primary=$(realpath -e -- "$1")
mkdir -p -- "$2"
out=$(realpath -e -- "$2")
bin=${ARGAND_BIN:-"$dir/target/release/argand"}
plan="$dir/division-first-hit-registration.json"
exec 9>"$out/extension.lock"
flock -x 9
base=$(bash "$dir/report_division_first_hits.sh" "$primary")
jq -en --argjson base "$base" --slurpfile plan "$plan" '
  $plan[0].three_quarters_test as $t |
  $base.updates==$t.wsd_budget and $base.all_draws.evaluated==12 and $base.prospective.evaluated==8 and
  $t.continue_only_after_every_prospective_success and
  (($plan[0].model_seeds+$t.additional_model_seeds) | length==(unique|length)) and
  $base.prospective.registered+($t.additional_model_seeds|length)==$t.required_consecutive_first_attempt_successes
  ' >/dev/null
if [[ -s "$out/registration.json" ]]; then
    cmp "$plan" "$out/registration.json"
else
    cp -- "$plan" "$out/registration.pending"
    mv -- "$out/registration.pending" "$out/registration.json"
fi
records=()
test_cap=$(jq -r '.three_quarters_test.wsd_budget' "$plan")
required=$(jq -r '.three_quarters_test.required_consecutive_first_attempt_successes' "$plan")
complete=$(jq -r '.prospective.first_attempt_successes_for_test' <<< "$base")
missed=''
run_cell() {
    local seed=$1 cap=$2 i=$3 record=$4 root="$out/wsd-$2"
    local cell="$root/grok-division-p97/seed$seed"
    mkdir -p -- "$cell"
    (
        flock -n 8 || exit 0
        [[ ! -s "$cell/result.json" && -s "$cell/observations.jsonl" ]] || exit 0
        mkdir -p "$cell/interrupted"
        archive=$(mktemp -d "$cell/interrupted/attempt.XXXXXXXX")
        for f in "$cell"/weights.f32 "$cell"/*.pending "$cell"/observations.jsonl; do
            [[ ! -f $f ]] || mv -- "$f" "$archive/"
        done
    ) 8>>"$cell/cell.lock"
    "$bin" adamw-until-perfect grok-division-p97 "$seed" "$root" "$cap" >/dev/null
    "$bin" division-certificate "$cell" terminal >/dev/null
    local clean=true
    for trace in "$cell"/interrupted/attempt.*/observations.jsonl; do
        if [[ -s $trace ]]; then clean=false; fi
    done
    jq -en --argjson seed "$seed" --argjson i "$i" --argjson cap "$cap" --arg cell "$cell" --arg kind terminal \
        --slurpfile r "$cell/replay.json" --slurpfile c "$cell/observed-certificate.json" \
        -f "$dir/verify_division_first_hit.jq" |
        jq --argjson clean "$clean" '.+{first_attempt:$clean}' > "$record.pending"
    mv -- "$record.pending" "$record"
}
if (( complete == 8 )); then
    while read -r seed; do
        i=${#records[@]}
        record="$out/draw-$i.json"
        run_cell "$seed" "$test_cap" "$i" "$record"
        records+=("$record")
        if ! jq -e '.success' "$record" >/dev/null; then missed=$seed; break; fi
        if ! jq -e '.first_attempt' "$record" >/dev/null; then break; fi
        ((complete+=1))
    done < <(jq -r '.three_quarters_test.additional_model_seeds[]' "$plan")
fi
for record in "${records[@]}"; do cat -- "$record"; done |
    jq -es --argjson base "$base" --argjson k "$complete" --argjson n "$required" '
    {fixed_wsd_budget:$base.updates,original_cohort:$base.all_draws,prospective_prefix:$base.prospective,
     additional_draws:.,required_first_attempt_successes:$n,credited_first_attempt_successes:$k,
     rejects_single_run_at_most_three_quarters:($k==$n),null_bound_for_all_successes:pow(0.75;$n)}
    ' > "$out/summary.pending"
mv -- "$out/summary.pending" "$out/summary.json"
if [[ -n $missed ]]; then
    run_cell "$missed" 200000 "$i" "$out/longer-diagnostic.json"
fi
