#!/usr/bin/env bash
# Count strict fixed-horizon successes after both checkpoint replay and
# reconstruction of the entire label table from the training equations.
set -euo pipefail
if (( $# != 1 )); then echo 'usage: bash report_division_horizon.sh COHORT' >&2; exit 2; fi
cd -- "$1"
summary=/dev/null
if [[ -s confirmation.json ]]; then summary=confirmation.json; fi
for ((i=0; i<12; i++)); do
    cell="draw-$i"
    [[ -s "$cell/replay.json" && -s "$cell/observed-certificate.json" ]] || continue
    interrupted=0
    for prefix in "$cell"/interrupted/step-*/observations.jsonl; do
        if [[ -s "$prefix" ]]; then ((interrupted+=1)); fi
    done
    jq -cn --argjson i "$i" --argjson interrupted "$interrupted" \
        --slurpfile r "$cell/replay.json" --slurpfile c "$cell/observed-certificate.json" \
        '{index:$i,interrupted:$interrupted,replay:$r[0],certificate:$c[0]}'
done | jq -es --slurpfile plan protocol.json --slurpfile summary "$summary" '
    $plan[0] as $p |
    if $p.case!="grok-division-p97" or $p.correction!="adamw-wsd" or $p.n!=12 or
       ($p.seeds|length)!=12 or ($p.updates|IN(100000,200000)|not)
    then error("unexpected division registration") else . end |
    def valid_count: type=="number" and .==floor and .>=0 and .<=4656;
    map(.index as $i | .replay as $r | .certificate as $c | $r.strict as $s |
        if $r.identity.seed!=$p.seeds[$i][0] or $r.identity.order_seed!=(($r.identity.seed+10000)%4294967296) or
           $r.identity.case!=$p.case or $r.identity.optimizer!=$p.correction or
           ($r.identity.fixed_horizon|not) or $r.step!=$p.updates or $r.identity.budget!=$p.updates or
           $r.identity.adamw.weight_decay!=0.1 or $r.identity.adamw.budget!=$p.updates or
           $c.case!=$p.case or $c.step!=$r.step or $c.seed!=$r.identity.seed or
           $r.weights_sha256!=$c.weights_sha256 or ($r.weights_sha256|test("^[0-9a-f]{64}$")|not) or
           $c.observed_sha256!="53e9d2fb9336a74529e88bf582a2fb37483c1f7e79ab8f1fd670de565baf0eef" or
           $c.observed_sha256!=$r.identity.split_sha256[0] or $c.output_tokens_compared!=239 or
           $s.total!=4656 or $c.hidden.total!=4656 or $r.splits.hidden.total!=4656 or $c.observed.total!=4656 or
           ([$s.strict_correct,$s.ties,$c.observed.strict_correct,$r.splits.hidden.correct]|all(.[];valid_count)|not) or
           $s.strict_correct!=$c.hidden.strict_correct or $s.ties!=$c.hidden.ties or
           $s.all_strict!=($s.strict_correct==4656 and $s.min_gap>0 and $s.ties==0) or
           $s.all_strict!=($c.hidden.minimum_gap>0) or ($s.all_strict and $r.splits.hidden.correct!=4656) or
           $c.accept!=($s.all_strict and $c.observed.strict_correct==4656)
        then error("inconsistent terminal evidence for draw \($i)") else
           {index:$i,seed:$c.seed,interrupted_attempts:.interrupted,weights_sha256:$c.weights_sha256,success:$s.all_strict,
            hidden_correct:$s.strict_correct,hidden_min_gap:$s.min_gap,
            observed_correct:$c.observed.strict_correct,all_inputs_certificate:$c.accept} end) as $rows |
    def count_at($n):
        (map(select(.success))|length) as $k |
        {registered:$n,evaluated:length,successes:$k,empirical_fraction:(if length==$n then $k/$n else null end)};
    def choose($n;$k): reduce range(1;$k+1) as $i (1; . * ($n-$k+$i) / $i);
    def tail_bound($k;$p): [1,2*choose(10;$k)*pow($p;$k)]|min;
    ($rows|count_at(12)) as $all |
    if $summary[0]!=null and ($summary[0].protocol!=$p or ($all.evaluated==12 and $summary[0].successes!=$all.successes))
    then error("confirmation disagrees with verified terminal outcomes") else . end |
    ($rows|map(select(.index>=2))) as $new |
    ($new|count_at(10)) as $fresh |
    ($new|map(select(.interrupted_attempts==0 and .success))|length) as $original |
    {updates:$p.updates,all_draws:$all,unseen_in_pilot:$fresh,
     two_budget_iid_null_tail_upper_bounds:(if $fresh.evaluated==10 then
       {registered:10,at_least_successes:$original,
        interrupted_draws:($new|map(select(.interrupted_attempts>0))|length),
        single_run_at_most_half:tail_bound($original;0.5),
        single_run_at_most_three_quarters:tail_bound($original;0.75)} else null end),
     draws:$rows}'
