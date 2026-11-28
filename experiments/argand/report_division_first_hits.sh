#!/usr/bin/env bash
# Report registered first hits. A matching analysis plan identifies the
# prospective draws. A rerun after interruption receives no test credit.
set -euo pipefail
if (( $# < 1 || $# > 2 )); then echo 'usage: bash report_division_first_hits.sh COHORT [ANALYSIS_PLAN]' >&2; exit 2; fi
dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
analysis=$(realpath -e -- "${2:-$dir/division-first-hit-registration.json}")
cd -- "$1"
for ((i=0; i<12; i++)); do
    seed=$(jq -er --argjson i "$i" '.seeds[$i][0]' protocol.json)
    interruptions=0
    for trace in "grok-division-p97/seed$seed"/interrupted/attempt.*/observations.jsonl; do
        if [[ -s $trace ]]; then ((interruptions+=1)); fi
    done
    echo "$interruptions"
done | jq -es -L "$dir" --slurpfile plan protocol.json --slurpfile verified summary.json --slurpfile analysis "$analysis" '
    include "division_binomial";
    . as $interruptions | $plan[0] as $p | $verified[0] as $s | $s.draws as $rows |
    if $p.case!="grok-division-p97" or $p.correction!="adamw-wsd" or $p.stop!="first-strict-100-percent" or
       $p.n!=12 or ($p.seeds|length)!=12 or ($p.seeds|map(.[0])|unique|length)!=12 or $s.updates!=$p.updates or
       $s.registered!=12 or $s.evaluated>12 or $s.empirical_fraction!=(if $s.evaluated==12 then $s.successes/12 else null end) or
       ($rows|map(.index))!=[range($s.evaluated)] or $s.successes!=($rows|map(select(.success))|length) or
       any($rows[]; .seed!=$p.seeds[.index][0] or .total!=4656 or .step<0 or .step>$s.updates or
         .success!=(.correct==4656 and .min_gap>0) or ((.success|not) and .step!=$s.updates))
    then error("inconsistent stopping protocol or draw records") else . end |
    $analysis[0] as $a | $a.first_prospective_index as $first | ($p.n-$first) as $n |
    ($rows|map(select(.index>=$first))) as $fresh |
    ($fresh|map(select(.success and .checkpoint=="terminal" and $interruptions[.index]==0))|length) as $k |
    def bounds($q): binomial_upper_tail($n;$k;$q) as $tail |
      {one_budget:$tail,any_registered_budget:([1,($a.wsd_budgets|length)*$tail]|min)};
    {updates:$s.updates,all_draws:($s|{registered,evaluated,successes,empirical_fraction}),
     prospective:(if $a.model_seeds==($p.seeds|map(.[0])) and ($a.wsd_budgets|index($p.updates))!=null and
       $first==($first|floor) and $first>=0 and $n>0 then
       {registered:$n,evaluated:($fresh|length),successes:($fresh|map(select(.success))|length),
       first_attempt_successes_for_test:$k,
       independent_null_upper_tails:(if ($fresh|length)==$n then
         {at_most_half:bounds(0.5),at_most_three_quarters:bounds(0.75)} else null end)} else null end),
     stopped_during_training:($rows|map(select(.checkpoint=="terminal"))|
       {evaluated:length,updates_omitted_after_stopping:(map($s.updates-.step)|add//0)})}'
