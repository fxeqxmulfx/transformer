#!/usr/bin/env bash
# Summarize the two predeclared controls without rerunning a candidate.
set -euo pipefail
if (( $# < 1 || $# > 2 )); then echo 'usage: bash report_division_optimizer_pairs.sh PRIMARY [HIGHER_RATE_CONTROL]' >&2; exit 2; fi
primary=$(realpath -e -- "$1")
shopt -s nullglob
files=("$primary"/[0-9]*-{control,candidate}.json)
if (( $#==2 )) && [[ -s "$2/protocol.json" ]]; then
    jq -es '.[0] as $p | .[1] as $h | $p.candidate==$h.candidate and $p.binary_sha256==$h.binary_sha256 and
      $h.seeds==$p.registration.seeds' "$primary/protocol.json" "$2/protocol.json" >/dev/null
    files+=("$2"/[0-9]*-higher_rate_control.json)
fi
jq -s --slurpfile p "$primary/protocol.json" '
  def median: sort | length as $n |
    if $n==0 then null elif $n%2==1 then .[$n/2|floor] else (.[$n/2-1]+.[$n/2])/2 end;
  def choose($n;$k): reduce range(1;$k+1) as $i (1; . * ($n-$i+1)/$i);
  def tail($n;$k;$q): [range($k;$n+1) as $i | choose($n;$i)*pow($q;$i)*pow(1-$q;$n-$i)] | add;
  def counts:
    map(select(.!=null)) as $all | ($all|map(select(.first_attempt=="complete"))) as $ok |
    {evaluated:($all|length),hits:($ok|map(select(.success))|length),
     misses:($ok|map(select(.success|not))|length),interrupted:(($all|length)-($ok|length))};
  def diff($candidate;$ctrl):
    if $candidate==null or $ctrl==null then {status:"pending"}
    elif $candidate.first_attempt!="complete" or $ctrl.first_attempt!="complete" then {status:"interrupted"}
    elif $candidate.success and $ctrl.success then
      ($candidate.step-$ctrl.step) as $d |
      {status:(if $d<0 then "earlier" elif $d==0 then "tied" else "later" end),difference:$d}
    elif $candidate.success then {status:"earlier"}
    elif $ctrl.success then {status:"later"} else {status:"both_censored"} end;
  def compare($arm):
    map(diff(.candidate;.[$arm])) as $x | {earlier:($x|map(select(.status=="earlier"))|length),
      tied:($x|map(select(.status=="tied"))|length),later:($x|map(select(.status=="later"))|length),
      both_censored:($x|map(select(.status=="both_censored"))|length),
      interrupted:($x|map(select(.status=="interrupted"))|length),pending:($x|map(select(.status=="pending"))|length),
      median_step_difference_both_hit:($x|map(select(.difference!=null)|.difference)|median)};
  . as $arms | $p[0].registration.seeds as $seeds |
  if any($arms[]; .seed as $s | ($seeds|index($s))==null) or
    (($arms|map([.seed,.arm])|unique|length)!=($arms|length)) then error("duplicate or unregistered arm") else . end |
  [range(0; $seeds|length) as $i | $seeds[$i] as $seed |
    {index:$i,seed:$seed} + ($arms | map(select(.seed==$seed)|{key:.arm,value:
      {step,success,first_attempt,min_gap:.hidden.minimum_gap,selected}}) | from_entries)] as $pairs |
  ($pairs|map(.candidate)|counts) as $candidate |
  {registered:($seeds|length),primary_rate:$p[0].control.identity.adamw.peak,
   candidate_optimizer:$p[0].candidate.identity.optimizer,
   candidate_peak:($p[0].candidate.identity|(.equal_moment//.softsign_lion//.rmsprop//.muon).peak),
   candidate:$candidate,control:($pairs|map(.control)|counts),
   candidate_null_tail_at_three_quarters:(if $candidate.evaluated==($seeds|length) then
      tail(($seeds|length);$candidate.hits;0.75) else null end),
   higher_rate_control:($pairs|map(.higher_rate_control)|counts),
   against_primary:($pairs|compare("control")),against_higher_rate:($pairs|compare("higher_rate_control")),
   pairs:$pairs}' "${files[@]}" < /dev/null
