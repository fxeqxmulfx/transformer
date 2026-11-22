# jq -s [--rawfile extensions extensions.tsv] -f compare.jq CONTROL CANDIDATE
# A frozen registry fixes the comparison cohort even after new cases are admitted.
def registry:
  (($ARGS.named.extensions // "") | split("\n") | map(select(length > 0) | split("\t"))) as $rows
  | if ($rows | all(.[]; length == 2 and (.[0] == "short" or .[0] == "long") and (.[1]|length) > 0))
       and ($rows | map(.[1]) | unique | length) == ($rows|length)
    then $rows | map({key:.[1], value:.[0]}) | from_entries
    else error("invalid extension registry") end;
def indexed($registry):
  [.tasks[] as $task
   | if $task.tier == "short" or $task.tier == "long" then $task.tier
     elif ($registry | has($task.case)) and $task.tier == ($registry[$task.case] + "-extension")
       then $registry[$task.case]
     else empty end as $tier
   | $task.seeds[]
   | {case:$task.case,tier:$tier,seed,status,step,examples_seen,correct,total,source:$task.tier}]
  | sort_by(.tier,.case,.seed);
def complete($registry):
  length == 105 + 3 * ($registry|length)
  and (map(select(.source == "short")) | length == 75)
  and (map(select(.source == "long")) | length == 30)
  and (group_by(.case) | length == 35 + ($registry|length) and all(.[]; map(.seed) == [0,1,2]))
  and all(.[]; .status == "pass" and (.examples_seen|type) == "number" and .examples_seen > 0
           and (.step|type) == "number" and .step > 0 and (.correct|type) == "number"
           and (.total|type) == "number" and .total > 0 and .correct <= .total
           and .correct * 100 >= .total * 99);
def identity: [.source,.case,.seed,.total];
if length != 2 then error("expected a control report and a candidate report") else . end
| registry as $registry | .[0] as $before | .[1] as $after
| ($before|indexed($registry)) as $a | ($after|indexed($registry)) as $b
| if (($a|complete($registry)) and ($b|complete($registry)) and ($a|map(identity)) == ($b|map(identity))) then
    [range(0; $a|length) as $i | ($b[$i] | del(.source,.correct,.total)) + {control_step:$a[$i].step,
        epoch_ratio:($b[$i].examples_seen / $a[$i].examples_seen)}] as $pairs
    | {control:$before.model,candidate:$after.model,complete:true,
       tiers:($pairs|group_by(.tier)|map({key:.[0].tier,value:{
          runs:length,faster:(map(select(.epoch_ratio < 1))|length),
          tied:(map(select(.epoch_ratio == 1))|length),slower:(map(select(.epoch_ratio > 1))|length),
          geometric_mean_epoch_ratio:(map(.epoch_ratio|log)|add/length|exp)}})|from_entries),
       pairs:$pairs}
    | .speed_gate = (all(.tiers[]; .geometric_mean_epoch_ratio <= 1.0))
  else error("incomplete or mismatched task, seed, tier, held-out size, or 99% crossing") end
