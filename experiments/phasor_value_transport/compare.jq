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
def nonnegative_integer: type == "number" and . >= 0 and . == floor;
def complete($registry):
  length == 105 + 3 * ($registry|length)
  and (map(select(.source == "short")) | length == 75)
  and (map(select(.source == "long")) | length == 30)
  and (group_by(.case) | length == 35 + ($registry|length) and all(.[]; map(.seed) == [0,1,2]))
  and all(.[]; .status == "pass" and (.examples_seen|nonnegative_integer) and (.step|nonnegative_integer)
           and ((.step == 0) == (.examples_seen == 0)) and (.correct|type) == "number"
           and (.total|type) == "number" and .total > 0 and .correct <= .total
           and .correct * 100 >= .total * 99);
def identity: [.source,.case,.seed,.total];
def tier_summary:
  [ .[] | select(.step > 0 and .control_step > 0) | .epoch_ratio | log ] as $logs
  | {runs:length,faster:(map(select(.step < .control_step))|length),
     tied:(map(select(.step == .control_step))|length),slower:(map(select(.step > .control_step))|length),
     geometric_mean_epoch_ratio:(if ($logs|length) > 0 then $logs|add/length|exp else null end)}
    + (if any(.[]; .step == 0 or .control_step == 0) then
         {initialization:{gained:(map(select(.step == 0 and .control_step > 0))|length),
                          retained:(map(select(.step == 0 and .control_step == 0))|length),
                          lost:(map(select(.step > 0 and .control_step == 0))|length)}}
       else {} end);
if length != 2 then error("expected a control report and a candidate report") else . end
| registry as $registry | .[0] as $before | .[1] as $after
| ($before|indexed($registry)) as $a | ($after|indexed($registry)) as $b
| if (($a|complete($registry)) and ($b|complete($registry)) and ($a|map(identity)) == ($b|map(identity))) then
    [range(0; $a|length) as $i | ($b[$i] | del(.source,.correct,.total)) + {control_step:$a[$i].step,
        epoch_ratio:(if $a[$i].examples_seen > 0 then $b[$i].examples_seen / $a[$i].examples_seen else null end)}] as $pairs
    | {control:$before.model,candidate:$after.model,complete:true,
       tiers:($pairs|group_by(.tier)|map({key:.[0].tier,value:tier_summary})|from_entries),
       pairs:$pairs}
    | .speed_gate = (all(.tiers[]; (.geometric_mean_epoch_ratio // 1) <= 1 and (.initialization.lost // 0) == 0))
  else error("incomplete or mismatched task, seed, tier, held-out size, or 99% crossing") end
