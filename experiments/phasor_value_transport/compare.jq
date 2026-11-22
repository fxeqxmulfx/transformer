# jq -s -f compare.jq CONTROL_REPORT CANDIDATE_REPORT
def indexed:
  [.tasks[] | select(.tier == "short" or .tier == "long") as $task
   | $task.seeds[] | {case:$task.case,tier:$task.tier,seed,status,step,examples_seen}]
  | sort_by(.tier,.case,.seed);
def complete:
  length == 105 and (map(select(.tier == "short")) | length == 75)
  and (group_by(.tier,.case) | length == 35 and all(.[]; map(.seed) == [0,1,2]))
  and all(.[]; .status == "pass" and (.examples_seen|type) == "number" and .examples_seen > 0);
def identity: [.tier,.case,.seed];
if length != 2 then error("expected a control report and a candidate report") else . end
| .[0] as $before | .[1] as $after | ($before|indexed) as $a | ($after|indexed) as $b
| if (($a|complete) and ($b|complete) and ($a|map(identity)) == ($b|map(identity))) then
    [range(0;105) as $i | $b[$i] + {control_step:$a[$i].step,
        epoch_ratio:($b[$i].examples_seen / $a[$i].examples_seen)}] as $pairs
    | {control:$before.model,candidate:$after.model,complete:true,
       tiers:($pairs|group_by(.tier)|map({key:.[0].tier,value:{
          runs:length,faster:(map(select(.epoch_ratio < 1))|length),
          tied:(map(select(.epoch_ratio == 1))|length),slower:(map(select(.epoch_ratio > 1))|length),
          geometric_mean_epoch_ratio:(map(.epoch_ratio|log)|add/length|exp)}})|from_entries),
       pairs:$pairs}
    | .speed_gate = (all(.tiers[]; .geometric_mean_epoch_ratio <= 1.0))
  else error("both reports must have the same 105 distinct passing original task/seed pairs") end
