# Read both screens' observations.jsonl files with jq -s. Supply their
# replay-verified summaries through --slurpfile control and candidate.
def rule: [.optimizer, .correction, .stop_on];
def cell: [.case, .seed] | @json;
def endpoints($screen):
  INDEX($screen.cases[] as $task | $task.cells[] + {case:$task.name, tier:$task.tier, total:$task.total}; cell);
def trajectory($ends):
  sort_by(.step) as $rows | $rows[0] as $start | $rows[-1] as $end |
  $start.identity as $id | $ends[$id|cell] as $verified |
  if $verified == null or $verified.tier != $id.tier or $verified.total != $end.total or
     $verified.step != $end.step or $verified.correct != $end.correct or $verified.strict != $end.strict.all_strict or
     $end.step > $id.budget or $id.observe <= 0 or
     ($rows|map(.step)) != [range(0;$end.step+1;$id.observe)] or
     any($rows[]; .identity != $id or .total != $verified.total or .correct < 0 or .correct > .total) or
     (if $id.stop_on=="first-strict-100-percent" then
        any($rows[]; .step < $end.step and .strict.all_strict==true) or
        ($end.step < $id.budget and $end.strict.all_strict!=true)
      else any($rows[]; .step < $end.step and .correct*100 >= 99*.total) end)
  then error("incomplete observations or endpoint mismatch at \($id.case), seed \($id.seed)") else . end |
  {identity:($id|del(.optimizer,.correction,.gaussian_weight_noise,.stop_on)), initial:($start|{loss,correct,total}),
   case:$id.case,seed:$id.seed,tier:$id.tier,budget:$id.budget,step:$end.step,
   first_99:([$rows[]|select(.correct*100 >= 99*.total)|.step]|first),
   strict:$verified.strict};
def traces($report):
  endpoints($report) as $ends | ($report|rule) as $rule |
  map(select((.identity|rule)==$rule)) | group_by(.identity|cell) | map(trajectory($ends)) |
  if (map(cell)|sort) == ($ends|keys) then sort_by(cell) else error("missing cells") end;
def counts:
  {runs:length,budget:map(.budget)|add,
   control_updates:map(.control.step)|add,candidate_updates:map(.candidate.step)|add,
   control_99:map(select(.control.first_99 != null))|length,
   candidate_99:map(select(.candidate.first_99 != null))|length,
   shared_99:map(select(.control.first_99 != null and .candidate.first_99 != null))|length,
   shared_99_steps:(map(select(.control.first_99 != null and .candidate.first_99 != null))|
     {earlier:map(select(.candidate.first_99 < .control.first_99))|length,
      tied:map(select(.candidate.first_99 == .control.first_99))|length,
      later:map(select(.candidate.first_99 > .control.first_99))|length}),
   gained_99:map(select(.control.first_99 == null and .candidate.first_99 != null))|length,
   lost_99:map(select(.control.first_99 != null and .candidate.first_99 == null))|length,
   control_exact:map(select(.control.strict))|length,candidate_exact:map(select(.candidate.strict))|length,
   control_stopped_before_cap_below_100:map(select((.control.strict|not) and .control.step < .budget))|length,
   candidate_updates_saved:map(.budget-.candidate.step)|add};
traces($control[0]) as $a | traces($candidate[0]) as $b |
if ($control[0]|rule)==($candidate[0]|rule) or ($a|map(cell))!=($b|map(cell))
then error("two matched screens with distinct training rules are required") else . end |
[range($a|length) as $i | $a[$i] as $x | $b[$i] as $y |
 if $x.identity!=$y.identity or $x.initial!=$y.initial then error("unmatched initial state or schedule") else
 {case:$x.case,seed:$x.seed,tier:$x.tier,budget:$x.budget,
  control:($x|{step,first_99,strict}),candidate:($y|{step,first_99,strict})} end] as $pairs |
{tiers:($pairs|group_by(.tier)|map({key:.[0].tier,value:counts})|from_entries),
 overall:($pairs|counts),pairs:$pairs}
