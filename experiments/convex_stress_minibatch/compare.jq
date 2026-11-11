# All planned cells remain in the output, including user-interrupted runs.
# Input: jq -s, all endpoint result.json files; --slurpfile old v1/summary.json
# and --slurpfile interrupted minibatch-v1/interrupted_cells.json.
def key: [.arm,.task,.seed] | tojson;
def require($condition; $message):
  if $condition then . else error($message) end;
. as $results |
($old[0].cells | INDEX(key)) as $before |
($results | INDEX(key)) as $complete |
($interrupted[0] | INDEX(key)) as $stopped |
require(($before|length) == 210; "expected original 210-cell population") |
require(($results|length) == ($complete|length); "duplicate endpoint") |
require(($stopped|length) == ($interrupted[0]|length); "duplicate interruption") |
require(($results|map(.source_fingerprint)|unique|length) == 1; "mixed source fingerprints") |
require((($complete|keys) + ($stopped|keys) | sort) == ($before|keys);
        "completed and interrupted cells must partition all 210 planned cells") |
[$before | keys[] as $k | $before[$k] as $oldcell |
  if $complete[$k] != null then
    $complete[$k] as $r | ($r.splits[$r.selection] // null) as $primary |
    require($r.tier == $oldcell.tier; "tier mismatch") |
    require($r.arm != "forward_only" or
      ($r.equivalence.equal == true and $primary == $oldcell.primary and
       $r.stop_reason == $oldcell.stop_reason); "forward tensor or metric mismatch") |
    {arm:$r.arm,task:$r.task,tier:$r.tier,seed:$r.seed,status:"complete",
     updates:$r.updates,stop_reason:$r.stop_reason,primary:$primary,
     old_primary:$oldcell.primary,training:$r.training,feasibility_error:$r.feasibility_error,
     pass:($primary != null and $primary.exact*100 >= $primary.examples*99)}
  else
    $stopped[$k] as $r |
    require($r.arm == "frank_wolfe" and $r.tier == $oldcell.tier;
      "unexpected interrupted population") |
    {arm:$r.arm,task:$r.task,tier:$r.tier,seed:$r.seed,status:"interrupted_by_user",
     pass:null,primary:null,old_primary:$oldcell.primary,last_observation:$r.last_observation}
  end] as $cells |
{complete_cohort:false,source_fingerprint:$results[0].source_fingerprint,
 old_source_fingerprint:$old[0].source_fingerprint,
 forward_metrics_and_tensors_identical:true,cells:$cells,
 totals:($cells | group_by([.arm,.tier]) | map({arm:.[0].arm,tier:.[0].tier,
   planned:length,completed:(map(select(.status=="complete"))|length),
   interrupted:(map(select(.status=="interrupted_by_user"))|length),
   passed:(map(select(.pass==true))|length),
   explicit_failures:(map(select(.stop_reason=="no_consistent_key"))|length)}))}
