# jq -es --slurpfile inventory INVENTORY -f catalog_results.jq INDEPENDENT_REPLAYS.jsonl
def observation($r; $scores):
  $scores[$r.protocol.selection];
def meets($c; $s):
  if $c.metric == "sequence_accuracy" then $s.correct / $s.total >= $c.target
  elif $c.metric == "negative_mse" then
    ($s.mse|type=="number" and isfinite and .>=0) and (-$s.mse >= $c.target)
  else error("unsupported recorded metric") end;
def source_summary:
  . as $r | .source_target as $t | .protocol as $p
  | {seed:$p.seed, search_updates:$p.updates, final_update:$t.final_step,
     outcome:(if $t.first != null then
                (if $t.final_target_met then "retained" else "crossed then dropped" end)
              elif $t.budget_exhausted then "budget miss" else "screen cutoff" end),
     first_update:$t.first.step, first_exact_99_update:.first_99,
     primary_at_first:observation($r; $t.first.scores), primary_at_final:observation($r; .scores),
     first_scores:$t.first.scores, final_scores:.scores,
     first_weights_sha256:$t.first.weights_sha256, final_weights_sha256:.weights_sha256};
if ($inventory|length) != 1 then error("expected one catalog inventory") else . end
| $inventory[0].cases as $cases
| INDEX($cases[]; .case) as $by_name
| if ($cases|length) == 0 or ($by_name|length) != ($cases|length)
     or (map(.protocol.model)|unique|length) > 1
     or (group_by([.protocol.case,.protocol.seed,.protocol.updates])|any(.[]; length!=1))
  then error("invalid source inventory, mixed models, or duplicate search") else . end
| map(. as $r | .protocol as $p | .source_target as $t | $by_name[$p.case] as $c
      | if $c.source_adapter == "ready" and $p.suite_sha256 == $c.catalog_sha256
           and $p.source_target.metric == $c.metric and $p.source_target.threshold == $c.target
           and $p.source_target.after_target == $c.after_target
           and $p.source_target.search_updates == $p.updates and $p.schedule_updates == $c.budget
           and $p.selection == $c.selection and $p.observe == $c.observe
           and $p.heldout_examples == $c.heldout_examples
           and ($p.seed==0 or $p.seed==1 or $p.seed==2)
           and ($r.scores[$c.selection].total==$c.heldout_examples)
           and ($t.final_step>=0 and $t.final_step<=$c.budget)
           and $t.budget_exhausted==($t.final_step==$c.budget)
           and $t.final_target_met==meets($c; $r.scores[$c.selection])
           and (if $t.first==null then ($t.final_step==$p.updates and ($t.final_target_met|not))
                else $t.first.step<=$p.updates and $t.first.scores[$c.selection].total==$c.heldout_examples
                     and meets($c; $t.first.scores[$c.selection]) end)
        then . else error("result differs from its catalog protocol: "+$p.case) end) as $runs
| [$cases[] as $c | [$runs[]|select(.protocol.case==$c.case)]
   | group_by(.protocol.seed) | map(max_by(.protocol.updates)|source_summary) as $seeds
   | $c + {seeds:$seeds}] as $catalog
| {model:($runs[0].protocol.model // null),source_plan_sha256:$inventory[0].source_plan_sha256,
   cohorts:($catalog|group_by(.membership)|map([.[].seeds[]] as $seeds |
     {key:.[0].membership,value:{
       cases:length, measured_cases:(map(select(.seeds|length>0))|length),
       unavailable_cases:(map(select(.source_adapter=="unavailable"))|length),
       measured_seeds:($seeds|length),
       retained:([$seeds[]|select(.outcome=="retained")]|length),
       crossed_then_dropped:([$seeds[]|select(.outcome=="crossed then dropped")]|length),
       budget_misses:([$seeds[]|select(.outcome=="budget miss")]|length),
       bounded_misses:([$seeds[]|select(.outcome=="screen cutoff")]|length)}})|from_entries),
   cases:$catalog}
