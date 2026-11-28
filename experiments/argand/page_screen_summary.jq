# Input is an alternating sequence of terminal records and their independent
# replays, in the same case and seed order as the canonical manifest.
. as $records |
[$suite[0].tasks[] | select($tier == "all" or .tier == $tier) as $task |
  $suite[0].seeds[] | {task:$task, seed:.}] as $cells |
if length != 2*($cells|length) then error("incomplete optimizer screen") else . end |
[range(0;$cells|length) as $i |
  $cells[$i] as $c | $records[2*$i] as $r | $records[2*$i+1] as $v |
  $v.splits[$c.task.selection] as $p |
  if $r.identity.case != $c.task.name or $r.identity.seed != $c.seed or
     $r.identity.optimizer != $records[0].identity.optimizer or $r.identity.correction != $records[0].identity.correction or
     $r.identity.stop_on != $records[0].identity.stop_on or
     (if ($r.identity.optimizer|IN("equal-moment-wsd","uncorrected-adam-wsd","normalized-adamw-wsd")) then
        ($r.identity.optimizer=="normalized-adamw-wsd") as $normalized |
        (if $r.identity.optimizer=="equal-moment-wsd" then [$r.identity.equal_moment,0.9]
         elif $normalized then [$r.identity.normalized_adamw,0.95] else [$r.identity.uncorrected_adam,0.95] end) |
        .[1] as $beta2 | .[0] | .beta1!=0.9 or .beta2!=$beta2 or .bias_correction!=$normalized or
        ($normalized and .normalization!="recursive-gains") or
        .epsilon!=1e-8 or .weight_decay!=0.1 or .peak!=0.003 or .budget!=$c.task.case.max_updates
      else false end) or
     ($r.identity.stop_on=="first-strict-100-percent" and $r.step<$r.identity.budget and
       ($r.strict.all_strict!=true or $r.correct!=$r.total)) or
     ($r.identity.optimizer=="adamw-wsd-first-noise" and
       ($r.identity.fixed_horizon or $r.identity.gaussian_weight_noise!={after_update:1,seed:(371+$c.seed),stddev:1e-6})) or
     $r.identity.tier != $c.task.tier or $r.identity != $v.identity or
     $r.weights_sha256 != $v.weights_sha256 or $r.step != $v.step or
     $r.strict != $v.strict or $p.correct != $r.correct or $p.loss != $r.loss or
     $p.total != $r.total or $r.total != $c.task.heldout_examples or
     $r.step > 1000 or $r.step % $c.task.case.observe != 0 or
     $r.passed != ($r.correct*100 >= 99*$r.total) or
     ($r.passed == false and $r.step != $r.identity.budget)
  then error("screen or replay mismatch at \($c.task.name), seed \($c.seed)") else $r end] |
def counts: {runs:length, passed_99:map(select(.passed))|length,
  exact:map(select(.strict.all_strict and .correct == .total))|length,
  cases_with_pass:map(select(.passed)|.identity.case)|unique|length,
  cases_all_seeds:group_by(.identity.case)|map(select(all(.[];.passed)))|length,
  updates:map(.step)|add, skipped:map(.skipped_batches)|add,
  train_and_score_seconds:map(.segment_seconds)|add};
{optimizer:.[0].identity.optimizer,correction:.[0].identity.correction,
 overall:counts,tiers:(group_by(.identity.tier) | map({key:.[0].identity.tier,value:counts}) | from_entries),
 cases:group_by(.identity.case) | map({name:.[0].identity.case, tier:.[0].identity.tier, total:.[0].total,
   cells:sort_by(.identity.seed) | map({seed:.identity.seed,step,correct,passed,strict:.strict.all_strict})})} +
  (if .[0].identity.stop_on then {stop_on:.[0].identity.stop_on} else {} end)
