. as $rows
| ($p[0] | [.tasks[] as $t | .arms[] as $a | .budgets[] | [$t,$a,.]]) as $expected
| ([$rows[] | [.task,.arm,.budget_name]]) as $actual
| if (($actual|sort) != ($expected|sort)) or (($actual|unique|length) != ($actual|length))
  then error("missing, duplicated or unplanned pilot cell") else . end
| if any($rows[]; .protocol != $p[0] or .source_fingerprint != $prep[0].source_fingerprint)
  then error("mixed producer/protocol") else . end
| if any($rows[]; .status != "complete" or .endpoint_kind != "selected")
  then error("incomplete or nonselected endpoint") else . end
| {planned:($expected|length), completed:length,
   endpoints:[.[] | {task,arm,budget_name,polish,ideal_ce_excess,fit_defect,postprocessing_seconds,
     parent_training,train:.metrics.train.sequence_accuracy,primary:.metrics[.selection].sequence_accuracy}],
   primary_passes:[group_by([.arm,.budget_name])[] |
     {arm:.[0].arm,budget:.[0].budget_name,
      passed:([.[] | select(.metrics[.selection].exact * 100 >= .metrics[.selection].examples * 99)]|length),total:length}]}
