. as $rows
| ($p[0] | [.tasks[] as $t | .arms[] as $a | .radii[] | [$t,$a,.]]) as $expected
| ([$rows[] | [.task,.arm,.radius_name]]) as $actual
| if (($actual|sort) != ($expected|sort)) or (($actual|unique|length) != ($actual|length))
  then error("missing, duplicated or unplanned pilot cell") else . end
| if any($rows[]; .protocol != $p[0]) or ([$rows[].source_fingerprint] | unique | length) != 1
  then error("mixed producer/protocol") else . end
| {planned:($expected|length), completed:length,
   endpoints:[.[] | {task,arm,radius_name,updates,epochs,stop_reason,
     ideal_ce_excess,fit_defect,full_operator_residual,uniform_mixture_alpha,
     max_operator_norm:(.operator_norms|max),elapsed_seconds,
     first_primary_99_update,train:.metrics.train.sequence_accuracy,
     primary:.metrics[.selection].sequence_accuracy,train_ce:.metrics.train.loss}],
   primary_passes:[group_by([.arm,.radius_name])[] |
     {arm:.[0].arm,radius:.[0].radius_name,passed:([.[] | select(.metrics[.selection].exact * 100 >= .metrics[.selection].examples * 99)]|length),total:length}]}
