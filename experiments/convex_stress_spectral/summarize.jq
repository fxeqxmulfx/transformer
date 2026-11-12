. as $rows
| ($p[0] | [.tasks[] as $t | .arms[] as $a | .radii[] | [$t,$a,.]]) as $expected
| ([$rows[] | [.task,.arm,.radius_name]]) as $actual
| if (($actual|sort) != ($expected|sort)) or (($actual|unique|length) != ($actual|length))
  then error("missing, duplicated or unplanned pilot cell") else . end
| {planned:($expected|length), completed:length,
   endpoints:[.[] | {task,arm,radius_name,updates,epochs,projection_caps,max_radial_shrink,
     projected_gradient_residual,max_entry_norm,operator_norm,elapsed_seconds,
     first_primary_99_update,train:.metrics.train.sequence_accuracy,
     primary:.metrics[.selection].sequence_accuracy,train_ce:.metrics.train.loss}],
   primary_passes:[group_by([.arm,.radius_name])[] |
     {arm:.[0].arm,radius:.[0].radius_name,passed:([.[] | select(.metrics[.selection].exact * 100 >= .metrics[.selection].examples * 99)]|length),total:length}]}
