# Deterministic descriptive summaries of every prescribed checkpoint.
# Usage: jq -s -f summarize.jq COHORT/*/*/*/*/scores.json
if length != 16 or ([.[].cell] | unique | length) != 16 then
  error("expected sixteen distinct complete cells")
elif any(.[]; (.checkpoints | keys) != ["0","1","2","4","8"]) then
  error("prescribed checkpoints changed")
else . end
| [ .[] as $run
    | $run.checkpoints | to_entries[]
    | .value as $r
    | {cell:$run.cell, stage:(.key|tonumber), components:$r.components_stored,
       capacity:$r.capacity_mass, original_success:$r.original_success,
       raw_and_exact_99:$r.raw_and_exact_99,
       train_raw:$r.splits.train.metrics.token_accuracy,
       train_exact:$r.splits.train.metrics.sequence_accuracy,
       train_ce:$r.splits.train.metrics.loss,
       novel_raw:$r.splits["validation/novel"].metrics.token_accuracy,
       novel_exact:$r.splits["validation/novel"].metrics.sequence_accuracy,
       novel_ce:$r.splits["validation/novel"].metrics.loss,
       test_raw:$r.splits.test.metrics.token_accuracy,
       test_exact:$r.splits.test.metrics.sequence_accuracy,
       test_ce:$r.splits.test.metrics.loss}
  ] as $rows
| if any($rows[]; [.train_raw,.train_exact,.train_ce,.novel_raw,.novel_exact,.novel_ce,.test_raw,.test_exact,.test_ce] | any(.==null)) then
    error("original metric missing")
  else . end
| {
    rows:($rows|sort_by(.cell.arm,.cell.geometry,.cell.variant,.cell.mode,.stage)),
    original_successes:([$rows[]|select(.original_success)]|length),
    raw_and_exact_99:([$rows[]|select(.raw_and_exact_99)]|length),
    final:[$rows[]|select(.stage==8)],
    paired_final:[
      $rows[]|select(.stage==8 and .cell.arm=="grow") as $g
      | $rows[]|select(.stage==8 and .cell.arm=="continue"
                      and .cell.geometry==$g.cell.geometry
                      and .cell.variant==$g.cell.variant
                      and .cell.mode==$g.cell.mode) as $c
      | {geometry:$g.cell.geometry,variant:$g.cell.variant,mode:$g.cell.mode,
         grow_minus_continue:{train_raw:($g.train_raw-$c.train_raw),
           train_ce:($g.train_ce-$c.train_ce),novel_raw:($g.novel_raw-$c.novel_raw),
           novel_ce:($g.novel_ce-$c.novel_ce),test_raw:($g.test_raw-$c.test_raw),
           test_ce:($g.test_ce-$c.test_ce)}}
    ],
    trajectories:($rows|group_by(.cell.arm,.cell.geometry,.stage)|map({
      arm:.[0].cell.arm,geometry:.[0].cell.geometry,stage:.[0].stage,cells:length,
      mean_train_raw:(map(.train_raw)|add/length),mean_train_ce:(map(.train_ce)|add/length),
      mean_novel_raw:(map(.novel_raw)|add/length),mean_novel_ce:(map(.novel_ce)|add/length),
      mean_test_raw:(map(.test_raw)|add/length),mean_test_ce:(map(.test_ce)|add/length)
    })),
    scope:"one original task and seed; four dependent starting-variant/mode cells per trajectory group; every checkpoint retained, none selected"
  }
