# Metadata-only fixture. No original examples, answers or learner are used.
$protocol[0] as $p
| [$p.tasks[] as $task | $p.arms[] | {task:$task, training_mode:.}] as $ids
| def model($id; $k; $ce): $id + {kind:"fixture", vocab:97, decoder_index:$k,
    train_ce:$ce, primary_pass:false, all_splits_pass:false,
    metrics:{validation:{sequence_accuracy:0.1}}};
{
  general:{source_fingerprint:$p.general_fingerprint, cases:27, planned_decoder_slots:108,
    candidates:[$ids[] as $id | [0,1][] as $k | model($id; $k; 3.7 + $k/10)],
    records:[$ids[] | {case:{task:.task,arm:.training_mode}, status:"complete",
      candidates:[{decoder_index:0,status:"complete"},{decoder_index:1,status:"complete"},
                  {decoder_index:2,status:"refused"},{decoder_index:3,status:"refused"}]}],
    choices:[$ids[] | . + {status:"complete"}]},
  certified:{source_fingerprint:$p.certified_fingerprint,
    choices:[$ids[] | select(.task != $p.added_task) | . + {selected:model(.; 0; 3.9)}]},
  hull:{source_fingerprint:$p.hull_fingerprint,
    choices:[$suite[0].tasks[].name as $task | $p.arms[] | {task:$task,arm:.,status:"selected",
      selected_candidate:"counts",training_ce:3.9,metrics:{validation:{sequence_accuracy:0.1}},
      primary_pass:false,all_splits_pass:false}]},
  prepared:{source_fingerprint:$p.hull_fingerprint,
    tasks:[$suite[0].tasks[] | {task:.name,vocab:97,scale:1}]}
}
