# Compare complete fixed fixtures; never count component timing as task accuracy.
def valid:
  .states == 8 and .records == 130 and .rounds == 3 and .original_data_used == false
  and .known_certificate == "OutwardIntervals" and .unknown_certificate == "ExactDyadics"
  and ([.forward_sha256,.mini_sha256] | all(type=="string" and test("^[0-9a-f]{64}$")))
  and ([.head_build_ns,.loss_130_ns,.gradient_127_ns,.proposal_ns,.projection_ns,
        .known_predict_ns,.unknown_predict_ns,.forward_total_ns,.mini_total_ns,
        .forward_work,.mini_work] | all(type=="number" and .>0));
($before|sort_by(.programs,.classes)) as $a
| ($after|sort_by(.programs,.classes)) as $b
| [[128,97],[128,193],[512,97],[512,193]] as $keys
| if ($a|map([.programs,.classes]))!=$keys or ($b|map([.programs,.classes]))!=$keys
     or (all($a[]; valid)|not) or (all($b[]; valid)|not)
  then error("missing or invalid complete phase profile") else . end
| [range(0;4) as $i | $a[$i] as $old | $b[$i] as $new
   | if $old.forward_sha256!=$new.forward_sha256 or $old.mini_sha256!=$new.mini_sha256
        or $old.forward_work!=$new.forward_work or $old.mini_work!=$new.mini_work
     then error("complete endpoint words or original work changed") else . end
   | {programs:$old.programs,classes:$old.classes,
      complete_endpoint_words_and_work_equal:true,
      head_speedup:($old.head_build_ns/$new.head_build_ns),
      known_prediction_speedup:($old.known_predict_ns/$new.known_predict_ns),
      exact_prediction_speedup:($old.unknown_predict_ns/$new.unknown_predict_ns),
      forward_total_speedup:($old.forward_total_ns/$new.forward_total_ns),
      minibatch_total_speedup:($old.mini_total_ns/$new.mini_total_ns),
      after_forward_ms:($new.forward_total_ns/1e6),
      after_minibatch_ms:($new.mini_total_ns/1e6),
      original_data_used:false}]
