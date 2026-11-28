if .replayed != true or .runs != 8 or (.records | length) != 8 or
   (.protocol.protocol | IN("argand-sum8-adaptive-page-v1","argand-sum8-adaptive-page-sqrt-v1",
     "argand-sum8-adaptive-page-retry-v1","argand-sum8-adamw-wsd-v1","argand-sum8-adamw-first-noise-v1") | not)
then error("expected the replayed eight-run sum8 cohort") else . end |
.records | map(. as $r |
  ([.initial + {step:0}] + (.milestones // []) +
    [{step:.optimizer.step, observed:.observed, hidden:.hidden}]) |
  unique_by(.step) | {points:., last:last, stop:$r.stop}) as $runs |
([$runs[].points[].step] | unique)[] as $step |
[$runs[] | ([.points[] | select(.step == $step)] | first) //
  (if .stop == "no_resolvable_full_step" and .last.step < $step then .last
   else error("missing observation from a continuing run") end)] |
def stats($key): map(100 * .[$key].correct / .[$key].total) |
  [(add/length), min, max];
([$step] + stats("observed") + stats("hidden")) | @tsv
