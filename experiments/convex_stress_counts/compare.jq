# Postprocessing only, with both immutable preceding cohorts supplied explicitly.
def identity: [.arm,.task,.seed,.scale]|tojson;
def require($p;$message): if $p then . else error($message) end;
($ARGS.named.expected//48) as $expected |
require(.planned==$expected and (.cells|length)==$expected; "wrong count-cohort size") |
require($adaptive[0].source_fingerprint=="5a35847751311093af3259b7070362d60304cafedfb0241e9fbf2b82c311c3c2";
  "wrong adaptive baseline") |
require($completion[0].source_fingerprint=="c896fdbd0a2a3026942e49100f2c315dfc78defb452ad31ebc3aac2fd27b0c54";
  "wrong completion baseline") |
require($association[0].source_fingerprint==$completion[0].source_fingerprint and
  ($association[0].cells|length)==6; "wrong associative completion baseline") |
($adaptive[0].cells|INDEX(identity)) as $a |
(($completion[0].cells+$association[0].cells)|INDEX(identity)) as $c |
require(($a|length)==210 and ($c|length)==30; "incomplete baseline inventory") |
. as $new |
[.cells[] | . as $cell | ($c[identity]//$a[identity]) as $old |
  require($old!=null and $old.replay_exit==0; "missing replayed baseline") |
  require(.result==null or .result.data_fingerprints==$old.result.data_fingerprints;
    "data fingerprint mismatch") |
  {arm,task,seed,scale,status,passed,
   primary_accuracy:.primary.sequence_accuracy,
   train_accuracy:.result.splits.train.sequence_accuracy,
   all_splits_99:(.result.splits!=null and ([.result.splits[]|select(.examples>0)|.exact*100>=.examples*99]|all)),
   updates:.result.updates,fresh_gap:.result.training.mean_fw_gap,
   baseline:(if $c[identity]!=null then "completion-v1" else "adaptive-v1" end),
   baseline_primary:$old.primary.sequence_accuracy,baseline_pass:$old.passed,
   baseline_updates:$old.result.updates,
   delta_primary:(if .primary!=null and $old.primary!=null then
     .primary.sequence_accuracy-$old.primary.sequence_accuracy else null end)}] as $cells |
{source_fingerprint:$new.source_fingerprint,
 interpretation:"Seed-zero comparison on identical frozen data and diagnostic budgets. Prefer the latest completion baseline where tested; use adaptive-v1 otherwise. No transformer or matched-time comparison.",
 cells:$cells,
 totals:($cells|group_by([.arm,.scale])|map({arm:.[0].arm,scale:.[0].scale,planned:length,
   passed:map(select(.passed))|length,
   all_splits_passed:map(select(.all_splits_99))|length,
   newly_passed:map(select(.passed and (.baseline_pass|not)))|length,
   lost_passes:map(select((.status=="replayed" or .status=="no_consistent_key") and
     (.passed|not) and .baseline_pass))|length}))}
