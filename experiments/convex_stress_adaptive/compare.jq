# Postprocessing only: historical full-budget endpoints versus capped repair.
# jq --slurpfile full OLD_FULL --slurpfile mini OLD_MINI -f compare.jq NEW_SUMMARY
def key: [.arm,.task,.seed]|tojson;
def require($p;$message): if $p then . else error($message) end;
require(.planned==210 and (.cells|length)==210; "expected all 210 adaptive cells") |
require($full[0].source_fingerprint=="4452feddd3eeb5cc82464029b2f733c4fd7209bf8a79156fe907ed3d3c19796e";
  "wrong full-batch baseline") |
require($mini[0].source_fingerprint=="8bca8579d8f1c2113e7c863199485f32bc1e2004a72b4c5966b546115e5555c4";
  "wrong minibatch baseline") |
($full[0].cells|INDEX(key)) as $full_cells |
($mini[0].cells|INDEX(key)) as $mini_cells |
require(($full_cells|length)==210 and ($mini_cells|length)==210; "incomplete baseline inventory") |
. as $new |
[.cells[] | . as $cell |
  {arm:(if .arm=="forward_only" then .arm else "frank_wolfe" end),task,seed} as $identity |
  (if .arm=="minibatch" then $mini_cells[$identity|key] else $full_cells[$identity|key] end) as $old |
  require($old != null; "missing matching baseline") |
  require(.primary==null or $old.primary==null or .primary.examples==$old.primary.examples;
    "primary split denominator mismatch") |
  {arm,task,seed,scale,status,passed,
   primary_accuracy:.primary.sequence_accuracy,
   train_accuracy:.result.splits.train.sequence_accuracy,
   all_reported_nonempty_splits_99:(.result.splits != null and
     ([.result.splits[]|select(.examples>0)|.exact*100>=.examples*99]|all)),
   updates:.result.updates,fresh_gap:.result.training.mean_fw_gap,
   baseline_status:($old.status//"complete"),baseline_primary:$old.primary.sequence_accuracy,
   baseline_pass:($old.pass//false),
   delta_primary:(if .primary!=null and $old.primary!=null then
     .primary.sequence_accuracy-$old.primary.sequence_accuracy else null end)}] as $cells |
{source_fingerprint:$new.source_fingerprint,
 interpretation:"Seed zero; exact same frozen data and metrics, different diagnostic update caps. Scale 10 also changes the fixed CE logit scale. No matched time or fresh transformer comparison.",
 cells:$cells,
 totals:($cells|group_by([.arm,.scale])|map({arm:.[0].arm,scale:.[0].scale,planned:length,
   passed:map(select(.passed))|length,
   all_reported_splits_passed:map(select(.all_reported_nonempty_splits_99))|length,
   newly_passed:map(select(.passed and (.baseline_pass|not)))|length,
   lost_passes:map(select((.status=="replayed" or .status=="no_consistent_key") and
     (.passed|not) and .baseline_pass))|length}))}
