# jq --slurpfile adaptive ADAPTIVE_SUMMARY -f compare.jq COMPLETION_SUMMARY
def key: [.arm,.task,.seed,.scale]|tojson;
def require($p;$message): if $p then . else error($message) end;
($ARGS.named.expected//24) as $expected |
require(.planned==$expected and (.cells|length)==$expected; "unexpected completion inventory") |
require(.source_fingerprint=="c896fdbd0a2a3026942e49100f2c315dfc78defb452ad31ebc3aac2fd27b0c54";
  "wrong completion source") |
require($adaptive[0].source_fingerprint=="5a35847751311093af3259b7070362d60304cafedfb0241e9fbf2b82c311c3c2";
  "wrong adaptive source") |
($adaptive[0].cells|INDEX(key)) as $old |
. as $summary |
[.cells[] | . as $cell | $old[key] as $before |
  require($before!=null; "missing paired adaptive cell") |
  require(.result==null or $before.result==null or
    .result.data_fingerprints==$before.result.data_fingerprints; "data mismatch") |
  (if .arm=="forward_only" and .replay_exit==0 and $before.replay_exit==0 then
    .result.model_sha256==$before.result.model_sha256 and
    .result.splits==$before.result.splits and .result.updates==$before.result.updates
    else null end) as $forward_equal |
  require($forward_equal!=false; "unchanged forward control differs") |
  {arm,task,seed,scale,status,passed,forward_equal:$forward_equal,
   primary:.primary.sequence_accuracy,adaptive_primary:$before.primary.sequence_accuracy,
   train:.result.splits.train.sequence_accuracy,adaptive_train:$before.result.splits.train.sequence_accuracy,
   updates:.result.updates,adaptive_updates:$before.result.updates,
   fresh_gap:.result.training.mean_fw_gap,adaptive_fresh_gap:$before.result.training.mean_fw_gap,
   adaptive_status:$before.status}] as $cells |
{source_fingerprint:$summary.source_fingerprint,
 interpretation:"Same frozen splits and fixed caps; only unseen selected-key completion changes CE training. Forward-only models and metrics are exact controls. Targeted follow-up cohorts, not a full-suite comparison.",
 cells:$cells,totals:($cells|group_by([.arm,.scale])|map({arm:.[0].arm,scale:.[0].scale,
   planned:length,passed:map(select(.passed))|length,forward_equal:map(select(.forward_equal==true))|length}))}
