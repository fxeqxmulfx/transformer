# jq --slurpfile counts runs/combined.json -f compare_control.jq CONTROL_SUMMARY
def key: [.arm,.task,.seed,.scale]|tojson;
def require($p;$message): if $p then . else error($message) end;
require(.planned==6 and (.cells|length)==6 and (.cells|map(key)|unique|length)==6;
  "expected six distinct control cells") |
require(.source_fingerprint=="c896fdbd0a2a3026942e49100f2c315dfc78defb452ad31ebc3aac2fd27b0c54";
  "wrong completion source") |
require($counts[0].planned==210 and
  $counts[0].source_fingerprint=="b06514ec0c08f583d74fe431b3e3a97ea898366c849c2c71f0f3742ae03de210";
  "wrong counts cohort") |
($counts[0].cells|INDEX(key)) as $new |
[.cells[] | $new[key] as $c |
  require(.task=="recurrence-fibonacci-k1-c2-lgt" and .seed==0 and
    .run_exit==0 and .replay_exit==0 and $c.run_exit==0 and $c.replay_exit==0;
    "missing completed recurrence pair") |
  require(.result.data_fingerprints==$c.result.data_fingerprints; "data mismatch") |
  {arm,scale,primary:.primary.sequence_accuracy,count_primary:$c.primary.sequence_accuracy,
   splits_equal:(.result.splits==$c.result.splits),
   updates_equal:(.result.updates==$c.result.updates),
   ce_difference:(if .arm=="forward_only" then null else
     .result.training.mean_ce-$c.result.training.mean_ce end),
   gap_difference:(if .arm=="forward_only" then null else
     .result.training.mean_fw_gap-$c.result.training.mean_fw_gap end)}]
