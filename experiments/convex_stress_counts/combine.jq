# jq -s --slurpfile suite ../convex_stress/suite.json -f combine.jq PILOT REMAINING
def identity: [.arm,.task,.seed,.scale];
def require($p;$message): if $p then . else error($message) end;
require(length==2 and .[0].planned==48 and .[1].planned==162; "expected the 48+162 cohorts") |
require(.[0].source_fingerprint==.[1].source_fingerprint and
  .[0].source_fingerprint=="b06514ec0c08f583d74fe431b3e3a97ea898366c849c2c71f0f3742ae03de210";
  "cohort source mismatch") |
.[0].source_fingerprint as $fingerprint |
([.[].cells[]]) as $cells |
([$suite[0].tasks[].name as $task | ["full_batch","minibatch","forward_only"][] as $arm |
  [1,10][] as $scale | [$arm,$task,0,$scale]]|sort) as $expected |
require(($cells|map(identity)|sort)==$expected and ($cells|map(identity)|unique|length)==210;
  "combined cohort is not the exact original suite") |
{source_fingerprint:$fingerprint,planned:210,
 interpretation:"Two disjoint cohorts: the initial 48-cell pilot, then all remaining 162 cells selected after the pilot parity outcomes. Unchanged model binary and training protocol.",
 statuses:($cells|group_by(.status)|map({key:.[0].status,value:length})|from_entries),
 totals:($cells|group_by([.arm,.scale])|map({arm:.[0].arm,scale:.[0].scale,planned:length,
   passed:map(select(.passed))|length,replay_verified:map(select(.replay_exit==0))|length})),
 cells:$cells}
