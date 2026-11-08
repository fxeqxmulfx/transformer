# Validate retained evidence against the frozen cohort and ordering, including
# raw observations rather than relying only on the runner's pass flags.
def require($ok;$why): if $ok then . else error($why) end;
def arms: ["real-swiglu","complex-diagonal","complex-product-radial","complex-product-plain"];
def states: [arms[] as $a|[7,11][] as $s|[0,5000,10000][] as $t|[$a,$s,$t]]|sort;
require(.status=="pass" and .inputs.unchanged and (.inputs.files|length)==73;"final status/input ledger") |
require(.preflight.passed and ([.preflight.cases[]|[.arm,.seed,.step]]|sort)==states;"saved-state preflight") |
require(all(.preflight.cases[];
  .passed and .audit.initial_match and .audit.metrics_match and
  (.audit.state_hashes|unique|length)==1 and
  all(.audit.warmup[];.passed and (.hashes|unique|length)==1));"preflight identity") |
require(([.preflight.cases[].audit.warmup[]]|length)==160;"warmup coverage") |
require((.preflight.edges.cases|length)==8 and
  all(.preflight.edges.cases[];.passed and (.hashes|unique|length)==1);"edge coverage") |
require(.benchmark.passed and ([.benchmark.cases[]|[.arm,.seed,.step]]|sort)==states;"benchmark completion") |
require(all(.benchmark.cases[];
  .passed and (.samples|length)==48 and ([.samples[]|select(.warmup==false)]|length)==36 and
  ([.samples[].round]|unique)==[range(12)] and
  (.samples as $s|all(range(12);. as $r|
    all("update","complete";. as $w|
      [$s[]|select(.round==$r and .workload==$w)] as $pair|
      ($pair|length)==2 and ($pair|map(.state_sha256)|unique|length)==1 and
      ($pair|map(.method))==(if $r%2==0 then ["source-adamw","compact-adamw"] else ["compact-adamw","source-adamw"] end) and
      all($pair[];.warmup==($r<3) and .updates==(if $w=="update" then 256 else 8 end) and .wall_seconds>0 and .gpu_seconds>=0)) and
    [$s[]|select(.round==$r)][0].workload==(if $r%2==0 then "update" else "complete" end))));"paired samples/order") |
require(.training.valid_completed_cohort and .training.completed_runs==16 and
  ([.training.groups[]|[.arm,.seed]]|sort)==([arms[] as $a|[29,31][] as $s|[$a,$s]]|sort);"training completion") |
require(all(.training.groups[];
  . as $g|
  (.runs|map(select(.method=="source-adamw"))|first) as $s|
  (.runs|map(select(.method=="compact-adamw"))|first) as $c|
  (.seed==29 or .seed==31) and (.runs|length)==2 and
  (.matches|length)==41 and all(.matches[];.passed) and
  (.runs|map(.method)|sort)==["compact-adamw","source-adamw"] and
  all(.runs[];.status=="complete" and .completed_steps==4000 and
    (.history|map(.step))==[range(0;4001;100)] and
    (.chunks|length)==40 and ([.chunks[]|.to-.from]|add)==4000 and
    all(.chunks[];.passed and .seconds>0) and .checkpoint.valid and
    .checkpoint.sha256==.history[-1].state_sha256 and
    .cost.full_backwards==4000 and .cost.training_forward_batches==4000 and
    .cost.optimizer_kernels==4000 and .cost.evaluation_forward_batches==820 and
    .cost.canonical_exports==41) and
  all(range(41);. as $i|$g.runs[0].history[$i] as $a|$g.runs[1].history[$i] as $b|
    $a.state_valid and $b.state_valid and $a.state_sha256==$b.state_sha256 and
    $a.train==$b.train and $a.validation==$b.validation) and
  $c.moment_bytes==1146880 and
  $s.moment_bytes==(if .arm=="real-swiglu" then 1146880 else 2194432 end) and
  $s.allocated_payload_bytes-$c.allocated_payload_bytes==(if .arm=="real-swiglu" then 0 else 1047552 end));"training identity/cost/memory") |
{verified:true,source_sha256,candidate_ptx_sha256,
  preflight_states:24,warmup_updates:160,edge_cases:8,
  paired_workloads:576,benchmark_samples:1152,
  training_runs:16,paired_observations:328,full_backwards:64000}
