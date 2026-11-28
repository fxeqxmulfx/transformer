# Verify a fresh arm against its frozen discovery configuration and its two
# independent re-evaluations. Observed correctness is reported separately.
$r[0] as $r | $v[0] as $v | $c[0] as $c |
($r.identity | del(.budget,.seed,.order_seed)) as $id |
if $id==$expected and $r.identity.case=="grok-division-p97" and $r.identity.seed==$seed and
   $r.identity.budget==$cap and $r.identity.order_seed==(($seed+10000)%4294967296) and
   $r.identity.observe==256 and $r.identity.stop_on=="first-strict-100-percent" and
   $r.identity==$v.identity and $r.weights_sha256==$v.weights_sha256 and $r.weights_sha256==$c.weights_sha256 and
   $r.step==$v.step and $r.step==$c.step and $c.seed==$seed and $r.strict==$v.strict and
   $c.output_tokens_compared==239 and $c.observed_sha256==$r.identity.split_sha256[0] and
   $c.hidden.total==4656 and $r.strict.total==4656 and $c.observed.total==4656 and
   $r.strict.strict_correct==$c.hidden.strict_correct and $r.strict.ties==$c.hidden.ties and
   $r.strict.min_gap==$c.hidden.minimum_gap and $r.step>=0 and $r.step<=$cap and
   $r.strict.all_strict==($c.hidden.strict_correct==4656 and $c.hidden.minimum_gap>0 and $c.hidden.ties==0) and
   (if $r.strict.all_strict then $r.first_strict_step==$r.step and $v.splits.hidden.correct==4656
    else $r.first_strict_step==null and $r.step==$cap end)
then {arm:$arm,seed:$seed,step:$r.step,success:$r.strict.all_strict,
      first_attempt:(if $interrupted then "interrupted" else "complete" end),
      hidden:$c.hidden,observed:$c.observed,gpu:$r.gpu,training_seconds:$r.segment_seconds,
      selected:($r.checkpoint_proposal.selected//"ordinary"),weights_sha256:$r.weights_sha256}
else error("paired replay, observed-data certificate or frozen optimizer settings disagree") end
