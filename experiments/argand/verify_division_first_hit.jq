# Check replay and observation-only reconstruction for one registered seed.
$r[0] as $r | $c[0] as $c |
if $r.identity.case=="grok-division-p97" and $r.identity.optimizer=="adamw-wsd" and
   $r.identity.seed==$seed and $r.identity.order_seed==(($seed+10000)%4294967296) and
   $r.identity.adamw.budget==$cap and $r.identity.budget==$cap and $r.identity.adamw.weight_decay==0.1 and
   $r.identity.split_sha256[0]=="53e9d2fb9336a74529e88bf582a2fb37483c1f7e79ab8f1fd670de565baf0eef" and
   $r.identity.split_sha256[1]=="1f679f635bd75862e28f1f6dec8cc121a2963f2fa258a0d8d291df0c8b9acc7d" and
   $r.weights_sha256==$c.weights_sha256 and $r.step==$c.step and $c.seed==$seed and
   $c.output_tokens_compared==239 and $c.observed_sha256==$r.identity.split_sha256[0] and
   $r.step>=0 and $r.step<=$cap and $r.strict.total==4656 and $c.hidden.total==4656 and
   $r.splits.hidden.total==4656 and $r.strict.strict_correct==$c.hidden.strict_correct and
   $r.strict.min_gap==$c.hidden.minimum_gap and $r.strict.ties==$c.hidden.ties and
   $r.strict.all_strict==($r.strict.strict_correct==4656 and $r.strict.ties==0 and $r.strict.min_gap>0) and
   (if $kind=="first-strict" then $r.identity.fixed_horizon and $r.strict.all_strict
    else $r.identity.stop_on=="first-strict-100-percent" end) and
   (if $r.strict.all_strict then $r.splits.hidden.correct==4656 else $r.step==$cap end)
then {index:$i,seed:$seed,step:$r.step,success:$r.strict.all_strict,correct:$r.strict.strict_correct,
  total:$r.strict.total,min_gap:$r.strict.min_gap,weights_sha256:$r.weights_sha256,cell:$cell,checkpoint:$kind}
else error("inconsistent first-passage evidence") end
