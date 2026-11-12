# Validate the complete retained scope before any new training or allocation.
. as $rows
| $prep[0] as $p
| $recovery[0] as $r
| [$p.tasks[].task as $task | $p.protocol.arms[] as $arm |
   $p.protocol.candidates[] | $arm+"--"+$task+"--"+.] as $expected
| [$rows[] | .arm+"--"+.task+"--"+.candidate] as $actual
| if ($expected|length)!=315 or ($expected|unique|length)!=315 or
     $p.planned_choices!=105 or ($p.tasks|length)!=35 or
     ($actual|length)!=311 or ($actual|unique|length)!=311 or
     ($actual-$expected|length)!=0 or
     (($expected-$actual)|sort)!=($r.missing|sort)
  then error("retained inventory is not exactly the original 311 cells and four gaps") else . end
| if $p.source_fingerprint!=$r.source_fingerprint or
     any($rows[]; .source_fingerprint!=$p.source_fingerprint or .protocol!=$p.protocol)
  then error("mixed source or protocol in retained inventory") else . end
| if any($rows[]; . as $row | ($p.tasks|map(select(.task==$row.task))|.[0]) as $task |
     .data_fingerprints!=$task.data_fingerprints or .vocab!=$task.vocab or .scale!=$task.scale or
     .seed!=$p.protocol.seed or
     (["complete","ineligible","fit_failure"]|index($row.status))==null or
     (if .status=="complete" then (.model_sha256|type)!="string" or
          (.model_sha256|test("^[0-9a-f]{64}$")|not)
      else .model_sha256!=null end))
  then error("nonterminal or inconsistent retained result") else . end
| {source_fingerprint:$p.source_fingerprint,retained:($actual|sort),missing:$r.missing,
   planned:($expected|length),choices:$p.planned_choices,recovery_protocol:$r}
