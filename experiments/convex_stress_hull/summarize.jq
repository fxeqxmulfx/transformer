. as $rows
| $prep[0] as $input
| [$input.tasks[].task as $t | $input.protocol.arms[] as $a | $input.protocol.candidates[] | [$t,$a,.]] as $expected
| [$rows[] | [.task,.arm,.candidate]] as $actual
| if ($actual|sort)!=($expected|sort) or ($actual|unique|length)!=($actual|length)
  then error("missing, duplicate or unplanned candidate record") else . end
| if any($rows[]; .source_fingerprint!=$input.source_fingerprint or .protocol!=$input.protocol)
  then error("mixed producer/protocol") else . end
| if any($rows[]; .status as $status | (["complete","ineligible","fit_failure"]|index($status))==null)
  then error("nonterminal candidate") else . end
| {planned:($expected|length),completed:length,
   statuses:[group_by([.candidate,.status])[] | {candidate:.[0].candidate,status:.[0].status,count:length}],
   results:[.[] | {task,tier,arm,candidate,status,updates,stop_reason,training_ce,ideal_ce_excess,
      elapsed_seconds,examples_presented,prefixes_presented,first_primary_99_update,
      primary:.metrics[.selection].sequence_accuracy,
      primary_pass:(if .status=="complete" then .metrics[.selection].exact*100>=.metrics[.selection].examples*99 else false end)}]}
