# Input order: completed .001, completed .01, early settling producer reports.
if length != 3 or any(.[]; .status != "pass" or (.seeds|length) != 8)
then error("three complete eight-seed cohorts are required") else . end |
. as $cohorts | range(0;3) as $i |
([$cohorts[$i].seeds[].geometry[]] | group_by(.step))[] |
if length != 8 then error("missing paired probe") else . end |
. as $rows |
([$rows[].metrics.validation.accuracy]) as $accuracy |
([$rows[].coverage.validation[] | select(.family=="readout" and .rank==32) | .coverage]) as $coverage |
[$i, $rows[0].step,
 ($accuracy|add/length), ($accuracy|min), ($accuracy|max),
 ($coverage|add/length), ($coverage|min), ($coverage|max)] | @tsv
