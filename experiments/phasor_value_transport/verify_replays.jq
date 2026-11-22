# jq -es --slurpfile report REPORT -f verify_replays.jq REPLAYS.jsonl
def integer: type == "number" and isfinite and . == floor;
def valid_run:
  (.case | type == "string" and length > 0)
  and (.weights_sha256 | type == "string" and test("^[0-9a-f]{64}$"))
  and (.kernite_version | type == "string" and length > 0)
  and (.seed | integer and . >= 0 and . <= 2)
  and (.correct | integer) and (.total | integer and . > 0)
  and .correct <= .total and .correct * 100 >= .total * 99;
def identity: {case,seed,correct,total,model,weights_sha256,kernite_version};
if ($report|length) != 1 then error("expected one cohort report") else . end
| $report[0] as $r
| [ $r.tasks[] as $t | $t.seeds[]
    | if .status != "pass" then error("reported cohort has a missing or failed seed") else . end
    | {case:$t.case,model:$r.model,seed,correct,total,weights_sha256,
       kernite_version:.timing_environment.kernite_version} ] as $expected
| if ($expected|length) == 0 or ($r.model|type) != "string" or ($r.model|length) == 0
     or ($r.tasks | group_by(.case) | any(.[]; length != 1))
     or ($r.tasks | any(.[]; (.seeds|map(.seed)|sort) != [0,1,2]))
     or (all($expected[]; valid_run)|not)
  then error("invalid cohort, checkpoint, crossing, version, or seed membership") else . end
| if all(.[]; valid_run and (.scored_with_kernite | type == "string" and length > 0))
     and (map(identity)|sort_by(.case,.seed)) == ($expected|sort_by(.case,.seed))
  then {verified:length,model:$r.model,scored_with_kernite:map(.scored_with_kernite)|unique}
  else error("replay identity, checkpoint, version, or score differs from the cohort report") end
