# jq -es --slurpfile report REPORT -f verify_replays.jq REPLAYS.jsonl
# Add --argjson passing_only true to count unfinished and exhausted seeds separately.
def integer: type == "number" and isfinite and . == floor;
def valid_run:
  (.case | type == "string" and length > 0)
  and (.weights_sha256 | type == "string" and test("^[0-9a-f]{64}$"))
  and (.kernite_version | type == "string" and length > 0)
  and (.seed | integer and . >= 0 and . <= 2)
  and (.step | integer and . >= 0)
  and (.correct | integer) and (.total | integer and . > 0)
  and .correct <= .total and .correct * 100 >= .total * 99;
def identity: {case,seed,step,correct,total,model,weights_sha256,kernite_version};
if ($report|length) != 1 then error("expected one cohort report") else . end
| $report[0] as $r
| ($ARGS.named | if has("passing_only") then .passing_only else false end) as $passing_only
| if ($passing_only|type) != "boolean" then error("passing_only must be boolean") else . end
| [ $r.tasks[] as $t | $t.seeds[]
    | if .status == "pass" then .
      elif $passing_only and (.status == "pending" or .status == "budget miss") then empty
      else error("reported cohort has a missing or failed seed") end
    | {case:$t.case,model:$r.model,seed,step,correct,total,weights_sha256,
       kernite_version:.timing_environment.kernite_version} ] as $expected
| if ($r.tasks | type != "array" or length == 0) or ($r.model|type) != "string" or ($r.model|length) == 0
     or ($r.tasks | group_by(.case) | any(.[]; length != 1))
     or ($r.tasks | any(.[]; (.case | type != "string" or length == 0)
                        or (.seeds|map(.seed)|sort) != [0,1,2]))
     or (all($expected[]; valid_run)|not)
  then error("invalid cohort, checkpoint, crossing, version, or seed membership") else . end
| map(. + {step:.observation.step})
| if all(.[]; valid_run and (.scored_with_kernite | type == "string" and length > 0)
         and .observation.pass == true
         and ([.case,.seed,.correct,.total] == (.observation | [.case,.seed,.correct,.total])))
     and (map(identity)|sort_by(.case,.seed)) == ($expected|sort_by(.case,.seed))
  then {verified:length,model:$r.model,scored_with_kernite:map(.scored_with_kernite)|unique}
       + (if $passing_only then
            {pending:([$r.tasks[].seeds[]|select(.status=="pending")]|length),
             budget_misses:([$r.tasks[].seeds[]|select(.status=="budget miss")]|length)}
          else {} end)
  else error("replay identity, checkpoint, version, or score differs from the cohort report") end
