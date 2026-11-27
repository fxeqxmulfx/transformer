# Compare two complete runs of run_argand.sh, or of run_v2_stress.sh at commit
# 7399c3bb, at the same observation cap.
# Load suite, control and candidate with --slurpfile and cap with --argjson.
def cell: [.identity.case, .identity.seed];
def sha256: type == "string" and test("^[0-9a-f]{64}$");
def valid_record($task):
  ($task.case.max_updates) as $budget | ($task.case.observe) as $every
  | (if $budget <= $cap then $budget else ($cap / $every | floor) * $every end) as $stop
  | .identity.tier == $task.tier and .budget == $budget
    and (.identity.model | type == "string" and length > 0) and (.identity.suite_sha256 | sha256)
    and .total == $task.heldout_examples and .total > 0
    and (.correct | type == "number" and . == floor and . >= 0) and .correct <= .total
    and (.loss | type == "number" and isfinite and . >= 0)
    and (.state_sha256 | sha256)
    and .passed == (.correct * 100 >= 99 * .total)
    and (.step | type == "number" and . == floor and . >= 0)
    and .step <= $stop and (.step == $budget or .step % $every == 0)
    and (.passed or .step == $stop);
def checked($tasks; $cells):
  sort_by(cell)
  | if map(cell) == $cells and (map(.identity | del(.case, .tier, .seed)) | unique | length) == 1
       and all(.[]; valid_record($tasks[.identity.case]))
    then . else error("incomplete screen or invalid terminal observation") end;
def summary:
  map(select(.control.passed and .candidate.passed)) as $shared
  | {runs:length,
     control_passes:map(select(.control.passed))|length,
     candidate_passes:map(select(.candidate.passed))|length,
     gained:map(select(.candidate.passed and (.control.passed|not)) | [.case,.seed]),
     lost:map(select(.control.passed and (.candidate.passed|not)) | [.case,.seed]),
     shared:($shared|length),
     earlier:($shared|map(select(.candidate.step < .control.step))|length),
     tied:($shared|map(select(.candidate.step == .control.step))|length),
     later:($shared|map(select(.candidate.step > .control.step))|length)};

($suite[0].tasks | map(select(.tier == "short" or .tier == "long")) |
  map({key:.name, value:.}) | from_entries) as $tasks
| ([$tasks|keys[] as $case | $suite[0].seeds[] | [$case,.]] | sort) as $cells
| ($control | checked($tasks; $cells)) as $a
| ($candidate | checked($tasks; $cells)) as $b
| [range(0; $cells|length) as $i | $a[$i] as $x | $b[$i] as $y
   | if ($x.identity|del(.model)) != ($y.identity|del(.model))
     then error("unmatched suite, seed, repeat or trainer") else
     {tier:$x.identity.tier, case:$x.identity.case, seed:$x.identity.seed, total:$x.total,
      control:($x|{passed,step,correct,loss}), candidate:($y|{passed,step,correct,loss})}
     end] as $pairs
| {control:$a[0].identity.model, candidate:$b[0].identity.model, cap:$cap,
   suite_sha256:$a[0].identity.suite_sha256,
   tiers:($pairs|group_by(.tier)|map({key:.[0].tier,value:summary})|from_entries), pairs:$pairs}
