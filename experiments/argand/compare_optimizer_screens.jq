# Compare two independently replayed summaries from page_screen_summary.jq.
# Supply each summary with --slurpfile and run with -n.
def cells:
  . as $screen |
  [.cases[] as $task | $task.cells[] |
   {case:$task.name, tier:$task.tier, total:$task.total, seed, step, correct, passed, strict}]
  | sort_by(.case,.seed)
  | if length == $screen.overall.runs and length > 0 and
       (unique_by(.case,.seed) | length) == length
    then . else error("missing or duplicated case and seed") end;
def design: {case, tier, total, seed};
def outcome: {step, correct, passed, strict};
def training_rule: {optimizer,correction} + (if .stop_on then {stop_on} else {} end);
def counts:
  {runs:length,
   control_99:map(select(.control.passed))|length,
   candidate_99:map(select(.candidate.passed))|length,
   control_exact:map(select(.control.strict))|length,
   candidate_exact:map(select(.candidate.strict))|length,
   gained_99:map(select(.candidate.passed and (.control.passed|not)))|length,
   lost_99:map(select(.control.passed and (.candidate.passed|not)))|length,
   shared_99:map(select(.control.passed and .candidate.passed))|length};
($control[0] | cells) as $a | ($candidate[0] | cells) as $b |
if ($a|map(design)) != ($b|map(design)) then error("unmatched evaluation cells") else . end |
[range($a|length) as $i | ($a[$i]|design) +
  {control:($a[$i]|outcome),candidate:($b[$i]|outcome)}] as $pairs |
{control:($control[0]|training_rule),candidate:($candidate[0]|training_rule),
 overall:($pairs|counts),tiers:($pairs|group_by(.tier)|map({key:.[0].tier,value:counts})|from_entries),
 pairs:$pairs}
