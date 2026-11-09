# Retain every seed, rank, method, probe crossing and sampled regression.
def coverage($method; $rank):
  if $method == "all" then .legacy_coverage[] | select(.family=="readout" and .rank==$rank)
  elif $method == "active" then .active_coverage[] | select(.rank==$rank)
  else .coverage[] | select(.family==$method and .rank==$rank) end;
def describe($method; $rank):
  . as $rows | ($rows[-1] | coverage($method;$rank)) as $end |
  {method:$method, rank:$rank, accuracy:$rows[-1].accuracy,
   final_coverage:$end.coverage, count:$end.count, correct:$end.correct, certified:$end.certified,
   incorrect:($end.count-$end.correct), correct_uncertified:($end.correct-$end.certified),
   gained_vs_active:(if $method=="support" or $method=="envelope" then $end.newly_certified else null end),
   lost_vs_active:(if $method=="support" or $method=="envelope" then $end.lost_certificates else null end),
   late_min:([$rows[]|select(.step>=5000)|coverage($method;$rank)|.coverage]|min),
   regressions:([range(1;$rows|length) as $i |
     select(($rows[$i]|coverage($method;$rank)|.coverage)<($rows[$i-1]|coverage($method;$rank)|.coverage))]|length),
   crossings:[0.5,0.9,0.99]|map(. as $level|{level:$level,
     accuracy:([$rows[]|select(.accuracy>=$level)|.step]|first),
     coverage:([$rows[]|select((coverage($method;$rank)|.coverage)>=$level)|.step]|first)})};
if .status!="pass" or .failed!=0 or (.cases|length)!=744
then error("complete construction required") else . end |
(.cases|group_by([.cohort,.seed])|map(sort_by(.step))) as $groups |
if ($groups|length)!=24 or any($groups[];length!=31)
then error("seed/probe inventory changed") else . end |
($groups|map(. as $rows|{cohort:$rows[0].cohort,seed:$rows[0].seed,
  methods:["all","active","support","envelope"]|map(. as $method|
    {method:$method,ranks:[0,32,33,128]|map(. as $rank|$rows|describe($method;$rank))})})) as $pairs |
{status,seconds,source_sha256,pairs:$pairs,
 margin_checks:([.cases[].margin_checks]|add),
 active_exact_matches:([.cases[].legacy_exact_matches]|add),
 false_certificates:([.cases[].false_certificates]|add), violations:([.cases[].violations]|add),
 moment_coordinates:([.cases[].moment_check.coordinates]|add),
 max_moment_invariant_excess:([.cases[].moment_check.max_relative_invariant_excess]|max),
 max_adaptive_direction:([.cases[].moment_check.max_adaptive_direction]|max),
 aggregates:["lr001","lr01","settle"]|map(. as $cohort|
   ["all","active","support","envelope"]|map(. as $method|
     [0,32,33,128]|map(. as $rank|
       ([$pairs[]|select(.cohort==$cohort)|.methods[]|select(.method==$method)|.ranks[]|select(.rank==$rank)]) as $rows|
       {cohort:$cohort,method:$method,rank:$rank,seeds:($rows|length),
        mean_final_accuracy:([$rows[].accuracy]|add/length),
        mean_final_coverage:([$rows[].final_coverage]|add/length),
        late_min_range:([$rows[].late_min]|[min,max]),
        endpoint_range:([$rows[].final_coverage]|[min,max]),
        endpoint99:([$rows[]|select(.final_coverage>=0.99)]|length),
        all_late99:([$rows[]|select(.late_min>=0.99)]|length),
        incorrect:([$rows[].incorrect]|add),correct_uncertified:([$rows[].correct_uncertified]|add),
        gained_vs_active:([$rows[].gained_vs_active]|add),lost_vs_active:([$rows[].lost_vs_active]|add)}))|add)|add}
