# Complete declared inventory; zero useful horizons and numerical failures stay visible.
def stats: if length==0 then null else {n:length,min:min,max:max,mean:(add/length)} end;
def tracks:
  [.cases[] as $c | $c.endpoints[]? as $e | $e.checks[]? as $f | $f.tracks[]? |
    . as $track | . + {cohort:$c.cohort,seed:$c.seed,anchor:$c.anchor,horizon:$e.horizon,family:$f.family,
      full_increase:([$e.full_potentials[]? | select(.metric==$track.metric) | .increase][0])}];
tracks as $tracks |
{
  status,seconds,total_seconds,charged_seconds,failed,error,
  cases:(.cases|length),
  endpoints:([.cases[].endpoints[]?]|length),
  failures:[.cases[] | select(.status!="pass") | {cohort,seed,anchor,status,error,
    endpoints:[.endpoints[]? | select(.status!="pass")]}],
  coefficients:($tracks | group_by([.family,.metric,.method]) | map({
    family:.[0].family,metric:.[0].metric,method:.[0].method,records:length,
    stability:([.[].stability]|stats),
    positive_passes:([.[]|select(.horizon>0 and .condition)]|length),
    moving_positive_passes:([.[]|select(.anchor>0 and .horizon>0 and .condition)]|length),
    anchors:(group_by([.cohort,.seed,.anchor]) | map({
      cohort:.[0].cohort,seed:.[0].seed,anchor:.[0].anchor,
      useful_horizons:[.[]|select(.condition)|.horizon],
      largest_useful_positive:([0]+[.[]|select(.horizon>0 and .condition)|.horizon]|max),
      minimum_positive_stability:([.[]|select(.horizon>0)|.stability]|min)
    })),
    ce_scaled_excess:([.[]|.actual_ce.scaled_excess // empty]|stats),
    reference_increases:([.[]|select(.actual_ce.reference_increase==true)]|length),
    reference_increases_with_condition:([.[]|select(.condition and .actual_ce.reference_increase==true)]|length),
    moving_positive_full_increases_with_condition:([.[]|select(.anchor>0 and .horizon>0 and .condition and .full_increase)]|length)
  })),
  full_potentials:([.cases[] as $c | $c.endpoints[]? as $e | $e.full_potentials[]? |
    . + {cohort:$c.cohort,seed:$c.seed,anchor:$c.anchor,horizon:$e.horizon}] |
    group_by(.metric) | map({metric:.[0].metric,records:length,
      positive_increases:([.[]|select(.horizon>0 and .increase)]|length),
      moving_positive_increases:([.[]|select(.anchor>0 and .horizon>0 and .increase)]|length),
      relative_changes:([.[].relative_change]|stats)})),
  numerical:{
    probability_excess:([.cases[].endpoints[]?.checks[]?.probability_excess]|stats),
    feature_scaled_excess:([.cases[].endpoints[]?.checks[]?.feature_scaled_excess]|stats),
    gradient_error:([.cases[].endpoints[]?.partial_gradient_error // empty]|stats)
  }
}
