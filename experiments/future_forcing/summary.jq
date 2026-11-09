# Complete task #205 inventory, with no dropped early or inconclusive windows.
if .passed != true or .failed != 0 or .prefixes != 444000
  or (.cases|length) != 144 or any(.cases[]; .status != "pass")
then error("incomplete future-forcing audit") else . end |
def span: [min,max];
def total(f): map(f)|add;
[.cases[] | . as $case |
  . + {tracks: [.tracks[] |
    . + {first_uncertified_steps: [.longest_prefix[] |
      if . == $case.prefixes then null else $case.anchor + . + 1 end]}]}] as $rows |
{
  source_sha256, seconds, combined_postprocessing_seconds, passed, failed, prefixes,
  cases: ($rows|length),
  max_scaled_excess: ([.cases[].max_scaled_excess[]]|max),
  violations: ([.cases[].violations[]]|add),
  memory_factor: {per_step:81/95, after_10:pow(81/95;10),
    after_50:pow(81/95;50), after_100:pow(81/95;100)},
  groups: [$rows|group_by([.cohort,.anchor])[] |
    {cohort:.[0].cohort, anchor:.[0].anchor, end:.[0].end,
      seeds:length, length:.[0].prefixes, prefixes:total(.prefixes),
      moving_updates:total(.moving_updates),
      minimum_observed_share_range:(map(.minimum_observed_share)|span),
      warm_minimum_share_gain_range:
        (map(.tracks[0].minimum_share_lower-.tracks[1].minimum_share_lower)|span),
      tracks: [range(2) as $m |
        {method:.[0].tracks[$m].method,
          minimum_share_lower_range:(map(.tracks[$m].minimum_share_lower)|span),
          endpoint_active_radius_range:(map(.tracks[$m].endpoint_radii[0])|span),
          endpoint_total_radius_range:(map(.tracks[$m].endpoint_radii[1])|span),
          thresholds:[range(4) as $j |
            {threshold:([0,0.5,0.6,2/3][$j]),
              longest_prefix_range:(map(.tracks[$m].longest_prefix[$j])|span),
              first_uncertified_step_range:
                (map(.tracks[$m].first_uncertified_steps[$j])|map(select(.!=null))|
                  if length == 0 then null else span end),
              prefixes_above:total(.tracks[$m].prefixes_above[$j]),
              full_windows:([.[]|select(.tracks[$m].longest_prefix[$j]==.prefixes)]|length),
              warm_prefix_gain_range:
                (map(.tracks[0].longest_prefix[$j]-.tracks[1].longest_prefix[$j])|span)}]}]}],
  case_rows:$rows
}
