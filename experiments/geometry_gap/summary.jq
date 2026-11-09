# Native jq summary of every declared state, with no selected seeds or probes.
def coverage($method; $rank):
  if $method == "all" then .legacy_coverage[] | select(.family == "readout" and .rank == $rank)
  else .coverage[] | select(.rank == $rank) end;
def describe($method):
  . as $rows | ($rows[-1] | coverage($method; 32)) as $end |
  {accuracy: $rows[-1].accuracy,
   readout32: $end.coverage,
   readout33: ($rows[-1] | coverage($method; 33) | .coverage),
   readout128: ($rows[-1] | coverage($method; 128) | .coverage),
   count: $end.count, correct: $end.correct, certified: $end.certified,
   uncertified_incorrect: ($end.count - $end.correct),
   uncertified_correct: ($end.correct - $end.certified),
   late_min_readout32: ([$rows[] | select(.step >= 5000) | coverage($method; 32) | .coverage] | min),
   readout32_regressions: ([range(1; $rows|length) as $i |
     select(($rows[$i] | coverage($method;32) | .coverage) <
            ($rows[$i-1] | coverage($method;32) | .coverage))] | length),
   crossings: [0.5, 0.9, 0.99] | map(. as $level |
     {level: $level,
      accuracy: ([$rows[] | select(.accuracy >= $level) | .step] | first),
      coverage: ([$rows[] | select((coverage($method;32)|.coverage) >= $level) | .step] | first)})};
if .status != "pass" or .failed != 0 or (.cases|length) != 744
then error("complete verified construction is required") else . end |
(.cases | group_by([.cohort,.seed]) | map(sort_by(.step) as $rows |
  {cohort: $rows[0].cohort, seed: $rows[0].seed,
   all: ($rows | describe("all")), active: ($rows | describe("active")),
   final_new: ($rows[-1].coverage[] | select(.rank==32) | .newly_certified),
   final_lost: ($rows[-1].coverage[] | select(.rank==32) | .lost_certificates)})) as $pairs |
{status, seconds, source_sha256, pairs:$pairs,
 margin_checks: ([.cases[].margin_checks]|add),
 legacy_exact_matches: ([.cases[].legacy_exact_matches]|add),
 false_certificates: ([.cases[].false_certificates]|add),
 violations: ([.cases[].violations]|add),
 cohorts: ["lr001","lr01","settle"] | map(. as $cohort |
   ["all","active"] | map(. as $method |
     ([$pairs[]|select(.cohort==$cohort)|.[$method]]) as $rows |
     {cohort:$cohort, method:$method, seeds:($rows|length),
      mean_final_accuracy: ([$rows[].accuracy]|add/length),
      mean_final_coverage: ([$rows[].readout32]|add/length),
      mean_late_min_coverage: ([$rows[].late_min_readout32]|add/length),
      late_min_range: ([$rows[].late_min_readout32]|[min,max]),
      endpoint_range: ([$rows[].readout32]|[min,max]),
      endpoint99: ([$rows[]|select(.readout32>=0.99)]|length),
      all_late90: ([$rows[]|select(.late_min_readout32>=0.9)]|length),
      all_late99: ([$rows[]|select(.late_min_readout32>=0.99)]|length),
      uncertified_incorrect: ([$rows[].uncertified_incorrect]|add),
      uncertified_correct: ([$rows[].uncertified_correct]|add)})) | add,
 conditioning32: {min_rank:([.cases[].conditioning32.min_rank]|min),
   max_rank:([.cases[].conditioning32.max_rank]|max),
   max_coefficient_norm:([.cases[].conditioning32.max_coefficient_norm]|max),
   max_orthogonality_penalty:([.cases[].conditioning32.max_orthogonality_penalty]|max),
   smallest_training_singular:([.cases[].conditioning32.smallest_training_singular]|min)}}
