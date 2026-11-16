# Reconcile named cases, retaining historical scope after raw-output cleanup.
# This consumes independently verified reports; it is not a raw-score verifier.
def require($ok; $message): if $ok then . else error($message) end;
def names_unique($xs; $n):
  ($xs | length) == $n and ($xs | unique | length) == $n;
def three_modes($xs):
  ($xs | map(.mode) | sort) == ["Forward", "Full", "Mini"];
def passed($xs):
  three_modes($xs) and all($xs[]; .all_heldout99 == true and .error == null);
def coordinate_cases($report):
  [ $report.cases[]
    | .results as $r
    | [$r.endpoints[] | select(.prior == "Coordinates")] as $m
    | require(three_modes($m); "historical coordinate modes missing or duplicated")
    | select(passed($m)) | $r.case ];
# These are the original probe split inventories, not model inputs.
def probe_splits($name):
  if $name == "gpt-probe-order-discrimination" then ["train", "validation"]
  elif $name == "gpt-probe-key-value-binding" then
    ["test", "train", "validation", "validation/novel"]
  else ["test", "train", "validation"] end;

$retained[0] as $r |
$original35[0] as $o |
$catalog77[0] as $c |
$historical77[0] as $h |
$bounded[0] as $b |
$verified[0] as $v |
$probes[0] as $p |
require($r.incomplete_report == false and $r.parse_error == null;
  "incomplete retained historical summary") |
require($r.source_sha256 ==
  ($c.references[] | select(.file == "baseline344.json") | .sha256);
  "retained baseline identity differs from the catalog77 reference") |
require($b.protocol.references["catalog77-report.json"] == $catalog77_sha256 and
  $b.protocol.references["catalog77-historical-verification.json"] == $historical77_sha256 and
  $b.protocol.references["original35-accuracy.json"] == $original35_sha256 and
  $v.report.sha256 == $bounded_sha256;
  "a report differs from its frozen or verified content hash") |
require($h.independently_reproduced == true and
  $h.catalog_all_modes_all_heldout99_cases == 77 and
  $v.independently_reproduced == true and $v.artifacts == 686 and
  $v.new_complete_raw_artifacts_verified == true and
  $v.historical_raw_artifacts_reverified == false and
  $p.identical_all_artifacts == true and $p.complete_inventory_verified == true and
  ($p.artifacts | length) == 66;
  "required historical or fresh verification is missing") |
require(($o | map(.initializer) | sort) == ["first_small_ce", "minimum_ce"];
  "original35 initialization policies differ") |
[$o[] |
  require((.choices | length) == 105 and
    all(.choices[]; .all_heldout_both99 == true);
    "original35 endpoint or all-heldout99 result missing") |
  [.choices | group_by(.task)[] |
    require((map(.training_mode) | sort) == ["forward_only", "full_batch", "minibatch"];
      "original35 modes missing or duplicated") | .[0].task] | sort] as $old |
require($old[0] == $old[1] and names_unique($old[0]; 35);
  "original35 policies disagree on the named case set") |
$r.summary as $s |
$s.baseline343.reference340.baseline339 as $g |
$g.baseline334 as $f |
require($s.complete == true and $g.complete == true and $f.complete == true;
  "a historical cohort is incomplete") |
[$f.cases[] | .results as $x |
  require(three_modes($x.modes); "prefix cohort modes missing or duplicated") |
  select(passed($x.modes)) | $x.case] as $prefix |
coordinate_cases($g) as $generation |
coordinate_cases($s) as $counts |
require(names_unique($prefix; 16) and names_unique($generation; 7) and
  names_unique($counts; 11);
  "historical cohort counts differ from their reviewed scope") |
($old[0] + $prefix + $generation + $counts | sort) as $old69 |
require(names_unique($old69; 69) and $s.catalog_all_modes_all_heldout99_cases == 69;
  "historical cohorts overlap or fail to reconstruct69") |
[$c.cases[] |
  [.endpoints[] | select(.prior == "Coordinates")] as $m |
  require(passed($m) and all($m[]; .original_success == true);
    "basis/language cohort is not complete in every mode") | .case] as $basis |
require($c.complete == true and names_unique($basis; 8);
  "basis/language cohort count differs") |
($old69 + $basis | sort) as $old77 |
require(names_unique($old77; 77); "historical basis/language cases overlap") |
require($b.complete == true and ($b.cases | length) == 54 and
  $b.data_seed == 1 and $v.data_seed == 1 and $p.seeds == [1];
  "cohort completeness, denominator or seed differs") |
[$b.cases[] | select(.all_modes_original_heldout99 == true and
  .case.existing35 == false) | .case.name] as $new |
require(names_unique($new; 19) and
  ($new | sort) == ($v.new_all_modes_all_heldout99_cases | sort) and
  ($new - $old77 | length) == 19 and $v.regressed_old_controls == [];
  "bounded additions overlap history or differ from verification") |
[$p.cases[] | select(.model_refusal == null and .source_refusal == null) |
  .name as $name |
  require((.modes | map(.mode) | sort) == ["Forward", "Mini"] and
    all(.modes[];
      (.splits | map(.split) | sort) == probe_splits($name) and
      all(.splits[]; .score.raw99 == true and .score.exact99 == true and
        .score.original_success == true and .score.raw.target_count > 0));
    "probe modes, complete splits or pass criteria differ") | .name] as $probe |
require(names_unique($probe; 5) and ($probe - $old77 | length) == 2;
  "probe additions differ from the reviewed two-case extension") |
($old77 + $new | unique) as $three |
($three + $probe | unique) as $two |
require(($three | length) == $v.catalog_union_cases and
  ($three | length) == 96 and ($two | length) == 98;
  "named union does not match the expected verified coverage") |
{
  scope: "Union of separately frozen seed1 cohorts, not one universal checkpoint or three seeds",
  historical_raw_outputs_reverified: false,
  bounded_artifacts_independently_matched: $v.artifacts,
  historical_three_mode_cases: $old77,
  bounded_additions: ($new | sort),
  probe_additions: ($probe - $old77 | sort),
  three_mode_union: $three,
  current_two_mode_union: $two,
  counts: {historical_three_mode: 77, bounded_additions: 19,
    additional_probes: 2, three_mode: 96, current_two_mode: 98,
    catalog: 165, remaining_without_current_two_mode_credit: 67},
  hashes: {retained_baseline: $retained_sha256, original35: $original35_sha256,
    catalog77: $catalog77_sha256, historical77: $historical77_sha256,
    bounded_report: $bounded_sha256, bounded_verification: $verified_sha256,
    probe_verification: $probes_sha256}
}
