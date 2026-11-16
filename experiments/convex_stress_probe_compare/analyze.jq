# Read-only analysis of the independently verified, complete original cohort.
if .producer != "1c67b444ce763c9b7674c2ef4b15c12b68583e1e"
   or .complete_inventory_verified != true
   or .identical_all_artifacts != true
   or .required_modes != ["Forward", "Mini"]
   or .seeds != [1] or (.cases|length) != 6
then error("expected the complete frozen two-mode verification") else . end
| [.cases[] | select(.source_refusal == null and .model_refusal == null)
   | select((.modes|length) == 2)
   | select(all(.modes[]; all(.splits[] | select(.split != "train");
         .score.raw99 and .score.exact99))) | .name] as $passing
| {
    producer, seeds, artifacts: (.artifacts|length),
    passing_cases: $passing,
    source_refusals: [.cases[] | select(.source_refusal != null) | {name,reason:.source_refusal}],
    modes: ([.cases[].modes[] | {mode,rows:[.splits[]|select(.split!="train")|.score]}]
      | group_by(.mode) | map({
        mode: .[0].mode,
        heldout_correct: ([.[].rows[].raw.correct]|add),
        heldout_targets: ([.[].rows[].raw.target_count]|add),
        heldout_exact: ([.[].rows[].raw.exact]|add),
        heldout_examples: ([.[].rows[].raw.example_count]|add),
        maximum_ce: ([.[].rows[].cross_entropy]|max),
        weighted_ce: (([.[].rows[]|.cross_entropy*.teacher_forced.target_count]|add)
          / ([.[].rows[].teacher_forced.target_count]|add)),
        maximum_numeric_mse: ([.[].rows[].mse|select(.!=null)]|max),
        all_original_success: (all(.[].rows[];.original_success))
      })),
    new_case_names: ([$passing[]|select(.=="gpt-probe-conditional-amplitudes"
      or .=="gpt-probe-order-discrimination")]),
    raw_replayed_calls_per_run: ([.cases[].raw_replayed_calls]|add),
    counts_retain_original_overlapping_subsets: true,
    minibatch_zero_ce_is_float64_rounding: true,
    unified_catalog_or_additional_seeds_measured: false
  }
