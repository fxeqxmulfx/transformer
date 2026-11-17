if length != 24 or
   ([.[] | [.input.kind,.input.geometry,.input.variant,.input.mode]] | unique | length) != 24 or
   any(.[]; .updates != 0 or .intact_raw_replay != "identical" or
       .original_success != false or .raw_and_exact_99 != false) then
  error("expected all24 unchanged original binding diagnostic inputs")
else
  {
    scope: "descriptive means across four correlated starting-variant/mode cells; one original task and seed",
    cells: (sort_by([.input.kind,.input.geometry,.input.variant,.input.mode]) | map({input, splits})),
    groups: (group_by([.input.kind,.input.geometry]) | map(
      . as $rows | {
        kind: .[0].input.kind, geometry: .[0].input.geometry,
        splits: (reduce (.[0].splits | keys[]) as $s ({};
          .[$s] = ([$rows[] | .splits[$s].overall] | . as $r | {
            original_accuracy: (map(.original_accuracy) | add / length),
            stored_value_prediction_rate: (map(.stored_value_prediction_rate) | add / length),
            two_value_choice_accuracy: (map(.two_value_choice_accuracy) | add / length),
            mean_full_ce: (map(.mean_full_ce) | add / length),
            mean_mass_ce: (map(.mean_mass_ce) | add / length),
            mean_conditional_pair_ce: (map(.mean_conditional_pair_ce) | add / length),
            mean_stored_probability_mass: (map(.mean_stored_probability_mass) | add / length),
            mean_correct_minus_other_logit: (map(.mean_correct_minus_other_logit) | add / length),
            maximum_ce_decomposition_error: (map(.maximum_ce_decomposition_error) | max)
          })))
      }))
  }
end
