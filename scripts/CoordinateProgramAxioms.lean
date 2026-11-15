import Transformer.Grokking.FlatSpectrum.ConvexDepth.CoordinatePrograms

/-! Direct CM57 audit. Run with
`LAKE_ARTIFACT_CACHE=false lake env lean -j 2 scripts/CoordinateProgramAxioms.lean`.
Checks the joint domain, actual complex products, full budgets, ordinary CE,
conditional identification and active/inactive/uncovered witnesses.
-/

open Transformer.Grokking.FlatSpectrum.ConvexDepth.CoordinatePrograms

#print axioms costs_pos
#print axioms center_combination
#print axioms charge_combination
#print axioms zero_mass
#print axioms domain_convex
#print axioms dot_combination
#print axioms augmented_anchors
#print axioms augmented_realizable
#print axioms query_identified
#print axioms score_difference
#print axioms all_centers_identified
#print axioms mixture_strict
#print axioms source_bound
#print axioms local_source_logits
#print axioms row_score
#print axioms local_expansion_bound
#print axioms bank_source_bound
#print axioms features_bound
#print axioms head_block
#print axioms bank_source_logits
#print axioms head_logits
#print axioms expansion_bound
#print axioms head_budget
#print axioms loss_convex
#print axioms batchLoss_convex
#print axioms domain_loss_convex
#print axioms domain_batchLoss_convex
#print axioms identified_prediction
#print axioms identified_gap
#print axioms example_feasible
#print axioms example_realizable
#print axioms example_span
#print axioms rejected_program
#print axioms example_query_coverage
#print axioms example_unseen_prediction
#print axioms example_copy_prediction
#print axioms hole_not_covered
#print axioms opposite_completions_feasible
#print axioms opposite_raw_winners
#print axioms alternative_feasible
