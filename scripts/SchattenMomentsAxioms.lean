import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten

open Lean Elab Command

#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.budget_convex
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.budget_at_card_iff
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.budget_blend
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.rankOne_missing_of_loss_le
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.hull_fit_minimum
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.intermediate_excluded
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.intermediate_admitted
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.intermediate_excluded_hull
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.masked_optimal_but_uncertain
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.paired_recovery
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.quotient_fit_minimum
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.quotient_observed_group
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.certified_decoded_blend
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.graph_nuclear_budget_recovers
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.constant_moment_nuclear
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.graph_probability_recovery
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.phaseCode_gram
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.phaseCandidate_budget
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.phaseCandidate_minimum
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.phaseCandidate_unique
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.rows_decode
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.decodedFeasible_convex
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.decoded_loss_convex
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.decoded_fit_minimum
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.decodedPhase_minimum
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.corner_no_injective_phase
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.corner_direct_excluded
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.corner_decoded_admitted
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.corner_decoded_ambiguity
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.corner_decoded_graph_coverage
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.decoderHull_convex
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.decoderHull_rows
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.decoderHull_fit_minimum
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.decodedPhase_hull_minimum
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.corner_decoderHull_ambiguity

run_cmd do
  let env ← getEnv
  let standard := [``propext, ``Classical.choice, ``Quot.sound]
  let mut count : Nat := 0
  for (name, _) in env.constants.toList do
    unless (`Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten).isPrefixOf name &&
        !name.isInternal do continue
    count := count + 1
    for axiomName in ← liftCoreM (collectAxioms name) do
      unless standard.contains axiomName do
        throwError "{name} depends on forbidden axiom {axiomName}"
  logInfo m!"Schatten moments: {count} declarations audited; no sorry or extra axioms"
