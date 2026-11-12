import Transformer.Grokking.FlatSpectrum.CoupledMoments

open Lean Elab Command

#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.feasible_convex
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.fit_minimum
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.nonzero_fourier_moments_injective
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.independent_readings_not_joint
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.feasible_blend
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.relabelHull_invariant
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.crossEntropy_relabel
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.swappedWitness_not_feasible
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.relabelHull_strict_witness

section CodingDiscovery
open Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery

#print axioms rectangleCodes_congr
#print axioms rectangleCodes_comp_surjective
#print axioms affine_of_two_rows
#print axioms exists_gauge
#print axioms bijective_of_mem_pinnedCodes
#print axioms exists_phases_of_mem_pinnedCodes
#print axioms finrank_pinnedCodes_eq_one_iff
#print axioms finrank_pinnedCodes_eq_one_of_two_rows
#print axioms finrank_pinnedCodes_eq_one_iff_rank
#print axioms completion_eq_target
#print axioms completes_iff
#print axioms discover_eq_accept
#print axioms discover_realizable
#print axioms discover_eq_accept_iff
#print axioms discover_congr
#print axioms discovered_minimum
#print axioms discovered_argmin
#print axioms pinned_argmin
#print axioms discovered_prediction
#print axioms coding_discovery
#print axioms five_coding_discovery
#print axioms five_heldout_card
#print axioms five_names_not_code
#print axioms discover_eq_ambiguous_of_two_tables
#print axioms cross_ambiguous
#print axioms ten_disconnected
#print axioms mem_rectangleCodes_iff_anchored
#print axioms finrank_pinnedCodes_ne_one_of_unobserved
#print axioms discover_eq_ambiguous_of_unobserved

end CodingDiscovery

section Paired
open Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten

#print axioms budget_pairLatent
#print axioms pairedFeasible_convex
#print axioms pairedFeasible_iff
#print axioms pairedPhase_budget
#print axioms paired_fit_minimum
#print axioms pairedPhase_minimum
#print axioms pairedPhase_hull_minimum
#print axioms foldDecoder_fiber
#print axioms folded_paired_minimum
#print axioms folded_new_capacity
#print axioms paired_unseen_ambiguity

end Paired

run_cmd do
  let env ← getEnv
  let standard := [``propext, ``Classical.choice, ``Quot.sound]
  let mut count : Nat := 0
  for (name, _) in env.constants.toList do
    unless (`Transformer.Grokking.FlatSpectrum.CoupledMoments).isPrefixOf name && !name.isInternal do
      continue
    count := count + 1
    for axiomName in ← liftCoreM (collectAxioms name) do
      unless standard.contains axiomName do
        throwError "{name} depends on forbidden axiom {axiomName}"
  logInfo m!"Coupled moments: {count} declarations audited; no sorry or extra axioms"
