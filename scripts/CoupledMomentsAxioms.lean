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

section DecodedRecovery
open Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery

#print axioms pairedDecoder_fold
#print axioms step_unique
#print axioms anchor_rigid
#print axioms pinned_rigid
#print axioms accepted_decoding
#print axioms fits_agree
#print axioms Anchor.accepts
#print axioms Reached.pinned
#print axioms covers_of_labels
#print axioms decoding_of_labels
#print axioms recover_sound
#print axioms recover_accept
#print axioms recover_of_labels
#print axioms recover_uncertified
#print axioms recover_unsupported
#print axioms recover_inconsistent
#print axioms phaseHull_subset_decoded
#print axioms phaseHull_subset_paired
#print axioms hull_defect_le
#print axioms hull_fit_decoding
#print axioms hull_minimizer_decoding
#print axioms certified_minimum
#print axioms certifiedFeasible_convex
#print axioms certified_restriction
#print axioms budget_iff_nuclear
#print axioms wrong_nuclear
#print axioms budget_wrong_minimum
#print axioms wrong_not_in_hull
#print axioms held_recovery
#print axioms held_cross_entropy
#print axioms threeCells_no_certificate
#print axioms paired_ambiguity_refused
#print axioms corner_unsupported
#print axioms conflict_inconsistent

end DecodedRecovery

section DecodedPropagation
open Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedPropagation

#print axioms step_preserves
#print axioms step_subset
#print axioms run_preserves
#print axioms run_subset
#print axioms live_of_contains
#print axioms normalize_sum
#print axioms normalize_fits
#print axioms initial_contains
#print axioms mem_outputs
#print axioms mem_answers
#print axioms identifies
#print axioms empty_refutes
#print axioms not_singleton_of_disagree
#print axioms mem_bankAnswers
#print axioms bank_identifies
#print axioms empty_bank_refutes
#print axioms bank_not_singleton_of_disagree
#print axioms certifiedFeasible_convex
#print axioms certified_loss_convex
#print axioms certified_witness
#print axioms certified_optimum
#print axioms certified_restriction
#print axioms wide_fits
#print axioms wide_fiber
#print axioms wide_not_paired
#print axioms wide_answer
#print axioms wide_recovery
#print axioms wide_latent_undetermined
#print axioms wide_bank_fits
#print axioms wide_shifted_fits
#print axioms wide_bank_answer
#print axioms ambiguity_fit_false
#print axioms ambiguity_fit_true
#print axioms wide_ambiguity_refused
#print axioms bank_ambiguity_fit_false
#print axioms bank_ambiguity_fit_true
#print axioms bank_ambiguity_refused
#print axioms conflict_empty
#print axioms conflict_bank_empty
#print axioms live_has_selection
#print axioms branch_mem_answers
#print axioms phase_endpoint
#print axioms branch_endpoint
#print axioms live_endpoint
#print axioms certified_rows_blend
#print axioms wide_partial_endpoint

end DecodedPropagation

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
