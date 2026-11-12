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
