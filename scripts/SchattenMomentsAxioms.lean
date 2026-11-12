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
