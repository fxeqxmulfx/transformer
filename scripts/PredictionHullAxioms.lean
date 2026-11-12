import Transformer.Grokking.FlatSpectrum.CoupledMoments.PredictionHull

open Lean Elab Command

#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.scaled_target_le
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.scaled_ideal_le
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.predictionHull_fit_minimum
#print axioms Transformer.Grokking.FlatSpectrum.CoupledMoments.predictionHull_excess_le

run_cmd do
  let env ← getEnv
  let standard := [``propext, ``Classical.choice, ``Quot.sound]
  let mut count : Nat := 0
  for (name, _) in env.constants.toList do
    unless (`Transformer.Grokking.FlatSpectrum.CoupledMoments).isPrefixOf name &&
        !name.isInternal do continue
    count := count + 1
    for axiomName in ← liftCoreM (collectAxioms name) do
      unless standard.contains axiomName do
        throwError "{name} depends on forbidden axiom {axiomName}"
  logInfo m!"Coupled predictions: {count} declarations audited; no sorry or extra axioms"
