import Transformer.Grokking.Convex.SpectralBall

open Lean Elab Command

#print axioms Transformer.Grokking.Convex.SpectralBall.feasible_convex
#print axioms Transformer.Grokking.Convex.SpectralBall.ceLoss_convex
#print axioms Transformer.Grokking.Convex.SpectralBall.missing_eq_one
#print axioms Transformer.Grokking.Convex.SpectralBall.eq_signKernel_of_loss_le
#print axioms Transformer.Grokking.Convex.SpectralBall.ambiguous_flat_minima

run_cmd do
  let env ← getEnv
  let standard := [``propext, ``Classical.choice, ``Quot.sound]
  let mut count : Nat := 0
  for (name, _) in env.constants.toList do
    unless (`Transformer.Grokking.Convex.SpectralBall).isPrefixOf name && !name.isInternal do
      continue
    count := count + 1
    for axiomName in ← liftCoreM (collectAxioms name) do
      unless standard.contains axiomName do
        throwError "{name} depends on forbidden axiom {axiomName}"
  logInfo m!"Spectral ball: {count} declarations audited; no sorry or extra axioms"
