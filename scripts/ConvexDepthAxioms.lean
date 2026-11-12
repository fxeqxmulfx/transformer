import Transformer.Grokking.FlatSpectrum.ConvexDepth

open Lean Elab Command

#print axioms Transformer.Grokking.FlatSpectrum.ConvexDepth.run_affine
#print axioms Transformer.Grokking.FlatSpectrum.ConvexDepth.budget_loss_convex
#print axioms Transformer.Grokking.FlatSpectrum.ConvexDepth.run_norm_le
#print axioms Transformer.Grokking.FlatSpectrum.ConvexDepth.run_difference_le
#print axioms Transformer.Grokking.FlatSpectrum.ConvexDepth.fourier4_unitary
#print axioms Transformer.Grokking.FlatSpectrum.ConvexDepth.unitary_singularValues
#print axioms Transformer.Grokking.FlatSpectrum.ConvexDepth.fourier_phase_transport_norm
#print axioms Transformer.Grokking.FlatSpectrum.ConvexDepth.twoBlockCE_not_convex
#print axioms Transformer.Grokking.FlatSpectrum.ConvexDepth.fourMixer_two_steps
#print axioms Transformer.Grokking.FlatSpectrum.ConvexDepth.pathRun_residual
#print axioms Transformer.Grokking.FlatSpectrum.ConvexDepth.pathLoss_convex
#print axioms Transformer.Grokking.FlatSpectrum.ConvexDepth.productWeights_mem
#print axioms Transformer.Grokking.FlatSpectrum.ConvexDepth.path_card
#print axioms Transformer.Grokking.FlatSpectrum.ConvexDepth.fourier_path_bound
#print axioms Transformer.Grokking.FlatSpectrum.ConvexDepth.residual_path_bound
#print axioms Transformer.Grokking.FlatSpectrum.ConvexDepth.correlatedWeights_not_product
#print axioms Transformer.Grokking.FlatSpectrum.ConvexDepth.correlated_logits
#print axioms Transformer.Grokking.FlatSpectrum.ConvexDepth.residual_interaction_margin
#print axioms Transformer.Grokking.FlatSpectrum.ConvexDepth.no_affine_interaction
#print axioms Transformer.Grokking.FlatSpectrum.ConvexDepth.headLoss_convex
#print axioms Transformer.Grokking.FlatSpectrum.ConvexDepth.headLogits_bound
#print axioms Transformer.Grokking.FlatSpectrum.ConvexDepth.liftHead_logits

run_cmd do
  let env ← getEnv
  let standard := [``propext, ``Classical.choice, ``Quot.sound]
  let mut count : Nat := 0
  for (name, _) in env.constants.toList do
    unless (`Transformer.Grokking.FlatSpectrum.ConvexDepth).isPrefixOf name &&
        !name.isInternal do continue
    count := count + 1
    for axiomName in ← liftCoreM (collectAxioms name) do
      unless standard.contains axiomName do
        throwError "{name} depends on forbidden axiom {axiomName}"
  logInfo m!"Convex depth: {count} declarations audited; no sorry or extra axioms"
