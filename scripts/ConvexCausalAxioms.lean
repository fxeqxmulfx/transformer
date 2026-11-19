import Transformer.Grokking.Convex.ConvexCausal

/-! Audit every public declaration of the Lean-only causal extension, including examples. -/

open Lean Elab Command

#print axioms Transformer.Grokking.Convex.ConvexCausal.normalized_logits
#print axioms Transformer.Grokking.Convex.ConvexCausal.loss_convex
#print axioms Transformer.Grokking.Convex.ConvexCausal.universal_rows
#print axioms Transformer.Grokking.Convex.ConvexCausal.encode_injective
#print axioms Transformer.Grokking.Convex.ConvexCausal.context_card
#print axioms Transformer.Grokking.Convex.ConvexCausal.trained_minimum
#print axioms Transformer.Grokking.Convex.ConvexCausal.optimizer_vertex
#print axioms Transformer.Grokking.Convex.ConvexCausal.learns_causal
#print axioms Transformer.Grokking.Convex.ConvexCausal.expected_minimum
#print axioms Transformer.Grokking.Convex.ConvexCausal.ideal_tendsto_zero
#print axioms Transformer.Grokking.Convex.ConvexCausal.spectral_loss_convex
#print axioms Transformer.Grokking.Convex.ConvexCausal.risk_decomposition
#print axioms Transformer.Grokking.Convex.ConvexCausal.universal_not_separates
#print axioms Transformer.Grokking.Convex.ConvexCausal.uniform_accuracy

run_cmd do
  let env ← getEnv
  let standard := [``propext, ``Classical.choice, ``Quot.sound]
  let mut count : Nat := 0
  for (name, _) in env.constants.toList do
    unless (`Transformer.Grokking.Convex.ConvexCausal).isPrefixOf name &&
        !name.isInternal do continue
    count := count + 1
    for axiomName in ← liftCoreM (collectAxioms name) do
      unless standard.contains axiomName do
        throwError "{name} depends on forbidden axiom {axiomName}"
  logInfo m!"Convex causal: {count} declarations audited; no sorry or extra axioms"
