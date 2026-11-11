import Transformer.Grokking.FlatSpectrum.ConvexRecall

/-!
# Axiom audit for convex answer-loss recall

Run `LAKE_ARTIFACT_CACHE=false lake env lean scripts/ConvexRecallAxioms.lean`.
Audit every declaration in the construction, including transitive proof
dependencies. Fail on any axiom other than the three standard Lean axioms.
-/

open Lean Elab Command

#print axioms Transformer.Grokking.FlatSpectrum.ConvexRecall.answerLoss_convex
#print axioms Transformer.Grokking.FlatSpectrum.ConvexRecall.eq_one_of_answerLoss_le
#print axioms Transformer.Grokking.FlatSpectrum.ConvexRecall.trained_solves_mqar
#print axioms Transformer.Grokking.FlatSpectrum.ConvexRecall.trained_isFlat
#print axioms Transformer.Grokking.FlatSpectrum.ConvexRecall.trained_not_nuclear_minimum
#print axioms Transformer.Grokking.FlatSpectrum.ConvexRecall.wrong_global_minimum_without_separation
#print axioms Transformer.Grokking.FlatSpectrum.ConvexRecall.tendsto_one_of_loss_tendsto
#print axioms Transformer.Grokking.FlatSpectrum.ConvexRecall.eventually_all_strict_argmax

run_cmd do
  let env ← getEnv
  let standard := [``propext, ``Classical.choice, ``Quot.sound]
  let mut count : Nat := 0
  for (name, _) in env.constants.toList do
    unless (`Transformer.Grokking.FlatSpectrum.ConvexRecall).isPrefixOf name && !name.isInternal do
      continue
    count := count + 1
    let axioms ← liftCoreM (collectAxioms name)
    for axiomName in axioms do
      unless standard.contains axiomName do
        throwError "{name} depends on forbidden axiom {axiomName}"
  logInfo m!"Convex recall: {count} declarations audited; no sorry or extra axioms"
