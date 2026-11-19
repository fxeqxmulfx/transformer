import Transformer.Grokking.Convex.ConvexChannel

/-!
# Axiom audit for the generated convex channel model

Run `LAKE_ARTIFACT_CACHE=false lake env lean scripts/ConvexChannelAxioms.lean`.
Every declaration and its transitive dependencies must use only standard
Lean axioms. This audit fails on `sorryAx` as well as any extra axiom.
-/

open Lean Elab Command

#print axioms Transformer.Grokking.Convex.ConvexChannel.convex_feasible
#print axioms Transformer.Grokking.Convex.ConvexChannel.answerLoss_convex
#print axioms Transformer.Grokking.Convex.ConvexChannel.generates_absent_value
#print axioms Transformer.Grokking.Convex.ConvexChannel.eq_generator_of_answerLoss_le
#print axioms Transformer.Grokking.Convex.ConvexChannel.trained_strict_argmax
#print axioms Transformer.Grokking.Convex.ConvexChannel.flat_error_bound
#print axioms Transformer.Grokking.Convex.ConvexChannel.flat_width_necessary
#print axioms Transformer.Grokking.Convex.ConvexChannel.flat_capacity_does_not_identify
#print axioms Transformer.Grokking.Convex.ConvexChannel.linear_oracle
#print axioms Transformer.Grokking.Convex.ConvexChannel.feasible_mixRows
#print axioms Transformer.Grokking.Convex.ConvexChannel.Forward.allows_scan
#print axioms Transformer.Grokking.Convex.ConvexChannel.Forward.learn_complete
#print axioms Transformer.Grokking.Convex.ConvexChannel.Forward.learn_failure_iff
#print axioms Transformer.Grokking.Convex.ConvexChannel.Forward.learned_isMinOn
#print axioms Transformer.Grokking.Convex.ConvexChannel.Forward.learned_eq_generator
#print axioms Transformer.Grokking.Convex.ConvexChannel.Forward.countedLearn_bound
#print axioms Transformer.Grokking.Convex.ConvexChannel.Forward.bool_unseen_generated
#print axioms Transformer.Grokking.Convex.ConvexChannel.Forward.bool_learned_flat
#print axioms Transformer.Grokking.Convex.ConvexChannel.Observation.unseen_logits_of_min
#print axioms Transformer.Grokking.Convex.ConvexChannel.Observation.unseen_logits_eq_targets
#print axioms Transformer.Grokking.Convex.ConvexChannel.Observation.logit_error_of_reconstruct
#print axioms Transformer.Grokking.Convex.ConvexChannel.Observation.every_optimizer_predicts_unseen

#print axioms Transformer.Grokking.Convex.ConvexChannel.Forward.learnBatches_eq_learn
#print axioms Transformer.Grokking.Convex.ConvexChannel.Forward.learnedBatches_eq_channel
#print axioms Transformer.Grokking.Convex.ConvexChannel.Forward.later_batch_changes_key
#print axioms Transformer.Grokking.Convex.ConvexChannel.Forward.Inference.feedback_fits
#print axioms Transformer.Grokking.Convex.ConvexChannel.Forward.Inference.finish_inferBatches
#print axioms Transformer.Grokking.Convex.ConvexChannel.Forward.Inference.certify_sound
#print axioms Transformer.Grokking.Convex.ConvexChannel.Forward.Inference.guarded_unchanged
#print axioms Transformer.Grokking.Convex.ConvexChannel.Forward.Inference.bool_self_label_blocks_feedback
#print axioms Transformer.Grokking.Convex.ConvexChannel.Forward.Inference.bool_guard_preserves_feedback
#print axioms Transformer.Grokking.Convex.ConvexChannel.Forward.Inference.no_unlabeled_answer_guarantee

run_cmd do
  let env ← getEnv
  let standard := [``propext, ``Classical.choice, ``Quot.sound]
  let mut count : Nat := 0
  for (name, _) in env.constants.toList do
    unless (`Transformer.Grokking.Convex.ConvexChannel).isPrefixOf name && !name.isInternal do
      continue
    count := count + 1
    let axioms ← liftCoreM (collectAxioms name)
    for axiomName in axioms do
      unless standard.contains axiomName do
        throwError "{name} depends on forbidden axiom {axiomName}"
  logInfo m!"Convex channels: {count} declarations audited; no sorry or extra axioms"
