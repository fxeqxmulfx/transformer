import Transformer.GPTMini.Sparsemax

/-!
# Reproducible statement, definition, and transitive-axiom inspection

Run with `lake env lean` after building `Transformer.GPTMini.Sparsemax`.
The local paper context is arXiv:2211.11052v1, §§3.1 and 4; the new
saturation results and prime-193 finite experiment are explicitly extensions.
This inspection complements, and does not replace, `scripts/Axioms.lean`.
-/

set_option pp.universes true
set_option pp.explicit true

#print HasFDerivAt
#print HasFDerivAtFilter
#print hasFDerivAt_const
#print hasFDerivAtFilter_const
#print HasFDerivAt.congr_of_eventuallyEq
#print Filter.EventuallyEq.hasFDerivAt_iff
#print Filter.EventuallyEq.hasFDerivAtFilter_iff
#print ContinuousAt
#print continuous_apply
#print Continuous.continuousAt
#print ContinuousAt.add
#print continuousAt_const
#print ContinuousAt.eventually_lt
#print Filter.Tendsto.eventually_lt
#print Filter.eventually_all
#print Filter.Eventually.of_forall
#print Filter.Eventually.mono
#print le_of_lt
#print congrArg
#print Fin.sum_univ_two
#print List.length
#print List.foldl
#print List.countP
#print List.countP.go
#print List.all
#print Nat.div
#print Nat.mod

#print axioms hasFDerivAt_const
#print axioms hasFDerivAtFilter_const
#print axioms HasFDerivAt.congr_of_eventuallyEq
#print axioms Filter.EventuallyEq.hasFDerivAt_iff
#print axioms Filter.EventuallyEq.hasFDerivAtFilter_iff
#print axioms continuous_apply
#print axioms Continuous.continuousAt
#print axioms ContinuousAt.add
#print axioms continuousAt_const
#print axioms ContinuousAt.eventually_lt
#print axioms Filter.Tendsto.eventually_lt
#print axioms Filter.eventually_all
#print axioms Filter.Eventually.of_forall
#print axioms Filter.Eventually.mono
#print axioms le_of_lt
#print axioms congrArg
#print axioms Fin.sum_univ_two
#print axioms List.length
#print axioms List.foldl
#print axioms List.countP
#print axioms List.countP.go
#print axioms List.all
#print axioms Nat.div
#print axioms Nat.mod

#print Transformer.ConvexRecall.simplexOn
#print Transformer.ConvexRecall.routingObjective
#print Transformer.GPTMini.Convex.sparseWeights
#print Transformer.GPTMini.Convex.sparseWeights_spec
#print Transformer.ConvexRecall.routing_minimizer_unique
#print Transformer.ConvexRecall.basis_mem_simplex
#print axioms Transformer.GPTMini.Convex.sparseWeights_spec
#print axioms Transformer.ConvexRecall.routing_minimizer_unique
#print axioms Transformer.ConvexRecall.basis_mem_simplex

#print axioms Transformer.GPTMini.Sparsemax.sparseWeights_eq_basis_of_gap
#print axioms Transformer.GPTMini.Sparsemax.sparseWeights_eventually_eq_basis
#print axioms Transformer.GPTMini.Sparsemax.sparseWeights_hasFDerivAt_zero
#print axioms Transformer.GPTMini.Sparsemax.rowLoss_hasFDerivAt_zero
#print axioms Transformer.GPTMini.Sparsemax.separatedScores_gap
#print axioms Transformer.GPTMini.Sparsemax.wrong_route_positive_stationary_point
#print axioms Transformer.GPTMini.Sparsemax.Certificate.train_certificate
#print axioms Transformer.GPTMini.Sparsemax.Certificate.heldout_certificate
#print axioms Transformer.GPTMini.Sparsemax.Certificate.train_accuracy_exact
#print axioms Transformer.GPTMini.Sparsemax.Certificate.heldout_accuracy_exact
#print axioms Transformer.GPTMini.Sparsemax.Certificate.heldout_percentage_rounding_interval
#print axioms Transformer.GPTMini.Sparsemax.Certificate.heldout_accuracy_below_target
