import Transformer.Grokking.Convex.ConvexDepth.ClassGeometry

/-! Direct CM56 audit: identification from observed anchors, actual complete
complex products, full/minibatch CE, and a nonconstant unseen-class witness.
Run with `LAKE_ARTIFACT_CACHE=false lake env lean scripts/UnseenClassGeometryAxioms.lean`.
-/

open Transformer.Grokking.Convex.ConvexDepth.ClassGeometry

#print axioms fitted_anchors
#print axioms anchors_identify
#print axioms score_difference
#print axioms strict_score
#print axioms fitted_prediction
#print axioms domain_convex
#print axioms relabeling
#print axioms features_bound
#print axioms expansion_bound
#print axioms source_logits
#print axioms head_logits
#print axioms head_budget
#print axioms loss_convex
#print axioms batchLoss_convex
#print axioms identified_prediction
#print axioms domain_loss_convex
#print axioms domain_batchLoss_convex
#print axioms example_realizable
#print axioms distinct_feasible
#print axioms missing_prior_ambiguity
#print axioms example_unseen_prediction
