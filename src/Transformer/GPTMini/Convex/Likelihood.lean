import Transformer.GPTMini.Convex.Likelihood.QueryKeyBoundary

/-!
# Compact nonlinear likelihood controls for the joint attention search

New constructions derived from Boyd and Vandenberghe (2004), §3.1.5,
§3.2.2 and §3.5. Sequential Bernoulli probabilities give genuinely nonlinear
logit contrasts with jointly convex ordinary categorical cross entropy.
Strict positivity and normalization also force convex prediction fibers
when every category loss is convex. Even nonconstant families fail this test.

The scalar learned Q/K product restores nonconvexity in the explicit control.
These are proved mathematical search criteria, not a compact drop-in
embedding/attention implementation, a trained Basis result, or a guarantee
for nonlinear downstream FFN and tied readout training.
-/
