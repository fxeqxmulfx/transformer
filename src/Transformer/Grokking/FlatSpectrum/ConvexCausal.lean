module

public import Transformer.Grokking.FlatSpectrum.ConvexCausal.Limits
public import Transformer.Grokking.FlatSpectrum.ConvexCausal.Compatibility
public import Transformer.Grokking.FlatSpectrum.ConvexCausal.Readout

/-!
# A calibrated convex causal model for finite-token stress tasks

Source: `review/convex_causal.md`, CCa1–CCa10, derived against the complete
165-case `diff-gpt/stress_tests_for_gpt.md` catalog. Joint probability coordinates
and a fixed floor yield convex ordinary CE via log-probability logits. Complete
positional interactions support every bounded-prefix conditional distribution,
with explicit exponential cost. A finite structured rule bank admits computed
CE minimizers and exact unseen/free-generation recovery under observable input
separation and realizability. Every optimum on that bank's full convex mixture
domain has the same recovered rule. Calibration and numeric readouts preserve
stochastic and decoded-output distinctions. Universal capacity does not imply
that a proper training split identifies arbitrary unseen labels. No original
benchmark pass, efficient universal solver, or automatic bank discovery is claimed.
-/
