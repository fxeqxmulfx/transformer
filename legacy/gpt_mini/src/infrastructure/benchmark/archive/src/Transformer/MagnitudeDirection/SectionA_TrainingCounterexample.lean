/-
# Original MD need not inherit training convergence

User-requested extension of arXiv:2606.25971v2, §3.1 and Appendix A,
Algorithm 2. On the smooth strongly convex loss `(w+1)^2/2`, normalized
positive scalar directions with relative LR `1/4` and positive softplus
gains stay positive forever. Their true gradient is always greater than
one. This refutes an unconditional optimizer-agnostic training guarantee,
not a numbered theorem in the paper, which gives no such theorem.
The example allows any stateful gain optimizers, including Adam/AMSGrad.
-/

import Transformer.MagnitudeDirection.SectionA_TrainingCounterexampleStep

open scoped Topology
open Filter

noncomputable section

namespace Transformer.MagnitudeDirection

open Optimization

variable {R C : Type*}

/-- Actual normalized-direction MD training on the opposite quadratic,
with both LRs positive and all gradients recomputed from the current loss.
Source: arXiv:2606.25971v2, Appendix A, training counterexample. -/
def counterexampleRun (rowOpt : StatefulGainStep R 1) (colOpt : StatefulGainStep C 1)
    (memory : Unit × R × C) : ℕ → (Unit × R × C) × FullState 1 1 :=
  fullTrainingRun oppositeQuadratic (normalizedMDProposal rowOpt colOpt (1 / 4) (1 / 4) 1)
    memory (unitGainState 1)

/-- Every finite original iterate stays in the positive sphere component,
regardless of the stateful gain-optimizer outputs. The true current gradient
is used at every step. Source: arXiv:2606.25971v2, Appendix A, training counterexample. -/
theorem originalMD_positive_component (rowOpt : StatefulGainStep R 1)
    (colOpt : StatefulGainStep C 1) (memory : Unit × R × C) (t : ℕ) :
    0 < (counterexampleRun rowOpt colOpt memory t).2.weight 0 0 ∧
      fullDirection softplus (counterexampleRun rowOpt colOpt memory t).2 = 1 := by
  induction t with
  | zero =>
    exact ⟨by norm_num [counterexampleRun, fullTrainingRun, unitGainState],
      unitGainState_direction _⟩
  | succ t ih =>
    let s := counterexampleRun rowOpt colOpt memory t
    have hG : 0 < toMatrix (gradient oppositeQuadratic (fromMatrix s.2.weight)) 0 0 := by
      rw [oppositeQuadratic_gradient, toMatrix_fromMatrix]
      linarith [ih.1]
    exact normalizedMDProposal_positive rowOpt colOpt (1 / 4) (1 / 4) s.1 s.2
      (toMatrix (gradient oppositeQuadratic (fromMatrix s.2.weight))) ih.1 hG (by norm_num)

/-- The norm of the actual full objective gradient is uniformly at least
one, not merely a factor-coordinate gradient or arbitrary signal.
Source: arXiv:2606.25971v2, Appendix A, training counterexample. -/
theorem originalMD_gradient_lower_bound (rowOpt : StatefulGainStep R 1)
    (colOpt : StatefulGainStep C 1) (memory : Unit × R × C) (t : ℕ) :
    1 ≤ ‖gradient oppositeQuadratic
      (fromMatrix (counterexampleRun rowOpt colOpt memory t).2.weight)‖ := by
  exact oppositeQuadratic_gradient_lower_bound _
    (originalMD_positive_component rowOpt colOpt memory t).1

/-- Refutation of unconditional inheritance of base-optimizer convergence
through MD. The objective is smooth, strongly convex and lower bounded;
every finite direction remains on its unit sphere, both fixed LRs are
positive and small, yet true full gradients do not tend to zero.
This is an allowed normalized-gradient base callback with arbitrary
stateful gain callbacks, not a claim about a particular experimental run.
Source: arXiv:2606.25971v2, §3.1 and Appendix A, training counterexample. -/
theorem originalMD_unconditional_training_counterexample
    (rowOpt : StatefulGainStep R 1) (colOpt : StatefulGainStep C 1)
    (memory : Unit × R × C) :
    SmoothObjective oppositeQuadratic 1 ∧ StrongLowerModel oppositeQuadratic 1 ∧
      (∀ x, 0 ≤ oppositeQuadratic x) ∧
      (∀ t : ℕ, frobeniusNorm
        (fullDirection softplus (counterexampleRun rowOpt colOpt memory t).2) = 1) ∧
      ¬ Tendsto (fun t : ℕ => ‖gradient oppositeQuadratic
        (fromMatrix (counterexampleRun rowOpt colOpt memory t).2.weight)‖) atTop (𝓝 0) := by
  refine ⟨oppositeQuadratic_models.1, oppositeQuadratic_models.2.1,
    oppositeQuadratic_models.2.2, ?_, ?_⟩
  · intro t
    rw [(originalMD_positive_component rowOpt colOpt memory t).2, single_frobeniusNorm]
    norm_num
  · intro h
    have hc : (1 : ℝ) ≤ 0 := ge_of_tendsto h
      (Eventually.of_forall (originalMD_gradient_lower_bound rowOpt colOpt memory))
    norm_num at hc

/-- The fused-step correction repairs this same counterexample: its
actual weights converge to the negative-unit global minimizer despite
the original normalized proposal's component obstruction. Base/gain
optimizer computations and all gain memories are retained.
Source: training correction to arXiv:2606.25971v2, Appendix A, Algorithm 2. -/
theorem safeguardedMD_repairs_component (rowOpt : StatefulGainStep R 1)
    (colOpt : StatefulGainStep C 1) (memory : Unit × R × C) :
    Tendsto (fun t : ℕ =>
      (safeguardedFullRun (1 / 4) 1 1 oppositeQuadratic
        (normalizedMDProposal rowOpt colOpt (1 / 4) (1 / 4) 1) memory (unitGainState 1) t).2)
      atTop (𝓝 (-(fromMatrix (1 : Matrix (Fin 1) (Fin 1) ℝ)))) := by
  have hstar : gradient oppositeQuadratic
      (-(fromMatrix (1 : Matrix (Fin 1) (Fin 1) ℝ))) = 0 := by
    rw [oppositeQuadratic, trainingQuadratic_gradient]
    exact sub_self _
  exact (safeguardedMD_minimum_convergence (1 / 4) 1 1 1 oppositeQuadratic
    (normalizedMDProposal rowOpt colOpt (1 / 4) (1 / 4) 1) memory (unitGainState 1) _
    oppositeQuadratic_models.1 oppositeQuadratic_models.2.1 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) hstar).2.1

end Transformer.MagnitudeDirection
