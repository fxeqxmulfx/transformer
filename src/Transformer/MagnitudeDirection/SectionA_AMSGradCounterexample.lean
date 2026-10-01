/-
# Original concrete AMSGradMD can fail on a strongly convex loss

arXiv:2606.25971v2, §3.1 and Appendix A, Algorithm 2, with the specified
AMSGrad direction and gain callbacks from arXiv:1904.03590v4, Algorithm 1
and §6. Epsilon is one, both moment coefficients are zero, both LRs are
one quarter and all histories start at zero. Every printed moment,
maximum, projection, raw gain and genuine objective gradient is retained.
The loss is `(w+1)^2/2`; the unguarded weights stay positive and their
true full gradients cannot tend to zero. The corrected algorithm converges.
-/

import Transformer.MagnitudeDirection.SectionA_AMSGradCounterexampleStep
import Transformer.MagnitudeDirection.SectionA_AMSGradConvergence

open scoped Topology
open Filter

noncomputable section

namespace Transformer.MagnitudeDirection

open Optimization

/-- Original, unguarded concrete AMSGradMD training with genuine loss
gradients and zero initial histories. Source: arXiv:2606.25971v2,
Appendix A, AMSGradMD training extension; arXiv:1904.03590v4, Algorithm 1. -/
def originalAMSGradMDRun {m n : ℕ} (c ε β β₂ etaW etaG : ℝ)
    (f : MatrixSpace m n → ℝ) (initial : FullState m n) :
    ℕ → AMSGradMDMemory m n × FullState m n :=
  fullTrainingRun f (amsgradMDProposal ε β β₂ etaW etaG c)
    (zeroAMSGradMDMemory m n) initial

/-- A concrete original AMSGradMD counterexample with strictly positive
epsilon/LRs and admissible moment coefficients, on a nonconstant smooth
strongly convex loss. Source: arXiv:2606.25971v2, Appendix A, AMSGradMD
counterexample; arXiv:1904.03590v4, Algorithm 1 and §6. -/
def amsgradMDCounterexampleRun : ℕ → AMSGradMDMemory 1 1 × FullState 1 1 :=
  originalAMSGradMDRun 1 1 0 0 (1 / 4) (1 / 4) oppositeQuadratic (unitGainState 1)

/-- Actual AMSGradMD weights remain positive and the actual recovered
direction is one at every training time, despite all gain and moment
updates being performed. Source: arXiv:2606.25971v2, Appendix A,
AMSGradMD counterexample; arXiv:1904.03590v4, Algorithm 1 and §6. -/
theorem originalAMSGradMD_positive_component (t : ℕ) :
    0 < (amsgradMDCounterexampleRun t).2.weight 0 0 ∧
      fullDirection softplus (amsgradMDCounterexampleRun t).2 = 1 := by
  induction t with
  | zero =>
    exact ⟨by norm_num [amsgradMDCounterexampleRun, originalAMSGradMDRun, fullTrainingRun,
      unitGainState], unitGainState_direction _⟩
  | succ t ih =>
    let s := amsgradMDCounterexampleRun t
    have hG : 0 < toMatrix (gradient oppositeQuadratic (fromMatrix s.2.weight)) 0 0 := by
      rw [oppositeQuadratic_gradient, toMatrix_fromMatrix]
      linarith [ih.1]
    exact amsgradMDProposal_positive s.1 s.2
      (toMatrix (gradient oppositeQuadratic (fromMatrix s.2.weight))) ih.2 hG

/-- The genuine full objective gradient is uniformly bounded away from
zero along the original concrete AMSGradMD training sequence.
Source: arXiv:2606.25971v2, Appendix A, AMSGradMD counterexample;
arXiv:1904.03590v4, Algorithm 1 and §6. -/
theorem originalAMSGradMD_gradient_lower_bound (t : ℕ) :
    1 ≤ ‖gradient oppositeQuadratic (fromMatrix (amsgradMDCounterexampleRun t).2.weight)‖ :=
  oppositeQuadratic_gradient_lower_bound _ (originalAMSGradMD_positive_component t).1

/-- Refutation of automatic transfer of ordinary AMSGrad stationarity
to original AMSGradMD: the actual concrete algorithm has a smooth strongly
convex lower-bounded loss and valid sphere storage, yet its true full
gradients do not tend to zero. This is an AMSGradMD variant, not a claim
that the MD paper's Adam-gain experiments use these hyperparameters.
Source: arXiv:2606.25971v2, Appendix A, AMSGradMD counterexample;
arXiv:1904.03590v4, Algorithm 1 and §6. -/
theorem originalAMSGradMD_training_counterexample :
    SmoothObjective oppositeQuadratic 1 ∧ StrongLowerModel oppositeQuadratic 1 ∧
      (∀ x, 0 ≤ oppositeQuadratic x) ∧
      (∀ t : ℕ, frobeniusNorm (fullDirection softplus (amsgradMDCounterexampleRun t).2) = 1) ∧
      ¬ Tendsto (fun t : ℕ => ‖gradient oppositeQuadratic
        (fromMatrix (amsgradMDCounterexampleRun t).2.weight)‖) atTop (𝓝 0) := by
  refine ⟨oppositeQuadratic_models.1, oppositeQuadratic_models.2.1,
    oppositeQuadratic_models.2.2, ?_, ?_⟩
  · intro t
    rw [(originalAMSGradMD_positive_component t).2, single_frobeniusNorm]
    norm_num
  · intro h
    have hc : (1 : ℝ) ≤ 0 := ge_of_tendsto h
      (Eventually.of_forall originalAMSGradMD_gradient_lower_bound)
    norm_num at hc

/-- The corrected concrete AMSGradMD converges to the actual negative-unit
global minimum on this same example. Every moment/max history is still
updated; correction acts on the complete fused step and its representation.
Source: arXiv:2606.25971v2, Appendix A, AMSGradMD training correction;
arXiv:1904.03590v4, Algorithm 1 and §6. -/
theorem safeguardedAMSGradMD_repairs_component :
    Tendsto (fun t : ℕ => (safeguardedAMSGradMDRun (1 / 4) 1 1 1 0 0 (1 / 4) (1 / 4)
      oppositeQuadratic (unitGainState 1) t).2)
      atTop (𝓝 (-(fromMatrix (1 : Matrix (Fin 1) (Fin 1) ℝ)))) := by
  have hstar : gradient oppositeQuadratic
      (-(fromMatrix (1 : Matrix (Fin 1) (Fin 1) ℝ))) = 0 := by
    rw [oppositeQuadratic, trainingQuadratic_gradient]
    exact sub_self _
  exact (safeguardedAMSGradMD_minimum_convergence (1 / 4) 1 1 1 0 0 (1 / 4) (1 / 4) 1
    oppositeQuadratic (unitGainState 1) _ oppositeQuadratic_models.1
    oppositeQuadratic_models.2.1 (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) hstar).2.1

end Transformer.MagnitudeDirection
