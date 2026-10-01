/-
# Algorithms used in the ablations

arXiv:2402.19449v2, Section 2.3 and Appendix A.6.
These are the stated heavy-ball updates with GD, normalized GD, and sign
directions. The article's Adam comparisons are empirical; its numbered
theorems analyze gradient flow and ordinary sign descent.
-/

import Transformer.Imbalance.Section3_Quadratic

noncomputable section

namespace Transformer.Imbalance

/-- Base directions compared in Section 2.3 and Appendix A.6. -/
inductive DirectionKind where
  | gradient
  | normalized
  | sign

/-- Actual directions from Appendix A.6; normalized GD uses Euclidean norm. -/
def direction {d : ℕ} (kind : DirectionKind) (g : EucSpace d) : EucSpace d :=
  match kind with
  | .gradient => g
  | .normalized => ‖g‖⁻¹ • g
  | .sign => WithLp.toLp 2 (fun r => Real.sign (g r))

/-- Heavy-ball momentum buffer update m_t=βm_(t-1)+d_t; Appendix A.6. -/
def momentumNext {d : ℕ} (kind : DirectionKind) (β : ℝ) (m g : EucSpace d) : EucSpace d :=
  β • m + direction kind g

/-- Parameter update x_(t+1)=x_t-αm_t; Appendix A.6. -/
def optimizerNext {d : ℕ} (kind : DirectionKind) (α β : ℝ) (x m g : EucSpace d) : EucSpace d :=
  x - α • momentumNext kind β m g

/-- Normalized GD has a unit direction whenever the gradient is nonzero;
Section 2.3 and Appendix A.6. -/
theorem normalized_direction_norm {d : ℕ} (g : EucSpace d) (hg : g ≠ 0) :
    ‖direction .normalized g‖ = 1 := by
  have hn : 0 < ‖g‖ := norm_pos_iff.2 hg
  rw [direction, norm_smul, Real.norm_eq_abs, abs_of_pos (inv_pos.2 hn)]
  exact inv_mul_cancel₀ hn.ne'

/-- Nonvacuity of the normalization hypothesis; Appendix A.6. -/
example : (WithLp.toLp 2 (fun _ : Fin 1 => (1 : ℝ)) : EucSpace 1) ≠ 0 := by
  intro h
  have := congrArg (fun g : EucSpace 1 => g 0) h
  norm_num at this

/-- Ordinary sign directions are invariant to positive scalar rescaling,
the mechanism used in Section 3.1 and Appendix A.6. -/
theorem sign_direction_scale {d : ℕ} (g : EucSpace d) (a : ℝ) (ha : 0 < a) :
    direction .sign (a • g) = direction .sign g := by
  ext r
  exact sign_positive_scale a (g r) ha

/-- Nonvacuity of the sign-scale hypothesis; Section 3.1 and Appendix A.6. -/
example : (0 : ℝ) < 1 / 1000 := by norm_num

/-- Setting β=0 recovers the three momentum-free updates;
Section 2.3 and Appendix A.6. -/
theorem optimizerNext_no_momentum {d : ℕ} (kind : DirectionKind) (α : ℝ)
    (x m g : EucSpace d) : optimizerNext kind α 0 x m g = x - α • direction kind g := by
  simp [optimizerNext, momentumNext]

end Transformer.Imbalance
