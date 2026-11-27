module

public import Transformer.Optimization.PAGE.FiniteSum

/-! Finite-sum PAGE, Li et al. (2021), §4 and Appendix B. Conditional
variance and Lyapunov descent for the sampled update. The constants below
use a conservative step so the same potential also controls PL convergence. -/

@[expose] public section

namespace Transformer.Optimization.PAGE

open scoped InnerProductSpace

variable {I E : Type*} [Fintype I] [Nonempty I]
  [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

/-- Average squared change of the actual component gradients (Assumption 2).
A component may itself be a minibatch loss. -/
def AverageSmooth (loss : I → E → ℝ) (V : ℝ) : Prop :=
  ∀ x y, mean (fun i => ‖gradient (loss i) y - gradient (loss i) x‖^2) ≤ V^2 * ‖y-x‖^2

noncomputable def error (loss : I → E → ℝ) (s : State E) : ℝ :=
  ‖s.estimate - full loss s.weights‖^2

noncomputable def energy (loss : I → E → ℝ) (η p lower : ℝ) (s : State E) : ℝ :=
  objective loss s.weights - lower + η/p * error loss s

omit [CompleteSpace E] in
private theorem centered (e : E) (u : I → E) :
    mean (fun i => ‖e + (u i - mean u)‖^2) ≤ ‖e‖^2 + mean (fun i => ‖u i‖^2) := by
  have h (i : I) : ‖e + (u i - mean u)‖^2 =
      ‖e‖^2 + (2*⟪e, u i⟫_ℝ - 2*⟪e, mean u⟫_ℝ) +
      (‖u i‖^2 - 2*⟪mean u, u i⟫_ℝ + ‖mean u‖^2) := by
    rw [norm_add_sq_real, norm_sub_sq_real, inner_sub_right, real_inner_comm (u i) (mean u)]
    ring
  simp_rw [h]
  simp only [mean_add, mean_sub, mean_mul, mean_const, mean_inner, real_inner_self_eq_norm_sq]
  linarith [sq_nonneg ‖mean u‖]

/-- Lemma 3, obtained from using the same component at both weights. -/
theorem next_error (loss : I → E → ℝ) {η p V : ℝ} (hp : p ≤ 1)
    (hv : AverageSmooth loss V) (s : State E) :
    next loss η p (error loss) s ≤
      (1-p) * (error loss s + V^2 * η^2 * ‖s.estimate‖^2) := by
  let x := s.weights
  let y := x - η • s.estimate
  let u := fun i => gradient (loss i) y - gradient (loss i) x
  have hu : mean u = full loss y - full loss x := mean_sub _ _
  have diff (i : I) : (step loss η s (some i)).estimate - full loss y =
      (s.estimate - full loss x) + (u i - mean u) := by
    change s.estimate + u i - full loss y = _
    rw [hu]
    abel
  have he := centered (s.estimate - full loss x) u
  have hs := hv x y
  have hy : y-x = -(η • s.estimate) := by dsimp [y]; abel
  rw [hy, norm_neg, norm_smul, Real.norm_eq_abs, mul_pow, sq_abs] at hs
  change p * ‖full loss y - full loss y‖^2 +
    (1-p) * mean (fun i => ‖(step loss η s (some i)).estimate - full loss y‖^2) ≤ _
  simp only [sub_self, norm_zero, zero_pow (by decide : 2 ≠ 0), mul_zero, zero_add, diff]
  apply mul_le_mul_of_nonneg_left _ (sub_nonneg.mpr hp)
  have hs' : mean (fun i => ‖u i‖^2) ≤ V^2 * η^2 * ‖s.estimate‖^2 := by
    simpa only [mul_assoc, u] using hs
  exact he.trans (add_le_add le_rfl hs')

omit [Nonempty I] in
private theorem weight_descent (loss : I → E → ℝ) (hd : ∀ i, Differentiable ℝ (loss i))
    {L η : ℝ} (hl : SmoothObjective (objective loss) L) (s : State E) :
    objective loss (s.weights - η • s.estimate) ≤ objective loss s.weights -
      η/2 * ‖full loss s.weights‖^2 + η/2 * error loss s -
      η/2 * (1-η*L) * ‖s.estimate‖^2 := by
  have h := hl.2 s.weights (s.weights - η • s.estimate)
  rw [gradient_objective loss hd, sub_sub_cancel_left, inner_neg_right, real_inner_smul_right,
    norm_neg, norm_smul, Real.norm_eq_abs, mul_pow, sq_abs] at h
  rw [error, norm_sub_sq_real, real_inner_comm (full loss s.weights) s.estimate]
  linarith

/-- With positive η and p, this upper bound is equivalently
η L + 2 η² (1-p) V²/p ≤ 1. It depends only on the objective and sampler. -/
def StableStep (η p L V : ℝ) : Prop :=
  η/p * ((1-p) * V^2 * η^2) ≤ η/2 * (1-η*L)

/-- The full gradient and estimator error both spend the same descent budget. -/
theorem next_energy (loss : I → E → ℝ) (hd : ∀ i, Differentiable ℝ (loss i))
    {η p L V : ℝ} (hη : 0 < η) (hp : 0 < p) (hp' : p ≤ 1)
    (hl : SmoothObjective (objective loss) L) (hv : AverageSmooth loss V)
    (hs : StableStep η p L V) (lower : ℝ) (s : State E) :
    next loss η p (energy loss η p lower) s ≤ energy loss η p lower s -
      η/2 * (‖full loss s.weights‖^2 + error loss s) := by
  let y := s.weights - η • s.estimate
  have he : next loss η p (energy loss η p lower) s =
      objective loss y - lower + η/p * next loss η p (error loss) s := by
    unfold next energy
    change p * (objective loss y - lower + η/p * _) +
      (1-p) * mean (fun i => objective loss y - lower + η/p * error loss (step loss η s (some i))) = _
    rw [mean_add, mean_const, mean_mul]
    ring
  rw [he]
  have h1 := weight_descent loss hd hl (η := η) s
  have h2 := mul_le_mul_of_nonneg_left (next_error (η := η) loss hp' hv s) (div_pos hη hp).le
  have h3 := mul_le_mul_of_nonneg_right hs (sq_nonneg ‖s.estimate‖)
  have h4 : η/p * (1-p) = η/p - η := by field_simp
  have h5 := congrArg (fun z : ℝ => z * error loss s) h4
  dsimp only [y, energy]
  linarith

end Transformer.Optimization.PAGE
