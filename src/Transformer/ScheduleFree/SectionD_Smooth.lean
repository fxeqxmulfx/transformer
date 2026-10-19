module

public import Transformer.ScheduleFree.SectionC_Bregman

/-!
# Smooth convex Bregman bounds

arXiv:2405.15682v4, Appendix D, proof of Corollary 1. The source first
prints `B_F(a,b) ≥ 2L‖∇F(a)-∇F(b)‖²`; its next displayed calculation
uses the correct coefficient `1/(2L)`. We prove the latter from the
supporting-plane and smooth quadratic upper models. These models keep
the analytic assumptions explicit and also cover nonscalar Hilbert spaces.
-/

@[expose] public section

noncomputable section

namespace Transformer.ScheduleFree

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Appendix D: the standard `L`-smooth quadratic upper model. -/
def SmoothUpper (F : E → ℝ) (grad : E → E) (L : ℝ) : Prop :=
  ∀ x y, F y ≤ F x + inner ℝ (grad x) (y - x) + L / 2 * ‖y - x‖ ^ 2

/-- Appendix D, corrected coefficient: smooth convex Bregman lower bound.
The proof takes a single gradient step on the loss minus its tangent at `b`. -/
theorem smooth_bregman {F : E → ℝ} {grad : E → E} {L : ℝ} (hL : 0 < L)
    (hgrad : ∀ x, IsSubgradient F x (grad x)) (hsmooth : SmoothUpper F grad L)
    (a b : E) : ‖grad a - grad b‖ ^ 2 / (2 * L) ≤ bregman F grad a b := by
  let d := grad a - grad b
  let v := a - L⁻¹ • d
  have hlow := hgrad b v
  have hupp := hsmooth a v
  have hv : v - a = -(L⁻¹ • d) := by dsimp [v]; module
  have hvb : v - b = (a - b) - L⁻¹ • d := by dsimp [v]; module
  rw [hv, norm_neg, norm_smul, Real.norm_eq_abs,
    abs_of_nonneg (inv_nonneg.mpr hL.le), inner_neg_right, real_inner_smul_right] at hupp
  rw [hvb, inner_sub_right, real_inner_smul_right] at hlow
  have hid : inner ℝ (grad a) d - inner ℝ (grad b) d = ‖d‖ ^ 2 := by
    rw [← inner_sub_left]
    exact real_inner_self_eq_norm_sq d
  apply (div_le_iff₀ (by positivity : 0 < 2 * L)).mpr
  unfold bregman
  have he : L * L⁻¹ = 1 := mul_inv_cancel₀ hL.ne'
  have h := sub_nonneg.mpr (hlow.trans hupp)
  field_simp at h
  nlinarith [hid]

example (a b : ℝ) : ‖(0 : ℝ) - 0‖ ^ 2 / (2 * 1) ≤
    bregman (fun _ => 0) (fun _ => 0) a b :=
  smooth_bregman (F := fun _ : ℝ => 0) (grad := fun _ => 0) zero_lt_one
    (by intro x u; simp) (by intro x y; simp only [inner_zero_left, zero_add]; positivity) a b

omit [InnerProductSpace ℝ E] in
/-- Appendix D: three vector increments have squared norm at most three
times the sum of their squared norms. -/
theorem norm_three_sq (a b c : E) :
    ‖a + b + c‖ ^ 2 ≤ 3 * (‖a‖ ^ 2 + ‖b‖ ^ 2 + ‖c‖ ^ 2) := by
  have h1 := sq_nonneg (‖a‖ - ‖b‖)
  have h2 := sq_nonneg (‖a‖ - ‖c‖)
  have h3 := sq_nonneg (‖b‖ - ‖c‖)
  have hn : ‖a + b + c‖ ≤ ‖a‖ + ‖b‖ + ‖c‖ := by
    calc
      _ ≤ ‖a + b‖ + ‖c‖ := norm_add_le _ _
      _ ≤ _ := by gcongr; exact norm_add_le a b
  nlinarith [sq_le_sq₀ (norm_nonneg (a + b + c))
    (by positivity : 0 ≤ ‖a‖ + ‖b‖ + ‖c‖) |>.mpr hn]

end Transformer.ScheduleFree
