module

public import Transformer.ScheduleFree.SectionE_Regret
public import Transformer.ScheduleFree.SectionA_Conversion

/-!
# Ball invariants and strong convexity

arXiv:2405.15682v4, Appendix E. The norm bounds used in its last line
follow from the actual projection and the convex combinations. The
initial base point must be in the ball, an implicit source assumption.
-/

/-!
# Strongly convex online-to-batch conversion

arXiv:2405.15682v4, Appendix E, Theorem 6, first inequality. The
negative comparator Bregman term pays for the surrogate curvature.
As in the general conversion, a minimizer is allowed but not required.
-/

@[expose] public section

open scoped BigOperators

noncomputable section

section

namespace Transformer.ScheduleFree

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Appendix E: a strongly convex supporting quadratic model. -/
def StrongLower (F : E → ℝ) (grad : E → E) (μ : ℝ) : Prop :=
  ∀ x y, F x + inner ℝ (grad x) (y - x) + μ / 2 * ‖y - x‖ ^ 2 ≤ F y

theorem strong_subgradient {F : E → ℝ} {grad : E → E} {μ : ℝ}
    (hμ : 0 ≤ μ) (hstrong : StrongLower F grad μ) (x : E) : IsSubgradient F x (grad x) := by
  intro y
  have h := hstrong x y
  have hp : 0 ≤ μ / 2 * ‖y - x‖ ^ 2 := by positivity
  linarith

/-- A nonvacuous family satisfying the strong model at every point. -/
theorem quadratic_strong (μ : ℝ) :
    StrongLower (fun x : E => μ / 2 * ‖x‖ ^ 2) (fun x => μ • x) μ := by
  intro x y
  rw [real_inner_smul_left, inner_sub_right, real_inner_self_eq_norm_sq,
    norm_sub_sq_real, real_inner_comm x y]
  ring_nf
  exact le_rfl

example (x : E) : IsSubgradient (fun v : E => (1 : ℝ) / 2 * ‖v‖ ^ 2) x ((1 : ℝ) • x) :=
  strong_subgradient zero_le_one (quadratic_strong 1) x

/-- Appendix E: weighted averages of points in the ball stay in the ball. -/
theorem average_norm {w : ℕ → ℝ} (hw : ∀ t, 0 ≤ w t) (hfirst : 0 < w 0)
    {D : ℝ} (hD : 0 ≤ D) (z : ℕ → E) (hz : ∀ t, ‖z t‖ ≤ D) (n : ℕ) :
    ‖average w z n‖ ≤ D := by
  cases n with
  | zero => simpa [average, weightSum] using hD
  | succ n =>
    have hW := weightSum_pos_succ hw hfirst n
    have hs : ‖∑ t ∈ Finset.range (n + 1), w t • z t‖ ≤ weightSum w (n + 1) * D := by
      calc
        _ ≤ ∑ t ∈ Finset.range (n + 1), ‖w t • z t‖ := norm_sum_le _ _
        _ ≤ ∑ t ∈ Finset.range (n + 1), w t * D := by
          apply Finset.sum_le_sum
          intro t ht
          rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (hw t)]
          exact mul_le_mul_of_nonneg_left (hz t) (hw t)
        _ = _ := (Finset.sum_mul _ _ D).symm
    rw [average, norm_smul, Real.norm_eq_abs, abs_of_nonneg (inv_nonneg.mpr hW.le)]
    calc
      _ ≤ (weightSum w (n + 1))⁻¹ * (weightSum w (n + 1) * D) :=
        mul_le_mul_of_nonneg_left hs (inv_nonneg.mpr hW.le)
      _ = D := by rw [← mul_assoc, inv_mul_cancel₀ hW.ne', one_mul]

example (z : ℕ → E) (hz : ∀ t, ‖z t‖ ≤ 1) (n : ℕ) :
    ‖average (fun _ => 1) z n‖ ≤ 1 :=
  average_norm (fun _ => zero_le_one) zero_lt_one zero_le_one z hz n

theorem mix_norm {α D : ℝ} (hα : 0 ≤ α) (hα1 : α ≤ 1) (x z : E)
    (hx : ‖x‖ ≤ D) (hz : ‖z‖ ≤ D) : ‖mix α x z‖ ≤ D := by
  have h := norm_add_le ((1 - α) • x) (α • z)
  rw [norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs,
    abs_of_nonneg (sub_nonneg.mpr hα1), abs_of_nonneg hα] at h
  have h1 := mul_le_mul_of_nonneg_left hx (sub_nonneg.mpr hα1)
  have h2 := mul_le_mul_of_nonneg_left hz hα
  unfold mix
  nlinarith

example (x : E) : ‖mix 0 x x‖ ≤ ‖x‖ := mix_norm le_rfl zero_le_one x x le_rfl le_rfl

/-- Appendix E's surrogate gradients are bounded by `G+2μD`. -/
theorem surrogate_gradient_bound {μ D G : ℝ} (hμ : 0 ≤ μ) (z y g : E)
    (hz : ‖z‖ ≤ D) (hy : ‖y‖ ≤ D) (hg : ‖g‖ ≤ G) :
    ‖g + μ • (z - y)‖ ≤ G + 2 * μ * D := by
  have h := norm_add_le g (μ • (z - y))
  rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hμ] at h
  have hs := mul_le_mul_of_nonneg_left ((norm_sub_le z y).trans (add_le_add hz hy)) hμ
  linarith

example (z y g : E) : ‖g + (0 : ℝ) • (z - y)‖ ≤ ‖g‖ + 2 * 0 * (‖z‖ + ‖y‖) :=
  surrogate_gradient_bound le_rfl z y g (by linarith [norm_nonneg y])
    (by linarith [norm_nonneg z]) le_rfl

theorem weightSum_linear (n : ℕ) :
    weightSum (fun t => (t : ℝ) + 1) n = (n : ℝ) * ((n : ℝ) + 1) / 2 := by
  induction n with
  | zero => simp [weightSum]
  | succ n ih => rw [weightSum_succ, ih]; push_cast; ring

end Transformer.ScheduleFree

end

section

namespace Transformer.ScheduleFree

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Appendix E, Theorem 6: exact pathwise strong conversion before taking
expectations. The RHS equals surrogate regret minus `Σwμ‖z-y‖²/2`. -/
theorem strong_conversion {F : E → ℝ} {grad : E → E} {μ : ℝ}
    (hμ : 0 ≤ μ) (hstrong : StrongLower F grad μ) {w α : ℕ → ℝ}
    (hw : ∀ t, 0 ≤ w t) (hfirst : 0 < w 0) (hα : ∀ t, 0 ≤ α t ∧ α t ≤ 1)
    (z : ℕ → E) (u : E) (T : ℕ) :
    weightSum w T * (F (average w z T) - F u) ≤
      ∑ t ∈ Finset.range T, w t *
        (inner ℝ (grad (query w α z t)) (z t - u) - μ / 2 * ‖u - query w α z t‖ ^ 2) := by
  have hgrad := strong_subgradient hμ hstrong
  rw [conversion_identity F grad w α z u T
    (fun t _ => (weightSum_pos_succ hw hfirst t).ne')]
  apply Finset.sum_le_sum
  intro t ht
  have hgap := mul_nonneg (hw t) (mixGap_nonneg (x := average w z (t + 1)) (z := z t)
    (hα t).1 (hα t).2 (hgrad _) (hgrad _))
  have hp := mul_nonneg (weightSum_nonneg hw t) (bregman_nonneg (hgrad (average w z (t + 1)))
    (a := average w z t))
  have hs : μ / 2 * ‖u - query w α z t‖ ^ 2 ≤ bregman F grad u (query w α z t) := by
    have h := hstrong (query w α z t) u
    unfold bregman
    linarith
  have hwHS := mul_le_mul_of_nonneg_left hs (hw t)
  unfold defect
  nlinarith

example (z : ℕ → ℝ) (u : ℝ) (T : ℕ) :
    weightSum (fun _ => 1) T *
      (‖average (fun _ => 1) z T‖ ^ 2 / 2 - ‖u‖ ^ 2 / 2) ≤
      ∑ t ∈ Finset.range T, (1 : ℝ) *
        (inner ℝ ((1 : ℝ) • query (fun _ => 1) (fun _ => 0) z t) (z t - u) -
          1 / 2 * ‖u - query (fun _ => 1) (fun _ => 0) z t‖ ^ 2) := by
  convert strong_conversion zero_le_one (quadratic_strong (E := ℝ) 1)
    (fun _ => zero_le_one) zero_lt_one (fun _ => ⟨le_rfl, zero_le_one⟩) z u T using 1
  ring

end Transformer.ScheduleFree

end
