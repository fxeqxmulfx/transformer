/-
# Perceptrons and attention's mean-field landscape — the transform off the sphere

`f_B^μ(x) = ∫ e^{xᵀ B y} dμ(y)` makes sense for every `x ∈ ℝ^d`, and there it
satisfies the discrete form of the source's `(Δ - 1) F_ν = 0`: with directions
`w_i` such that `w_i · B y = y_i`,
`Σ_i [f(x + t w_i) + f(x - t w_i) - 2 f(x)] = t² f(x) + O(t⁴ f(x))`
(`abs_secondDiff_sub_le`).  With the discrete maximum principle of
`SecondDiff`, two such transforms that agree on the unit sphere agree on the
closed unit ball (`attentionTransformMapExt_eq_of_eq_on_sphere`) — the
uniqueness for the Dirichlet problem that `rem: general-attention` invokes.
The remark itself is `TransformMap`.

Source: arXiv:2601.21366v2, `rem: general-attention`.
-/

import Transformer.Perceptron.Transform
import Transformer.Perceptron.SecondDiff

open scoped BigOperators
open Real MeasureTheory Filter

namespace Transformer
namespace Perceptron

variable {d : ℕ}

/-- `f_B^μ` on all of `ℝ^d`, `x ↦ ∫ e^{xᵀ B y} dμ(y)`; on the sphere it is
`attentionTransformMap`. -/
noncomputable def attentionTransformMapExt (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (μ : Measure (SSphere d)) (x : EucSpace d) : ℝ :=
  ∫ y, exp (inner (𝕜 := ℝ) x (B (y : EucSpace d))) ∂μ

theorem continuous_attentionTransformMapExt (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (μ : Measure (SSphere d)) [IsFiniteMeasure μ] :
    Continuous (attentionTransformMapExt B μ) := by
  have := continuous_parametric_integral_of_continuous (μ := μ)
    (f := fun (x : EucSpace d) (y : SSphere d) => exp (inner (𝕜 := ℝ) x (B (y : EucSpace d))))
    (Real.continuous_exp.comp (continuous_fst.inner
      (B.continuous_of_finiteDimensional.comp (continuous_subtype_val.comp continuous_snd))))
    isCompact_univ
  rw [Measure.restrict_univ] at this
  exact this

/-- **`Σ_i [f(x + t w_i) + f(x - t w_i) - 2 f(x)] = t² f(x) + O(t⁴ f(x))`**, for
`f = f_B^μ` and directions with `w_i · B y = y_i`: the discrete form of
`Δ F = F`, which the source obtains by differentiating under the integral. -/
theorem abs_secondDiff_sub_le (B : EucSpace d →ₗ[ℝ] EucSpace d) (w : Fin d → EucSpace d)
    (hw : ∀ i y, inner (𝕜 := ℝ) (w i) (B y) = y i) (μ : Measure (SSphere d))
    [IsFiniteMeasure μ] (x : EucSpace d) {t : ℝ} (ht : |t| ≤ 1) :
    |∑ i, (attentionTransformMapExt B μ (x + t • w i) +
        attentionTransformMapExt B μ (x - t • w i) - 2 * attentionTransformMapExt B μ x) -
      t ^ 2 * attentionTransformMapExt B μ x| ≤ t ^ 4 * attentionTransformMapExt B μ x := by
  have hBc := B.continuous_of_finiteDimensional
  have hint : ∀ g : SSphere d → ℝ, Continuous g → Integrable g μ := fun g hg =>
    Perspective.integrable_of_continuous_compact hg μ
  set e : SSphere d → ℝ := fun y => exp (inner (𝕜 := ℝ) x (B (y : EucSpace d))) with he_def
  have he : Continuous e := by fun_prop
  have hshift : ∀ (s : ℝ) (i : Fin d), attentionTransformMapExt B μ (x + s • w i) =
      ∫ y, e y * exp (s * (y : EucSpace d) i) ∂μ := fun s i => by
    refine integral_congr_ae (Eventually.of_forall fun y => ?_)
    simp only [he_def, inner_add_left, real_inner_smul_left, hw, Real.exp_add]
  have hminus : ∀ i, x - t • w i = x + (-t) • w i := fun i => by
    rw [neg_smul, sub_eq_add_neg]
  have hsplit : ∑ i, (attentionTransformMapExt B μ (x + t • w i) +
        attentionTransformMapExt B μ (x - t • w i) - 2 * attentionTransformMapExt B μ x) -
      t ^ 2 * attentionTransformMapExt B μ x =
      ∫ y, e y * (∑ i, (exp (t * (y : EucSpace d) i) + exp (-(t * (y : EucSpace d) i)) - 2)
        - t ^ 2) ∂μ := by
    simp_rw [hminus, hshift]
    have h0 : attentionTransformMapExt B μ x = ∫ y, e y ∂μ := rfl
    rw [h0]
    have hexp : ∀ y : SSphere d, e y * (∑ i, (exp (t * (y : EucSpace d) i) +
        exp (-(t * (y : EucSpace d) i)) - 2) - t ^ 2) =
        ∑ i, (e y * exp (t * (y : EucSpace d) i) + e y * exp (-t * (y : EucSpace d) i) -
          2 * e y) - t ^ 2 * e y := fun y => by
      rw [mul_sub, Finset.mul_sum]
      congr 1
      · exact Finset.sum_congr rfl fun i _ => by rw [neg_mul]; ring
      · ring
    simp_rw [hexp]
    rw [integral_sub (integrable_finsetSum _ fun i _ => hint _ (by fun_prop))
        (hint _ (by fun_prop)), integral_finsetSum _ fun i _ => hint _ (by fun_prop),
      integral_const_mul]
    congr 1
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [integral_sub, integral_add, integral_const_mul]
    all_goals exact hint _ (by fun_prop)
  rw [hsplit, ← Real.norm_eq_abs]
  calc ‖∫ y, e y * (∑ i, (exp (t * (y : EucSpace d) i) + exp (-(t * (y : EucSpace d) i)) - 2)
        - t ^ 2) ∂μ‖
      ≤ ∫ y, t ^ 4 * e y ∂μ := by
        refine norm_integral_le_of_norm_le (hint _ (by fun_prop)) (Eventually.of_forall fun y => ?_)
        rw [Real.norm_eq_abs, abs_mul, abs_of_pos (exp_pos _), mul_comm]
        exact mul_le_mul_of_nonneg_right
          (abs_sum_exp_sub_le (mem_sphere_zero_iff_norm.1 y.2) ht) (exp_pos _).le
    _ = t ^ 4 * attentionTransformMapExt B μ x := integral_const_mul _ _

/-- The hypotheses of `abs_secondDiff_sub_le` are satisfiable: `B = id` and the
coordinate axes `w_i = e_i`, with `t = 0`. -/
example : (∀ i (y : EucSpace 1), inner (𝕜 := ℝ) (EuclideanSpace.single i 1 : EucSpace 1)
      (LinearMap.id (R := ℝ) y) = y i) ∧ |(0 : ℝ)| ≤ 1 :=
  ⟨fun i y => by simp [EuclideanSpace.inner_single_left], by simp⟩

/-- **`f_B^{μ₁} = f_B^{μ₂}` on the sphere forces it on the closed ball**, given
directions with `w_i · B y = y_i`: the uniqueness half of the source's
Dirichlet problem for `Δ - 1`, through `le_zero_of_secondDiff_ge` applied to
`f_B^{μ₁} - f_B^{μ₂}` and to `f_B^{μ₂} - f_B^{μ₁}`. -/
theorem attentionTransformMapExt_eq_of_eq_on_sphere (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (w : Fin d → EucSpace d) (hw : ∀ i y, inner (𝕜 := ℝ) (w i) (B y) = y i)
    (μ₁ μ₂ : Measure (SSphere d)) [IsFiniteMeasure μ₁] [IsFiniteMeasure μ₂]
    (h : ∀ x : EucSpace d, ‖x‖ = 1 →
      attentionTransformMapExt B μ₁ x = attentionTransformMapExt B μ₂ x)
    {x : EucSpace d} (hx : ‖x‖ ≤ 1) :
    attentionTransformMapExt B μ₁ x = attentionTransformMapExt B μ₂ x := by
  have key : ∀ (a b : Measure (SSphere d)) [IsFiniteMeasure a] [IsFiniteMeasure b],
      (∀ x : EucSpace d, ‖x‖ = 1 →
        attentionTransformMapExt B a x = attentionTransformMapExt B b x) →
      attentionTransformMapExt B a x ≤ attentionTransformMapExt B b x := by
    intro a b _ _ hab
    refine sub_nonpos.1 (le_zero_of_secondDiff_ge
      (G := fun x => attentionTransformMapExt B a x - attentionTransformMapExt B b x)
      (S := fun x => attentionTransformMapExt B a x + attentionTransformMapExt B b x)
      ((continuous_attentionTransformMapExt B a).sub (continuous_attentionTransformMapExt B b))
      w (fun x t ht => ?_) (fun x hx => by simp [hab x hx]) hx)
    have e1 := abs_le.1 (abs_secondDiff_sub_le B w hw a x ht)
    have e2 := abs_le.1 (abs_secondDiff_sub_le B w hw b x ht)
    generalize attentionTransformMapExt B a = F₁ at e1 e2 ⊢
    generalize attentionTransformMapExt B b = F₂ at e1 e2 ⊢
    have hs : ∑ i, (F₁ (x + t • w i) - F₂ (x + t • w i) + (F₁ (x - t • w i) - F₂ (x - t • w i))
        - 2 * (F₁ x - F₂ x)) = ∑ i, (F₁ (x + t • w i) + F₁ (x - t • w i) - 2 * F₁ x) -
          ∑ i, (F₂ (x + t • w i) + F₂ (x - t • w i) - 2 * F₂ x) := by
      rw [← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun i _ => by ring
    rw [hs]
    linarith [e1.1, e2.2]
  exact le_antisymm (key μ₁ μ₂ h) (key μ₂ μ₁ fun x hx => (h x hx).symm)

/-- The hypotheses of `attentionTransformMapExt_eq_of_eq_on_sphere` are
satisfiable: `B = id`, `w_i = e_i`, `μ₁ = μ₂ = 0`. -/
example : (∀ i (y : EucSpace 1), inner (𝕜 := ℝ) (EuclideanSpace.single i 1 : EucSpace 1)
      (LinearMap.id (R := ℝ) y) = y i) ∧
    ∀ x : EucSpace 1, ‖x‖ = 1 → attentionTransformMapExt LinearMap.id 0 x =
      attentionTransformMapExt LinearMap.id 0 x :=
  ⟨fun i y => by simp [EuclideanSpace.inner_single_left], fun _ _ => rfl⟩

end Perceptron
end Transformer
