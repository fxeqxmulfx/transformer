/-
# Appendix B — the modified metric costs `O(β)`, proved

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*, Appendix B, `eq: metric.grad`.

The statement compares two directional derivatives along one and the same
curve of tuples: that of `𝖤_β = (2β)⁻¹ Σ_i Σ_j e^{β ⟨x_i, x_j⟩}` and that of
`𝖤_0 = n⁻¹ Σ_i Σ_j ⟨x_i, x_j⟩`, and asserts that they agree up to `O(β)`
once the normalisations are matched by the factor `n/2`.

The proof is the two derivatives, computed and subtracted:

* `hasDerivAt_selfEnergy` differentiates the double exponential sum;
* `symmetrized_double_sum` uses the symmetry of the weight `e^{β ⟨x_i, x_j⟩}`
  to fold the two halves of `d/dt ⟨x_i, x_j⟩` into one, cancelling the `2β`;
* `hasDerivAt_E0` (from `Perspective.AppendixA_Beta0`) does the same for `𝖤_0`,
  whose weight is the constant `1`.

What is left is `Σ_i Σ_j (e^{β ⟨x_i, x_j⟩} - 1) ⟨x_j, v_i⟩`, and on the sphere
`|⟨x_i, x_j⟩| ≤ 1`, so `|e^{β ⟨x_i, x_j⟩} - 1| ≤ 2β` for `β ≤ 1`.  The constant
is `2 n Σ_i ‖v_i‖ + 1`, explicit in the initial tuple and the velocity.
-/

import Transformer.Perspective.AppendixB_HighD
import Mathlib.Analysis.Complex.Exponential

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- The derivative of `𝖤_β` along a curve of tuples: with `ẋ_i(t) = b_i`,

  `d/dt 𝖤_β(X(t)) = (2β)⁻¹ Σ_i Σ_j e^{β ⟨x_i, x_j⟩} β (⟨x_i, b_j⟩ + ⟨x_j, b_i⟩)`.

This is `Perspective.hasDerivAt_E0` for the `β > 0` energy, and the `V = Id`
case of the computation inside `Perspective.sa_is_gradient_flow`. -/
theorem hasDerivAt_selfEnergy (β : ℝ) (Y : ℝ → SphereTuple d n)
    (b : Idx n → EucSpace d) (t : ℝ)
    (hY : ∀ i : Idx n, HasDerivAt (fun s => (Y s i : EucSpace d)) (b i) t) :
    HasDerivAt (fun s => selfEnergy d n β (Y s))
      ((2 * β)⁻¹ * ∑ i : Idx n, ∑ j : Idx n,
        Real.exp (β * inner (𝕜 := ℝ) ((Y t i : EucSpace d)) ((Y t j : EucSpace d)))
          * (β * (inner (𝕜 := ℝ) ((Y t i : EucSpace d)) (b j)
              + inner (𝕜 := ℝ) ((Y t j : EucSpace d)) (b i)))) t := by
  have hterm : ∀ i j : Idx n,
      HasDerivAt
        (fun s => Real.exp (β * inner (𝕜 := ℝ) ((Y s i : EucSpace d)) ((Y s j : EucSpace d))))
        (Real.exp (β * inner (𝕜 := ℝ) ((Y t i : EucSpace d)) ((Y t j : EucSpace d)))
          * (β * (inner (𝕜 := ℝ) ((Y t i : EucSpace d)) (b j)
              + inner (𝕜 := ℝ) ((Y t j : EucSpace d)) (b i)))) t := by
    intro i j
    have hi : HasDerivAt
        (fun s => inner (𝕜 := ℝ) ((Y s i : EucSpace d)) ((Y s j : EucSpace d)))
        (inner (𝕜 := ℝ) ((Y t i : EucSpace d)) (b j)
          + inner (𝕜 := ℝ) (b i) ((Y t j : EucSpace d))) t :=
      HasDerivAt.inner ℝ (hY i) (hY j)
    have h := (hi.const_mul β).exp
    rwa [real_inner_comm ((Y t j : EucSpace d)) (b i)] at h
  have hsum := HasDerivAt.const_mul ((2 * β)⁻¹) (HasDerivAt.fun_sum
    (fun i (_ : i ∈ (Finset.univ : Finset (Idx n))) =>
      HasDerivAt.fun_sum
        (fun j (_ : j ∈ (Finset.univ : Finset (Idx n))) => hterm i j)))
  simpa only [selfEnergy, particleEnergy, ContinuousLinearMap.coe_id', id_eq] using hsum

/-- A symmetric weight folds the two halves of a symmetrized double sum into
one: for `w i j = w j i` and `β ≠ 0`,

  `(2β)⁻¹ Σ_i Σ_j w_{ij} β (g_{ji} + g_{ij}) = Σ_i Σ_j w_{ij} g_{ij}`.

This is what turns the derivative of `𝖤_β` into the modified metric of §3.4
paired against the velocity. -/
theorem symmetrized_double_sum (β : ℝ) (hβ : β ≠ 0) (w g : Idx n → Idx n → ℝ)
    (hw : ∀ i j : Idx n, w i j = w j i) :
    (2 * β)⁻¹ * ∑ i : Idx n, ∑ j : Idx n, w i j * (β * (g j i + g i j))
      = ∑ i : Idx n, ∑ j : Idx n, w i j * g i j := by
  have h1 : ∑ i : Idx n, ∑ j : Idx n, w i j * g j i
      = ∑ i : Idx n, ∑ j : Idx n, w i j * g i j := by
    rw [Finset.sum_comm]
    exact Finset.sum_congr rfl fun i _ =>
      Finset.sum_congr rfl fun j _ => by rw [hw]
  have h2 : ∑ i : Idx n, ∑ j : Idx n, w i j * (β * (g j i + g i j))
      = β * (∑ i : Idx n, ∑ j : Idx n, w i j * g j i)
        + β * ∑ i : Idx n, ∑ j : Idx n, w i j * g i j := by
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun j _ => by ring
  rw [h2, h1]
  field_simp
  ring

/-- **Equation (eq: metric.grad).** *The modified metric costs `O(β)`.*

  `g_β(∇_{g_β} 𝖤_β(x), v) = g(∇_g 𝖤_0(x), v) + O(β)`.

Both sides are directional derivatives: the left one of `𝖤_β` and the right
one of `𝖤_0`, along a curve with velocity `v`.  The factor `n/2` is the
difference of normalisations, `𝖤_0 = n⁻¹ Σ_i Σ_j ⟨x_i, x_j⟩` against
`𝖤_β = (2β)⁻¹ Σ_i Σ_j e^{β ⟨x_i, x_j⟩}`, whose `β → 0` derivative is
`Σ_i Σ_j ⟨v_i, x_j⟩`.

The constant is `2 n Σ_i ‖v_i‖ + 1`: the bound `|e^u - 1| ≤ 2|u|` for
`|u| ≤ 1` against `|⟨x_i, x_j⟩| ≤ 1` and `|⟨x_j, v_i⟩| ≤ ‖v_i‖`.

Source: arXiv:2312.10794v5, Appendix B, `eq: metric.grad`. -/
theorem metric_grad_comparison
    (X : SphereTuple d n) (v : Idx n → EucSpace d) :
    ∃ C : ℝ, 0 < C ∧
      ∀ Y : ℝ → SphereTuple d n, Y 0 = X →
        (∀ i : Idx n, HasDerivAt (fun s => (Y s i : EucSpace d)) (v i) 0) →
        ∀ c₀ : ℝ, HasDerivAt (fun s => E0 d n (Y s)) c₀ 0 →
          ∀ β : ℝ, 0 < β → β ≤ 1 →
            ∀ cβ : ℝ, HasDerivAt (fun s => selfEnergy d n β (Y s)) cβ 0 →
              |cβ - ((n : ℝ) / 2) * c₀| ≤ C * β := by
  have hvnn : (0 : ℝ) ≤ ∑ i : Idx n, ‖v i‖ := Finset.sum_nonneg fun i _ => norm_nonneg _
  refine ⟨2 * (n : ℝ) * (∑ i : Idx n, ‖v i‖) + 1, ?_, ?_⟩
  · have : (0 : ℝ) ≤ 2 * (n : ℝ) * (∑ i : Idx n, ‖v i‖) :=
      mul_nonneg (by positivity) hvnn
    linarith
  intro Y _hY0 hYd c₀ hc₀ β hβ hβ1 cβ hcβ
  -- The unit vectors of the tuple at `t = 0`.
  have hnorm : ∀ i : Idx n, ‖(Y 0 i : EucSpace d)‖ = 1 :=
    fun i => mem_sphere_zero_iff_norm.mp (Y 0 i).2
  -- The derivative of `𝖤_β`, with the `2β` cancelled.
  have hcβeq : cβ = ∑ i : Idx n, ∑ j : Idx n,
      Real.exp (β * inner (𝕜 := ℝ) ((Y 0 i : EucSpace d)) ((Y 0 j : EucSpace d)))
        * inner (𝕜 := ℝ) ((Y 0 j : EucSpace d)) (v i) := by
    rw [hcβ.unique (hasDerivAt_selfEnergy d n β Y v 0 hYd)]
    exact symmetrized_double_sum n β (ne_of_gt hβ) _ _
      (fun i j => by rw [real_inner_comm ((Y 0 i : EucSpace d))])
  -- The derivative of `𝖤_0`, with the normalisation matched.
  have hc₀eq : c₀ = (2 * (n : ℝ)⁻¹) *
      ∑ i : Idx n, inner (𝕜 := ℝ) (v i) (∑ j : Idx n, (Y 0 j : EucSpace d)) :=
    hc₀.unique (hasDerivAt_E0 d n Y v 0 hYd)
  have hc₀' : ((n : ℝ) / 2) * c₀
      = ∑ i : Idx n, ∑ j : Idx n, inner (𝕜 := ℝ) ((Y 0 j : EucSpace d)) (v i) := by
    have hT : ∀ i : Idx n, inner (𝕜 := ℝ) (v i) (∑ j : Idx n, (Y 0 j : EucSpace d))
        = ∑ j : Idx n, inner (𝕜 := ℝ) ((Y 0 j : EucSpace d)) (v i) := by
      intro i
      rw [inner_sum]
      exact Finset.sum_congr rfl fun j _ => real_inner_comm _ _
    rw [hc₀eq, Finset.sum_congr rfl (fun i (_ : i ∈ Finset.univ) => hT i)]
    rcases Finset.eq_empty_or_nonempty (Finset.univ : Finset (Idx n)) with he | hne
    · rw [he]; simp
    · obtain ⟨i₀, -⟩ := hne
      have hn0 : (n : ℝ) ≠ 0 :=
        Nat.cast_ne_zero.mpr (Fin.pos_iff_nonempty.mpr ⟨i₀⟩).ne'
      have h1 : ((n : ℝ) / 2) * (2 * (n : ℝ)⁻¹) = 1 := by field_simp
      rw [← mul_assoc, h1, one_mul]
  -- What is left is the deviation of the weight from `1`.
  have hsub : (∑ i : Idx n, ∑ j : Idx n,
        Real.exp (β * inner (𝕜 := ℝ) ((Y 0 i : EucSpace d)) ((Y 0 j : EucSpace d)))
          * inner (𝕜 := ℝ) ((Y 0 j : EucSpace d)) (v i))
      - ∑ i : Idx n, ∑ j : Idx n, inner (𝕜 := ℝ) ((Y 0 j : EucSpace d)) (v i)
      = ∑ i : Idx n, ∑ j : Idx n,
          (Real.exp (β * inner (𝕜 := ℝ) ((Y 0 i : EucSpace d)) ((Y 0 j : EucSpace d))) - 1)
            * inner (𝕜 := ℝ) ((Y 0 j : EucSpace d)) (v i) := by
    rw [← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun j _ => by ring
  have hbound : ∀ i j : Idx n,
      |(Real.exp (β * inner (𝕜 := ℝ) ((Y 0 i : EucSpace d)) ((Y 0 j : EucSpace d))) - 1)
          * inner (𝕜 := ℝ) ((Y 0 j : EucSpace d)) (v i)| ≤ 2 * β * ‖v i‖ := by
    intro i j
    have hij : |inner (𝕜 := ℝ) ((Y 0 i : EucSpace d)) ((Y 0 j : EucSpace d))| ≤ 1 := by
      have := abs_real_inner_le_norm ((Y 0 i : EucSpace d)) ((Y 0 j : EucSpace d))
      rwa [hnorm i, hnorm j, one_mul] at this
    have hu : |β * inner (𝕜 := ℝ) ((Y 0 i : EucSpace d)) ((Y 0 j : EucSpace d))| ≤ 1 := by
      rw [abs_mul, abs_of_pos hβ]
      nlinarith [abs_nonneg (inner (𝕜 := ℝ) ((Y 0 i : EucSpace d)) ((Y 0 j : EucSpace d)))]
    have hexp : |Real.exp (β * inner (𝕜 := ℝ) ((Y 0 i : EucSpace d)) ((Y 0 j : EucSpace d)))
        - 1| ≤ 2 * β := by
      refine (Real.abs_exp_sub_one_le hu).trans ?_
      rw [abs_mul, abs_of_pos hβ]
      nlinarith [abs_nonneg (inner (𝕜 := ℝ) ((Y 0 i : EucSpace d)) ((Y 0 j : EucSpace d)))]
    have hg : |inner (𝕜 := ℝ) ((Y 0 j : EucSpace d)) (v i)| ≤ ‖v i‖ := by
      have := abs_real_inner_le_norm ((Y 0 j : EucSpace d)) (v i)
      rwa [hnorm j, one_mul] at this
    rw [abs_mul]
    exact mul_le_mul hexp hg (abs_nonneg _) (by positivity)
  rw [hcβeq, hc₀', hsub]
  calc |∑ i : Idx n, ∑ j : Idx n,
          (Real.exp (β * inner (𝕜 := ℝ) ((Y 0 i : EucSpace d)) ((Y 0 j : EucSpace d))) - 1)
            * inner (𝕜 := ℝ) ((Y 0 j : EucSpace d)) (v i)|
      ≤ ∑ i : Idx n, ∑ j : Idx n, 2 * β * ‖v i‖ := by
        refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun i _ => ?_)
        exact (Finset.abs_sum_le_sum_abs _ _).trans
          (Finset.sum_le_sum fun j _ => hbound i j)
    _ = 2 * (n : ℝ) * (∑ i : Idx n, ‖v i‖) * β := by
        simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
          ← Finset.mul_sum]
        ring
    _ ≤ (2 * (n : ℝ) * (∑ i : Idx n, ‖v i‖) + 1) * β := by nlinarith

/-- The hypotheses are satisfiable: the constant curve at a tuple has velocity
`0`, and then both directional derivatives are `0`. -/
example (X : SphereTuple d n) :
    (fun _ : ℝ => X) 0 = X ∧
      ∀ i : Idx n, HasDerivAt (fun _ : ℝ => (X i : EucSpace d)) 0 0 :=
  ⟨rfl, fun _ => hasDerivAt_const _ _⟩

end Perspective
end Transformer
