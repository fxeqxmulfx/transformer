/-
# Metastability — one framework for `SA` and `USA`

The theorem of §2 of arXiv:2410.06833v1 is stated for `(SA)` *or* `(USA)`:

  `(SA)   ẋ_i = Proj_{x_i} Σ_j  e^{β⟨x_i,x_j⟩} / Z_i · x_j`,    `Z_i = Σ_k e^{β⟨x_i,x_k⟩}`,
  `(USA)  ẋ_i = n⁻¹ Proj_{x_i} Σ_j e^{β(⟨x_i,x_j⟩ - 1)} x_j`.

The proof of the paper uses of the weights `a_{ij}` in `ẋ_i = Proj_{x_i} Σ_j a_{ij} x_j` only

  `n⁻¹ e^{β(⟨x_i,x_j⟩ - 1)} ≤ a_{ij} ≤ e^{β(⟨x_i,x_j⟩ - 1)}`,

and both models satisfy it: for `SA` because `e^β ≤ Z_i ≤ n e^β`, and for `USA`
with `a_{ij} = n⁻¹ e^{β(⟨x_i,x_j⟩ - 1)}` itself.  `IsAttnFlow` is the class of
trajectories with such weights; `isAttnFlow_of_SA` and `isAttnFlow_of_unnormalizedSA`
put both models in it, so the proof of the metastability theorem is written once.

`unnormalizedSA` is `(USA)` exactly as the metastability paper prints it, with the
factor `e^{-β}` inside the sum.  It is *not* `Perspective.USA`, which drops that factor
(the survey's normalisation): the two differ by the constant time change `t ↦ e^{β} t`.

Source: arXiv:2410.06833v1, §1 (`SA`, `USA`) and §2 (the proof uses only the bounds above).
-/

import Transformer.Basic
import Transformer.Perspective.Section1_IPS

open scoped BigOperators InnerProductSpace
open Real

namespace Transformer
namespace Metastability

variable (d n : ℕ)

/-- **Equation (USA)** of arXiv:2410.06833v1, as printed there:

  `ẋ_i(t) = n⁻¹ Proj_{x_i(t)} Σ_j e^{β(⟨x_i(t), x_j(t)⟩ - 1)} x_j(t)`. -/
def unnormalizedSA (β : ℝ) (X : ℝ → SphereTuple d n) : Prop :=
  ∀ t : ℝ, ∀ i : Idx n,
    HasDerivAt (fun s => (X s i : EucSpace d))
      (proj d (X t i : EucSpace d)
        ((n : ℝ)⁻¹ • ∑ j : Idx n,
          Real.exp (β * (⟪(X t i : EucSpace d), (X t j : EucSpace d)⟫_ℝ - 1))
            • (X t j : EucSpace d))) t

/-- **The bounds on the attention weights that the proof uses.**  `a i j` is the weight
of token `j` in the velocity of token `i` of the configuration `x`:

  `n⁻¹ e^{β(⟨x_i,x_j⟩ - 1)} ≤ a_{ij} ≤ e^{β(⟨x_i,x_j⟩ - 1)}`. -/
structure IsAttnWeights (β : ℝ) (x : Idx n → EucSpace d) (a : Idx n → Idx n → ℝ) :
    Prop where
  /-- The lower bound `a_{ij} ≥ n⁻¹ e^{β(⟨x_i,x_j⟩ - 1)}`. -/
  lower : ∀ i j, (1 / (n : ℝ)) * Real.exp (β * (⟪x i, x j⟫_ℝ - 1)) ≤ a i j
  /-- The upper bound `a_{ij} ≤ e^{β(⟨x_i,x_j⟩ - 1)}`. -/
  upper : ∀ i j, a i j ≤ Real.exp (β * (⟪x i, x j⟫_ℝ - 1))

/-- **An attention flow**: a trajectory on the sphere with
`ẋ_i = Proj_{x_i} Σ_j a_{ij}(t) x_j` for time-dependent weights `a(t)` obeying
`IsAttnWeights` at every time. -/
def IsAttnFlow (β : ℝ) (X : ℝ → SphereTuple d n) : Prop :=
  ∃ a : ℝ → Idx n → Idx n → ℝ,
    (∀ t, IsAttnWeights d n β (fun k => (X t k : EucSpace d)) (a t)) ∧
    ∀ t : ℝ, ∀ i : Idx n,
      HasDerivAt (fun s => (X s i : EucSpace d))
        (proj d (X t i : EucSpace d) (∑ j : Idx n, a t i j • (X t j : EucSpace d))) t

variable {d n}

/-- The inner product of two points of the sphere is at most `1`. -/
theorem inner_sphere_le_one (x y : SSphere d) :
    ⟪(x : EucSpace d), (y : EucSpace d)⟫_ℝ ≤ 1 := by
  have h := abs_real_inner_le_norm (x : EucSpace d) (y : EucSpace d)
  rw [mem_sphere_zero_iff_norm.mp x.2, mem_sphere_zero_iff_norm.mp y.2, one_mul] at h
  exact (abs_le.mp h).2

/-- The inner product of a point of the sphere with itself is `1`. -/
theorem inner_sphere_self (x : SSphere d) : ⟪(x : EucSpace d), (x : EucSpace d)⟫_ℝ = 1 := by
  rw [real_inner_self_eq_norm_mul_norm, mem_sphere_zero_iff_norm.mp x.2]; ring

/-- **`SA` is an attention flow**, for `β ≥ 0`: the normalised weights
`e^{β⟨x_i,x_j⟩}/Z_i` satisfy `n⁻¹ e^{β(⟨x_i,x_j⟩ - 1)} ≤ · ≤ e^{β(⟨x_i,x_j⟩ - 1)}`
because `e^β ≤ Z_i ≤ n e^β`. -/
theorem isAttnFlow_of_SA {β : ℝ} (hβ : 0 ≤ β) {X : ℝ → SphereTuple d n}
    (hX : Perspective.SA d n β X) : IsAttnFlow d n β X := by
  have hpart : ∀ t i, Real.exp β ≤ Perspective.partitionSA d n β X t i
      ∧ Perspective.partitionSA d n β X t i ≤ (n : ℝ) * Real.exp β := by
    intro t i
    refine ⟨?_, ?_⟩
    · have h := Finset.single_le_sum (f := fun k : Idx n =>
        Real.exp (β * ⟪(X t i : EucSpace d), (X t k : EucSpace d)⟫_ℝ))
        (fun k _ => (Real.exp_pos _).le) (Finset.mem_univ i)
      simp only [inner_sphere_self, mul_one] at h
      exact h
    · calc Perspective.partitionSA d n β X t i
          ≤ ∑ _k : Idx n, Real.exp β :=
            Finset.sum_le_sum fun k _ => Real.exp_le_exp.mpr
              (by nlinarith [inner_sphere_le_one (X t i) (X t k)])
        _ = (n : ℝ) * Real.exp β := by
            rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  refine ⟨fun t i j => Real.exp (β * ⟪(X t i : EucSpace d), (X t j : EucSpace d)⟫_ℝ)
      / Perspective.partitionSA d n β X t i, fun t => ⟨fun i j => ?_, fun i j => ?_⟩,
    fun t i => ?_⟩
  · have hn : (0 : ℝ) < n := by exact_mod_cast Fin.pos i
    have hZ : 0 < Perspective.partitionSA d n β X t i :=
      lt_of_lt_of_le (Real.exp_pos _) (hpart t i).1
    rw [le_div_iff₀ hZ]
    have hexp : Real.exp (β * (⟪(X t i : EucSpace d), (X t j : EucSpace d)⟫_ℝ - 1))
        * Real.exp β = Real.exp (β * ⟪(X t i : EucSpace d), (X t j : EucSpace d)⟫_ℝ) := by
      rw [← Real.exp_add]; congr 1; ring
    calc 1 / (n : ℝ) * Real.exp (β * (⟪(X t i : EucSpace d), (X t j : EucSpace d)⟫_ℝ - 1))
          * Perspective.partitionSA d n β X t i
        ≤ 1 / (n : ℝ) * Real.exp (β * (⟪(X t i : EucSpace d), (X t j : EucSpace d)⟫_ℝ - 1))
          * ((n : ℝ) * Real.exp β) :=
          mul_le_mul_of_nonneg_left (hpart t i).2 (by positivity)
      _ = Real.exp (β * ⟪(X t i : EucSpace d), (X t j : EucSpace d)⟫_ℝ) := by
          rw [← hexp]; field_simp
  · have hZ : 0 < Perspective.partitionSA d n β X t i :=
      lt_of_lt_of_le (Real.exp_pos _) (hpart t i).1
    rw [div_le_iff₀ hZ]
    have hexp : Real.exp (β * (⟪(X t i : EucSpace d), (X t j : EucSpace d)⟫_ℝ - 1))
        * Real.exp β = Real.exp (β * ⟪(X t i : EucSpace d), (X t j : EucSpace d)⟫_ℝ) := by
      rw [← Real.exp_add]; congr 1; ring
    rw [← hexp]
    exact mul_le_mul_of_nonneg_left (hpart t i).1 (Real.exp_pos _).le
  · have h := hX t i
    have hsm : (Perspective.partitionSA d n β X t i)⁻¹ •
          ∑ j : Idx n, Real.exp (β * ⟪(X t i : EucSpace d), (X t j : EucSpace d)⟫_ℝ)
            • (X t j : EucSpace d)
        = ∑ j : Idx n, (Real.exp (β * ⟪(X t i : EucSpace d), (X t j : EucSpace d)⟫_ℝ)
            / Perspective.partitionSA d n β X t i) • (X t j : EucSpace d) := by
      rw [Finset.smul_sum]
      exact Finset.sum_congr rfl fun j _ => by rw [smul_smul, div_eq_inv_mul]
    rw [hsm] at h
    exact h

/-- **`USA` is an attention flow**: its weights `n⁻¹ e^{β(⟨x_i,x_j⟩ - 1)}` are the lower
bound itself, and satisfy the upper bound because `n ≥ 1`. -/
theorem isAttnFlow_of_unnormalizedSA {β : ℝ} {X : ℝ → SphereTuple d n}
    (hX : unnormalizedSA d n β X) : IsAttnFlow d n β X := by
  refine ⟨fun t i j => (1 / (n : ℝ)) *
      Real.exp (β * (⟪(X t i : EucSpace d), (X t j : EucSpace d)⟫_ℝ - 1)),
    fun t => ⟨fun i j => le_rfl, fun i j => ?_⟩, fun t i => ?_⟩
  · have hn : (1 : ℝ) ≤ n := by exact_mod_cast Fin.pos i
    have h1 : 1 / (n : ℝ) ≤ 1 := by rw [div_le_one (by linarith)]; exact hn
    calc 1 / (n : ℝ) * Real.exp (β * (⟪(X t i : EucSpace d), (X t j : EucSpace d)⟫_ℝ - 1))
        ≤ 1 * Real.exp (β * (⟪(X t i : EucSpace d), (X t j : EucSpace d)⟫_ℝ - 1)) :=
          mul_le_mul_of_nonneg_right h1 (Real.exp_pos _).le
      _ = _ := one_mul _
  · have h := hX t i
    have hsm : (n : ℝ)⁻¹ • ∑ j : Idx n,
          Real.exp (β * (⟪(X t i : EucSpace d), (X t j : EucSpace d)⟫_ℝ - 1))
            • (X t j : EucSpace d)
        = ∑ j : Idx n, (1 / (n : ℝ) *
            Real.exp (β * (⟪(X t i : EucSpace d), (X t j : EucSpace d)⟫_ℝ - 1)))
            • (X t j : EucSpace d) := by
      rw [Finset.smul_sum]
      exact Finset.sum_congr rfl fun j _ => by rw [smul_smul, one_div]
    rw [hsm] at h
    exact h

end Metastability
end Transformer
