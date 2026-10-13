/-
# A global quadratic bound for the actual angular energy

ArXiv:2410.06833v1, §3.2, `lem: PL.borjan`.
The scalar quadratic estimate is summed over the actual pair potential.
Its linear term is the true gradient pairing; the quadratic term is
bounded by the Euclidean coordinate-square displacement.
No cap, flow, or convergence assumption enters this global energy bound.
-/

import Transformer.Metastability.Section3_EnergyAlgebra
import Transformer.Metastability.Section3_ClusterSpread

open scoped BigOperators
open Real

namespace Transformer.Metastability

/-- **A global Taylor upper bound for the actual angular interaction energy.**
Source: arXiv:2410.06833v1, §3.2, the energy-discrepancy step in
`lem: PL.borjan`. The quadratic coefficient `2(1+β)` is a loose bound;
it is chosen uniformly over configurations and all positive token counts. -/
theorem angularEβ_quadratic_upper (n : ℕ) (hn : 0 < n) (β : ℝ) (hβ : 0 < β)
    (Θ v : Idx n → ℝ) :
    angularEβ n β (Θ + v) - angularEβ n β Θ ≤
      (∑ i : Idx n, angularGrad n β Θ i * v i) +
        2 * (1 + β) * ∑ i : Idx n, (v i) ^ 2 := by
  have hnR : (0 : ℝ) < n := by exact_mod_cast hn
  let C := 1 / (2 * (n : ℝ) ^ 2)
  have hC : 0 ≤ C := by dsimp only [C]; positivity
  have hpair (i j : Idx n) :
      angularPairPotential β ((Θ + v) i - (Θ + v) j) - angularPairPotential β (Θ i - Θ j) ≤
        -Causal.h_pot β (Θ i - Θ j) * (v i - v j) + (1 + β) * (v i - v j) ^ 2 := by
    have h := angularPairPotential_upper β hβ (Θ i - Θ j) ((Θ + v) i - (Θ + v) j)
    have heq : (Θ + v) i - (Θ + v) j - (Θ i - Θ j) = v i - v j := by
      simp only [Pi.add_apply]
      ring
    simpa only [heq] using h
  have hsum := Finset.sum_le_sum (s := (Finset.univ : Finset (Idx n))) fun i _ =>
    Finset.sum_le_sum (s := (Finset.univ : Finset (Idx n))) fun j _ => hpair i j
  have hscaled := mul_le_mul_of_nonneg_left hsum hC
  have hE : angularEβ n β (Θ + v) - angularEβ n β Θ =
      C * ∑ i : Idx n, ∑ j : Idx n,
        (angularPairPotential β ((Θ + v) i - (Θ + v) j) - angularPairPotential β (Θ i - Θ j)) := by
    rw [angularEβ_eq_pairPotential n hn β (ne_of_gt hβ),
      angularEβ_eq_pairPotential n hn β (ne_of_gt hβ)]
    rw [← mul_sub]
    simp only [Finset.sum_sub_distrib, C]
  have hR : C * (∑ i : Idx n, ∑ j : Idx n,
      (-Causal.h_pot β (Θ i - Θ j) * (v i - v j) + (1 + β) * (v i - v j) ^ 2)) =
      (∑ i : Idx n, angularGrad n β Θ i * v i) +
        C * (1 + β) * (∑ i : Idx n, ∑ j : Idx n, (v i - v j) ^ 2) := by
    simp only [Finset.sum_add_distrib, ← Finset.mul_sum, mul_add]
    rw [pair_linear_eq_angularGrad n hn]
    dsimp only [C]
    ring
  rw [← hE, hR] at hscaled
  have hquad := normalized_pair_sq_le n hn β hβ.le v
  dsimp only [C] at hscaled
  linarith

/-- **The energy gain is controlled by the gradient-square and displacement-square norms.**
Source: arXiv:2410.06833v1, §3.2, `lem: PL.borjan`.
This is the previous actual-energy bound with the coordinate Young inequality. -/
theorem angularEβ_gain_le_squares (n : ℕ) (hn : 0 < n) (β : ℝ) (hβ : 0 < β)
    (Θ v : Idx n → ℝ) :
    angularEβ n β (Θ + v) - angularEβ n β Θ ≤
      (1 / 2 : ℝ) * ∑ i : Idx n, (angularGrad n β Θ i) ^ 2 +
        (3 + 2 * β) * ∑ i : Idx n, (v i) ^ 2 := by
  have h := angularEβ_quadratic_upper n hn β hβ Θ v
  have hlinear := sum_mul_le_half_squares n (angularGrad n β Θ) v
  have hnonneg : 0 ≤ ∑ i : Idx n, (v i) ^ 2 := Finset.sum_nonneg fun _ _ => sq_nonneg _
  nlinarith

/-- **A displacement bound by the true gradient norm gives a uniform PL coefficient.**
Source: arXiv:2410.06833v1, §3.2, `lem: PL.borjan`. The coarse factor
`512n⁵(1+β)` is explicit and independent of the configuration, cap radius,
and slow-manifold threshold. -/
theorem angularEβ_gain_le_gradient (n : ℕ) (hn : 0 < n) (β : ℝ) (hβ : 0 < β)
    (Θ v : Idx n → ℝ)
    (hspread : (∑ i : Idx n, (v i) ^ 2) ≤
      64 * (n : ℝ) ^ 5 * ∑ i : Idx n, (angularGrad n β Θ i) ^ 2) :
    angularEβ n β (Θ + v) - angularEβ n β Θ ≤
      512 * (n : ℝ) ^ 5 * (1 + β) * ∑ i : Idx n, (angularGrad n β Θ i) ^ 2 := by
  have hn1 : (1 : ℝ) ≤ n := by exact_mod_cast (show 1 ≤ n by omega)
  have hn5 : (1 : ℝ) ≤ (n : ℝ) ^ 5 := by
    simpa only [one_pow] using pow_le_pow_left₀ (by norm_num : (0 : ℝ) ≤ 1) hn1 5
  have hG : 0 ≤ ∑ i : Idx n, (angularGrad n β Θ i) ^ 2 :=
    Finset.sum_nonneg fun _ _ => sq_nonneg _
  have hscaled := mul_le_mul_of_nonneg_left hspread (by linarith : 0 ≤ 3 + 2 * β)
  have hcoef : (1 / 2 : ℝ) + (3 + 2 * β) * (64 * (n : ℝ) ^ 5) ≤
      512 * (n : ℝ) ^ 5 * (1 + β) := by
    nlinarith [mul_nonneg hβ.le (by linarith : 0 ≤ (n : ℝ) ^ 5)]
  have hfinal := mul_le_mul_of_nonneg_right hcoef hG
  have h := angularEβ_gain_le_squares n hn β hβ Θ v
  nlinarith

/-- **The actual collapsed configuration has a uniform energy-discrepancy bound.**
Recharting preserves the original energy and gradient. The true force
controls the collapse displacement, and the global Taylor estimate
controls its energy cost. Source: arXiv:2410.06833v1, §3.2, `lem: PL.borjan`. -/
theorem collapsed_angularEβ_gain_le_gradient (n k : ℕ) (hn : 0 < n) (β τ δ α : ℝ)
    (ω : Idx k → ℝ) (p : Idx n → Idx k) (Θ : Idx n → ℝ)
    (hβ : 1 < β) (hτ : 0 < τ) (hτ16 : τ < 1 / 16) (hδ : 0 ≤ δ) (hδ1 : δ < 1)
    (hα : ∀ q q' : Idx k, q ≠ q' → ∀ u v : ℝ,
      1 - 2 * τ ≤ Real.cos (u - ω q) → 1 - 2 * τ ≤ Real.cos (v - ω q') →
        Real.cos (u - v) ≤ α)
    (hγ : 0 < 1 - α - 8 * τ - β⁻¹ * Real.log (2 * (n : ℝ) ^ 2 / τ))
    (hsmall : ∀ q : Idx k, ∀ u v : ℝ, |u - ω q| ≤ π → |v - ω q| ≤ π →
      1 - 2 * τ ≤ Real.cos (u - ω q) → 1 - 2 * τ ≤ Real.cos (v - ω q) →
        |u - v| ≤ (1 / 8) * Real.sqrt ((1 - δ) / (β + 1 / 2)))
    (hcaps : ∀ i : Idx n, 1 - 2 * τ ≤ Real.cos (Θ i - ω (p i))) :
    angularEβ n β (collapseTuple p (chartTuple ω p Θ)) - angularEβ n β Θ ≤
      512 * (n : ℝ) ^ 5 * (1 + β) * ∑ i : Idx n, (angularGrad n β Θ i) ^ 2 := by
  have hown (i : Idx n) : 1 - 2 * τ ≤ Real.cos (chartTuple ω p Θ i - ω (p i)) := by
    rw [chartTuple_cap_cos]
    exact hcaps i
  have hspread := collapsed_spread_le_gradient n k hn β τ δ α ω p (chartTuple ω p Θ)
    hβ hτ hτ16 hδ hδ1 hα hγ hsmall hown (chartTuple_own_chart ω p Θ)
  let v : Idx n → ℝ := fun i => collapseTuple p (chartTuple ω p Θ) i - chartTuple ω p Θ i
  have hgain := angularEβ_gain_le_gradient n hn β (by linarith) (chartTuple ω p Θ) v hspread
  have hU : chartTuple ω p Θ + v = collapseTuple p (chartTuple ω p Θ) := by
    funext i
    simp only [Pi.add_apply, v]
    ring
  rw [hU, angularEβ_chartTuple] at hgain
  simpa only [angularGrad_chartTuple] using hgain

/-- The complete input restrictions of the two global Taylor bounds hold
at `n=2`, `β=1`, with unrestricted actual angle and displacement tuples. -/
example : 0 < 2 ∧ (0 : ℝ) < 1 := ⟨by norm_num, one_pos⟩

/-- The displacement hypothesis of the PL-coefficient consequence is
satisfiable on a nonempty consensus tuple with zero displacement. -/
example : 0 < 2 ∧ (0 : ℝ) < 1 ∧
    (∑ i : Idx 2, ((fun _ : Idx 2 => (0 : ℝ)) i) ^ 2) ≤
      64 * (2 : ℝ) ^ 5 * ∑ i : Idx 2,
        (angularGrad 2 1 (fun _ => (0 : ℝ)) i) ^ 2 := by
  refine ⟨by norm_num, one_pos, ?_⟩
  simp only [zero_pow (by norm_num : 2 ≠ 0), Finset.sum_const_zero]
  positivity

/-- All geometric hypotheses of the actual-collapse energy bound are
realized by the source-compatible three-token, two-cap configuration. -/
example : let τ : ℝ := 1 / 10 ^ 8
    let α : ℝ := 1 - 2 * (1 - 2 * τ) ^ 2
    let ω : Idx 2 → ℝ := ![0, π]
    let Θ : Idx 3 → ℝ := ![0, 1 / 10 ^ 4, π]
    let p : Idx 3 → Idx 2 := ![0, 0, 1]
    0 < 3 ∧ (1 : ℝ) < 1000 ∧ 0 < τ ∧ τ < 1 / 16 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) < 1 ∧
    (∀ q q' : Idx 2, q ≠ q' → ∀ u v : ℝ,
      1 - 2 * τ ≤ Real.cos (u - ω q) → 1 - 2 * τ ≤ Real.cos (v - ω q') →
        Real.cos (u - v) ≤ α) ∧
    0 < 1 - α - 8 * τ - (1000 : ℝ)⁻¹ * Real.log (2 * (3 : ℝ) ^ 2 / τ) ∧
    (∀ q : Idx 2, ∀ u v : ℝ, |u - ω q| ≤ π → |v - ω q| ≤ π →
      1 - 2 * τ ≤ Real.cos (u - ω q) → 1 - 2 * τ ≤ Real.cos (v - ω q) →
        |u - v| ≤ (1 / 8) * Real.sqrt ((1 - 1 / 2) / (1000 + 1 / 2))) ∧
    (∀ i : Idx 3, 1 - 2 * τ ≤ Real.cos (Θ i - ω (p i))) := by
  intro τ α ω Θ p
  obtain ⟨-, hα, hγ, hsmall, -⟩ := circle_witness
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    hα, hγ, hsmall, fun i => ?_⟩
  fin_cases i
  · norm_num [p, Θ, ω, τ]
  · have h := Real.one_sub_sq_div_two_le_cos (x := 1 / 10 ^ 4)
    norm_num [p, Θ, ω, τ] at h ⊢
    linarith
  · norm_num [p, Θ, ω, τ]

end Transformer.Metastability
