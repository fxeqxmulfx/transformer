/-
# A single cap gives metastability at an explicit temperature threshold

For fixed token count `n ≥ 1`, height `ε = 1/100` and
`β ≥ 1000 n² + 2` satisfy the regime of the already proved
`isMetastable_of_cover`. With one cap `α = 0`; the elementary bound
`log x ≤ x - 1` gives positive `γ` and a positive second rate bound.
Half the smaller rate bound provides an admissible rate.

The single-cap case uses `alphaDist_one`: there are no pairs of distinct
caps, so the existing separation convention is `sSup ∅ = 0`.
The proof below also bounds `γ` away from zero uniformly in `β`.
This meets the source's asymptotic lower-bound requirement for `γ`.
The dimension restriction is supplied by the final energy-window theorem.

This applies the full theorem, including confinement and collapse for every
solution, rather than a purely geometric definition of metastability.
Source: arXiv:2410.06833v1, §2, `thm: metastability`, `eq: gamma`,
`eq: lambda.3`; §4, `sec: energy.levels`.
-/

import Transformer.Metastability.DirectProofWitness
import Transformer.Metastability.Section4_EnergyGap

open Real

namespace Transformer.Metastability
variable {d n : ℕ}

/-- A uniform lower bound for `γ` in the single-cap regime. In particular,
this supplies the source's `γ = Ω(1)` condition for this choice of cap height.
The threshold is deliberately loose: `log x ≤ x - 1` bounds its logarithmic
term without any limiting argument or numerical approximation.
Source: arXiv:2410.06833v1, §2, `eq: gamma`; §4, `sec: energy.levels`. -/
theorem one_cap_gamma_lower_bound (hn : 1 ≤ n) (β : ℝ)
    (hβ : 1000 * (n : ℝ) ^ 2 + 2 ≤ β) : 1 / 2 < γβ n β 0 (1 / 100) := by
  have hn0 : (0 : ℝ) < n := by exact_mod_cast (show 0 < n by omega)
  have hβ0 : 0 < β := by nlinarith [sq_nonneg (n : ℝ)]
  have hl := Real.log_le_sub_one_of_pos (x := 2 * (n : ℝ) ^ 2 / (1 / 100))
    (by positivity)
  unfold γβ
  norm_num at hl ⊢
  rw [inv_mul_eq_div]
  have hb : Real.log (2 * (n : ℝ) ^ 2 / (1 / 100)) / β < 21 / 50 := by
    apply (div_lt_iff₀ hβ0).mpr
    have ha : 2 * (n : ℝ) ^ 2 / (1 / 100) = 200 * (n : ℝ) ^ 2 := by ring
    rw [ha] at hl ⊢
    nlinarith
  linarith

/-- At the explicit threshold for two tokens, all hypotheses hold and
`γ` is bounded away from zero; this is also a witness for the stronger bound. -/
example : 1 ≤ 2 ∧ 1000 * ((2 : ℕ) : ℝ) ^ 2 + 2 ≤ (4002 : ℝ) ∧
    1 / 2 < γβ 2 4002 0 (1 / 100) := by
  exact ⟨by norm_num, by norm_num,
    one_cap_gamma_lower_bound (by norm_num) 4002 (by norm_num)⟩

/-- A height-`1/100` cap containing every token gives actual metastability
for `β ≥ 1000 n² + 2`. The constant is a sufficient bound, not a sharp one.

The corrected logarithm sign already documented in `metastability` is
used in the first rate bound; all other conditions and conclusions are
those of `IsMetastable`, including the times and both dynamical estimates.
Source: arXiv:2410.06833v1, §2, `thm: metastability`, `eq: gamma`,
`eq: lambda.3`; §4, `sec: energy.levels` (auxiliary regime). -/
theorem one_cap_metastable (hn : 1 ≤ n) (β : ℝ)
    (hβ : 1000 * (n : ℝ) ^ 2 + 2 ≤ β) (X : SphereTuple d n) (w : SSphere d)
    (hcover : ∀ i, X i ∈ sphericalCap d w (1 / 100)) :
    IsMetastable d n β X := by
  have hn0 : (0 : ℝ) < n := by exact_mod_cast (show 0 < n by omega)
  have hβ1 : 1 < β := by nlinarith [sq_nonneg (n : ℝ)]
  have hβ0 : 0 < β := by linarith
  have hγ : 0 < γβ n β 0 (1 / 100) :=
    lt_trans (by norm_num) (one_cap_gamma_lower_bound hn β hβ)
  have hexp : Real.exp (-(lamStar β (1 / 100) * β)) = 8 * (1 / 100) :=
    exp_neg_lamStar_mul hβ0 (by norm_num)
  let R₁ := Real.exp ((1 + β⁻¹ * Real.log ((β - 1) * (1 / 100) /
      (β ^ 2 * (n : ℝ) ^ 2 * Real.exp 1))) * β) *
      (1 - Real.exp (-(γβ n β 0 (1 / 100) * β)))
  let R₂ := 1 - Real.log (2 * (n : ℝ) ^ 2 /
      (1 - Real.exp (-(lamStar β (1 / 100) * β)))) / β -
      Real.exp (-(lamStar β (1 / 100) * β))
  have hR₁ : 0 < R₁ := by
    apply mul_pos (Real.exp_pos _) (sub_pos.mpr (Real.exp_lt_one_iff.mpr ?_))
    have := mul_pos hγ hβ0
    linarith
  have hR₂ : 0 < R₂ := by
    dsimp [R₂]
    rw [hexp]
    have hl := Real.log_le_sub_one_of_pos (x := 2 * (n : ℝ) ^ 2 / (1 - 8 * (1 / 100)))
      (by positivity)
    norm_num at hl ⊢
    have hb : Real.log (2 * (n : ℝ) ^ 2 / (23 / 25)) / β < 23 / 25 := by
      apply (div_lt_iff₀ hβ0).mpr
      have ha : 2 * (n : ℝ) ^ 2 / (23 / 25) = 50 / 23 * (n : ℝ) ^ 2 := by ring
      rw [ha] at hl ⊢
      nlinarith
    linarith
  let lam := min R₁ R₂ / 2
  have hlam : 0 < lam := half_pos (lt_min hR₁ hR₂)
  have hlam₁ : lam < R₁ :=
    lt_of_lt_of_le (half_lt_self (lt_min hR₁ hR₂)) (min_le_left _ _)
  have hlam₂ : lam < R₂ :=
    lt_of_lt_of_le (half_lt_self (lt_min hR₁ hR₂)) (min_le_right _ _)
  apply isMetastable_of_cover hβ1 (by norm_num) (by norm_num) hn X (k := 1) hn
    (fun _ => w) (fun i => ⟨0, hcover i⟩) (lam := lam)
  · rw [alphaDist_one]
    exact hγ
  · exact hlam
  · simpa only [alphaDist_one, sub_zero] using hlam₁
  · simpa only [alphaDist_one, sub_zero] using hlam₂

/-- The hypotheses of the one-cap regime have a concrete witness:
two tokens at `e₂`, `β = 10000`, and height `1/100`. -/
example : 1 ≤ 2 ∧ 1000 * ((2 : ℕ) : ℝ) ^ 2 + 2 ≤ (10000 : ℝ) ∧
    (∀ i : Idx 2, (fun _ : Idx 2 => basePoint 1) i ∈
      sphericalCap 2 (basePoint 1) (1 / 100)) ∧
    IsMetastable 2 2 10000 (fun _ : Idx 2 => basePoint 1) := by
  refine ⟨by norm_num, by norm_num, ?_, ?_⟩
  · exact fun _ => basePoint_mem_cap 1 (by norm_num)
  · exact one_cap_metastable (by norm_num) 10000 (by norm_num) _ _
      (fun _ => basePoint_mem_cap 1 (by norm_num))

/-- Sufficiently high energy alone forces metastability at the explicit
large-temperature threshold. The cap is centred at token zero, which exists
because `n ≥ 1`; no cap data are assumed in the statement.
Source: arXiv:2410.06833v1, §4, `sec: energy.levels`. -/
theorem isMetastable_of_high_energy (hn : 1 ≤ n) (β : ℝ)
    (hβ : 1000 * (n : ℝ) ^ 2 + 2 ≤ β) (X : SphereTuple d n)
    (hE : 1 / (2 * β) - (Real.exp β - Real.exp (β * (1 - (1 / 100)))) /
      (2 * β * Real.exp β * (n : ℝ) ^ 2) < Eβ d n β X) :
    IsMetastable d n β X := by
  have hβ0 : 0 < β := by nlinarith [sq_nonneg (n : ℝ)]
  let j : Idx n := ⟨0, by omega⟩
  exact one_cap_metastable hn β hβ X (X j)
    (high_energy_mem_cap β (1 / 100) hβ0 hn X j hE)

/-- The high-energy hypothesis is satisfied by actual consensus, and hence
the implication to metastability is nonvacuous at `β = 10000`. -/
example : ∃ X : SphereTuple 2 2,
    1000 * ((2 : ℕ) : ℝ) ^ 2 + 2 ≤ (10000 : ℝ) ∧
    1 / (2 * (10000 : ℝ)) -
      (Real.exp 10000 - Real.exp (10000 * (1 - (1 / 100)))) /
        (2 * 10000 * Real.exp 10000 * ((2 : ℕ) : ℝ) ^ 2) < Eβ 2 2 10000 X := by
  refine ⟨fun _ => basePoint 1, by norm_num, ?_⟩
  rw [energy_consensus (by norm_num) 10000 (by norm_num)]
  have := cap_energy_tolerance_pos (n := 2) 10000 (1 / 100)
    (by norm_num) (by norm_num) (by norm_num)
  linarith

end Transformer.Metastability
