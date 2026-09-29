/-
# Metastability — the hypotheses of `metastability` are satisfiable

A concrete instance of every hypothesis of `Metastability.metastability`: `n = 2` tokens at
the point `e_{d+1}` of `𝕊^d`, one cap (`k = 1`, so there is no pair of distinct caps and `α = 0`),
`β = 10⁴`, `ε = 1/100`, and `λ` half of the smaller of the two bounds of `eq: lambda.3`
(`witness_regime`).  The same data show that `IsMetastable` is satisfiable
(`isMetastable_basePoint`).
-/

import Transformer.Metastability.IsMetastable

open scoped BigOperators InnerProductSpace
open Real Set

namespace Transformer
namespace Metastability

/-- With a single cap there are no two distinct caps, so `α(ε) = sSup ∅ = 0`. -/
theorem alphaDist_one (d : ℕ) (w : Idx 1 → SSphere d) (ε : ℝ) : αDist d 1 w ε = 0 := by
  unfold αDist
  have h : {c : ℝ | ∃ i j : Idx 1, i ≠ j ∧
      ∃ x ∈ sphericalCap d (w i) (2 * ε), ∃ y ∈ sphericalCap d (w j) (2 * ε),
        c = inner (𝕜 := ℝ) ((x : EucSpace d)) ((y : EucSpace d))} = ∅ := by
    ext c
    simp only [Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false]
    rintro ⟨i, j, hij, -⟩
    exact hij (Subsingleton.elim i j)
  rw [h, Real.sSup_empty]

/-- **The parameter conditions of `metastability` are satisfiable**: at `n = 2`, `k = 1`,
`β = 10⁴`, `ε = 1/100` and one cap centred at `e_{d+1}`, `γ(β) > 0` and there is a `λ > 0` below
both bounds of `eq: lambda.3`. -/
theorem witness_regime (d : ℕ) : ∃ lam : ℝ, 0 < lam ∧
    0 < γβ 2 10000 (αDist (d + 1) 1 (fun _ : Idx 1 => basePoint d) (1 / 100)) (1 / 100) ∧
    lam < Real.exp ((1 - αDist (d + 1) 1 (fun _ : Idx 1 => basePoint d) (1 / 100)
        + (10000 : ℝ)⁻¹ * Real.log ((10000 - 1) * (1 / 100) /
          (10000 ^ 2 * ((2 : ℕ) : ℝ) ^ 2 * Real.exp 1))) * 10000) *
      (1 - Real.exp (-(γβ 2 10000 (αDist (d + 1) 1 (fun _ : Idx 1 => basePoint d) (1 / 100))
        (1 / 100) * 10000))) ∧
    lam < 1 - αDist (d + 1) 1 (fun _ : Idx 1 => basePoint d) (1 / 100)
      - Real.log (2 * ((2 : ℕ) : ℝ) ^ 2 / (1 - Real.exp (-(lamStar 10000 (1 / 100) * 10000)))) / 10000
      - Real.exp (-(lamStar 10000 (1 / 100) * 10000)) := by
  have hα : αDist (d + 1) 1 (fun _ : Idx 1 => basePoint d) (1 / 100) = 0 :=
    alphaDist_one (d + 1) _ _
  have hγ : 0 < γβ 2 10000 (αDist (d + 1) 1 (fun _ : Idx 1 => basePoint d) (1 / 100)) (1 / 100) := by
    rw [hα]
    unfold γβ
    have h := Real.log_le_sub_one_of_pos (x := 2 * ((2 : ℕ) : ℝ) ^ 2 / (1 / 100)) (by norm_num)
    norm_num at h ⊢
    linarith
  have hexp : Real.exp (-(lamStar 10000 (1 / 100) * 10000)) = 8 * (1 / 100) :=
    exp_neg_lamStar_mul (by norm_num) (by norm_num)
  have hpos₁ : 0 < Real.exp ((1 - αDist (d + 1) 1 (fun _ : Idx 1 => basePoint d) (1 / 100)
        + (10000 : ℝ)⁻¹ * Real.log ((10000 - 1) * (1 / 100) /
          (10000 ^ 2 * ((2 : ℕ) : ℝ) ^ 2 * Real.exp 1))) * 10000) *
      (1 - Real.exp (-(γβ 2 10000 (αDist (d + 1) 1 (fun _ : Idx 1 => basePoint d) (1 / 100))
        (1 / 100) * 10000))) := by
    refine mul_pos (Real.exp_pos _) (sub_pos.2 (Real.exp_lt_one_iff.2 ?_))
    have := mul_pos hγ (by norm_num : (0 : ℝ) < 10000)
    linarith
  have hpos₂ : 0 < 1 - αDist (d + 1) 1 (fun _ : Idx 1 => basePoint d) (1 / 100)
      - Real.log (2 * ((2 : ℕ) : ℝ) ^ 2 / (1 - Real.exp (-(lamStar 10000 (1 / 100) * 10000)))) / 10000
      - Real.exp (-(lamStar 10000 (1 / 100) * 10000)) := by
    rw [hα, hexp]
    have h := Real.log_le_sub_one_of_pos
      (x := 2 * ((2 : ℕ) : ℝ) ^ 2 / (1 - 8 * (1 / 100))) (by norm_num)
    norm_num at h ⊢
    linarith
  set R₁ := Real.exp ((1 - αDist (d + 1) 1 (fun _ : Idx 1 => basePoint d) (1 / 100)
        + (10000 : ℝ)⁻¹ * Real.log ((10000 - 1) * (1 / 100) /
          (10000 ^ 2 * ((2 : ℕ) : ℝ) ^ 2 * Real.exp 1))) * 10000) *
      (1 - Real.exp (-(γβ 2 10000 (αDist (d + 1) 1 (fun _ : Idx 1 => basePoint d) (1 / 100))
        (1 / 100) * 10000))) with hR₁
  set R₂ := 1 - αDist (d + 1) 1 (fun _ : Idx 1 => basePoint d) (1 / 100)
      - Real.log (2 * ((2 : ℕ) : ℝ) ^ 2 / (1 - Real.exp (-(lamStar 10000 (1 / 100) * 10000)))) / 10000
      - Real.exp (-(lamStar 10000 (1 / 100) * 10000)) with hR₂
  exact ⟨min R₁ R₂ / 2, half_pos (lt_min hpos₁ hpos₂), hγ,
    lt_of_lt_of_le (half_lt_self (lt_min hpos₁ hpos₂)) (min_le_left _ _),
    lt_of_lt_of_le (half_lt_self (lt_min hpos₁ hpos₂)) (min_le_right _ _)⟩

/-- The centre of a cap lies in the cap: `e_{d+1} ∈ 𝒮_{e_{d+1}}(c)` for `c ≥ 0`. -/
theorem basePoint_mem_cap (d : ℕ) {c : ℝ} (hc : 0 ≤ c) :
    basePoint d ∈ sphericalCap (d + 1) (basePoint d) c := by
  show (1 : ℝ) - c ≤ ⟪((basePoint d : SSphere (d + 1)) : EucSpace (d + 1)),
    ((basePoint d : SSphere (d + 1)) : EucSpace (d + 1))⟫_ℝ
  rw [inner_sphere_self]
  linarith

/-- **`IsMetastable` is satisfiable**: two tokens at `e_{d+1}` are metastable at `β = 10⁴`, with
one cap, `ε = 1/100` and the rate of `witness_regime`. -/
theorem isMetastable_basePoint (d : ℕ) :
    IsMetastable (d + 1) 2 10000 (fun _ : Idx 2 => basePoint d) := by
  obtain ⟨lam, hlam, hγ, hlam₁, hlam₂⟩ := witness_regime d
  exact isMetastable_of_cover (β := 10000) (ε := 1 / 100) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (fun _ => basePoint d) (k := 1) (by norm_num)
    (fun _ => basePoint d) (fun _ => ⟨0, basePoint_mem_cap d (by norm_num)⟩) hγ hlam hlam₁ hlam₂

/-- **The hypotheses of `metastability` are satisfiable**, together with its conclusion: at
`d = 1`, `n = 2`, `k = 1`, `β = 10⁴`, `ε = 1/100` there is a `λ > 0` below both bounds of
`eq: lambda.3`, and the times `T₁ < T₂` of the theorem exist. -/
example : ∃ (β ε lam : ℝ) (T : MetastabilityTimes β ε
    (αDist 1 1 (fun _ : Idx 1 => basePoint 0) ε) 2 lam), 0 < T.T1 ∧ T.T1 < T.T2 := by
  obtain ⟨lam, hlam, hγ, hlam₁, hlam₂⟩ := witness_regime 0
  obtain ⟨T, -⟩ := metastability (d := 1) (n := 2) (β := 10000) (ε := 1 / 100)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (fun _ => basePoint 0)
    (fun _ : Idx 1 => basePoint 0) (fun _ => ⟨0, basePoint_mem_cap 0 (by norm_num)⟩)
    hγ hlam hlam₁ hlam₂
  exact ⟨10000, 1 / 100, lam, T, T.pos, T.ord⟩

end Metastability
end Transformer
