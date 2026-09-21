/-
# Metastability — a separated configuration on the circle (§3.2 of 2410.06833v1)

Three particles on `𝕋` meeting every condition of §3.2 at once, and the chart
estimate for the caps `𝒮_q(τ)` it rests on.  The hypotheses of
`Metastability.PL_borjan` and `Metastability.claim_one` are witnessed here.
-/

import Transformer.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds

open Real

namespace Transformer
namespace Metastability

/-- A point of a `2τ`-cap within `π` of its centre is within `4√τ` of it:
the chart estimate behind `eq: tau.small`.  Source: arXiv:2410.06833v1, §3.2,
`eq: cones_2`. -/
theorem sq_le_of_mem_cap (τ u : ℝ) (hu : |u| ≤ π) (hc : 1 - 2 * τ ≤ Real.cos u) :
    u ^ 2 ≤ 16 * τ := by
  have h := Real.cos_le_one_sub_mul_cos_sq hu
  have hπ0 : 0 < π ^ 2 := by positivity
  have hπ16 : π ^ 2 ≤ 16 := by nlinarith [Real.pi_le_four, Real.pi_pos]
  have h2 : 2 / π ^ 2 * u ^ 2 ≤ 2 * τ := by linarith
  rw [div_mul_eq_mul_div, div_le_iff₀ hπ0] at h2
  have hτ : 0 ≤ τ := by nlinarith [sq_nonneg u]
  nlinarith [mul_le_mul_of_nonneg_left hπ16 hτ]

/-- The hypotheses of `sq_le_of_mem_cap` are satisfiable: the centre itself. -/
example : (0 : ℝ) ^ 2 ≤ 16 * 0 :=
  sq_le_of_mem_cap 0 0 (by simp [Real.pi_pos.le]) (by simp)

/-- **A separated configuration on the circle.**  Three particles, two of them
`10⁻⁴` apart in the cap around `0` and one at `π`, with `β = 1000`,
`τ = 10⁻⁸`, `δ = 1/2`, `λ = 1` and `α = 1 - 2(1 - 2τ)²`, the bound on the
cosine across the two caps, satisfy every condition §3.2 places on a
configuration: the caps (`eq: cones_2`), `alpha.dist.2`, `γ > 0`
(`d: condition_ineq_2`), `eq: tau.small`, the range of `δ`, `eq: lambda.3`,
and `eq: cond.sine` in the cap around `0`, whose particles are ordered.

Source: arXiv:2410.06833v1, §3.2, `hyp: init.theta`, `lem: PL.borjan`. -/
theorem circle_witness : let τ : ℝ := 1 / 10 ^ 8
    let α : ℝ := 1 - 2 * (1 - 2 * τ) ^ 2
    let ω : Idx 2 → ℝ := ![0, π]
    let Θ : Idx 3 → ℝ := ![0, 1 / 10 ^ 4, π]
    let idx : Fin 2 → Idx 3 := ![0, 1]
    (∀ i : Idx 3, ∃ p : Idx 2, 1 - 2 * τ ≤ Real.cos (Θ i - ω p)) ∧
    (∀ p p' : Idx 2, p ≠ p' → ∀ u v : ℝ,
      1 - 2 * τ ≤ Real.cos (u - ω p) → 1 - 2 * τ ≤ Real.cos (v - ω p') →
        Real.cos (u - v) ≤ α) ∧
    0 < 1 - α - 8 * τ - (1000 : ℝ)⁻¹ * Real.log (2 * ((3 : ℕ) : ℝ) ^ 2 / τ) ∧
    (∀ p : Idx 2, ∀ u v : ℝ, |u - ω p| ≤ π → |v - ω p| ≤ π →
      1 - 2 * τ ≤ Real.cos (u - ω p) → 1 - 2 * τ ≤ Real.cos (v - ω p) →
        |u - v| ≤ (1 / 8) * Real.sqrt ((1 - 1 / 2) / (1000 + 1 / 2))) ∧
    8 * (1 + 1000) * Real.exp (-((1 - α) * 1000)) * Real.exp (-(1 / 2 : ℝ)) < 1 / 2 ∧
    (1 : ℝ) < min
      (Real.exp ((1 - α - (1000 : ℝ)⁻¹ * Real.log ((1000 - 1) * τ
          / (1000 ^ 2 * ((3 : ℕ) : ℝ) ^ 2 * Real.exp 1))) * 1000)
        * (1 - Real.exp (-((1 - α - 8 * τ
            - (1000 : ℝ)⁻¹ * Real.log (2 * ((3 : ℕ) : ℝ) ^ 2 / τ)) * 1000))))
      (1 - α - (1000 : ℝ)⁻¹ * Real.log (2 * ((3 : ℕ) : ℝ) ^ 2
          / (1 - Real.exp (-((1000 : ℝ)⁻¹ * Real.log (1 / (8 * τ)) * 1000))))
        - Real.exp (-((1000 : ℝ)⁻¹ * Real.log (1 / (8 * τ)) * 1000))) ∧
    StrictMono (Θ ∘ idx) ∧
    (∀ j : Fin 2, |Θ (idx j) - ω 0| ≤ π) ∧
    (∀ i : Idx 3, 1 - 2 * τ ≤ Real.cos (Θ i - ω 0) ↔ i ∈ Set.range idx) ∧
    (∃ a b : Fin 2, Real.exp (-(1 * 1000 / 2)) ≤ |Θ (idx a) - Θ (idx b)|) ∧
    (∀ i : Idx 3, ∃ p : Idx 2, 1 - τ ≤ Real.cos (Θ i - ω p)) := by
  intro τ α ω Θ idx
  have hτ : τ = 1 / 10 ^ 8 := rfl
  have hα : α = 1 - 2 * (1 - 2 * τ) ^ 2 := rfl
  have hπ := Real.pi_pos
  have hπ4 := Real.pi_le_four
  -- `1 - α ≥ 1`, and `γ` is large
  have h1α : 1 ≤ 1 - α := by rw [hα, hτ]; norm_num
  have hlog : Real.log (2 * ((3 : ℕ) : ℝ) ^ 2 / τ) < 100 := by
    rw [Real.log_lt_iff_lt_exp (by rw [hτ]; norm_num)]
    refine lt_of_lt_of_le ?_ (Real.pow_div_factorial_le_exp 100 (by norm_num) 10)
    rw [hτ]; norm_num [Nat.factorial]
  have hlog0 : 0 ≤ Real.log (2 * ((3 : ℕ) : ℝ) ^ 2 / τ) :=
    Real.log_nonneg (by rw [hτ]; norm_num)
  have hγ : 1 / 2 ≤ 1 - α - 8 * τ - (1000 : ℝ)⁻¹ * Real.log (2 * ((3 : ℕ) : ℝ) ^ 2 / τ) := by
    have : 8 * τ ≤ 1 / 100 := by rw [hτ]; norm_num
    nlinarith
  have hexp1000 : (1000 : ℝ) ^ 2 / 2 ≤ Real.exp 1000 := by
    simpa [Nat.factorial] using Real.pow_div_factorial_le_exp 1000 (by norm_num) 2
  refine ⟨?_, ?_, by linarith, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro i
    fin_cases i
    · exact ⟨0, by simp [Θ, ω]; rw [hτ]; norm_num⟩
    · refine ⟨0, ?_⟩
      simp only [Θ, ω]
      simp
      have := Real.one_sub_sq_div_two_le_cos (x := 1 / 10 ^ 4)
      rw [hτ]; norm_num at this ⊢; linarith
    · exact ⟨1, by simp [Θ, ω]; rw [hτ]; norm_num⟩
  · intro p p' hpp' u v hu hv
    have hc : 0 < 1 - 2 * τ := by rw [hτ]; norm_num
    have key : ∀ a b : ℝ, 1 - 2 * τ ≤ Real.cos a → 1 - 2 * τ ≤ -Real.cos b →
        Real.cos (a - b) ≤ α := by
      intro a b ha hb
      rw [Real.cos_sub, hα]
      nlinarith [Real.sin_sq_add_cos_sq a, Real.sin_sq_add_cos_sq b,
        sq_nonneg (Real.sin a - Real.sin b), mul_nonneg (sub_nonneg.mpr ha) (sub_nonneg.mpr hb),
        mul_le_mul ha ha hc.le (hc.le.trans ha), mul_le_mul hb hb hc.le (hc.le.trans hb)]
    fin_cases p <;> fin_cases p'
    · exact absurd rfl hpp'
    · simp only [ω] at hu hv
      simp only [Fin.zero_eta, Matrix.cons_val_zero, sub_zero, Fin.mk_one,
        Matrix.cons_val_one, Matrix.cons_val_fin_one, Real.cos_sub_pi] at hu hv
      exact key u v hu hv
    · simp only [ω] at hu hv
      simp only [Fin.zero_eta, Matrix.cons_val_zero, sub_zero, Fin.mk_one,
        Matrix.cons_val_one, Matrix.cons_val_fin_one, Real.cos_sub_pi] at hu hv
      rw [← Real.cos_neg, neg_sub]
      exact key v u hv hu
    · exact absurd rfl hpp'
  · intro p u v hu hv hcu hcv
    have hX : (1 / 8 : ℝ) * Real.sqrt ((1 - 1 / 2) / (1000 + 1 / 2))
        = Real.sqrt ((1 / 64) * ((1 - 1 / 2) / (1000 + 1 / 2))) := by
      rw [Real.sqrt_mul (by norm_num), show (1 / 64 : ℝ) = (1 / 8) ^ 2 by norm_num,
        Real.sqrt_sq (by norm_num)]
    rw [hX]
    refine Real.abs_le_sqrt ?_
    have hu2 := sq_le_of_mem_cap τ _ hu hcu
    have hv2 := sq_le_of_mem_cap τ _ hv hcv
    have : (u - v) ^ 2 ≤ 2 * ((u - ω p) ^ 2 + (v - ω p) ^ 2) := by
      nlinarith [sq_nonneg (u - ω p + (v - ω p))]
    rw [hτ] at hu2 hv2
    nlinarith
  · have h1 : Real.exp (-((1 - α) * 1000)) ≤ Real.exp (-1000) :=
      Real.exp_le_exp.mpr (by nlinarith)
    have h2 : Real.exp (-1000) ≤ 2 / 1000 ^ 2 := by
      rw [Real.exp_neg, inv_le_comm₀ (Real.exp_pos _) (by norm_num)]
      linarith
    have h3 : Real.exp (-(1 / 2 : ℝ)) ≤ 1 := Real.exp_le_one_iff.mpr (by norm_num)
    have h4 := Real.exp_pos (-((1 - α) * 1000))
    have h5 := Real.exp_pos (-(1 / 2 : ℝ))
    nlinarith [mul_le_mul h1 h3 h5.le (Real.exp_pos _).le]
  · have hexp500 : (2 : ℝ) ≤ Real.exp 500 := by
      linarith [Real.add_one_le_exp (500 : ℝ)]
    refine lt_min ?_ ?_
    · have hsmall : Real.log ((1000 - 1) * τ / (1000 ^ 2 * ((3 : ℕ) : ℝ) ^ 2 * Real.exp 1)) ≤ 0 := by
        refine Real.log_nonpos (by rw [hτ]; positivity) ?_
        rw [div_le_one (by positivity), hτ]
        nlinarith [Real.add_one_le_exp (1 : ℝ)]
      have hE : 1001 ≤ Real.exp ((1 - α - (1000 : ℝ)⁻¹ * Real.log ((1000 - 1) * τ
          / (1000 ^ 2 * ((3 : ℕ) : ℝ) ^ 2 * Real.exp 1))) * 1000) := by
        have := Real.add_one_le_exp ((1 - α - (1000 : ℝ)⁻¹ * Real.log ((1000 - 1) * τ
          / (1000 ^ 2 * ((3 : ℕ) : ℝ) ^ 2 * Real.exp 1))) * 1000)
        nlinarith
      have hF : Real.exp (-((1 - α - 8 * τ
            - (1000 : ℝ)⁻¹ * Real.log (2 * ((3 : ℕ) : ℝ) ^ 2 / τ)) * 1000)) ≤ 1 / 2 := by
        calc _ ≤ Real.exp (-500) := Real.exp_le_exp.mpr (by nlinarith)
          _ ≤ 1 / 2 := by
            rw [Real.exp_neg, inv_le_comm₀ (Real.exp_pos _) (by norm_num)]
            linarith
      nlinarith
    · have hL : Real.exp (-((1000 : ℝ)⁻¹ * Real.log (1 / (8 * τ)) * 1000)) = 8 * τ := by
        rw [show (1000 : ℝ)⁻¹ * Real.log (1 / (8 * τ)) * 1000 = Real.log (1 / (8 * τ)) by ring,
          Real.exp_neg, Real.exp_log (by rw [hτ]; norm_num)]
        rw [hτ]; norm_num
      rw [hL]
      have hlog18 : Real.log (2 * ((3 : ℕ) : ℝ) ^ 2 / (1 - 8 * τ)) ≤ 18 := by
        refine (Real.log_le_sub_one_of_pos (by rw [hτ]; norm_num)).trans ?_
        rw [hτ]; norm_num
      have h1α' : 1 - α = 2 * (1 - 2 * τ) ^ 2 := by rw [hα]; ring
      rw [h1α', hτ] at *
      norm_num at hlog18 ⊢
      linarith
  · refine Fin.strictMono_iff_lt_succ.mpr fun i => ?_
    fin_cases i
    simp [Θ, idx]
  · intro j
    fin_cases j <;> simp [Θ, ω, idx] <;> linarith [Real.two_le_pi]
  · intro i
    fin_cases i
    · simp [Θ, ω, idx]; rw [hτ]; norm_num
    · simp only [Θ, ω, idx]
      have := Real.one_sub_sq_div_two_le_cos (x := 1 / 10 ^ 4)
      constructor
      · intro; exact ⟨1, rfl⟩
      · intro; simp; rw [hτ]; norm_num at this ⊢; linarith
    · simp only [Θ, ω, idx]
      constructor
      · intro h; simp at h; rw [hτ] at h; norm_num at h
      · rintro ⟨j, hj⟩; fin_cases j <;> simp at hj
  · refine ⟨1, 0, ?_⟩
    simp only [Θ, idx]
    simp
    have h500 : (500 : ℝ) ^ 2 / 2 ≤ Real.exp 500 := by
      simpa [Nat.factorial] using Real.pow_div_factorial_le_exp 500 (by norm_num) 2
    rw [show -(1000 / 2 : ℝ) = -500 by norm_num, Real.exp_neg,
      inv_le_comm₀ (Real.exp_pos _) (by norm_num)]
    norm_num
    linarith
  · intro i
    fin_cases i
    · exact ⟨0, by simp [Θ, ω]; rw [hτ]; norm_num⟩
    · refine ⟨0, ?_⟩
      simp only [Θ, ω]
      simp
      have := Real.one_sub_sq_div_two_le_cos (x := 1 / 10 ^ 4)
      rw [hτ]; norm_num at this ⊢; linarith
    · exact ⟨1, by simp [Θ, ω]; rw [hτ]; norm_num⟩

end Metastability
end Transformer
