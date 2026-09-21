/-
# Metastability — `claim: 1` of the PL inequality (§3.2 of 2410.06833v1)

The comparison of the partial derivatives of `𝖤_β` inside one cluster that
the proof of `lem: PL.borjan` rests on (`eq: Ht.third.lb`), stated at the
configuration where the source uses it, with its hypotheses witnessed.
-/

import Transformer.Metastability.OttoReznikoff
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds

open Real

namespace Transformer
namespace Metastability

/-- **Claim (claim: 1), equation (eq: Ht.third.lb).**

Let `θ_{i_1} < ⋯ < θ_{i_r}` be the particles of one cap `𝒮_q(2τ)`, `1 ≤ r < n`,
at a configuration off the slow manifold through that cap (`eq: cond.sine`).
Then

  `max_ℓ |∂_{θ_{i_ℓ}} 𝖤_β(Θ)| ≤ (e/2) max{ |∂_{θ_{i_1}} 𝖤_β(Θ)|, |∂_{θ_{i_r}} 𝖤_β(Θ)| }`.

**The setting, as the source uses it.**  The claim sits inside the proof of
`lem: PL.borjan` and is about `Θ(t)`, `t ∈ [0, T)`, on the flow started at a
`(β, τ)`-separated configuration.  Its proof uses of `Θ(t)` exactly the
following, which are the hypotheses here, at one configuration `Θ`:

* every particle lies in some cap `𝒮_p(2τ)` (`hcaps`, from `eq: stick`), and
  `α` bounds the cosine between points of distinct caps (`hα`, `eq:
  alpha.dist.2` — the maximum itself satisfies this, and every condition below
  only weakens as `α` falls);
* `γ > 0` (`d: condition_ineq_2`), `eq: tau.small` with its `δ`, and `λ` below
  the bound `eq: lambda.3` of `rem: lambda.gamma` (with `τ` for `ε`, as in §3);
* `q` is a cap satisfying `eq: cond.sine`, which is what `Θ(t) ∉ 𝒩_β` provides;
* the particles of `𝒮_q(2τ)` are relabelled `idx 0, …, idx (r-1)` with
  increasing angles.

The torus is carried by real representatives.  The relabelling
`θ_1 < ⋯ < θ_r` and the distances of `eq: tau.small` are those of the chart
centred at `ω_q`, so the members of a cap are taken within `π` of its centre
(`hchart`, and the same restriction inside `hsmall`); without it a particle and
its translate by `2π` would both sit in the cap, and `eq: tau.small` could not
hold.  The asymptotic requirements `γ(β) = Ω(1)` and `λ(β) = Ω(1)` concern a
family in `β`, not one configuration, and are dropped.

Not proved here.  The source's own last step — "using the fact that
`Θ(t) ∉ 𝒩_β`, `max_ℓ |∂_{θ_ℓ} 𝖤_β| ≥ 2e(1+β)e^{-(1-α)β}`" — is asserted
without derivation.

Source: arXiv:2410.06833v1, §3.2, proof of `lem: PL.borjan`, `claim: 1`,
`eq: Ht.third.lb`. -/
theorem claim_one (n : ℕ) (β τ δ α lam : ℝ) (k : ℕ) (ω : Idx k → ℝ) (Θ : Idx n → ℝ)
    (hβ : 1 < β) (hτ : 0 < τ) (hτ16 : τ < 1 / 16) (hk : k ≤ n)
    (hcaps : ∀ i : Idx n, ∃ p : Idx k, 1 - 2 * τ ≤ Real.cos (Θ i - ω p))
    (hα : ∀ p p' : Idx k, p ≠ p' → ∀ u v : ℝ,
      1 - 2 * τ ≤ Real.cos (u - ω p) → 1 - 2 * τ ≤ Real.cos (v - ω p') →
        Real.cos (u - v) ≤ α)
    (hγ : 0 < 1 - α - 8 * τ - β⁻¹ * Real.log (2 * (n : ℝ) ^ 2 / τ))
    (hsmall : ∀ p : Idx k, ∀ u v : ℝ, |u - ω p| ≤ π → |v - ω p| ≤ π →
      1 - 2 * τ ≤ Real.cos (u - ω p) → 1 - 2 * τ ≤ Real.cos (v - ω p) →
        |u - v| ≤ (1 / 8) * Real.sqrt ((1 - δ) / (β + 1 / 2)))
    (hδ : 8 * (1 + β) * Real.exp (-((1 - α) * β)) * Real.exp (-(1 / 2 : ℝ)) < δ)
    (hδ1 : δ < 1) (hlam : 0 < lam)
    (hlam3 : lam < min
      (Real.exp ((1 - α - β⁻¹ * Real.log ((β - 1) * τ / (β ^ 2 * (n : ℝ) ^ 2 * Real.exp 1)))
          * β)
        * (1 - Real.exp (-((1 - α - 8 * τ - β⁻¹ * Real.log (2 * (n : ℝ) ^ 2 / τ)) * β))))
      (1 - α - β⁻¹ * Real.log (2 * (n : ℝ) ^ 2
          / (1 - Real.exp (-(β⁻¹ * Real.log (1 / (8 * τ)) * β))))
        - Real.exp (-(β⁻¹ * Real.log (1 / (8 * τ)) * β))))
    (q : Idx k) (r : ℕ) (idx : Fin r → Idx n) (hr : 0 < r) (hrn : r < n)
    (hmono : StrictMono (Θ ∘ idx))
    (hchart : ∀ j : Fin r, |Θ (idx j) - ω q| ≤ π)
    (hmem : ∀ i : Idx n, 1 - 2 * τ ≤ Real.cos (Θ i - ω q) ↔ i ∈ Set.range idx)
    (hsine : ∃ a b : Fin r, Real.exp (-(lam * β / 2)) ≤ |Θ (idx a) - Θ (idx b)|) :
    ∀ l : Fin r, |angularGrad n β Θ (idx l)|
      ≤ (Real.exp 1 / 2) * max |angularGrad n β Θ (idx ⟨0, hr⟩)|
          |angularGrad n β Θ (idx ⟨r - 1, by omega⟩)| := by
  sorry

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

/-- The hypotheses of `claim_one` are satisfiable: three particles on the
circle, two of them `10⁻⁴` apart in the cap around `0` and one at `π`, with
`β = 1000`, `τ = 10⁻⁸`, `δ = 1/2`, `λ = 1` and `α` the bound
`1 - 2(1 - 2τ)²` on the cosine across the two caps. -/
example : let τ : ℝ := 1 / 10 ^ 8
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
    (∃ a b : Fin 2, Real.exp (-(1 * 1000 / 2)) ≤ |Θ (idx a) - Θ (idx b)|) := by
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
  refine ⟨?_, ?_, by linarith, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
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

end Metastability
end Transformer
