import Transformer.AdamBeyond.Section3_StochCoins

/-
# Adam and beyond — §3: the expectations in the lemma of Theorem 3

`m_n` and `v_n` are functions of the coins `b_1, …, b_n` (`coinM`, `coinV`),
and the pointwise bounds of `Section3_StochBound` integrate to:

* `E[T₁] ≤ 1/√(1-β₂)`, `stoch_int_T1`;
* `E[β₁ m_n/√(β₂ v_n + 1 - β₂)] ≤ β₁ K p Σ_{j≤n} γ^{n-j}`, `stoch_int_T2`;
* `E[β₂ v_n + 1 - β₂] ≤ 1 + p C²`, `stoch_int_v`, whence
  `E[1/√(β₂ v_n + 1 - β₂)] ≥ 1/√a` for every `a ≥ 1 + p C²`, `stoch_int_inv`,
  by the tangent `inv_sqrt_ge` in place of Jensen's inequality.

Source: arXiv:1904.09237, Appendix, proof of Theorem 3, the lemma: the bounds
on `E[T₁]`, `E[T₂]` and `E[T₃]`.
-/

open MeasureTheory ProbabilityTheory Finset Function

namespace Transformer
namespace AdamBeyond

open AMSGrad

/-- `m_n` as a function of the coins. -/
noncomputable def coinM (C β₁ β₂ : ℝ) (α : ℕ → ℝ) (n : ℕ) (u : ℕ → Bool) : ℝ :=
  (stochSetup C β₁ β₂ α (fun j (u : ℕ → Bool) => u j) u).m adamRule n 0

/-- `v_n` as a function of the coins. -/
noncomputable def coinV (C β₁ β₂ : ℝ) (α : ℕ → ℝ) (n : ℕ) (u : ℕ → Bool) : ℝ :=
  (stochSetup C β₁ β₂ α (fun j (u : ℕ → Bool) => u j) u).v adamRule n 0

variable {C β₁ β₂ : ℝ} {α : ℕ → ℝ}

theorem coinM_dep (n : ℕ) : DependsOn (coinM C β₁ β₂ α n) (Icc 1 n : Set ℕ) := fun x y h => by
  rw [coinM, coinM, stoch_m_eq, stoch_m_eq]
  congr 1
  exact dependsOn_sum n (fun j c => β₁ ^ (n - j) * coinGrad C c) h

theorem coinV_dep (n : ℕ) : DependsOn (coinV C β₁ β₂ α n) (Icc 1 n : Set ℕ) := fun x y h => by
  rw [coinV, coinV, stoch_v_eq, stoch_v_eq]
  congr 1
  exact dependsOn_sum n (fun j c => β₂ ^ (n - j) * coinGrad C c ^ 2) h

/-- A function of `m_n` and `v_n` depends on the coins `b_1, …, b_n`. -/
theorem coinMV_dep (n : ℕ) (g : ℝ → ℝ → ℝ) :
    DependsOn (fun u => g (coinM C β₁ β₂ α n u) (coinV C β₁ β₂ α n u)) (Icc 1 n : Set ℕ) :=
  dependsOn_comp₂ (coinM_dep n) (coinV_dep n) g

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
  {b : ℕ → Ω → Bool} {p : ℝ}

/-- `E[T₁] ≤ 1/√(1-β₂)`.  arXiv:1904.09237, Appendix, proof of Theorem 3,
(eq:T_1-bound). -/
theorem stoch_int_T1 (hb : IsBernoulliSeq μ b p) (hC : 1 ≤ C) (hβ₁ : 0 ≤ β₁) (hβ₁' : β₁ ≤ 1)
    (hβ₂ : 0 ≤ β₂) (hβ₂' : β₂ < 1) (n : ℕ) :
    ∫ ω, (β₁ * coinM C β₁ β₂ α n (fun j => b j ω) + (1 - β₁) * C) /
        Real.sqrt (β₂ * coinV C β₁ β₂ α n (fun j => b j ω) + (1 - β₂) * C ^ 2) ∂μ ≤
      1 / Real.sqrt (1 - β₂) := by
  have h := integral_mono (coin_integrable hb (coinMV_dep (C := C) (β₁ := β₁) (β₂ := β₂)
    (α := α) n fun m v => (β₁ * m + (1 - β₁) * C) / Real.sqrt (β₂ * v + (1 - β₂) * C ^ 2)))
    (integrable_const (1 / Real.sqrt (1 - β₂))) fun ω =>
      stoch_T1_le hC hβ₁ hβ₂ hβ₂' (stoch_m_le hC hβ₁ hβ₁' n) (stoch_v_nonneg hβ₂ hβ₂'.le n)
  simpa using h

/-- `E[β₁ m_n/√(β₂ v_n + 1 - β₂)] ≤ β₁ K p Σ_{j≤n} γ^{n-j}`, with
`K = 2(1-β₁)/√(β₂(1-β₂))` and `γ = β₁/√β₂`.  arXiv:1904.09237, Appendix,
proof of Theorem 3, (eq:T_2-bound), by another argument. -/
theorem stoch_int_T2 (hb : IsBernoulliSeq μ b p) (hp : 0 ≤ p) (hC : 1 ≤ C) (hβ₁ : 0 ≤ β₁)
    (hβ₁' : β₁ ≤ 1) (hβ₂ : 0 < β₂) (hβ₂' : β₂ < 1) (n : ℕ) :
    ∫ ω, β₁ * coinM C β₁ β₂ α n (fun j => b j ω) /
        Real.sqrt (β₂ * coinV C β₁ β₂ α n (fun j => b j ω) + (1 - β₂)) ∂μ ≤
      β₁ * (2 * (1 - β₁) / Real.sqrt (β₂ * (1 - β₂)) *
        (p * ∑ j ∈ Icc 1 n, (β₁ / Real.sqrt β₂) ^ (n - j))) := by
  have hS := coin_integrable hb
    (dependsOn_sum n fun j c => (β₁ / Real.sqrt β₂) ^ (n - j) * coinInd c)
  have h := integral_mono (coin_integrable hb (coinMV_dep (C := C) (β₁ := β₁) (β₂ := β₂)
    (α := α) n fun m v => β₁ * m / Real.sqrt (β₂ * v + (1 - β₂))))
    ((hS.const_mul (2 * (1 - β₁) / Real.sqrt (β₂ * (1 - β₂)))).const_mul β₁) fun ω =>
      stoch_T2_le hC hβ₁ hβ₁' hβ₂ hβ₂' n
  refine h.trans (le_of_eq ?_)
  rw [integral_const_mul, integral_const_mul, coin_integral_sum hb hp]

/-- `E[β₂ v_n + 1 - β₂] ≤ 1 + p C²`.  arXiv:1904.09237, Appendix, proof of
Theorem 3, the bound on `E[v_{t-1}]`. -/
theorem stoch_int_v (hb : IsBernoulliSeq μ b p) (hp : 0 ≤ p) (hβ₂ : 0 ≤ β₂) (hβ₂' : β₂ < 1)
    (n : ℕ) :
    ∫ ω, (β₂ * coinV C β₁ β₂ α n (fun j => b j ω) + (1 - β₂)) ∂μ ≤ 1 + p * C ^ 2 := by
  set S2 := ∑ j ∈ Icc 1 n, β₂ ^ (n - j) with hS2
  have hS := coin_integrable hb (dependsOn_sum n fun j c => β₂ ^ (n - j) * C ^ 2 * coinInd c)
  have hpt : ∀ ω, β₂ * coinV C β₁ β₂ α n (fun j => b j ω) + (1 - β₂) ≤
      (β₂ * (1 - β₂) * S2 + (1 - β₂)) +
        β₂ * (1 - β₂) * ∑ j ∈ Icc 1 n, β₂ ^ (n - j) * C ^ 2 * coinInd (b j ω) := by
    intro ω
    have h1 : coinV C β₁ β₂ α n (fun j => b j ω) ≤
        (1 - β₂) * ∑ j ∈ Icc 1 n, β₂ ^ (n - j) * (1 + C ^ 2 * coinInd (b j ω)) :=
      stoch_v_le_ind hβ₂ hβ₂'.le n
    have e : ∑ j ∈ Icc 1 n, β₂ ^ (n - j) * (1 + C ^ 2 * coinInd (b j ω)) =
        S2 + ∑ j ∈ Icc 1 n, β₂ ^ (n - j) * C ^ 2 * coinInd (b j ω) := by
      rw [hS2, ← sum_add_distrib]
      exact sum_congr rfl fun _ _ => by ring
    rw [e] at h1
    nlinarith [mul_le_mul_of_nonneg_left h1 hβ₂]
  have hR : Integrable (fun ω => (β₂ * (1 - β₂) * S2 + (1 - β₂)) +
      β₂ * (1 - β₂) * ∑ j ∈ Icc 1 n, β₂ ^ (n - j) * C ^ 2 * coinInd (b j ω)) μ :=
    (integrable_const _).add (hS.const_mul _)
  have h := integral_mono (coin_integrable hb (coinMV_dep (C := C) (β₁ := β₁) (β₂ := β₂)
    (α := α) n fun _ v => β₂ * v + (1 - β₂))) hR hpt
  have hc : ∫ _ : Ω, (β₂ * (1 - β₂) * S2 + (1 - β₂)) ∂μ = β₂ * (1 - β₂) * S2 + (1 - β₂) := by
    simp
  rw [integral_add (integrable_const _) (hS.const_mul _), hc, integral_const_mul,
    coin_integral_sum hb hp, ← sum_mul] at h
  have hS2le : (1 - β₂) * S2 ≤ 1 := by
    have := sum_pow_sub_le hβ₂ hβ₂' n
    rw [le_div_iff₀ (by linarith)] at this
    linarith
  have hS20 : 0 ≤ S2 := sum_nonneg fun _ _ => by positivity
  have hpC : 0 ≤ p * C ^ 2 := by positivity
  refine h.trans ?_
  have h0 : 0 ≤ β₂ * (1 + p * C ^ 2) := mul_nonneg hβ₂ (by linarith)
  nlinarith [mul_le_mul_of_nonneg_left hS2le h0]

/-- `E[1/√(β₂ v_n + 1 - β₂)] ≥ 1/√a` for every `a ≥ 1 + p C²`: the tangent of
`1/√x` at `a`.  arXiv:1904.09237, Appendix, proof of Theorem 3, the bound on
`E[T₃]`, by the tangent in place of Jensen's inequality. -/
theorem stoch_int_inv (hb : IsBernoulliSeq μ b p) (hp : 0 ≤ p) (hβ₂ : 0 ≤ β₂) (hβ₂' : β₂ < 1)
    {a : ℝ} (ha : 1 + p * C ^ 2 ≤ a) (n : ℕ) :
    1 / Real.sqrt a ≤
      ∫ ω, 1 / Real.sqrt (β₂ * coinV C β₁ β₂ α n (fun j => b j ω) + (1 - β₂)) ∂μ := by
  have ha0 : 0 < a := by nlinarith [mul_nonneg hp (sq_nonneg C)]
  have hs : 0 < Real.sqrt a := Real.sqrt_pos.2 ha0
  have hx := coin_integrable hb (coinMV_dep (C := C) (β₁ := β₁) (β₂ := β₂) (α := α) n
    fun _ v => β₂ * v + (1 - β₂))
  have hI := coin_integrable hb (coinMV_dep (C := C) (β₁ := β₁) (β₂ := β₂) (α := α) n
    fun _ v => 1 / Real.sqrt (β₂ * v + (1 - β₂)))
  have hv : ∀ ω, 0 ≤ coinV C β₁ β₂ α n (fun j => b j ω) := fun _ =>
    stoch_v_nonneg hβ₂ hβ₂'.le n
  have hL : Integrable (fun ω => 3 / (2 * Real.sqrt a) -
      (β₂ * coinV C β₁ β₂ α n (fun j => b j ω) + (1 - β₂)) / (2 * a * Real.sqrt a)) μ :=
    (integrable_const _).sub (hx.div_const _)
  have h := integral_mono hL hI fun ω => inv_sqrt_ge
    (by nlinarith [hv ω] : 0 < β₂ * coinV C β₁ β₂ α n (fun j => b j ω) + (1 - β₂)) ha0
  have hc : ∫ _ : Ω, 3 / (2 * Real.sqrt a) ∂μ = 3 / (2 * Real.sqrt a) := by simp
  rw [integral_sub (integrable_const _) (hx.div_const _), hc, integral_div] at h
  refine le_trans ?_ h
  have h1 := div_le_div_of_nonneg_right
    ((stoch_int_v hb hp hβ₂ hβ₂' n (α := α) (β₁ := β₁)).trans ha)
    (by positivity : (0 : ℝ) ≤ 2 * a * Real.sqrt a)
  have e1 : a / (2 * a * Real.sqrt a) = 1 / (2 * Real.sqrt a) := by field_simp
  have e2 : 3 / (2 * Real.sqrt a) = 3 * (1 / (2 * Real.sqrt a)) := by ring
  have e3 : 1 / Real.sqrt a = 2 * (1 / (2 * Real.sqrt a)) := by field_simp
  linarith

/-- The hypotheses are satisfiable: fair coins, `C = 1`, `β₁ = 0`, `β₂ = 1/2`,
`a = 2 = 1 + p C²`. -/
example : (∃ μ : Measure (ℕ → Bool), IsProbabilityMeasure μ ∧
    IsBernoulliSeq μ (fun t ω => ω t) 1) ∧ (1 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 / 2 ∧
    (1 / 2 : ℝ) < 1 ∧ 1 + 1 * (1 : ℝ) ^ 2 ≤ 2 :=
  ⟨exists_isBernoulliSeq (by norm_num) (by norm_num), by norm_num⟩

end AdamBeyond
end Transformer
