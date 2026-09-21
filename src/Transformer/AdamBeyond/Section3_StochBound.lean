import Transformer.AdamBeyond.Section3_Stoch

/-
# Adam and beyond — §3: the pointwise bounds of the lemma of Theorem 3

What the lemma in the proof of Theorem 3 bounds outcome by outcome, with
`C ≥ 1`, `0 ≤ β₁ ≤ 1` and `0 < β₂ < 1`:

* `T₁`: `(β₁ m + (1-β₁) C)/√(β₂ v + (1-β₂) C²) ≤ 1/√(1-β₂)` for `m ≤ C`,
  `stoch_T1_le`, as in the source;
* `T₂`: `β₁ m_n/√(β₂ v_n + 1 - β₂) ≤ β₁ K Σ_{j≤n} γ^{n-j} [b_j]`, with
  `K = 2(1-β₁)/√(β₂(1-β₂))` and `γ = β₁/√β₂`, `stoch_T2_le`: every coin
  `C` in `m_n` is paid for by its own `C²` in `v_n`.  The source bounds `T₂`
  on the event that the last `k ≈ log C/log(1/β₁)` coins are all `-1`, and by
  Cauchy-Schwarz elsewhere, which gives `O(p log C)` in expectation; this bound
  gives `O(p)` with no event;
* `T₃`: the tangent of `1/√x` at `a`, `inv_sqrt_ge`, which replaces Jensen's
  inequality, and `v_n ≤ (1-β₂) Σ_{j≤n} β₂^{n-j} (1 + C² [b_j])`.

Source: arXiv:1904.09237, Appendix, proof of Theorem 3, the lemma: the bounds
on `T₁`, `T₂` and `T₃`.
-/

open Finset

namespace Transformer
namespace AdamBeyond

open AMSGrad

/-- The indicator of a coin: `1` if `c`, else `0`. -/
def coinInd (c : Bool) : ℝ := if c then 1 else 0

theorem coinGrad_le {C : ℝ} (hC : 1 ≤ C) (c : Bool) : coinGrad C c ≤ C := by
  unfold coinGrad; split_ifs <;> linarith

theorem coinGrad_le_ind (C : ℝ) (c : Bool) : coinGrad C c ≤ (C + 1) * coinInd c := by
  cases c <;> simp [coinGrad, coinInd]

theorem coinGrad_sq_le (C : ℝ) (c : Bool) :
    coinGrad C c ^ 2 ≤ 1 + C ^ 2 * coinInd c := by
  cases c <;> simp [coinGrad, coinInd]

/-- `Σ_{j=1}^n r^{n-j} ≤ 1/(1-r)` for `0 ≤ r < 1`. -/
theorem sum_pow_sub_le {r : ℝ} (hr : 0 ≤ r) (hr' : r < 1) (n : ℕ) :
    ∑ j ∈ Icc 1 n, r ^ (n - j) ≤ 1 / (1 - r) := by
  have h1 : 0 < 1 - r := by linarith
  induction n with
  | zero => rw [Icc_eq_empty (by omega), sum_empty]; positivity
  | succ n ih =>
    rw [sum_Icc_succ_top (by omega), Nat.sub_self, pow_zero]
    have e : ∑ j ∈ Icc 1 n, r ^ (n + 1 - j) = r * ∑ j ∈ Icc 1 n, r ^ (n - j) := by
      rw [mul_sum]
      refine sum_congr rfl fun j hj => ?_
      rw [show n + 1 - j = n - j + 1 by have := (mem_Icc.1 hj).2; omega, pow_succ]
      ring
    rw [e]
    calc r * ∑ j ∈ Icc 1 n, r ^ (n - j) + 1 ≤ r * (1 / (1 - r)) + 1 := by gcongr
      _ = 1 / (1 - r) := by field_simp; ring

/-- The tangent of `1/√x` at `a > 0`: `1/√x ≥ 3/(2√a) - x/(2a√a)` for `x > 0`.
It replaces Jensen's inequality in the bound on `E[T₃]`. -/
theorem inv_sqrt_ge {x a : ℝ} (hx : 0 < x) (ha : 0 < a) :
    3 / (2 * Real.sqrt a) - x / (2 * a * Real.sqrt a) ≤ 1 / Real.sqrt x := by
  obtain ⟨s, hs, rfl⟩ : ∃ s, 0 < s ∧ x = s ^ 2 :=
    ⟨Real.sqrt x, Real.sqrt_pos.2 hx, (Real.sq_sqrt hx.le).symm⟩
  obtain ⟨r, hr, rfl⟩ : ∃ r, 0 < r ∧ a = r ^ 2 :=
    ⟨Real.sqrt a, Real.sqrt_pos.2 ha, (Real.sq_sqrt ha.le).symm⟩
  rw [Real.sqrt_sq hs.le, Real.sqrt_sq hr.le, div_sub_div _ _ (by positivity) (by positivity),
    div_le_div_iff₀ (by positivity) hs]
  nlinarith [mul_nonneg (mul_nonneg (sq_nonneg (r - s)) (by positivity : (0 : ℝ) ≤ 2 * r + s))
    (by positivity : (0 : ℝ) ≤ 2 * r ^ 2 * r)]

/-- The bound on `T₁`: `(β₁ m + (1-β₁) C)/√(β₂ v + (1-β₂) C²) ≤ 1/√(1-β₂)`
for `m ≤ C` and `v ≥ 0`.  arXiv:1904.09237, Appendix, proof of Theorem 3,
(eq:T_1-bound). -/
theorem stoch_T1_le {C β₁ β₂ m v : ℝ} (hC : 1 ≤ C) (hβ₁ : 0 ≤ β₁) (hβ₂ : 0 ≤ β₂)
    (hβ₂' : β₂ < 1) (hm : m ≤ C) (hv : 0 ≤ v) :
    (β₁ * m + (1 - β₁) * C) / Real.sqrt (β₂ * v + (1 - β₂) * C ^ 2) ≤ 1 / Real.sqrt (1 - β₂) := by
  have hq : 0 < Real.sqrt (1 - β₂) := Real.sqrt_pos.2 (by linarith)
  have hs : Real.sqrt (1 - β₂) * C ≤ Real.sqrt (β₂ * v + (1 - β₂) * C ^ 2) := by
    rw [Real.le_sqrt (by positivity) (by nlinarith), mul_pow, Real.sq_sqrt (by linarith)]
    nlinarith
  calc (β₁ * m + (1 - β₁) * C) / Real.sqrt (β₂ * v + (1 - β₂) * C ^ 2)
      ≤ C / Real.sqrt (β₂ * v + (1 - β₂) * C ^ 2) := by gcongr; nlinarith
    _ ≤ C / (Real.sqrt (1 - β₂) * C) := by gcongr
    _ = 1 / Real.sqrt (1 - β₂) := by field_simp

section Run

variable {Ω : Type*} {C β₁ β₂ : ℝ} {α : ℕ → ℝ} {b : ℕ → Ω → Bool} {ω : Ω}

/-- `m_n ≤ C`. -/
theorem stoch_m_le (hC : 1 ≤ C) (hβ₁ : 0 ≤ β₁) (hβ₁' : β₁ ≤ 1) (n : ℕ) :
    (stochSetup C β₁ β₂ α b ω).m adamRule n 0 ≤ C := by
  induction n with
  | zero => show (0 : ℝ) ≤ C; linarith
  | succ n ih =>
    rw [stoch_m_succ]
    have := coinGrad_le hC (b (n + 1) ω)
    nlinarith

/-- `v_n ≥ 0`. -/
theorem stoch_v_nonneg (hβ₂ : 0 ≤ β₂) (hβ₂' : β₂ ≤ 1) (n : ℕ) :
    0 ≤ (stochSetup C β₁ β₂ α b ω).v adamRule n 0 := by
  rw [stoch_v_eq]
  exact mul_nonneg (by linarith) (sum_nonneg fun _ _ => by positivity)

/-- `v_n ≥ (1-β₂) β₂^{n-j} C²` for every coin `b_j = true`, `j ≤ n`. -/
theorem stoch_v_ge (hβ₂ : 0 ≤ β₂) (hβ₂' : β₂ ≤ 1) {n j : ℕ} (hj : j ∈ Icc 1 n)
    (hb : b j ω = true) :
    (1 - β₂) * β₂ ^ (n - j) * C ^ 2 ≤ (stochSetup C β₁ β₂ α b ω).v adamRule n 0 := by
  rw [stoch_v_eq, mul_assoc]
  refine mul_le_mul_of_nonneg_left ?_ (by linarith)
  have := single_le_sum (f := fun j => β₂ ^ (n - j) * coinGrad C (b j ω) ^ 2)
    (fun _ _ => by positivity) hj
  simpa [coinGrad, hb] using this

/-- `v_n ≤ (1-β₂) Σ_{j≤n} β₂^{n-j} (1 + C² [b_j])`. -/
theorem stoch_v_le_ind (hβ₂ : 0 ≤ β₂) (hβ₂' : β₂ ≤ 1) (n : ℕ) :
    (stochSetup C β₁ β₂ α b ω).v adamRule n 0 ≤
      (1 - β₂) * ∑ j ∈ Icc 1 n, β₂ ^ (n - j) * (1 + C ^ 2 * coinInd (b j ω)) := by
  rw [stoch_v_eq]
  have : 0 ≤ 1 - β₂ := by linarith
  gcongr with j
  exact coinGrad_sq_le C _

/-- `m_n ≤ (1-β₁) Σ_{j≤n} β₁^{n-j} (C + 1) [b_j]`. -/
theorem stoch_m_le_ind (hβ₁ : 0 ≤ β₁) (hβ₁' : β₁ ≤ 1) (n : ℕ) :
    (stochSetup C β₁ β₂ α b ω).m adamRule n 0 ≤
      (1 - β₁) * ∑ j ∈ Icc 1 n, β₁ ^ (n - j) * ((C + 1) * coinInd (b j ω)) := by
  rw [stoch_m_eq]
  have : 0 ≤ 1 - β₁ := by linarith
  gcongr with j
  exact coinGrad_le_ind C _

/-- The bound on `T₂`, outcome by outcome:
`β₁ m_n/√(β₂ v_n + 1 - β₂) ≤ β₁ K Σ_{j≤n} γ^{n-j} [b_j]` with
`K = 2(1-β₁)/√(β₂(1-β₂))` and `γ = β₁/√β₂`.  arXiv:1904.09237, Appendix,
proof of Theorem 3, the bound on `T₂` (eq:T_2-bound), by another argument. -/
theorem stoch_T2_le (hC : 1 ≤ C) (hβ₁ : 0 ≤ β₁) (hβ₁' : β₁ ≤ 1) (hβ₂ : 0 < β₂)
    (hβ₂' : β₂ < 1) (n : ℕ) :
    β₁ * (stochSetup C β₁ β₂ α b ω).m adamRule n 0 /
        Real.sqrt (β₂ * (stochSetup C β₁ β₂ α b ω).v adamRule n 0 + (1 - β₂)) ≤
      β₁ * (2 * (1 - β₁) / Real.sqrt (β₂ * (1 - β₂)) *
        ∑ j ∈ Icc 1 n, (β₁ / Real.sqrt β₂) ^ (n - j) * coinInd (b j ω)) := by
  set V := (stochSetup C β₁ β₂ α b ω).v adamRule n 0 with hVdef
  have hV : 0 ≤ V := stoch_v_nonneg hβ₂.le hβ₂'.le n
  set D := Real.sqrt (β₂ * V + (1 - β₂)) with hDdef
  have hD : 0 < D := Real.sqrt_pos.2 (by nlinarith)
  have hB : 0 ≤ 1 - β₁ := by linarith
  have key : ∀ j ∈ Icc 1 n, (1 - β₁) * (β₁ ^ (n - j) * ((C + 1) * coinInd (b j ω))) / D ≤
      2 * (1 - β₁) / Real.sqrt (β₂ * (1 - β₂)) *
        ((β₁ / Real.sqrt β₂) ^ (n - j) * coinInd (b j ω)) := by
    intro j hj
    cases hbj : b j ω
    · simp [coinInd]
    simp only [coinInd, ite_true, mul_one]
    set L := Real.sqrt (β₂ * (1 - β₂)) * Real.sqrt β₂ ^ (n - j) with hLdef
    have hL0 : 0 < L := by
      have : 0 < β₂ * (1 - β₂) := by nlinarith
      positivity
    have hL : L * C ≤ D := by
      rw [hDdef, Real.le_sqrt (by positivity) (by nlinarith)]
      have hv := stoch_v_ge (α := α) (β₁ := β₁) (C := C) hβ₂.le hβ₂'.le hj hbj
      have e : (L * C) ^ 2 = β₂ * ((1 - β₂) * β₂ ^ (n - j) * C ^ 2) := by
        rw [hLdef, mul_pow, mul_pow, Real.sq_sqrt (by nlinarith), ← pow_mul, mul_comm (n - j) 2,
          pow_mul, Real.sq_sqrt hβ₂.le]
        ring
      rw [e]
      nlinarith
    have hw : 0 ≤ (1 - β₁) * β₁ ^ (n - j) := by positivity
    calc (1 - β₁) * (β₁ ^ (n - j) * (C + 1)) / D
        ≤ (1 - β₁) * (β₁ ^ (n - j) * (C + 1)) / (L * C) := by gcongr
      _ ≤ 2 * ((1 - β₁) * β₁ ^ (n - j)) / L := by
          rw [div_le_div_iff₀ (by positivity) hL0]
          have := mul_nonneg hw hL0.le
          nlinarith
      _ = _ := by rw [div_pow, hLdef]; field_simp
  calc β₁ * (stochSetup C β₁ β₂ α b ω).m adamRule n 0 / D
      = β₁ * ((stochSetup C β₁ β₂ α b ω).m adamRule n 0 / D) := by ring
    _ ≤ β₁ * ((1 - β₁) * (∑ j ∈ Icc 1 n, β₁ ^ (n - j) * ((C + 1) * coinInd (b j ω))) / D) := by
        gcongr
        exact stoch_m_le_ind hβ₁ hβ₁' n
    _ = β₁ * ∑ j ∈ Icc 1 n, (1 - β₁) * (β₁ ^ (n - j) * ((C + 1) * coinInd (b j ω))) / D := by
        rw [← sum_div, ← mul_sum]
    _ ≤ β₁ * ∑ j ∈ Icc 1 n, 2 * (1 - β₁) / Real.sqrt (β₂ * (1 - β₂)) *
          ((β₁ / Real.sqrt β₂) ^ (n - j) * coinInd (b j ω)) := by
        gcongr with j hj
        exact key j hj
    _ = _ := by congr 1; rw [mul_sum]

end Run

/-- The hypotheses of the bounds are satisfiable: `C = 1`, `β₁ = 0`, `β₂ = 1/2`. -/
example : (1 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) < 1 := by norm_num

end AdamBeyond
end Transformer
