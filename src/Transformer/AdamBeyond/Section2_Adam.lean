import Transformer.AMSGrad.Section4_Lemmas

/-
# Adam and beyond — §2 and §3: the adaptive methods and `Γ_t`

§2 of arXiv:1904.09237 (Reddi, Kale, Kumar) sets up Algorithm 1, the generic
adaptive method, and its instances SGD, AdaGrad and Adam; §3 opens with the
quantity `Γ_{t+1} = √V_{t+1}/α_{t+1} - √V_t/α_t` and the observation that it is
`⪰ 0` for SGD and AdaGrad.

**What the source says and what is carried here.**

* The run is `Transformer.AMSGrad.Setup.state` (Algorithm 1 of
  arXiv:1904.03590, which is Algorithm 2 of this paper) for a rule
  `(t, v̂_{t-1}, v_t) ↦ v̂_t`; `V_t` of Algorithm 1 is `diag(v̂_t)`.  Every
  instance of this paper has diagonal `V_t` and a moment `m_t` given by the
  exponential moving average with `β_{1,t}`, so each is one rule:
  - Adam (without the debiasing step, as in the source) is `adamRule`,
    `v̂_t = v_t`;
  - SGD is `sgdRule`, `V_t = I`, run with `β_{1,t} = 0`;
  - AdaGrad is `adagradRule`, run with `β₂ = 0`, so that `v_t = g_t²` and
    `v̂_t = ((t-1)v̂_{t-1} + g_t²)/t = Σ_{j≤t} g_j²/t` (`adagrad_vhat`).
  `V_0 = 0`, as in Algorithm 2; `V_0` enters only `Γ_1`.

* `eq:adam` and `eq:adam-recur` are the same `m_t` and `v_t`: `m_sum`, `v_sum`.

* `Γ_t` is `Gamma`, a vector since every `V_t` is diagonal; `Γ_t ⪰ 0` is
  `0 ≤ Gamma`.  The source leaves `Γ_1` implicit; here `Γ_t` is defined for
  every `t` through `V_0 = 0` and `α_0`, which for `α_t = α/√t` is `α/0 = 0` in
  Lean, so `√V_0/α_0 = 0` and `Γ_1 = √V_1/α_1`.  With `α_t = α/√t`, `Γ_t ⪰ 0`
  for SGD (`gamma_sgd_nonneg`), AdaGrad (`gamma_adagrad_nonneg`), and, as §4
  states, AMSGrad (`gamma_amsgrad_nonneg`).

Source: arXiv:1904.09237, §2, Algorithm 1, (SGD), (AdaGrad), (Adam),
eq:adam-recur; §3, eq:gamma-t; §4, the paragraph after Algorithm 2.
-/

open Finset

namespace Transformer
namespace AdamBeyond

open AMSGrad

variable {d : ℕ}

/-! ### The instances of Algorithm 1 -/

/-- Adam, without the debiasing step: `V_t = diag(v_t)`.
arXiv:1904.09237, §2, (Adam). -/
def adamRule : Rule d := fun _ _ b => b

/-- SGD: `V_t = I`.  Run with `β_{1,t} = 0`.  arXiv:1904.09237, §2, (SGD). -/
def sgdRule : Rule d := fun _ _ _ => 1

/-- AdaGrad: `v̂_t = ((t-1)v̂_{t-1} + v_t)/t`.  Run with `β₂ = 0`, so that
`v_t = g_t²` and `v̂_t = Σ_{j≤t} g_j²/t` (`adagrad_vhat`).
arXiv:1904.09237, §2, (AdaGrad). -/
noncomputable def adagradRule : Rule d :=
  fun t a b => (((t : ℝ) - 1) / t) • a + (1 / (t : ℝ)) • b

/-! ### `eq:adam` is `eq:adam-recur` -/

/-- `v_t = (1-β₂) Σ_{j≤t} β₂^{t-j} g_j²`, for every rule.
arXiv:1904.09237, §2, (Adam) and eq:adam-recur. -/
theorem v_sum (S : Setup d) (R : Rule d) (n : ℕ) (i : Fin d) :
    S.v R n i = (1 - S.β₂) * ∑ j ∈ Icc 1 n, S.β₂ ^ (n - j) * S.g R j i ^ 2 := by
  induction n with
  | zero => simp [Setup.v, Setup.state]
  | succ n ih =>
    change S.β₂ * S.v R n i + (1 - S.β₂) * S.g R (n + 1) i ^ 2 = _
    rw [ih, sum_Icc_succ_top (by omega), Nat.sub_self, pow_zero, one_mul, mul_add, ← mul_assoc,
      mul_comm S.β₂, mul_assoc, mul_sum]
    congr 2
    refine sum_congr rfl fun j hj => ?_
    rw [show n + 1 - j = n - j + 1 by have := (mem_Icc.mp hj).2; omega, pow_succ]
    ring

/-- `m_t = (1-β₁) Σ_{j≤t} β₁^{t-j} g_j`, for constant `β₁` and every rule.
arXiv:1904.09237, §2, (Adam) and eq:adam-recur. -/
theorem m_sum {S : Setup d} {b : ℝ} (hb : ∀ t, S.β₁ t = b) (R : Rule d) (n : ℕ) (i : Fin d) :
    S.m R n i = (1 - b) * ∑ j ∈ Icc 1 n, b ^ (n - j) * S.g R j i := by
  induction n with
  | zero => simp [Setup.m, Setup.state]
  | succ n ih =>
    change S.β₁ (n + 1) * S.m R n i + (1 - S.β₁ (n + 1)) * S.g R (n + 1) i = _
    rw [hb, ih, sum_Icc_succ_top (by omega), Nat.sub_self, pow_zero, one_mul, mul_add,
      ← mul_assoc, mul_comm b, mul_assoc, mul_sum]
    congr 2
    refine sum_congr rfl fun j hj => ?_
    rw [show n + 1 - j = n - j + 1 by have := (mem_Icc.mp hj).2; omega, pow_succ]
    ring

/-- With `β₂ = 0`, the AdaGrad rule gives `v̂_t = Σ_{j≤t} g_j²/t`.
arXiv:1904.09237, §2, (AdaGrad). -/
theorem adagrad_vhat {S : Setup d} (hβ₂ : S.β₂ = 0) (n : ℕ) (i : Fin d) :
    S.vhat adagradRule n i = (∑ t ∈ Icc 1 n, S.g adagradRule t i ^ 2) / n := by
  induction n with
  | zero => simp [Setup.vhat, Setup.state]
  | succ n ih =>
    change (((n + 1 : ℕ) : ℝ) - 1) / ((n + 1 : ℕ) : ℝ) * S.vhat adagradRule n i
      + 1 / ((n + 1 : ℕ) : ℝ) * (S.β₂ * S.v adagradRule n i
        + (1 - S.β₂) * S.g adagradRule (n + 1) i ^ 2) = _
    rw [ih, hβ₂, sum_Icc_succ_top (by omega)]
    rcases Nat.eq_zero_or_pos n with rfl | hn
    · simp
    · have : (n : ℝ) ≠ 0 := by positivity
      push_cast
      field_simp
      ring

/-! ### `Γ_t` -/

/-- `Γ_t = √V_t/α_t - √V_{t-1}/α_{t-1}`, coordinatewise.
arXiv:1904.09237, §3, eq:gamma-t. -/
noncomputable def Gamma (S : Setup d) (R : Rule d) (t : ℕ) : Vec d :=
  fun i => Real.sqrt (S.vhat R t i) / S.α t - Real.sqrt (S.vhat R (t - 1) i) / S.α (t - 1)

/-- With `α_t = α/√t`, `Γ_{n+1} = (√(v̂_{n+1}(n+1)) - √(v̂_n n))/α`. -/
theorem gamma_succ {S : Setup d} {α : ℝ} (hα : S.α = fun t : ℕ => α / Real.sqrt t) (R : Rule d)
    (n : ℕ) (i : Fin d) : Gamma S R (n + 1) i =
      (Real.sqrt (S.vhat R (n + 1) i) * Real.sqrt ((n + 1 : ℕ) : ℝ)
        - Real.sqrt (S.vhat R n i) * Real.sqrt (n : ℝ)) / α := by
  simp only [Gamma, hα, Nat.add_sub_cancel, div_div_eq_mul_div, sub_div]

/-- `Γ_t ⪰ 0` as soon as `√v̂_t · √t` is non-decreasing. -/
theorem gamma_nonneg_of {S : Setup d} {α : ℝ} (hα : S.α = fun t : ℕ => α / Real.sqrt t)
    (hα₀ : 0 < α) (R : Rule d) (i : Fin d)
    (h : ∀ n : ℕ, Real.sqrt (S.vhat R n i) * Real.sqrt (n : ℝ)
      ≤ Real.sqrt (S.vhat R (n + 1) i) * Real.sqrt ((n + 1 : ℕ) : ℝ)) (t : ℕ) :
    0 ≤ Gamma S R t i := by
  cases t with
  | zero => simp [Gamma]
  | succ n => rw [gamma_succ hα]; exact div_nonneg (sub_nonneg.mpr (h n)) hα₀.le

/-- **SGD.**  With `α_t = α/√t`, `Γ_t ⪰ 0`.  arXiv:1904.09237, §3, after eq:gamma-t. -/
theorem gamma_sgd_nonneg {S : Setup d} {α : ℝ} (hα : S.α = fun t : ℕ => α / Real.sqrt t)
    (hα₀ : 0 < α) (t : ℕ) : 0 ≤ Gamma S sgdRule t := fun i => by
  refine gamma_nonneg_of hα hα₀ _ i (fun n => ?_) t
  have h1 : S.vhat sgdRule n i ≤ 1 := by cases n <;> simp [Setup.vhat, Setup.state, Setup.step, sgdRule]
  have h2 : S.vhat sgdRule (n + 1) i = 1 := rfl
  rw [h2, Real.sqrt_one, one_mul]
  calc Real.sqrt (S.vhat sgdRule n i) * Real.sqrt n ≤ 1 * Real.sqrt n :=
        mul_le_mul_of_nonneg_right (Real.sqrt_le_one.mpr h1) (Real.sqrt_nonneg _)
    _ ≤ Real.sqrt ((n + 1 : ℕ) : ℝ) := by
        rw [one_mul]; exact Real.sqrt_le_sqrt (by push_cast; linarith)

/-- **AdaGrad.**  With `α_t = α/√t` and `β₂ = 0`, `Γ_t ⪰ 0`.
arXiv:1904.09237, §3, after eq:gamma-t. -/
theorem gamma_adagrad_nonneg {S : Setup d} {α : ℝ} (hα : S.α = fun t : ℕ => α / Real.sqrt t)
    (hα₀ : 0 < α) (hβ₂ : S.β₂ = 0) (t : ℕ) : 0 ≤ Gamma S adagradRule t := fun i => by
  have key : ∀ n : ℕ, Real.sqrt (S.vhat adagradRule n i) * Real.sqrt (n : ℝ)
      = Real.sqrt (∑ t ∈ Icc 1 n, S.g adagradRule t i ^ 2) := by
    intro n
    rw [adagrad_vhat hβ₂, ← Real.sqrt_mul' _ (Nat.cast_nonneg n)]
    rcases Nat.eq_zero_or_pos n with rfl | hn
    · simp
    · rw [div_mul_cancel₀ _ (by positivity)]
  refine gamma_nonneg_of hα hα₀ _ i (fun n => ?_) t
  rw [key, key]
  exact Real.sqrt_le_sqrt (sum_le_sum_of_subset_of_nonneg (Icc_subset_Icc_right (by omega))
    fun _ _ _ => sq_nonneg _)

/-- **AMSGrad.**  With `α_t = α/√t`, `Γ_t ⪰ 0`, whatever `β₂`.
arXiv:1904.09237, §4, the paragraph after Algorithm 2. -/
theorem gamma_amsgrad_nonneg {S : Setup d} {α : ℝ} (hα : S.α = fun t : ℕ => α / Real.sqrt t)
    (hα₀ : 0 < α) (t : ℕ) : 0 ≤ Gamma S amsgradRule t := fun i => by
  refine gamma_nonneg_of hα hα₀ _ i (fun n => ?_) t
  exact mul_le_mul (Real.sqrt_le_sqrt (vhat_le_succ S n i))
    (Real.sqrt_le_sqrt (by push_cast; linarith)) (Real.sqrt_nonneg _) (Real.sqrt_nonneg _)

/-- The hypotheses of `m_sum`, `adagrad_vhat` and the `Γ_t ⪰ 0` theorems are
satisfiable; `gamma_nonneg_of`'s monotonicity is `gamma_amsgrad_nonneg`'s. -/
example : (zeroSetup (d := 1) (fun t : ℕ => 1 / Real.sqrt t) (fun _ => 0) 0).α
      = (fun t : ℕ => 1 / Real.sqrt t) ∧ (0 : ℝ) < 1 ∧
      (zeroSetup (d := 1) (fun t : ℕ => 1 / Real.sqrt t) (fun _ => 0) 0).β₂ = 0 ∧
      ∀ t, (zeroSetup (d := 1) (fun t : ℕ => 1 / Real.sqrt t) (fun _ => 0) 0).β₁ t = 0 :=
  ⟨rfl, one_pos, rfl, fun _ => rfl⟩

end AdamBeyond
end Transformer
