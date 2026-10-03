import Transformer.AMSGrad.Section4_Theorem
import Transformer.AMSGrad.Section4_Rate

/-
# AMSGrad — the corrected convergence rate

§4 of arXiv:1904.03590v4: Corollary 4.5, from Theorem 4.1.

**What the source says and what is carried here.**

* Corollary 4.5 asserts `lim R(T)/T = 0`.  Theorem 4.1 is an upper bound and
  gives only the upper half, `limsup R(T)/T ≤ 0`, which is what is stated here,
  `cor_lambda` and `cor_inv`.  The lower half is false: `Section4_Counter`.

Source: arXiv:1904.03590v4, §4, Corollary 4.5.
-/

open Finset Filter

namespace Transformer
namespace AMSGrad

variable {d : ℕ}

/-- **Corollary 4.5, `β_{1,t} = β₁λ^{t-1}`, the upper half.**  Under the
assumptions of `mainthm_lambda`, for every `ε > 0`, eventually
`R(T)/T ≤ ε` for all `x* ∈ F`.  The source asserts `lim R(T)/T = 0`; see the
module docstring.

Source: arXiv:1904.03590v4, §4, Corollary 4.5. -/
theorem cor_lambda {S : Setup d} {F : Set (Vec d)} {D G : ℝ} (hS : IsOnlineConvex S F D G)
    {α β₁ lam : ℝ} (hα : 0 < α) (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₁ : 0 ≤ β₁) (hβ₁' : β₁ < 1) (hl : 0 < lam) (hl' : lam < 1)
    (hb : ∀ t, S.β₁ t = β₁ * lam ^ (t - 1))
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) (hγ : β₁ / Real.sqrt S.β₂ < 1)
    {ε : ℝ} (hε : 0 < ε) :
    ∀ᶠ T : ℕ in atTop, ∀ xstar ∈ F, S.regret amsgradRule xstar T / T ≤ ε := by
  obtain ⟨t₀, -, h⟩ := mainthm_lambda hS hα hαt hβ₁ hβ₁' hl hl' hb hβ₂ hβ₂' hγ
  set A := d * D ^ 2 * G / (2 * α * (1 - β₁))
  obtain ⟨K, hKd⟩ : ∃ K : ℝ,
      K = α / ((1 - β₁) ^ 2 * Real.sqrt (1 - S.β₂) * (1 - β₁ / Real.sqrt S.β₂)) := ⟨_, rfl⟩
  obtain ⟨a, had⟩ : ∃ a : ℝ, a = A * ∑ t ∈ Icc (1 : ℕ) t₀, Real.sqrt t +
      d * D ^ 2 * G / (2 * α * (1 - β₁) * (1 - lam) ^ 2) := ⟨_, rfl⟩
  obtain ⟨b, hbd⟩ : ∃ b : ℝ, b = A := ⟨_, rfl⟩
  obtain ⟨c, hcd⟩ : ∃ c : ℝ, c = K * (d * G) := ⟨_, rfl⟩
  set M := |a| + |b| + |c|
  have hM : 0 ≤ M := by positivity
  filter_upwards [tendsto_rate.eventually
    (eventually_le_nhds (div_pos hε (by linarith : 0 < M + 1))), eventually_ge_atTop 1]
    with T hq hT xstar hx
  have hK : 0 ≤ K := by
    rw [hKd]
    have : 0 < 1 - β₁ / Real.sqrt S.β₂ := by linarith
    have : 0 < 1 - S.β₂ := by linarith
    have : 0 < 1 - β₁ := by linarith
    positivity
  have hg : K * Real.sqrt (Real.log T + 1) * ∑ i, S.gnorm amsgradRule T i ≤
      K * Real.sqrt (Real.log T + 1) * (d * G * Real.sqrt T) := by
    refine mul_le_mul_of_nonneg_left ((sum_le_sum fun i _ => gnorm_le hS amsgradRule T i).trans
      (le_of_eq ?_)) (by positivity)
    rw [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
    ring
  have hr := h T hT xstar hx
  have hT0 : (0 : ℝ) < T := by exact_mod_cast hT
  calc S.regret amsgradRule xstar T / T
      ≤ (a + b * Real.sqrt T + c * Real.sqrt (Real.log T + 1) * Real.sqrt T) / T := by
        refine div_le_div_of_nonneg_right ?_ hT0.le
        have e : α * Real.sqrt (Real.log T + 1) /
            ((1 - β₁) ^ 2 * Real.sqrt (1 - S.β₂) * (1 - β₁ / Real.sqrt S.β₂)) *
              ∑ i, S.gnorm amsgradRule T i =
            K * Real.sqrt (Real.log T + 1) * ∑ i, S.gnorm amsgradRule T i := by
          rw [hKd]; ring
        subst had hbd hcd
        linarith
    _ ≤ M * Real.sqrt ((Real.log T + 1) / T) := rate_le a b c hT
    _ ≤ M * (ε / (M + 1)) := mul_le_mul_of_nonneg_left hq hM
    _ ≤ ε := by
        rw [mul_div_assoc', div_le_iff₀ (by linarith)]
        nlinarith

/-- **Corollary 4.5, `β_{1,t} = β₁/t`, the upper half.**  Under the
assumptions of `mainthm_inv`, for every `ε > 0`, eventually `R(T)/T ≤ ε` for
all `x* ∈ F`.

Source: arXiv:1904.03590v4, §4, Corollary 4.5. -/
theorem cor_inv {S : Setup d} {F : Set (Vec d)} {D G : ℝ} (hS : IsOnlineConvex S F D G)
    {α β₁ : ℝ} (hα : 0 < α) (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₁ : 0 ≤ β₁) (hβ₁' : β₁ < 1) (hb : ∀ t, S.β₁ t = β₁ / t)
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) (hγ : β₁ / Real.sqrt S.β₂ < 1)
    {ε : ℝ} (hε : 0 < ε) :
    ∀ᶠ T : ℕ in atTop, ∀ xstar ∈ F, S.regret amsgradRule xstar T / T ≤ ε := by
  obtain ⟨t₀, -, h⟩ := mainthm_inv hS hα hαt hβ₁ hβ₁' hb hβ₂ hβ₂' hγ
  set A := d * D ^ 2 * G / (2 * α * (1 - β₁))
  obtain ⟨K, hKd⟩ : ∃ K : ℝ,
      K = α / ((1 - β₁) ^ 2 * Real.sqrt (1 - S.β₂) * (1 - β₁ / Real.sqrt S.β₂)) := ⟨_, rfl⟩
  obtain ⟨a, had⟩ : ∃ a : ℝ, a = A * ∑ t ∈ Icc (1 : ℕ) t₀, Real.sqrt t := ⟨_, rfl⟩
  obtain ⟨b, hbd⟩ : ∃ b : ℝ, b = A + d * D ^ 2 * G / (α * (1 - β₁)) := ⟨_, rfl⟩
  obtain ⟨c, hcd⟩ : ∃ c : ℝ, c = K * (d * G) := ⟨_, rfl⟩
  set M := |a| + |b| + |c|
  have hM : 0 ≤ M := by positivity
  filter_upwards [tendsto_rate.eventually
    (eventually_le_nhds (div_pos hε (by linarith : 0 < M + 1))), eventually_ge_atTop 1]
    with T hq hT xstar hx
  have hK : 0 ≤ K := by
    rw [hKd]
    have : 0 < 1 - β₁ / Real.sqrt S.β₂ := by linarith
    have : 0 < 1 - S.β₂ := by linarith
    have : 0 < 1 - β₁ := by linarith
    positivity
  have hg : K * Real.sqrt (Real.log T + 1) * ∑ i, S.gnorm amsgradRule T i ≤
      K * Real.sqrt (Real.log T + 1) * (d * G * Real.sqrt T) := by
    refine mul_le_mul_of_nonneg_left ((sum_le_sum fun i _ => gnorm_le hS amsgradRule T i).trans
      (le_of_eq ?_)) (by positivity)
    rw [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
    ring
  have hr := h T hT xstar hx
  have hT0 : (0 : ℝ) < T := by exact_mod_cast hT
  calc S.regret amsgradRule xstar T / T
      ≤ (a + b * Real.sqrt T + c * Real.sqrt (Real.log T + 1) * Real.sqrt T) / T := by
        refine div_le_div_of_nonneg_right ?_ hT0.le
        have e : α * Real.sqrt (Real.log T + 1) /
            ((1 - β₁) ^ 2 * Real.sqrt (1 - S.β₂) * (1 - β₁ / Real.sqrt S.β₂)) *
              ∑ i, S.gnorm amsgradRule T i =
            K * Real.sqrt (Real.log T + 1) * ∑ i, S.gnorm amsgradRule T i := by
          rw [hKd]; ring
        have e2 : d * D ^ 2 * G * Real.sqrt T / (α * (1 - β₁)) =
            d * D ^ 2 * G / (α * (1 - β₁)) * Real.sqrt T := by ring
        subst had hbd hcd
        linarith
    _ ≤ M * Real.sqrt ((Real.log T + 1) / T) := rate_le a b c hT
    _ ≤ M * (ε / (M + 1)) := mul_le_mul_of_nonneg_left hq hM
    _ ≤ ε := by
        rw [mul_div_assoc', div_le_iff₀ (by linarith)]
        nlinarith

/-- The hypotheses of Corollary 4.5, in both settings, are satisfiable: the zero
cost on `[-1, 1]`, `α = 1`, `β₁ = 0`, `λ = 1/2`, `β₂ = 1/2`, `ε = 1`. -/
example :
    let S := zeroSetup (d := 1) (fun t : ℕ => 1 / Real.sqrt t) (fun _ => 0) (1 / 2)
    IsOnlineConvex S (Set.Icc (fun _ => -1) (fun _ => 1)) 2 0 ∧ (0 : ℝ) < 1 ∧
      S.α = (fun t : ℕ => 1 / Real.sqrt t) ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 ∧
      (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) < 1 ∧ (∀ t, S.β₁ t = 0 * (1 / 2) ^ (t - 1)) ∧
      (∀ t, S.β₁ t = 0 / t) ∧ 0 < S.β₂ ∧ S.β₂ < 1 ∧ 0 / Real.sqrt S.β₂ < 1 ∧ (0 : ℝ) < 1 :=
  ⟨isOnlineConvex_zero _ _ _, one_pos, rfl, le_rfl, one_pos, by norm_num, by norm_num,
    fun _ => by simp [zeroSetup], fun _ => by simp [zeroSetup], by norm_num [zeroSetup],
    by norm_num [zeroSetup], by simp, one_pos⟩

end AMSGrad
end Transformer
