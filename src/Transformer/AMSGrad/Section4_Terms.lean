import Transformer.AMSGrad.Section4_MainLemma
import Transformer.AMSGrad.Section4_Lemmas

/-
# AMSGrad — the second term of Lemma 3.1

§4 of arXiv:1904.03590v4, proof of Theorem 4.1: the bound (eqsecond2) on the
second term of Lemma 3.1, from Lemma 4.4, and the series bound used for the
third term when `β_{1,t} = β₁λ^{t-1}`.

**What the source says and what is carried here.**

* For `β_{1,t} = β₁λ^{t-1}` the source bounds `Σ_{t=2}^T √(t-1) λ^{t-1}` by
  `Σ_t t λ^{t-1} ≤ 1/(1-λ)²`, Lemma 2.3.  The bound used here is
  `√s ≤ s + 1` and the series `Σ_{s≥0} (s+1)λ^s = 1/(1-λ)²`, `taylor_deriv`.

Source: arXiv:1904.03590v4, §4, proof of Theorem 4.1.
-/

open Finset

namespace Transformer
namespace AMSGrad

variable {d : ℕ}

/-- `Σ_{t=2}^T √(t-1) λ^{t-1} ≤ 1/(1-λ)²` for `0 ≤ λ < 1`.
arXiv:1904.03590v4, §4, proof of Theorem 4.1, via Lemma 2.3. -/
theorem sum_sqrt_mul_pow_le {lam : ℝ} (hl : 0 ≤ lam) (hl' : lam < 1) (T : ℕ) :
    ∑ t ∈ Icc 2 T, Real.sqrt ((t - 1 : ℕ) : ℝ) * lam ^ (t - 1) ≤ 1 / (1 - lam) ^ 2 := by
  set g : ℕ → ℝ := fun t => ((t : ℝ) + 1) * lam ^ t
  have hg : ∀ t, 0 ≤ g t := fun t => by positivity
  have hsum : Summable g := by
    by_contra h
    have h0 := tsum_eq_zero_of_not_summable h
    rw [taylor_deriv hl hl'] at h0
    exact (by have : 0 < 1 - lam := by linarith
              positivity : (0 : ℝ) < 1 / (1 - lam) ^ 2).ne' h0
  have key : ∀ n, ∑ t ∈ Icc 2 n, g (t - 1) ≤ ∑ t ∈ range n, g t := by
    intro n
    induction n with
    | zero => simp
    | succ n ih =>
      rcases Nat.lt_or_ge n 1 with h | h
      · obtain rfl : n = 0 := by omega
        simpa using hg 0
      · rw [sum_Icc_succ_top (by omega), sum_range_succ, Nat.add_sub_cancel]
        linarith
  rw [← taylor_deriv hl hl']
  refine le_trans (sum_le_sum fun t _ => ?_) ((key T).trans (hsum.sum_le_tsum _ (fun t _ => hg t)))
  have hs := Real.sq_sqrt (Nat.cast_nonneg (α := ℝ) (t - 1))
  exact mul_le_mul_of_nonneg_right (by nlinarith [Real.sqrt_nonneg ((t - 1 : ℕ) : ℝ)])
    (pow_nonneg hl _)

/-- The hypotheses of `sum_sqrt_mul_pow_le` are satisfiable: `λ = 1/2`. -/
example : (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) < 1 := by norm_num

/-- **(eqsecond2).**  For a rule with `v_t ≤ v̂_t`, `α_t = α/√t`,
`0 ≤ β_{1,t} ≤ β₁ < 1`, `0 < β₂ < 1` and `γ < 1`,

  `Σᵢ Σ_{t=1}^T α_t/(1-β₁) m²_{t,i}/√v̂_{t,i}`
  `    ≤ α√(ln T + 1)/((1-β₁)²√(1-β₂)(1-γ)) Σᵢ ‖g_{1:T,i}‖₂`.

Source: arXiv:1904.03590v4, §4, proof of Theorem 4.1, (eqsecond2). -/
theorem eqsecond_le {S : Setup d} {R : Rule d} (hR : ∀ t a b, b ≤ R t a b) {α β₁ : ℝ}
    (hα : 0 ≤ α) (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₁ : ∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ β₁) (hβ₁' : β₁ < 1)
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) (hγ : β₁ / Real.sqrt S.β₂ < 1) (T : ℕ) :
    ∑ i, ∑ t ∈ Icc 1 T, S.α t / (1 - β₁) * S.m R t i ^ 2 / Real.sqrt (S.vhat R t i) ≤
      α * Real.sqrt (Real.log T + 1) /
        ((1 - β₁) ^ 2 * Real.sqrt (1 - S.β₂) * (1 - β₁ / Real.sqrt S.β₂)) *
          ∑ i, S.gnorm R T i := by
  have hB : 0 < 1 - β₁ := by linarith
  rw [mul_sum]
  refine sum_le_sum fun i _ => ?_
  have hm := mainlem hR hβ₁ hβ₁' hβ₂ hβ₂' hγ T i
  set B := 1 - β₁
  set Γ := 1 - β₁ / Real.sqrt S.β₂
  have e1 : ∑ t ∈ Icc 1 T, S.α t / B * S.m R t i ^ 2 / Real.sqrt (S.vhat R t i) =
      α / B * ∑ t ∈ Icc 1 T, S.m R t i ^ 2 / Real.sqrt (t * S.vhat R t i) := by
    rw [mul_sum]
    refine sum_congr rfl fun t _ => ?_
    rw [hαt, Real.sqrt_mul (Nat.cast_nonneg _)]
    simp only [div_eq_mul_inv, mul_inv]
    ring
  rw [e1]
  refine (mul_le_mul_of_nonneg_left hm (div_nonneg hα hB.le)).trans (le_of_eq ?_)
  simp only [div_eq_mul_inv, mul_inv, sq]
  ring

/-- The hypotheses of `eqsecond_le` are satisfiable: the zero cost on `[-1, 1]`,
AMSGrad, `α = 1`, `β₁ = 0`, `β₂ = 1/2`. -/
example :
    let S := zeroSetup (d := 1) (fun t : ℕ => 1 / Real.sqrt t) (fun _ => 0) (1 / 2)
    (∀ t (a b : Vec 1), b ≤ amsgradRule t a b) ∧ (0 : ℝ) ≤ 1 ∧
      S.α = (fun t : ℕ => 1 / Real.sqrt t) ∧ (∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ 0) ∧
      (0 : ℝ) < 1 ∧ 0 < S.β₂ ∧ S.β₂ < 1 ∧ 0 / Real.sqrt S.β₂ < 1 :=
  ⟨fun _ _ _ => le_sup_right, zero_le_one, rfl, fun _ _ => ⟨le_rfl, le_rfl⟩, one_pos,
    by norm_num [zeroSetup], by norm_num [zeroSetup], by simp⟩

end AMSGrad
end Transformer
