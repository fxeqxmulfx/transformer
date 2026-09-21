import Transformer.AMSGrad.Section3_Issue
import Transformer.AMSGrad.Section4_Lemmas

/-
# AMSGrad — the first term of Lemma 3.1 under Lemma 4.3

§4 of arXiv:1904.03590v4, proof of Theorem 4.1: the bound (eqmain2) on the
first term (eqmain) of Lemma 3.1, from the Abel summation (eqtemp2) and the
`t₀` of Lemma 4.3.

**What the source says and what is carried here.**

* The source splits the Abel sum at `t₀` and uses `1 ≤ t₀ ≤ T`.  Here `t₀` is
  fixed before `T` (see `Section4_Theorem`), so `T ≤ t₀` is allowed; then the
  sum over `t₀ < t ≤ T` is empty, and the bound still holds, since every term
  of `Σ_{t=1}^{t₀} √t` is non-negative.  The split is carried as an induction
  on `T`, `telescope_le`.

Source: arXiv:1904.03590v4, §4, proof of Theorem 4.1, (eqmain2).
-/

open Finset

namespace Transformer
namespace AMSGrad

variable {d : ℕ}

/-- The telescoping of (eqmain2).  For `c_t ≥ 0` (`t ≥ 1`), non-decreasing past
`t₀`, and `0 ≤ e_t ≤ K`,

  `c₁e₁ + Σ_{t=2}^T e_t(c_t - c_{t-1}) ≤ K (Σ_{1 ≤ t ≤ T, t ≤ t₀} c_t + [t₀ < T] c_T)`.

Source: arXiv:1904.03590v4, §4, proof of Theorem 4.1, (eqmain2). -/
theorem telescope_le {c e : ℕ → ℝ} {K : ℝ} {t₀ : ℕ} (hc : ∀ t, 1 ≤ t → 0 ≤ c t)
    (he : ∀ t, 0 ≤ e t ∧ e t ≤ K) (hmono : ∀ t, t₀ < t → c (t - 1) ≤ c t)
    {T : ℕ} (hT : 1 ≤ T) :
    c 1 * e 1 + ∑ t ∈ Icc 2 T, e t * (c t - c (t - 1)) ≤
      K * (∑ t ∈ Icc 1 T, (if t ≤ t₀ then c t else 0) + if t₀ < T then c T else 0) := by
  induction T, hT using Nat.le_induction with
  | base =>
    have h1 := mul_le_mul_of_nonneg_left (he 1).2 (hc 1 le_rfl)
    rw [Icc_self, sum_singleton, Icc_eq_empty (by norm_num), sum_empty]
    split_ifs <;> first | omega | nlinarith
  | succ n hn ih =>
    rw [sum_Icc_succ_top (by omega), sum_Icc_succ_top (by omega), Nat.add_sub_cancel]
    have hcn := hc n hn
    have hc' := hc (n + 1) (by omega)
    have he' := he (n + 1)
    have h1 := mul_le_mul_of_nonneg_left he'.2 hc'
    have h2 := mul_nonneg he'.1 hcn
    by_cases h : n + 1 ≤ t₀
    · rw [ite_eq_right (by omega)] at ih
      rw [ite_eq_left h, ite_eq_right (by omega)]
      nlinarith
    · have hm := hmono (n + 1) (by omega)
      rw [Nat.add_sub_cancel] at hm
      rw [ite_eq_right h, ite_eq_left (by omega)]
      by_cases h' : t₀ < n
      · rw [ite_eq_left h'] at ih
        nlinarith [mul_nonneg (sub_nonneg.2 he'.2) (sub_nonneg.2 hm)]
      · rw [ite_eq_right h'] at ih
        nlinarith

/-- The hypotheses of `telescope_le` are satisfiable: `c = e = K = 0`. -/
example : (∀ t : ℕ, 1 ≤ t → (0 : ℝ) ≤ (fun _ => 0) t) ∧
    (∀ t : ℕ, (0 : ℝ) ≤ (fun _ => 0) t ∧ (fun _ => (0 : ℝ)) t ≤ 0) ∧
    (∀ t : ℕ, 1 < t → (fun _ => (0 : ℝ)) (t - 1) ≤ (fun _ => 0) t) ∧ 1 ≤ 1 :=
  ⟨fun _ _ => le_rfl, fun _ => ⟨le_rfl, le_rfl⟩, fun _ _ => le_rfl, le_rfl⟩

/-- **(eqmain2).**  Under AMSGrad with `α_t = α/√t`, `0 ≤ β_{1,t} ≤ β₁ < 1`,
`0 ≤ β₂ ≤ 1`, and a `t₀` as in Lemma 4.3, for all `T ≥ 1` and `x* ∈ F`,

  `Σᵢ Σ_{t=1}^T √v̂_{t,i}/(2α_t(1-β_{1,t})) ((x_{t,i} - x*ᵢ)² - (x_{t+1,i} - x*ᵢ)²)`
  `    ≤ dD²G/(2α(1-β₁)) (Σ_{t=1}^{t₀} √t + √T)`.

Source: arXiv:1904.03590v4, §4, proof of Theorem 4.1, (eqmain2). -/
theorem eqmain_le {S : Setup d} {F : Set (Vec d)} {D G : ℝ} (hS : IsOnlineConvex S F D G)
    {α β₁ : ℝ} (hα : 0 < α) (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₁ : ∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ β₁) (hβ₁' : β₁ < 1)
    (hβ₂ : 0 ≤ S.β₂) (hβ₂' : S.β₂ ≤ 1) {t₀ : ℕ}
    (ht₀ : ∀ t, t₀ < t → ∀ i : Fin d,
      Real.sqrt (((t - 1 : ℕ) : ℝ) * S.vhat amsgradRule (t - 1) i) / (1 - S.β₁ (t - 1)) ≤
        Real.sqrt ((t : ℝ) * S.vhat amsgradRule t i) / (1 - S.β₁ t))
    {T : ℕ} (hT : 1 ≤ T) {xstar : Vec d} (hx : xstar ∈ F) :
    ∑ i, ∑ t ∈ Icc 1 T, Real.sqrt (S.vhat amsgradRule t i) / (2 * S.α t * (1 - S.β₁ t)) *
        ((S.x amsgradRule t i - xstar i) ^ 2 - (S.x amsgradRule (t + 1) i - xstar i) ^ 2) ≤
      d * D ^ 2 * G / (2 * α * (1 - β₁)) *
        (∑ t ∈ Icc (1 : ℕ) t₀, Real.sqrt t + Real.sqrt T) := by
  have hB : 0 < 1 - β₁ := by linarith
  set B := 1 - β₁
  have hb : ∀ t, 1 ≤ t → 0 < 1 - S.β₁ t := fun t ht => by linarith [(hβ₁ t ht).2]
  calc _ ≤ ∑ _i : Fin d, D ^ 2 * G / (2 * α * B) *
        (∑ t ∈ Icc (1 : ℕ) t₀, Real.sqrt t + Real.sqrt T) := sum_le_sum fun i _ => ?_
    _ = _ := by rw [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]; ring
  have hG : 0 ≤ G := (abs_nonneg _).trans (hS.grad_le 0 _ hS.x₁_mem i)
  set c : ℕ → ℝ := fun t =>
    Real.sqrt ((t : ℝ) * S.vhat amsgradRule t i) / (α * (1 - S.β₁ t)) with hc_def
  set e : ℕ → ℝ := fun t => (S.x amsgradRule t i - xstar i) ^ 2 with he_def
  have hterm : ∀ t ∈ Icc 1 T, Real.sqrt (S.vhat amsgradRule t i) / (2 * S.α t * (1 - S.β₁ t)) *
      ((S.x amsgradRule t i - xstar i) ^ 2 - (S.x amsgradRule (t + 1) i - xstar i) ^ 2) =
      c t / 2 * (e t - e (t + 1)) := by
    intro t ht
    have h0 : 0 < Real.sqrt (t : ℝ) :=
      Real.sqrt_pos.2 (Nat.cast_pos.2 (mem_Icc.1 ht).1)
    have h1 := hb t (mem_Icc.1 ht).1
    simp only [hc_def, he_def, hαt, Real.sqrt_mul (Nat.cast_nonneg t)]
    field_simp
  rw [sum_congr rfl hterm]
  have hc0 : ∀ t, 1 ≤ t → 0 ≤ c t := fun t ht =>
    div_nonneg (Real.sqrt_nonneg _) (mul_nonneg hα.le (hb t ht).le)
  have he : ∀ t, 0 ≤ e t ∧ e t ≤ D ^ 2 := fun t => by
    refine ⟨sq_nonneg _, ?_⟩
    have := hS.diam _ (x_mem hS amsgradRule t) _ hx i
    simp only [he_def]
    rw [← sq_abs]
    exact pow_le_pow_left₀ (abs_nonneg _) this 2
  have hmono : ∀ t, t₀ < t → c (t - 1) ≤ c t := fun t ht => by
    simp only [hc_def, div_mul_eq_div_div_swap]
    exact div_le_div_of_nonneg_right (ht₀ t ht i) hα.le
  have hcb : ∀ t, 1 ≤ t → c t ≤ G / (α * B) * Real.sqrt t := fun t ht => by
    simp only [hc_def]
    rw [Real.sqrt_mul (Nat.cast_nonneg t), mul_comm (G / (α * B)), ← mul_div_assoc]
    exact div_le_div₀ (by positivity)
      (mul_le_mul_of_nonneg_left (vt hS hβ₂ hβ₂' t i) (Real.sqrt_nonneg _))
      (mul_pos hα hB) (mul_le_mul_of_nonneg_left (by linarith [(hβ₁ t ht).2]) hα.le)
  have hab := abel_le c e hT (hc0 T hT) (he (T + 1)).1
  have htel := telescope_le hc0 he hmono hT
  have hs1 : ∑ t ∈ Icc 1 T, (if t ≤ t₀ then c t else 0) ≤
      G / (α * B) * ∑ t ∈ Icc (1 : ℕ) t₀, Real.sqrt t := by
    rw [← sum_filter, mul_sum]
    refine (sum_le_sum_of_subset_of_nonneg (fun t ht => ?_)
      fun t ht _ => hc0 t (mem_Icc.1 ht).1).trans
        (sum_le_sum fun t ht => hcb t (mem_Icc.1 ht).1)
    simp only [mem_filter, mem_Icc] at ht ⊢
    omega
  have hs2 : (if t₀ < T then c T else 0) ≤ G / (α * B) * Real.sqrt T := by
    split_ifs
    · exact hcb T hT
    · exact mul_nonneg (div_nonneg hG (mul_pos hα hB).le) (Real.sqrt_nonneg _)
  have h3 := mul_le_mul_of_nonneg_left (add_le_add hs1 hs2) (sq_nonneg D)
  have e1 : D ^ 2 * G / (2 * α * B) * (∑ t ∈ Icc (1 : ℕ) t₀, Real.sqrt t + Real.sqrt T) =
      1 / 2 * (D ^ 2 * (G / (α * B) * ∑ t ∈ Icc (1 : ℕ) t₀, Real.sqrt t +
        G / (α * B) * Real.sqrt T)) := by
    field_simp
  rw [e1]
  linarith

/-- The hypotheses of `eqmain_le` are satisfiable: the zero cost on `[-1, 1]`,
`α = 1`, `β_{1,t} = β₁ = 0`, `β₂ = 1/2`, the `t₀` of Lemma 4.3, `T = 1`,
`x* = 0`. -/
example :
    let S := zeroSetup (d := 1) (fun t : ℕ => 1 / Real.sqrt t) (fun _ => 0) (1 / 2)
    IsOnlineConvex S (Set.Icc (fun _ => -1) (fun _ => 1)) 2 0 ∧ (0 : ℝ) < 1 ∧
      S.α = (fun t : ℕ => 1 / Real.sqrt t) ∧ (∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ 0) ∧
      (0 : ℝ) < 1 ∧ 0 ≤ S.β₂ ∧ S.β₂ ≤ 1 ∧
      (∃ t₀ : ℕ, ∀ t, t₀ < t → ∀ i : Fin 1,
        Real.sqrt (((t - 1 : ℕ) : ℝ) * S.vhat amsgradRule (t - 1) i) / (1 - S.β₁ (t - 1)) ≤
          Real.sqrt ((t : ℝ) * S.vhat amsgradRule t i) / (1 - S.β₁ t)) ∧
      1 ≤ 1 ∧ (0 : Vec 1) ∈ Set.Icc (fun _ => -1) (fun _ => 1) :=
  ⟨isOnlineConvex_zero _ _ _, one_pos, rfl, fun _ _ => ⟨le_rfl, le_rfl⟩, one_pos,
    by norm_num [zeroSetup], by norm_num [zeroSetup],
    (t_0_inv (S := zeroSetup (d := 1) (fun t : ℕ => 1 / Real.sqrt t) (fun _ => 0) (1 / 2))
      le_rfl zero_lt_one fun _ => by simp [zeroSetup]).imp fun _ h => h.2,
    le_rfl, fun _ => by norm_num, fun _ => by norm_num⟩

end AMSGrad
end Transformer
