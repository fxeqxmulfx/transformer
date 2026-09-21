import Transformer.AMSGrad.Section3_Step

/-
# AMSGrad — the issue in the convergence proof

§3 of arXiv:1904.03590v4: the regret bound every later theorem starts from
(Lemma 3.1), and the rewriting of its first term (`eqtemp2`) that replaces the
step of Reddi et al. the paper shows to be unjustified.

**What the source says and what is carried here.**

* Lemma 3.1 is stated for Algorithm 1; §5 applies it to AdamX "similarly".
  Its proof uses of `v̂_t` only `v̂_t ≥ v_t`, so it is stated once, for every
  rule with `v_t ≤ v̂_t`, `prepare_lem`; `amsgradRule` is such a rule,
  `le_amsgradRule`.  It needs `0 < α_t`, `0 ≤ β_{1,t} ≤ β₁ < 1` and
  `0 < β₂ < 1`, which the source assumes throughout.

* `eqtemp2` is Abel summation plus the omission of a non-negative term, and
  holds for any sequences: `abel_le`, one coordinate at a time.

Source: arXiv:1904.03590v4, §3, Lemma 3.1 and (3.4).
-/

open Finset

namespace Transformer
namespace AMSGrad

variable {d : ℕ}

/-- AMSGrad's rule keeps `v̂_t ≥ v_t`. -/
theorem le_amsgradRule (t : ℕ) (a b : Vec d) : b ≤ amsgradRule t a b := le_sup_right

/-- **Lemma 3.1.**  For a rule with `v_t ≤ v̂_t` (AMSGrad, and AdamX of §5),
under the standing assumptions, for all `T ≥ 1` and `x* ∈ F`,

  `R(T) ≤ Σᵢ Σ_{t=1}^T √v̂_{t,i}/(2α_t(1-β_{1,t})) ((x_{t,i} - x*ᵢ)² - (x_{t+1,i} - x*ᵢ)²)`
  `     + Σᵢ Σ_{t=1}^T α_t/(1-β₁) m²_{t,i}/√v̂_{t,i}`
  `     + Σᵢ Σ_{t=2}^T β_{1,t}√v̂_{t-1,i}/(2α_{t-1}(1-β₁)) (x_{t,i} - x*ᵢ)²`.

Source: arXiv:1904.03590v4, §3, Lemma 3.1. -/
theorem prepare_lem {S : Setup d} {F : Set (Vec d)} {D G : ℝ} (hS : IsOnlineConvex S F D G)
    {R : Rule d} (hR : ∀ t a b, b ≤ R t a b)
    (hα : ∀ t, 1 ≤ t → 0 < S.α t)
    (hβ₁ : ∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1) (hβ₁' : S.β₁ 1 < 1)
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1)
    {T : ℕ} (hT : 1 ≤ T) {xstar : Vec d} (hxstar : xstar ∈ F) :
    S.regret R xstar T ≤
      ∑ i, ∑ t ∈ Icc 1 T, Real.sqrt (S.vhat R t i) / (2 * S.α t * (1 - S.β₁ t)) *
          ((S.x R t i - xstar i) ^ 2 - (S.x R (t + 1) i - xstar i) ^ 2)
      + ∑ i, ∑ t ∈ Icc 1 T, S.α t / (1 - S.β₁ 1) * S.m R t i ^ 2 / Real.sqrt (S.vhat R t i)
      + ∑ i, ∑ t ∈ Icc 2 T, S.β₁ t * Real.sqrt (S.vhat R (t - 1) i) /
          (2 * S.α (t - 1) * (1 - S.β₁ 1)) * (S.x R t i - xstar i) ^ 2 := by
  have hb1 : 0 < 1 - S.β₁ 1 := by linarith
  set A : ℕ → Fin d → ℝ := fun t i => Real.sqrt (S.vhat R t i) / (2 * S.α t * (1 - S.β₁ t)) *
    ((S.x R t i - xstar i) ^ 2 - (S.x R (t + 1) i - xstar i) ^ 2) with hA
  set H : ℕ → Fin d → ℝ := fun t i =>
    S.α t / (1 - S.β₁ 1) * S.m R t i ^ 2 / Real.sqrt (S.vhat R t i) with hH
  set E : ℕ → Fin d → ℝ := fun t i => S.β₁ t * Real.sqrt (S.vhat R (t - 1) i) /
    (2 * S.α (t - 1) * (1 - S.β₁ 1)) * (S.x R t i - xstar i) ^ 2 with hE
  have hH0 : ∀ i, H 0 i = 0 := fun i => by
    simp only [hH]; rw [show S.m R 0 i = 0 from rfl]; simp
  -- One step: (tvheq), with the `m_t` term halved and the `m_{t-1}` term split by Young.
  have hstep : ∀ t ∈ Icc 1 T, S.f t (S.x R t) - S.f t xstar ≤
      ∑ i, (A t i + H t i / 2 + H (t - 1) i / 2 + E t i) := by
    intro t ht
    rw [mem_Icc] at ht
    obtain ⟨hb0, hb⟩ := hβ₁ t ht.1
    have hαt := hα t ht.1
    have hc : S.f t (S.x R t) + ∑ i, S.g R t i * (xstar i - S.x R t i) ≤ S.f t xstar :=
      convex_first_order (hS.convexOn t) (hS.differentiable t) (S.x R t) xstar
    have hneg : ∑ i, S.g R t i * (S.x R t i - xstar i) =
        -∑ i, S.g R t i * (xstar i - S.x R t i) := by
      rw [← sum_neg_distrib]; exact sum_congr rfl fun i _ => by ring
    have hst := step_ineq hS hR ht.1 hαt hb0 (by linarith) hβ₂ hβ₂' hxstar
    refine le_trans (by linarith) (hst.trans (sum_le_sum fun i _ => ?_))
    have hk : 0 ≤ S.m R t i ^ 2 / Real.sqrt (S.vhat R t i) := by positivity
    have hB : S.α t / (2 * (1 - S.β₁ t)) * S.m R t i ^ 2 / Real.sqrt (S.vhat R t i) ≤
        H t i / 2 := by
      have e1 : H t i / 2 = S.α t / (2 * (1 - S.β₁ 1)) *
          (S.m R t i ^ 2 / Real.sqrt (S.vhat R t i)) := by
        simp only [hH, div_eq_mul_inv, mul_inv]; ring
      rw [e1, mul_div_assoc]
      exact mul_le_mul_of_nonneg_right
        (div_le_div_of_nonneg_left hαt.le (by linarith) (by linarith)) hk
    have hbb : S.β₁ t / (1 - S.β₁ t) ≤ S.β₁ t / (1 - S.β₁ 1) :=
      div_le_div_of_nonneg_left hb0 hb1 (by linarith)
    have hC : S.β₁ t / (1 - S.β₁ t) * S.m R (t - 1) i * (xstar i - S.x R t i) ≤
        H (t - 1) i / 2 + E t i := by
      obtain ⟨s, rfl⟩ : ∃ s, t = s + 1 := ⟨t - 1, by omega⟩
      simp only [Nat.add_sub_cancel]
      rcases Nat.eq_zero_or_pos s with rfl | hs
      · have hv : S.vhat R 0 i = 0 := rfl
        rw [show S.m R 0 i = 0 from rfl, hH0]
        simp [hE, hv]
      have hαs := hα s hs
      set w := Real.sqrt (S.vhat R s i)
      set m := S.m R s i
      set z := xstar i - S.x R (s + 1) i
      have hy := young (m := m) (z := z) hαs (Real.sqrt_nonneg _)
        (m_eq_zero_of_vhat hR hβ₂ hβ₂')
      have hm2 : 0 ≤ S.α s * m ^ 2 / (2 * w) := by positivity
      have hz2 : 0 ≤ w / (2 * S.α s) * z ^ 2 := by positivity
      have hb2 : S.β₁ (s + 1) / (1 - S.β₁ 1) ≤ 1 / (1 - S.β₁ 1) :=
        div_le_div_of_nonneg_right (by linarith) hb1.le
      have hq : 0 ≤ S.β₁ (s + 1) / (1 - S.β₁ (s + 1)) := div_nonneg hb0 (by linarith)
      have e2 : H s i / 2 = 1 / (1 - S.β₁ 1) * (S.α s * m ^ 2 / (2 * w)) := by
        simp only [hH, div_eq_mul_inv, mul_inv]; ring
      have e3 : E (s + 1) i = S.β₁ (s + 1) / (1 - S.β₁ 1) * (w / (2 * S.α s) * z ^ 2) := by
        simp only [hE, Nat.add_sub_cancel, z, div_eq_mul_inv, mul_inv]; ring
      rw [e2, e3, mul_assoc]
      calc S.β₁ (s + 1) / (1 - S.β₁ (s + 1)) * (m * z)
          ≤ S.β₁ (s + 1) / (1 - S.β₁ (s + 1)) * (w / (2 * S.α s) * z ^ 2)
            + S.β₁ (s + 1) / (1 - S.β₁ (s + 1)) * (S.α s * m ^ 2 / (2 * w)) := by
            rw [← mul_add]; exact mul_le_mul_of_nonneg_left hy hq
        _ ≤ _ := by
          nlinarith [mul_le_mul_of_nonneg_right hbb hz2,
            mul_le_mul_of_nonneg_right (hbb.trans hb2) hm2]
    linarith
  -- Sum over `t`, then shift the `m_{t-1}` terms back by one.
  show _ ≤ ∑ i, ∑ t ∈ Icc 1 T, A t i + ∑ i, ∑ t ∈ Icc 1 T, H t i +
    ∑ i, ∑ t ∈ Icc 2 T, E t i
  rw [← sum_add_distrib, ← sum_add_distrib]
  calc S.regret R xstar T
      ≤ ∑ t ∈ Icc 1 T, ∑ i, (A t i + H t i / 2 + H (t - 1) i / 2 + E t i) := sum_le_sum hstep
    _ = ∑ i, ∑ t ∈ Icc 1 T, (A t i + H t i / 2 + H (t - 1) i / 2 + E t i) := sum_comm
    _ ≤ _ := sum_le_sum fun i _ => ?_
  have hshift : ∀ n, 1 ≤ n → ∑ t ∈ Icc 1 n, H (t - 1) i + H n i = ∑ t ∈ Icc 1 n, H t i := by
    intro n hn
    induction n, hn using Nat.le_induction with
    | base => simp [hH0]
    | succ n hn ih =>
      rw [sum_Icc_succ_top (by omega), sum_Icc_succ_top (by omega), ← ih, Nat.add_sub_cancel]
  have hE1 : ∑ t ∈ Icc 1 T, E t i = ∑ t ∈ Icc 2 T, E t i := by
    rw [← add_sum_Ioc_eq_sum_Icc hT, show Ioc 1 T = Icc 2 T by ext; simp; omega]
    have hv : S.vhat R 0 i = 0 := rfl
    simp [hE, hv]
  have hHT : 0 ≤ H T i := by
    have := hα T hT
    simp only [hH]; positivity
  have := hshift T hT
  rw [sum_add_distrib, sum_add_distrib, sum_add_distrib, ← sum_div, ← sum_div, hE1]
  linarith

/-- The hypotheses of `prepare_lem` are satisfiable: the zero cost on `[-1, 1]`
under AMSGrad, `α_t = 1`, `β_{1,t} = 0`, `β₂ = 1/2`, `T = 1`, `x* = 0`. -/
example :
    let S := zeroSetup (d := 1) (fun _ => 1) (fun _ => 0) (1 / 2)
    IsOnlineConvex S (Set.Icc (fun _ => -1) (fun _ => 1)) 2 0 ∧
      (∀ t (a b : Vec 1), b ≤ amsgradRule t a b) ∧ (∀ t, 1 ≤ t → 0 < S.α t) ∧
      (∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1) ∧ S.β₁ 1 < 1 ∧ 0 < S.β₂ ∧ S.β₂ < 1 ∧
      1 ≤ 1 ∧ (0 : Vec 1) ∈ Set.Icc (fun _ => -1) (fun _ => 1) :=
  ⟨isOnlineConvex_zero _ _ _, le_amsgradRule, fun _ _ => one_pos,
    fun _ _ => ⟨le_rfl, le_rfl⟩, by simp [zeroSetup], by norm_num [zeroSetup],
    by norm_num [zeroSetup], le_rfl, fun _ => by norm_num, fun _ => by norm_num⟩

/-! ### Abel summation of the first term -/

/-- Abel summation: `Σ_{t=1}^T c_t(D_t - D_{t+1})
= c_1 D_1 + Σ_{t=2}^T D_t(c_t - c_{t-1}) - c_T D_{T+1}`.
arXiv:1904.03590v4, §3, the display before (3.4). -/
theorem abel_eq (c D : ℕ → ℝ) {T : ℕ} (hT : 1 ≤ T) :
    ∑ t ∈ Icc 1 T, c t * (D t - D (t + 1)) =
      c 1 * D 1 + ∑ t ∈ Icc 2 T, D t * (c t - c (t - 1)) - c T * D (T + 1) := by
  induction T, hT using Nat.le_induction with
  | base => simp; ring
  | succ n hn ih =>
    rw [Finset.sum_Icc_succ_top (by omega), ih, Finset.sum_Icc_succ_top (by omega)]
    simp only [Nat.add_sub_cancel]
    ring

/-- **Equation (3.4)**, one coordinate at a time: for `c_T ≥ 0` and
`D_{T+1} ≥ 0`, `Σ_{t=1}^T c_t/2 (D_t - D_{t+1})
≤ c_1/2 D_1 + 1/2 Σ_{t=2}^T D_t(c_t - c_{t-1})`.  With
`c_t = √v̂_{t,i}/(α_t(1 - β_{1,t}))` and `D_t = (x_{t,i} - x*ᵢ)²` this is
(3.4).

Source: arXiv:1904.03590v4, §3, (3.4). -/
theorem abel_le (c D : ℕ → ℝ) {T : ℕ} (hT : 1 ≤ T) (hc : 0 ≤ c T) (hD : 0 ≤ D (T + 1)) :
    ∑ t ∈ Icc 1 T, c t / 2 * (D t - D (t + 1)) ≤
      c 1 / 2 * D 1 + 1 / 2 * ∑ t ∈ Icc 2 T, D t * (c t - c (t - 1)) := by
  have h := abel_eq c D hT
  have h2 : ∑ t ∈ Icc 1 T, c t / 2 * (D t - D (t + 1))
      = 1 / 2 * ∑ t ∈ Icc 1 T, c t * (D t - D (t + 1)) := by
    rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun t _ => by ring
  rw [h2, h]
  nlinarith [mul_nonneg hc hD]

/-- The hypotheses of `abel_le` are satisfiable. -/
example : 1 ≤ 1 ∧ (0 : ℝ) ≤ (fun _ : ℕ => 0) 1 ∧ (0 : ℝ) ≤ (fun _ : ℕ => 0) 2 :=
  ⟨le_rfl, le_rfl, le_rfl⟩

end AMSGrad
end Transformer
