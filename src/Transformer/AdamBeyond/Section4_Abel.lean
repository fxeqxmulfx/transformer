import Transformer.AMSGrad.Section3_Issue

/-
# Adam and beyond — §4: Abel summation with a varying `β_{1,t}`

The scalar inequality behind the first term of Theorem 4 of arXiv:1904.09237:
Abel summation of `Σ_t √v̂_{t,i}/(2α_t(1-β_{1,t})) ((x_{t,i}-x*ᵢ)² - (x_{t+1,i}-x*ᵢ)²)`
together with the `β_{1,t}` term of `prepare_lem`, one coordinate at a time.

Source: arXiv:1904.09237, appendix, §"Proof of Theorem 4".
-/

open Finset

namespace Transformer
namespace AdamBeyond

open AMSGrad

/-- `Σ_{t=2}^T (a_t - a_{t-1}) = a_T - a_1`.
arXiv:1904.09237, appendix, §"Proof of Theorem 4". -/
theorem sum_Icc_two_sub (a : ℕ → ℝ) {T : ℕ} (hT : 1 ≤ T) :
    ∑ t ∈ Icc 2 T, (a t - a (t - 1)) = a T - a 1 := by
  induction T, hT using Nat.le_induction with
  | base => simp
  | succ n hn ih =>
    rw [sum_Icc_succ_top (by omega), ih, Nat.add_sub_cancel]; ring

/-- Abel summation with a varying `β_t`, one coordinate: for `a_t ≥ 0`
non-decreasing, `0 ≤ β_t ≤ β < 1` and `0 ≤ u_t ≤ E`,

  `Σ_{t=1}^T a_t/(2(1-β_t)) (u_t - u_{t+1}) + Σ_{t=2}^T β_t a_{t-1}/(2(1-β)) u_t`
  `  ≤ E a_T/(2(1-β)) + E/(1-β) Σ_{t=2}^T β_t a_{t-1}`.

Source: arXiv:1904.09237, appendix, §"Proof of Theorem 4". -/
theorem abel_beta_le (a b u : ℕ → ℝ) {β E : ℝ} {T : ℕ} (hT : 1 ≤ T) (hβ : β < 1)
    (ha : ∀ t ∈ Icc 1 T, 0 ≤ a t) (hmono : ∀ t ∈ Icc 2 T, a (t - 1) ≤ a t)
    (hb : ∀ t ∈ Icc 1 T, 0 ≤ b t ∧ b t ≤ β) (hu : ∀ t ∈ Icc 1 (T + 1), 0 ≤ u t ∧ u t ≤ E) :
    ∑ t ∈ Icc 1 T, a t / (2 * (1 - b t)) * (u t - u (t + 1))
        + ∑ t ∈ Icc 2 T, b t * a (t - 1) / (2 * (1 - β)) * u t ≤
      E * a T / (2 * (1 - β)) + E / (1 - β) * ∑ t ∈ Icc 2 T, b t * a (t - 1) := by
  have h1 : (1 : ℕ) ∈ Icc 1 T := mem_Icc.2 ⟨le_rfl, hT⟩
  have hβ0 : 0 ≤ β := (hb 1 h1).1.trans (hb 1 h1).2
  have hk : 0 < 2 * (1 - β) := by linarith
  have hstep : ∀ t ∈ Icc 2 T, u t * (a t / (2 * (1 - b t)) - a (t - 1) / (2 * (1 - b (t - 1))))
      + b t * a (t - 1) / (2 * (1 - β)) * u t ≤
      E * (a t - a (t - 1)) / (2 * (1 - β)) + E / (1 - β) * (b t * a (t - 1)) := by
    intro t ht
    obtain ⟨ht2, htT⟩ := mem_Icc.1 ht
    have hat := ha (t - 1) (mem_Icc.2 ⟨by omega, by omega⟩)
    obtain ⟨hbt, hbt'⟩ := hb t (mem_Icc.2 ⟨by omega, htT⟩)
    obtain ⟨hbs, hbs'⟩ := hb (t - 1) (mem_Icc.2 ⟨by omega, by omega⟩)
    obtain ⟨hu0, huE⟩ := hu t (mem_Icc.2 ⟨by omega, by omega⟩)
    have hm := hmono t ht
    have hnum : 0 ≤ a t - a (t - 1) + b t * a (t - 1) := by nlinarith
    have hp : 0 < 2 * (1 - b t) := by linarith
    have e1 : a t / (2 * (1 - b t)) - a (t - 1) / 2 =
        (a t - a (t - 1) + b t * a (t - 1)) / (2 * (1 - b t)) := by
      have : 1 - b t ≠ 0 := by linarith
      field_simp; ring
    have c1 : (a t - a (t - 1) + b t * a (t - 1)) / (2 * (1 - b t)) ≤
        (a t - a (t - 1) + b t * a (t - 1)) / (2 * (1 - β)) :=
      div_le_div_of_nonneg_left hnum hk (by linarith)
    have c2 : a (t - 1) / 2 ≤ a (t - 1) / (2 * (1 - b (t - 1))) :=
      div_le_div_of_nonneg_left hat (by linarith) (by linarith)
    obtain ⟨X, hX⟩ : ∃ X, X = (a t - a (t - 1) + b t * a (t - 1)) / (2 * (1 - β)) := ⟨_, rfl⟩
    have hX0 : 0 ≤ X := hX ▸ div_nonneg hnum hk.le
    have hdiff : a t / (2 * (1 - b t)) - a (t - 1) / (2 * (1 - b (t - 1))) ≤ X := by
      rw [hX]; linarith
    have q1 : u t * (a t / (2 * (1 - b t)) - a (t - 1) / (2 * (1 - b (t - 1)))) ≤ E * X :=
      (mul_le_mul_of_nonneg_left hdiff hu0).trans (mul_le_mul_of_nonneg_right huE hX0)
    obtain ⟨Y, hY⟩ : ∃ Y, Y = b t * a (t - 1) / (2 * (1 - β)) := ⟨_, rfl⟩
    have hY0 : 0 ≤ Y := hY ▸ div_nonneg (mul_nonneg hbt hat) hk.le
    have q2 : Y * u t ≤ Y * E := mul_le_mul_of_nonneg_left huE hY0
    have e2 : E * (a t - a (t - 1)) / (2 * (1 - β)) + E / (1 - β) * (b t * a (t - 1)) =
        E * X + Y * E := by
      rw [hX, hY]; field_simp; ring
    rw [e2, ← hY]
    linarith
  have hsum := sum_le_sum hstep
  have eR : ∑ t ∈ Icc 2 T, (E * (a t - a (t - 1)) / (2 * (1 - β))
      + E / (1 - β) * (b t * a (t - 1))) =
      E * (a T - a 1) / (2 * (1 - β)) + E / (1 - β) * ∑ t ∈ Icc 2 T, b t * a (t - 1) := by
    rw [sum_add_distrib, ← mul_sum, ← sum_div, ← mul_sum, sum_Icc_two_sub a hT]
  rw [eR, sum_add_distrib] at hsum
  rw [abel_eq (fun t => a t / (2 * (1 - b t))) u hT]
  obtain ⟨hb1, hb1'⟩ := hb 1 h1
  obtain ⟨hu1, hu1'⟩ := hu 1 (mem_Icc.2 ⟨le_rfl, by omega⟩)
  have ha1 := ha 1 h1
  have hc1 : a 1 / (2 * (1 - b 1)) * u 1 ≤ E * a 1 / (2 * (1 - β)) := by
    rw [mul_comm E, mul_div_right_comm]
    exact mul_le_mul (div_le_div_of_nonneg_left ha1 hk (by linarith)) hu1' hu1
      (div_nonneg ha1 hk.le)
  have hcT : 0 ≤ a T / (2 * (1 - b T)) * u (T + 1) :=
    mul_nonneg (div_nonneg (ha T (mem_Icc.2 ⟨hT, le_rfl⟩))
      (by linarith [(hb T (mem_Icc.2 ⟨hT, le_rfl⟩)).2]))
      (hu (T + 1) (mem_Icc.2 ⟨by omega, le_rfl⟩)).1
  have e3 : E * (a T - a 1) / (2 * (1 - β)) =
      E * a T / (2 * (1 - β)) - E * a 1 / (2 * (1 - β)) := by ring
  rw [e3] at hsum
  linarith

/-- The hypotheses of `abel_beta_le` are satisfiable: all sequences zero. -/
example : (1 : ℕ) ≤ 1 ∧ (0 : ℝ) < 1 ∧ (∀ t ∈ Icc 1 1, (0 : ℝ) ≤ (fun _ : ℕ => 0) t) ∧
    (∀ t ∈ Icc 2 1, (fun _ : ℕ => (0 : ℝ)) (t - 1) ≤ (fun _ : ℕ => 0) t) ∧
    (∀ t ∈ Icc 1 1, (0 : ℝ) ≤ (fun _ : ℕ => 0) t ∧ (fun _ : ℕ => (0 : ℝ)) t ≤ 0) ∧
    (∀ t ∈ Icc 1 (1 + 1), (0 : ℝ) ≤ (fun _ : ℕ => 0) t ∧ (fun _ : ℕ => (0 : ℝ)) t ≤ 0) :=
  ⟨le_rfl, one_pos, fun _ _ => le_rfl, fun _ _ => le_rfl, fun _ _ => ⟨le_rfl, le_rfl⟩,
    fun _ _ => ⟨le_rfl, le_rfl⟩⟩

end AdamBeyond
end Transformer
