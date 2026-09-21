import Transformer.AdamBeyond.Section3_GenBlock
import Transformer.AdamBeyond.AppendixG_Auxiliary

/-
# Adam and beyond — §3: one block of Theorem 2 returns to `x = 1`

Past the warm-up, `t ≥ C`, the steps of a block add up to at least
`N α/(2√t) - 2α/(√t √(1-β₂)(1-γ)) ≥ 0` (`gen_block_sum`): the `N` last steps of the
block have `v ≤ 2` and move up by at least `α/(2√t)` each, while all the downward
pull of the gradient `C` is at most a geometric series in `γ = β₁/√β₂`.  Since
the steps change sign once, from down to up, the projected run then ends the
block no lower than it started (`proj_1d_of_mono`), so a block that starts at
`x = 1` ends at `x = 1` (`gen_block`).

Source: arXiv:1904.09237, Appendix, proof of Theorem 2, eq:gen-claim.
-/

open Finset

namespace Transformer
namespace AdamBeyond

/-- `proj_1d` for steps that, once positive, stay positive. -/
theorem proj_1d_of_mono {a b : ℝ} {y δ : ℕ → ℝ}
    (hy : ∀ t, 1 ≤ t → y (t + 1) = max a (min b (y t + δ t))) (hy₁ : y 1 ∈ Set.Icc a b)
    {T : ℕ} (hmono : ∀ j k, 1 ≤ j → j ≤ k → k ≤ T → 0 < δ j → 0 < δ k) :
    min b (y 1 + ∑ j ∈ Icc 1 T, δ j) ≤ y (T + 1) := by
  classical
  by_cases h : ∃ j, 1 ≤ j ∧ j ≤ T ∧ 0 < δ j
  · have hj₀ := Nat.find_spec h
    refine proj_1d hy hy₁ (i := Nat.find h - 1) (fun j hj => ?_) (fun j hj => ?_)
    · rw [mem_Icc] at hj
      by_contra hc
      exact Nat.find_min h (show j < Nat.find h by omega) ⟨hj.1, by omega, lt_of_not_ge hc⟩
    · rw [mem_Ioc] at hj
      exact hmono _ j hj₀.1 (by omega) hj.2 hj₀.2.2
  · push Not at h
    refine proj_1d hy hy₁ (i := T) (fun j hj => ?_) (fun j hj => ?_)
    · rw [mem_Icc] at hj; exact h j hj.1 hj.2
    · rw [mem_Ioc] at hj; omega

/-- `(1 - γ) Σ_{j=1}^T γ^{j-1} = 1 - γ^T`. -/
theorem geom_Icc (γ : ℝ) (T : ℕ) : (1 - γ) * ∑ j ∈ Icc 1 T, γ ^ (j - 1) = 1 - γ ^ T := by
  induction T with
  | zero => simp
  | succ T ih => rw [sum_Icc_succ_top (by omega), mul_add, ih, Nat.add_sub_cancel, pow_succ]; ring

variable {C : ℕ} {β₁ β₂ α : ℝ}
  (hC : 1 ≤ C) (hβ₁ : 0 ≤ β₁) (hβ₁' : β₁ ≤ 1) (hβ₂0 : 0 < β₂) (hβ₂1 : β₂ < 1) (hα : 0 < α)
  {t : ℕ} (h₁ : genSlope C (t + 1) = C) (h₂ : ∀ j, 2 ≤ j → j ≤ C → genSlope C (t + j) = -1)
  (ht : C ≤ t) {N : ℕ} (hCN : C = 2 * N) (hN : β₂ ^ N * (C : ℝ) ^ 2 ≤ 1)
  (hγ1 : β₁ / Real.sqrt β₂ < 1) (hγ : 4 ≤ N * Real.sqrt (1 - β₂) * (1 - β₁ / Real.sqrt β₂))
include hC hβ₁ hβ₁' hβ₂0 hβ₂1 hα h₁ h₂ ht hCN hN hγ1 hγ

/-- The steps of a block add up to a non-negative number.
arXiv:1904.09237, Appendix, proof of Theorem 2. -/
theorem gen_block_sum : 0 ≤ ∑ j ∈ Icc 1 C, genD C β₁ β₂ α (t + j) := by
  have hβ₂ := hβ₂0.le
  have hβ₂' := hβ₂1.le
  have htpos : (0 : ℝ) < t := by exact_mod_cast (show 0 < t by omega)
  set u := α / Real.sqrt t
  set γ := β₁ / Real.sqrt β₂
  set q := Real.sqrt (1 - β₂)
  have hq : 0 < q := Real.sqrt_pos.2 (by linarith)
  have hγ0 : 0 ≤ γ := by positivity
  have hu : 0 < u := by positivity
  -- the lower bound on each step, with `α_{t+j} ≤ α/√t`
  have hlow : ∀ j ∈ Icc 1 C, α / Real.sqrt (t + j : ℕ) / Real.sqrt (genV C β₁ β₂ α (t + j)) -
      2 * u / q * γ ^ (j - 1) ≤ genD C β₁ β₂ α (t + j) := by
    intro j hj
    rw [mem_Icc] at hj
    have hd := genD_ge hC hβ₁ hβ₁' hβ₂ hβ₂' h₁ h₂ hβ₂0 hβ₂1 hα.le hj.1 hj.2
    have ha : α / Real.sqrt (t + j : ℕ) ≤ u :=
      div_le_div_of_nonneg_left hα.le (Real.sqrt_pos.2 htpos)
        (Real.sqrt_le_sqrt (by push_cast; linarith))
    have : 2 * (α / Real.sqrt (t + j : ℕ)) * γ ^ (j - 1) / q ≤ 2 * u / q * γ ^ (j - 1) := by
      calc _ = α / Real.sqrt (t + j : ℕ) * (2 * γ ^ (j - 1) / q) := by ring
        _ ≤ u * (2 * γ ^ (j - 1) / q) := mul_le_mul_of_nonneg_right ha (by positivity)
        _ = _ := by ring
    linarith
  -- the `N` last steps move up by at least `α/(2√t)` each
  have hup : ∀ j ∈ Ioc N C, u / 2 ≤
      α / Real.sqrt (t + j : ℕ) / Real.sqrt (genV C β₁ β₂ α (t + j)) := by
    intro j hj
    rw [mem_Ioc] at hj
    obtain ⟨-, v1, v2⟩ := gen_inblock hC hβ₁ hβ₁' hβ₂ hβ₂' h₁ h₂ (α := α) (j := j) (by omega) hj.2
    have hC1 : (1 : ℝ) ≤ C := by exact_mod_cast hC
    have hV0 : 0 < genV C β₁ β₂ α (t + j) :=
      lt_of_lt_of_le (mul_pos (mul_pos (pow_pos hβ₂0 _) (by linarith)) (by positivity)) v1
    have hV2 : genV C β₁ β₂ α (t + j) ≤ 2 := by
      have := pow_le_pow_of_le_one hβ₂ hβ₂' (show N ≤ j - 1 by omega)
      nlinarith
    have htj : ((t + j : ℕ) : ℝ) ≤ 2 * t := by push_cast; exact_mod_cast (by omega : t + j ≤ 2 * t)
    rw [div_div, div_div]
    have hn : (0 : ℝ) < (t + j : ℕ) := by exact_mod_cast (show 0 < t + j by omega)
    refine div_le_div_of_nonneg_left hα.le
      (mul_pos (Real.sqrt_pos.2 hn) (Real.sqrt_pos.2 hV0)) ?_
    rw [← Real.sqrt_mul (Nat.cast_nonneg _), show Real.sqrt t * 2 = Real.sqrt (t * 4) by
      rw [Real.sqrt_mul htpos.le, show (4 : ℝ) = 2 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]]
    exact Real.sqrt_le_sqrt (by nlinarith)
  have hP : N * (u / 2) ≤ ∑ j ∈ Icc 1 C,
      α / Real.sqrt (t + j : ℕ) / Real.sqrt (genV C β₁ β₂ α (t + j)) := by
    calc N * (u / 2) = (Ioc N C).card • (u / 2) := by
          rw [Nat.card_Ioc, hCN, show 2 * N - N = N by omega, nsmul_eq_mul]
      _ ≤ ∑ j ∈ Ioc N C, α / Real.sqrt (t + j : ℕ) / Real.sqrt (genV C β₁ β₂ α (t + j)) :=
          card_nsmul_le_sum _ _ _ hup
      _ ≤ _ := sum_le_sum_of_subset_of_nonneg
          (fun j hj => by rw [mem_Ioc] at hj; rw [mem_Icc]; omega)
          (fun _ _ _ => by positivity)
  have hG : ∑ j ∈ Icc 1 C, γ ^ (j - 1) ≤ 1 / (1 - γ) := by
    rw [le_div_iff₀ (by linarith), mul_comm, geom_Icc]
    linarith [pow_nonneg hγ0 C]
  have hsum := sum_le_sum hlow
  rw [sum_sub_distrib, ← mul_sum] at hsum
  have hfin : 2 * u / q * (1 / (1 - γ)) ≤ N * (u / 2) := by
    rw [mul_one_div, div_div, div_le_iff₀ (mul_pos hq (by linarith))]
    nlinarith
  have := mul_le_mul_of_nonneg_left hG (by positivity : 0 ≤ 2 * u / q)
  linarith

/-- **The claim eq:gen-claim of the proof of Theorem 2.**  A block that starts at
`x = 1` ends at `x = 1`.  arXiv:1904.09237, Appendix, proof of Theorem 2. -/
theorem gen_block (hx : genX C β₁ β₂ α t = 1) : genX C β₁ β₂ α (t + C) = 1 := by
  have hβ₂ := hβ₂0.le
  have hβ₂' := hβ₂1.le
  have key := proj_1d_of_mono (a := -1) (b := 1) (y := fun j => genX C β₁ β₂ α (t + j - 1))
    (δ := fun j => genD C β₁ β₂ α (t + j)) (T := C) (fun j hj => by
      rw [show t + (j + 1) - 1 = t + j - 1 + 1 by omega, genX_succ',
        show t + j - 1 + 1 = t + j by omega])
    (by simp [hx]) (fun j k hj hjk hkC hδ => ?_)
  · rw [show t + 1 - 1 = t by omega, hx, show t + (C + 1) - 1 = t + C by omega,
      min_eq_left (by linarith [gen_block_sum hC hβ₁ hβ₁' hβ₂0 hβ₂1 hα h₁ h₂ ht hCN hN hγ1 hγ])]
      at key
    exact le_antisymm (genX_le_one ..) key
  · have hC1 : (1 : ℝ) ≤ C := by exact_mod_cast hC
    have hM : genM C β₁ β₂ α (t + j) < 0 := by
      by_contra hc
      push Not at hc
      unfold genD at hδ
      have : 0 ≤ α / Real.sqrt (t + j : ℕ) * (genM C β₁ β₂ α (t + j) /
          Real.sqrt (genV C β₁ β₂ α (t + j))) := by positivity
      linarith
    have hMk := genM_block_anti hC hβ₁ hβ₁' hβ₂ hβ₂' h₂ (α := α) hj hjk hkC
    obtain ⟨-, v1, -⟩ := gen_inblock hC hβ₁ hβ₁' hβ₂ hβ₂' h₁ h₂ (α := α) (by omega) hkC
    have hV0 : 0 < genV C β₁ β₂ α (t + k) :=
      lt_of_lt_of_le (mul_pos (mul_pos (pow_pos hβ₂0 _) (by linarith)) (by positivity)) v1
    have hn : (0 : ℝ) < (t + k : ℕ) := by exact_mod_cast (show 0 < t + k by omega)
    unfold genD
    have : genM C β₁ β₂ α (t + k) / Real.sqrt (genV C β₁ β₂ α (t + k)) < 0 :=
      div_neg_of_neg_of_pos (by linarith) (Real.sqrt_pos.2 hV0)
    have : 0 < α / Real.sqrt (t + k : ℕ) := by positivity
    nlinarith

omit hC hβ₁ hβ₁' hβ₂0 hβ₂1 hα h₁ h₂ ht hCN hN hγ1 hγ in
/-- The hypotheses are satisfiable: `C = 10 = 2·5`, `β₁ = 0`, `β₂ = 1/4`, `α = 1`,
the first block `t = 10`. -/
example : (1 : ℕ) ≤ 10 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 / 4 ∧ (1 / 4 : ℝ) < 1 ∧
    (0 : ℝ) < 1 ∧ genSlope 10 (10 + 0 * 10 + 1) = 10 ∧
    (∀ j, 2 ≤ j → j ≤ 10 → genSlope 10 (10 + 0 * 10 + j) = -1) ∧ 10 ≤ 10 + 0 * 10 ∧
    10 = 2 * 5 ∧ (1 / 4 : ℝ) ^ 5 * ((10 : ℕ) : ℝ) ^ 2 ≤ 1 ∧ 0 / Real.sqrt (1 / 4) < (1 : ℝ) ∧
    4 ≤ ((5 : ℕ) : ℝ) * Real.sqrt (1 - 1 / 4) * (1 - 0 / Real.sqrt (1 / 4)) := by
  obtain ⟨h₁, h₂⟩ := genSlope_block (C := 10) (k := 0) (by norm_num)
  have hs : (4 / 5 : ℝ) ≤ Real.sqrt (1 - 1 / 4) := Real.le_sqrt_of_sq_le (by norm_num)
  refine ⟨by norm_num, le_rfl, by norm_num, by norm_num, by norm_num, by norm_num,
    by exact_mod_cast h₁, h₂, by norm_num, rfl, by norm_num, by simp, ?_⟩
  simp only [zero_div, sub_zero, mul_one]
  push_cast
  linarith

end AdamBeyond
end Transformer
