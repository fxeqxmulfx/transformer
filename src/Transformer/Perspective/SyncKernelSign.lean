module

public import Transformer.Perspective.SyncKernelQuartic

/-!
# The sign of the quartic `p` on `[-1, 1]`

Polyanskiy, Rigollet, Yao — arXiv:2507.22857v1, *Synchronization of mean-field
models on the circle*, Appendices A–B, proofs of `lemma:positiveregion` and
`lemma:positiveregion2`.

The values of the quartic `p` (`Perspective.SyncKernelQuartic`) at the source's
points change sign four times, so the four roots in between are all the roots of
`p`, and the sign of `p` on `[-1, 1]` follows.

* `β > 1/3`: `p(-2 - 6/β) < 0 < p(-1 - 2/β)`, `p(-1) < 0 < p(1 - 0.3/β)` and
  `p(1) < 0` give roots `s₁ < s₂ < -1 < r₁ < r₂ < 1`; on `[-1, 1]`, `p > 0`
  exactly on `(r₁, r₂)` (`exists_kernelQuartic_roots`).
* `0 < β ≤ 1/3`: `p(-2 - 6/β) < 0 < p(-1 - 2/β)`, `p(-1 - 1/β) < 0 ≤ p(-1)`,
  `p(0) > 0` and `p(1) < 0` give roots `s₁ < s₂ < s₃ ≤ -1 < 0 < r < 1`; on
  `[-1, 1]`, `p ≥ 0` exactly on `[-1, r]` (`exists_kernelQuartic_root`).

The source takes the lowest root from `p(z) → -∞` as `z → -∞`; the point
`-2 - 6/β`, where `p < 0`, stands for that limit.
-/

@[expose] public section

open Real

namespace Transformer
namespace Perspective

/-- **`β > 1/3`: the sign of `p`.**  `p` has roots `-1 < r₁ < r₂ < 1`, and for
`z ∈ [-1, 1]`, `p(z) > 0` iff `r₁ < z < r₂`.

Source: arXiv:2507.22857v1, Appendix A, proof of `lemma:positiveregion`. -/
theorem exists_kernelQuartic_roots {β : ℝ} (hβ : 1 / 3 < β) :
    ∃ r₁ r₂, -1 < r₁ ∧ r₁ < r₂ ∧ r₂ < 1 ∧ kernelQuartic β r₁ = 0 ∧
      kernelQuartic β r₂ = 0 ∧
      ∀ z ∈ Set.Icc (-1 : ℝ) 1, 0 < kernelQuartic β z ↔ r₁ < z ∧ z < r₂ := by
  have hβ0 : 0 < β := by linarith
  have h2 : 0 < 2 / β := by positivity
  have h6 : 6 / β = 3 * (2 / β) := by ring
  have hm0 : 0 < 3 / (10 * β) := by positivity
  have hm2 : 3 / (10 * β) < 2 := by
    rw [div_lt_iff₀ (by positivity)]
    linarith
  have hp1 : kernelQuartic β 1 < 0 := by rw [kernelQuartic_one]; linarith
  have hpn1 : kernelQuartic β (-1) < 0 := by rw [kernelQuartic_neg_one]; linarith
  have hpa := kernelQuartic_neg_one_sub_two_div_pos hβ0
  have hpm := kernelQuartic_one_sub_div_pos hβ
  obtain ⟨s₁, hs₁, hs₁0⟩ := exists_kernelQuartic_eq_zero β
    (show -2 - 6 / β ≤ -1 - 2 / β by linarith)
    (mul_neg_of_neg_of_pos (kernelQuartic_neg_two_sub_lt hβ0) hpa)
  obtain ⟨s₂, hs₂, hs₂0⟩ := exists_kernelQuartic_eq_zero β
    (show -1 - 2 / β ≤ -1 by linarith) (mul_neg_of_pos_of_neg hpa hpn1)
  obtain ⟨r₁, hr₁, hr₁0⟩ := exists_kernelQuartic_eq_zero β
    (show (-1 : ℝ) ≤ 1 - 3 / (10 * β) by linarith) (mul_neg_of_neg_of_pos hpn1 hpm)
  obtain ⟨r₂, hr₂, hr₂0⟩ := exists_kernelQuartic_eq_zero β
    (show 1 - 3 / (10 * β) ≤ 1 by linarith) (mul_neg_of_pos_of_neg hpm hp1)
  -- the roots of `p` in `[-1, ∞)` are `r₁` and `r₂`
  have hzero : ∀ w, -1 ≤ w → kernelQuartic β w = 0 → w = r₁ ∨ w = r₂ := by
    intro w hw hw0
    rcases eq_of_kernelQuartic_eq_zero hβ0 (hs₁.2.trans hs₂.1) (hs₂.2.trans hr₁.1)
      (hr₁.2.trans hr₂.1) hs₁0 hs₂0 hr₁0 hr₂0 hw0 with h | h | h | h
    · exfalso; linarith [hs₁.2, hs₂.1, hs₂.2]
    · exfalso; linarith [hs₂.2]
    · exact Or.inl h
    · exact Or.inr h
  refine ⟨r₁, r₂, hr₁.1, hr₁.2.trans hr₂.1, hr₂.2, hr₁0, hr₂0, fun z hz =>
    ⟨fun hpz => ⟨?_, ?_⟩, fun hzr => ?_⟩⟩
  · -- otherwise a root in `(-1, z)`
    refine lt_of_not_ge fun h => ?_
    obtain ⟨w, hw, hw0⟩ := exists_kernelQuartic_eq_zero β hz.1 (mul_neg_of_neg_of_pos hpn1 hpz)
    rcases hzero w hw.1.le hw0 with h' | h' <;> linarith [hw.2, hr₁.2, hr₂.1]
  · -- otherwise a root in `(z, 1)`
    refine lt_of_not_ge fun h => ?_
    obtain ⟨w, hw, hw0⟩ := exists_kernelQuartic_eq_zero β hz.2 (mul_neg_of_pos_of_neg hpz hp1)
    rcases hzero w (by linarith [hw.1, hz.1]) hw0 with h' | h' <;>
      linarith [hw.1, hr₁.2, hr₂.1]
  · -- otherwise a root between `z` and `1 - 0.3/β`
    refine lt_of_not_ge fun h => ?_
    have hpz : kernelQuartic β z < 0 := lt_of_le_of_ne h fun h0 => by
      rcases hzero z hz.1 h0 with h' | h' <;> linarith [hzr.1, hzr.2]
    rcases le_total z (1 - 3 / (10 * β)) with hzm | hzm
    · obtain ⟨w, hw, hw0⟩ := exists_kernelQuartic_eq_zero β hzm (mul_neg_of_neg_of_pos hpz hpm)
      rcases hzero w (by linarith [hw.1, hz.1]) hw0 with h' | h' <;>
        linarith [hw.1, hw.2, hzr.1, hr₂.1]
    · obtain ⟨w, hw, hw0⟩ := exists_kernelQuartic_eq_zero β hzm (mul_neg_of_pos_of_neg hpm hpz)
      rcases hzero w (by linarith [hw.1, hr₁.1, hr₁.2]) hw0 with h' | h' <;>
        linarith [hw.1, hw.2, hzr.2, hr₁.2]

/-- **`0 < β ≤ 1/3`: the sign of `p`.**  `p` has a root `r ∈ (0, 1)`, and for
`z ∈ [-1, 1]`, `p(z) ≥ 0` iff `z ≤ r`.

Source: arXiv:2507.22857v1, Appendix B, proof of `lemma:positiveregion2`. -/
theorem exists_kernelQuartic_root {β : ℝ} (hβ : 0 < β) (hβ' : β ≤ 1 / 3) :
    ∃ r, 0 < r ∧ r < 1 ∧ kernelQuartic β r = 0 ∧
      ∀ z ∈ Set.Icc (-1 : ℝ) 1, 0 ≤ kernelQuartic β z ↔ z ≤ r := by
  have h1 : 0 < 1 / β := by positivity
  have h2 : 2 / β = 2 * (1 / β) := by ring
  have h6 : 6 / β = 6 * (1 / β) := by ring
  have hp1 : kernelQuartic β 1 < 0 := by rw [kernelQuartic_one]; linarith
  have hpn1 : 0 ≤ kernelQuartic β (-1) := by rw [kernelQuartic_neg_one]; linarith
  have hp0 : 0 < kernelQuartic β 0 := by
    rw [kernelQuartic_zero]
    nlinarith [mul_pos hβ (show (0 : ℝ) < 4 - β ^ 2 by nlinarith)]
  have hpa := kernelQuartic_neg_one_sub_two_div_pos hβ
  have hpc := kernelQuartic_neg_one_sub_inv_lt hβ hβ'
  obtain ⟨s₁, hs₁, hs₁0⟩ := exists_kernelQuartic_eq_zero β
    (show -2 - 6 / β ≤ -1 - 2 / β by linarith)
    (mul_neg_of_neg_of_pos (kernelQuartic_neg_two_sub_lt hβ) hpa)
  obtain ⟨s₂, hs₂, hs₂0⟩ := exists_kernelQuartic_eq_zero β
    (show -1 - 2 / β ≤ -1 - 1 / β by linarith) (mul_neg_of_pos_of_neg hpa hpc)
  obtain ⟨s₃, hs₃, hs₃0⟩ := intermediate_value_Ioc (show -1 - 1 / β ≤ -1 by linarith)
    (continuous_kernelQuartic β).continuousOn ⟨hpc, hpn1⟩
  obtain ⟨r, hr, hr0⟩ := exists_kernelQuartic_eq_zero β zero_le_one
    (mul_neg_of_pos_of_neg hp0 hp1)
  -- the only root of `p` in `(-1, ∞)` is `r`
  have hzero : ∀ w, -1 < w → kernelQuartic β w = 0 → w = r := by
    intro w hw hw0
    rcases eq_of_kernelQuartic_eq_zero hβ (hs₁.2.trans hs₂.1) (hs₂.2.trans hs₃.1)
      (hs₃.2.trans_lt (by linarith [hr.1])) hs₁0 hs₂0 hs₃0 hr0 hw0 with h | h | h | h
    · exfalso; linarith [hs₁.2, hs₂.1, hs₂.2, hs₃.1, hs₃.2]
    · exfalso; linarith [hs₂.2, hs₃.1, hs₃.2]
    · exfalso; linarith [hs₃.2]
    · exact h
  refine ⟨r, hr.1, hr.2, hr0, fun z hz => ⟨fun hpz => le_of_not_gt fun hrz => ?_,
    fun hzr => le_of_not_gt fun hpz => ?_⟩⟩
  · -- otherwise a root in `(z, 1)`
    have hpz' : 0 < kernelQuartic β z :=
      lt_of_le_of_ne hpz fun h0 => by linarith [hzero z (by linarith [hr.1]) h0.symm]
    obtain ⟨w, hw, hw0⟩ := exists_kernelQuartic_eq_zero β hz.2 (mul_neg_of_pos_of_neg hpz' hp1)
    linarith [hzero w (by linarith [hw.1, hr.1]) hw0, hw.1]
  · -- otherwise a root between `z` and `0`
    have hz1 : -1 < z := lt_of_le_of_ne hz.1 fun h' => by rw [← h'] at hpz; linarith
    rcases le_total z 0 with hz0 | hz0
    · obtain ⟨w, hw, hw0⟩ := exists_kernelQuartic_eq_zero β hz0 (mul_neg_of_neg_of_pos hpz hp0)
      linarith [hzero w (by linarith [hw.1]) hw0, hw.2, hr.1]
    · obtain ⟨w, hw, hw0⟩ := exists_kernelQuartic_eq_zero β hz0 (mul_neg_of_pos_of_neg hp0 hpz)
      linarith [hzero w (by linarith [hw.1]) hw0, hw.2]

/-- The hypotheses of the two theorems above are satisfiable: `β = 1` for
`1/3 < β`, and `β = 1/4` for `0 < β ≤ 1/3`. -/
example : (1 : ℝ) / 3 < 1 ∧ (0 : ℝ) < 1 / 4 ∧ (1 : ℝ) / 4 ≤ 1 / 3 := by norm_num

end Perspective
end Transformer
