/-
# Metastability — the scalar function behind the collapse of a cap

In the direct proof of `thm: metastability` (arXiv:2410.06833v1, §2) the within-cap
minimum `ρ = 1 - s` obeys `ρ̇ ≥ (2/n) ρ(1 - ρ) e^{β(ρ-1)} - 2 n e^{-(1-α)β}`, and in terms of
the distance `s = 1 - ρ` to `1`,

  `(2/n) ρ(1 - ρ) e^{β(ρ-1)} = (2/n) F_β(s)`,   `F_β(s) = s (1 - s) e^{-β s}`.

This file collects the elementary facts about `F_β` used to build the subsolutions of
`Metastability.CollapseTime`:

* `collapseRate_min_le` — `F_β` is log-concave on `(0, 1)`, so `F_β ≥ min(F_β(a), F_β(b))`
  on `[a, b]`; a level that `F_β` exceeds at both ends of `[δ, 8ε]` is exceeded throughout;
* `collapseRate_ge` — `F_β(r) ≤ F_β(s)` for `1/β ≤ s ≤ r`;
* `collapseRate_phase2` — `F_β(s) ≥ (β-1)/(βe) · s` for `0 ≤ s ≤ 1/β`;
* `sub_of_rate` — a profile with `-s' ≤ F_β(s)/n`, where `F_β(s)` exceeds the leakage level
  `2n² e^{-(1-α)β}`, is a strict subsolution of the differential inequality.
-/

import Mathlib.Analysis.Convex.Jensen
import Mathlib.Analysis.Convex.SpecificFunctions.Basic

open Set Real

namespace Transformer
namespace Metastability

/-- **The collapse rate** `F_β(s) = s (1 - s) e^{-β s}`: with `s = 1 - ρ` the distance of the
within-cap minimum to `1`, the gain term of `eq: ze.equation` is `(2/n) F_β(s)`. -/
noncomputable def collapseRate (β s : ℝ) : ℝ := s * (1 - s) * Real.exp (-(β * s))

/-- **`F_β` is log-concave on `(0, 1)`**: `log F_β(s) = log s + log (1 - s) - β s` is a sum of
concave functions.  Hence `F_β(s) ≥ min (F_β(a)) (F_β(b))` for `s ∈ [a, b] ⊂ (0, 1)`. -/
theorem collapseRate_min_le {β a b s : ℝ} (ha : 0 < a) (hb : b < 1) (hs : s ∈ Icc a b) :
    min (collapseRate β a) (collapseRate β b) ≤ collapseRate β s := by
  have h1 : ConcaveOn ℝ (Ioo (0 : ℝ) 1) Real.log :=
    strictConcaveOn_log_Ioi.concaveOn.subset (fun x hx => hx.1) (convex_Ioo 0 1)
  have h2 : ConcaveOn ℝ (Ioo (0 : ℝ) 1) (fun x => Real.log (1 - x)) := by
    have h := strictConcaveOn_log_Ioi.concaveOn.comp_affineMap (AffineMap.lineMap (1 : ℝ) 0)
    refine (h.subset (fun x hx => ?_) (convex_Ioo 0 1)).congr ?_
    · simp only [mem_preimage, AffineMap.lineMap_apply_ring', mem_Ioi]
      linarith [hx.2]
    · intro x hx
      simp only [Function.comp_apply, AffineMap.lineMap_apply_ring']
      congr 1; ring
  have h3 : ConvexOn ℝ (Ioo (0 : ℝ) 1) (fun x => β * x) :=
    ⟨convex_Ioo 0 1, fun x _ y _ a b _ _ _ => le_of_eq (by simp only [smul_eq_mul]; ring)⟩
  have hg : ConcaveOn ℝ (Ioo (0 : ℝ) 1) (fun x => Real.log x + Real.log (1 - x) - β * x) :=
    ((h1.add h2).sub h3)
  have hF : ∀ x ∈ Ioo (0 : ℝ) 1, collapseRate β x
      = Real.exp (Real.log x + Real.log (1 - x) - β * x) := by
    intro x hx
    rw [Real.exp_sub, Real.exp_add, Real.exp_log hx.1, Real.exp_log (by linarith [hx.2]),
      collapseRate, Real.exp_neg, div_eq_mul_inv]
  have hmem : ∀ x, a ≤ x → x ≤ b → x ∈ Ioo (0 : ℝ) 1 := fun x h1 h2 =>
    ⟨lt_of_lt_of_le ha h1, lt_of_le_of_lt h2 hb⟩
  have hab : a ≤ b := hs.1.trans hs.2
  have hmin := hg.min_le_of_mem_Icc (hmem a le_rfl hab) (hmem b hab le_rfl) hs
  rw [hF a (hmem a le_rfl hab), hF b (hmem b hab le_rfl), hF s (hmem s hs.1 hs.2),
    ← Real.exp_monotone.map_min]
  exact Real.exp_le_exp.2 hmin

/-- **Phase 1.**  `F_β(r) ≤ F_β(s)` for `1/β ≤ s ≤ r ≤ 1`, `s > 0`: with `u = β(r - s)`,
`e^u ≥ 1 + u`, and `s (1 - s)(1 + β(r - s)) ≥ r (1 - s) ≥ r (1 - r)` because `β s ≥ 1`. -/
theorem collapseRate_ge {β s r : ℝ} (hβs : 1 ≤ β * s) (hs : 0 < s) (hsr : s ≤ r)
    (hr : r ≤ 1) : collapseRate β r ≤ collapseRate β s := by
  have he : 1 + β * (r - s) ≤ Real.exp (β * (r - s)) := by
    linarith [Real.add_one_le_exp (β * (r - s))]
  have hsplit : Real.exp (-(β * s)) = Real.exp (-(β * r)) * Real.exp (β * (r - s)) := by
    rw [← Real.exp_add]; congr 1; ring
  have hpos : 0 < Real.exp (-(β * r)) := Real.exp_pos _
  have hpoly : r * (1 - r) ≤ s * (1 - s) * (1 + β * (r - s)) := by
    have h1 : (1 - s) * (r - s) ≤ (1 - s) * (β * s) * (r - s) := by
      have : (0 : ℝ) ≤ (1 - s) * (r - s) := mul_nonneg (by linarith) (by linarith)
      nlinarith
    nlinarith
  unfold collapseRate
  rw [hsplit]
  calc r * (1 - r) * Real.exp (-(β * r))
      ≤ s * (1 - s) * (1 + β * (r - s)) * Real.exp (-(β * r)) :=
        mul_le_mul_of_nonneg_right hpoly hpos.le
    _ ≤ s * (1 - s) * Real.exp (β * (r - s)) * Real.exp (-(β * r)) := by
        have : 0 ≤ s * (1 - s) := mul_nonneg hs.le (by linarith)
        exact mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left he this) hpos.le
    _ = s * (1 - s) * (Real.exp (-(β * r)) * Real.exp (β * (r - s))) := by ring

/-- **Phase 2.**  For `β > 1` and `0 ≤ s ≤ 1/β`, `F_β(s) ≥ (β - 1)/(β e) · s`: the factor
`(1 - s) e^{-β s}` is at least `(1 - 1/β) e^{-1}`. -/
theorem collapseRate_phase2 {β s : ℝ} (hβ : 1 < β) (hs : 0 ≤ s) (hsβ : s ≤ 1 / β) :
    (β - 1) / (β * Real.exp 1) * s ≤ collapseRate β s := by
  have hβ0 : 0 < β := by linarith
  have hβs : β * s ≤ 1 := by rwa [le_div_iff₀ hβ0, mul_comm] at hsβ
  have h1 : (β - 1) / β ≤ 1 - s := by
    rw [div_le_iff₀ hβ0]; nlinarith
  have h2 : Real.exp (-1) ≤ Real.exp (-(β * s)) := Real.exp_le_exp.2 (by linarith)
  have h3 : (β - 1) / (β * Real.exp 1) ≤ (1 - s) * Real.exp (-(β * s)) := by
    calc (β - 1) / (β * Real.exp 1) = (β - 1) / β * Real.exp (-1) := by
          rw [Real.exp_neg]; field_simp
      _ ≤ (1 - s) * Real.exp (-(β * s)) :=
          mul_le_mul h1 h2 (Real.exp_pos _).le (by nlinarith)
  unfold collapseRate
  calc (β - 1) / (β * Real.exp 1) * s ≤ (1 - s) * Real.exp (-(β * s)) * s :=
        mul_le_mul_of_nonneg_right h3 hs
    _ = s * (1 - s) * Real.exp (-(β * s)) := by ring

/-- **A profile with `-s' ≤ F_β(s)/n` above the leakage level is a strict subsolution.**
If `2 n² E < F_β(s)` and `-s' ≤ F_β(s)/n`, then with `B = 1 - s`,

  `B' = -s' < (2/n) B (1 - B) e^{β(B-1)} - 2 n E`. -/
theorem sub_of_rate {n : ℕ} {β s s' E : ℝ} (hn : 1 ≤ n)
    (hF : 2 * (n : ℝ) ^ 2 * E < collapseRate β s) (hs : -s' ≤ collapseRate β s / n) :
    -s' < (2 / (n : ℝ)) * (1 - s) * (1 - (1 - s)) * Real.exp (β * ((1 - s) - 1))
      - 2 * (n : ℝ) * E := by
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  have hid : (2 / (n : ℝ)) * (1 - s) * (1 - (1 - s)) * Real.exp (β * ((1 - s) - 1))
      = (2 / (n : ℝ)) * collapseRate β s := by
    unfold collapseRate
    have : β * ((1 - s) - 1) = -(β * s) := by ring
    rw [this]; ring
  rw [hid]
  have h1 : 2 * (n : ℝ) * E < collapseRate β s / n := by
    rw [lt_div_iff₀ hn']; nlinarith
  have h2 : (2 / (n : ℝ)) * collapseRate β s = 2 * (collapseRate β s / n) := by ring
  rw [h2]
  linarith

end Metastability
end Transformer
