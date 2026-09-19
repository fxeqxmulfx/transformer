/-
# Four Over Six is biased

arXiv:2601.22813v2, §4.2 and Appendix A.

Averaging `Transformer.Quartet.Section4_Rounding.q46At_biasWitness` over the
`16` coins of the group finishes the refutation.  The value depends on the
first coin alone, so the unit cube splits into two boxes — the coin below
`14/17` and the coin above it — whose volumes are `14/17` and `3/17`, and the
mean is

    (14/17) · (17/16) + (3/17) · (51/64) = 7/8 + 9/64 = 65/64,

against an entry of `1`.  The bias is `1/64`, about `1.6%`, and it is positive:
Four Over Six systematically *overestimates* this entry.  That is what
Appendix A measures as a plateau in the concentration curve, and why §4.2
keeps the scheme on the forward pass and refuses it on the backward one.

Both branches are unbiased on their own — `integral_qSRAt_four_and_six` — so
nothing here contradicts §3.1; what the selection adds is a correlation
between the branch and the coins, and `65/64 ≠ 1` is its size.
-/

import Transformer.Quartet.Section4_Rounding
import Mathlib.MeasureTheory.Constructions.Pi

namespace Transformer
namespace Quartet

open MeasureTheory

/-- The coins on which the witness's entry comes back rounded up: the unit
cube cut at `14/17` in the first coordinate (§4.2). -/
def coinsUp : Set (Fin 16 → ℝ) :=
  Set.univ.pi fun i => if i = 0 then Set.Ico (0 : ℝ) (14 / 17) else Set.Icc (0 : ℝ) 1

/-- The coins on which it comes back rounded down: the rest of the cube. -/
def coinsDown : Set (Fin 16 → ℝ) :=
  Set.univ.pi fun i => if i = 0 then Set.Icc (14 / 17 : ℝ) 1 else Set.Icc (0 : ℝ) 1

theorem measurableSet_coinsUp : MeasurableSet coinsUp :=
  MeasurableSet.univ_pi fun i => by
    by_cases hi : i = 0
    · simp only [hi, reduceIte]; exact measurableSet_Ico
    · simp only [hi, reduceIte]; exact measurableSet_Icc

theorem measurableSet_coinsDown : MeasurableSet coinsDown :=
  MeasurableSet.univ_pi fun i => by
    by_cases hi : i = 0
    · simp only [hi, reduceIte]; exact measurableSet_Icc
    · simp only [hi, reduceIte]; exact measurableSet_Icc

/-- The probability that the first coin falls below `14/17` (§4.2). -/
theorem volume_coinsUp : volume coinsUp = ENNReal.ofReal (14 / 17) := by
  rw [coinsUp, volume_pi_pi]
  simp [apply_ite volume]

/-- And the probability that it does not. -/
theorem volume_coinsDown : volume coinsDown = ENNReal.ofReal (3 / 17) := by
  rw [coinsDown, volume_pi_pi]
  simp [apply_ite volume]
  norm_num

/-- The two boxes exhaust the cube of coins `meanGroup` integrates over. -/
theorem coinCube_eq_union :
    (Set.univ.pi fun _ : Fin 16 => Set.Icc (0 : ℝ) 1) = coinsUp ∪ coinsDown := by
  ext t
  simp only [coinsUp, coinsDown, Set.mem_union, Set.mem_pi, Set.mem_univ, true_implies]
  constructor
  · intro h
    rcases lt_or_ge (t 0) (14 / 17 : ℝ) with hlt | hge
    · refine Or.inl fun i => ?_
      by_cases hi : i = 0
      · subst hi; simpa using ⟨(h 0).1, hlt⟩
      · simpa [hi] using h i
    · refine Or.inr fun i => ?_
      by_cases hi : i = 0
      · subst hi; simpa using ⟨hge, (h 0).2⟩
      · simpa [hi] using h i
  · rintro (h | h) i
    · by_cases hi : i = 0
      · subst hi
        have h0 := h 0
        simp only [reduceIte, Set.mem_Ico] at h0
        exact Set.mem_Icc.mpr ⟨h0.1, by linarith [h0.2]⟩
      · simpa [hi] using h i
    · by_cases hi : i = 0
      · subst hi
        have h0 := h 0
        simp only [reduceIte, Set.mem_Icc] at h0
        exact Set.mem_Icc.mpr ⟨by linarith [h0.1], h0.2⟩
      · simpa [hi] using h i

/-- And they do not overlap. -/
theorem disjoint_coinsUp_coinsDown : Disjoint coinsUp coinsDown := by
  rw [Set.disjoint_left]
  intro t h1 h2
  have a1 := h1 0 (Set.mem_univ _)
  have a2 := h2 0 (Set.mem_univ _)
  simp only [reduceIte, Set.mem_Ico, Set.mem_Icc] at a1 a2
  linarith [a1.2, a2.1]

/-- On the first box the entry comes back as `17/16` (§4.2). -/
theorem eqOn_coinsUp :
    Set.EqOn (fun t : Fin 16 → ℝ => q46At biasWitness 0 0 t) (fun _ => (17 / 16 : ℝ)) coinsUp := by
  intro t ht
  have h0 := ht 0 (Set.mem_univ _)
  simp only [reduceIte, Set.mem_Ico] at h0
  simp [q46At_biasWitness t, h0.2]

/-- On the second it comes back as `51/64` (§4.2). -/
theorem eqOn_coinsDown :
    Set.EqOn (fun t : Fin 16 → ℝ => q46At biasWitness 0 0 t) (fun _ => (51 / 64 : ℝ))
      coinsDown := by
  intro t ht
  have h0 := ht 0 (Set.mem_univ _)
  simp only [reduceIte, Set.mem_Icc] at h0
  simp [q46At_biasWitness t, not_lt.mpr h0.1]

/-- **The mean of Four Over Six on the witness is `65/64`** (§4.2): the
`14/17` of the coins that round up contribute `17/16` and the remaining
`3/17` contribute `51/64`. -/
theorem meanGroup_q46At_biasWitness :
    meanGroup (fun t => q46At biasWitness 0 0 t) = 65 / 64 := by
  have hup : IntegrableOn (fun t : Fin 16 → ℝ => q46At biasWitness 0 0 t) coinsUp :=
    (integrableOn_const (by rw [volume_coinsUp]; exact ENNReal.ofReal_ne_top)).congr_fun
      eqOn_coinsUp.symm measurableSet_coinsUp
  have hdown : IntegrableOn (fun t : Fin 16 → ℝ => q46At biasWitness 0 0 t) coinsDown :=
    (integrableOn_const (by rw [volume_coinsDown]; exact ENNReal.ofReal_ne_top)).congr_fun
      eqOn_coinsDown.symm measurableSet_coinsDown
  unfold meanGroup
  rw [coinCube_eq_union,
    setIntegral_union disjoint_coinsUp_coinsDown measurableSet_coinsDown hup hdown,
    setIntegral_congr_fun measurableSet_coinsUp eqOn_coinsUp,
    setIntegral_congr_fun measurableSet_coinsDown eqOn_coinsDown,
    setIntegral_const, setIntegral_const, Measure.real, Measure.real,
    volume_coinsUp, volume_coinsDown,
    ENNReal.toReal_ofReal (by norm_num : (0 : ℝ) ≤ 14 / 17),
    ENNReal.toReal_ofReal (by norm_num : (0 : ℝ) ≤ 3 / 17)]
  norm_num

/-- **Their combination is biased** (§4.2, "it does not constitute an unbiased
estimation, as the act of picking a lower MSE scale branch introduces bias"):
some tensor has an entry whose expectation under Four Over Six is not that
entry.  This is why the paper drops the scheme from its backward pass, and
Appendix A measures the leftover bias as a plateau in the concentration
curve.

The paper offers no proof — the claim is argued in one sentence and checked by
experiment — so the witness is `biasWitness`, built here, and the expectation
it moves is `1 ↦ 65/64`. -/
theorem exists_mean_q46At_ne :
    ∃ (x : Fin (2 ^ 0) → Fin 16 → ℝ) (i : Fin (2 ^ 0)) (j : Fin 16),
      meanGroup (fun t => q46At x i j t) ≠ x i j := by
  refine ⟨biasWitness, 0, 0, ?_⟩
  rw [meanGroup_q46At_biasWitness, show biasWitness 0 0 = 1 from by simp [biasWitness]]
  norm_num

end Quartet
end Transformer
