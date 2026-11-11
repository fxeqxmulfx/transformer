module

public import Transformer.Grokking.Protocol.Refutation

/-!
# The target for division fails in probability too

Source: arXiv:2201.02177v1 (Power et al.), Figure 1; `experiments/grok_division/PROTOCOL.md`.
The modes of convergence are those of `Transformer.Grokking.Protocol.Target`, and the bound
used is `uniformPerm_uncovered_le` of `Transformer.Grokking.Protocol.Refutation`.

A wrong held-out answer keeps the held-out accuracy at most `1 - 1/9312`
(`wrongAt_subset`). At every update the samples with one have at least half the probability
of the uncovered splits, a positive constant. So the held-out accuracy does not tend to `1`
in probability (`not_division_tendsToOneInProbability`), under Adam and under AdamW at every
weight decay: the weakest of the three modes fails, not only the almost sure one.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Filter Topology MeasureTheory

/-- **A wrong held-out answer keeps the held-out accuracy `1/9312` away from `1`.** -/
lemma wrongAt_subset (decay : ℝ) (t : ℕ) :
    wrongAt decay t ⊆ {ω | ENNReal.ofReal (1 / 9312) ≤
      edist (heldoutAccuracy answerLogits (fun x => residueToken x.1) (Hyper.source decay)
        divisionGrad equations (by norm_num : 4656 ≤ 97 * 96) 512 t ω) 1} := by
  rintro ω ⟨x, hx, hne⟩
  have hacc := accuracy_le_of_ne (y := fun x : Equation => residueToken x.1) hx hne
  have hpos : (0 : ℝ) < (heldout equations 4656 ω.1).card :=
    Nat.cast_pos.2 (Finset.card_pos.2 ⟨x, hx⟩)
  have hcard : ((heldout equations 4656 ω.1).card : ℝ) ≤ 9312 := by
    have h := (Finset.card_image_le (s := Finset.univ.filter fun j : Fin (97 * 96) =>
      4656 ≤ (j : ℕ)) (f := fun j => equations (ω.1 j))).trans (Finset.card_filter_le _ _)
    rw [Finset.card_univ, Fintype.card_fin] at h
    exact_mod_cast h
  have hgap : (1 : ℝ) / 9312 ≤ 1 / (heldout equations 4656 ω.1).card :=
    one_div_le_one_div_of_le hpos hcard
  rw [Set.mem_ofPred_eq, edist_dist, Real.dist_eq]
  refine ENNReal.ofReal_le_ofReal ?_
  rw [abs_sub_comm, heldoutAccuracy]
  exact (by linarith : (1 : ℝ) / 9312 ≤ 1 - _).trans (le_abs_self _)

/-- **The held-out accuracy does not tend to `1` in probability**, for Adam and for AdamW at
every weight decay. -/
theorem not_division_tendsToOneInProbability (decay : ℝ) :
    ¬ TendsToOneInProbability (law (97 * 96) 4656 (initMeasure paperSizes))
      (heldoutAccuracy answerLogits (fun x => residueToken x.1) (Hyper.source decay) divisionGrad
        equations (by norm_num : 4656 ≤ 97 * 96) 512) := fun h => by
  have hlim := h (ENNReal.ofReal (1 / 9312)) (ENNReal.ofReal_pos.2 (by norm_num))
  have hbound : ∀ t : ℕ, uniformPerm (97 * 96) uncovered ≤
      2 * law (97 * 96) 4656 (initMeasure paperSizes) {ω | ENNReal.ofReal (1 / 9312) ≤
        edist (heldoutAccuracy answerLogits (fun x => residueToken x.1) (Hyper.source decay)
          divisionGrad equations (by norm_num : 4656 ≤ 97 * 96) 512 t ω) 1} := fun t =>
    (uniformPerm_uncovered_le decay t).trans
      (mul_le_mul_right (measure_mono (wrongAt_subset decay t)) 2)
  have hzero := ge_of_tendsto' (ENNReal.Tendsto.const_mul hlim (Or.inr ENNReal.ofNat_ne_top))
    hbound
  rw [mul_zero] at hzero
  exact uniformPerm_uncovered_pos.ne' (le_antisymm hzero zero_le)

end Transformer.Grokking.Protocol
