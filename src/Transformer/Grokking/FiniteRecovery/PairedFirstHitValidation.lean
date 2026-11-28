module

public import Transformer.Grokking.FiniteRecovery.TerminalTail
public import Mathlib.Data.Fin.Tuple.Sort

/-! One-sided validation of paired first-hit times. Each sample contains
both optimizers on the same draw. Their times may depend arbitrarily on
each other. Ties and penalties for reaching the cap are retained. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

open MeasureTheory Finset
open scoped ENNReal Classical

/-- A completed miss is assigned the first grid time after the cap.
Negative differences mean that the candidate hit earlier. -/
def pairedHitDifference (cap interval : ℕ) (candidate control : Option ℕ) : ℤ :=
  (candidate.getD (cap+interval) : ℤ) - (control.getD (cap+interval) : ℤ)

variable {D α : Type*} [LinearOrder α]

/-- Zero-based order statistic of n observations, including duplicates. -/
noncomputable def pairedOrderStatistic {n : ℕ} (value : D → α) (z : ℕ → D) (rank : Fin n) : α :=
  let sample (i : Fin n) := value (z i.val)
  sample (Tuple.sort sample rank)

theorem rank_le_below_count {n : ℕ} (value : D → α) (z : ℕ → D) (rank : Fin n) (m : α)
    (h : pairedOrderStatistic value z rank < m) :
    rank.val+1 ≤ terminalPassCount {d | value d < m} (range n) z := by
  classical
  let sample (i : Fin n) := value (z i.val)
  let lift (i : Fin (rank.val+1)) : Fin n := ⟨i.val,by omega⟩
  let idx (i : Fin (rank.val+1)) := (Tuple.sort sample (lift i)).val
  have inj : Function.Injective idx := by
    intro i j hij
    have he : lift i = lift j := (Tuple.sort sample).injective (Fin.ext hij)
    exact Fin.ext (congrArg (fun k : Fin n => k.val) he)
  have size : (univ.image idx).card = rank.val+1 := by
    rw [card_image_of_injective _ inj]
    simp
  rw [← size]
  apply card_le_card
  intro k hk
  obtain ⟨i,_,rfl⟩ := mem_image.mp hk
  simp only [mem_filter]
  refine ⟨mem_range.mpr (Tuple.sort sample (lift i)).isLt,?_⟩
  have hr : lift i ≤ rank := by change i.val ≤ rank.val; omega
  exact ((Tuple.monotone_sort sample) hr).trans_lt h

/-- With eleven iid pairs, the ninth-smallest observed difference is a
distribution-free upper confidence bound for any population median m.
The chance of undershooting m is at most 67/2048, including ties. -/
theorem paired_upper_median_eleven [MeasurableSpace D] (μ : Measure D) [IsProbabilityMeasure μ]
    (value : D → α) (m : α) (hm : MeasurableSet {d | value d < m})
    (hlower : μ {d | value d < m} ≤ 1/2) :
    Measure.infinitePi (fun _ : ℕ => μ) {z | pairedOrderStatistic value z (8 : Fin 11) < m} ≤ 67/2048 := by
  let U := {d | value d < m}
  calc
    _ ≤ Measure.infinitePi (fun _ : ℕ => μ) {z | 9 ≤ terminalPassCount U (range 11) z} :=
      measure_mono fun z hz => rank_le_below_count value z (8 : Fin 11) m hz
    _ = terminalTail 11 9 (μ U) := by rw [iid_terminal_count_tail μ hm,card_range,terminalTail]
    _ ≤ terminalTail 11 9 (1/2) := terminalTail_monotone _ _ hlower (by norm_num)
    _ = _ := by
      rw [terminalTail,show Finset.Icc 9 11 = ({9,10,11} : Finset ℕ) by decide]
      have half : (1:ℝ≥0∞)-1/2 = 1/2 := by
        apply (ENNReal.toReal_eq_toReal_iff' (by finiteness) (by finiteness)).mp
        rw [ENNReal.toReal_sub_of_le (by norm_num) (by norm_num)]
        norm_num
      norm_num [Nat.choose,half]
      apply (ENNReal.toReal_eq_toReal_iff' (by finiteness) (by finiteness)).mp
      rw [ENNReal.toReal_add (by finiteness) (by finiteness),ENNReal.toReal_add (by finiteness) (by finiteness)]
      norm_num

example : pairedHitDifference 20000 256 (some 1536) (some 2816) = -1280 ∧
    pairedHitDifference 20000 256 none (some 2816) = 17440 := by decide

example := paired_upper_median_eleven (Measure.dirac (0 : ℕ)) (fun _ => pairedHitDifference 20000 256 none none) 0
  (by simp [pairedHitDifference]) (by simp [pairedHitDifference])

end Transformer.Grokking.FiniteRecovery
