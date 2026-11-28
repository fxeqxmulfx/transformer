module

public import Transformer.Grokking.FiniteRecovery.TerminalBinomial

/-! Exact binomial upper tails at the boundary of a composite one-sided
null. Arbitrary dependence between the executions at different budgets is
allowed. Crediting only uninterrupted successes preserves the bound. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

open MeasureTheory Set Finset
open scoped ENNReal Classical

noncomputable def terminalTail (n k : ℕ) (p : ℝ≥0∞) : ℝ≥0∞ :=
  ∑ j ∈ Finset.Icc k n, (n.choose j : ℝ≥0∞) * p^j * (1-p)^(n-j)

private theorem count_mono_success {D : Type*} {A B : Set D} (h : A ⊆ B) (I : Finset ℕ) (z : ℕ → D) :
    terminalPassCount A I z ≤ terminalPassCount B I z := by
  classical
  unfold terminalPassCount
  apply Finset.card_le_card
  intro i hi
  exact mem_filter.mpr ⟨(mem_filter.mp hi).1,h (mem_filter.mp hi).2⟩

/-- Couple a Bernoulli trial of mass p to a trial of mass q by using
three atoms of masses p, q-p and 1-q. Every success at p is also a
success at q, including when either probability lies on the boundary. -/
theorem terminalTail_monotone (n k : ℕ) {p q : ℝ≥0∞} (hp : p ≤ q) (hq : q ≤ 1) :
    terminalTail n k p ≤ terminalTail n k q := by
  let ν : Measure ℕ := p • Measure.dirac 0 + (q-p) • Measure.dirac 1 + (1-q) • Measure.dirac 2
  have hsum : p + (q-p) = q := add_tsub_cancel_of_le hp
  have hprob : q + (1-q) = 1 := add_tsub_cancel_of_le hq
  let : IsProbabilityMeasure ν := ⟨by simp [ν,smul_eq_mul,hsum,hprob]⟩
  let A : Set ℕ := {0}
  let B : Set ℕ := insert 1 A
  have ha : MeasurableSet A := measurableSet_singleton _
  have hb : MeasurableSet B := ha.insert _
  have pa : ν A = p := by norm_num [A,ν,add_apply,smul_apply,smul_eq_mul,Measure.dirac_apply']
  have pb : ν B = q := by norm_num [B,A,ν,add_apply,smul_apply,smul_eq_mul,Measure.dirac_apply',hsum]
  have inc : {z | k ≤ terminalPassCount A (range n) z} ⊆ {z | k ≤ terminalPassCount B (range n) z} := by
    intro z hz
    exact hz.trans (count_mono_success (subset_insert 1 A) _ z)
  have h := measure_mono (μ := Measure.infinitePi (fun _ : ℕ => ν)) inc
  simpa only [iid_terminal_count_tail ν ha,iid_terminal_count_tail ν hb,pa,pb,card_range,terminalTail] using h

variable {D B : Type*} [MeasurableSpace D]

/-- Compare every candidate budget to the exact binomial tail under the
null upper bound. The credited set may depend on the whole execution,
so interrupted runs can be excluded from the success count. -/
theorem selected_budget_certified_tail (μ : Measure D) [IsProbabilityMeasure μ]
    (budgets : Finset B) (U : B → Set D) (hU : ∀ b ∈ budgets, MeasurableSet (U b))
    (p : ℝ≥0∞) (hp : p ≤ 1) (I : Finset ℕ) (credit : (ℕ → D) → B → Finset ℕ) (k : ℕ) :
    Measure.infinitePi (fun _ : ℕ => μ)
      {z | ∃ b ∈ budgets, μ (U b) ≤ p ∧ k ≤ terminalPassCount (U b) (I ∩ credit z b) z} ≤
      (budgets.card : ℝ≥0∞) * terminalTail I.card k p := by
  refine le_trans ?_ (selected_event_bound (Measure.infinitePi (fun _ : ℕ => μ)) budgets (fun b => μ (U b) ≤ p)
    (fun b => {z | k ≤ terminalPassCount (U b) I z}) _ ?_)
  · apply measure_mono
    rintro z ⟨b,hb,hw,hk⟩
    exact ⟨b,hb,hw,hk.trans (card_le_card (filter_subset_filter _ Finset.inter_subset_left))⟩
  · intro b hb hw
    rw [iid_terminal_count_tail μ (hU b hb)]
    exact terminalTail_monotone _ _ hw hp

example (μ : Measure D) [IsProbabilityMeasure μ] (U : ℕ → Set D) (hU : ∀ b, MeasurableSet (U b))
    (credit : (ℕ → D) → ℕ → Finset ℕ) :
    Measure.infinitePi (fun _ : ℕ => μ) {z | ∃ b ∈ ({100000,200000} : Finset ℕ), μ (U b) ≤ 1/2 ∧
      9 ≤ terminalPassCount (U b) (Finset.Ico 2 12 ∩ credit z b) z} ≤ 11/512 := by
  apply (selected_budget_certified_tail μ _ U (fun b _ => hU b) _ (by norm_num) _ credit 9).trans
  rw [show (Finset.Ico 2 12).card = 10 by decide,terminalTail,show Finset.Icc 9 10 = ({9,10} : Finset ℕ) by decide]
  norm_num [Nat.choose]
  apply (ENNReal.toReal_le_toReal (by finiteness) (by finiteness)).mp
  norm_num
  rw [ENNReal.toReal_add (by finiteness) (by finiteness)]
  norm_num

end Transformer.Grokking.FiniteRecovery
