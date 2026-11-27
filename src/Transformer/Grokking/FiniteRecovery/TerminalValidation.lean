module

public import Transformer.Grokking.FiniteRecovery.VerifiedRestarts

/-! Validation of terminal classifiers on independent draws. One draw can
include the native seed, the minibatch order and the GPU execution of
each candidate budget. The budgets may be arbitrarily dependent within
a draw. Only the validation draws are independent across trials. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

open MeasureTheory Set
open scoped ENNReal

variable {D B X A : Type*} [MeasurableSpace D]

/-- Any law on complete training executions gives the same probability
for the true terminal test and for its observation-only certificate. -/
theorem observed_terminal_measure (μ : Measure D) (S : Set X) (family : Set (X → A))
    (hid : InjOn (fun g : X → A => fun x : S => g x) family) (f : X → A) (hf : f ∈ family)
    (test : D → (X → A) → Prop) :
    μ {z | ∃ g ∈ family, (∀ x : S, g x = f x) ∧ test z g} = μ {z | test z f} := by
  exact congrArg μ (Set.ext fun z => identified_property_iff S family hid f hf (test z))

/-- All terminal observations in a specified consecutive window succeed. -/
def allPassWindow (U : Set D) (start n : ℕ) : Set (ℕ → D) :=
  {z | ∀ i ∈ Finset.Ico start (start+n), z i ∈ U}

theorem iid_allPassWindow (μ : Measure D) [IsProbabilityMeasure μ] {U : Set D}
    (hU : MeasurableSet U) (start n : ℕ) :
    Measure.infinitePi (fun _ : ℕ => μ) (allPassWindow U start n) = (μ U)^n := by
  change Measure.infinitePi _ (Set.pi (Finset.Ico start (start+n) : Set ℕ) (fun _ => U)) = _
  rw [Measure.infinitePi_pi _ (fun _ _ => hU)]
  simp

/-- Even when the reported budget is chosen using the validation results,
the probability that some candidate with single-run success at most p
passes every validation draw is bounded by the number of candidates
times p^n. No independence between different budgets is required. -/
theorem selected_budget_all_pass_bound (μ : Measure D) [IsProbabilityMeasure μ]
    (budgets : Finset B) (U : B → Set D) (hU : ∀ b ∈ budgets, MeasurableSet (U b))
    (p : ℝ≥0∞) (start n : ℕ) :
    Measure.infinitePi (fun _ : ℕ => μ)
      {z | ∃ b ∈ budgets, μ (U b) ≤ p ∧ z ∈ allPassWindow (U b) start n}
      ≤ (budgets.card : ℝ≥0∞) * p^n := by
  classical
  let P := Measure.infinitePi (fun _ : ℕ => μ)
  let W (b : B) : Set (ℕ → D) := if μ (U b) ≤ p then allPassWindow (U b) start n else ∅
  have bound (b) (hb : b ∈ budgets) : P (W b) ≤ p^n := by
    by_cases h : μ (U b) ≤ p
    · simpa [P,W,h,iid_allPassWindow μ (hU b hb)] using pow_le_pow_left' h n
    · simp [W,h]
  calc
    _ ≤ P (⋃ b ∈ budgets, W b) := by
      apply measure_mono
      rintro z ⟨b,hb,h,hz⟩
      exact mem_iUnion₂.mpr ⟨b,hb,by simpa [W,h] using hz⟩
    _ ≤ ∑ b ∈ budgets, P (W b) := measure_biUnion_finset_le _ _
    _ ≤ ∑ _b ∈ budgets, p^n := Finset.sum_le_sum bound
    _ = _ := by simp

example (μ : Measure D) [IsProbabilityMeasure μ] (U : ℕ → Set D) (hU : ∀ b, MeasurableSet (U b)) :
    Measure.infinitePi (fun _ : ℕ => μ)
      {z | ∃ b ∈ ({100000,200000} : Finset ℕ), μ (U b) ≤ 3/4 ∧ z ∈ allPassWindow (U b) 2 10}
      ≤ 2 * (3/4 : ℝ≥0∞)^10 := by
  simpa using selected_budget_all_pass_bound μ ({100000,200000} : Finset ℕ) U
    (fun b _ => hU b) (3/4) 2 10

example (μ : Measure D) (test : D → (Unit → Bool) → Prop) :
    μ {z | ∃ g ∈ ({fun _ : Unit => true} : Set (Unit → Bool)),
      (∀ x : (∅ : Set Unit), g x = true) ∧ test z g} = μ {z | test z (fun _ => true)} :=
  observed_terminal_measure μ ∅ _ (Set.injOn_singleton _ _) _ (Set.mem_singleton _) test

end Transformer.Grokking.FiniteRecovery
