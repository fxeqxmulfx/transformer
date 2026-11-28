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

/-- A decision to extend an all-success series can depend on the results
already seen. At a fixed budget, rejection still requires success on
every draw of the original window, including any continuation. -/
theorem continued_all_pass_bound (μ : Measure D) [IsProbabilityMeasure μ]
    {U : Set D} (hU : MeasurableSet U) (start n : ℕ) (report : Set (ℕ → D))
    (h : report ⊆ allPassWindow U start n) {q : ℝ≥0∞} (hq : μ U ≤ q) :
    Measure.infinitePi (fun _ : ℕ => μ) report ≤ q ^ n := by
  calc
    _ ≤ _ := measure_mono h
    _ = _ := iid_allPassWindow μ hU start n
    _ ≤ _ := pow_le_pow_left' hq n

/-- Eleven consecutive first-attempt hits at a single budget reject
a single-draw hit probability of at most three quarters at 5%. -/
theorem eleven_hit_three_quarters (μ : Measure D) [IsProbabilityMeasure μ]
    {U : Set D} (hU : MeasurableSet U) (start : ℕ) (report : Set (ℕ → D))
    (h : report ⊆ allPassWindow U start 11) (hq : μ U ≤ 3/4) :
    Measure.infinitePi (fun _ : ℕ => μ) report < 1/20 := by
  apply lt_of_le_of_lt (continued_all_pass_bound μ hU start 11 report h hq)
  apply (ENNReal.toReal_lt_toReal (by finiteness) (by finiteness)).mp
  norm_num

theorem selected_event_bound {Z : Type*} [MeasurableSpace Z] (P : Measure Z)
    (budgets : Finset B) (weak : B → Prop) (event : B → Set Z) (q : ℝ≥0∞)
    (h : ∀ b ∈ budgets, weak b → P (event b) ≤ q) :
    P {z | ∃ b ∈ budgets, weak b ∧ z ∈ event b} ≤ (budgets.card : ℝ≥0∞) * q := by
  classical
  let W (b : B) := if weak b then event b else ∅
  have bound (b) (hb : b ∈ budgets) : P (W b) ≤ q := by
    by_cases hw : weak b
    · simpa [W,hw] using h b hb hw
    · simp [W,hw]
  calc
    _ ≤ P (⋃ b ∈ budgets, W b) := by
      apply measure_mono
      rintro z ⟨b,hb,h,hz⟩
      exact mem_iUnion₂.mpr ⟨b,hb,by simpa [W,h] using hz⟩
    _ ≤ ∑ b ∈ budgets, P (W b) := measure_biUnion_finset_le _ _
    _ ≤ ∑ _b ∈ budgets, q := Finset.sum_le_sum bound
    _ = _ := by simp

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
  apply selected_event_bound
  intro b hb hw
  rw [iid_allPassWindow μ (hU b hb)]
  exact pow_le_pow_left' hw n

/-- Number of strictly successful terminal evaluations on the chosen trial indices. -/
noncomputable def terminalPassCount (U : Set D) (I : Finset ℕ) (z : ℕ → D) : ℕ := by
  classical exact (I.filter fun i => z i ∈ U).card

/-- At least k successes contain a k-element subset of successes. Each
such subset has probability p^k. This bound also applies when some of
the terminal checkpoints fail. -/
theorem iid_terminal_count_bound (μ : Measure D) [IsProbabilityMeasure μ] {U : Set D}
    (hU : MeasurableSet U) (I : Finset ℕ) (k : ℕ) :
    Measure.infinitePi (fun _ : ℕ => μ) {z | k ≤ terminalPassCount U I z}
      ≤ (I.card.choose k : ℝ≥0∞) * (μ U)^k := by
  classical
  let P := Measure.infinitePi (fun _ : ℕ => μ)
  let E (T : Finset ℕ) := Set.pi (T : Set ℕ) (fun _ => U)
  have bound (T) (hT : T ∈ I.powersetCard k) : P (E T) = (μ U)^k := by
    dsimp only [P,E]
    rw [Measure.infinitePi_pi _ (fun _ _ => hU)]
    simp [(Finset.mem_powersetCard.mp hT).2]
  calc
    _ ≤ P (⋃ T ∈ I.powersetCard k, E T) := by
      apply measure_mono
      intro z hz
      change k ≤ (I.filter fun i => z i ∈ U).card at hz
      obtain ⟨T,ht,he⟩ := Finset.exists_subset_card_eq hz
      exact mem_iUnion₂.mpr ⟨T,Finset.mem_powersetCard.mpr
        ⟨fun _ hi => (Finset.mem_filter.mp (ht hi)).1,he⟩,fun _ hi => (Finset.mem_filter.mp (ht hi)).2⟩
    _ ≤ ∑ T ∈ I.powersetCard k, P (E T) := measure_biUnion_finset_le _ _
    _ = ∑ _T ∈ I.powersetCard k, (μ U)^k := Finset.sum_congr rfl bound
    _ = _ := by simp

theorem selected_budget_count_bound (μ : Measure D) [IsProbabilityMeasure μ]
    (budgets : Finset B) (U : B → Set D) (hU : ∀ b ∈ budgets, MeasurableSet (U b))
    (p : ℝ≥0∞) (I : Finset ℕ) (k : ℕ) :
    Measure.infinitePi (fun _ : ℕ => μ)
      {z | ∃ b ∈ budgets, μ (U b) ≤ p ∧ k ≤ terminalPassCount (U b) I z}
      ≤ (budgets.card : ℝ≥0∞) * ((I.card.choose k : ℝ≥0∞) * p^k) := by
  refine selected_event_bound _ budgets (fun b => μ (U b) ≤ p)
    (fun b => {z | k ≤ terminalPassCount (U b) I z}) _ ?_
  intro b hb hw
  exact (iid_terminal_count_bound μ (hU b hb) I k).trans (mul_le_mul' le_rfl (pow_le_pow_left' hw k))

/-- Credit only successful original executions retained by an arbitrary audit.
The retained indices may depend on every outcome and on the budget. Keeping
the full registered sample size in the tail bound also covers interruptions
whose cause or dependence on the training trajectory is unknown. -/
theorem selected_budget_retained_count_bound (μ : Measure D) [IsProbabilityMeasure μ]
    (budgets : Finset B) (U : B → Set D) (hU : ∀ b ∈ budgets, MeasurableSet (U b))
    (p : ℝ≥0∞) (I : Finset ℕ) (retain : B → (ℕ → D) → Finset ℕ) (k : ℕ) :
    Measure.infinitePi (fun _ : ℕ => μ)
      {z | ∃ b ∈ budgets, μ (U b) ≤ p ∧ k ≤ terminalPassCount (U b) (I ∩ retain b z) z}
      ≤ (budgets.card : ℝ≥0∞) * ((I.card.choose k : ℝ≥0∞) * p^k) := by
  classical
  refine le_trans ?_ (selected_budget_count_bound μ budgets U hU p I k)
  apply measure_mono
  rintro z ⟨b,hb,hp,hk⟩
  exact ⟨b,hb,hp,hk.trans (Finset.card_le_card
    (Finset.filter_subset_filter _ Finset.inter_subset_left))⟩

example := selected_budget_retained_count_bound (Measure.dirac (0 : ℕ))
  ({100000,200000} : Finset ℕ) (fun _ => Set.univ) (fun _ _ => MeasurableSet.univ)
  (1/2) (Finset.Ico 2 12) (fun _ _ => Finset.Ico 3 12) 9

/-- With ten fresh independent draws and two candidate budgets, even
nine successes reject a single-run probability at most one half at 5%. -/
theorem nine_of_ten_two_budgets (μ : Measure D) [IsProbabilityMeasure μ]
    (U : ℕ → Set D) (hU : ∀ b, MeasurableSet (U b)) :
    Measure.infinitePi (fun _ : ℕ => μ)
      {z | ∃ b ∈ ({100000,200000} : Finset ℕ), μ (U b) ≤ 1/2 ∧
        9 ≤ terminalPassCount (U b) (Finset.Ico 2 12) z} ≤ 5/128 := by
  apply (selected_budget_count_bound μ ({100000,200000} : Finset ℕ) U
    (fun b _ => hU b) (1/2) (Finset.Ico 2 12) 9).trans
  norm_num [Nat.choose]
  apply (ENNReal.toReal_le_toReal (by finiteness) (by finiteness)).mp
  norm_num

example := nine_of_ten_two_budgets (Measure.dirac (0 : ℕ)) (fun _ => Set.univ) (fun _ => MeasurableSet.univ)

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
