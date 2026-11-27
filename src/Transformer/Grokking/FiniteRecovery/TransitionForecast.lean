module

public import Transformer.Grokking.FiniteRecovery.ArgandTransition
public import Transformer.Grokking.FiniteRecovery.ArgandSamples
public import Transformer.Grokking.FiniteRecovery.Series

/-! Observations and forecasts for a law of the current token. A sample
can be compressed to one observation for each current token. Compile the
recovered law into ordinary Argand, then use the exact margin expression
for the softmax and the cross-entropy. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Filter
open scoped Topology

variable {A : Type*} {H d T : ℕ}

/-- The last token is the state for this deterministic transition. -/
def transitionTask (next : A → A) (word : Word A (T + 1)) : A :=
  next (word (Fin.last T))

/-- Take one representative history with each possible final state. -/
def transitionInputs (A : Type*) [Fintype A] (T : ℕ) : Finset (Word A (T + 1)) :=
  Finset.univ.map ⟨fun a _ => a, fun _ _ h => congrFun h (Fin.last T)⟩

@[simp] theorem card_transitionInputs (A : Type*) [Fintype A] (T : ℕ) :
    (transitionInputs A T).card = Fintype.card A := by
  rw [transitionInputs, Finset.card_map, Finset.card_univ]

/-- Read the next-state law from the labels of histories ending at this state. -/
noncomputable def transitionFromData (part : Finset (Word A (T + 1) × A)) (state : A) : A := by
  classical
  exact if h : ∃ label, ∃ word, (word, label) ∈ part ∧ word (Fin.last T) = state
    then Classical.choose h else state

theorem transitionFromData_eq_of_covers (next : A → A) (inputs : Finset (Word A (T + 1)))
    (hcover : ∀ a, ∃ word ∈ inputs, word (Fin.last T) = a) :
    transitionFromData (observe inputs (transitionTask next)) = next := by
  funext state
  obtain ⟨word, hw, hlast⟩ := hcover state
  have h : ∃ label, ∃ w, (w, label) ∈ observe inputs (transitionTask next) ∧
      w (Fin.last T) = state :=
    ⟨transitionTask next word, word, mem_observe.mpr ⟨hw, rfl⟩, hlast⟩
  rw [transitionFromData, dite_eq_left h]
  obtain ⟨w, hw', hs⟩ := Classical.choose_spec h
  simpa [transitionTask, hs] using (mem_observe.mp hw').2.symm

theorem transitionFromData_eq [Fintype A] (next : A → A) (T : ℕ) :
    transitionFromData (observe (transitionInputs A T) (transitionTask next)) = next := by
  apply transitionFromData_eq_of_covers
  intro a
  exact ⟨fun _ => a, Finset.mem_map.mpr ⟨a, Finset.mem_univ _, rfl⟩, rfl⟩

/-- For an unknown transition law on a nontrivial finite alphabet, exactly
one observation per last token is needed in the best fixed mask. -/
theorem isLeast_transition_observations (A : Type*) [Fintype A] [Nontrivial A] (T : ℕ) :
    IsLeast {n : ℕ | ∃ inputs : Finset (Word A (T + 1)), inputs.card = n ∧
      ∃ learner, Recovers inputs (Set.range (transitionTask (A := A) (T := T))) learner}
      (Fintype.card A) := by
  classical
  let last : Word A (T + 1) → A := fun word => word (Fin.last T)
  have hiff (inputs : Finset (Word A (T + 1))) :
      (∃ learner, Recovers inputs (Set.range (transitionTask (A := A) (T := T))) learner) ↔
      DeterminesOnContexts last (Set.univ : Set (A → A)) inputs := by
    rw [exists_learner_iff_injOn]
    constructor
    · intro hinj f _ g _ hagree a ha
      obtain ⟨word, rfl⟩ := ha
      have hobs : observe inputs (transitionTask f) = observe inputs (transitionTask g) :=
        observe_eq_iff_eqOn.mpr fun w hw => hagree (Set.mem_image_of_mem last hw)
      exact congrFun (hinj (Set.mem_range_self f) (Set.mem_range_self g) hobs) word
    · rintro h _ ⟨f, rfl⟩ _ ⟨g, rfl⟩ hobs
      have hagree : Set.EqOn f g (last '' (inputs : Set _)) := by
        rintro _ ⟨w, hw, rfl⟩
        exact observe_eq_iff_eqOn.mp hobs hw
      exact funext fun w => h f (Set.mem_univ _) g (Set.mem_univ _) hagree (Set.mem_range_self w)
  have hcard : Finset.univ.image last = Finset.univ := by
    apply Finset.eq_univ_of_forall
    intro a
    exact Finset.mem_image.mpr ⟨fun _ => a, Finset.mem_univ _, rfl⟩
  simpa only [hiff, hcard, Finset.card_univ] using
    isLeast_unrestricted_series_observations (A := A) last

/-- At any positive margin, greedy decoding of ordinary Argand realizes
the exact transition on every word, including unobserved prefixes. -/
theorem transitionModel_realizes (hc : 0 < H * d) (next : Fin (H * d) → Fin (H * d))
    {margin : ℝ} (hm : 0 < margin) :
    (transitionModel next margin).realizes 1 (transitionTask (T := T) next) := by
  intro word token hne
  rw [transitionModel_logits hc next hm, transitionModel_logits hc next hm]
  simp only [transitionTask] at hne ⊢
  simpa [Ne.symm hne] using hm

/-- A proof using the parameter table recovered from data. It identifies
the entire class of words, without enumerating the words. -/
theorem transitionFromData_realizes (hc : 0 < H * d) (next : Fin (H * d) → Fin (H * d))
    {margin : ℝ} (hm : 0 < margin) :
    (transitionModel
      (transitionFromData (observe (transitionInputs (Fin (H * d)) T) (transitionTask next)))
      margin).realizes 1 (transitionTask (T := T) next) := by
  rw [transitionFromData_eq]
  exact transitionModel_realizes hc next hm

/-- Exact softmax probability for the correct label. All other logits are
zero, so there are vocab minus one unit contributions to the denominator. -/
theorem transitionModel_probability (hc : 0 < H * d) (next : Fin (H * d) → Fin (H * d))
    {margin : ℝ} (hm : 0 < margin) (word : Word (Fin (H * d)) T) (i : Fin T) :
    (transitionModel next margin).probability 1 word i (next (word i)) =
      Real.exp margin / (Real.exp margin + ((H * d - 1 : ℕ) : ℝ)) := by
  let target := next (word i)
  let p := fun token => Real.exp ((transitionModel next margin).logits 1 word i token)
  have hp : p target = Real.exp margin := by
    simp [p, target, transitionModel_logits hc next hm]
  have hother : (∑ a ∈ Finset.univ.erase target, p a) =
      ((H * d - 1 : ℕ) : ℝ) := by
    calc
      _ = ∑ _ ∈ Finset.univ.erase target, (1 : ℝ) := by
        refine Finset.sum_congr rfl fun a ha => ?_
        rw [show p a = Real.exp 0 by
          unfold p
          rw [transitionModel_logits hc next hm, ite_eq_right (Finset.mem_erase.mp ha).1.symm]]
        exact Real.exp_zero
      _ = _ := by simp
  change p target / (∑ a, p a) = _
  rw [← Finset.sum_erase_add Finset.univ p (Finset.mem_univ target), hp, hother, add_comm]

/-- Cross-entropy of any correctly labeled sample word, as a function
of the common margin. -/
noncomputable def marginLoss (competitors : ℕ) (margin : ℝ) : ℝ :=
  Real.log (Real.exp margin + competitors) - margin

/-- With a competing token the loss at any finite margin is positive. -/
theorem marginLoss_pos {competitors : ℕ} (h : 0 < competitors) (margin : ℝ) :
    0 < marginLoss competitors margin := by
  apply sub_pos.mpr
  calc
    margin = Real.log (Real.exp margin) := (Real.log_exp _).symm
    _ < Real.log (Real.exp margin + competitors) :=
      Real.log_lt_log (Real.exp_pos _) (lt_add_of_pos_right _ (Nat.cast_pos.mpr h))

/-- Along a path of increasing unbounded margins the exact cross-entropy
approaches zero for any finite vocabulary. -/
theorem marginLoss_tendsto_zero (competitors : ℕ) :
    Tendsto (marginLoss competitors) atTop (𝓝 0) := by
  have he (x : ℝ) : marginLoss competitors x = Real.log (1 + competitors * Real.exp (-x)) := by
    calc
      marginLoss competitors x = Real.log ((Real.exp x + competitors) / Real.exp x) := by
        rw [marginLoss, Real.log_div (by positivity) (Real.exp_ne_zero _), Real.log_exp]
      _ = _ := by
        rw [add_div, div_self (Real.exp_ne_zero _), Real.exp_neg, div_eq_mul_inv]
  have ht : Tendsto (fun x : ℝ => 1 + (competitors : ℝ) * Real.exp (-x)) atTop (𝓝 1) := by
    simpa only [mul_zero, add_zero] using
      (Real.tendsto_exp_neg_atTop_nhds_zero.const_mul (competitors : ℝ)).const_add 1
  have hl := (Real.continuousAt_log (show (1 : ℝ) ≠ 0 by norm_num)).tendsto.comp ht
  simpa only [Function.comp_def, Real.log_one, ← he] using hl

theorem transitionModel_labelLoss (hc : 0 < H * d) (next : Fin (H * d) → Fin (H * d))
    {margin : ℝ} (hm : 0 < margin) (word : Word (Fin (H * d)) (T + 1)) :
    (transitionModel next margin).labelLoss 1 word (transitionTask next word) =
      marginLoss (H * d - 1) margin := by
  unfold Model.labelLoss transitionTask
  rw [transitionModel_probability hc next hm, Real.log_div (Real.exp_ne_zero _) (by positivity),
    Real.log_exp, neg_sub, marginLoss]

theorem transitionModel_observedLoss (hc : 0 < H * d) (next : Fin (H * d) → Fin (H * d))
    {margin : ℝ} (hm : 0 < margin) (inputs : Finset (Word (Fin (H * d)) (T + 1)))
    (hne : inputs.Nonempty) :
    (transitionModel next margin).observedLoss 1 (observe inputs (transitionTask next)) =
      marginLoss (H * d - 1) margin := by
  unfold Model.observedLoss
  have he : (∑ pair ∈ observe inputs (transitionTask next),
      (transitionModel next margin).labelLoss 1 pair.1 pair.2) =
      ∑ _ ∈ observe inputs (transitionTask next), marginLoss (H * d - 1) margin := by
    apply Finset.sum_congr rfl
    intro pair hp
    rw [← (mem_observe.mp hp).2, transitionModel_labelLoss hc next hm]
  rw [he, Finset.sum_const, nsmul_eq_mul, card_observe,
    mul_comm, mul_div_cancel_right₀ _ (Nat.cast_ne_zero.mpr (Finset.card_pos.mpr hne).ne')]

/-- The recovered law yields every finite rollout exactly when its
argmax is iterated. -/
theorem transitionFromData_rollout [Fintype A] (next : A → A) (T horizon : ℕ) (initial : A) :
    ((transitionFromData (observe (transitionInputs A T) (transitionTask next)))^[horizon]) initial =
      (next^[horizon]) initial := by
  rw [transitionFromData_eq]

example := isLeast_transition_observations (Fin 256) 31

example (next : Fin 256 → Fin 256) :=
  transitionFromData_realizes (H := 4) (d := 64) (T := 31) (by decide) next (by norm_num : (0 : ℝ) < 1)

example (next : Fin 64 → Fin 64) :=
  transitionFromData_realizes (H := 4) (d := 16) (T := 31) (by decide) next (by norm_num : (0 : ℝ) < 1)

example : 0 < marginLoss 255 1 := marginLoss_pos (by decide) 1

example (next : Fin 256 → Fin 256) (initial : Fin 256) :=
  transitionFromData_rollout next 31 1000000 initial

example : Fintype.card (Word (Fin 256) 32) = 2 ^ 256 := by
  rw [card_word, Fintype.card_fin, show 256 = 2 ^ 8 by decide, ← pow_mul]
  norm_num

end Transformer.Grokking.FiniteRecovery.Argand
