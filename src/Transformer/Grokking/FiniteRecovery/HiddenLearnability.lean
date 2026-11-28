module

public import Transformer.Grokking.FiniteRecovery.LabelOrbit

/-! # Checkable conditions for learning the hidden answers

`HiddenSuccess S s f` is the event that from some update on every input
outside `S` ranks the answer `f x` strictly above every other token.
Three properties, each checked from the task and the architecture without
running the optimizer, are necessary for success above one half:

* **identifiability**: a run that sees the target only through its observed
  labels gives two members of the class with the same labels a combined
  success probability of at most one (`hidden_success_add_le_one`);
* **strict representability**: success with positive probability needs
  weights ranking every hidden answer first (`representable_of_pos`);
* **seen answers**: ordinary AdamW from an exchangeable Gaussian start
  succeeds with probability at most one half if a hidden answer and another
  token occur in no observed prompt, observed label or hidden prompt
  (`Argand.adamW_unseen_answer_le_half`).

The 3×3 phase table of `TwoHiddenAnswers` has the first two, yet its hidden
answers 2, 3 and 63 occur nowhere, and AdamW succeeds there with probability
at most 1/2756. In modular division each answer is an observed denominator
token or the observed zero-row answer (`Argand.division_answers_seen`). The
three properties are not shown to be sufficient. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

open Set MeasureTheory Filter
open scoped NNReal

variable {X A D : Type*}

/-- From some update `N` on, every input outside `S` ranks its answer
`f x` strictly above each other token `a`. -/
def HiddenSuccess (S : Set X) (s : ℕ → X → A → ℝ) (f : X → A) : Prop :=
  ∃ N, ∀ t ≥ N, ∀ x ∉ S, ∀ a, a ≠ f x → s t x a < s t x (f x)

/-- **Identifiability is necessary.** The scores of a run depend on the
target only through its observed labels `y`. Two targets with these labels
differ only on hidden inputs, where their strict rankings exclude each other. -/
theorem hidden_success_add_le_one [MeasurableSpace D] (μ : Measure D) [IsProbabilityMeasure μ]
    (S : Set X) (score : (S → A) → D → ℕ → X → A → ℝ)
    (meas : ∀ y t (f : X → A), MeasurableSet {z | ∀ x ∉ S, ∀ a, a ≠ f x → score y z t x a < score y z t x (f x)})
    (y : S → A) {f g : X → A} (hf : ∀ x : S, f x = y x) (hg : ∀ x : S, g x = y x) (hne : f ≠ g) :
    μ.real {z | HiddenSuccess S (score y z) f} + μ.real {z | HiddenSuccess S (score y z) g} ≤ 1 := by
  obtain ⟨x, hx⟩ := Function.ne_iff.mp hne
  have hidden : x ∉ S := fun h => hx ((hf ⟨x, h⟩).trans (hg ⟨x, h⟩).symm)
  have disj : Disjoint {z | HiddenSuccess S (score y z) f} {z | HiddenSuccess S (score y z) g} := by
    refine Set.disjoint_left.mpr fun z ⟨N, hN⟩ ⟨M, hM⟩ => ?_
    exact lt_asymm (hN (max N M) (le_max_left _ _) x hidden (g x) (Ne.symm hx))
      (hM (max N M) (le_max_right _ _) x hidden (f x) hx)
  have mg : MeasurableSet {z | HiddenSuccess S (score y z) g} := by
    have he : {z | HiddenSuccess S (score y z) g} = ⋃ N : ℕ, ⋂ t : ℕ, ⋂ (_ : t ≥ N),
        {z | ∀ x ∉ S, ∀ a, a ≠ g x → score y z t x a < score y z t x (g x)} := by
      ext z
      simp only [HiddenSuccess, mem_ofPred_eq, mem_iUnion, mem_iInter]
    rw [he]
    exact .iUnion fun N => .iInter fun t => .iInter fun _ => meas y t g
  rw [← measureReal_union disj mg]
  exact measureReal_le_one

/-- Success above one half for every member of the class forces the
observed labels to separate the class. -/
theorem identified_of_above_half [MeasurableSpace D] (μ : Measure D) [IsProbabilityMeasure μ]
    (S : Set X) (ℋ : Set (X → A)) (score : (S → A) → D → ℕ → X → A → ℝ)
    (meas : ∀ y t (f : X → A), MeasurableSet {z | ∀ x ∉ S, ∀ a, a ≠ f x → score y z t x a < score y z t x (f x)})
    (above : ∀ f ∈ ℋ, 1/2 < μ.real {z | HiddenSuccess S (score (fun x => f x) z) f}) :
    InjOn (fun g : X → A => fun x : S => g x) ℋ := by
  intro f hf g hg heq
  have same : (fun x : S => f x) = (fun x : S => g x) := heq
  by_contra hne
  have hg' := above g hg
  rw [← congrArg score same] at hg'
  have sum := hidden_success_add_le_one μ S score meas (fun x => f x) (fun _ => rfl)
    (fun x => (congrFun same x).symm) hne
  linarith [above f hf]

/-- **Strict representability is necessary.** A run whose scores come from
its weights succeeds with positive probability only if some weights rank
every hidden answer strictly first. -/
theorem representable_of_pos [MeasurableSpace D] (μ : Measure D) {W : Type*} (S : Set X)
    (run : D → ℕ → W) (score : W → X → A → ℝ) (f : X → A)
    (pos : 0 < μ.real {z | HiddenSuccess S (fun t => score (run z t)) f}) :
    ∃ w, ∀ x ∉ S, ∀ a, a ≠ f x → score w x a < score w x (f x) := by
  have ne : μ {z | HiddenSuccess S (fun t => score (run z t)) f} ≠ 0 := fun h =>
    pos.ne' (by simp [measureReal_def, h])
  obtain ⟨z, N, hN⟩ := nonempty_of_measure_ne_zero ne
  exact ⟨run z N, hN N le_rfl⟩

/-- Swapping an unseen hidden answer with another unseen token fixes every
seen token and moves the hidden assignment. -/
theorem orbit_two_le_of_unseen [Finite A] [Finite X] (seen : Set A) (f : X → A) {x : X} {b : A}
    (hx : f x ∉ seen) (hb : b ∉ seen) (hne : b ≠ f x) :
    2 ≤ (hiddenLabelOrbit (fixingSubgroup (Equiv.Perm A) seen) f).card := by
  classical
  let e : fixingSubgroup (Equiv.Perm A) seen := ⟨Equiv.swap (f x) b, (mem_fixingSubgroup_iff (Equiv.Perm A)).mpr
    fun a ha => Equiv.swap_apply_of_ne_of_ne (fun h => hx (h ▸ ha)) (fun h => hb (h ▸ ha))⟩
  have moved : (hiddenLabelOrbit (fixingSubgroup (Equiv.Perm A) seen) f).card ≠ 1 := fun h =>
    hne (by simpa [e] using (hiddenLabelOrbit_singleton_iff _ f).mp h e x)
  have pos : 0 < (hiddenLabelOrbit (fixingSubgroup (Equiv.Perm A) seen) f).card :=
    Finset.card_pos.mpr ⟨f, mem_hiddenLabelOrbit.mpr ⟨1, rfl⟩⟩
  omega

/-- If every hidden answer is seen, the stabilizer of the seen tokens fixes
the hidden assignment, and the orbit bound excludes nothing. -/
theorem orbit_singleton_of_seen [Finite A] [Finite X] (seen : Set A) (f : X → A)
    (h : ∀ x, f x ∈ seen) : (hiddenLabelOrbit (fixingSubgroup (Equiv.Perm A) seen) f).card = 1 :=
  (hiddenLabelOrbit_singleton_iff _ f).mpr fun e x => (mem_fixingSubgroup_iff (Equiv.Perm A)).mp e.property _ (h x)

example := orbit_singleton_of_seen (univ : Set Bool) (id : Bool → Bool) fun _ => mem_univ _

example := representable_of_pos (Measure.dirac ()) (univ : Set Bool) (fun _ _ => ())
  (fun _ _ _ => (0 : ℝ)) id (by simp [HiddenSuccess])

example := hidden_success_add_le_one (Measure.dirac ()) (∅ : Set Bool) (fun _ _ _ _ _ => (0 : ℝ))
  (fun _ _ _ => MeasurableSet.of_discrete) (fun x => x.val) (f := id) (g := not)
  (fun x => x.2.elim) (fun x => x.2.elim) (fun h => absurd (congrFun h true) (by decide))

example := identified_of_above_half (Measure.dirac ()) (univ : Set Bool) (univ : Set (Bool → Bool))
  (fun _ _ _ _ _ => (0 : ℝ)) (fun _ _ _ => MeasurableSet.of_discrete) (fun _ _ => by simp [HiddenSuccess]; norm_num)

end Transformer.Grokking.FiniteRecovery
