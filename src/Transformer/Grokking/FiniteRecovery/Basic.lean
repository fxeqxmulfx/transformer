module

public import Mathlib.Data.Fintype.BigOperators
public import Mathlib.Data.Fintype.EquivFin
public import Mathlib.Logic.Function.Basic

/-! The finite recovery problem from the introduction of the Grokking Book.
One learner must recover every admissible target on the same observed inputs. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

/-- A word of length T over A, with repetition allowed. -/
abbrev Word (A : Type*) (T : ℕ) := Fin T → A

variable {A : Type*} {T : ℕ}

/-- The finite alphabet can be numbered from zero to its cardinality minus one. -/
noncomputable def tokenEnumeration (A : Type*) [Fintype A] :
    Fin (Fintype.card A) ≃ A :=
  (Fintype.equivFin A).symm

/-- Each of the T positions independently admits any token in A. -/
theorem card_word (A : Type*) [Fintype A] (T : ℕ) :
    Fintype.card (Word A T) = Fintype.card A ^ T := by
  simp only [Word, Fintype.card_fun, Fintype.card_fin]

/-- Extend an input word to its labeled pair on the graph. -/
def graphEmbedding (f : Word A T → A) : Word A T ↪ Word A T × A :=
  ⟨fun word => (word, f word), fun _ _ h => congrArg Prod.fst h⟩

/-- The labeled pairs revealed by the selected input words. -/
def observe (inputs : Finset (Word A T)) (f : Word A T → A) :
    Finset (Word A T × A) :=
  inputs.map (graphEmbedding f)

/-- The full finite graph has one labeled pair for each word. -/
def graph [Fintype A] (f : Word A T → A) : Finset (Word A T × A) :=
  observe Finset.univ f

@[simp] theorem card_observe (inputs : Finset (Word A T)) (f : Word A T → A) :
    (observe inputs f).card = inputs.card :=
  Finset.card_map _

@[simp] theorem mem_observe {inputs : Finset (Word A T)} {f : Word A T → A}
    {pair : Word A T × A} :
    pair ∈ observe inputs f ↔ pair.1 ∈ inputs ∧ f pair.1 = pair.2 := by
  rw [observe, Finset.mem_map]
  constructor
  · rintro ⟨word, hword, rfl⟩
    exact ⟨hword, rfl⟩
  · rintro ⟨hword, hlabel⟩
    exact ⟨pair.1, hword, Prod.ext rfl hlabel⟩

@[simp] theorem mem_graph [Fintype A] {f : Word A T → A} {pair : Word A T × A} :
    pair ∈ graph f ↔ f pair.1 = pair.2 := by
  simp [graph]

theorem card_graph [Fintype A] (f : Word A T → A) :
    (graph f).card = Fintype.card A ^ T := by
  rw [graph, card_observe, Finset.card_univ, card_word]

/-- Revealing selected words gives part of the full graph. -/
theorem observe_subset_graph [Fintype A] (inputs : Finset (Word A T)) (f : Word A T → A) :
    observe inputs f ⊆ graph f := by
  intro pair hpair
  exact mem_graph.mpr (mem_observe.mp hpair).2

/-- Every part of the graph can be represented by selecting its input words. -/
theorem exists_observe_eq_of_subset_graph [Fintype A] (f : Word A T → A)
    (part : Finset (Word A T × A)) (hpart : part ⊆ graph f) :
    ∃ inputs, observe inputs f = part := by
  classical
  refine ⟨part.image Prod.fst, ?_⟩
  ext pair
  constructor
  · intro hpair
    obtain ⟨hinput, hlabel⟩ := mem_observe.mp hpair
    obtain ⟨other, hother, hfst⟩ := Finset.mem_image.mp hinput
    have hotherLabel := mem_graph.mp (hpart hother)
    have heq : other = pair := Prod.ext hfst (by rw [← hotherLabel, hfst, hlabel])
    exact heq ▸ hother
  · intro hpair
    exact mem_observe.mpr ⟨Finset.mem_image_of_mem _ hpair, mem_graph.mp (hpart hpair)⟩

/-- A proper part of the graph leaves at least one input unobserved. -/
theorem observe_proper_iff [Fintype A] (inputs : Finset (Word A T)) (f : Word A T → A) :
    observe inputs f ⊂ graph f ↔ ∃ word, word ∉ inputs := by
  classical
  rw [Finset.ssubset_iff_subset_ne]
  constructor
  · rintro ⟨_, hne⟩
    by_contra! hall
    exact hne (congrArg (fun selected => observe selected f) (Finset.eq_univ_of_forall hall))
  · rintro ⟨word, hword⟩
    refine ⟨observe_subset_graph inputs f, ?_⟩
    intro heq
    have hp : (word, f word) ∈ graph f := mem_graph.mpr rfl
    rw [← heq] at hp
    exact hword (mem_observe.mp hp).1

/-- Two graphs reveal the same observations exactly when their labels agree on the selected words. -/
theorem observe_eq_iff_eqOn {inputs : Finset (Word A T)} {f g : Word A T → A} :
    observe inputs f = observe inputs g ↔ Set.EqOn f g (inputs : Set (Word A T)) := by
  constructor
  · intro heq word hword
    have hp : (word, f word) ∈ observe inputs f := mem_observe.mpr ⟨hword, rfl⟩
    rw [heq] at hp
    exact (mem_observe.mp hp).2.symm
  · intro hagree
    ext pair
    by_cases hword : pair.1 ∈ inputs
    · simp [hword, hagree hword]
    · simp [hword]

/-- A deterministic training procedure returns a predictor from labeled pairs. -/
abbrev Learner (A : Type*) (T : ℕ) := Finset (Word A T × A) → Word A T → A

/-- A single learner exactly recovers every target in the given class. -/
def Recovers (inputs : Finset (Word A T)) (family : Set (Word A T → A))
    (learn : Learner A T) : Prop :=
  ∀ ⦃f⦄, f ∈ family → learn (observe inputs f) = f

/-- A recovery rule exists exactly when observations distinguish the admissible targets. -/
theorem exists_learner_iff_injOn [Nonempty A] (inputs : Finset (Word A T))
    (family : Set (Word A T → A)) :
    (∃ learn, Recovers inputs family learn) ↔ Set.InjOn (observe inputs) family := by
  constructor
  · rintro ⟨learn, hlearn⟩ f hf g hg heq
    calc
      f = learn (observe inputs f) := (hlearn hf).symm
      _ = learn (observe inputs g) := congrArg learn heq
      _ = g := hlearn hg
  · intro hinjective
    classical
    let learn : Learner A T := fun part =>
      if h : ∃ f ∈ family, observe inputs f = part then Classical.choose h
      else fun _ => Classical.choice (inferInstance : Nonempty A)
    refine ⟨learn, ?_⟩
    intro f hf
    have h : ∃ g ∈ family, observe inputs g = observe inputs f := ⟨f, hf, rfl⟩
    simpa only [learn, dite_eq_left h] using
      hinjective (Classical.choose_spec h).1 hf (Classical.choose_spec h).2

/-- A hidden label can be changed for every target if the alphabet has two tokens. -/
theorem exists_distinct_target [Fintype A] (inputs : Finset (Word A T)) (f : Word A T → A)
    (hcard : 2 ≤ Fintype.card A) (hmissing : ∃ word, word ∉ inputs) :
    ∃ g : Word A T → A, g ≠ f ∧ observe inputs g = observe inputs f := by
  classical
  have : Nontrivial A := Fintype.one_lt_card_iff_nontrivial.mp hcard
  obtain ⟨word, hword⟩ := hmissing
  obtain ⟨token, htoken⟩ := exists_ne (f word)
  refine ⟨Function.update f word token, ?_, observe_eq_iff_eqOn.mpr ?_⟩
  · intro h
    apply htoken
    simpa using congrFun h word
  · intro other hother
    apply Function.update_of_ne
    intro heq
    exact hword (heq ▸ hother)

/-- No learner, including any fixed Argand training procedure, recovers every
unrestricted target from a proper graph part with at least two tokens. -/
theorem no_unrestricted_recovery [Fintype A] (inputs : Finset (Word A T))
    (hcard : 2 ≤ Fintype.card A) (hmissing : ∃ word, word ∉ inputs) :
    ¬ ∃ learn, Recovers inputs Set.univ learn := by
  have : Nontrivial A := Fintype.one_lt_card_iff_nontrivial.mp hcard
  let f : Word A T → A := fun _ => Classical.choice (inferInstance : Nonempty A)
  obtain ⟨g, hne, heq⟩ := exists_distinct_target inputs f hcard hmissing
  intro h
  have hinjective := (exists_learner_iff_injOn inputs Set.univ).mp h
  exact hne (hinjective (Set.mem_univ g) (Set.mem_univ f) heq)

private def zeroWord : Word (Fin 2) 1 := fun _ => 0
private def oneInput : Finset (Word (Fin 2) 1) := {zeroWord}
private def zeroTarget : Word (Fin 2) 1 → Fin 2 := fun _ => 0

private theorem oneInput_has_hidden_word : ∃ word, word ∉ oneInput := by
  refine ⟨fun _ => 1, ?_⟩
  intro h
  have heq : (1 : Fin 2) = 0 := congrFun (Finset.mem_singleton.mp h) 0
  exact (by decide : (1 : Fin 2) ≠ 0) heq

example : Fintype.card (Word (Fin 2) 3) = 8 := by decide

example : ∃ inputs, observe inputs zeroTarget = {(zeroWord, 0)} :=
  exists_observe_eq_of_subset_graph _ _ (Finset.singleton_subset_iff.mpr (mem_graph.mpr rfl))

example : ∃ g, g ≠ zeroTarget ∧ observe oneInput g = observe oneInput zeroTarget :=
  exists_distinct_target oneInput zeroTarget (by decide) oneInput_has_hidden_word

example : ¬ ∃ learn, Recovers oneInput Set.univ learn :=
  no_unrestricted_recovery oneInput (by decide) oneInput_has_hidden_word

/-- One labeled word determines a constant target on a binary alphabet. -/
example : ∃ learn, Recovers oneInput {f | ∃ token, f = fun _ => token} learn := by
  apply (exists_learner_iff_injOn oneInput _).mpr
  rintro f ⟨a, rfl⟩ g ⟨b, rfl⟩ heq
  have hzero : zeroWord ∈ oneInput := Finset.mem_singleton_self _
  have hab : a = b := (observe_eq_iff_eqOn.mp heq) hzero
  exact congrArg (fun token : Fin 2 => fun _ : Word (Fin 2) 1 => token) hab

end Transformer.Grokking.FiniteRecovery
