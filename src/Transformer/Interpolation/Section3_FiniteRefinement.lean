module

public import Mathlib.Data.Fintype.EquivFin
public import Mathlib.SetTheory.Cardinal.NatCard
public import Mathlib.Data.Set.Finite.Range
public import Mathlib.Logic.Function.Iterate

/-!
# Finite progress of barycenter class refinement

Auxiliary combinatorics for arXiv:2411.04551v3, §3.2,
`prop: separation`. Preserving distinct original classes while splitting
one class strictly increases the number of class values. Equivalently,
the difference between the finite index count and the class count falls.

A deterministic sequence of such refinements reaches an injective class
map after at most the number of indices, provided completed states remain
completed. The hypotheses are explicit transition properties; the actual
ridge phase proving them for barycenter directions is supplied separately.
No measure, control or weak-solution conclusion is assumed in this kernel.
-/

@[expose] public section

open Set

namespace Transformer.Interpolation

/-- A strict refinement has more actual class values. Source auxiliary:
arXiv:2411.04551v3, §3.2, finite progress toward `prop: separation`. -/
theorem card_range_lt_of_strict_refinement {ι α β : Type*} [Fintype ι]
    (f : ι → α) (g : ι → β)
    (hpreserve : ∀ i j, f i ≠ f j → g i ≠ g j)
    (hsplit : ∃ i j, f i = f j ∧ g i ≠ g j) :
    Nat.card (Set.range f) < Nat.card (Set.range g) := by
  classical
  let rep (y : Set.range g) : ι := Classical.choose y.2
  have hrep (y : Set.range g) : g (rep y) = y.1 := Classical.choose_spec y.2
  let φ : Set.range g → Set.range f := fun y => ⟨f (rep y), mem_range_self _⟩
  have hfactor (i : ι) :
      φ (rangeFactorization g i) = rangeFactorization f i := by
    apply Subtype.ext
    by_contra hne
    exact hpreserve (rep (rangeFactorization g i)) i hne (hrep _)
  have hsurj : Function.Surjective φ := by
    rintro ⟨_, i, rfl⟩
    exact ⟨rangeFactorization g i, hfactor i⟩
  -- Two newly distinct classes still have the same original class value.
  have hni : ¬Function.Injective φ := by
    intro hi
    obtain ⟨i, j, hf, hg⟩ := hsplit
    have he : φ (rangeFactorization g i) = φ (rangeFactorization g j) := by
      rw [hfactor, hfactor]
      exact Subtype.ext hf
    exact hg (congrArg Subtype.val (hi he))
  simpa only [Nat.card_eq_fintype_card] using
    Fintype.card_lt_of_surjective_not_injective φ hsurj hni

/-- A strict refinement lowers the finite class deficit. Source auxiliary:
arXiv:2411.04551v3, §3.2, the bound on perturbation phases. -/
theorem refinement_defect_lt {ι α β : Type*} [Fintype ι]
    (f : ι → α) (g : ι → β)
    (hpreserve : ∀ i j, f i ≠ f j → g i ≠ g j)
    (hsplit : ∃ i j, f i = f j ∧ g i ≠ g j) :
    Fintype.card ι - Nat.card (Set.range g) <
      Fintype.card ι - Nat.card (Set.range f) := by
  have hlt := card_range_lt_of_strict_refinement f g hpreserve hsplit
  have hbound := Finite.card_range_le g
  rw [Nat.card_eq_fintype_card (α := ι)] at hbound
  omega

/-- Each uncompleted iterate has more distinct values than elapsed
steps. Source auxiliary: arXiv:2411.04551v3, §3.2, finite separation. -/
theorem refinement_iterate_progress {S ι α : Type*} [Fintype ι] [Nonempty ι]
    (F : S → S) (f : S → ι → α)
    (hstep : ∀ x, ¬Function.Injective (f x) →
      (∀ i j, f x i ≠ f x j → f (F x) i ≠ f (F x) j) ∧
      ∃ i j, f x i = f x j ∧ f (F x) i ≠ f (F x) j)
    (hdone : ∀ x, Function.Injective (f x) → Function.Injective (f (F x)))
    (x : S) (n : ℕ) :
    Function.Injective (f ((F^[n]) x)) ∨
      n < Nat.card (Set.range (f ((F^[n]) x))) := by
  classical
  induction n with
  | zero =>
    right
    let k : ι := Classical.arbitrary ι
    have : Nonempty (Set.range (f x)) := ⟨⟨f x k, mem_range_self k⟩⟩
    exact Nat.card_pos
  | succ n ih =>
    rw [Function.iterate_succ_apply']
    rcases ih with hi | hn
    · exact Or.inl (hdone _ hi)
    · by_cases hi : Function.Injective (f ((F^[n]) x))
      · exact Or.inl (hdone _ hi)
      · right
        obtain ⟨hp, hs⟩ := hstep ((F^[n]) x) hi
        have hh := card_range_lt_of_strict_refinement _ _ hp hs
        omega

/-- At most the number of original indices suffices for an injective
class map. Source auxiliary: arXiv:2411.04551v3, §3.2, switch count. -/
theorem injective_after_card_refinements {S ι α : Type*} [Fintype ι] [Nonempty ι]
    (F : S → S) (f : S → ι → α)
    (hstep : ∀ x, ¬Function.Injective (f x) →
      (∀ i j, f x i ≠ f x j → f (F x) i ≠ f (F x) j) ∧
      ∃ i j, f x i = f x j ∧ f (F x) i ≠ f (F x) j)
    (hdone : ∀ x, Function.Injective (f x) → Function.Injective (f (F x)))
    (x : S) : Function.Injective (f ((F^[Fintype.card ι]) x)) := by
  have hh := refinement_iterate_progress F f hstep hdone x (Fintype.card ι)
  rcases hh with hi | hlt
  · exact hi
  · have hbound := Finite.card_range_le (f ((F^[Fintype.card ι]) x))
    rw [Nat.card_eq_fintype_card (α := ι)] at hbound
    omega

/-- Three indices start in two classes. The first genuine refinement
splits the first class while preserving its distinction from the second;
later states retain all three classes. This realizes the strict and
iterated progress hypotheses with nonempty finite data. -/
example : ∃ (F : ℕ → ℕ) (f : ℕ → Fin 3 → Fin 3),
    (∀ x, ¬Function.Injective (f x) →
      (∀ i j, f x i ≠ f x j → f (F x) i ≠ f (F x) j) ∧
      ∃ i j, f x i = f x j ∧ f (F x) i ≠ f (F x) j) ∧
    (∀ x, Function.Injective (f x) → Function.Injective (f (F x))) ∧
    (∃ i j, f 0 i = f 0 j ∧ f (F 0) i ≠ f (F 0) j) ∧
    ∃ i j, f 0 i ≠ f 0 j := by
  let F : ℕ → ℕ := Nat.succ
  let f (n : ℕ) (i : Fin 3) : Fin 3 := if n = 0 then if i = 2 then 1 else 0 else i
  have hs (n : ℕ) : f (F n) = id := by
    funext i
    simp [f, F]
  -- Every positive state already has one value per original index.
  have hi (n : ℕ) : Function.Injective (f (F n)) := by
    rw [hs]
    exact Function.injective_id
  have hstep (n : ℕ) (hn : ¬Function.Injective (f n)) :
      (∀ i j, f n i ≠ f n j → f (F n) i ≠ f (F n) j) ∧
      ∃ i j, f n i = f n j ∧ f (F n) i ≠ f (F n) j := by
    have hn₀ : n = 0 := by
      by_contra hne
      apply hn
      intro i j hij
      simpa only [f, ite_eq_right hne] using hij
    subst n
    constructor
    · intro i j hne he
      exact hne (congrArg (f 0) (hi 0 he))
    · exact ⟨0, 1, by decide, by decide⟩
  exact ⟨F, f, hstep, fun n _ => hi n,
    ⟨0, 1, by decide, by decide⟩,
    ⟨0, 2, by decide⟩⟩

end Transformer.Interpolation
