module

public import Mathlib.InformationTheory.Hamming

/-!
# What a part of the table determines: the two definitions

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT1. A class `H` of
tables on a set `B` of cells is *determined* by a part `Ω ⊆ B` when two members of `H` that agree
on `Ω` are equal (`Determines`). `δ(H)` is the least number of cells on which two different
members of `H` differ (`minDistance`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart

variable {B A : Type*}

/-- A part `Ω` of the cells *determines* the class `H` (`review/flat_spectrum.md`, PT1): two
members of `H` that agree on `Ω` are equal. -/
def Determines (H : Set (B → A)) (Ω : Set B) : Prop :=
  ∀ f ∈ H, ∀ f' ∈ H, (∀ x ∈ Ω, f x = f' x) → f = f'

/-- A larger part determines whatever a smaller one does. -/
lemma Determines.mono {H : Set (B → A)} {Ω Ω' : Set B} (h : Determines H Ω) (hs : Ω ⊆ Ω') :
    Determines H Ω' :=
  fun f hf f' hf' hag => h f hf f' hf' fun x hx => hag x (hs hx)

example : Determines (Set.univ : Set (Fin 1 → Bool)) Set.univ :=
  Determines.mono (Ω := Set.univ) (fun _ _ _ _ h => funext fun x => h x (Set.mem_univ x))
    subset_rfl

/-- `δ(H)` (`review/flat_spectrum.md`, PT1): the least number of cells on which two different
members of `H` differ. -/
noncomputable def minDistance [Fintype B] [DecidableEq A] (H : Set (B → A)) : ℕ :=
  sInf {d | ∃ f ∈ H, ∃ f' ∈ H, f ≠ f' ∧ hammingDist f f' = d}

variable [Fintype B] [DecidableEq A] {H : Set (B → A)}

/-- With two different members, the least distance is attained by a pair of them. -/
lemma exists_hammingDist_eq_minDistance (hH : H.Nontrivial) :
    ∃ f ∈ H, ∃ f' ∈ H, f ≠ f' ∧ hammingDist f f' = minDistance H := by
  obtain ⟨f, hf, f', hf', hne⟩ := hH
  exact Nat.sInf_mem (s := {d | ∃ f ∈ H, ∃ f' ∈ H, f ≠ f' ∧ hammingDist f f' = d})
    ⟨_, f, hf, f', hf', hne, rfl⟩

example : ∃ f ∈ (Set.univ : Set (Fin 2 → Bool)), ∃ f' ∈ (Set.univ : Set (Fin 2 → Bool)),
    f ≠ f' ∧ hammingDist f f' = minDistance (Set.univ : Set (Fin 2 → Bool)) :=
  exists_hammingDist_eq_minDistance Set.nontrivial_univ

/-- Two different members differ on at least `δ(H)` cells. -/
lemma minDistance_le {f f' : B → A} (hf : f ∈ H) (hf' : f' ∈ H) (hne : f ≠ f') :
    minDistance H ≤ hammingDist f f' :=
  Nat.sInf_le ⟨f, hf, f', hf', hne, rfl⟩

example : minDistance (Set.univ : Set (Fin 1 → Bool)) ≤
    hammingDist (fun _ => true : Fin 1 → Bool) (fun _ => false) :=
  minDistance_le (Set.mem_univ _) (Set.mem_univ _) fun h => by simpa using congrFun h 0

end Transformer.Grokking.FlatSpectrum.TablePart
