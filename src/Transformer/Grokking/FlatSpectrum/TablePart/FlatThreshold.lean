module

public import Transformer.Grokking.FlatSpectrum.TablePart.FlatWorst

/-!
# PT4: what the worst part says

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT4, *The worst*. Every
part of more than `q^{k+l} − min(q^k, q^l)` cells determines the flat class, and no smaller number
guarantees it (`forall_determines_flatClass_iff`). The table without a row or a column does not
determine it (`not_determines_off_row`, `not_determines_off_col`). At `q = 2`, `k = l = 1`, the one
exception, the member `a b` has a single nonzero cell (`minDistance_flatClass_two`). At `p = 97`,
`δ = 97` (`minDistance_flatClass_97`), every 9313 cells determine the class
(`forall_determines_flat_9313`), and the 9312 cells outside one row do not (`card_off_row_97`,
`not_determines_off_row`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart

open Finset Matrix Module

variable {F ι κ : Type*} [Field F] [Fintype F] [DecidableEq F]
  [Fintype ι] [Fintype κ] [DecidableEq ι] [DecidableEq κ]

omit [Fintype F] [DecidableEq ι] [DecidableEq κ] in
/-- The flat class has two different members. -/
lemma flatClass_nontrivial : (flatClass F ι κ : Set ((ι → F) × (κ → F) → F)).Nontrivial := by
  refine ⟨0, Submodule.zero_mem _, _, row_indicator_mem (κ := κ) (0 : ι → F), fun h => ?_⟩
  simpa using congrFun h (0, 0)

/-- **The threshold of the flat class** (`review/flat_spectrum.md`, PT4): outside `q = 2`,
`k = l = 1`, every part of `m` cells determines it exactly when `m > q^{k+l} − min(q^k, q^l)`. -/
theorem forall_determines_flatClass_iff
    (hne : ¬ (Fintype.card F = 2 ∧ Fintype.card ι = 1 ∧ Fintype.card κ = 1)) (m : ℕ) :
    (∀ Ω : Finset ((ι → F) × (κ → F)), #Ω = m →
        Determines (flatClass F ι κ : Set ((ι → F) × (κ → F) → F)) Ω) ↔
      Fintype.card F ^ Fintype.card ι * Fintype.card F ^ Fintype.card κ -
        min (Fintype.card F ^ Fintype.card ι) (Fintype.card F ^ Fintype.card κ) < m := by
  rw [forall_determines_iff flatClass_nontrivial, minDistance_flatClass hne, Fintype.card_prod,
    Fintype.card_fun, Fintype.card_fun]

example := forall_determines_flatClass_iff (F := ZMod 3) (ι := Fin 1) (κ := Fin 1)
  (by simp [ZMod.card]) 7

omit [Fintype F] [DecidableEq ι] [DecidableEq κ] in
/-- **The table without a row does not determine the class** (`review/flat_spectrum.md`, PT4): the
indicator of the row vanishes on the rest. -/
theorem not_determines_off_row (u₀ : ι → F) :
    ¬ Determines (flatClass F ι κ : Set ((ι → F) × (κ → F) → F)) {c | c.1 ≠ u₀} := by
  intro h
  have h0 := (determines_submodule_iff _ _).1 h _ (row_indicator_mem (κ := κ) u₀) fun c hc => by
    have hc' : c.1 ≠ u₀ := hc
    simp [hc']
  simpa using congrFun h0 (u₀, 0)

omit [Fintype F] [DecidableEq ι] [DecidableEq κ] in
/-- The table without a column does not determine the class (`review/flat_spectrum.md`, PT4). -/
theorem not_determines_off_col (v₀ : κ → F) :
    ¬ Determines (flatClass F ι κ : Set ((ι → F) × (κ → F) → F)) {c | c.2 ≠ v₀} := by
  intro h
  have h0 := (determines_submodule_iff _ _).1 h _ (col_indicator_mem (ι := ι) v₀) fun c hc => by
    have hc' : c.2 ≠ v₀ := hc
    simp [hc']
  simpa using congrFun h0 (0, v₀)

/-- The table without a row has `q^{k+l} − q^l` cells, written without subtraction. -/
lemma card_off_row (u₀ : ι → F) :
    #{c : (ι → F) × (κ → F) | c.1 ≠ u₀} + Fintype.card F ^ Fintype.card κ =
      Fintype.card F ^ Fintype.card ι * Fintype.card F ^ Fintype.card κ := by
  have hrow : #{c : (ι → F) × (κ → F) | c.1 = u₀} = Fintype.card F ^ Fintype.card κ := by
    rw [← hammingNorm_row (κ := κ) u₀]
    refine congrArg card (filter_congr fun c _ => ?_)
    by_cases h : c.1 = u₀ <;> simp [h]
  have hsplit := card_filter_add_card_filter_not (s := (univ : Finset ((ι → F) × (κ → F))))
    (fun c => c.1 = u₀)
  rw [hrow, card_univ, Fintype.card_prod, Fintype.card_fun, Fintype.card_fun] at hsplit
  have hne : #{c : (ι → F) × (κ → F) | c.1 ≠ u₀} = #{c : (ι → F) × (κ → F) | ¬ c.1 = u₀} := rfl
  omega

/-- **The exception** (`review/flat_spectrum.md`, PT4): at `q = 2`, `k = l = 1`, the member `a b`
has one nonzero cell, so `δ = 1`. -/
theorem minDistance_flatClass_two :
    minDistance (flatClass (ZMod 2) (Fin 1) (Fin 1) :
      Set ((Fin 1 → ZMod 2) × (Fin 1 → ZMod 2) → ZMod 2)) = 1 := by
  rw [minDistance_submodule]
  refine IsLeast.csInf_eq ⟨⟨fun c => c.1 0 * c.2 0, ?_, ?_, ?_⟩, ?_⟩
  · exact mem_flatClass.2 ⟨0, 0, 1, fun u v => by simp [dotProduct, mulVec]⟩
  · intro h
    simpa using congrFun h (fun _ => 1, fun _ => 1)
  · decide +kernel
  · rintro _ ⟨f, -, h0, rfl⟩
    exact hammingNorm_pos_iff.2 h0

section Numbers

attribute [local instance] fact_prime_97

/-- At `p = 97`, `δ = 97` (`review/flat_spectrum.md`, PT4). -/
theorem minDistance_flatClass_97 :
    minDistance (flatClass (ZMod 97) (Fin 1) (Fin 1) :
      Set ((Fin 1 → ZMod 97) × (Fin 1 → ZMod 97) → ZMod 97)) = 97 := by
  rw [minDistance_flatClass (by simp [ZMod.card])]
  simp [ZMod.card]

/-- **At `p = 97` every 9313 cells determine the flat class** (`review/flat_spectrum.md`, PT4). -/
theorem forall_determines_flat_9313 (Ω : Finset ((Fin 1 → ZMod 97) × (Fin 1 → ZMod 97)))
    (hΩ : #Ω = 9313) : Determines (flatClass (ZMod 97) (Fin 1) (Fin 1) :
      Set ((Fin 1 → ZMod 97) × (Fin 1 → ZMod 97) → ZMod 97)) Ω :=
  (forall_determines_flatClass_iff (F := ZMod 97) (ι := Fin 1) (κ := Fin 1)
    (by simp [ZMod.card]) 9313).2 (by simp [ZMod.card]) Ω hΩ

example : ∃ Ω : Finset ((Fin 1 → ZMod 97) × (Fin 1 → ZMod 97)), #Ω = 9313 ∧
    Determines (flatClass (ZMod 97) (Fin 1) (Fin 1) :
      Set ((Fin 1 → ZMod 97) × (Fin 1 → ZMod 97) → ZMod 97)) Ω := by
  obtain ⟨Ω, -, hΩ⟩ := exists_subset_card_eq
    (s := (univ : Finset ((Fin 1 → ZMod 97) × (Fin 1 → ZMod 97)))) (n := 9313)
    (by simp [card_univ, ZMod.card])
  exact ⟨Ω, hΩ, forall_determines_flat_9313 Ω hΩ⟩

/-- The table without a row has 9312 cells at `p = 97` (`review/flat_spectrum.md`, PT4). -/
theorem card_off_row_97 (u₀ : Fin 1 → ZMod 97) :
    #{c : (Fin 1 → ZMod 97) × (Fin 1 → ZMod 97) | c.1 ≠ u₀} = 9312 := by
  have h := card_off_row (κ := Fin 1) u₀
  simp only [ZMod.card, Fintype.card_fin] at h
  omega

end Numbers

end Transformer.Grokking.FlatSpectrum.TablePart
