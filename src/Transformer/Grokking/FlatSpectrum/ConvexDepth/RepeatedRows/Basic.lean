module

public import Mathlib.Data.Fintype.Card
public import Mathlib.Tactic

/-!
# Lossless rows with a shared value and explicit exceptions

Source: `review/flat_spectrum.md`, CM70. Values need only have equality: the
representation applies equally to real coefficients and their stored bit words.
Every differing coordinate has its own explicit entry. No arithmetic average,
uniformity condition, or restriction on admissible rows enters the encoding.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.RepeatedRows

variable {C A : Type*}

/-- A shared value and all explicit exceptions, CM70. The dependent value
function stores entries only for the finite exception set. -/
structure Row (C A : Type*) where
  baseline : A
  exceptions : Finset C
  values : {c // c ∈ exceptions} → A

/-- Recover an actual coordinate, including each explicit exception, CM70. -/
noncomputable def read (row : Row C A) (c : C) : A := by
  classical
  exact if h : c ∈ row.exceptions then row.values ⟨c, h⟩ else row.baseline

/-- Number of stored values, excluding index and container overhead, CM70.
Native memory forecasts must charge those additional bytes separately. -/
def slots (row : Row C A) : ℕ := 1 + row.exceptions.card

variable [Fintype C]

/-- Encode every coordinate differing from an arbitrary shared value, CM70.
The shared value can be selected for storage efficiency without changing a row. -/
noncomputable def encode (baseline : A) (row : C → A) : Row C A := by
  classical
  exact ⟨baseline, Finset.univ.filter (fun c => row c ≠ baseline), fun c => row c.1⟩

/-- The exception list is exactly the set of differing original coordinates,
CM70; none is silently discarded or rounded. -/
theorem mem_encode (baseline : A) (row : C → A) (c : C) :
    c ∈ (encode baseline row).exceptions ↔ row c ≠ baseline := by
  classical
  change c ∈ Finset.univ.filter (fun d => row d ≠ baseline) ↔ _
  exact Finset.mem_filter_univ c

/-- Reading the encoding returns every original value, CM70. This identity
is generic in the value type and therefore also preserves exact bit words. -/
theorem read_encode (baseline : A) (row : C → A) : read (encode baseline row) = row := by
  classical
  funext c
  by_cases h : row c = baseline
  · have hc : c ∉ (encode baseline row).exceptions := by
      intro hm
      exact (mem_encode baseline row c).mp hm h
    simp only [read, dite_eq_right hc]
    exact h.symm
  · have hc := (mem_encode baseline row c).mpr h
    simp only [read, dite_eq_left hc]
    rfl

/-- Distinct complete rows cannot collapse to one encoding, CM70. -/
theorem encode_injective (baseline : A) : Function.Injective (encode (C := C) baseline) := by
  intro a b h
  have he := congrArg read h
  rw [read_encode, read_encode] at he
  exact he

/-- A uniform row needs just its single shared value, including a zero row,
CM70; this is a storage fact rather than a new feasible-set constraint. -/
theorem slots_constant (value : A) : slots (encode value (fun _ : C => value)) = 1 := by
  classical
  unfold slots encode
  simp

/-- Selecting any actual class value as the baseline uses at most the original
number of values, CM70. Completely nonuniform rows remain representable. -/
theorem slots_le_card (row : C → A) (c₀ : C) :
    slots (encode (row c₀) row) ≤ Fintype.card C := by
  classical
  have hsub : (encode (row c₀) row).exceptions ⊆ Finset.univ :=
    Finset.filter_subset _ _
  have hne : (encode (row c₀) row).exceptions ≠ Finset.univ := by
    intro he
    have hm : c₀ ∈ (encode (row c₀) row).exceptions := by rw [he]; exact Finset.mem_univ _
    exact (mem_encode (row c₀) row c₀).mp hm rfl
  have hc := Finset.card_lt_card (Finset.ssubset_iff_subset_ne.mpr ⟨hsub, hne⟩)
  rw [Finset.card_univ] at hc
  unfold slots
  omega

/-- If a row agrees with its baseline outside an explicit class set, every
stored exception lies inside it, CM70. This also applies to complex columns. -/
theorem exceptions_subset (baseline : A) (row : C → A) (allowed : Finset C)
    (h : ∀ c, c ∉ allowed → row c = baseline) :
    (encode baseline row).exceptions ⊆ allowed := by
  intro c hc
  by_contra hn
  exact (mem_encode baseline row c).mp hc (h c hn)

/-- A bound on actual exceptional classes bounds stored values, CM70.
Every omitted class still reads back as its exact original baseline value. -/
theorem slots_le_exceptions (baseline : A) (row : C → A) (allowed : Finset C)
    (h : ∀ c, c ∉ allowed → row c = baseline) :
    slots (encode baseline row) ≤ 1 + allowed.card := by
  have hc := Finset.card_le_card (exceptions_subset baseline row allowed h)
  exact Nat.add_le_add_left hc 1

example := mem_encode (0 : ℕ) (fun b : Bool => if b then 3 else 0) true
example := read_encode (0 : ℕ) (fun b : Bool => if b then 3 else 0)
example := encode_injective (C := Bool) (0 : ℕ)
example := slots_constant (C := Bool) (0 : ℕ)
example := slots_le_card (fun b : Bool => if b then (3 : ℕ) else 1) false
example := exceptions_subset (0 : ℕ) (fun b : Bool => if b then 3 else 0) {true}
  (by intro c hc; cases c <;> simp_all)
example := slots_le_exceptions (0 : ℕ) (fun b : Bool => if b then 3 else 0) {true}
  (by intro c hc; cases c <;> simp_all)

end Transformer.Grokking.FlatSpectrum.ConvexDepth.RepeatedRows
