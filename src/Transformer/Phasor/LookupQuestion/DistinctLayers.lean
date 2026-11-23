module

public import Transformer.Phasor.LookupQuestion.LongChains
public import Mathlib.Logic.Equiv.Fin.Basic
public import Mathlib.Data.Fintype.Prod

/-! The synthetic length-only control prints a separate name for every key at
every layer. Five keys and forty lookup tables require 205 token names even
when all five questions reuse the same printed tables and choices. -/
@[expose] public section
namespace Transformer.Phasor.LookupQuestion.DistinctLayers

/-- The source and final layers both require names. This is the exact
alphabet size for injective single-token names across all lanes and layers. -/
theorem capacity (h k vocab : ℕ) :
    (∃ labels : Fin (h+1) × Fin k → Fin vocab, Function.Injective labels) ↔ (h+1)*k ≤ vocab := by
  constructor
  · rintro ⟨labels, separate⟩
    simpa using Fintype.card_le_of_injective labels separate
  · intro enough
    refine ⟨fun i => Fin.castLE enough (finProdFinEquiv i), ?_⟩
    exact (Fin.castLE_injective enough).comp finProdFinEquiv.injective

example : ¬∃ labels : Fin 21 × Fin 5 → Fin 60, Function.Injective labels := by
  simpa using mt (capacity 20 5 60).mp (by decide)

example : ∃ labels : Fin 41 × Fin 5 → Fin 205, Function.Injective labels :=
  (capacity 40 5 205).mpr (by decide)

end Transformer.Phasor.LookupQuestion.DistinctLayers
