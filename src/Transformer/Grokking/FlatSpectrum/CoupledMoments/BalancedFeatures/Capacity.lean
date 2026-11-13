module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.BalancedFeatures.Completion

/-!
# A capacity obstruction of surjective phase assignments

Source: `review/flat_spectrum.md`, CM24. If one axis covers the entire latent
group, a constant opposite row forces the decoder and the whole table to be
constant. This restriction belongs to the native permutation proposal family;
CM15's broader convex hull does not impose surjective phase assignments.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.BalancedFeatures

variable {G m n C : Type*} [AddGroup G]

/-- A fully covered latent axis and one constant row force the whole decoded
phase table to be constant, CM24. More optimization cannot remove this obstruction. -/
theorem constant_row_forces_constant (d : G → C) (r : m → G) (s : n → G)
    (i₀ : m) (c : C) (hs : Function.Surjective s)
    (hrow : ∀ j, d (r i₀ + s j) = c) : ∀ i j, d (r i + s j) = c := by
  intro i j
  obtain ⟨k, hk⟩ := hs (-r i₀ + (r i + s j))
  have h := hrow k
  rw [hk, add_neg_cancel_left] at h
  exact h

/-- An observed constant row cannot coexist with a different value when the
row observations cover every column phase, CM24. This is a representation
obstruction, not a lower bound inferred from partially observed labels. -/
theorem no_constant_row (d : G → C) (r : m → G) (s : n → G)
    (i₀ : m) (c : C) (hs : Function.Surjective s)
    (hdiff : ∃ i j, d (r i + s j) ≠ c) : ¬ ∀ j, d (r i₀ + s j) = c := by
  intro hrow
  obtain ⟨i, j, hij⟩ := hdiff
  exact hij (constant_row_forces_constant d r s i₀ c hs hrow i j)

example := constant_row_forces_constant (fun _ : ZMod 2 => false)
  (id : ZMod 2 → ZMod 2) id 0 false (by intro z; exact ⟨z, rfl⟩) (by simp)
example := no_constant_row (id : ZMod 2 → ZMod 2) id id 0 0
  (by intro z; exact ⟨z, rfl⟩) ⟨0, 1, by decide⟩

end Transformer.Grokking.FlatSpectrum.CoupledMoments.BalancedFeatures
