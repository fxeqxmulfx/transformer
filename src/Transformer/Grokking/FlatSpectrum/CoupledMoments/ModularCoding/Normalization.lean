module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.ModularCoding.Recombine

/-!
# Unit gauges in composite categorical phase domains

Source: `review/flat_spectrum.md`, CM27. Unit scaling and two shifts preserve
observed categorical equations and every decoded prediction. No prime-domain
or one-dimensional-kernel assumption appears here.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.ModularCoding

variable {q : ℕ} {I C S : Type*}

/-- Shift and scale a categorical phase or class code, CM27. -/
def normalize (u b : ZMod q) (z : C → ZMod q) (c : C) : ZMod q := u * (z c - b)

/-- The opposite-axis shift preserving the sum relation, CM27. -/
def column (u a b : ZMod q) (s : I → ZMod q) (i : I) : ZMod q := u * (s i + a - b)

/-- Transform an actual proposed decoder with the checked inverse unit, CM27. -/
def redecode (v b : ZMod q) (d : ZMod q → C) (z : ZMod q) : C := d (v * z + b)

/-- Categorical sample relations survive the explicit unit gauge; the equality
itself even holds for arbitrary scalars, CM27. Categories carry no numeric rule. -/
theorem normalize_fit {cell : S → I × I} {labels : S → C}
    {r s : I → ZMod q} {z : C → ZMod q}
    (h : ∀ t, r (cell t).1 + s (cell t).2 = z (labels t)) (u a b : ZMod q) :
    ∀ t, normalize u a r (cell t).1 + column u a b s (cell t).2 =
      normalize u b z (labels t) := by
  intro t
  unfold normalize column
  linear_combination u * h t

/-- The chosen row and class anchors become zero, and a checked unit-scaled
class difference becomes one, CM27. These are the actual proposed gauges. -/
theorem normalized_anchors (r : I → ZMod q) (z : C → ZMod q) (i₀ : I) (c₀ c₁ : C)
    (u : ZMod q) (h : u * (z c₁ - z c₀) = 1) :
    normalize u (r i₀) r i₀ = 0 ∧ normalize u (z c₀) z c₀ = 0 ∧
      normalize u (z c₀) z c₁ = 1 := by
  refine ⟨?_, ?_, h⟩ <;> simp only [normalize, sub_self, mul_zero]

/-- A checked inverse preserves injectivity of the learned output code, CM27.
This remains valid at a composite modulus. -/
theorem normalize_injective {u v b : ZMod q} {z : C → ZMod q}
    (hu : v * u = 1) (hz : Function.Injective z) :
    Function.Injective (normalize u b z) := by
  intro x y h
  apply hz
  unfold normalize at h
  linear_combination v * h - (z x - z y) * hu

/-- The transformed actual decoder preserves the full prediction table under
the checked unit gauge, CM27. This is an identity for a supplied proposal,
not a claim that training labels identify the unknown decoder. -/
theorem redecode_identity {u v : ZMod q} (hu : v * u = 1)
    (d : ZMod q → C) (r s a b : ZMod q) :
    redecode v b d (u * (r - a) + u * (s + a - b)) = d (r + s) := by
  unfold redecode
  congr 1
  linear_combination (r + s - b) * hu

example := normalize_fit
  (cell := (![(1, 2), (3, 5)] : Fin 2 → ZMod 6 × ZMod 6))
  (labels := (![3, 2] : Fin 2 → ZMod 6)) (r := id) (s := id) (z := id)
  (by decide) (5 : ZMod 6) 3 2

example := normalized_anchors (id : ZMod 6 → ZMod 6) id 3 2 1 5 (by decide)

example := normalize_injective (u := (5 : ZMod 6)) (v := 5) (b := 2)
  (z := id) (by decide) (by intro x y h; exact h)

example := redecode_identity (u := (5 : ZMod 6)) (v := 5) (by decide)
  (fun z : ZMod 6 => decide (z = 2 ∨ z = 3)) 4 5 3 2

end Transformer.Grokking.FlatSpectrum.CoupledMoments.ModularCoding
