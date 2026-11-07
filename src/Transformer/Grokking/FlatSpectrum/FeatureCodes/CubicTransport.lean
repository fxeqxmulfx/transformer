module

public import Transformer.Grokking.FlatSpectrum.FeatureCodes.CubicQuotient

/-!
# Transfer an observed cubic label along an explicit input route

Source: `review/flat_spectrum.md`, *Training-state transport*, TT4.
Translations `(a,b) -> (a+t,b-t)` preserve the sum. Simultaneous multiplication
by 1, 35 or 61 preserves its cube modulo 97. Thus a training anchor supplies
the answer for a query connected by this checked route; no query label is
needed to construct the route. This is task symmetry, not model equivariance.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.FeatureCodes

/-- Translation followed by an allowed multiplier preserves the cubic task,
TT4. The parameters describe an exact route between raw operand pairs. -/
theorem cubic97_route (a b t u : ZMod 97) (hu : u = 1 ∨ u = 35 ∨ u = 61) :
    ((u * a + t) + (u * b - t)) ^ 3 = (a + b) ^ 3 := by
  have hs : (u * a + t) + (u * b - t) = u * (a + b) := by ring
  have hc : u ^ 3 = 1 := by rcases hu with rfl | rfl | rfl <;> decide
  rw [hs, mul_pow, hc, one_mul]

/-- A sum-multiplier check produces the full raw-input route, TT4. -/
theorem cubic97_route_of_sum (a b c d u : ZMod 97)
    (hs : c + d = u * (a + b)) :
    ∃ t, c = u * a + t ∧ d = u * b - t := by
  refine ⟨c - u * a, by ring, ?_⟩
  calc
    d = u * (a + b) - c := by linear_combination hs
    _ = u * b - (c - u * a) := by ring

/-- A checked route transports a training answer to the query, TT4. -/
theorem cubic97_transfer_label (a b c d u label : ZMod 97)
    (hu : u = 1 ∨ u = 35 ∨ u = 61) (hs : c + d = u * (a + b))
    (hlabel : (a + b) ^ 3 = label) : (c + d) ^ 3 = label := by
  obtain ⟨t, hc, hd⟩ := cubic97_route_of_sum a b c d u hs
  rw [hc, hd, cubic97_route a b t u hu, hlabel]

example := cubic97_route 1 2 7 35 (Or.inr (Or.inl rfl))
example := cubic97_route_of_sum 1 2 42 63 35 (by decide)
example := cubic97_transfer_label 1 2 42 63 35 27
  (Or.inr (Or.inl rfl)) (by decide) (by decide)
example := cubic97_transfer_label 1 96 7 90 1 0
  (Or.inl rfl) (by decide) (by decide)

end Transformer.Grokking.FlatSpectrum.FeatureCodes
