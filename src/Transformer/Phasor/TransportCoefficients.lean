module

public import Transformer.Phasor.SuccessorDisplacement

/-!
# Joint emission of transported masses and displaced embedding coefficients

Source: the transport-normalization timing trial in
`experiments/phasor_value_transport/README.md`. The normalization pass
already visits each vocabulary destination once. It can write the untouched
mass for the softmax derivative and the displaced value for the embedding
multiply at that destination during the same visit.
-/

@[expose] public section

namespace Transformer.Phasor

open scoped RealInnerProductSpace
open Transformer

variable {p D : ℕ}

/-- Two outputs written at destination `s + offset` when the unshifted
source `s` is normalized. A self read copies the mass to both outputs. -/
def transportCoefficients (mass : Fin p → ℝ) (offset key dest : Fin p) (strength : ℝ)
    (s : Fin p) : ℝ × ℝ :=
  (mass s, if key = dest then mass s
    else if s + offset = dest then mass s + strength
    else if s + offset = key then mass s - strength else mass s)

/-- At each output coordinate the two values equal the shifted normalized
mass and the coefficient from the separate displacement pass. -/
theorem transportCoefficients_at [NeZero p] (mass : Fin p → ℝ) (offset key dest c : Fin p)
    (strength : ℝ) :
    transportCoefficients mass offset key dest strength (c-offset) =
      (mass (c-offset), displacementWeight (fun v => mass (v-offset)) strength key dest c) := by
  by_cases self : key = dest
  · subst dest
    simp [transportCoefficients, displacementWeight]
  · by_cases hit : c = dest
    · subst c
      simp [transportCoefficients, displacementWeight, self, Ne.symm self]
    · by_cases leave : c = key <;> simp [transportCoefficients, displacementWeight, self, hit, leave]

/-- The embedding multiply sees exactly the transported vocabulary read
plus the signed difference of tied embeddings. The retained mass can be
arbitrary and the displacement strength can be negative. -/
theorem transportCoefficients_read [NeZero p] (E : Fin p → EucSpace D) (mass : Fin p → ℝ)
    (offset key dest : Fin p) (strength : ℝ) :
    ∑ s, (transportCoefficients mass offset key dest strength s).2 • E (s+offset) =
      transportedValue E mass offset + strength • (E dest - E key) := by
  let perm := Equiv.addRight offset
  let q := fun c => mass (c-offset)
  have moved := Equiv.sum_comp perm (fun c => displacementWeight q strength key dest c • E c)
  have ordinary := Equiv.sum_comp perm (fun c => q c • E c)
  have coeff (s : Fin p) : (transportCoefficients mass offset key dest strength s).2 =
      displacementWeight q strength key dest (s+offset) := by
    simpa [q] using congrArg Prod.snd (transportCoefficients_at mass offset key dest (s+offset) strength)
  change (∑ s, (transportCoefficients mass offset key dest strength s).2 • E (perm s)) = _
  simp_rw [coeff]
  change (∑ s, displacementWeight q strength key dest (perm s) • E (perm s)) = _
  rw [moved, displacementWeight_read, ← ordinary]
  simp [perm, q, transportedValue]

/-- For fixed input addresses the displaced coefficient has the same
finite variation in its normalized mass as the saved ordinary coefficient.
Thus its score derivative uses the ordinary softmax masses. -/
theorem transportCoefficients_mass_change (mass delta : Fin p → ℝ) (offset key dest s : Fin p)
    (strength : ℝ) :
    (transportCoefficients (mass + delta) offset key dest strength s).2 -
      (transportCoefficients mass offset key dest strength s).2 = delta s := by
  simp only [transportCoefficients, Pi.add_apply]
  split_ifs <;> ring

/-- The last source coordinate wraps through zero, and a negative
displacement produces a signed effective coefficient. -/
example : transportCoefficients (fun _ => 1/4) 2 0 1 (-1) (4 : Fin 5) = (1/4, -3/4) := by
  norm_num [transportCoefficients]

example (mass delta : Fin 3 → ℝ) :
    (transportCoefficients (mass + delta) 1 0 0 2 2).2 -
      (transportCoefficients mass 1 0 0 2 2).2 = delta 2 :=
  transportCoefficients_mass_change _ _ _ _ _ _ _

end Transformer.Phasor
