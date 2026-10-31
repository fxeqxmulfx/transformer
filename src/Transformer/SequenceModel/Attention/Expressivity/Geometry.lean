module

public import Transformer.SequenceModel.Attention.Expressivity.Normalization

/-!
# Ordered pair codes retained by XSA and a residual

Source: `review/flat_spectrum.md`, *Transformer learning bridges*, E3.
Two disjoint coordinate banks store the current token and the earlier token.
Equal tokens have no context component; the residual identifies that case.
All codes and requested residual displacements have fixed small norms.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Expressivity

open GPTMini
open scoped RealInnerProductSpace

/-- A unit token coordinate in the actual residual space. -/
def unit {d : ℕ} (i : Fin d) : EucSpace d := EuclideanSpace.single i 1

/-- E3: token basis vectors have exact norm one. -/
theorem norm_unit {d : ℕ} (i : Fin d) : ‖unit i‖ = 1 := by
  simp [unit, EuclideanSpace.single]

/-- E3: disjoint banks are exactly orthogonal. -/
theorem inner_unit {d : ℕ} (i j : Fin d) : ⟪unit i, unit j⟫ = if i = j then 1 else 0 := by
  simp [unit, EuclideanSpace.inner_single_left]

/-- Current-token residual plus the earlier-token workspace, unless tokens agree. -/
def pairCode {p d : ℕ} (e : (Fin p ⊕ Fin p) ↪ Fin d) (a b : Fin p) : EucSpace d :=
  unit (e (.inl b)) + if a = b then 0 else unit (e (.inr a))

/-- E3: the diagonal and off-diagonal codes have squared lengths one and two. -/
theorem pairCode_normSq {p d : ℕ} (e : (Fin p ⊕ Fin p) ↪ Fin d) (a b : Fin p) :
    ‖pairCode e a b‖ ^ 2 = if a = b then 1 else 2 := by
  by_cases h : a = b
  · simp [pairCode, h, norm_unit]
  · norm_num [pairCode, h, norm_add_sq_real, norm_unit, inner_unit, e.injective.eq_iff]

/-- E3: the desired answer displacement fits strictly in every width above
three; both equal and unequal operands, and repeated answer labels, are included. -/
theorem displacement_normSq_le {p d : ℕ} (e : (Fin p ⊕ Fin p) ↪ Fin d) (a b q : Fin p) :
    ‖unit (e (.inl q)) - pairCode e a b‖ ^ 2 ≤ 3 := by
  rw [norm_sub_sq_real, norm_unit, pairCode_normSq]
  simp only [pairCode, inner_add_right, inner_unit]
  by_cases hab : a = b <;> by_cases hqb : q = b <;>
    norm_num [hab, hqb, inner_unit, e.injective.eq_iff]

/-- E3: exact preimages of unit embeddings are the same positive scalar
multiple in every token coordinate, retaining a tied embedding/readout. -/
theorem unnormalize_unit {d : ℕ} (eps : ℝ) (i : Fin d) :
    unnormalize d eps (unit i) = Real.sqrt ((d : ℝ) * eps / ((d : ℝ) - 1)) • unit i := by
  simp only [unnormalize, norm_unit, one_pow]

/-- E3: the code has a unique current token in the first bank. -/
theorem pairCode_left {p d : ℕ} (e : (Fin p ⊕ Fin p) ↪ Fin d) (a b q : Fin p) :
    pairCode e a b (e (.inl q)) = if q = b then 1 else 0 := by
  by_cases hab : a = b <;>
    simp [pairCode, hab, unit, EuclideanSpace.single, e.injective.eq_iff]

/-- E3: the context bank identifies the earlier token on every unequal pair. -/
theorem pairCode_right {p d : ℕ} (e : (Fin p ⊕ Fin p) ↪ Fin d) (a b q : Fin p) :
    pairCode e a b (e (.inr q)) = if a = b then 0 else if q = a then 1 else 0 := by
  by_cases hab : a = b <;>
    simp [pairCode, hab, unit, EuclideanSpace.single, e.injective.eq_iff]

end Transformer.SequenceModel.Attention.Expressivity
