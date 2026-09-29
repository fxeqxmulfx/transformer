/-
# Zero padding and cropping commute with row-major layout

Arora et al., arXiv:2312.04927v1, Appendix `def: kaleidoscope`.
The scalar upper-left submatrix S E Sᵀ becomes row padding and cropping
in the original n × d layout. Feature width remains d throughout.
-/

import Transformer.Zoology.Appendix_KaleidoscopeCast

namespace Transformer.Zoology

/-- A scalar row-major index is inside the original vector exactly when
its row is inside the original sequence. Source: Appendix `def: kaleidoscope`. -/
theorem butterflyGrid_index_bound (p k : ℕ) (i : ℕ)
    (q : Fin (butterflyWidth k)) :
    i * butterflyWidth k + q.val < butterflyWidth (k + p) ↔
      i < butterflyWidth p := by
  rw [butterflyGrid_width]
  constructor
  · intro h
    by_contra hi
    have hmul := Nat.mul_le_mul_right (butterflyWidth k) (Nat.le_of_not_gt hi)
    omega
  · intro hi
    calc
      i * butterflyWidth k + q.val < i * butterflyWidth k + butterflyWidth k :=
        Nat.add_lt_add_left q.isLt _
      _ = (i + 1) * butterflyWidth k := by ring
      _ ≤ butterflyWidth p * butterflyWidth k :=
        Nat.mul_le_mul_right _ hi

/-- Padding by complete feature rows is exactly the original scalar
zero embedding Sᵀ. Source: Appendix `def: kaleidoscope`, power-of-two expansion. -/
theorem butterflyGrid_pad_decode {p k e : ℕ}
    (v : Fin (butterflyWidth (k + p)) → ℝ) :
    (padSequence (butterflyGridDecode (p := p) (k := k) v) :
      RealSequence (butterflyWidth (p + e)) (butterflyWidth k)) =
      butterflyGridDecode (p := p + e) (k := k)
        (fun j => padButterfly (e := e) v
          (Fin.cast (butterflyGridExpandedWidth p k e).symm j)) := by
  funext i q
  let j : Fin (butterflyWidth ((k + p) + e)) :=
    Fin.cast (butterflyGridExpandedWidth p k e).symm
      (butterflyGridFlatten (p + e) k i q)
  have hj : j.val = i.val * butterflyWidth k + q.val := by
    exact butterflyGridFlatten_val (p + e) k i q
  simp only [padSequence, dite_eq_left q.isLt]
  change (if hi : i.val < butterflyWidth p then
    butterflyGridDecode v ⟨i.val, hi⟩ q else 0) = padButterfly v j
  by_cases hi : i.val < butterflyWidth p
  · have hflat : j.val < butterflyWidth (k + p) := by
      rw [hj, butterflyGrid_index_bound]
      exact hi
    rw [dite_eq_left hi]
    change v (butterflyGridFlatten p k ⟨i.val, hi⟩ q) =
      if h : j.val < butterflyWidth (k + p) then v ⟨j.val, h⟩ else 0
    rw [dite_eq_left hflat]
    apply congrArg v
    apply Fin.ext
    change (butterflyGridFlatten p k ⟨i.val, hi⟩ q).val = j.val
    rw [butterflyGridFlatten_val, hj]
  · have hflat : ¬ j.val < butterflyWidth (k + p) := by
      rw [hj, butterflyGrid_index_bound]
      exact hi
    rw [dite_eq_right hi]
    change 0 = if h : j.val < butterflyWidth (k + p) then v ⟨j.val, h⟩ else 0
    rw [dite_eq_right hflat]

/-- Cropping complete feature rows is exactly the original scalar
coordinate extraction S. Source: Appendix `def: kaleidoscope`. -/
theorem butterflyGrid_crop_decode {p k e : ℕ}
    (v : Fin (butterflyWidth ((k + p) + e)) → ℝ) :
    cropSequence (butterflyWidth_le_expanded p e) (le_refl _)
      (butterflyGridDecode (p := p + e) (k := k)
        (fun j => v (Fin.cast (butterflyGridExpandedWidth p k e).symm j))) =
      butterflyGridDecode (p := p) (k := k) (cropButterfly (e := e) v) := by
  funext i q
  change v (Fin.cast (butterflyGridExpandedWidth p k e).symm
    (butterflyGridFlatten (p + e) k
      ⟨i.val, lt_of_lt_of_le i.isLt (butterflyWidth_le_expanded p e)⟩ q)) =
    v ⟨(butterflyGridFlatten p k i q).val,
      lt_of_lt_of_le (butterflyGridFlatten p k i q).isLt
        (butterflyWidth_le_expanded (k + p) e)⟩
  apply congrArg v
  apply Fin.ext
  change (butterflyGridFlatten (p + e) k
      ⟨i.val, lt_of_lt_of_le i.isLt (butterflyWidth_le_expanded p e)⟩ q).val =
    (butterflyGridFlatten p k i q).val
  rw [butterflyGridFlatten_val, butterflyGridFlatten_val]

end Transformer.Zoology
