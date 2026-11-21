module

public import Mathlib.Tactic.Linarith
public import Transformer.GainMatroid.Section2_MatrixMinors
public import Transformer.GainMatroid.Section4_Relations

/-!
# Lemma 2.5 for contractions of links

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.3, Lemma 2.5 (ii) and (iii) for a link `e`.

For lift matrices the printed statement holds for every induced gain function, because switching
is a row operation (`liftMatrix_contract_of_isAddInducedGain`).

For frame matrices it does not. The source defines `A / e` for a link by row operations alone and
states `A_F((G, φ) / e) = A_F(G, φ) / e`, but switching a frame matrix also scales its columns.
`frameMatrix_contract_fails_without_scaling` is a balanced triangle with `φ(e) = 2` on which no row
operations give the canonical matrix of any gain function on `G / e`. The statement holds as printed
when `φ(e) = 1` (`frameMatrix_contract_of_eq_one`), and for every induced gain function after a
scaling of the columns (`frameMatrix_contract_of_isInducedGain`).
-/

@[expose] public section

open Matrix

namespace Transformer.GainMatroid

variable {V E F R R' : Type*} [DecidableEq V] [Field F]

/-- Row operations applied beforehand do not change what a matrix contracts to. -/
theorem IsMatrixContraction.of_mul_left [Fintype R] [DecidableEq R] {A : Matrix R E F}
    {P : Matrix R R F} {e : E} {i : R} {σ : R' → R} {B : Matrix R' ↥({e}ᶜ : Set E) F}
    (hP : IsUnit P) (h : IsMatrixContraction (P * A) e i σ B) : IsMatrixContraction A e i σ B := by
  obtain ⟨T, hT, h₁, h₂⟩ := h.exists_rowOp
  refine ⟨h.injective, h.range_eq, T * P, hT.mul hP, fun r => ?_, fun r' f => ?_⟩
  · rw [Matrix.mul_assoc]
    exact h₁ r
  · rw [Matrix.mul_assoc]
    exact h₂ r' f

namespace OrientedGraph

variable (G : OrientedGraph V E) {e : E}

/-- **Lemma 2.5 (ii)** for a link with identity gain: add the row of `tail e` to the row of
`head e`, and remove the row of `tail e` and column `e`. -/
theorem frameMatrix_contract_of_eq_one [Fintype V]
    [DecidableEq (Quotient (G.reachableSetoid {e}))] (he : G.IsLink e) {φ : E → Fˣ} (h1 : φ e = 1) :
    IsMatrixContraction (frameMatrix G φ) e (G.tail e) (G.contractRow e)
      (frameMatrix (G.contract {e}) fun f => φ f) := by
  refine ⟨G.contractRow_injective e, G.range_contractRow he,
    transvection (G.head e) (G.tail e) 1, ?_, fun r => ?_, ?_⟩
  · rw [Matrix.isUnit_iff_isUnit_det, det_transvection_of_ne _ _ he.symm]
    exact isUnit_one
  · rw [transvection_one_mul_apply]
    simp only [frameMatrix_apply, h1, Units.val_one, one_mul]
    by_cases hr : r = G.head e <;> simp [hr, he, he.symm]
  · rintro ⟨x⟩ f
    change (transvection (G.head e) (G.tail e) (1 : F) * frameMatrix G φ)
        (G.contractVertex e x) f =
      (if Quotient.mk _ x = Quotient.mk (G.reachableSetoid {e}) (G.tail f) then (1 : F) else 0) -
        (φ f : F) *
          if Quotient.mk _ x = Quotient.mk (G.reachableSetoid {e}) (G.head f) then 1 else 0
    rw [G.ite_mk_eq, G.ite_mk_eq, G.incidence_contractVertex (G.contractVertex_ne he x),
      transvection_one_mul_apply]
    simp only [frameMatrix_apply]

/-- **Lemma 2.5 (iii)** for a link with zero gain: the same row operation, off the gains row. -/
theorem liftMatrix_contract_of_eq_zero [Fintype V]
    [DecidableEq (Quotient (G.reachableSetoid {e}))] (he : G.IsLink e) {ψ : E → F} (h0 : ψ e = 0) :
    IsMatrixContraction (liftMatrix G ψ) e (some (G.tail e)) (Option.map (G.contractRow e))
      (liftMatrix (G.contract {e}) fun f => ψ f) := by
  refine ⟨Option.map_injective (G.contractRow_injective e),
    range_optionMap (G.range_contractRow he),
    transvection (some (G.head e)) (some (G.tail e)) 1, ?_, fun r => ?_, ?_⟩
  · rw [Matrix.isUnit_iff_isUnit_det,
      det_transvection_of_ne _ _ fun h => he (Option.some_inj.1 h).symm]
    exact isUnit_one
  · rw [transvection_one_mul_apply]
    cases r with
    | none => simp [liftMatrix_none, h0]
    | some w =>
      simp only [liftMatrix_some, Option.some.injEq]
      by_cases hw : w = G.head e <;> simp [hw, he, he.symm]
  · rintro (_ | q) f
    · rw [transvection_one_mul_apply]
      simp [liftMatrix_none]
    · obtain ⟨x, rfl⟩ := Quotient.exists_rep q
      change (transvection (some (G.head e)) (some (G.tail e)) (1 : F) * liftMatrix G ψ)
          (some (G.contractVertex e x)) f =
        (if Quotient.mk _ x = Quotient.mk (G.reachableSetoid {e}) (G.tail f) then (1 : F) else 0) -
          if Quotient.mk _ x = Quotient.mk (G.reachableSetoid {e}) (G.head f) then 1 else 0
      have h := G.incidence_contractVertex (G.contractVertex_ne he x) (1 : F) (G.tail f) (G.head f)
      simp only [one_mul] at h
      rw [G.ite_mk_eq, G.ite_mk_eq, h, transvection_one_mul_apply]
      simp only [liftMatrix_some, Option.some.injEq]

/-- **Lemma 2.5 (ii)** for a link, corrected. For every gain function induced on `G / e`, the
canonical matrix is `A / e` for `A` the canonical frame matrix with its columns scaled. The source
writes `A_F((G, φ) / e) = A_F(G, φ) / e` with no scaling, which
`frameMatrix_contract_fails_without_scaling` refutes. -/
theorem frameMatrix_contract_of_isInducedGain [Fintype V] [Fintype E] [DecidableEq E]
    [DecidableEq (Quotient (G.reachableSetoid {e}))] (he : G.IsLink e) {φ : E → Fˣ}
    {ι : ↥({e}ᶜ : Set E) → Fˣ} (hι : G.IsInducedGain {e} φ ι) :
    ∃ s : E → Fˣ, IsMatrixContraction (scaleCols s (frameMatrix G φ)) e (G.tail e)
      (G.contractRow e) (frameMatrix (G.contract {e}) ι) := by
  obtain ⟨η, h1, hι⟩ := hι
  obtain rfl : ι = fun f : ↥({e}ᶜ : Set E) => switch G η φ f := funext hι
  refine ⟨fun f => (η (G.tail f))⁻¹,
    IsMatrixContraction.of_mul_left (P := diagonal fun w => (η w : F)) ?_ ?_⟩
  · rw [Matrix.isUnit_iff_isUnit_det, det_diagonal, isUnit_iff_ne_zero]
    exact Finset.prod_ne_zero_iff.2 fun w _ => (η w).ne_zero
  · have hM : diagonal (fun w => (η w : F)) *
        scaleCols (fun f => (η (G.tail f))⁻¹) (frameMatrix G φ) =
          frameMatrix G (switch G η φ) := by
      rw [scaleCols_eq_mul_diagonal, ← Matrix.mul_assoc]
      exact diagonal_mul_frameMatrix_mul_diagonal G η φ
    rw [hM]
    exact G.frameMatrix_contract_of_eq_one he (h1 e rfl)

/-- **Lemma 2.5 (iii)** for a link: for every gain function induced on `G / e`, its canonical lift
matrix is `A_L(G, ψ) / e`, with the removed row other than the gains row. -/
theorem liftMatrix_contract_of_isAddInducedGain [Fintype V]
    [DecidableEq (Quotient (G.reachableSetoid {e}))] (he : G.IsLink e) {ψ : E → F}
    {ι : ↥({e}ᶜ : Set E) → F} (hι : G.IsAddInducedGain {e} ψ ι) :
    IsMatrixContraction (liftMatrix G ψ) e (some (G.tail e)) (Option.map (G.contractRow e))
      (liftMatrix (G.contract {e}) ι) := by
  obtain ⟨η, h0, hι⟩ := hι
  obtain rfl : ι = fun f : ↥({e}ᶜ : Set E) => addSwitch G η ψ f := funext hι
  have hM : liftRowOp ((1 : Fˣ) : F) (fun v => -(((1 : Fˣ) : F) * η v)) * liftMatrix G ψ =
      liftMatrix G (addSwitch G η ψ) := by
    rw [liftRowOp_mul_liftMatrix, Units.val_one, one_smul]
  refine IsMatrixContraction.of_mul_left
    (P := liftRowOp ((1 : Fˣ) : F) fun v => -(((1 : Fˣ) : F) * η v))
    (isUnit_liftRowOp (Units.ne_zero 1) _) ?_
  rw [hM]
  exact G.liftMatrix_contract_of_eq_zero he (h0 e rfl)

end OrientedGraph

open OrientedGraph

/-- A triangle for the failure of Lemma 2.5 (ii): `0 : 0 → 1`, `1 : 0 → 2` and `2 : 1 → 2`. -/
def contractTriangle : OrientedGraph (Fin 3) (Fin 3) := ⟨![0, 0, 1], ![1, 2, 2]⟩

/-- A balanced gain function on `contractTriangle` with gain `2` on the edge to contract. -/
noncomputable def contractTriangleGain : Fin 3 → ℚˣ := ![ratTwo, ratTwo, 1]

/-- **Lemma 2.5 (ii) fails as printed** for a link whose gain is not the identity: on a balanced
triangle with `φ(e) = 2`, no row operations turn `A_F(G, φ)` into the canonical matrix of any gain
function on `G / e`. In the row of the merged vertex both remaining columns must be `1`, while any
row combination vanishing on column `e` has its first entry twice its second. -/
theorem frameMatrix_contract_fails_without_scaling
    [DecidableEq (Quotient (contractTriangle.reachableSetoid {(0 : Fin 3)}))]
    (ι : ↥({(0 : Fin 3)}ᶜ : Set (Fin 3)) → ℚˣ) (i : Fin 3)
    (σ : Quotient (contractTriangle.reachableSetoid {(0 : Fin 3)}) → Fin 3) :
    ¬IsMatrixContraction (frameMatrix contractTriangle contractTriangleGain) 0 i σ
      (frameMatrix (contractTriangle.contract {0}) ι) := by
  intro h
  obtain ⟨T, -, h₁, h₂⟩ := h.exists_rowOp
  let x := Quotient.mk (contractTriangle.reachableSetoid {(0 : Fin 3)}) (0 : Fin 3)
  have hx : σ x ≠ i := by
    have hmem : σ x ∈ ({i}ᶜ : Set (Fin 3)) := h.range_eq ▸ Set.mem_range_self x
    exact hmem
  have h01 : x = Quotient.mk (contractTriangle.reachableSetoid {(0 : Fin 3)}) (1 : Fin 3) :=
    contractTriangle.contract_class_eq (F := {(0 : Fin 3)}) (f := 0) rfl
  have h02 : x ≠ Quotient.mk (contractTriangle.reachableSetoid {(0 : Fin 3)}) (2 : Fin 3) := by
    intro hc
    have := contractTriangle.mk_eq_mk_iff_contractVertex.1 hc
    simp [contractVertex, contractTriangle] at this
  have he := h₁ (σ x)
  rw [ite_eq_right hx, mul_frameMatrix_apply] at he
  have hg := h₂ x ⟨1, by simp⟩
  have hf := h₂ x ⟨2, by simp⟩
  rw [mul_frameMatrix_apply] at hg hf
  have hgB : frameMatrix (contractTriangle.contract {0}) ι x ⟨1, by simp⟩ = 1 := by
    change (if x = x then (1 : ℚ) else 0) - _ * (if x = Quotient.mk _ (2 : Fin 3) then 1 else 0) = 1
    simp [h02]
  have hfB : frameMatrix (contractTriangle.contract {0}) ι x ⟨2, by simp⟩ = 1 := by
    change (if x = Quotient.mk _ (1 : Fin 3) then (1 : ℚ) else 0) -
      _ * (if x = Quotient.mk _ (2 : Fin 3) then 1 else 0) = 1
    simp [h02, ← h01]
  rw [hgB] at hg
  rw [hfB] at hf
  simp [contractTriangle, contractTriangleGain, ratTwo] at he hg hf
  linarith

end Transformer.GainMatroid
