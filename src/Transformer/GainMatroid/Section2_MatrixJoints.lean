module

public import Transformer.GainMatroid.Section2_MatrixContractions

/-!
# Lemma 2.5 for contractions of joints

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.3, Lemma 2.5 (ii) and (iii) for a joint `e` at
`v`, which is an unbalanced loop.

Frame type: scale row `v` so that column `e` is a unit vector, scale each column with a nonzero
entry in row `v` and a second nonzero entry so that the second entry is `1 - γ(e)`, as the source
prescribes, and remove row `v` and column `e`. The result is the canonical matrix of the contraction
without the row of `v`, which is zero there (`frameMatrix_frameContract_row`), since `v` carries
only balanced loops after the contraction. The source deletes such isolated vertices silently.

Lift type: scale the gains row so that column `e` is a unit vector and remove it. The result is the
canonical lift matrix of `G \ e` with identity gains, without its gains row, which is zero.

A balanced loop has a zero column, so no row operation makes it a unit vector. The source sets
`G / e = G \ e` for it, and that case is `frameMatrix_delete` and `liftMatrix_delete`.
-/

@[expose] public section

open Matrix

namespace Transformer.GainMatroid

variable {V E F : Type*} [DecidableEq V] [Field F]

namespace BiasedGraph

variable {Ω : BiasedGraph V E} {e : E} {v : V} {φ : E → Fˣ} {ι : ↥({e}ᶜ : Set E) → Fˣ}

/-- The row of `v` vanishes in the canonical frame matrix of a frame-type contraction at `v`. -/
theorem frameMatrix_frameContract_row (hι : Ω.IsFrameInducedGain e v φ ι)
    (f : ↥({e}ᶜ : Set E)) :
    frameMatrix ((Ω.graph.frameContract v).delete {e}) ι v f = 0 := by
  change (if v = (Ω.graph.frameContract v).tail f then (1 : F) else 0) -
    (ι f : F) * (if v = (Ω.graph.frameContract v).head f then 1 else 0) = 0
  rcases hι f with ⟨h₁, h₂, -⟩ | ⟨h₁, h₂, h₃⟩ | ⟨hl, h₁ | h₁, -⟩
  · simp [OrientedGraph.frameContract, h₁, h₂, Ne.symm h₁, Ne.symm h₂]
  · simp [OrientedGraph.frameContract, h₁, h₂, h₃]
  · have hw : v ≠ Ω.graph.head f := h₁ ▸ hl
    simp [OrientedGraph.frameContract, h₁, hw, hw.symm]
  · have hw : v ≠ Ω.graph.tail f := fun hc => hl (hc ▸ h₁.symm)
    simp [OrientedGraph.frameContract, h₁, hw, hw.symm]

/-- The scaling of one column in a frame-type contraction of the joint `e` at `v`. -/
private theorem exists_column_scale (c₀ : Fˣ) (hc₀ : (c₀ : F) = 1 - φ e)
    (hι : Ω.IsFrameInducedGain e v φ ι) (f : E) :
    ∃ c : Fˣ, ((frameMatrix Ω.graph φ v f = 0 ∨
        ∀ x, x ≠ v → frameMatrix Ω.graph φ x f = 0) → c = 1) ∧
      ∀ (hf : f ∈ ({e}ᶜ : Set E)) (x : V), x ≠ v →
        frameMatrix Ω.graph φ x f * c =
          frameMatrix ((Ω.graph.frameContract v).delete {e}) ι x ⟨f, hf⟩ := by
  by_cases hfe : f = e
  · exact ⟨1, fun _ => rfl, fun hf => absurd hfe hf⟩
  have hf : f ∈ ({e}ᶜ : Set E) := hfe
  have hB : ∀ x, frameMatrix ((Ω.graph.frameContract v).delete {e}) ι x ⟨f, hf⟩ =
      (if x = (Ω.graph.frameContract v).tail f then (1 : F) else 0) -
        (ι ⟨f, hf⟩ : F) * (if x = (Ω.graph.frameContract v).head f then 1 else 0) := fun _ => rfl
  rcases hι ⟨f, hf⟩ with ⟨h₁, h₂, h₃⟩ | ⟨h₁, h₂, h₃⟩ | ⟨hl, h₁ | h₁, h₃⟩
  · refine ⟨1, fun _ => rfl, fun hf' x _ => ?_⟩
    rw [hB, h₃]
    simp [frameMatrix_apply, OrientedGraph.frameContract, show Ω.graph.tail f ≠ v from h₁,
      show Ω.graph.head f ≠ v from h₂]
  · refine ⟨1, fun _ => rfl, fun hf' x hx => ?_⟩
    rw [hB, h₃]
    simp [frameMatrix_apply, OrientedGraph.frameContract, show Ω.graph.tail f = v from h₁,
      show Ω.graph.head f = v from h₂, hx]
  · have ht : Ω.graph.tail f = v := h₁
    have hw : Ω.graph.head f ≠ v := fun hc => hl (ht.trans hc.symm)
    refine ⟨-c₀ * (φ f)⁻¹, fun h => ?_, fun hf' x hx => ?_⟩
    · exfalso
      rcases h with h | h
      · simp [frameMatrix_apply, ht, Ne.symm hw] at h
      · have := h _ hw
        simp [frameMatrix_apply, ht, hw] at this
    · rw [hB, h₃]
      have hφ : (φ f : F) ≠ 0 := (φ f).ne_zero
      by_cases hxw : x = Ω.graph.head f
      · simp [frameMatrix_apply, OrientedGraph.frameContract, ht, hw, hxw, hc₀]
        field_simp
      · simp [frameMatrix_apply, OrientedGraph.frameContract, ht, hw, hxw, hx]
  · have hh : Ω.graph.head f = v := h₁
    have hw : Ω.graph.tail f ≠ v := fun hc => hl (hc.trans hh.symm)
    refine ⟨c₀, fun h => ?_, fun hf' x hx => ?_⟩
    · exfalso
      rcases h with h | h
      · simp [frameMatrix_apply, hh, Ne.symm hw] at h
      · have := h _ hw
        simp [frameMatrix_apply, hh, hw] at this
    · rw [hB, h₃]
      by_cases hxw : x = Ω.graph.tail f
      · simp [frameMatrix_apply, OrientedGraph.frameContract, hh, hw, hxw, hc₀]
      · simp [frameMatrix_apply, OrientedGraph.frameContract, hh, hw, hxw, hx]

/-- **Lemma 2.5 (ii)** for a joint, frame type. Up to the zero row of `v`, the canonical frame
matrix of the frame-type contraction is `A / e`, for `A` with exactly the columns scaled that the
source allows: those with a nonzero entry in row `v` and a second nonzero entry. -/
theorem frameMatrix_frameContract [Fintype V] (ht : Ω.graph.tail e = v) (hh : Ω.graph.head e = v)
    (he : φ e ≠ 1) (hι : Ω.IsFrameInducedGain e v φ ι) :
    ∃ s : E → Fˣ, (∀ f, (frameMatrix Ω.graph φ v f = 0 ∨
        ∀ x, x ≠ v → frameMatrix Ω.graph φ x f = 0) → s f = 1) ∧
      IsMatrixContraction (scaleCols s (frameMatrix Ω.graph φ)) e v
        (Subtype.val : {x // x ≠ v} → V)
        ((frameMatrix ((Ω.graph.frameContract v).delete {e}) ι).submatrix Subtype.val id) := by
  have hne : (1 : F) - φ e ≠ 0 := fun h => he (Units.ext (by simpa using (sub_eq_zero.1 h).symm))
  let c₀ : Fˣ := Units.mk0 _ hne
  have hc₀ : (c₀ : F) = 1 - φ e := rfl
  choose s hs₁ hs₂ using exists_column_scale (Ω := Ω) (e := e) (v := v) (φ := φ) (ι := ι) c₀ hc₀ hι
  have hcol : ∀ r, frameMatrix Ω.graph φ r e = if r = v then (c₀ : F) else 0 := fun r => by
    by_cases hr : r = v <;> simp [frameMatrix_apply, ht, hh, hr, hc₀]
  have hse : s e = 1 := hs₁ e (Or.inr fun x hx => by rw [hcol, ite_eq_right hx])
  refine ⟨s, hs₁, Subtype.val_injective, by ext y; simp,
    diagonal fun x => if x = v then ((c₀⁻¹ : Fˣ) : F) else 1, ?_, fun r => ?_, fun x f => ?_⟩
  · rw [Matrix.isUnit_iff_isUnit_det, det_diagonal, isUnit_iff_ne_zero]
    refine Finset.prod_ne_zero_iff.2 fun w _ => ?_
    by_cases hw : w = v
    · rw [ite_eq_left hw]
      exact Units.ne_zero _
    · rw [ite_eq_right hw]
      exact one_ne_zero
  · rw [diagonal_mul]
    change _ * (frameMatrix Ω.graph φ r e * (s e : F)) = _
    rw [hcol, hse, Units.val_one, mul_one]
    by_cases hr : r = v <;> simp [hr]
  · rw [diagonal_mul]
    change _ * (frameMatrix Ω.graph φ x f * (s f : F)) = _
    rw [ite_eq_right x.2, one_mul]
    exact hs₂ f f.2 x x.2

end BiasedGraph

/-- **Lemma 2.5 (iii)** for a joint, lift type: scale the gains row and remove it. The result is the
canonical lift matrix of `G \ e` with identity gains, without its gains row, which is zero. -/
theorem liftMatrix_liftContract [Fintype V] (G : OrientedGraph V E) {e : E}
    (hloop : G.tail e = G.head e) {ψ : E → F} (he : ψ e ≠ 0) :
    IsMatrixContraction (liftMatrix G ψ) e none some
      ((liftMatrix (G.delete {e}) (0 : ↥({e}ᶜ : Set E) → F)).submatrix some id) := by
  refine ⟨Option.some_injective _, by rw [← Set.compl_range_some, compl_compl],
    liftRowOp (ψ e)⁻¹ 0, isUnit_liftRowOp (inv_ne_zero he) 0, fun r => ?_, fun x f => ?_⟩
  · cases r with
    | none =>
      rw [liftRowOp_mul_none]
      simp [liftMatrix_none, he]
    | some w =>
      rw [liftRowOp_mul_some, liftMatrix_loop G ψ hloop]
      simp
  · rw [liftRowOp_mul_some]
    rfl

end Transformer.GainMatroid
