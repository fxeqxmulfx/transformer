module

public import Transformer.GainMatroid.Section2_MatrixJoints

/-!
# Projective equivalence passes to minors of a matrix

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.3, the sentence after Lemma 2.5: projectively
equivalent matrices have projectively equivalent deletions and contractions.

The row space of `A / e` is the part of the row space of `A` that vanishes at `e`, read on the other
coordinates (`IsMatrixContraction.rowSpace_eq`). So it does not depend on the row operations or on
the row removed, and a scaling of columns carries one contraction to the other.

The examples at the end meet the hypotheses of the theorems of Lemma 2.5.
-/

@[expose] public section

open Matrix

namespace Transformer.GainMatroid

open OrientedGraph

variable {E F R R' S S' : Type*} [Field F]

/-- The row space of a matrix with its columns reindexed. -/
theorem rowSpace_submatrix (M : Matrix R E F) {E' : Type*} (g : E' → E) :
    rowSpace (M.submatrix id g) = (rowSpace M).map (LinearMap.funLeft F F g) := by
  unfold rowSpace
  rw [Submodule.map_span, ← Set.range_comp]
  rfl

/-- §2.3, after Lemma 2.5: projectively equivalent matrices have projectively equivalent
deletions. -/
theorem ProjEquiv.delete {A : Matrix R E F} {B : Matrix R' E F} (h : ProjEquiv A B) (D : Set E) :
    ProjEquiv (A.submatrix id (Subtype.val : ↥Dᶜ → E)) (B.submatrix id Subtype.val) := by
  obtain ⟨s, hs⟩ := h
  refine ⟨fun f => s f, ?_⟩
  have hsub : scaleCols (fun f : ↥Dᶜ => s f) (A.submatrix id Subtype.val) =
      (scaleCols s A).submatrix id Subtype.val := rfl
  rw [hsub, rowSpace_submatrix, hs, rowSpace_submatrix]

/-- The row space of `A / e`: the vectors of the row space of `A` vanishing at `e`, without their
coordinate `e`. -/
theorem IsMatrixContraction.rowSpace_eq [Fintype R] [DecidableEq R] {A : Matrix R E F} {e : E}
    {i : R} {σ : R' → R} {B : Matrix R' ↥({e}ᶜ : Set E) F} (h : IsMatrixContraction A e i σ B) :
    rowSpace B = (rowSpace A ⊓ LinearMap.ker (LinearMap.proj e : (E → F) →ₗ[F] F)).map
      (LinearMap.funLeft F F (Subtype.val : ↥({e}ᶜ : Set E) → E)) := by
  obtain ⟨T, hT, h₁, h₂⟩ := h.exists_rowOp
  rw [← rowSpace_mul_of_isUnit hT A]
  refine le_antisymm (Submodule.span_le.2 ?_) ?_
  · rintro _ ⟨r', rfl⟩
    have hne : σ r' ≠ i := by
      have hmem : σ r' ∈ ({i}ᶜ : Set R) := h.range_eq ▸ Set.mem_range_self r'
      exact hmem
    refine ⟨(T * A) (σ r'), ⟨row_mem_rowSpace _ _, ?_⟩, ?_⟩
    · have h0 := h₁ (σ r')
      rw [ite_eq_right hne] at h0
      exact h0
    · ext f
      exact h₂ r' f
  · rintro _ ⟨x, ⟨hx, hxe⟩, rfl⟩
    obtain ⟨c, rfl⟩ := (Submodule.mem_span_range_iff_exists_fun F).1 hx
    have hci : c i = 0 := by
      have he : ∑ r, c r * (T * A) r e = 0 := by
        simpa [Finset.sum_apply, Matrix.row] using hxe
      simpa [h₁] using he
    rw [map_sum]
    refine Submodule.sum_mem _ fun r _ => ?_
    by_cases hr : r = i
    · subst hr
      simp [hci]
    · obtain ⟨r', rfl⟩ : r ∈ Set.range σ := h.range_eq ▸ hr
      rw [map_smul]
      refine Submodule.smul_mem _ _ (Submodule.subset_span ⟨r', ?_⟩)
      ext f
      exact (h₂ r' f).symm

/-- §2.3, after Lemma 2.5: projectively equivalent matrices have projectively equivalent
contractions, whatever row operations and removed rows are used. -/
theorem IsMatrixContraction.projEquiv [Fintype R] [DecidableEq R] [Fintype S] [DecidableEq S]
    {A : Matrix R E F} {C : Matrix S E F} {e : E} {i : R} {j : S} {σ : R' → R} {τ : S' → S}
    {A' : Matrix R' ↥({e}ᶜ : Set E) F} {C' : Matrix S' ↥({e}ᶜ : Set E) F}
    (hA : IsMatrixContraction A e i σ A') (hAC : ProjEquiv A C)
    (hC : IsMatrixContraction C e j τ C') : ProjEquiv A' C' := by
  obtain ⟨s, hs⟩ := hAC
  refine ⟨fun f => s f, ?_⟩
  rw [rowSpace_scaleCols, hA.rowSpace_eq, hC.rowSpace_eq, ← hs, rowSpace_scaleCols]
  ext y
  simp only [Submodule.mem_map, Submodule.mem_inf, LinearMap.mem_ker, LinearMap.proj_apply]
  constructor
  · rintro ⟨_, ⟨x, ⟨hx, hxe⟩, rfl⟩, rfl⟩
    refine ⟨scaleEquiv s x, ⟨⟨x, hx, rfl⟩, ?_⟩, rfl⟩
    change x e * (s e : F) = 0
    rw [hxe, zero_mul]
  · rintro ⟨_, ⟨⟨x, hx, rfl⟩, hxe⟩, rfl⟩
    have hxe' : x e * (s e : F) = 0 := hxe
    refine ⟨_, ⟨x, ⟨hx, (mul_eq_zero.1 hxe').resolve_right (s e).ne_zero⟩, rfl⟩, rfl⟩

/-- A single link is a forest, so every link has an induced gain function. -/
theorem OrientedGraph.isForest_singleton {V : Type*} {G : OrientedGraph V E} {e : E}
    (he : G.IsLink e) : G.IsForest {e} := by
  rintro C ⟨u, W, hW, rfl⟩ hsub
  obtain ⟨x, hx⟩ := List.exists_mem_of_ne_nil _ hW.edges_ne_nil
  obtain rfl : x = e := hsub hx
  exact he (hW.eq_of_forall_ne hx id fun f hf hfe => absurd (hsub hf) hfe)

theorem contractTriangle_isLink : contractTriangle.IsLink 0 := by
  simp [contractTriangle]

/-- A joint `0` at the vertex `0`, with two links to the vertex `1`, for the examples. -/
def jointGraph : OrientedGraph (Fin 2) (Fin 3) := ⟨![0, 0, 1], ![0, 1, 0]⟩

theorem jointGraph_isFrameInducedGain :
    jointGraph.contrabalanced.IsFrameInducedGain 0 0 (fun _ : Fin 3 => ratTwo) fun _ => ratTwo := by
  have h : ∀ f : Fin 3, f ≠ 0 → jointGraph.tail f ≠ jointGraph.head f ∧
      (jointGraph.tail f = 0 ∨ jointGraph.head f = 0) := by
    simp only [jointGraph]
    decide
  rintro ⟨f, hf⟩
  exact Or.inr (Or.inr ⟨(h f hf).1, (h f hf).2, rfl⟩)

example := linkGraph.contractVertex_ne (e := ()) linkGraph_isLink 2

example := linkGraph.range_contractRow (e := ()) linkGraph_isLink

example := linkGraph.incidence_contractVertex (e := ()) (y := 1) (by decide) (3 : ℚ) 0 1

example := range_optionMap (linkGraph.range_contractRow (e := ()) linkGraph_isLink)

example := isForest_singleton linkGraph_isLink

open Classical in
example := linkGraph.frameMatrix_contract_of_eq_one linkGraph_isLink (φ := (1 : Unit → ℚˣ)) rfl

open Classical in
example := linkGraph.liftMatrix_contract_of_eq_zero linkGraph_isLink (ψ := (0 : Unit → ℚ)) rfl

open Classical in
example := contractTriangle.frameMatrix_contract_of_isInducedGain contractTriangle_isLink
  (exists_isInducedGain (isForest_singleton contractTriangle_isLink)
    contractTriangleGain).choose_spec

open Classical in
example := contractTriangle.liftMatrix_contract_of_isAddInducedGain contractTriangle_isLink
  (exists_isAddInducedGain (isForest_singleton contractTriangle_isLink)
    (![1, 2, 3] : Fin 3 → ℚ)).choose_spec

open Classical in
example : IsMatrixContraction (frameMatrix linkGraph (1 : Unit → ℚˣ)) () (linkGraph.tail ())
    (linkGraph.contractRow ()) (frameMatrix (linkGraph.contract {()}) fun f => (1 : Unit → ℚˣ) f) :=
  IsMatrixContraction.of_mul_left (P := 1) isUnit_one (by
    rw [Matrix.one_mul]
    exact linkGraph.frameMatrix_contract_of_eq_one linkGraph_isLink rfl)

example := BiasedGraph.frameMatrix_frameContract_row jointGraph_isFrameInducedGain ⟨1, by simp⟩

example := BiasedGraph.frameMatrix_frameContract (Ω := jointGraph.contrabalanced) (e := 0) (v := 0)
  rfl rfl ratTwo_ne_one jointGraph_isFrameInducedGain

example := liftMatrix_liftContract jointGraph (e := 0) rfl (ψ := fun _ => (1 : ℚ)) one_ne_zero

example := (ProjEquiv.refl (frameMatrix contractTriangle contractTriangleGain)).delete {0}

open Classical in
example := (contractTriangle.frameMatrix_contract_of_eq_one contractTriangle_isLink
  (φ := ![1, ratTwo, 1]) rfl).rowSpace_eq

open Classical in
/-- Two lift matrices related by a scaling of the gains contract to equivalent matrices. -/
example := (contractTriangle.liftMatrix_contract_of_eq_zero contractTriangle_isLink
    (ψ := (![0, 1, 2] : Fin 3 → ℚ)) rfl).projEquiv
  (projEquiv_liftMatrix_of_switchScaleEquiv contractTriangle
    ⟨0, Units.mk0 2 two_ne_zero, by
      ext f
      fin_cases f <;> simp [addSwitch_apply]
      norm_num⟩)
  (contractTriangle.liftMatrix_contract_of_eq_zero contractTriangle_isLink
    (ψ := (![0, 2, 4] : Fin 3 → ℚ)) rfl)

end Transformer.GainMatroid
