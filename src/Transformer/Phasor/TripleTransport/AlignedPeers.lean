module

public import Transformer.Phasor.TripleTransport

/-!
# Transport through aligned output pairs

Source: “Aligned peer reads” in the Phasor triple-transport experiment.
The partial maps record matches between learned vectors. Shared keys
in two contexts relate ordered output pairs, including repeated images.
-/

@[expose] public section

namespace Transformer.Phasor.TripleTransport

variable {I X V T : Type*}

/-- Two observations in the same two context rows relate pairs of output slots.
Row and key slots have independent roles and row maps may repeat a value. -/
def AlignedEdge (P : I → X → Option V) (a b : V × V) : Prop :=
  ∃ i j x y, i ≠ j ∧ x ≠ y ∧
    P i x = some a.1 ∧ P j x = some a.2 ∧ P i y = some b.1 ∧ P j y = some b.2

/-- Every aligned component preserves a row-pair invariant of a total map
when the partial observations agree with the total map. -/
theorem aligned_component_preserves (P : I → X → Option V) (A : I → X → V)
    (truth : ∀ i x v, P i x = some v → A i x = v) (F : V × V → T)
    (invariant : ∀ i j x y, F (A i x, A j x) = F (A i y, A j y))
    {a b : V × V} (connected : Relation.EqvGen (AlignedEdge P) a b) : F a = F b := by
  induction connected with
  | rel a b edge =>
    obtain ⟨i, j, x, y, _, _, h₁, h₂, h₃, h₄⟩ := edge
    simpa [truth i x _ h₁, truth j x _ h₂, truth i y _ h₃, truth j y _ h₄]
      using invariant i j x y
  | refl => rfl
  | symm _ _ _ ih => exact ih.symm
  | trans _ _ _ _ _ ih₁ ih₂ => exact ih₁.trans ih₂

/-- Under a separating row-pair invariant, the output of a peer context at
the query and any connected candidate determine the true target. No output
at the target query is supplied. -/
theorem aligned_read_sound (P : I → X → Option V) (A : I → X → V)
    (truth : ∀ i x v, P i x = some v → A i x = v) (F : V × V → T)
    (invariant : ∀ i j x y, F (A i x, A j x) = F (A i y, A j y))
    (separates : ∀ v, Function.Injective (fun w => F (v, w)))
    (i j : I) (anchor query : X) (u v q y : V)
    (left : P i anchor = some u) (right : P j anchor = some v)
    (peer : P i query = some q)
    (path : Relation.EqvGen (AlignedEdge P) (u, v) (q, y)) : y = A j query := by
  apply separates q
  have observed := aligned_component_preserves P A truth F invariant path
  have same := invariant i j anchor query
  rw [truth i anchor _ left, truth j anchor _ right, truth i query _ peer] at same
  exact observed.symm.trans same

/-- Any renamed additive row family admits a separating invariant even when
multiple input keys have the same learned column code. The names and the
row and column codes appear only in this sufficient condition. -/
theorem additive_aligned_read {G : Type*} [AddCommGroup G] (rename : G ≃ V)
    (row : I → G) (key : X → G) (P : I → X → Option V)
    (truth : ∀ i x v, P i x = some v → rename (row i + key x) = v)
    (i j : I) (anchor query : X) (u v q y : V)
    (left : P i anchor = some u) (right : P j anchor = some v)
    (peer : P i query = some q)
    (path : Relation.EqvGen (AlignedEdge P) (u, v) (q, y)) :
    y = rename (row j + key query) := by
  apply aligned_read_sound P (fun c x => rename (row c + key x)) truth
    (fun p => rename.symm p.2 - rename.symm p.1) _ _ i j anchor query u v q y left right peer path
  · intro r s x z
    simp only [Equiv.symm_apply_apply]
    abel
  · intro a b c h
    exact rename.symm.injective (sub_left_inj.mp h)

-- The target row is missing its last cell. A third context supplies the
-- same translation between two output pairs, without a numeric operation
-- on the hidden names in the inference rule.
private def exampleBank (r k : ℤ) : Option ℤ :=
  if r = 1 ∧ k = 2 then none else some (r + k / 2)

private theorem examplePath :
    Relation.EqvGen (AlignedEdge exampleBank) ((0 : ℤ), 1) (1, 2) := by
  apply Relation.EqvGen.rel
  exact ⟨2, 3, -4, -2, by decide, by decide, by decide, by decide, by decide, by decide⟩

example : exampleBank 1 2 = none ∧ exampleBank 2 0 = exampleBank 2 1 ∧
    ∀ y : ℤ, Relation.EqvGen (AlignedEdge exampleBank) (0, 1) (1, y) → y = 2 := by
  refine ⟨by decide, by decide, fun y path => ?_⟩
  apply additive_aligned_read (Equiv.refl ℤ) id (fun z : ℤ => z / 2) exampleBank
    _ 0 1 0 2 0 1 1 y (by decide) (by decide) (by decide) path
  intro i x v h
  by_cases missing : i = 1 ∧ x = 2
  · simp [exampleBank, missing] at h
  · simpa [exampleBank, missing] using h

example : ∃ y, Relation.EqvGen (AlignedEdge exampleBank) ((0 : ℤ), 1) (1, y) :=
  ⟨2, examplePath⟩


section Relabel

variable {J K W : Type*}

/-- Renaming the context, key and output dictionaries separately preserves
every aligned path, including chains traversed against a witnessed edge. -/
theorem aligned_rename (P : I → X → Option V) (Q : J → K → Option W)
    (r : I → J) (k : X → K) (v : V → W)
    (hr : Function.Injective r) (hk : Function.Injective k)
    (mapped : ∀ i x, Q (r i) (k x) = (P i x).map v) {a b : V × V}
    (path : Relation.EqvGen (AlignedEdge P) a b) :
    Relation.EqvGen (AlignedEdge Q) (Prod.map v v a) (Prod.map v v b) := by
  induction path with
  | rel a b edge =>
    obtain ⟨i, j, x, y, hij, hxy, h₁, h₂, h₃, h₄⟩ := edge
    apply Relation.EqvGen.rel
    exact ⟨r i, r j, k x, k y, hr.ne hij, hk.ne hxy,
      by simp [mapped, h₁], by simp [mapped, h₂], by simp [mapped, h₃], by simp [mapped, h₄]⟩
  | refl => exact .refl _
  | symm _ _ _ ih => exact .symm _ _ ih
  | trans _ _ _ _ _ h₁ h₂ => exact .trans _ _ _ h₁ h₂

/-- Apply independent changes of names to the context, key and output slots. -/
def renameAligned (P : I → X → Option V) (r : I ≃ J) (k : X ≃ K) (v : V ≃ W)
    (j : J) (l : K) : Option W := (P (r.symm j) (k.symm l)).map v

/-- Every path in bijectively renamed observations decodes back to a path in
the original bank. This includes paths used to reject an ambiguous image. -/
theorem aligned_rename_iff (P : I → X → Option V) (r : I ≃ J) (k : X ≃ K) (v : V ≃ W)
    (a b : V × V) :
    Relation.EqvGen (AlignedEdge (renameAligned P r k v)) (Prod.map v v a) (Prod.map v v b) ↔
      Relation.EqvGen (AlignedEdge P) a b := by
  constructor
  · intro path
    have back (j : J) (z : K) :
        P (r.symm j) (k.symm z) = (renameAligned P r k v j z).map v.symm := by
      cases h : P (r.symm j) (k.symm z) <;> simp [renameAligned, h]
    have original := aligned_rename _ P r.symm k.symm v.symm
      r.symm.injective k.symm.injective back path
    simpa [Prod.map] using original
  · exact aligned_rename P _ r k v r.injective k.injective (by simp [renameAligned])

/-- An available result of the original triple search is kept. At an
abstention the peer read contributes exactly its own uniquely decoded value. -/
theorem first_read_eq (primary peer : Option V) (v : V) :
    primary.or peer = some v ↔ primary = some v ∨ (primary = none ∧ peer = some v) := by
  cases primary <;> simp

example : Relation.EqvGen (AlignedEdge exampleBank) ((0 : ℤ), 1) (1, 2) :=
  aligned_rename exampleBank exampleBank id id id (fun _ _ h => h) (fun _ _ h => h)
    (fun i x => by cases h : exampleBank i x <;> simp [h]) examplePath

end Relabel

end Transformer.Phasor.TripleTransport
