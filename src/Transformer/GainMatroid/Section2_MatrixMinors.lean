module

public import Mathlib.LinearAlgebra.Matrix.Transvection
public import Transformer.GainMatroid.Section2_LinkMinors
public import Transformer.GainMatroid.Section4_Switching

/-!
# Canonical matrices of deletions and of contractions of normalized links

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.3, Lemma 2.5. `A \ e` removes column `e`. A
matrix `A / e` (`IsMatrixContraction`) comes from row operations that make column `e` a unit vector,
by removing that column and the row of its `1`.

Deleting edges deletes their columns from both canonical matrices (Lemma 2.5 (i)). If the link `e`
has identity gain, adding the row of `tail e` to the row of `head e` and removing the row of
`tail e` and column `e` gives the canonical matrix of `G / e` with the restricted gain function.
The rows that remain are the vertices of `G / e`, through `OrientedGraph.contractRow`.
-/

@[expose] public section

open Matrix

namespace Transformer.GainMatroid

variable {V E F R R' : Type*} [DecidableEq V] [Field F]

/-- `B` is a matrix `A / e` (§2.3, before Lemma 2.5): the row operations `T` make column `e` the
unit vector with its `1` in row `i`, and `B` is `T * A` without row `i` and column `e`. The map `σ`
carries the rows of `B` bijectively onto the rows other than `i`. -/
structure IsMatrixContraction [Fintype R] [DecidableEq R] (A : Matrix R E F) (e : E) (i : R)
    (σ : R' → R) (B : Matrix R' ↥({e}ᶜ : Set E) F) : Prop where
  /-- Distinct rows of `B` come from distinct rows of `A`. -/
  injective : Function.Injective σ
  /-- The rows of `B` are all the rows other than `i`. -/
  range_eq : Set.range σ = {i}ᶜ
  /-- The row operations. -/
  exists_rowOp : ∃ T : Matrix R R F, IsUnit T ∧ (∀ r, (T * A) r e = if r = i then 1 else 0) ∧
    ∀ r' (f : ↥({e}ᶜ : Set E)), (T * A) (σ r') f = B r' f

/-- **Lemma 2.5 (i)**, frame case: deleting edges deletes their columns. -/
theorem frameMatrix_delete (G : OrientedGraph V E) (D : Set E) (φ : E → Fˣ) :
    frameMatrix (G.delete D) (fun f => φ f) = (frameMatrix G φ).submatrix id Subtype.val :=
  rfl

/-- **Lemma 2.5 (i)**, lift case: deleting edges deletes their columns. The source writes
the `Fˣ`-gain `φ` here, but `A_L` takes an `F⁺`-gain function, so `ψ` is meant. -/
theorem liftMatrix_delete (G : OrientedGraph V E) (D : Set E) (ψ : E → F) :
    liftMatrix (G.delete D) (fun f => ψ f) = (liftMatrix G ψ).submatrix id Subtype.val := by
  ext r f
  cases r <;> rfl

/-- Adding row `j` to row `i`, entry by entry. -/
theorem transvection_one_mul_apply {n m : Type*} [Fintype n] [DecidableEq n] (i j a : n) (b : m)
    (M : Matrix n m F) :
    (transvection i j (1 : F) * M) a b = M a b + if a = i then M j b else 0 := by
  by_cases h : a = i
  · subst h
    rw [transvection_mul_apply_same, ite_eq_left rfl, one_mul]
  · rw [transvection_mul_apply_of_ne i j a b h, ite_eq_right h, add_zero]

/-- The rows of a matrix with an added first row, when `σ` misses exactly `i`. -/
theorem range_optionMap {σ : R' → R} {i : R} (h : Set.range σ = {i}ᶜ) :
    Set.range (Option.map σ) = {some i}ᶜ := by
  ext r
  cases r with
  | none => exact ⟨fun _ hc => by simp at hc, fun _ => ⟨none, rfl⟩⟩
  | some y =>
    have hy := Set.ext_iff.1 h y
    constructor
    · rintro ⟨_ | q, hq⟩ hc
      · simp at hq
      · have hqy : σ q = y := by simpa using hq
        exact hy.1 ⟨q, hqy⟩ (Option.some_inj.1 hc)
    · intro hne
      obtain ⟨q, hq⟩ := hy.2 fun hyi => hne (congrArg some hyi)
      exact ⟨some q, by simp [hq]⟩

namespace OrientedGraph

variable (G : OrientedGraph V E) {e : E}

/-- The vertex map of contracting `e`: `tail e` goes to `head e`, the others stay. -/
def contractVertex (e : E) (x : V) : V := if x = G.tail e then G.head e else x

theorem contractVertex_tail (e : E) : G.contractVertex e (G.tail e) = G.head e := by
  simp [contractVertex]

theorem contractVertex_head (e : E) : G.contractVertex e (G.head e) = G.head e := by
  by_cases h : G.head e = G.tail e <;> simp [contractVertex, h]

/-- Two vertices lie in one vertex of `G / e` exactly when `contractVertex` identifies them. -/
theorem mk_eq_mk_iff_contractVertex {a b : V} :
    Quotient.mk (G.reachableSetoid {e}) a = Quotient.mk _ b ↔
      G.contractVertex e a = G.contractVertex e b := by
  constructor
  · intro h
    obtain ⟨p, hp⟩ : G.ReachableIn {e} a b := Quotient.exact h
    refine Walk.eq_of_forall_mem_edges _ p fun f hf => ?_
    obtain rfl : f = e := hp f hf
    rw [contractVertex_tail, contractVertex_head]
  · intro h
    have hc := G.contract_class_eq (F := {e}) (f := e) rfl
    unfold contractVertex at h
    by_cases ha : a = G.tail e
    · by_cases hb : b = G.tail e
      · rw [ha, hb]
      · rw [ite_eq_left ha, ite_eq_right hb] at h
        rw [ha, hc, h]
    · rw [ite_eq_right ha] at h
      by_cases hb : b = G.tail e
      · rw [ite_eq_left hb] at h
        rw [h, hb, hc]
      · rw [ite_eq_right hb] at h
        rw [h]

/-- The row of `A` that carries a vertex of `G / e`: its vertex other than `tail e`. -/
def contractRow (e : E) : Quotient (G.reachableSetoid {e}) → V :=
  Quotient.lift (G.contractVertex e) fun _ _ h => G.mk_eq_mk_iff_contractVertex.1 (Quotient.sound h)

theorem contractRow_injective (e : E) : Function.Injective (G.contractRow e) := by
  rintro ⟨a⟩ ⟨b⟩ h
  exact G.mk_eq_mk_iff_contractVertex.2 h

theorem contractVertex_ne (he : G.IsLink e) (x : V) : G.contractVertex e x ≠ G.tail e := by
  unfold contractVertex
  by_cases hx : x = G.tail e
  · rw [ite_eq_left hx]
    exact he.symm
  · rw [ite_eq_right hx]
    exact hx

/-- For a link `e`, the rows of `G / e` are the rows of `G` other than `tail e`. -/
theorem range_contractRow (he : G.IsLink e) : Set.range (G.contractRow e) = {G.tail e}ᶜ := by
  ext y
  constructor
  · rintro ⟨⟨a⟩, rfl⟩
    exact G.contractVertex_ne he a
  · intro hy
    have hy' : y ≠ G.tail e := hy
    exact ⟨Quotient.mk _ y, by simp [contractRow, contractVertex, hy']⟩

private theorem ite_contractVertex {y : V} (hy : y ≠ G.tail e) (t : V) :
    (if y = G.contractVertex e t then (1 : F) else 0) =
      (if y = t then 1 else 0) + if y = G.head e then (if G.tail e = t then 1 else 0) else 0 := by
  by_cases ht : t = G.tail e
  · rw [ht, contractVertex_tail]
    simp [hy]
  · have ht' : G.tail e ≠ t := Ne.symm ht
    simp [contractVertex, ht, ht']

/-- Entries of a column in a row `y ≠ tail e`, after adding row `tail e` to row `head e`. -/
theorem incidence_contractVertex {y : V} (hy : y ≠ G.tail e) (c : F) (a b : V) :
    (if y = G.contractVertex e a then (1 : F) else 0) -
        c * (if y = G.contractVertex e b then 1 else 0) =
      (if y = a then 1 else 0) - c * (if y = b then 1 else 0) +
        if y = G.head e then
          (if G.tail e = a then 1 else 0) - c * (if G.tail e = b then 1 else 0) else 0 := by
  rw [ite_contractVertex G hy a, ite_contractVertex G hy b]
  by_cases hyh : y = G.head e
  · simp only [ite_eq_left hyh]
    ring
  · simp only [ite_eq_right hyh]
    ring

/-- An indicator on the vertices of `G / e`, read on `contractVertex`. -/
theorem ite_mk_eq [DecidableEq (Quotient (G.reachableSetoid {e}))] (x a : V) :
    (if Quotient.mk (G.reachableSetoid {e}) x = Quotient.mk _ a then (1 : F) else 0) =
      if G.contractVertex e x = G.contractVertex e a then 1 else 0 :=
  ite_congr (propext G.mk_eq_mk_iff_contractVertex) (fun _ => rfl) fun _ => rfl

end OrientedGraph

end Transformer.GainMatroid
