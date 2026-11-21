module

public import Transformer.GainMatroid.Section1_Prop11

/-!
# Oriented graphs and canonical matrices

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §1.1 and §2.3. A graph carries the reference
orientation that §2.3 chooses: every edge has a tail and a head, equal for a loop. A gain function
then assigns a group element to each edge in this orientation.

`frameMatrix` and `liftMatrix` are the canonical frame and lift matrices `A_F(G, φ)` and
`A_L(G, ψ)` of §1.1, given by one formula for links and loops. The lemmas `frameMatrix_tail`,
`frameMatrix_head`, `frameMatrix_loop`, `frameMatrix_of_ne` and their `liftMatrix` versions recover
the source's description entry by entry. The extra row `v₀` of the lift matrix is `none : Option V`.
-/

@[expose] public section

open Matrix

namespace Transformer.GainMatroid

/-- A graph with vertices `V` and edges `E`, each edge oriented from its tail to its head (§2.3).
A loop has equal tail and head. -/
structure OrientedGraph (V E : Type*) where
  /-- The tail of an edge. -/
  tail : E → V
  /-- The head of an edge. -/
  head : E → V

variable {V E F : Type*}

/-- `e` is a link: its ends are distinct (§2.1). -/
abbrev OrientedGraph.IsLink (G : OrientedGraph V E) (e : E) : Prop := G.tail e ≠ G.head e

/-- `G` has no loops. -/
abbrev OrientedGraph.Loopless (G : OrientedGraph V E) : Prop := ∀ e, G.IsLink e

variable [DecidableEq V] [Field F]

/-- The **canonical frame matrix** `A_F(G, φ)` of an `Fˣ`-gain function (§1.1): a link `e` from `u`
to `v` with gain `g` has `1` in row `u`, `-g` in row `v` and zero elsewhere; a loop at `v` has
`1 - g` in row `v` and zero elsewhere. -/
def frameMatrix (G : OrientedGraph V E) (φ : E → Fˣ) : Matrix V E F :=
  Matrix.of fun w e => (if w = G.tail e then 1 else 0) - (φ e : F) * (if w = G.head e then 1 else 0)

/-- The **canonical lift matrix** `A_L(G, ψ)` of an `F⁺`-gain function (§1.1): the oriented
incidence matrix of `G` with one more row `v₀ = none` holding the gains. A link from `u` to `v` has
`1` in row `u` and `-1` in row `v`; a loop has zero in every row of `V`. -/
def liftMatrix (G : OrientedGraph V E) (ψ : E → F) : Matrix (Option V) E F :=
  Matrix.of fun r e =>
    match r with
    | none => ψ e
    | some w => (if w = G.tail e then 1 else 0) - (if w = G.head e then 1 else 0)

section Entries

variable (G : OrientedGraph V E) (φ : E → Fˣ) (ψ : E → F) {e : E} {w : V}

theorem frameMatrix_apply :
    frameMatrix G φ w e =
      (if w = G.tail e then 1 else 0) - (φ e : F) * (if w = G.head e then 1 else 0) := rfl

/-- A link has `1` in the row of its tail. -/
theorem frameMatrix_tail (he : G.IsLink e) : frameMatrix G φ (G.tail e) e = 1 := by
  simp [frameMatrix_apply, he]

/-- A link with gain `g` has `-g` in the row of its head. -/
theorem frameMatrix_head (he : G.IsLink e) : frameMatrix G φ (G.head e) e = -(φ e : F) := by
  simp [frameMatrix_apply, he.symm]

/-- A loop with gain `g` has `1 - g` in the row of its vertex. -/
theorem frameMatrix_loop (he : G.tail e = G.head e) :
    frameMatrix G φ (G.tail e) e = 1 - (φ e : F) := by
  simp [frameMatrix_apply, ← he]

/-- All other entries of a column of the frame matrix vanish. -/
theorem frameMatrix_of_ne (h₁ : w ≠ G.tail e) (h₂ : w ≠ G.head e) : frameMatrix G φ w e = 0 := by
  simp [frameMatrix_apply, h₁, h₂]

theorem liftMatrix_none : liftMatrix G ψ none e = ψ e := rfl

theorem liftMatrix_some :
    liftMatrix G ψ (some w) e =
      (if w = G.tail e then 1 else 0) - (if w = G.head e then 1 else 0) := rfl

/-- A link has `1` in the row of its tail. -/
theorem liftMatrix_tail (he : G.IsLink e) : liftMatrix G ψ (some (G.tail e)) e = 1 := by
  simp [liftMatrix_some, he]

/-- A link has `-1` in the row of its head. -/
theorem liftMatrix_head (he : G.IsLink e) : liftMatrix G ψ (some (G.head e)) e = -1 := by
  simp [liftMatrix_some, he.symm]

/-- A loop has zero in every row of `V`. -/
theorem liftMatrix_loop (he : G.tail e = G.head e) (w : V) : liftMatrix G ψ (some w) e = 0 := by
  simp [liftMatrix_some, he]

/-- All other entries in the rows of `V` vanish. -/
theorem liftMatrix_of_ne (h₁ : w ≠ G.tail e) (h₂ : w ≠ G.head e) :
    liftMatrix G ψ (some w) e = 0 := by
  simp [liftMatrix_some, h₁, h₂]

end Entries

/-- One link from `0` to `1` on three vertices, for the examples. -/
def linkGraph : OrientedGraph (Fin 3) Unit := ⟨fun _ => 0, fun _ => 1⟩

/-- One loop at `0` on two vertices, for the examples. -/
def loopGraph : OrientedGraph (Fin 2) Unit := ⟨fun _ => 0, fun _ => 0⟩

theorem linkGraph_isLink : linkGraph.IsLink () := by
  simp [linkGraph]

theorem linkGraph_ne_tail : (2 : Fin 3) ≠ linkGraph.tail () := by
  simp [linkGraph]

theorem linkGraph_ne_head : (2 : Fin 3) ≠ linkGraph.head () := by
  simp [linkGraph]

example (g : ℚˣ) : frameMatrix linkGraph (fun _ => g) (linkGraph.tail ()) () = (1 : ℚ) :=
  frameMatrix_tail _ _ linkGraph_isLink

example (g : ℚˣ) : frameMatrix linkGraph (fun _ => g) (linkGraph.head ()) () = -(g : ℚ) :=
  frameMatrix_head _ _ linkGraph_isLink

example (g : ℚˣ) : frameMatrix loopGraph (fun _ => g) (loopGraph.tail ()) () = 1 - (g : ℚ) :=
  frameMatrix_loop _ _ rfl

example (g : ℚˣ) : frameMatrix linkGraph (fun _ => g) 2 () = (0 : ℚ) :=
  frameMatrix_of_ne _ _ linkGraph_ne_tail linkGraph_ne_head

example (g : ℚ) : liftMatrix linkGraph (fun _ => g) (some (linkGraph.tail ())) () = 1 :=
  liftMatrix_tail _ _ linkGraph_isLink

example (g : ℚ) : liftMatrix linkGraph (fun _ => g) (some (linkGraph.head ())) () = -1 :=
  liftMatrix_head _ _ linkGraph_isLink

example (g : ℚ) : liftMatrix loopGraph (fun _ => g) (some 1) () = 0 :=
  liftMatrix_loop _ _ rfl 1

example (g : ℚ) : liftMatrix linkGraph (fun _ => g) (some 2) () = 0 :=
  liftMatrix_of_ne _ _ linkGraph_ne_tail linkGraph_ne_head

end Transformer.GainMatroid
