module

public import Transformer.GainMatroid.Section1_MainTheorems
public import Transformer.GainMatroid.Section2_CycleDegrees
public import Transformer.GainMatroid.Section2_DeltaY

/-!
# Theorem 2 and Conjecture 3 fail as printed; Conjecture 4

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §1.2. Theorem 2 claims that every representation
of a 3-connected matroid is projectively equivalent to a canonical representation particular to any
biased graph representing it, "together with a straightforward check for the case of rank 2".

In rank two it fails: `U_{2,4}` is represented over `𝔽₃` and by the contrabalanced `fourK2` of
four parallel links, but a realization of that biased graph needs four distinct gains on parallel
links, which neither `𝔽₃ˣ` nor `𝔽₃⁺` has (`theorem2_fails_rank_two`). Over `GF(2)` it fails in rank
three: the triangle with a joint at each vertex and its triangle balanced represents `M(K₄)` as a
frame matroid, `GF(2)ˣ` is trivial, so no gain function realizes the joints, and a canonical lift
matrix represents the lift matroid, in which the joints are parallel (`theorem2_fails_gf2`).

The second example, with any `k`-connected graph in place of the triangle, refutes Conjecture 3 as
literally stated (`conjecture3_fails`). Conjecture 4 asks only for a canonical representation, which
need not be particular to the given biased graph (`conjecture4`, open).
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph BiasedGraph

/-- Four parallel links from `0` to `1`. -/
def fourK2 : OrientedGraph (Fin 2) (Fin 4) := ⟨fun _ => 0, fun _ => 1⟩

/-- The triangle `0 → 1 → 2 ← 0` with a loop at each vertex. -/
def triangleLoops : OrientedGraph (Fin 3) (Fin 6) := ⟨![0, 1, 0, 0, 1, 2], ![1, 2, 2, 0, 1, 2]⟩

theorem triangleLoops_isCycle : triangleLoops.IsCycle {0, 1, 2} :=
  isCycle_of_three (x := 0) (y := 1) (z := 2) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩)
    (Or.inr ⟨rfl, rfl⟩) (by decide) (by decide)

/-- `triangleLoops` with its triangle balanced and its three loops joints. -/
def triangleLoopsBiased : BiasedGraph (Fin 3) (Fin 6) :=
  triangleLoops.singleCycleBiased {0, 1, 2} triangleLoops_isCycle

/-- **Theorem 2 fails in rank two.** `U_{2,4}` is 3-connected, represented over `𝔽₃` and by the
contrabalanced `fourK2`, and no `𝔽₃`-representation of it is particular to that biased graph. -/
theorem theorem2_fails_rank_two :
    ∃ A : Matrix (Fin 2) (Fin 4) (ZMod 3), IsNConnected (uniformMatroid (Fin 4) 2) 3 ∧
      IsMatrixRepresentation (uniformMatroid (Fin 4) 2) A ∧
      fourK2.contrabalanced.Represents (uniformMatroid (Fin 4) 2) ∧
      ¬fourK2.contrabalanced.IsFrameParticular A ∧ ¬fourK2.contrabalanced.IsLiftParticular A := by
  sorry

/-- **Theorem 2 fails over `GF(2)`**, in rank three: the frame matroid `M` of `triangleLoopsBiased`
is `M(K₄)`, 3-connected and binary, and no `GF(2)`-representation of it is particular to
`triangleLoopsBiased`. -/
theorem theorem2_fails_gf2 :
    ∃ (M : Matroid (Fin 6)) (A : Matrix (Fin 3) (Fin 6) (ZMod 2)), IsNConnected M 3 ∧
      2 < M.eRank ∧ IsMatrixRepresentation M A ∧ triangleLoopsBiased.Represents M ∧
      ¬triangleLoopsBiased.IsFrameParticular A ∧ ¬triangleLoopsBiased.IsLiftParticular A := by
  sorry

/-- **Conjecture 3** (Zaslavsky) **fails as literally stated.** "Let `(G, B)` be a biased graph,
where `G` is sufficiently connected, and let `𝔽` be a field. If `F(G, B)` (resp., `L(G, B)`) is
linearly representable over `𝔽` then `F(G, B)` (resp., `L(G, B)`) has a canonical representation
particular to `(G, B)`." For every `k`, a `k`-connected graph with two joints at distinct vertices
and every other cycle balanced has a binary frame matroid with no canonical `GF(2)`-representation
particular to it. -/
theorem conjecture3_fails : ∀ k : ℕ, ∃ (V E : Type) (_ : Finite V) (_ : Finite E)
    (_ : DecidableEq V) (Ω : BiasedGraph V E) (M : Matroid E), Ω.graph.IsKConnected k ∧
      Ω.IsFrameMatroid M ∧ IsRepresentable (ZMod 2) M ∧
      ∀ (m : ℕ) (A : Matrix (Fin m) E (ZMod 2)), IsMatrixRepresentation M A →
        ¬Ω.IsFrameParticular A ∧ ¬Ω.IsLiftParticular A := by
  sorry

/-- **Conjecture 4** (Zaslavsky, personal communication), open. "Let `(G, B)` be a biased graph,
where `G` is sufficiently connected, and let `𝔽` be a field. Every `𝔽`-representation of `F(G, B)`
(respectively `L(G, B)`) is projectively equivalent to a canonical representation." "Sufficiently
connected" is read as `k`-connected for some `k`, and a canonical representation as one particular
to some biased graph representing the same matroid: the source calls Theorem 2, which asks for one
particular to `(G, B)`, stronger. -/
theorem conjecture4 : ∃ k : ℕ, ∀ (V E : Type) [Finite V] [Finite E] [DecidableEq V]
    (Ω : BiasedGraph V E) (M : Matroid E), Ω.graph.IsKConnected k → Ω.Represents M →
      ∀ (F : Type) [Field F] (m : ℕ) (A : Matrix (Fin m) E F), IsMatrixRepresentation M A →
        ∃ (W : Type) (_ : Finite W) (_ : DecidableEq W) (Ω' : BiasedGraph W E),
          Ω'.Represents M ∧ (Ω'.IsFrameParticular A ∨ Ω'.IsLiftParticular A) := by
  sorry

end Transformer.GainMatroid
