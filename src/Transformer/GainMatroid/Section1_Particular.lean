module

public import Transformer.GainMatroid.Section2_CanonicalMatrices
public import Transformer.GainMatroid.Section3_Families

/-!
# Canonical matrices particular to a biased graph

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §1.2: "We say a matrix representation is
*particular to* the biased graph `(G, B)` when it is a canonical matrix arising from a gain function
realizing `(G, B)`." The statements of §4.2 and §5 say that a matrix `A` "is projectively equivalent
to a canonical frame (lift) matrix particular to `(G, B)`": `IsFrameParticular Ω A`
(`IsLiftParticular Ω A`). The canonical matrices are read in the orientation of `Ω.graph`, where
the source chooses one freely.

A set `A` of edges is *balanced* when every cycle within it is (`IsBalancedSet`), as in "no vertical
2-separation with one side balanced" (Theorem 5.4).
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

namespace BiasedGraph

variable {V E R F : Type*} [DecidableEq V] [Field F]

/-- `A` is projectively equivalent to a canonical frame matrix particular to `Ω` (§1.2): the frame
matrix of an `Fˣ`-gain function realizing `Ω`. -/
def IsFrameParticular (Ω : BiasedGraph V E) (A : Matrix R E F) : Prop :=
  ∃ φ : E → Fˣ, Ω.Realizes φ ∧ ProjEquiv A (frameMatrix Ω.graph φ)

/-- `A` is projectively equivalent to a canonical lift matrix particular to `Ω` (§1.2): the lift
matrix of an `F⁺`-gain function realizing `Ω`. -/
def IsLiftParticular (Ω : BiasedGraph V E) (A : Matrix R E F) : Prop :=
  ∃ ψ : E → F, Ω.AddRealizes ψ ∧ ProjEquiv A (liftMatrix Ω.graph ψ)

/-- Every cycle within `A` is balanced: `A` is a balanced side of a separation (§5.1). -/
def IsBalancedSet (Ω : BiasedGraph V E) (A : Set E) : Prop :=
  ∀ C, Ω.graph.IsCycle C → C ⊆ A → C ∈ Ω.balanced

variable {Ω : BiasedGraph V E}

theorem IsFrameParticular.of_projEquiv {R' : Type*} {A : Matrix R E F} {B : Matrix R' E F}
    (h : Ω.IsFrameParticular B) (hAB : ProjEquiv A B) : Ω.IsFrameParticular A :=
  h.imp fun _ hφ => ⟨hφ.1, hAB.trans hφ.2⟩

theorem IsLiftParticular.of_projEquiv {R' : Type*} {A : Matrix R E F} {B : Matrix R' E F}
    (h : Ω.IsLiftParticular B) (hAB : ProjEquiv A B) : Ω.IsLiftParticular A :=
  h.imp fun _ hψ => ⟨hψ.1, hAB.trans hψ.2⟩

omit [DecidableEq V] [Field F] in
/-- The empty set of edges is balanced. -/
theorem isBalancedSet_empty (Ω : BiasedGraph V E) : Ω.IsBalancedSet ∅ := by
  rintro C ⟨u, W, hW, rfl⟩ hsub
  obtain ⟨e, he⟩ := List.exists_mem_of_ne_nil _ hW.edges_ne_nil
  exact absurd (hsub he) (Set.notMem_empty e)

end BiasedGraph

open BiasedGraph

example : (twoC3.gainBiased twoC3Gain).IsFrameParticular (frameMatrix twoC3 twoC3Gain) :=
  ⟨twoC3Gain, rfl, ProjEquiv.refl _⟩

example : (twoC3.addGainBiased twoC3AddGain).IsLiftParticular (liftMatrix twoC3 twoC3AddGain) :=
  ⟨twoC3AddGain, rfl, ProjEquiv.refl _⟩

example : (twoC3.gainBiased twoC3Gain).IsFrameParticular
    ((frameMatrix twoC3 twoC3Gain).submatrix id id) :=
  IsFrameParticular.of_projEquiv ⟨twoC3Gain, rfl, ProjEquiv.refl _⟩ (ProjEquiv.refl _)

example : (twoC3.addGainBiased twoC3AddGain).IsLiftParticular
    ((liftMatrix twoC3 twoC3AddGain).submatrix id id) :=
  IsLiftParticular.of_projEquiv ⟨twoC3AddGain, rfl, ProjEquiv.refl _⟩ (ProjEquiv.refl _)

example := isBalancedSet_empty (twoC3.gainBiased twoC3Gain)

end Transformer.GainMatroid
