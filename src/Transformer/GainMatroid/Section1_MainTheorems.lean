module

public import Transformer.GainMatroid.Section2_ExchangeClasses
public import Transformer.GainMatroid.Section5_MainCases

/-!
# Theorems 1, 2 and 3

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §1.2 and §5. For a 3-connected matroid `M` of rank
greater than two and a loopless biased graph representing it, canonical lift matrices are
projectively equivalent exactly when their gain functions are switching-and-scaling equivalent, and
without a balancing vertex no canonical frame matrix is projectively equivalent to a canonical lift
matrix (Theorem 1 (ii), (iii)). Theorem 1 (i) fails as printed on an almost-balanced biased graph
(`theorem1_i_fails`); it holds when the biased graph is not almost balanced.

Every representation of `M` is particular to the biased graph when the field is not `GF(2)`
(Theorem 2, corrected: `Section1_Theorem2` shows that the printed statement fails in rank two and
over `GF(2)`), and the projective classes of representations correspond to the classes of gain
functions of `S_𝔽(G, B)` (Theorem 3). The source defines `S_𝔽(G, B)` for four cases, none of which
covers a balanced `(G, B)`; there `S_{𝔽ˣ}(G, B)`, a single class, is used.

Each of these first needs that a biased graph representing a 3-connected matroid of rank greater
than two is 2-connected and has no 2-separation with one side balanced
(`Represents.isKConnected_two`, §5.1).
-/

@[expose] public section

namespace Transformer.GainMatroid

universe uV uE

variable {V : Type uV} {E : Type uE} {R F : Type*} [Field F]

open OrientedGraph

namespace BiasedGraph

variable [Finite V] [Finite E] [DecidableEq V] {Ω : BiasedGraph V E}

/-- §5.1: "A biased graph representing a 3-connected matroid is 2-connected and has no
2-separation with one side inducing a balanced subgraph." The rank of the matroid is assumed greater
than two, as in Theorem 1: the contrabalanced biased graph of four parallel links on two vertices
represents the 3-connected `U_{2,4}` and is not 2-connected. -/
theorem Represents.isKConnected_two {M : Matroid E} (hM : IsNConnected M 3) (hr : 2 < M.eRank)
    (hΩ : Ω.Represents M) :
    Ω.graph.IsKConnected 2 ∧ ∀ A B, Ω.graph.IsSeparation 2 A B → ¬Ω.IsBalancedSet A := by
  sorry

/-- **Theorem 1.** "Let `M` be a 3-connected matroid of rank greater than two. Let `(G, B)` be a
loopless biased graph representing `M` and let `𝔽` be a field. (i) The canonical frame matrices
given by two `𝔽ˣ`-gain functions `φ` and `ψ` realizing `(G, B)` are projectively equivalent if and
only if `φ` and `ψ` are switching equivalent. (ii) The canonical lift matrices given by two
`𝔽⁺`-gain functions `φ` and `ψ` realizing `(G, B)` are projectively equivalent if and only if `φ`
and `ψ` are switching-and-scaling equivalent. (iii) Provided `(G, B)` has no vertex whose deletion
leaves a biased graph with no unbalanced cycles, no canonical frame matrix representation is
projectively equivalent to any canonical lift matrix representation of `M`."

Part (i) fails as printed (`theorem1_i_fails`); here it assumes that `(G, B)` is not almost
balanced. -/
theorem theorem1 {M : Matroid E} (hM : IsNConnected M 3) (hr : 2 < M.eRank)
    (hG : Ω.graph.Loopless) (hΩ : Ω.Represents M) :
    (¬Ω.IsAlmostBalanced → ∀ φ ψ : E → Fˣ, Ω.Realizes φ → Ω.Realizes ψ →
      (ProjEquiv (frameMatrix Ω.graph φ) (frameMatrix Ω.graph ψ) ↔ SwitchEquiv Ω.graph φ ψ)) ∧
    (∀ φ ψ : E → F, Ω.AddRealizes φ → Ω.AddRealizes ψ →
      (ProjEquiv (liftMatrix Ω.graph φ) (liftMatrix Ω.graph ψ) ↔ SwitchScaleEquiv Ω.graph φ ψ)) ∧
    ((∀ v, ¬Ω.IsBalancingVertex v) → ∀ (φ : E → Fˣ) (ψ : E → F), Ω.Realizes φ →
      Ω.AddRealizes ψ → ¬ProjEquiv (frameMatrix Ω.graph φ) (liftMatrix Ω.graph ψ)) := by
  sorry

/-- **Theorem 2**, corrected. The source: "Let `M` be a 3-connected matroid, and let `𝔽` be a
field. Let `A` be a matrix over `𝔽` representing `M` and let `(G, B)` be a biased graph representing
`M`. Then `A` is projectively equivalent to a canonical representation particular to `(G, B)`." It
fails in rank two (`theorem2_fails_rank_two`) and over `GF(2)` (`theorem2_fails_gf2`); here the rank
is greater than two and the field is not `GF(2)`. -/
theorem theorem2 {M : Matroid E} (hM : IsNConnected M 3) (hr : 2 < M.eRank)
    (hF : ∃ a : Fˣ, a ≠ 1) {A : Matrix R E F} (hA : IsMatrixRepresentation M A)
    (hΩ : Ω.Represents M) : Ω.IsFrameParticular A ∨ Ω.IsLiftParticular A := by
  sorry

/-- **Theorem 3.** "Let `M` be a 3-connected matroid of rank greater than two. Let `𝔽` be a field
and let `(G, B)` be a loopless biased graph representing `M`. The projective equivalence classes of
matrices over `𝔽` representing `M` are in one-to-one correspondence with the switching classes of
gain functions in `S_𝔽(G, B)`", where `S_𝔽(G, B)` is `S_{𝔽ˣ}(G, B)` if `M = F(G, B) ≠ L(G, B)`,
`S_{𝔽⁺}(G, B)` if `M = L(G, B) ≠ F(G, B)`, `S_{𝔽ˣ}(G, B) ∪ S_{𝔽⁺}(G, B)` if `(G, B)` is tangled,
and `S_{𝔽⁺}(Ĝ, B̂)` if `(G, B)` is almost balanced. For a balanced `(G, B)`, which the source's
cases leave out, `S_{𝔽ˣ}(G, B)` is used. The correspondence sends the class of a canonical matrix to
the class of its gain function. -/
theorem theorem3 {M : Matroid E} (hM : IsNConnected M 3) (hr : 2 < M.eRank)
    (hG : Ω.graph.Loopless) (hΩ : Ω.Represents M) (F : Type*) [Field F] :
    (Ω.IsFrameMatroid M ∧ ¬Ω.IsLiftMatroid M → ∃ e : ProjClass F M ≃ RealizationClass Ω Fˣ,
      ∀ (A : RepMatrix F M) (φ : E → Fˣ) (hφ : Ω.Realizes φ),
        ProjEquiv A.2.1 (frameMatrix Ω.graph φ) → e (Quotient.mk _ A) = Quotient.mk _ ⟨φ, hφ⟩) ∧
    (Ω.IsLiftMatroid M ∧ ¬Ω.IsFrameMatroid M → ∃ e : ProjClass F M ≃ AddRealizationClass Ω F,
      ∀ (A : RepMatrix F M) (ψ : E → F) (hψ : Ω.AddRealizes ψ),
        ProjEquiv A.2.1 (liftMatrix Ω.graph ψ) → e (Quotient.mk _ A) = Quotient.mk _ ⟨ψ, hψ⟩) ∧
    (Ω.IsTangled →
      ∃ e : ProjClass F M ≃ RealizationClass Ω Fˣ ⊕ AddRealizationClass Ω F,
        (∀ (A : RepMatrix F M) (φ : E → Fˣ) (hφ : Ω.Realizes φ),
          ProjEquiv A.2.1 (frameMatrix Ω.graph φ) →
            e (Quotient.mk _ A) = Sum.inl (Quotient.mk _ ⟨φ, hφ⟩)) ∧
        ∀ (A : RepMatrix F M) (ψ : E → F) (hψ : Ω.AddRealizes ψ),
          ProjEquiv A.2.1 (liftMatrix Ω.graph ψ) →
            e (Quotient.mk _ A) = Sum.inr (Quotient.mk _ ⟨ψ, hψ⟩)) ∧
    (Ω.IsAlmostBalanced → ∃ u, Ω.IsRollable u ∧ ∀ Ω', Ω.IsUnrolling u Ω' →
      ∃ e : ProjClass F M ≃ AddRealizationClass Ω' F,
        ∀ (A : RepMatrix F M) (ψ : E → F) (hψ : Ω'.AddRealizes ψ),
          ProjEquiv A.2.1 (liftMatrix Ω'.graph ψ) → e (Quotient.mk _ A) = Quotient.mk _ ⟨ψ, hψ⟩) ∧
    (Ω.IsBalanced → ∃ e : ProjClass F M ≃ RealizationClass Ω Fˣ,
      ∀ (A : RepMatrix F M) (φ : E → Fˣ) (hφ : Ω.Realizes φ),
        ProjEquiv A.2.1 (frameMatrix Ω.graph φ) → e (Quotient.mk _ A) = Quotient.mk _ ⟨φ, hφ⟩) := by
  sorry

end BiasedGraph

open BiasedGraph

local instance instFactPrimeSevenMainTheorems : Fact (Nat.Prime 7) := ⟨by decide⟩

/-- The hypotheses of the statements of this file hold for `U_{3,5}`, the loopless contrabalanced
`fanGraph`, the field `𝔽₇` and the frame matrix of `fanGain₁`. -/
example : IsNConnected (uniformMatroid (Fin 5) 3) 3 ∧ 2 < (uniformMatroid (Fin 5) 3).eRank ∧
    fanGraph.contrabalanced.graph.Loopless ∧
    fanGraph.contrabalanced.Represents (uniformMatroid (Fin 5) 3) ∧ (∃ a : (ZMod 7)ˣ, a ≠ 1) ∧
    IsMatrixRepresentation (uniformMatroid (Fin 5) 3) (frameMatrix fanGraph fanGain₁) :=
  ⟨uniformMatroid_three_five_isNConnected,
    by rw [uniformMatroid_three_five_eRank]; exact_mod_cast (by omega : (2 : ℕ) < 3),
    fanGraph_loopless, Or.inl fan_isFrameMatroid, ⟨fanTwo, by decide⟩,
    fan_isMatrixRepresentation⟩

end Transformer.GainMatroid
