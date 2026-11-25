module

public import Transformer.GainMatroid.Section3_Families

/-!
# Counting the biased `2C₃`, tubes and `K₄`

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §3.1. Proposition 3.1 says that there are six
unlabelled properly unbalanced biased `2C₃`'s (`proposition3_1`); the text adds that there are three
properly unbalanced biased tubes (`three_properlyUnbalanced_tubes`) and seven biased `K₄`'s,
`D_{t,q}` with `t` balanced triangles and `q` balanced 4-cycles, for
`(t, q) ∈ {(0,0), (0,1), (0,2), (0,3), (1,0), (2,1), (4,2)}` (`seven_biased_K4`).

"Unlabelled" means up to isomorphism (`BiasedGraph.IsIsomorphic`). Each count is stated as a list of
representatives on the labelled graphs `twoC3`, `tube`, `K4` of §4, pairwise non-isomorphic, to one
of which every biased graph on that graph is isomorphic. The figures of §3.1 name the
representatives by their balanced cycles, and the statements record the number of balanced
triangles (`T₀, T₁, T₂, T₂', T₃, T₄`) and 4-cycles (`𝖡₀, 𝖡₁, 𝖡₂`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

namespace BiasedGraph

variable {V E : Type*}

/-- The number of balanced cycles of `Ω` with `k` edges. -/
noncomputable def balancedCount (Ω : BiasedGraph V E) (k : ℕ) : ℕ :=
  {C | C ∈ Ω.balanced ∧ C.encard = k}.ncard

/-- No balanced cycle of the contrabalanced graph is counted. -/
theorem balancedCount_contrabalanced (G : OrientedGraph V E) (k : ℕ) :
    G.contrabalanced.balancedCount k = 0 := by
  simp [balancedCount, contrabalanced]

end BiasedGraph

open BiasedGraph

/-- **Proposition 3.1.** "There are six unlabelled properly unbalanced biased `2C₃`'s": `T₀`, `T₁`,
`T₂`, `T₂'`, `T₃`, `T₄`, with `0, 1, 2, 2, 3, 4` balanced triangles. -/
theorem proposition3_1 : ∃ T : Fin 6 → BiasedGraph (Fin 3) (Fin 6),
    (∀ i, (T i).graph = twoC3 ∧ (T i).IsProperlyUnbalanced) ∧
    (fun i => (T i).balancedCount 3) = ![0, 1, 2, 2, 3, 4] ∧
    (∀ i j, (T i).IsIsomorphic (T j) → i = j) ∧
    ∀ Ω : BiasedGraph (Fin 3) (Fin 6), Ω.graph = twoC3 → Ω.IsProperlyUnbalanced →
      ∃ i, Ω.IsIsomorphic (T i) := by
  sorry

/-- **§3.1.** "A biased tube is properly unbalanced if and only if it has no balanced 2-cycle. There
are three such tubes": `𝖡₀`, `𝖡₁`, `𝖡₂`, with `0, 1, 2` balanced 4-cycles. The first sentence is
`isProperlyUnbalanced_iff_of_tube`. -/
theorem three_properlyUnbalanced_tubes : ∃ B : Fin 3 → BiasedGraph (Fin 4) (Fin 6),
    (∀ i, (B i).graph = tube ∧ (B i).IsProperlyUnbalanced) ∧
    (fun i => (B i).balancedCount 4) = ![0, 1, 2] ∧
    (∀ i j, (B i).IsIsomorphic (B j) → i = j) ∧
    ∀ Ω : BiasedGraph (Fin 4) (Fin 6), Ω.graph = tube → Ω.IsProperlyUnbalanced →
      ∃ i, Ω.IsIsomorphic (B i) := by
  sorry

/-- **§3.1.** "There are seven biased `K₄`'s: `D_{0,0}`, `D_{0,1}`, `D_{0,2}`, `D_{0,3}`, `D_{1,0}`,
`D_{2,1}`, and `D_{4,2}`", where `D_{t,q}` has exactly `t` balanced triangles and `q` balanced
4-cycles. That the properly unbalanced ones are the first four is
`isProperlyUnbalanced_iff_of_K4`. -/
theorem seven_biased_K4 : ∃ D : Fin 7 → BiasedGraph (Fin 4) (Fin 6),
    (∀ i, (D i).graph = K4) ∧
    (fun i => ((D i).balancedCount 3, (D i).balancedCount 4)) =
      ![(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (2, 1), (4, 2)] ∧
    (∀ i j, (D i).IsIsomorphic (D j) → i = j) ∧
    ∀ Ω : BiasedGraph (Fin 4) (Fin 6), Ω.graph = K4 → ∃ i, Ω.IsIsomorphic (D i) := by
  sorry

/-- The contrabalanced `K₄` is `D_{0,0}`: it has no balanced triangle and no balanced 4-cycle. -/
example : (K4.contrabalanced.balancedCount 3, K4.contrabalanced.balancedCount 4) = (0, 0) := by
  simp [balancedCount_contrabalanced]

end Transformer.GainMatroid
