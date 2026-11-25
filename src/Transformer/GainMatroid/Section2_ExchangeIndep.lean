module

public import Transformer.GainMatroid.Section2_ExchangeContraction
public import Transformer.GainMatroid.Section2_DeltaIndep
public import Transformer.GainMatroid.Section2_FrameContractIndep
public import Transformer.GainMatroid.Section2_LiftContraction

/-!
# Frame and lift matroids across an exchange pair

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, toward Proposition 2.8. Let `Ω` on `T` and
`Ω'` on `S` be biased graphs of an exchange pair on `X = {a, b, c}` whose balanced cycles agree
across the exchange (`BiasedGraph.BalanceAgree`). A set off `X` is frame- or lift-independent in
both or in neither (`IsExchange.isFrameIndep_iff_of_disjoint`), since `T \ X` is carried into `S`.
For `I` off `c`, the set `I` with `a` and `b` exchanged and `c` added is independent in `Ω'`
exactly when `I` is independent in `Ω` (`IsExchange.isFrameIndep_insert_swap_iff`), since `S / c`
is `T \ c` with `a` and `b` exchanged and a cycle `D` of `T` off `c` is balanced in `Ω` exactly when
`D` with `a` and `b` exchanged, with or without `c`, is balanced in `Ω'`
(`IsExchange.exists_swap_mem_iff`). So the frame and lift matroids of `Ω'` are `Δ_X` of those of
`Ω` (`IsExchange.isDeltaMatroid_frame`, `IsExchange.isDeltaMatroid_lift`), by
`isDeltaMatroid_of_indep_iff`.
-/

@[expose] public section

namespace Transformer.GainMatroid

open scoped symmDiff

variable {V W E : Type*}

namespace BiasedGraph

/-- The balanced cycles of `Ω` and `Ω'` agree across an exchange of `X` (§2.4): a cycle of `Ω`
meeting `X` in no edge or in two is balanced in both or in neither, and a cycle of `Ω` meeting `X`
in one edge is balanced in `Ω` exactly when its symmetric difference with `X` is balanced in
`Ω'`. -/
structure BalanceAgree (Ω : BiasedGraph V E) (Ω' : BiasedGraph W E) (X : Set E) : Prop where
  zero : ∀ D, Ω.graph.IsCycle D → D ∩ X = ∅ → (D ∈ Ω.balanced ↔ D ∈ Ω'.balanced)
  one : ∀ D e, Ω.graph.IsCycle D → D ∩ X = {e} → (D ∈ Ω.balanced ↔ D ∆ X ∈ Ω'.balanced)
  two : ∀ D e f, Ω.graph.IsCycle D → e ≠ f → D ∩ X = {e, f} →
    (D ∈ Ω.balanced ↔ D ∈ Ω'.balanced)

end BiasedGraph

/-- An exchange of two elements that a set holds both or neither of fixes the set. -/
theorem image_swap_eq_self [DecidableEq E] {a b : E} {D : Set E} (h : a ∈ D ↔ b ∈ D) :
    Equiv.swap a b '' D = D := by
  ext e
  rw [Set.mem_image_equiv, Equiv.symm_swap, Equiv.swap_apply_def]
  split_ifs with h₁ h₂
  · rw [h₁]
    exact h.symm
  · rw [h₂]
    exact h
  · exact Iff.rfl

namespace OrientedGraph.IsExchange

open BiasedGraph

variable {ι : V → W} {v : W} {a b c : E} {x y z : V} {Ω : BiasedGraph V E} {Ω' : BiasedGraph W E}

/-- A cycle of `S` meets `X` in no edge or in two. -/
theorem ncard_inter_of_isCycle {T : OrientedGraph V E} {S : OrientedGraph W E}
    (h : IsExchange T S ι v a b c x y z) {D : Set E} (hD : S.IsCycle D) :
    (D ∩ {a, b, c}).ncard = 0 ∨ (D ∩ {a, b, c}).ncard = 2 := by
  rcases h.inter_cases_of_isCycle hD with h0 | ⟨a', b', c', x', y', z', h', -, h2⟩
  · rw [h0, Set.ncard_empty]
    exact Or.inl rfl
  · rw [h2, Set.ncard_pair h'.triangle.ne_bc]
    exact Or.inr rfl

variable [DecidableEq E]

/-- **A cycle off `c` across the exchange.** A cycle `D` of `T` off `c` is balanced in `Ω` exactly
when `D` with `a` and `b` exchanged, with or without `c`, is balanced in `Ω'`. -/
theorem exists_swap_mem_iff (h : IsExchange Ω.graph Ω'.graph ι v a b c x y z)
    (hB : Ω.BalanceAgree Ω' {a, b, c}) {D : Set E} (hD : Ω.graph.IsCycle D) (hcD : c ∉ D) :
    (∃ K₀ ⊆ ({c} : Set E), Equiv.swap a b '' D ∪ K₀ ∈ Ω'.balanced) ↔ D ∈ Ω.balanced := by
  have hab := h.triangle.ne_ab
  have hac := h.triangle.ne_ac
  have hbc := h.triangle.ne_bc
  have hS : ∀ D', D' ∈ Ω'.balanced → (D' ∩ {a, b, c}).ncard ≠ 1 ∧ (D' ∩ {a, b, c}).ncard ≠ 3 :=
    fun D' hD' => by
      rcases h.ncard_inter_of_isCycle (Ω'.isCycle_of_mem D' hD') with h0 | h2 <;> omega
  have hK : (∃ K₀ ⊆ ({c} : Set E), Equiv.swap a b '' D ∪ K₀ ∈ Ω'.balanced) ↔
      Equiv.swap a b '' D ∈ Ω'.balanced ∨ insert c (Equiv.swap a b '' D) ∈ Ω'.balanced := by
    refine ⟨fun ⟨K₀, hK₀, hmem⟩ => ?_, fun hmem => ?_⟩
    · rcases Set.subset_singleton_iff_eq.1 hK₀ with rfl | rfl
      · exact Or.inl (by rwa [Set.union_empty] at hmem)
      · exact Or.inr (by rwa [Set.union_singleton] at hmem)
    · rcases hmem with hmem | hmem
      · exact ⟨∅, Set.empty_subset _, by rwa [Set.union_empty]⟩
      · exact ⟨{c}, subset_rfl, by rwa [Set.union_singleton]⟩
  rw [hK]
  by_cases ha : a ∈ D <;> by_cases hb : b ∈ D
  · rw [image_swap_eq_self (iff_of_true ha hb)]
    have hDX : D ∩ {a, b, c} = {a, b} := by grind
    have h3 : (insert c D ∩ {a, b, c}).ncard = 3 := by
      rw [Set.inter_eq_right.2 (by grind)]
      exact Set.ncard_eq_three.2 ⟨a, b, c, hab, hac, hbc, rfl⟩
    rw [hB.two D a b hD hab hDX]
    exact ⟨fun h' => h'.elim id fun h'' => absurd h3 (hS _ h'').2, Or.inl⟩
  · rw [Equiv.image_swap_of_mem_of_notMem ha hb]
    have hDX : D ∩ {a, b, c} = {a} := by grind
    have h1 : ((insert b D \ {a}) ∩ {a, b, c}).ncard = 1 := by
      rw [show (insert b D \ {a}) ∩ {a, b, c} = {b} by grind, Set.ncard_singleton]
    rw [show insert c (insert b D \ {a}) = D ∆ {a, b, c} by grind, hB.one D a hD hDX]
    exact ⟨fun h' => h'.elim (fun h'' => absurd h1 (hS _ h'').1) id, Or.inr⟩
  · rw [Equiv.swap_comm, Equiv.image_swap_of_mem_of_notMem hb ha]
    have hDX : D ∩ {a, b, c} = {b} := by grind
    have h1 : ((insert a D \ {b}) ∩ {a, b, c}).ncard = 1 := by
      rw [show (insert a D \ {b}) ∩ {a, b, c} = {a} by grind, Set.ncard_singleton]
    rw [show insert c (insert a D \ {b}) = D ∆ {a, b, c} by grind, hB.one D b hD hDX]
    exact ⟨fun h' => h'.elim (fun h'' => absurd h1 (hS _ h'').1) id, Or.inr⟩
  · rw [image_swap_eq_self (iff_of_false ha hb)]
    have hDX : D ∩ {a, b, c} = ∅ := by grind
    have h1 : (insert c D ∩ {a, b, c}).ncard = 1 := by
      rw [show insert c D ∩ {a, b, c} = {c} by grind, Set.ncard_singleton]
    rw [hB.zero D hD hDX]
    exact ⟨fun h' => h'.elim id fun h'' => absurd h1 (hS _ h'').1, Or.inl⟩

end OrientedGraph.IsExchange

end Transformer.GainMatroid
