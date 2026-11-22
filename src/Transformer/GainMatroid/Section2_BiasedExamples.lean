module

public import Mathlib.Algebra.Group.Conj
public import Mathlib.Data.Rat.Init
public import Mathlib.Data.ZMod.Basic
public import Mathlib.Algebra.Field.Rat
public import Transformer.GainMatroid.Section2_Biased

/-!
# Gains on three parallel links

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.3, on the smallest theta: three parallel links.
A gain function on it is a triple of group elements, and two of the links form a balanced cycle
exactly when their gains agree (`pumpkin_balanced_iff`). The gains `(1, 1, g)` with `g ≠ 1` give a
theta with exactly one balanced cycle, and the examples below apply each theorem of the gain layer
to these graphs.
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

variable {Γ : Type*} [Group Γ]

/-- Two different links of `pumpkin` traverse a cycle. -/
theorem pumpkin_cycleWalk {i j : Fin 3} (hij : i ≠ j) :
    ((pumpkinWalk i).append (pumpkinWalk j).reverse).IsCycleWalk :=
  (pumpkinWalk_openly hij).isCycleWalk (by decide) (pumpkinWalk_isPath i) (pumpkinWalk_isPath j)

/-- The gain of a one-edge walk along its orientation is the gain of the edge. -/
theorem pumpkinWalk_gain (γ : Fin 3 → Γ) (i : Fin 3) : (pumpkinWalk i).gain γ = γ i := by
  simp [Walk.gain, pumpkinWalk, Walk.steps, dart, pumpkin]

/-- Two links of `pumpkin` form a balanced cycle exactly when their gains agree. -/
theorem pumpkin_balanced_iff (γ : Fin 3 → Γ) {i j : Fin 3} (hij : i ≠ j) :
    (pumpkinWalk i).edgeSet ∪ (pumpkinWalk j).edgeSet ∈ pumpkin.balancedCycles γ ↔ γ i = γ j := by
  rw [(pumpkinWalk_openly hij).mem_balancedCycles_iff γ (by decide) (pumpkinWalk_isPath i)
    (pumpkinWalk_isPath j), pumpkinWalk_gain, pumpkinWalk_gain]

/-- The gains `(1, 1, g)` with `g ≠ 1` balance the cycle of the first two links. -/
theorem pumpkin_first_balanced (g : Γ) :
    (pumpkinWalk 0).edgeSet ∪ (pumpkinWalk 1).edgeSet ∈ pumpkin.balancedCycles ![1, 1, g] :=
  (pumpkin_balanced_iff _ (by decide)).2 rfl

/-- The gains `(1, 1, g)` with `g ≠ 1` leave the cycle of the first and third links unbalanced. -/
theorem pumpkin_second_unbalanced {g : Γ} (hg : g ≠ 1) :
    (pumpkinWalk 0).edgeSet ∪ (pumpkinWalk 2).edgeSet ∉ pumpkin.balancedCycles ![1, 1, g] :=
  fun h => hg ((pumpkin_balanced_iff _ (by decide)).1 h).symm

/-- The gains `(1, 1, g)` with `g ≠ 1` leave the cycle of the last two links unbalanced. -/
theorem pumpkin_third_unbalanced {g : Γ} (hg : g ≠ 1) :
    (pumpkinWalk 1).edgeSet ∪ (pumpkinWalk 2).edgeSet ∉ pumpkin.balancedCycles ![1, 1, g] :=
  fun h => hg ((pumpkin_balanced_iff _ (by decide)).1 h).symm

/-- The source (§2.3, after the definition of `B_γ`) observes that any two closed Eulerian walks in
a cycle have conjugate gains. That is false for a walk and its reversal: around the digon of the
first two links, with gains `g, 1` for `g` of order three in an abelian group, the two gains are
`g` and `g⁻¹`, which are not conjugate. What the source draws from the observation, that `B_γ` is
well defined, is `Walk.gain_eq_one_iff_of_edgeSet_eq`. -/
theorem not_isConj_gain_of_reversed_walk :
    ∃ γ : Fin 3 → Multiplicative (ZMod 3),
      ¬IsConj (((pumpkinWalk 0).append (pumpkinWalk 1).reverse).gain γ)
        (((pumpkinWalk 1).append (pumpkinWalk 0).reverse).gain γ) := by
  refine ⟨![Multiplicative.ofAdd 1, 1, 1], fun h => ?_⟩
  rw [isConj_iff_eq, Walk.gain_append, Walk.gain_append, Walk.gain_reverse _ (pumpkinWalk_isPath 1),
    Walk.gain_reverse _ (pumpkinWalk_isPath 0), pumpkinWalk_gain, pumpkinWalk_gain] at h
  revert h
  decide

example (γ : Fin 3 → Γ) : pumpkin.dart γ 0 0 = γ 0 := dart_of_tail γ rfl

example (γ : Fin 3 → Γ) : pumpkin.dart γ 0 1 = (γ 0)⁻¹ := dart_of_ne γ (by simp [pumpkin])

example (γ : Fin 3 → Γ) : pumpkin.dart γ 0 1 = (pumpkin.dart γ 0 0)⁻¹ :=
  dart_symm γ (Or.inl ⟨rfl, rfl⟩) (by decide)

example (γ : Fin 3 → Γ) (η : Fin 2 → Γ) :
    switch pumpkin η γ 0 = 1 ↔ η 1 = (pumpkin.dart γ 0 0)⁻¹ * η 0 :=
  switch_eq_one_iff η (Or.inl ⟨rfl, rfl⟩)

example (γ : Fin 3 → Γ) : (pumpkinWalk 0).reverse.gain γ = ((pumpkinWalk 0).gain γ)⁻¹ :=
  Walk.gain_reverse γ (pumpkinWalk_isPath 0)

example (η : Fin 2 → Γ) :
    (pumpkinWalk 0).gain (fun _ => η 0 * (η 1)⁻¹) = η 0 * (η 1)⁻¹ :=
  Walk.gain_eq_of_potential η _ fun e _ => by simp [switch_apply, pumpkin]

example (γ : Fin 3 → Γ) (a : Γ) :
    ∃ η : Fin 2 → Γ, η 0 = a ∧ η 1 = ((pumpkinWalk 0).gain γ)⁻¹ * a ∧
      ∀ e ∈ (pumpkinWalk 0).edges, switch pumpkin η γ e = 1 :=
  Walk.exists_potential_of_isPath γ _ (pumpkinWalk_isPath 0) a

example (γ : Fin 3 → Γ) :
    ((pumpkinWalk 0).append (pumpkinWalk 1).reverse).gain γ = 1 ↔
      ∃ η : Fin 2 → Γ, ∀ e ∈ ((pumpkinWalk 0).append (pumpkinWalk 1).reverse).edges,
        switch pumpkin η γ e = 1 :=
  (pumpkin_cycleWalk (by decide)).gain_eq_one_iff γ

example (γ : Fin 3 → Γ) :
    ((pumpkinWalk 0).append (pumpkinWalk 1).reverse).gain γ = 1 ↔
      ((pumpkinWalk 1).append (pumpkinWalk 0).reverse).gain γ = 1 :=
  Walk.gain_eq_one_iff_of_edgeSet_eq γ (pumpkin_cycleWalk (by decide))
    (pumpkin_cycleWalk (by decide))
    (by rw [Walk.edgeSet_append_reverse, Walk.edgeSet_append_reverse, Set.union_comm])

example (γ : Fin 3 → Γ) :
    ((pumpkinWalk 0).append (pumpkinWalk 1).reverse).edgeSet ∈ pumpkin.balancedCycles γ ↔
      ((pumpkinWalk 0).append (pumpkinWalk 1).reverse).gain γ = 1 :=
  (pumpkin_cycleWalk (by decide)).mem_balancedCycles_iff γ

example : IsTheta (pumpkinWalk 1) (pumpkinWalk 0) (pumpkinWalk 2) := pumpkin_isTheta.swap₁₂

example : IsTheta (pumpkinWalk 0) (pumpkinWalk 2) (pumpkinWalk 1) := pumpkin_isTheta.swap₂₃

/-- The theta property applied to equal gains on all three links. -/
example (g : Γ) :
    (pumpkinWalk 1).edgeSet ∪ (pumpkinWalk 2).edgeSet ∈ pumpkin.balancedCycles ![g, g, g] :=
  pumpkin_isTheta.balanced_of_two _ ((pumpkin_balanced_iff _ (by decide)).2 rfl)
    ((pumpkin_balanced_iff _ (by decide)).2 rfl)

/-- The source's wording of the theta property on the biased graph of `(1, 1, g)`: its one balanced
cycle does not contradict it. -/
example (g : Γ) := BiasedGraph.theta_property (pumpkin.gainBiased ![1, 1, g])
  (P := pumpkinWalk 0) (Q := pumpkinWalk 1) (R := pumpkinWalk 2) pumpkin_isTheta

example (γ : Fin 3 → Γ) (η : Fin 2 → Γ) :
    (pumpkin.gainBiased γ).Realizes (switch pumpkin η γ) :=
  BiasedGraph.Realizes.switch (Ω := pumpkin.gainBiased γ) rfl η

example (φ : Fin 3 → ℚ) :
    pumpkin.addBalancedCycles ((2 : ℚ) • φ) = pumpkin.addBalancedCycles φ :=
  addBalancedCycles_smul pumpkin two_ne_zero φ

example (φ : Fin 3 → ℚ) (η : Fin 2 → ℚ) :
    (pumpkin.addGainBiased φ).AddRealizes (addSwitch pumpkin η φ) :=
  BiasedGraph.AddRealizes.addSwitch (Ω := pumpkin.addGainBiased φ) rfl η

/-- The three parallel links connect the two vertices. -/
theorem pumpkin_connected (x y : Fin 2) : Nonempty (pumpkin.Walk x y) := by
  have h2 : ∀ u : Fin 2, u = 0 ∨ u = 1 := by decide
  rcases h2 x with rfl | rfl <;> rcases h2 y with rfl | rfl
  · exact ⟨.nil _⟩
  · exact ⟨.cons 0 (show pumpkin.Joins 0 0 1 from Or.inl ⟨rfl, rfl⟩) (.nil _)⟩
  · exact ⟨.cons 0 (show pumpkin.Joins 0 1 0 from Or.inr ⟨rfl, rfl⟩) (.nil _)⟩
  · exact ⟨.nil _⟩

/-- The first two links of `pumpkin` form a cycle. -/
theorem pumpkin_pair_isCycle : pumpkin.IsCycle ({0, 1} : Set (Fin 3)) :=
  ⟨0, _, pumpkin_cycleWalk (i := 0) (j := 1) (by decide), by
    rw [Walk.edgeSet_append_reverse]
    ext f
    simp [pumpkinWalk, Walk.edgeSet, Walk.edges, or_comm]⟩

/-- That cycle is unbalanced for the additive gain `(1, 0, 0)`. -/
theorem pumpkin_pair_unbalanced :
    ({0, 1} : Set (Fin 3)) ∉ pumpkin.addBalancedCycles (![1, 0, 0] : Fin 3 → ℚ) := fun h => by
  obtain ⟨-, η, hη⟩ := mem_addBalancedCycles_iff.1 h
  have h0 : -η 0 + 1 + η 1 = 0 := hη 0 (by simp)
  have h1 : -η 0 + 0 + η 1 = 0 := hη 1 (by simp)
  exact one_ne_zero (add_left_cancel (add_right_cancel (h0.trans h1.symm)))

end Transformer.GainMatroid
