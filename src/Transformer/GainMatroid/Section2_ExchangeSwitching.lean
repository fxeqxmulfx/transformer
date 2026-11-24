module

public import Transformer.GainMatroid.Section2_ExchangeLabels

/-!
# Switching across an exchange pair

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4. For an exchange pair `T`, `S` on
`X = {a, b, c}` (`IsExchange`), with the triangle `X` in `T` and the star `X` at `v` in `S`, a
switching function moves between the two graphs:

* a set of edges off `X` is balanced in `T` exactly when it is balanced in `S`
  (`IsExchange.mem_balancedCycles_iff_of_disjoint`): switching functions agree on the vertices of
  `T` and the value at `v` is free;
* for a gain function with value one on `X`, a cycle `D` of `S` through `b` and `c` is balanced
  exactly when the cycle `D ∆ X` of `T` is (`IsExchange.mem_balancedCycles_iff_symmDiff`): an edge
  with gain one has gain one after switching exactly when its ends get one value
  (`Joins.switch_eq_one_iff_of_eq_one`), so `b` and `c` force the values at `y`, `v` and `z` to be
  equal, as `a` forces those at `y` and `z`.
-/

@[expose] public section

namespace Transformer.GainMatroid

open scoped symmDiff

namespace OrientedGraph

variable {V W E Γ : Type*} [Group Γ]

/-- An edge of gain one keeps gain one under a switching exactly when its ends get one value. -/
@[to_additive Joins.addSwitch_eq_zero_iff_of_eq_zero
  /-- An edge of gain zero keeps gain zero under a switching exactly when its ends get one
  value. -/]
theorem Joins.switch_eq_one_iff_of_eq_one {G : OrientedGraph V E} {γ : E → Γ} (η : V → Γ)
    {e : E} {s t : V} (h : G.Joins e s t) (hγ : γ e = 1) : switch G η γ e = 1 ↔ η s = η t := by
  rw [switch_apply, hγ, mul_one, inv_mul_eq_one]
  rcases h with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
  · rfl
  · exact eq_comm

namespace IsExchange

variable {T : OrientedGraph V E} {S : OrientedGraph W E} {ι : V → W} {v : W} {a b c : E}
  {x y z : V}

/-- Off `X`, switching `S` by `η'` is switching `T` by `η` when `η'` agrees with `η` on the vertices
of `T` not mapped to `v`. -/
@[to_additive addSwitch_eq /-- Off `X`, switching `S` by `η'` is switching `T` by `η` when `η'`
agrees with `η` on the vertices of `T` not mapped to `v`. -/]
theorem switch_eq (h : IsExchange T S ι v a b c x y z) (γ : E → Γ) {η : V → Γ} {η' : W → Γ}
    (hη : ∀ u, ι u ≠ v → η' (ι u) = η u) {e : E} (he : e ∉ ({a, b, c} : Set E)) :
    switch S η' γ e = switch T η γ e := by
  obtain ⟨h₁, h₂⟩ := h.agree e he
  rw [switch_apply, switch_apply, h₁, h₂, hη _ (h.tail_ne e), hη _ (h.head_ne e)]

variable [DecidableEq V] [DecidableEq W]

/-- A set of edges off `X` is balanced in `T` exactly when it is balanced in `S`. -/
@[to_additive mem_addBalancedCycles_iff_of_disjoint /-- A set of edges off `X` is balanced in `T`
exactly when it is balanced in `S`. -/]
theorem mem_balancedCycles_iff_of_disjoint (h : IsExchange T S ι v a b c x y z) (γ : E → Γ)
    {C : Set E} (hC : Disjoint C {a, b, c}) :
    C ∈ T.balancedCycles γ ↔ C ∈ S.balancedCycles γ := by
  rw [mem_balancedCycles_iff, mem_balancedCycles_iff, h.isCycle_iff_of_disjoint hC]
  refine and_congr_right fun _ => ⟨?_, ?_⟩
  · rintro ⟨η, hη⟩
    refine ⟨Function.extend ι η 1, fun e he => ?_⟩
    rw [h.switch_eq γ (fun u _ => h.inj.extend_apply η 1 u) (Set.disjoint_left.1 hC he)]
    exact hη e he
  · rintro ⟨η', hη'⟩
    refine ⟨η' ∘ ι, fun e he => ?_⟩
    rw [← h.switch_eq γ (η := η' ∘ ι) (fun u _ => rfl) (Set.disjoint_left.1 hC he)]
    exact hη' e he

/-- For a gain function with value one on `X`, a cycle `D` of `S` meeting `X` in `b` and `c` alone
is balanced exactly when the cycle `D ∆ X` of `T` is. -/
@[to_additive mem_addBalancedCycles_iff_symmDiff /-- For an additive gain function with value zero
on `X`, a cycle `D` of `S` meeting `X` in `b` and `c` alone is balanced exactly when the cycle
`D ∆ X` of `T` is. -/]
theorem mem_balancedCycles_iff_symmDiff (h : IsExchange T S ι v a b c x y z) {γ : E → Γ}
    (hγ : ∀ e ∈ ({a, b, c} : Set E), γ e = 1) {D : Set E} (hD : S.IsCycle D)
    (hDX : D ∩ {a, b, c} = {b, c}) :
    D ∈ S.balancedCycles γ ↔ D ∆ {a, b, c} ∈ T.balancedCycles γ := by
  have hb : b ∈ D := ((Set.ext_iff.1 hDX b).2 (Or.inl rfl)).1
  have hc : c ∈ D := ((Set.ext_iff.1 hDX c).2 (Or.inr rfl)).1
  have ha : a ∉ D := fun ha => by
    rcases (Set.ext_iff.1 hDX a).1 ⟨ha, Or.inl rfl⟩ with h' | h'
    · exact h.triangle.ne_ab h'
    · exact h.triangle.ne_ac h'
  have hsa := fun η : V → Γ =>
    h.triangle.joins_a.switch_eq_one_iff_of_eq_one η (hγ a (Or.inl rfl))
  have hsb := fun η' : W → Γ =>
    h.joins_b.switch_eq_one_iff_of_eq_one η' (hγ b (Or.inr (Or.inl rfl)))
  have hsc := fun η' : W → Γ =>
    h.joins_c.switch_eq_one_iff_of_eq_one η' (hγ c (Or.inr (Or.inr rfl)))
  rw [mem_balancedCycles_iff, mem_balancedCycles_iff, and_iff_right hD,
    and_iff_right (h.isCycle_symmDiff hD hDX)]
  constructor
  · rintro ⟨η', hη'⟩
    refine ⟨η' ∘ ι, fun e he => ?_⟩
    rcases Set.mem_symmDiff.1 he with ⟨heD, heX⟩ | ⟨heX, heD⟩
    · rw [← h.switch_eq γ (η := η' ∘ ι) (fun u _ => rfl) heX]
      exact hη' e heD
    · obtain rfl : e = a := by
        rcases heX with rfl | rfl | rfl
        · rfl
        · exact absurd hb heD
        · exact absurd hc heD
      exact (hsa _).2 (((hsb η').1 (hη' b hb)).symm.trans ((hsc η').1 (hη' c hc)))
  · classical
    rintro ⟨η, hη⟩
    have hyz : η y = η z := (hsa η).1 (hη a (Set.mem_symmDiff.2 (Or.inr ⟨Or.inl rfl, ha⟩)))
    let η' : W → Γ := fun w => if w = v then η y else Function.extend ι η 1 w
    have hη₀ : ∀ u, ι u ≠ v → η' (ι u) = η u := fun u hu => by
      simp [η', hu, h.inj.extend_apply]
    refine ⟨η', fun e he => ?_⟩
    by_cases heX : e ∈ ({a, b, c} : Set E)
    · rcases heX with rfl | rfl | rfl
      · exact absurd he ha
      · rw [hsb, hη₀ y (h.ne_of_joins h.triangle.joins_a)]
        simp [η']
      · rw [hsc, hη₀ z (h.ne_of_joins h.triangle.joins_a.symm)]
        simp [η', hyz]
    · rw [h.switch_eq γ hη₀ heX]
      exact hη e (Set.mem_symmDiff.2 (Or.inl ⟨he, heX⟩))

end IsExchange

example := K4_isLabelledTriangle.joins_a.switch_eq_one_iff_of_eq_one (γ := (1 : Fin 6 → ℚˣ)) 1 rfl
example := K4_isLabelledTriangle.joins_a.addSwitch_eq_zero_iff_of_eq_zero (γ := (0 : Fin 6 → ℚ)) 0
  rfl
example := K4_isLabelledTriangle.isExchange.switch_eq (1 : Fin 6 → ℚˣ) (η := 1) (η' := 1)
  (fun _ _ => rfl) (e := 0) (by simp)
example := K4_isLabelledTriangle.isExchange.addSwitch_eq (0 : Fin 6 → ℚ) (η := 0) (η' := 0)
  (fun _ _ => rfl) (e := 0) (by simp)
example := K4_isLabelledTriangle.isExchange.mem_balancedCycles_iff_of_disjoint (1 : Fin 6 → ℚˣ)
  (Set.empty_disjoint _)
example := K4_isLabelledTriangle.isExchange.mem_addBalancedCycles_iff_of_disjoint (0 : Fin 6 → ℚ)
  (Set.empty_disjoint _)
example := K4_isLabelledTriangle.isExchange.mem_balancedCycles_iff_symmDiff
  (γ := (1 : Fin 6 → ℚˣ)) (fun _ _ => rfl)
  (K4_isLabelledTriangle.isExchange.isCycle_of_inter_eq_pair K4_isCycle_1243 K4_inter_1243)
  K4_inter_1243
example := K4_isLabelledTriangle.isExchange.mem_addBalancedCycles_iff_symmDiff
  (γ := (0 : Fin 6 → ℚ)) (fun _ _ => rfl)
  (K4_isLabelledTriangle.isExchange.isCycle_of_inter_eq_pair K4_isCycle_1243 K4_inter_1243)
  K4_inter_1243

end OrientedGraph

end Transformer.GainMatroid
