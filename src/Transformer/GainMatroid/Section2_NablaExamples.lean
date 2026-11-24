module

public import Transformer.GainMatroid.Section2_NablaBiased

/-!
# Examples of `∇_Y` on a claw with parallel edges

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, Proposition 2.7. `clawPlus` is a claw at `0`
with the spokes `a = 0`, `b = 1`, `c = 2` to `1`, `2`, `3`, two edges `3`, `4` from `1` to `2` and
two edges `5`, `6` from `2` to `3`; `clawPlusNabla` is its `∇_Y` (`clawPlusNabla_isExchange`). Its
thetas cover the cases of the proof of Proposition 2.7: `Y` on one path through one edge
(`plus_isTheta_single`) or two (`plus_isTheta_pair`), two paths through `b` and `c` at the corner
`1` (`plus_isTheta_corner`), and `Y` as a cycle (`plus_isTheta_triangle`). The examples apply the
lemmas on cycles, paths and detours of `∇_Y G` to them.
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

open scoped symmDiff

/-- A claw at `0` with spokes `0`, `1`, `2` to `1`, `2`, `3`, the edges `3`, `4` from `1` to `2`
and the edges `5`, `6` from `2` to `3`. -/
def clawPlus : OrientedGraph (Fin 4) (Fin 7) := ⟨![0, 0, 0, 1, 1, 2, 2], ![1, 2, 3, 2, 2, 3, 3]⟩

theorem clawPlus_isLabelledClaw : clawPlus.IsLabelledClaw 0 0 1 2 1 2 3 := by
  refine ⟨Or.inl ⟨rfl, rfl⟩, Or.inl ⟨rfl, rfl⟩, Or.inl ⟨rfl, rfl⟩, by decide, by decide, by decide,
    by decide, by decide, by decide, by decide, by decide, by decide, ?_⟩
  simp only [clawPlus]
  decide

/-- `∇_Y` of `clawPlus`: `0` joins `2` to `3`, `1` joins `1` to `3`, `2`, `3`, `4` join `1` to `2`
and `5`, `6` join `2` to `3`. -/
def clawPlusNabla : OrientedGraph (Fin 4) (Fin 7) :=
  ⟨![2, 1, 1, 1, 1, 2, 2], ![3, 3, 2, 2, 2, 3, 3]⟩

theorem clawPlus_nablaGraph : clawPlus.nablaGraph 0 1 2 1 2 3 = clawPlusNabla := by
  refine eq_of_tail_head ?_ ?_ <;> funext e <;> fin_cases e <;>
    simp [nablaGraph, clawPlus, clawPlusNabla]

/-- `clawPlusNabla` and `clawPlus` are an exchange pair on `{0, 1, 2}`, with the star at `0`. -/
theorem clawPlusNabla_isExchange : IsExchange clawPlusNabla clawPlus id 0 0 1 2 1 2 3 :=
  clawPlus_nablaGraph ▸ clawPlus_isLabelledClaw.isExchange

/-- The edge `c = 2` of `clawPlusNabla` from `1` to `2`. -/
def plusC : clawPlusNabla.Walk 1 2 := .cons 2 (Or.inl ⟨rfl, rfl⟩) (.nil 2)

/-- The edge `3` of `clawPlusNabla` from `1` to `2`. -/
def plusE3 : clawPlusNabla.Walk 1 2 := .cons 3 (Or.inl ⟨rfl, rfl⟩) (.nil 2)

/-- The edge `4` of `clawPlusNabla` from `1` to `2`. -/
def plusE4 : clawPlusNabla.Walk 1 2 := .cons 4 (Or.inl ⟨rfl, rfl⟩) (.nil 2)

/-- The path `1, 3, 2` of `clawPlusNabla` through `b = 1` and `5`. -/
def plusB5 : clawPlusNabla.Walk 1 2 :=
  .cons 1 (Or.inl ⟨rfl, rfl⟩) (.cons 5 (Or.inr ⟨rfl, rfl⟩) (.nil 2))

/-- The path `1, 3, 2` of `clawPlusNabla` through `b = 1` and `a = 0`. -/
def plusBA : clawPlusNabla.Walk 1 2 :=
  .cons 1 (Or.inl ⟨rfl, rfl⟩) (.cons 0 (Or.inr ⟨rfl, rfl⟩) (.nil 2))

/-- The path `3, 1, 2` of `clawPlusNabla` through `b = 1` and `c = 2`. -/
def plusBC : clawPlusNabla.Walk 3 2 :=
  .cons 1 (Or.inr ⟨rfl, rfl⟩) (.cons 2 (Or.inl ⟨rfl, rfl⟩) (.nil 2))

/-- The edge `5` of `clawPlusNabla` from `3` to `2`. -/
def plusE5 : clawPlusNabla.Walk 3 2 := .cons 5 (Or.inr ⟨rfl, rfl⟩) (.nil 2)

/-- The edge `6` of `clawPlusNabla` from `3` to `2`. -/
def plusE6 : clawPlusNabla.Walk 3 2 := .cons 6 (Or.inr ⟨rfl, rfl⟩) (.nil 2)

/-- The edge `a = 0` of `clawPlusNabla` from `2` to `3`. -/
def plusA : clawPlusNabla.Walk 2 3 := .cons 0 (Or.inl ⟨rfl, rfl⟩) (.nil 3)

/-- The edge `3` of `clawPlus` from `1` to `2`. -/
def plusS3 : clawPlus.Walk 1 2 := .cons 3 (Or.inl ⟨rfl, rfl⟩) (.nil 2)

/-- The edge `5` of `clawPlus` from `3` to `2`. -/
def plusS5 : clawPlus.Walk 3 2 := .cons 5 (Or.inr ⟨rfl, rfl⟩) (.nil 2)

/-- The path `1, 0, 2` of `clawPlus` through the spokes `a = 0` and `b = 1`: the detour of `c`. -/
def plusSAB : clawPlus.Walk 1 2 :=
  .cons 0 (Or.inr ⟨rfl, rfl⟩) (.cons 1 (Or.inl ⟨rfl, rfl⟩) (.nil 2))

/-- A theta of `clawPlusNabla` with `Y` on one path, through `c`. -/
theorem plus_isTheta_single : IsTheta plusC plusE3 plusE4 := by
  refine ⟨by decide, ?_, ?_, ?_, ⟨by decide, by decide⟩, ⟨by decide, by decide⟩,
    ⟨by decide, by decide⟩⟩ <;> (unfold Walk.IsPath; decide)

/-- A theta of `clawPlusNabla` with `Y` on one path, through `b` and `c`. -/
theorem plus_isTheta_pair : IsTheta plusBC plusE5 plusE6 := by
  refine ⟨by decide, ?_, ?_, ?_, ⟨by decide, by decide⟩, ⟨by decide, by decide⟩,
    ⟨by decide, by decide⟩⟩ <;> (unfold Walk.IsPath; decide)

/-- A theta of `clawPlusNabla` from the corner `1`, through `b` and through `c`. -/
theorem plus_isTheta_corner : IsTheta plusB5 plusC plusE3 := by
  refine ⟨by decide, ?_, ?_, ?_, ⟨by decide, by decide⟩, ⟨by decide, by decide⟩,
    ⟨by decide, by decide⟩⟩ <;> (unfold Walk.IsPath; decide)

/-- A theta of `clawPlusNabla` with the triangle `Y` as a cycle. -/
theorem plus_isTheta_triangle : IsTheta plusC plusBA plusE3 := by
  refine ⟨by decide, ?_, ?_, ?_, ⟨by decide, by decide⟩, ⟨by decide, by decide⟩,
    ⟨by decide, by decide⟩⟩ <;> (unfold Walk.IsPath; decide)

/-- The detour of the edge `c` through the centre of `clawPlus`. -/
theorem plusSAB_isDetour : IsDetour {0, 1, 2} id 0 plusC plusSAB :=
  ⟨by unfold Walk.IsPath; decide, by decide, by simp [plusSAB, Walk.edges]⟩

theorem plusNabla_isCycle₃₄ : clawPlusNabla.IsCycle {3, 4} :=
  isCycle_pair (u := 1) (v := 2) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (by decide) (by decide)

theorem plusNabla_isCycle₂₃ : clawPlusNabla.IsCycle {2, 3} :=
  isCycle_pair (u := 1) (v := 2) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (by decide) (by decide)

theorem plusNabla_isCycle₁₂₅ : clawPlusNabla.IsCycle {1, 2, 5} :=
  isCycle_of_three (x := 3) (y := 1) (z := 2) (Or.inr ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩)
    (Or.inl ⟨rfl, rfl⟩) (by decide) (by decide)

theorem plusNabla_isCycle₁₀₃ : clawPlusNabla.IsCycle {1, 0, 3} :=
  isCycle_of_three (x := 1) (y := 3) (z := 2) (Or.inl ⟨rfl, rfl⟩) (Or.inr ⟨rfl, rfl⟩)
    (Or.inr ⟨rfl, rfl⟩) (by decide) (by decide)

theorem clawPlus_isCycle₃₄ : clawPlus.IsCycle {3, 4} :=
  isCycle_pair (u := 1) (v := 2) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (by decide) (by decide)

example := mem_nablaFamily_iff_of_inter_eq_empty (B := (∅ : Set (Set (Fin 7)))) (a := 0) (b := 1)
  (c := 2) (D := {3, 4}) (by decide) (by decide) (by decide) (by ext g; fin_cases g <;> simp)

example := mem_nablaFamily_iff_of_inter_eq_pair (B := (∅ : Set (Set (Fin 7)))) (a := 0) (b := 1)
  (c := 2) (D := {1, 2, 5}) (by decide) (by decide) (by decide) (by ext g; fin_cases g <;> simp)

example := mem_nablaFamily_iff_of_inter_eq_singleton (B := (∅ : Set (Set (Fin 7)))) (a := 2)
  (b := 0) (c := 1) (D := {2, 3}) (by decide) (by decide) (by decide)
  (by ext g; fin_cases g <;> simp)

example := clawPlusNabla_isExchange.triangle.mem_nablaBalanced (∅ : Set (Set (Fin 7)))

example := clawPlusNabla_isExchange.minimal_of_mem_nablaFamily
  (Ω := clawPlus.gainBiased (1 : Fin 7 → ℚˣ)) plusNabla_isCycle₃₄
  (fun h => by simpa using Set.ext_iff.1 h 3)
  ((mem_nablaFamily_iff_of_inter_eq_empty (by decide) (by decide) (by decide)
    (by ext g; fin_cases g <;> simp)).2 (gainBiased_one_isBalanced clawPlus _ clawPlus_isCycle₃₄))

example := clawPlusNabla_isExchange.mem_nablaBalanced_iff
  (Ω := clawPlus.gainBiased (1 : Fin 7 → ℚˣ)) plusNabla_isCycle₃₄
  (fun h => by simpa using Set.ext_iff.1 h 3)

example := clawPlusNabla_isExchange.mem_nablaBalanced_iff_of_disjoint
  (Ω := clawPlus.gainBiased (1 : Fin 7 → ℚˣ)) plusNabla_isCycle₃₄ (by ext g; fin_cases g <;> simp)

example := clawPlusNabla_isExchange.mem_nablaBalanced_iff_of_singleton
  (Ω := clawPlus.gainBiased (1 : Fin 7 → ℚˣ)) plusNabla_isCycle₂₃ (e := 2) (by simp)
  (by ext g; fin_cases g <;> simp)

example := clawPlusNabla_isExchange.mem_nablaBalanced_iff_of_pair
  (Ω := clawPlus.gainBiased (1 : Fin 7 → ℚˣ)) plusNabla_isCycle₁₂₅ (e₁ := 1) (e₂ := 2) (by simp)
  (by simp) (by decide) (by ext g; fin_cases g <;> simp)

example := clawPlusNabla_isExchange.mem_nablaBalanced_iff_symmDiff
  (Ω := clawPlus.gainBiased (1 : Fin 7 → ℚˣ)) plusNabla_isCycle₂₃
  (by
    rw [show ({2, 3} : Set (Fin 7)) ∆ {0, 1, 2} = {1, 0, 3} by
      ext g; fin_cases g <;> simp [Set.mem_symmDiff]]
    exact plusNabla_isCycle₁₀₃)
  (fun h => by simpa using Set.ext_iff.1 h 3)

example := clawPlusNabla_isExchange.exists_isDetour_of_off (by unfold Walk.IsPath; decide :
  plusE3.IsPath) (by simp [plusE3, Walk.edges])

example := clawPlusNabla_isExchange.openly_of_isDetour (by decide)
  (⟨by decide, by decide⟩ : Walk.Openly plusC plusE3) plusSAB_isDetour (Q' := plusS3)
  (by simp [plusE3, Walk.edges]) rfl rfl

example := clawPlusNabla_isExchange.exists_isTheta_of_isDetour plus_isTheta_single
  (by simp [plusE3, Walk.edges]) (by simp [plusE4, Walk.edges]) plusSAB_isDetour

example := clawPlusNabla_isExchange.exists_isPath_through (A := plusE3) (B := .nil 3) (by decide)
  (by simp [plusE3, Walk.edges]) (by simp [Walk.edges])

example := clawPlusNabla_isExchange.exists_isPath_of_joins (Or.inl ⟨rfl, rfl⟩) (A := plusE3)
  (B := .nil 3) (by decide) (by simp [plusE3, Walk.edges]) (by simp [Walk.edges])

example := clawPlusNabla_isExchange.exists_isDetour_of_inter_eq_singleton
  (by unfold Walk.IsPath; decide : plusA.IsPath)
  (by ext g; fin_cases g <;> simp [plusA, Walk.edgeSet, Walk.edges])

example := clawPlusNabla_isExchange.exists_isDetour_of_cons_cons (A := .nil 3) (B := .nil 2)
  (e₁ := 1) (e₂ := 2) (w := 1) (h₁ := Or.inr ⟨rfl, rfl⟩) (h₂ := Or.inl ⟨rfl, rfl⟩)
  (by unfold Walk.IsPath; decide)
  (by ext g; fin_cases g <;> simp [Walk.append, Walk.edgeSet, Walk.edges])

example := clawPlusNabla_isExchange.exists_isDetour_of_inter_eq_pair
  (by unfold Walk.IsPath; decide : plusBC.IsPath)
  (by ext g; fin_cases g <;> simp [plusBC, Walk.edgeSet, Walk.edges])

end Transformer.GainMatroid
