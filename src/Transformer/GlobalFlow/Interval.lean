/-
# Global scalar flows between two equilibria

Auxiliary to arXiv:2312.10794v5, §6.2, `eq: ybeta`.
The scalar angle equation has two equilibria bounding its initial value.
Its solution therefore exists for every real time and stays between them.

We clamp the field to that interval, where local Lipschitz regularity gives
one Lipschitz constant. Translating the lower equilibrium to zero puts this
field under `GlobalFlow.exists_global`. The intermediate value theorem and
uniqueness prevent a solution from crossing either equilibrium. The clamp is
then the identity along the solution, which solves the original equation.
-/

import Transformer.GlobalFlow.Existence
import Mathlib.Topology.Order.IntermediateValue

open Set Filter Topology
open scoped NNReal

namespace Transformer
namespace GlobalFlow

/-- A globally Lipschitz scalar flow meeting an equilibrium is constant.

Source: arXiv:2312.10794v5, §6.2, `eq: ybeta`; the uniqueness argument used
to keep its solution between the two equilibria. -/
theorem eq_const_of_hits_equilibrium {F : ℝ → ℝ} {K : ℝ≥0}
    (hF : LipschitzWith K F) {γ : ℝ → ℝ}
    (hγ : ∀ t, HasDerivAt γ (F (γ t)) t) {z s : ℝ}
    (hz : F z = 0) (hs : γ s = z) : γ = fun _ => z := by
  exact ODE_solution_unique_univ (v := fun _ => F) (s := fun _ => univ)
    (fun _ => hF.lipschitzOnWith) (fun t => ⟨hγ t, mem_univ _⟩)
    (fun t => ⟨by simpa only [hz] using hasDerivAt_const t z, mem_univ _⟩) hs

/-- The hypotheses of equilibrium uniqueness hold for the zero field and
its constant zero solution. -/
example : LipschitzWith 0 (fun _ : ℝ => (0 : ℝ)) ∧
    (∀ t : ℝ, HasDerivAt (fun _ : ℝ => (0 : ℝ)) 0 t) ∧
    (fun _ : ℝ => (0 : ℝ)) 0 = 0 :=
  ⟨LipschitzWith.const 0, fun t => hasDerivAt_const t 0, rfl⟩

/-- A scalar flow starting between two equilibria stays between them.

Source: arXiv:2312.10794v5, §6.2, `eq: ybeta`. This applies in both time
directions, using continuity to turn any crossing into a meeting. -/
theorem range_Icc_of_lipschitz_equilibria {F : ℝ → ℝ} {K : ℝ≥0}
    (hF : LipschitzWith K F) {a b : ℝ} (ha : F a = 0) (hb : F b = 0)
    {γ : ℝ → ℝ} (hγ : ∀ t, HasDerivAt γ (F (γ t)) t)
    (h0 : γ 0 ∈ Icc a b) : ∀ t, γ t ∈ Icc a b := by
  have hc : Continuous γ := continuous_iff_continuousAt.2 fun t => (hγ t).continuousAt
  intro t
  constructor
  · by_contra h
    have ht : γ t < a := lt_of_not_ge h
    obtain ⟨s, hs⟩ := intermediate_value_univ t 0 hc ⟨ht.le, h0.1⟩
    have heq := congrFun (eq_const_of_hits_equilibrium hF hγ ha hs) t
    linarith
  · by_contra h
    have ht : b < γ t := lt_of_not_ge h
    obtain ⟨s, hs⟩ := intermediate_value_univ 0 t hc ⟨h0.2, ht.le⟩
    have heq := congrFun (eq_const_of_hits_equilibrium hF hγ hb hs) t
    linarith

/-- The invariant interval can have distinct endpoints and an interior
initial value, as witnessed by the constant zero flow in `[-1, 1]`. -/
example : range (fun _ : ℝ => (0 : ℝ)) ⊆ Icc (-1 : ℝ) 1 := by
  rintro _ ⟨t, rfl⟩
  exact range_Icc_of_lipschitz_equilibria (LipschitzWith.const (0 : ℝ)) rfl rfl
    (fun s => hasDerivAt_const s 0) (by norm_num) t

/-- A flow starting strictly between the equilibria never reaches either.

Source: arXiv:2312.10794v5, §6.2, `eq: ybeta`; its boundary equilibria are
approached only at infinite time. -/
theorem range_Ioo_of_lipschitz_equilibria {F : ℝ → ℝ} {K : ℝ≥0}
    (hF : LipschitzWith K F) {a b : ℝ} (ha : F a = 0) (hb : F b = 0)
    {γ : ℝ → ℝ} (hγ : ∀ t, HasDerivAt γ (F (γ t)) t)
    (h0 : γ 0 ∈ Ioo a b) : ∀ t, γ t ∈ Ioo a b := by
  have hmem := range_Icc_of_lipschitz_equilibria hF ha hb hγ ⟨h0.1.le, h0.2.le⟩
  intro t
  refine ⟨lt_of_le_of_ne (hmem t).1 ?_, lt_of_le_of_ne (hmem t).2 ?_⟩
  · intro ht
    have heq := congrFun (eq_const_of_hits_equilibrium hF hγ ha ht.symm) 0
    linarith [h0.1]
  · intro ht
    have heq := congrFun (eq_const_of_hits_equilibrium hF hγ hb ht) 0
    linarith [h0.2]

/-- The strict interval invariance hypotheses hold for the same interior
zero solution. -/
example : range (fun _ : ℝ => (0 : ℝ)) ⊆ Ioo (-1 : ℝ) 1 := by
  rintro _ ⟨t, rfl⟩
  exact range_Ioo_of_lipschitz_equilibria (LipschitzWith.const (0 : ℝ)) rfl rfl
    (fun s => hasDerivAt_const s 0) (by norm_num) t

/-- A locally Lipschitz scalar field has a global solution from any point
of an interval whose endpoints are equilibria.

Source: arXiv:2312.10794v5, §6.2, `eq: ybeta`. This is a general existence
lemma, rather than an additional hypothesis on the angle equation. -/
theorem exists_global_Icc {F : ℝ → ℝ} (hF : LocallyLipschitz F)
    {a b x₀ : ℝ} (ha : F a = 0) (hb : F b = 0) (hx₀ : x₀ ∈ Icc a b) :
    ∃ γ : ℝ → ℝ, γ 0 = x₀ ∧ ∀ t, HasDerivAt γ (F (γ t)) t ∧ γ t ∈ Icc a b := by
  have hab : a ≤ b := hx₀.1.trans hx₀.2
  let p : ℝ → ℝ := fun x => max a (min x b)
  have hp : LipschitzWith 1 p := (LipschitzWith.id.min_const b).const_max a
  have hpmem : ∀ x, p x ∈ Icc a b := fun x =>
    ⟨le_max_left _ _, max_le hab (min_le_right _ _)⟩
  have hp_eq : ∀ x ∈ Icc a b, p x = x := fun x hx => by
    dsimp only [p]
    rw [min_eq_left hx.2, max_eq_right hx.1]
  obtain ⟨K, hK⟩ := (hF.locallyLipschitzOn (s := Icc a b)).exists_lipschitzOnWith_of_compact
    isCompact_Icc
  let H : ℝ → ℝ := F ∘ p
  have hH : LipschitzWith K H := by
    simpa only [mul_one] using lipschitzOnWith_univ.1
      (hK.comp hp.lipschitzOnWith (fun x _ => hpmem x))
  let G : ℝ → ℝ := fun x => H (x + a)
  have hG : LipschitzWith K G := by
    rw [lipschitzWith_iff_norm_sub_le]
    intro x y
    simpa only [G, add_sub_add_right_eq_sub] using hH.norm_sub_le (x + a) (y + a)
  have hG0 : G 0 = 0 := by
    simp only [G, H, Function.comp_apply, zero_add, hp_eq a ⟨le_rfl, hab⟩, ha]
  obtain ⟨ψ, hψ0, hψ⟩ := exists_global hG.locallyLipschitz K.coe_nonneg
    (fun x => by simpa only [hG0, sub_zero] using hG.norm_sub_le x 0) (x₀ - a)
  let γ : ℝ → ℝ := fun t => ψ t + a
  have hγ0 : γ 0 = x₀ := by dsimp only [γ]; rw [hψ0]; ring
  have hγ : ∀ t, HasDerivAt γ (H (γ t)) t := fun t => (hψ t).add_const a
  have hHa : H a = 0 := by
    simp only [H, Function.comp_apply, hp_eq a ⟨le_rfl, hab⟩, ha]
  have hHb : H b = 0 := by
    simp only [H, Function.comp_apply, hp_eq b ⟨hab, le_rfl⟩, hb]
  have hmem := range_Icc_of_lipschitz_equilibria hH hHa hHb hγ (hγ0 ▸ hx₀)
  refine ⟨γ, hγ0, fun t => ⟨?_, hmem t⟩⟩
  simpa only [H, Function.comp_apply, hp_eq (γ t) (hmem t)] using hγ t

/-- A nonzero polynomial field satisfies the existence hypotheses: the
two-particle angle drift `1 - x²` at `β = 0`, with initial value zero. -/
example : ∃ γ : ℝ → ℝ, γ 0 = 0 ∧
    ∀ t, HasDerivAt γ (1 - γ t ^ 2) t ∧ γ t ∈ Icc (-1 : ℝ) 1 := by
  exact exists_global_Icc
    (contDiff_const.sub (contDiff_id.pow 2) :
      ContDiff ℝ 1 (fun x : ℝ => 1 - x ^ 2)).locallyLipschitz
    (a := -1) (b := 1) (x₀ := 0)
    (by norm_num) (by norm_num) (by norm_num)

end GlobalFlow
end Transformer
