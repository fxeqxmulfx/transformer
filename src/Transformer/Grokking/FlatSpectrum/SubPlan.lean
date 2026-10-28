module

public import Transformer.Grokking.FlatSpectrum.SupplyDemand

/-!
# Sub-plans: masses sent within capacities

Source: `review/flat_spectrum.md`, *Exact recovery of a rank-one kernel*, Theorem F, item 3: the
supply–demand theorem (Gale, 1957). Its proof builds a sub-plan: `T ≥ 0`, vanishing off `Ω`,
with row sums `α` and column sums at most `β` (`IsSubPlan`). A transport plan is one
(`IsTransportPlan.isSubPlan`). Two moves of the proof assemble sub-plans: putting back the mass
`t` shipped along a cell (`IsSubPlan.ship`), and adding the sub-plans of the two sides of a set
of rows `R₀`, the rows `R₀` on their own cells and the other rows outside the columns `N(R₀)`
(`IsSubPlan.add_split`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset

variable {ι κ : Type*} [Fintype ι] [Fintype κ]

/-- `T` sends the row masses `α` into `Ω` within the column capacities `β`: it is nonnegative,
vanishes off `Ω`, and has row sums `α` and column sums at most `β`. -/
def IsSubPlan (Ω : Set (ι × κ)) (α : ι → ℝ) (β : κ → ℝ) (T : ι → κ → ℝ) : Prop :=
  (∀ i j, 0 ≤ T i j) ∧ (∀ i j, (i, j) ∉ Ω → T i j = 0) ∧ (∀ i, ∑ j, T i j = α i) ∧
    ∀ j, ∑ i, T i j ≤ β j

/-- A transport plan is a sub-plan. -/
lemma IsTransportPlan.isSubPlan {Ω : Set (ι × κ)} {α : ι → ℝ} {β : κ → ℝ} {T : ι → κ → ℝ}
    (hT : IsTransportPlan Ω α β T) : IsSubPlan Ω α β T :=
  ⟨hT.1, hT.2.1, hT.2.2.1, fun j => (hT.2.2.2 j).le⟩

example := (isTransportPlan_perm (ι := Fin 2) (Ω := Set.univ) (σ := 1)
  (fun _ => Set.mem_univ _) (w := fun _ => 1) fun _ => zero_le_one).isSubPlan

/-- A sub-plan on a smaller pattern is a sub-plan. -/
lemma IsSubPlan.mono {Ω Ω' : Set (ι × κ)} (h : Ω' ⊆ Ω) {α : ι → ℝ} {β : κ → ℝ}
    {T : ι → κ → ℝ} (hT : IsSubPlan Ω' α β T) : IsSubPlan Ω α β T :=
  ⟨hT.1, fun i j hij => hT.2.1 i j fun h' => hij (h h'), hT.2.2⟩

example := (isTransportPlan_perm (ι := Fin 2) (Ω := {ij | ij.2 = ij.1}) (σ := 1)
  (fun _ => rfl) (w := fun _ => 1) fun _ => zero_le_one).isSubPlan.mono (Set.subset_univ _)

/-- Putting back `t ≥ 0` on a cell `(i, j)` of `Ω` turns a sub-plan of the masses left after
shipping `t` into a sub-plan of the original masses. -/
lemma IsSubPlan.ship [DecidableEq ι] [DecidableEq κ] {Ω : Set (ι × κ)} {α : ι → ℝ} {β : κ → ℝ}
    {T : ι → κ → ℝ} {i : ι} {j : κ} (hij : (i, j) ∈ Ω) {t : ℝ} (ht : 0 ≤ t)
    (hT : IsSubPlan Ω (fun k => α k - if k = i then t else 0)
      (fun l => β l - if l = j then t else 0) T) :
    IsSubPlan Ω α β fun k l => T k l + if k = i then (if l = j then t else 0) else 0 := by
  obtain ⟨h0, hΩ, hrow, hcol⟩ := hT
  dsimp only at hrow hcol
  refine ⟨fun k l => add_nonneg (h0 k l) ?_, fun k l hkl => ?_, fun k => ?_, fun l => ?_⟩
  · split_ifs <;> simp [ht]
  · dsimp only
    rw [hΩ k l hkl, zero_add]
    split_ifs with hk hl
    · subst hk hl
      exact absurd hij hkl
    · rfl
    · rfl
  · dsimp only
    rw [sum_add_distrib, hrow k]
    by_cases hk : k = i <;> simp [hk]
  · dsimp only
    rw [sum_add_distrib, sum_ite_eq', ite_eq_left (mem_univ _)]
    linarith [hcol l]

/-- Shipping `1` along the only cell of a `1 × 1` pattern empties it; putting it back gives the
plan `1`. -/
example := IsSubPlan.ship (Ω := Set.univ) (α := fun _ : Fin 1 => (1 : ℝ))
  (β := fun _ : Fin 1 => (1 : ℝ)) (T := fun _ _ => 0) (i := 0) (j := 0) (Set.mem_univ _)
  zero_le_one ⟨fun _ _ => le_rfl, fun _ _ h => absurd (Set.mem_univ _) h,
    fun k => by fin_cases k; simp, fun l => by fin_cases l; simp⟩

/-- Sub-plans on the two sides of a set of rows `R₀`, the rows `R₀` on their own cells and the
other rows outside the columns `N(R₀)`, add up to a sub-plan. -/
lemma IsSubPlan.add_split [DecidableEq ι] [DecidableEq κ] {Ω : Set (ι × κ)} {α : ι → ℝ} {β : κ → ℝ}
    {R₀ : Finset ι} {T₁ T₂ : ι → κ → ℝ}
    (h₁ : IsSubPlan {kl ∈ Ω | kl.1 ∈ R₀} (fun k => if k ∈ R₀ then α k else 0) β T₁)
    (h₂ : IsSubPlan {kl ∈ Ω | kl.2 ∉ nbhd Ω R₀} (fun k => if k ∈ R₀ then 0 else α k)
      (fun l => if l ∈ nbhd Ω R₀ then 0 else β l) T₂) :
    IsSubPlan Ω α β fun k l => T₁ k l + T₂ k l := by
  obtain ⟨h0₁, hΩ₁, hrow₁, hcol₁⟩ := h₁
  obtain ⟨h0₂, hΩ₂, hrow₂, hcol₂⟩ := h₂
  dsimp only at hrow₁ hrow₂ hcol₂
  refine ⟨fun k l => add_nonneg (h0₁ k l) (h0₂ k l), fun k l hkl => ?_, fun k => ?_,
    fun l => ?_⟩
  · dsimp only
    rw [hΩ₁ k l fun h => hkl h.1, hΩ₂ k l fun h => hkl h.1, add_zero]
  · dsimp only
    rw [sum_add_distrib, hrow₁ k, hrow₂ k]
    split_ifs <;> simp
  · dsimp only
    rw [sum_add_distrib]
    have h₂ := hcol₂ l
    by_cases hl : l ∈ nbhd Ω R₀
    · rw [ite_eq_left hl] at h₂
      linarith [hcol₁ l]
    · rw [ite_eq_right hl] at h₂
      have h₁ : ∑ k, T₁ k l = 0 :=
        sum_eq_zero fun k _ => hΩ₁ k l fun h => hl (mem_nbhd.mpr ⟨k, h.2, h.1⟩)
      linarith

/-- With `R₀` all rows of a `1 × 1` pattern, the inner sub-plan carries everything. -/
example := IsSubPlan.add_split (Ω := Set.univ) (α := fun _ : Fin 1 => (1 : ℝ))
  (β := fun _ : Fin 1 => (1 : ℝ)) (R₀ := univ) (T₁ := fun _ _ => 1) (T₂ := fun _ _ => 0)
  ⟨fun _ _ => zero_le_one, fun k _ h => (h (by fin_cases k; simp)).elim,
    fun k => by fin_cases k; simp, fun _ => by simp⟩
  ⟨fun _ _ => le_rfl, fun _ _ _ => rfl, fun k => by fin_cases k; simp, fun _ => by
    simp only [sum_const_zero]
    split_ifs <;> norm_num⟩

end Transformer.Grokking.FlatSpectrum
