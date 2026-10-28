module

public import Mathlib.Data.Set.Card
public import Transformer.Grokking.FlatSpectrum.SubPlan

/-!
# The supply–demand theorem

Source: `review/flat_spectrum.md`, *Exact recovery of a rank-one kernel*, Theorem F, item 3: the
supply–demand theorem (Gale, 1957). Let `α ≥ 0` be masses on the rows and `β ≥ 0` capacities on
the columns. If every set of rows `R` has `∑_R α ≤ ∑_{N(R)} β`, some `T ≥ 0` that vanishes off
`Ω` has row sums `α` and column sums at most `β` (`exists_isSubPlan`). When the totals agree the
column sums are `β`, so `Ω` carries a transport plan exactly when the totals agree and the
condition holds (`exists_isTransportPlan_iff`).

The proof is an induction on the number of cells of `Ω`. Take a cell `(i, j)` and ship along it
the largest `t ≤ min (α_i, β_j)` that keeps the condition (`exists_ship`,
`sum_le_sum_nbhd_ship`). Either `t` empties row `i` or column `j`, and the cell goes; or a set of
rows `R₀ ∌ i` that meets `j` becomes tight. Then the rows `R₀` on their own cells and the other
rows outside the columns `N(R₀)` are two smaller problems, since neither holds `(i, j)`
(`sum_le_sum_nbhd_inside`, `sum_le_sum_nbhd_outside`, `IsSubPlan.add_split`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset

variable {ι κ : Type*} [Fintype ι] [Fintype κ]

/-- **How much to ship along a cell**: some `t ∈ [0, min (α_i, β_j)]` is at most the slack
`∑_{N(R)} β − ∑_R α` of every set of rows `R` that avoids `i` but meets `j`, and either
`t = α_i`, or `t = β_j`, or `t` is the slack of one such `R₀`. -/
lemma exists_ship [DecidableEq ι] [DecidableEq κ] {Ω : Set (ι × κ)} {α : ι → ℝ} {β : κ → ℝ}
    (hα : ∀ i, 0 ≤ α i) (hβ : ∀ j, 0 ≤ β j)
    (hall : ∀ R : Finset ι, ∑ k ∈ R, α k ≤ ∑ l ∈ nbhd Ω R, β l) (i : ι) (j : κ) :
    ∃ t, 0 ≤ t ∧ t ≤ α i ∧ t ≤ β j ∧
      (∀ R : Finset ι, i ∉ R → j ∈ nbhd Ω R → t ≤ ∑ l ∈ nbhd Ω R, β l - ∑ k ∈ R, α k) ∧
      (t = α i ∨ t = β j ∨ ∃ R₀ : Finset ι, i ∉ R₀ ∧ j ∈ nbhd Ω R₀ ∧
        ∑ l ∈ nbhd Ω R₀, β l - ∑ k ∈ R₀, α k = t) := by
  by_cases hA : ∃ R : Finset ι, i ∉ R ∧ j ∈ nbhd Ω R ∧
      ∑ l ∈ nbhd Ω R, β l - ∑ k ∈ R, α k < min (α i) (β j)
  · -- The least slack among the sets that avoid `i` but meet `j`.
    obtain ⟨R, hiR, hjR, hRm⟩ := hA
    have hF : R ∈ (univ : Finset (Finset ι)).filter fun R => i ∉ R ∧ j ∈ nbhd Ω R :=
      mem_filter.mpr ⟨mem_univ _, hiR, hjR⟩
    obtain ⟨R₀, hR₀, hmin⟩ := ((univ : Finset (Finset ι)).filter
      fun R => i ∉ R ∧ j ∈ nbhd Ω R).exists_min_image
        (fun R => ∑ l ∈ nbhd Ω R, β l - ∑ k ∈ R, α k) ⟨R, hF⟩
    obtain ⟨-, hiR₀, hjR₀⟩ := mem_filter.mp hR₀
    have hle : ∑ l ∈ nbhd Ω R₀, β l - ∑ k ∈ R₀, α k ≤
        ∑ l ∈ nbhd Ω R, β l - ∑ k ∈ R, α k := hmin R hF
    refine ⟨_, sub_nonneg.mpr (hall R₀), by linarith [min_le_left (α i) (β j)],
      by linarith [min_le_right (α i) (β j)],
      fun R' h1 h2 => hmin R' (mem_filter.mpr ⟨mem_univ _, h1, h2⟩),
      Or.inr (Or.inr ⟨R₀, hiR₀, hjR₀, rfl⟩)⟩
  · push Not at hA
    refine ⟨min (α i) (β j), le_min (hα i) (hβ j), min_le_left _ _, min_le_right _ _,
      fun R h1 h2 => hA R h1 h2, ?_⟩
    rcases min_choice (α i) (β j) with h | h
    · exact Or.inl h
    · exact Or.inr (Or.inl h)

example := exists_ship (Ω := {ij : Fin 2 × Fin 2 | ij.2 = ij.1}) (α := ![2, 1]) (β := ![2, 1])
  (by simp [Fin.forall_fin_two]) (by simp [Fin.forall_fin_two]) (fun R => by rw [nbhd_diag]) 0 0

/-- **The supply–demand theorem**, capacity form (Gale, 1957; `review/flat_spectrum.md`,
Theorem F, item 3): if `α, β ≥ 0` and every set of rows `R` has `∑_R α ≤ ∑_{N(R)} β`, some
`T ≥ 0` vanishing off `Ω` has row sums `α` and column sums at most `β`. -/
theorem exists_isSubPlan (Ω : Set (ι × κ)) {α : ι → ℝ} {β : κ → ℝ} (hα : ∀ i, 0 ≤ α i)
    (hβ : ∀ j, 0 ≤ β j) (hall : ∀ R : Finset ι, ∑ i ∈ R, α i ≤ ∑ j ∈ nbhd Ω R, β j) :
    ∃ T, IsSubPlan Ω α β T := by
  classical
  obtain ⟨N, hN⟩ : ∃ N, Ω.ncard = N := ⟨_, rfl⟩
  induction N using Nat.strong_induction_on generalizing Ω α β with
  | _ N ih =>
    rcases Ω.eq_empty_or_nonempty with rfl | ⟨⟨i, j⟩, hij⟩
    · -- Without cells every mass vanishes, and `T = 0` works.
      refine ⟨fun _ _ => 0, fun _ _ => le_rfl, fun _ _ _ => rfl, fun k => ?_, fun l => ?_⟩
      · have h := hall {k}
        rw [sum_singleton, show nbhd (∅ : Set (ι × κ)) {k} = ∅ by ext; simp [mem_nbhd],
          sum_empty] at h
        simp only [sum_const_zero]
        linarith [hα k]
      · simp only [sum_const_zero]
        exact hβ l
    obtain ⟨t, ht0, htα, htβ, hslack, hcase⟩ := exists_ship hα hβ hall i j
    have hα' (k : ι) : 0 ≤ α k - if k = i then t else 0 := by
      split_ifs with hk
      · rw [hk]
        linarith
      · linarith [hα k]
    have hβ' (l : κ) : 0 ≤ β l - if l = j then t else 0 := by
      split_ifs with hl
      · rw [hl]
        linarith
      · linarith [hβ l]
    have hall' := sum_le_sum_nbhd_ship hall hij hslack
    suffices ∃ T, IsSubPlan Ω (fun k => α k - if k = i then t else 0)
        (fun l => β l - if l = j then t else 0) T by
      obtain ⟨T, hT⟩ := this
      exact ⟨_, hT.ship hij ht0⟩
    have hlt (Ω' : Set (ι × κ)) (h : Ω' ⊂ Ω) : Ω'.ncard < N := hN ▸ Set.ncard_lt_ncard h
    rcases hcase with h | h | ⟨R₀, hiR₀, hjR₀, hR₀⟩
    · -- Row `i` is empty: the cell `(i, j)` goes.
      obtain ⟨T, hT⟩ := ih _ (hlt _ (Set.sdiff_singleton_ssubset.mpr hij)) (Ω \ {(i, j)}) hα'
        hβ' (sum_le_sum_nbhd_diff_of_row hβ' hall' (by simp [h]) j) rfl
      exact ⟨T, hT.mono Set.sdiff_subset⟩
    · -- Column `j` is empty: the cell `(i, j)` goes.
      obtain ⟨T, hT⟩ := ih _ (hlt _ (Set.sdiff_singleton_ssubset.mpr hij)) (Ω \ {(i, j)}) hα'
        hβ' (sum_le_sum_nbhd_diff_of_col hβ' hall' (by simp [h]) i) rfl
      exact ⟨T, hT.mono Set.sdiff_subset⟩
    · -- The rows `R₀` are tight: split there.
      have htight : ∑ l ∈ nbhd Ω R₀, (β l - if l = j then t else 0) ≤
          ∑ k ∈ R₀, (α k - if k = i then t else 0) := by
        rw [sum_sub_distrib, sum_sub_distrib, sum_ite_eq', sum_ite_eq', ite_eq_left hjR₀,
          ite_eq_right hiR₀]
        linarith
      obtain ⟨T₁, h₁⟩ := ih _ (hlt {kl ∈ Ω | kl.1 ∈ R₀}
          ((Set.ssubset_iff_of_subset (Set.sep_subset _ _)).mpr
            ⟨(i, j), hij, fun h => hiR₀ h.2⟩)) _
        (fun k => by
          by_cases hk : k ∈ R₀
          · rw [ite_eq_left hk]
            exact hα' k
          · rw [ite_eq_right hk])
        hβ' (sum_le_sum_nbhd_inside hβ' hall' R₀) rfl
      obtain ⟨T₂, h₂⟩ := ih _ (hlt {kl ∈ Ω | kl.2 ∉ nbhd Ω R₀}
          ((Set.ssubset_iff_of_subset (Set.sep_subset _ _)).mpr
            ⟨(i, j), hij, fun h => h.2 hjR₀⟩)) _
        (fun k => by
          by_cases hk : k ∈ R₀
          · rw [ite_eq_left hk]
          · rw [ite_eq_right hk]
            exact hα' k)
        (fun l => by
          by_cases hl : l ∈ nbhd Ω R₀
          · rw [ite_eq_left hl]
          · rw [ite_eq_right hl]
            exact hβ' l)
        (sum_le_sum_nbhd_outside hβ' hall' htight) rfl
      exact ⟨_, IsSubPlan.add_split h₁ h₂⟩

example := exists_isSubPlan {ij : Fin 2 × Fin 2 | ij.2 = ij.1} (α := ![2, 1]) (β := ![2, 1])
  (by simp [Fin.forall_fin_two]) (by simp [Fin.forall_fin_two]) fun R => by rw [nbhd_diag]

/-- **The supply–demand theorem** (Gale, 1957; `review/flat_spectrum.md`, Theorem F, item 3):
for `α, β ≥ 0`, `Ω` carries a transport plan from `α` to `β` if and only if the totals agree and
every set of rows `R` has `∑_R α ≤ ∑_{N(R)} β`. -/
theorem exists_isTransportPlan_iff {Ω : Set (ι × κ)} {α : ι → ℝ} {β : κ → ℝ}
    (hα : ∀ i, 0 ≤ α i) (hβ : ∀ j, 0 ≤ β j) :
    (∃ T, IsTransportPlan Ω α β T) ↔
      ∑ i, α i = ∑ j, β j ∧ ∀ R : Finset ι, ∑ i ∈ R, α i ≤ ∑ j ∈ nbhd Ω R, β j := by
  refine ⟨fun ⟨T, hT⟩ => ⟨hT.sum_eq, hT.sum_le_sum_nbhd⟩, fun ⟨hsum, hall⟩ => ?_⟩
  obtain ⟨T, h0, hΩ, hrow, hcol⟩ := exists_isSubPlan Ω hα hβ hall
  have h := (sum_eq_sum_iff_of_le fun j (_ : j ∈ univ) => hcol j).mp (by
    rw [sum_comm, ← hsum]
    exact sum_congr rfl fun i _ => hrow i)
  exact ⟨T, h0, hΩ, hrow, fun j => h j (mem_univ j)⟩

example := (exists_isTransportPlan_iff (Ω := {ij : Fin 2 × Fin 2 | ij.2 = ij.1})
  (α := ![2, 1]) (β := ![2, 1]) (by simp [Fin.forall_fin_two])
  (by simp [Fin.forall_fin_two])).mpr ⟨rfl, fun R => by rw [nbhd_diag]⟩

end Transformer.Grokking.FlatSpectrum
