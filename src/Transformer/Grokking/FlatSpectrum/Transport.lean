module

public import Mathlib.Algebra.BigOperators.Field
public import Mathlib.Algebra.BigOperators.Group.Finset.Piecewise
public import Mathlib.Algebra.BigOperators.Group.Finset.Sigma
public import Mathlib.Algebra.Group.End
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Basic.Real.Basic
public import Mathlib.Tactic.FieldSimp

/-!
# Transport plans on a pattern

Source: `review/flat_spectrum.md`, *Exact recovery of a rank-one kernel*, Theorem F (added).
A transport plan on `Ω ⊆ ι × κ` from row masses `α` to column masses `β` is a nonnegative `T`
that vanishes off `Ω`, with row sums `α` and column sums `β` (`IsTransportPlan`). Its masses
agree in total (`IsTransportPlan.sum_eq`). The rows `R` send all their mass into the columns
`N(R)` they meet in `Ω` (`nbhd`), so `∑_R α ≤ ∑_{N(R)} β`: Hall's weighted condition
(`IsTransportPlan.sum_le_sum_nbhd`). The graph of a permutation inside `Ω` carries a plan
(`isTransportPlan_perm`), and so does the full pattern (`isTransportPlan_univ`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset

variable {ι κ : Type*} [Fintype ι] [Fintype κ]

/-- `T` is a transport plan on `Ω` from the row masses `α` to the column masses `β`: it is
nonnegative, vanishes off `Ω`, and has row sums `α` and column sums `β`. -/
def IsTransportPlan (Ω : Set (ι × κ)) (α : ι → ℝ) (β : κ → ℝ) (T : ι → κ → ℝ) : Prop :=
  (∀ i j, 0 ≤ T i j) ∧ (∀ i j, (i, j) ∉ Ω → T i j = 0) ∧ (∀ i, ∑ j, T i j = α i) ∧
    ∀ j, ∑ i, T i j = β j

/-- The columns that meet the rows `R` in `Ω`. -/
noncomputable def nbhd (Ω : Set (ι × κ)) (R : Finset ι) : Finset κ :=
  open Classical in univ.filter fun j => ∃ i ∈ R, (i, j) ∈ Ω

omit [Fintype ι] in
lemma mem_nbhd {Ω : Set (ι × κ)} {R : Finset ι} {j : κ} :
    j ∈ nbhd Ω R ↔ ∃ i ∈ R, (i, j) ∈ Ω := by
  simp [nbhd]

omit [Fintype ι] in
lemma nbhd_mono {Ω : Set (ι × κ)} {R R' : Finset ι} (h : R ⊆ R') : nbhd Ω R ⊆ nbhd Ω R' :=
  fun j hj => by
    obtain ⟨i, hi, hij⟩ := mem_nbhd.mp hj
    exact mem_nbhd.mpr ⟨i, h hi, hij⟩

example := nbhd_mono (ι := Fin 2) (κ := Fin 2) (Ω := Set.univ) (subset_univ {0})

omit [Fintype ι] in
lemma nbhd_union [DecidableEq ι] [DecidableEq κ] (Ω : Set (ι × κ)) (R R' : Finset ι) :
    nbhd Ω (R ∪ R') = nbhd Ω R ∪ nbhd Ω R' := by
  ext j
  simp only [mem_union, mem_nbhd, or_and_right, exists_or]

/-- The two masses of a transport plan have the same total. -/
lemma IsTransportPlan.sum_eq {Ω : Set (ι × κ)} {α : ι → ℝ} {β : κ → ℝ} {T : ι → κ → ℝ}
    (hT : IsTransportPlan Ω α β T) : ∑ i, α i = ∑ j, β j :=
  calc ∑ i, α i = ∑ i, ∑ j, T i j := sum_congr rfl fun i _ => (hT.2.2.1 i).symm
    _ = ∑ j, ∑ i, T i j := sum_comm
    _ = ∑ j, β j := sum_congr rfl fun j _ => hT.2.2.2 j

/-- **A plan meets Hall's weighted condition**, `review/flat_spectrum.md`, Theorem F: the rows
`R` send their mass into the columns `N(R)`, so `∑_R α ≤ ∑_{N(R)} β`. -/
theorem IsTransportPlan.sum_le_sum_nbhd {Ω : Set (ι × κ)} {α : ι → ℝ} {β : κ → ℝ}
    {T : ι → κ → ℝ} (hT : IsTransportPlan Ω α β T) (R : Finset ι) :
    ∑ i ∈ R, α i ≤ ∑ j ∈ nbhd Ω R, β j := by
  obtain ⟨h0, hΩ, hrow, hcol⟩ := hT
  calc ∑ i ∈ R, α i = ∑ i ∈ R, ∑ j ∈ nbhd Ω R, T i j := by
        refine sum_congr rfl fun i hi => ?_
        rw [← hrow i]
        exact (sum_subset (subset_univ _) fun j _ hj =>
          hΩ i j fun h => hj (mem_nbhd.mpr ⟨i, hi, h⟩)).symm
    _ = ∑ j ∈ nbhd Ω R, ∑ i ∈ R, T i j := sum_comm
    _ ≤ ∑ j ∈ nbhd Ω R, β j := sum_le_sum fun j _ => by
        rw [← hcol j]
        exact sum_le_sum_of_subset_of_nonneg (subset_univ _) fun i _ _ => h0 i j

/-- The graph of a permutation `σ` inside `Ω` carries the mass `w_i` from row `i` to column
`σ i`: a plan from `w` to `w ∘ σ⁻¹`. -/
lemma isTransportPlan_perm [DecidableEq ι] {Ω : Set (ι × ι)} {σ : Equiv.Perm ι}
    (hσ : ∀ i, (i, σ i) ∈ Ω) {w : ι → ℝ} (hw : ∀ i, 0 ≤ w i) :
    IsTransportPlan Ω w (fun j => w (σ.symm j)) fun i j => if j = σ i then w i else 0 := by
  refine ⟨fun i j => by dsimp only; split_ifs <;> simp [hw i],
    fun i j hij => ite_eq_right fun (h : j = σ i) => hij (h ▸ hσ i), fun i => by simp,
    fun j => ?_⟩
  rw [← Equiv.sum_comp σ.symm]
  simp

example := isTransportPlan_perm (ι := Fin 2) (Ω := Set.univ) (σ := 1) (fun _ => Set.mem_univ _)
  (w := fun _ => 1) fun _ => zero_le_one

/-- On the full pattern, `α_i β_j / ∑ β` is a plan when the masses agree in total. -/
lemma isTransportPlan_univ {α : ι → ℝ} {β : κ → ℝ} (hα : ∀ i, 0 ≤ α i) (hβ : ∀ j, 0 ≤ β j)
    (hsum : ∑ i, α i = ∑ j, β j) (hpos : 0 < ∑ j, β j) :
    IsTransportPlan Set.univ α β fun i j => α i * β j / ∑ k, β k := by
  refine ⟨fun i j => div_nonneg (mul_nonneg (hα i) (hβ j)) hpos.le,
    fun i j h => absurd (Set.mem_univ _) h, fun i => ?_, fun j => ?_⟩
  · rw [← sum_div, ← mul_sum, mul_div_assoc, div_self hpos.ne', mul_one]
  · rw [← sum_div, ← sum_mul, hsum]
    field_simp

example := isTransportPlan_univ (ι := Fin 2) (κ := Fin 2) (α := fun _ => 1) (β := fun _ => 1)
  (fun _ => zero_le_one) (fun _ => zero_le_one) rfl (by simp)

example := (isTransportPlan_univ (ι := Fin 2) (κ := Fin 2) (α := fun _ => 1) (β := fun _ => 1)
  (fun _ => zero_le_one) (fun _ => zero_le_one) rfl (by simp)).sum_le_sum_nbhd {0}

example := (isTransportPlan_univ (ι := Fin 2) (κ := Fin 2) (α := fun _ => 1) (β := fun _ => 1)
  (fun _ => zero_le_one) (fun _ => zero_le_one) rfl (by simp)).sum_eq

end Transformer.Grokking.FlatSpectrum
