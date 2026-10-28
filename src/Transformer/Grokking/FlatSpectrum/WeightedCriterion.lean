module

public import Transformer.Grokking.FlatSpectrum.Gale
public import Transformer.Grokking.FlatSpectrum.WeightedCross

/-!
# Theorems F and G: when M recovers a rank-one kernel

Source: `review/flat_spectrum.md`, *Exact recovery of a rank-one kernel*, Theorem G, and
Theorem F, items 2 and 3. Let `a ∈ ℂ^m` and `b ∈ ℂ^n`, `A = ‖a‖²` and `B = ‖b‖²`. M recovers
`a bᵀ` from `Ω` exactly when `Ω` is rigid on the supports of `a` and `b` and every set of rows
`R` has `B ∑_R |a_i|² ≤ A ∑_{N(R)} |b_j|²` (`forall_isMinOn_vecMulVec_iff_support`).

* Sufficiency. The supply–demand theorem turns the condition into a plan `T` from `B |a_i|²`
  to `A |b_j|²` (`exists_isTransportPlan_iff`), and item 2 applies
  (`isMinOn_vecMulVec_iff_of_transport_mul`).
* Necessity. Without rigidity on the supports item 3's phases give a second completion of the
  same norm (`not_forall_isMinOn_vecMulVec_of_not_rigid`). If the condition fails at `R`,
  item 3's witness is shorter (`not_isMinOn_vecMulVec_of_nbhd`).

So M recovers `a bᵀ` exactly when `Ω` is rigid on the supports and carries a plan from
`B |a_i|²` to `A |b_j|²` (`forall_isMinOn_vecMulVec_iff_exists_transport_support`). Without
zero entries, rigidity on the supports is rigidity (`rigid_support_iff_of_ne_zero`), and these
are Theorem F (`forall_isMinOn_vecMulVec_iff_of_ne_zero`,
`forall_isMinOn_vecMulVec_iff_exists_transport`).

The rigid triangle `{(0, 0), (0, 1), (1, 1)}` recovers `(2, 1)(1, 2)ᵀ` but not `(1, 2)(2, 1)ᵀ`:
row 1 meets only column 1, which must take `|a_1|² / ‖a‖²` of the mass. In the `1 × 2` table the
cell `(0, 0)` recovers `(1)(1, 0)ᵀ` but not `(1)(1, 1)ᵀ`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]

/-- **Theorem G** (`review/flat_spectrum.md`): M recovers `a bᵀ` from `Ω` exactly when `Ω` is
rigid on the supports of `a` and `b` and every set of rows `R` has
`‖b‖² ∑_R |a_i|² ≤ ‖a‖² ∑_{N(R)} |b_j|²`. -/
theorem forall_isMinOn_vecMulVec_iff_support (a : m → ℂ) (b : n → ℂ) (Ω : Set (m × n)) :
    (∀ X ∈ {X' : Matrix m n ℂ | ∀ ij ∈ Ω, X' ij.1 ij.2 = vecMulVec a b ij.1 ij.2},
      (IsMinOn nuclearNorm {X' : Matrix m n ℂ | ∀ ij ∈ Ω,
        X' ij.1 ij.2 = vecMulVec a b ij.1 ij.2} X ↔ X = vecMulVec a b)) ↔
      (∀ (r : m → ℂ) (s : n → ℂ),
        (∀ ij ∈ Ω, a ij.1 ≠ 0 → b ij.2 ≠ 0 → r ij.1 + s ij.2 = 0) →
          ∀ i j, a i ≠ 0 → b j ≠ 0 → r i + s j = 0) ∧
        ∀ R : Finset m, (∑ i ∈ R, ‖a i‖ ^ 2) * ∑ j, ‖b j‖ ^ 2 ≤
          (∑ i, ‖a i‖ ^ 2) * ∑ j ∈ nbhd Ω R, ‖b j‖ ^ 2 := by
  refine ⟨fun H => ⟨fun r s hrs i j hi hj => ?_, fun R => ?_⟩, fun ⟨hΩ, hall⟩ X hX => ?_⟩
  · by_contra hne
    exact not_forall_isMinOn_vecMulVec_of_not_rigid hrs hne hi hj H
  · by_contra hlt
    exact not_isMinOn_vecMulVec_of_nbhd (not_le.mp hlt) ((H _ fun _ _ => rfl).mpr rfl)
  · obtain ⟨T, hT⟩ := (exists_isTransportPlan_iff (Ω := Ω)
      (α := fun i => (∑ j, ‖b j‖ ^ 2) * ‖a i‖ ^ 2) (β := fun j => (∑ i, ‖a i‖ ^ 2) * ‖b j‖ ^ 2)
      (fun i => mul_nonneg (sum_nonneg fun j _ => sq_nonneg ‖b j‖) (sq_nonneg ‖a i‖))
      (fun j => mul_nonneg (sum_nonneg fun i _ => sq_nonneg ‖a i‖) (sq_nonneg ‖b j‖))).mpr
      ⟨by rw [← mul_sum, ← mul_sum, mul_comm], fun R => by
        rw [← mul_sum, ← mul_sum, mul_comm]
        exact hall R⟩
    exact isMinOn_vecMulVec_iff_of_transport_mul hT hΩ hX

/-- **Theorem G with plans**: M recovers `a bᵀ` from `Ω` exactly when `Ω` is rigid on the
supports of `a` and `b` and carries a plan from `‖b‖² |a_i|²` to `‖a‖² |b_j|²`. -/
theorem forall_isMinOn_vecMulVec_iff_exists_transport_support (a : m → ℂ) (b : n → ℂ)
    (Ω : Set (m × n)) :
    (∀ X ∈ {X' : Matrix m n ℂ | ∀ ij ∈ Ω, X' ij.1 ij.2 = vecMulVec a b ij.1 ij.2},
      (IsMinOn nuclearNorm {X' : Matrix m n ℂ | ∀ ij ∈ Ω,
        X' ij.1 ij.2 = vecMulVec a b ij.1 ij.2} X ↔ X = vecMulVec a b)) ↔
      (∀ (r : m → ℂ) (s : n → ℂ),
        (∀ ij ∈ Ω, a ij.1 ≠ 0 → b ij.2 ≠ 0 → r ij.1 + s ij.2 = 0) →
          ∀ i j, a i ≠ 0 → b j ≠ 0 → r i + s j = 0) ∧
        ∃ T, IsTransportPlan Ω (fun i => (∑ j, ‖b j‖ ^ 2) * ‖a i‖ ^ 2)
          (fun j => (∑ i, ‖a i‖ ^ 2) * ‖b j‖ ^ 2) T := by
  rw [forall_isMinOn_vecMulVec_iff_support, exists_isTransportPlan_iff
    (fun i => mul_nonneg (sum_nonneg fun j _ => sq_nonneg ‖b j‖) (sq_nonneg ‖a i‖))
    (fun j => mul_nonneg (sum_nonneg fun i _ => sq_nonneg ‖a i‖) (sq_nonneg ‖b j‖))]
  simp only [← mul_sum]
  exact and_congr_right fun _ => ⟨fun h => ⟨mul_comm _ _, fun R => (mul_comm _ _).trans_le (h R)⟩,
    fun h R => (mul_comm _ _).trans_le (h.2 R)⟩

omit [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n] in
/-- Without zero entries, rigidity on the supports is rigidity. -/
lemma rigid_support_iff_of_ne_zero {a : m → ℂ} {b : n → ℂ} (ha : ∀ i, a i ≠ 0)
    (hb : ∀ j, b j ≠ 0) (Ω : Set (m × n)) :
    (∀ (r : m → ℂ) (s : n → ℂ), (∀ ij ∈ Ω, a ij.1 ≠ 0 → b ij.2 ≠ 0 → r ij.1 + s ij.2 = 0) →
      ∀ i j, a i ≠ 0 → b j ≠ 0 → r i + s j = 0) ↔
      ∀ (r : m → ℂ) (s : n → ℂ), (∀ ij ∈ Ω, r ij.1 + s ij.2 = 0) → ∀ i j, r i + s j = 0 :=
  ⟨fun h r s hrs i j => h r s (fun ij hij _ _ => hrs ij hij) i j (ha i) (hb j),
    rigid_support_of_ne_zero ha hb⟩

example := rigid_support_iff_of_ne_zero (a := ![2, 1]) (b := ![1, 2])
  (by simp [Fin.forall_fin_two]) (by simp [Fin.forall_fin_two]) {ij | ij ≠ (1, 0)}

/-- **Theorem F** (`review/flat_spectrum.md`, items 2 and 3): if no entry of `a` or `b`
vanishes, M recovers `a bᵀ` from `Ω` exactly when `Ω` is rigid and every set of rows `R` has
`‖b‖² ∑_R |a_i|² ≤ ‖a‖² ∑_{N(R)} |b_j|²`. -/
theorem forall_isMinOn_vecMulVec_iff_of_ne_zero {a : m → ℂ} {b : n → ℂ} (ha : ∀ i, a i ≠ 0)
    (hb : ∀ j, b j ≠ 0) (Ω : Set (m × n)) :
    (∀ X ∈ {X' : Matrix m n ℂ | ∀ ij ∈ Ω, X' ij.1 ij.2 = vecMulVec a b ij.1 ij.2},
      (IsMinOn nuclearNorm {X' : Matrix m n ℂ | ∀ ij ∈ Ω,
        X' ij.1 ij.2 = vecMulVec a b ij.1 ij.2} X ↔ X = vecMulVec a b)) ↔
      (∀ (r : m → ℂ) (s : n → ℂ), (∀ ij ∈ Ω, r ij.1 + s ij.2 = 0) → ∀ i j, r i + s j = 0) ∧
        ∀ R : Finset m, (∑ i ∈ R, ‖a i‖ ^ 2) * ∑ j, ‖b j‖ ^ 2 ≤
          (∑ i, ‖a i‖ ^ 2) * ∑ j ∈ nbhd Ω R, ‖b j‖ ^ 2 := by
  rw [forall_isMinOn_vecMulVec_iff_support, rigid_support_iff_of_ne_zero ha hb]

/-- **Theorem F, items 2 and 3, with plans**: if no entry of `a` or `b` vanishes, M recovers
`a bᵀ` from `Ω` exactly when `Ω` is rigid and carries a plan from `‖b‖² |a_i|²` to
`‖a‖² |b_j|²`. -/
theorem forall_isMinOn_vecMulVec_iff_exists_transport {a : m → ℂ} {b : n → ℂ}
    (ha : ∀ i, a i ≠ 0) (hb : ∀ j, b j ≠ 0) (Ω : Set (m × n)) :
    (∀ X ∈ {X' : Matrix m n ℂ | ∀ ij ∈ Ω, X' ij.1 ij.2 = vecMulVec a b ij.1 ij.2},
      (IsMinOn nuclearNorm {X' : Matrix m n ℂ | ∀ ij ∈ Ω,
        X' ij.1 ij.2 = vecMulVec a b ij.1 ij.2} X ↔ X = vecMulVec a b)) ↔
      (∀ (r : m → ℂ) (s : n → ℂ), (∀ ij ∈ Ω, r ij.1 + s ij.2 = 0) → ∀ i j, r i + s j = 0) ∧
        ∃ T, IsTransportPlan Ω (fun i => (∑ j, ‖b j‖ ^ 2) * ‖a i‖ ^ 2)
          (fun j => (∑ i, ‖a i‖ ^ 2) * ‖b j‖ ^ 2) T := by
  rw [forall_isMinOn_vecMulVec_iff_exists_transport_support, rigid_support_iff_of_ne_zero ha hb]

/-- The rigid triangle recovers `(2, 1)(1, 2)ᵀ` through the plan `[[5, 15], [0, 5]]` from
`5 (4, 1)` to `5 (1, 4)`. -/
example := (forall_isMinOn_vecMulVec_iff_exists_transport (n := Fin 2) (a := ![2, 1])
  (b := ![1, 2]) (by simp [Fin.forall_fin_two]) (by simp [Fin.forall_fin_two])
  {ij | ij ≠ (1, 0)}).mpr ⟨rigid_ne_one_zero, ![![5, 15], ![0, 5]], by
    refine ⟨?_, ?_, ?_, ?_⟩ <;> norm_num [Fin.forall_fin_two, Fin.sum_univ_two]⟩

/-- In the rigid triangle `{(0, 0), (0, 1), (1, 1)}`, row 0 meets both columns and row 1 only
column 1. -/
lemma nbhd_ne_one_zero (R : Finset (Fin 2)) :
    nbhd {ij : Fin 2 × Fin 2 | ij ≠ (1, 0)} R = if (0 : Fin 2) ∈ R then univ else R := by
  ext j
  rw [mem_nbhd]
  split_ifs with h0
  · exact iff_of_true ⟨0, h0, fun h => zero_ne_one (congrArg Prod.fst h)⟩ (mem_univ j)
  · constructor
    · rintro ⟨i, hi, hij⟩
      fin_cases i
      · exact absurd hi h0
      · fin_cases j
        · exact (hij rfl).elim
        · exact hi
    · intro hj
      refine ⟨j, hj, fun h => h0 ?_⟩
      rwa [← show j = 0 from congrArg Prod.snd h]

/-- The rigid triangle recovers `(2, 1)(1, 2)ᵀ`: row 1 carries `1/5` of the mass, and column 1
takes `4/5`. -/
example := (forall_isMinOn_vecMulVec_iff_of_ne_zero (n := Fin 2) (a := ![2, 1]) (b := ![1, 2])
  (by simp [Fin.forall_fin_two]) (by simp [Fin.forall_fin_two]) {ij | ij ≠ (1, 0)}).mpr
  ⟨rigid_ne_one_zero, fun R => by
    rw [nbhd_ne_one_zero]
    split_ifs with h0
    · exact mul_le_mul_of_nonneg_right
        (sum_le_sum_of_subset_of_nonneg (subset_univ R) fun _ _ _ => sq_nonneg _)
        (sum_nonneg fun _ _ => sq_nonneg _)
    · have h : ∀ k ∈ R, ‖(![2, 1] : Fin 2 → ℂ) k‖ ^ 2 ≤ ‖(![1, 2] : Fin 2 → ℂ) k‖ ^ 2 :=
        fun k hk => by
          fin_cases k
          · exact absurd hk h0
          · norm_num
      have := sum_le_sum h
      norm_num [Fin.sum_univ_two] at this ⊢
      linarith⟩

/-- The rigid triangle does not recover `(1, 2)(2, 1)ᵀ`: row 1 carries `4/5` of the mass, and
column 1 takes only `1/5`. -/
example := mt (forall_isMinOn_vecMulVec_iff_of_ne_zero (n := Fin 2) (a := ![1, 2])
  (b := ![2, 1]) (by simp [Fin.forall_fin_two]) (by simp [Fin.forall_fin_two])
  {ij | ij ≠ (1, 0)}).mp fun h => absurd (h.2 {1}) (by
    rw [nbhd_ne_one_zero]
    norm_num [Fin.sum_univ_two])

/-- In the `1 × 2` table the cell `(0, 0)` recovers `(1)(1, 0)ᵀ`: the plan `[[1, 0]]` carries
the masses, and the supports meet in that cell. -/
example := (forall_isMinOn_vecMulVec_iff_exists_transport_support ![(1 : ℂ)] ![1, 0]
  {((0 : Fin 1), (0 : Fin 2))}).mpr ⟨fun r s h i j _ hj => by
    fin_cases i
    fin_cases j
    · exact h _ rfl (by simp) (by simp)
    · simp at hj, fun _ j => ![1, 0] j, by simp [Fin.forall_fin_two],
    by simp [Fin.forall_fin_two], by simp [Fin.sum_univ_two], by simp [Fin.forall_fin_two]⟩

/-- In the `1 × 2` table the cell `(0, 0)` does not recover `(1)(1, 1)ᵀ`: its row carries the
mass `‖b‖² = 2`, and the one column it meets takes only `‖a‖² |b_0|² = 1`. -/
example := mt (forall_isMinOn_vecMulVec_iff_support ![(1 : ℂ)] ![1, 1]
  {((0 : Fin 1), (0 : Fin 2))}).mp fun h => absurd (h.2 {0}) (by
    have hN : nbhd {((0 : Fin 1), (0 : Fin 2))} {0} = {0} := by
      ext j
      fin_cases j <;> simp [mem_nbhd]
    rw [hN]
    norm_num [Fin.sum_univ_two])

end Transformer.Grokking.FlatSpectrum
