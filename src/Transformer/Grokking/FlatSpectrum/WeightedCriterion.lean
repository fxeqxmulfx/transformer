module

public import Transformer.Grokking.FlatSpectrum.Gale
public import Transformer.Grokking.FlatSpectrum.WeightedCross

/-!
# Theorem F: when M recovers a rank-one kernel

Source: `review/flat_spectrum.md`, *Exact recovery of a rank-one kernel*, Theorem F, items 2
and 3. Let no entry of `a` or `b` vanish, and let `A = ‖a‖²` and `B = ‖b‖²`. M recovers `a bᵀ`
from `Ω` exactly when `Ω` is rigid and every set of rows `R` has
`B ∑_R |a_i|² ≤ A ∑_{N(R)} |b_j|²` (`forall_isMinOn_vecMulVec_iff_of_ne_zero`).

* Sufficiency. The supply–demand theorem turns the condition into a plan `T` from `B |a_i|²`
  to `A |b_j|²` (`exists_isTransportPlan_iff`). With `q² A = B` and `c² = q`, the plan
  `(q / B) T` goes from `|c a_i|²` to `|c⁻¹ b_j|²`, and `(c a)(c⁻¹ b)ᵀ = a bᵀ`, so item 2
  applies (`isMinOn_vecMulVec_iff_of_transport_mul`).
* Necessity. Without rigidity item 3's phases give a second completion of the same norm
  (`not_forall_isMinOn_vecMulVec_of_not_rigid`). If the condition fails at `R`, item 3's
  witness is shorter (`not_isMinOn_vecMulVec_of_nbhd`).

The rigid triangle `{(0, 0), (0, 1), (1, 1)}` recovers `(2, 1)(1, 2)ᵀ` but not `(1, 2)(2, 1)ᵀ`:
row 1 meets only column 1, which must take `|a_1|² / ‖a‖²` of the mass.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- **Theorem F, item 2, without the scaling**: if no entry of `a`, `b` vanishes, `Ω` is rigid
and carries a plan from `‖b‖² |a_i|²` to `‖a‖² |b_j|²`, then `a bᵀ` is the unique minimum of the
nuclear norm among the matrices equal to it on `Ω`. -/
theorem isMinOn_vecMulVec_iff_of_transport_mul {a b : n → ℂ} (ha : ∀ i, a i ≠ 0)
    (hb : ∀ j, b j ≠ 0) {Ω : Set (n × n)} {T : n → n → ℝ}
    (hT : IsTransportPlan Ω (fun i => (∑ j, ‖b j‖ ^ 2) * ‖a i‖ ^ 2)
      (fun j => (∑ i, ‖a i‖ ^ 2) * ‖b j‖ ^ 2) T)
    (hΩ : ∀ r s : n → ℂ, (∀ ij ∈ Ω, r ij.1 + s ij.2 = 0) → ∀ i j, r i + s j = 0)
    {X : Matrix n n ℂ} (hX : ∀ ij ∈ Ω, X ij.1 ij.2 = vecMulVec a b ij.1 ij.2) :
    IsMinOn nuclearNorm {X' | ∀ ij ∈ Ω, X' ij.1 ij.2 = vecMulVec a b ij.1 ij.2} X ↔
      X = vecMulVec a b := by
  rcases isEmpty_or_nonempty n with hn | ⟨⟨i₀⟩⟩
  · refine ⟨fun _ => Subsingleton.elim _ _, fun _ => isMinOn_iff.mpr fun X' _ => ?_⟩
    rw [Subsingleton.elim X X']
  set A := ∑ i, ‖a i‖ ^ 2 with hA
  set B := ∑ j, ‖b j‖ ^ 2 with hB
  have hA0 : 0 < A := (pow_pos (norm_pos_iff.mpr (ha i₀)) 2).trans_le
    (single_le_sum (f := fun i => ‖a i‖ ^ 2) (fun i _ => sq_nonneg _) (mem_univ i₀))
  have hB0 : 0 < B := (pow_pos (norm_pos_iff.mpr (hb i₀)) 2).trans_le
    (single_le_sum (f := fun j => ‖b j‖ ^ 2) (fun j _ => sq_nonneg _) (mem_univ i₀))
  have hA0' := hA0.ne'
  have hB0' := hB0.ne'
  -- `c² = q` with `q² A = B` balances the masses.
  set q := √(B / A) with hq
  have hq0 : 0 < q := Real.sqrt_pos.mpr (div_pos hB0 hA0)
  have hq0' := hq0.ne'
  have hqB : q ^ 2 * A = B := by
    rw [hq, Real.sq_sqrt (div_pos hB0 hA0).le, div_mul_cancel₀ B hA0']
  set c := √q with hc
  have hc0 : 0 < c := Real.sqrt_pos.mpr hq0
  have hc2 : c ^ 2 = q := Real.sq_sqrt hq0.le
  have hc' : (c : ℂ) ≠ 0 := Complex.ofReal_ne_zero.mpr hc0.ne'
  have hK : vecMulVec (fun i => (c : ℂ) * a i) (fun j => ((c⁻¹ : ℝ) : ℂ) * b j) =
      vecMulVec a b := by
    ext i j
    simp only [vecMulVec_apply, Complex.ofReal_inv]
    field_simp
  have hna (i : n) : ‖(c : ℂ) * a i‖ ^ 2 = q * ‖a i‖ ^ 2 := by
    rw [norm_mul, Complex.norm_real, Real.norm_of_nonneg hc0.le, mul_pow, hc2]
  have hnb (j : n) : ‖((c⁻¹ : ℝ) : ℂ) * b j‖ ^ 2 = q⁻¹ * ‖b j‖ ^ 2 := by
    rw [norm_mul, Complex.norm_real, Real.norm_of_nonneg (inv_nonneg.mpr hc0.le), mul_pow,
      inv_pow, hc2]
  -- The plan scaled by `q / B`.
  obtain ⟨h0, hΩT, hrow, hcol⟩ := hT
  dsimp only at hrow hcol
  have hT' : IsTransportPlan Ω (fun i => ‖(c : ℂ) * a i‖ ^ 2)
      (fun j => ‖((c⁻¹ : ℝ) : ℂ) * b j‖ ^ 2) fun i j => q / B * T i j := by
    refine ⟨fun i j => mul_nonneg (div_nonneg hq0.le hB0.le) (h0 i j),
      fun i j hij => by dsimp only; rw [hΩT i j hij, mul_zero], fun i => ?_, fun j => ?_⟩
    · dsimp only
      rw [← mul_sum, hrow i, hna]
      field_simp
    · dsimp only
      rw [← mul_sum, hcol j, hnb, ← hqB]
      field_simp
  have := isMinOn_vecMulVec_iff_of_transport (fun i => mul_ne_zero hc' (ha i))
    (fun j => mul_ne_zero (Complex.ofReal_ne_zero.mpr (inv_pos.mpr hc0).ne') (hb j)) hT' hΩ
    (X := X) (by rw [hK]; exact hX)
  rwa [hK] at this

example := (isMinOn_vecMulVec_iff_of_transport_mul (n := Fin 2) (a := ![2, 1]) (b := ![2, 1])
  (by simp [Fin.forall_fin_two]) (by simp [Fin.forall_fin_two]) (Ω := {ij | ij ≠ (1, 0)})
  (isTransportPlan_perm (σ := 1) (Fin.forall_fin_two.mpr ⟨by simp, by simp⟩)
    fun _ => mul_nonneg (sum_nonneg fun _ _ => sq_nonneg _) (sq_nonneg _))
  rigid_ne_one_zero (X := vecMulVec ![2, 1] ![2, 1]) fun _ _ => rfl).mpr rfl

/-- **Theorem F** (`review/flat_spectrum.md`, items 2 and 3): if no entry of `a` or `b`
vanishes, M recovers `a bᵀ` from `Ω` exactly when `Ω` is rigid and every set of rows `R` has
`‖b‖² ∑_R |a_i|² ≤ ‖a‖² ∑_{N(R)} |b_j|²`. -/
theorem forall_isMinOn_vecMulVec_iff_of_ne_zero {a b : n → ℂ} (ha : ∀ i, a i ≠ 0)
    (hb : ∀ j, b j ≠ 0) (Ω : Set (n × n)) :
    (∀ X ∈ {X' : Matrix n n ℂ | ∀ ij ∈ Ω, X' ij.1 ij.2 = vecMulVec a b ij.1 ij.2},
      (IsMinOn nuclearNorm {X' : Matrix n n ℂ | ∀ ij ∈ Ω,
        X' ij.1 ij.2 = vecMulVec a b ij.1 ij.2} X ↔ X = vecMulVec a b)) ↔
      (∀ r s : n → ℂ, (∀ ij ∈ Ω, r ij.1 + s ij.2 = 0) → ∀ i j, r i + s j = 0) ∧
        ∀ R : Finset n, (∑ i ∈ R, ‖a i‖ ^ 2) * ∑ j, ‖b j‖ ^ 2 ≤
          (∑ i, ‖a i‖ ^ 2) * ∑ j ∈ nbhd Ω R, ‖b j‖ ^ 2 := by
  refine ⟨fun H => ⟨fun r s hrs i j => ?_, fun R => ?_⟩, fun ⟨hΩ, hall⟩ X hX => ?_⟩
  · by_contra hne
    exact not_forall_isMinOn_vecMulVec_of_not_rigid hrs hne (ha i) (hb j) H
  · by_contra hlt
    exact not_isMinOn_vecMulVec_of_nbhd (not_le.mp hlt) ((H _ fun _ _ => rfl).mpr rfl)
  · obtain ⟨T, hT⟩ := (exists_isTransportPlan_iff (Ω := Ω)
      (α := fun i => (∑ j, ‖b j‖ ^ 2) * ‖a i‖ ^ 2) (β := fun j => (∑ i, ‖a i‖ ^ 2) * ‖b j‖ ^ 2)
      (fun i => mul_nonneg (sum_nonneg fun j _ => sq_nonneg ‖b j‖) (sq_nonneg ‖a i‖))
      (fun j => mul_nonneg (sum_nonneg fun i _ => sq_nonneg ‖a i‖) (sq_nonneg ‖b j‖))).mpr
      ⟨by rw [← mul_sum, ← mul_sum, mul_comm], fun R => by
        rw [← mul_sum, ← mul_sum, mul_comm]
        exact hall R⟩
    exact isMinOn_vecMulVec_iff_of_transport_mul ha hb hT hΩ hX

/-- **Theorem F, items 2 and 3, with plans**: if no entry of `a` or `b` vanishes, M recovers
`a bᵀ` from `Ω` exactly when `Ω` is rigid and carries a plan from `‖b‖² |a_i|²` to
`‖a‖² |b_j|²`. -/
theorem forall_isMinOn_vecMulVec_iff_exists_transport {a b : n → ℂ} (ha : ∀ i, a i ≠ 0)
    (hb : ∀ j, b j ≠ 0) (Ω : Set (n × n)) :
    (∀ X ∈ {X' : Matrix n n ℂ | ∀ ij ∈ Ω, X' ij.1 ij.2 = vecMulVec a b ij.1 ij.2},
      (IsMinOn nuclearNorm {X' : Matrix n n ℂ | ∀ ij ∈ Ω,
        X' ij.1 ij.2 = vecMulVec a b ij.1 ij.2} X ↔ X = vecMulVec a b)) ↔
      (∀ r s : n → ℂ, (∀ ij ∈ Ω, r ij.1 + s ij.2 = 0) → ∀ i j, r i + s j = 0) ∧
        ∃ T, IsTransportPlan Ω (fun i => (∑ j, ‖b j‖ ^ 2) * ‖a i‖ ^ 2)
          (fun j => (∑ i, ‖a i‖ ^ 2) * ‖b j‖ ^ 2) T := by
  rw [forall_isMinOn_vecMulVec_iff_of_ne_zero ha hb, exists_isTransportPlan_iff
    (fun i => mul_nonneg (sum_nonneg fun j _ => sq_nonneg ‖b j‖) (sq_nonneg ‖a i‖))
    (fun j => mul_nonneg (sum_nonneg fun i _ => sq_nonneg ‖a i‖) (sq_nonneg ‖b j‖))]
  simp only [← mul_sum]
  exact and_congr_right fun _ => ⟨fun h => ⟨mul_comm _ _, fun R => (mul_comm _ _).trans_le (h R)⟩,
    fun h R => (mul_comm _ _).trans_le (h.2 R)⟩

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

end Transformer.Grokking.FlatSpectrum
