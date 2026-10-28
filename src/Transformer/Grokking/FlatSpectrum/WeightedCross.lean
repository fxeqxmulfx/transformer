module

public import Transformer.Grokking.FlatSpectrum.CrossCertificate
public import Transformer.Grokking.FlatSpectrum.WeightedMinimum
public import Transformer.Grokking.FlatSpectrum.WeightedWitness

/-!
# Rows that carry too much, and the weighted cross

Source: `review/flat_spectrum.md`, *Exact recovery of a rank-one kernel*, Theorem G, item 3,
and Theorem F, item 4 (added). Write `A = ‖a‖²` and `B = ‖b‖²`. If the rows `R` carry more
than the columns `N(R)` they meet in `Ω` hold, `A ∑_{N(R)} |b_j|² < B ∑_R |a_i|²`, then `Ω`
misses the block `R × N(R)ᶜ`, which breaks the inequality of
`exists_nuclearNorm_lt_of_weighted_block`. So a completion is shorter than `a bᵀ`
(`exists_nuclearNorm_lt_of_nbhd`), and `a bᵀ` is not a minimum
(`not_isMinOn_vecMulVec_of_nbhd`).

On the cross of row `i₀` and column `j₀` the plan is explicit. For masses with `∑ α = ∑ β`,
column `j₀` takes `α i` from each row `i ≠ i₀`, row `i₀` gives `β j` to each column `j ≠ j₀`,
and the corner keeps `α i₀ + β j₀ − ∑ β`. This is a plan when the corner is nonnegative
(`isTransportPlan_cross`). Scaling `a` by `c` and `b` by `c⁻¹`, with `c⁴ = B / A`, balances
`|a_i|²` against `|b_j|²` and keeps `a bᵀ`. So, when no entry vanishes, M recovers `a bᵀ` from
the cross exactly when `A B ≤ |a_{i₀}|² B + A |b_{j₀}|²`, that is
`|a_{i₀}|² / A + |b_{j₀}|² / B ≥ 1` (`forall_isMinOn_vecMulVec_cross_iff`). The plan and
Theorem F give one direction, the block `(univ \ {i₀}) × (univ \ {j₀})` the other. For
`|a_i| = |b_j| = 1` the condition reads `N ≤ 2`, in line with
`vecMulVec_cross_determined_not_isMinOn` for `N ≥ 3`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]

/-- **Rows that carry more than their columns hold have a shorter completion**,
`review/flat_spectrum.md`, Theorem G, item 3: if `‖a‖² ∑_{N(R)} |b_j|² < ‖b‖² ∑_R |a_i|²`,
some `X` equal to `a bᵀ` on `Ω` has `‖X‖_S1 < ‖a bᵀ‖_S1`. -/
theorem exists_nuclearNorm_lt_of_nbhd {a : m → ℂ} {b : n → ℂ} {Ω : Set (m × n)} {R : Finset m}
    (h : (∑ i, ‖a i‖ ^ 2) * ∑ j ∈ nbhd Ω R, ‖b j‖ ^ 2 < (∑ i ∈ R, ‖a i‖ ^ 2) * ∑ j, ‖b j‖ ^ 2) :
    ∃ X : Matrix m n ℂ, (∀ ij ∈ Ω, X ij.1 ij.2 = vecMulVec a b ij.1 ij.2) ∧
      nuclearNorm X < nuclearNorm (vecMulVec a b) := by
  refine exists_nuclearNorm_lt_of_weighted_block (C := (nbhd Ω R)ᶜ)
    (fun i hi j hj hij => mem_compl.mp hj (mem_nbhd.mpr ⟨i, hi, hij⟩)) ?_
  have hB : ∑ j, ‖b j‖ ^ 2 = ∑ j ∈ nbhd Ω R, ‖b j‖ ^ 2 + ∑ j ∈ (nbhd Ω R)ᶜ, ‖b j‖ ^ 2 :=
    (sum_add_sum_compl _ _).symm
  rw [hB] at h ⊢
  linarith

/-- On the diagonal of `Fin 2` the first row meets only the first column. -/
lemma nbhd_diag_zero : nbhd {ij : Fin 2 × Fin 2 | ij.2 = ij.1} {0} = {0} := by
  ext j
  simp [mem_nbhd]

example := exists_nuclearNorm_lt_of_nbhd (a := ![2, 1]) (b := ![1, 1])
  (Ω := {ij | ij.2 = ij.1}) (R := {0}) (by rw [nbhd_diag_zero]; norm_num [Fin.sum_univ_two])

/-- **Without the weighted Hall condition M does not recover `a bᵀ`**,
`review/flat_spectrum.md`, Theorem G, item 3: if `‖a‖² ∑_{N(R)} |b_j|² < ‖b‖² ∑_R |a_i|²`,
then `a bᵀ` is not a minimum of the nuclear norm among the matrices equal to it on `Ω`. -/
theorem not_isMinOn_vecMulVec_of_nbhd {a : m → ℂ} {b : n → ℂ} {Ω : Set (m × n)} {R : Finset m}
    (h : (∑ i, ‖a i‖ ^ 2) * ∑ j ∈ nbhd Ω R, ‖b j‖ ^ 2 < (∑ i ∈ R, ‖a i‖ ^ 2) * ∑ j, ‖b j‖ ^ 2) :
    ¬ IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = vecMulVec a b ij.1 ij.2}
      (vecMulVec a b) := fun hmin => by
  obtain ⟨X, hX, hlt⟩ := exists_nuclearNorm_lt_of_nbhd h
  exact lt_irrefl _ (hlt.trans_le (isMinOn_iff.mp hmin X hX))

example := not_isMinOn_vecMulVec_of_nbhd (a := ![2, 1]) (b := ![1, 1])
  (Ω := {ij | ij.2 = ij.1}) (R := {0}) (by rw [nbhd_diag_zero]; norm_num [Fin.sum_univ_two])

/-- The plan on the cross of `(i₀, j₀)`: column `j₀` takes `α i` from each row `i ≠ i₀`, row
`i₀` gives `β j` to each column `j ≠ j₀`, and the corner keeps `α i₀ + β j₀ − ∑ β`. -/
noncomputable def crossPlan (i₀ j₀ : n) (α β : n → ℝ) (i j : n) : ℝ :=
  if i = i₀ then (if j = j₀ then α i₀ + β j₀ - ∑ k, β k else β j) else if j = j₀ then α i else 0

/-- **The cross carries a plan when its corner is heavy enough**: for nonnegative masses with
`∑ α = ∑ β` and `∑ β ≤ α i₀ + β j₀`, `crossPlan` is a plan on the cross. -/
lemma isTransportPlan_cross {i₀ j₀ : n} {α β : n → ℝ} (hα : ∀ i, 0 ≤ α i) (hβ : ∀ j, 0 ≤ β j)
    (hsum : ∑ i, α i = ∑ j, β j) (h : ∑ j, β j ≤ α i₀ + β j₀) :
    IsTransportPlan {ij | ij.1 = i₀ ∨ ij.2 = j₀} α β (crossPlan i₀ j₀ α β) := by
  refine ⟨fun i j => ?_, fun i j hij => ?_, fun i => ?_, fun j => ?_⟩
  · rw [crossPlan]
    split_ifs
    · linarith
    · exact hβ j
    · exact hα i
    · exact le_rfl
  · rw [crossPlan, ite_eq_right fun h => hij (Or.inl h), ite_eq_right fun h => hij (Or.inr h)]
  · by_cases hi : i = i₀
    · have hrow (j : n) : crossPlan i₀ j₀ α β i j =
          β j + if j = j₀ then α i₀ - ∑ k, β k else 0 := by
        rw [crossPlan, ite_eq_left hi]
        by_cases hj : j = j₀
        · rw [ite_eq_left hj, ite_eq_left hj, hj]
          ring
        · rw [ite_eq_right hj, ite_eq_right hj, add_zero]
      rw [sum_congr rfl fun j _ => hrow j, sum_add_distrib, sum_ite_eq', ite_eq_left (mem_univ _),
        add_sub_cancel, hi]
    · simp [crossPlan, hi]
  · by_cases hj : j = j₀
    · have hcol (i : n) : crossPlan i₀ j₀ α β i j =
          α i + if i = i₀ then β j₀ - ∑ k, β k else 0 := by
        rw [crossPlan]
        by_cases hi : i = i₀
        · rw [ite_eq_left hi, ite_eq_left hi, ite_eq_left hj, hi]
          ring
        · rw [ite_eq_right hi, ite_eq_right hi, ite_eq_left hj, add_zero]
      rw [sum_congr rfl fun i _ => hcol i, sum_add_distrib, sum_ite_eq', ite_eq_left (mem_univ _),
        hsum, hj, add_sub_cancel]
    · simp [crossPlan, hj]

example := isTransportPlan_cross (n := Fin 2) (i₀ := 0) (j₀ := 0) (α := ![2, 1]) (β := ![2, 1])
  (by simp [Fin.forall_fin_two]) (by simp [Fin.forall_fin_two]) rfl
  (by norm_num [Fin.sum_univ_two])

/-- **Theorem F, item 4: the weighted cross**, `review/flat_spectrum.md`: if no entry of `a`,
`b` vanishes, M recovers `a bᵀ` from row `i₀` and column `j₀` exactly when
`‖a‖² ‖b‖² ≤ |a_{i₀}|² ‖b‖² + ‖a‖² |b_{j₀}|²`. -/
theorem forall_isMinOn_vecMulVec_cross_iff {a b : n → ℂ} (ha : ∀ i, a i ≠ 0)
    (hb : ∀ j, b j ≠ 0) (i₀ j₀ : n) :
    (∀ X ∈ {X' : Matrix n n ℂ | ∀ ij ∈ {ij : n × n | ij.1 = i₀ ∨ ij.2 = j₀},
        X' ij.1 ij.2 = vecMulVec a b ij.1 ij.2},
      (IsMinOn nuclearNorm {X' : Matrix n n ℂ | ∀ ij ∈ {ij : n × n | ij.1 = i₀ ∨ ij.2 = j₀},
        X' ij.1 ij.2 = vecMulVec a b ij.1 ij.2} X ↔ X = vecMulVec a b)) ↔
      (∑ i, ‖a i‖ ^ 2) * ∑ j, ‖b j‖ ^ 2 ≤
        ‖a i₀‖ ^ 2 * ∑ j, ‖b j‖ ^ 2 + (∑ i, ‖a i‖ ^ 2) * ‖b j₀‖ ^ 2 := by
  refine ⟨fun H => ?_, fun h X hX => ?_⟩
  · -- A light corner leaves the block off row `i₀` and column `j₀` too heavy.
    by_contra hlt
    obtain ⟨X, hX, hXlt⟩ := exists_nuclearNorm_lt_of_weighted_block (a := a) (b := b)
      (Ω := {ij : n × n | ij.1 = i₀ ∨ ij.2 = j₀}) (R := univ.erase i₀) (C := univ.erase j₀)
      (fun i hi j hj hij => Or.elim hij (ne_of_mem_erase hi) (ne_of_mem_erase hj))
      (by
        rw [sum_erase_eq_sub (mem_univ _), sum_erase_eq_sub (mem_univ _)]
        linarith [not_le.mp hlt])
    exact lt_irrefl _ (hXlt.trans_le (isMinOn_iff.mp ((H _ fun _ _ => rfl).mpr rfl) X hX))
  · set A := ∑ i, ‖a i‖ ^ 2 with hA
    set B := ∑ j, ‖b j‖ ^ 2 with hB
    have hA0 : 0 < A := (pow_pos (norm_pos_iff.mpr (ha i₀)) 2).trans_le
      (single_le_sum (f := fun i => ‖a i‖ ^ 2) (fun i _ => sq_nonneg _) (mem_univ i₀))
    have hB0 : 0 < B := (pow_pos (norm_pos_iff.mpr (hb j₀)) 2).trans_le
      (single_le_sum (f := fun j => ‖b j‖ ^ 2) (fun j _ => sq_nonneg _) (mem_univ j₀))
    -- `c² = q` with `q² A = B` balances the masses.
    set q := √(B / A) with hq
    have hq0 : 0 < q := Real.sqrt_pos.mpr (div_pos hB0 hA0)
    have hq0' := hq0.ne'
    have hqB : q ^ 2 * A = B := by
      rw [hq, Real.sq_sqrt (div_pos hB0 hA0).le, div_mul_cancel₀ B hA0.ne']
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
    have hbal : ∑ i, ‖(c : ℂ) * a i‖ ^ 2 = ∑ j, ‖((c⁻¹ : ℝ) : ℂ) * b j‖ ^ 2 := by
      simp only [hna, hnb, ← mul_sum]
      show q * A = q⁻¹ * B
      rw [← hqB]
      field_simp
    have hcorner : ∑ j, ‖((c⁻¹ : ℝ) : ℂ) * b j‖ ^ 2 ≤
        ‖(c : ℂ) * a i₀‖ ^ 2 + ‖((c⁻¹ : ℝ) : ℂ) * b j₀‖ ^ 2 := by
      simp only [hna, hnb, ← mul_sum]
      show q⁻¹ * B ≤ _
      have h' : q ^ 2 * A ≤ q ^ 2 * ‖a i₀‖ ^ 2 + ‖b j₀‖ ^ 2 :=
        le_of_mul_le_mul_left (by rw [← hqB] at h; linarith) hA0
      calc q⁻¹ * B = q⁻¹ * (q ^ 2 * A) := by rw [hqB]
        _ ≤ q⁻¹ * (q ^ 2 * ‖a i₀‖ ^ 2 + ‖b j₀‖ ^ 2) :=
          mul_le_mul_of_nonneg_left h' (inv_nonneg.mpr hq0.le)
        _ = q * ‖a i₀‖ ^ 2 + q⁻¹ * ‖b j₀‖ ^ 2 := by field_simp
    have hT := isTransportPlan_cross (i₀ := i₀) (j₀ := j₀)
      (α := fun i => ‖(c : ℂ) * a i‖ ^ 2) (β := fun j => ‖((c⁻¹ : ℝ) : ℂ) * b j‖ ^ 2)
      (fun _ => sq_nonneg _) (fun _ => sq_nonneg _) hbal hcorner
    have := isMinOn_vecMulVec_iff_of_transport hT
      (rigid_support_of_ne_zero (fun i => mul_ne_zero hc' (ha i))
        (fun j => mul_ne_zero (Complex.ofReal_ne_zero.mpr (inv_pos.mpr hc0).ne') (hb j))
        fun r s h => rigid_cross h) (X := X) (by rw [hK]; exact hX)
    rwa [hK] at this

/-- `(3, 1, 1)(1, 1, 1)ᵀ` is recovered from its first row and column: `9/11 + 1/3 ≥ 1`. -/
example := (forall_isMinOn_vecMulVec_cross_iff (n := Fin 3) (a := ![3, 1, 1])
  (b := ![1, 1, 1]) (by simp [Fin.forall_fin_succ]) (by simp [Fin.forall_fin_succ]) 0 0).mpr
  (by norm_num [Fin.sum_univ_three])

/-- The all-ones `3 × 3` matrix is not recovered from a cross: `1/3 + 1/3 < 1`. -/
example := mt (forall_isMinOn_vecMulVec_cross_iff (n := Fin 3) (a := fun _ => 1)
  (b := fun _ => 1) (by simp) (by simp) 0 0).mp (by norm_num)

end Transformer.Grokking.FlatSpectrum
