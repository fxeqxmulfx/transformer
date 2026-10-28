module

public import Transformer.Grokking.FlatSpectrum.OnesRecovery

/-!
# A missed block bounds every certificate

Source: `review/flat_spectrum.md`, *Exact recovery of a rank-one kernel*, *Patterns* (added).
`isMinOn_onesMatrix_iff` asks for `Y` vanishing off `Ω` with unit row and column sums and
`c (Y − Q)` a contraction, `c > 1`. If `Ω` misses `R × C`, `k = #R`, `l = #C`, the centered
indicators `u = 1_R − k/N`, `v = 1_C − l/N` give `∑ u_i ((Y − Q) v)_i = −k l/N`
(`sum_centeredIndicator_mul_mulVec`) and `‖u‖² = k (N − k)/N` (`sum_sq_centeredIndicator`), so
Cauchy–Schwarz gives `c² k l ≤ (N − k)(N − l)` (`sq_mul_le_of_block`). Hence no `c > 1` works when
`k, l > 0` and `k + l ≥ N` (`not_isContraction_of_block`), e.g. for a row with a single cell in `Ω`
(`not_isContraction_of_row`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset
open scoped ComplexOrder

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- A contraction shortens every vector. -/
lemma IsContraction.sum_norm_sq_mulVec_le {Z : Matrix n n ℂ} (hZ : IsContraction Z)
    (v : n → ℂ) : ∑ i, ‖(Z *ᵥ v) i‖ ^ 2 ≤ ∑ j, ‖v j‖ ^ 2 := by
  have hsq (w : n → ℂ) : star w ⬝ᵥ w = ((∑ i, ‖w i‖ ^ 2 : ℝ) : ℂ) := by
    simp only [dotProduct, Pi.star_apply, Complex.star_def, Complex.conj_mul',
      Complex.ofReal_sum, Complex.ofReal_pow]
  have h := (posSemidef_iff_dotProduct_mulVec.mp hZ).2 v
  rwa [sub_mulVec, one_mulVec, dotProduct_sub, ← mulVec_mulVec, dotProduct_mulVec,
    ← star_mulVec, hsq, hsq, ← Complex.ofReal_sub, Complex.zero_le_real, sub_nonneg] at h

example := (isContraction_one (n := Fin 2)).sum_norm_sq_mulVec_le ![1, 0]

/-- The centered indicator `1_R − #R/N` of `R`. -/
noncomputable def centeredIndicator (R : Finset n) (i : n) : ℝ :=
  (if i ∈ R then 1 else 0) - (#R : ℝ) / Fintype.card n

/-- The centered indicator sums to zero. -/
lemma sum_centeredIndicator (R : Finset n) (hN : (Fintype.card n : ℝ) ≠ 0) :
    ∑ i, centeredIndicator R i = 0 := by
  simp only [centeredIndicator, sum_sub_distrib, sum_boole, filter_mem_eq_inter, univ_inter,
    sum_const, card_univ, nsmul_eq_mul]
  field_simp
  ring

example := sum_centeredIndicator (n := Fin 2) {0} (by simp)

/-- The centered indicator of `R` has squared length `#R (N − #R)/N`. -/
lemma sum_sq_centeredIndicator (R : Finset n) (hN : (Fintype.card n : ℝ) ≠ 0) :
    ∑ i, centeredIndicator R i ^ 2 = #R * (Fintype.card n - #R) / Fintype.card n := by
  have h1 (i : n) (hi : i ∈ R) :
      centeredIndicator R i ^ 2 = (1 - (#R : ℝ) / Fintype.card n) ^ 2 := by
    rw [centeredIndicator, ite_eq_left hi]
  have h2 (i : n) (hi : i ∈ Rᶜ) : centeredIndicator R i ^ 2 = ((#R : ℝ) / Fintype.card n) ^ 2 := by
    rw [centeredIndicator, ite_eq_right (mem_compl.mp hi), zero_sub, neg_sq]
  rw [← sum_add_sum_compl R, sum_congr rfl h1, sum_congr rfl h2, sum_const, sum_const,
    nsmul_eq_mul, nsmul_eq_mul, card_compl, Nat.cast_sub (card_le_univ R)]
  field_simp
  ring

example := sum_sq_centeredIndicator (n := Fin 2) {0} (by simp)

/-- **Pairing with a missed block**: if `Ω` misses `R × C` and `Y` vanishes off `Ω` with unit
row and column sums, then `∑ u_i ((Y − Q) v)_i = −#R #C / N` for the centered indicators `u`
of `R` and `v` of `C`. -/
lemma sum_centeredIndicator_mul_mulVec {Ω : Set (n × n)} {R C : Finset n}
    (hRC : ∀ i ∈ R, ∀ j ∈ C, (i, j) ∉ Ω) {Y : Matrix n n ℂ}
    (hYΩ : ∀ i j, (i, j) ∉ Ω → Y i j = 0) (hrow : ∀ i, ∑ j, Y i j = 1)
    (hcol : ∀ j, ∑ i, Y i j = 1) (hN : (Fintype.card n : ℝ) ≠ 0) :
    ∑ i, (centeredIndicator R i : ℂ) *
        ((Y - meanProj n) *ᵥ fun j => (centeredIndicator C j : ℂ)) i =
      -((#R * #C / Fintype.card n : ℝ) : ℂ) := by
  have hsum (S : Finset n) : ∑ i, (centeredIndicator S i : ℂ) = 0 := by
    rw [← Complex.ofReal_sum, sum_centeredIndicator S hN, Complex.ofReal_zero]
  have hQ (i : n) : (meanProj n *ᵥ fun j => (centeredIndicator C j : ℂ)) i = 0 := by
    simp only [mulVec, dotProduct, meanProj, of_apply, ← mul_sum, hsum, mul_zero]
  have hY (i : n) : (Y *ᵥ fun j => (centeredIndicator C j : ℂ)) i =
      ∑ j ∈ C, Y i j - ((#C / Fintype.card n : ℝ) : ℂ) := by
    simp only [mulVec, dotProduct, centeredIndicator, Complex.ofReal_sub, mul_sub,
      sum_sub_distrib, ← sum_mul, hrow, one_mul, apply_ite Complex.ofReal, Complex.ofReal_one,
      Complex.ofReal_zero, mul_ite, mul_one, mul_zero, sum_ite_mem, univ_inter]
  have hmv (i : n) : ((Y - meanProj n) *ᵥ fun j => (centeredIndicator C j : ℂ)) i =
      ∑ j ∈ C, Y i j - ((#C / Fintype.card n : ℝ) : ℂ) := by
    rw [sub_mulVec, Pi.sub_apply, hY, hQ, sub_zero]
  have hblock : ∑ i ∈ R, ∑ j ∈ C, Y i j = 0 :=
    sum_eq_zero fun i hi => sum_eq_zero fun j hj => hYΩ i j (hRC i hi j hj)
  have hcols : ∑ i, ∑ j ∈ C, Y i j = #C := by
    rw [show ∑ i, ∑ j ∈ C, Y i j = ∑ j ∈ C, ∑ i, Y i j from sum_comm,
      sum_congr rfl fun j _ => hcol j, sum_const, nsmul_eq_mul, mul_one]
  have hu (i : n) : (centeredIndicator R i : ℂ) =
      (if i ∈ R then 1 else 0) - ((#R / Fintype.card n : ℝ) : ℂ) := by
    rw [centeredIndicator, Complex.ofReal_sub, apply_ite Complex.ofReal, Complex.ofReal_one,
      Complex.ofReal_zero]
  rw [sum_congr rfl fun i _ => by rw [hmv i]]
  simp only [mul_sub, sum_sub_distrib, ← sum_mul, hsum, zero_mul, sub_zero]
  simp only [hu, sub_mul, sum_sub_distrib, ite_mul, one_mul, zero_mul, sum_ite_mem, univ_inter,
    hblock, ← mul_sum, hcols]
  push_cast
  ring

example := sum_centeredIndicator_mul_mulVec (n := Fin 2) (Ω := {ij | ij ≠ (0, 1)}) (R := {0})
  (C := {1}) (by simp) (Y := 1) (fun i j h => by simp_all) (fun i => by simp [Matrix.one_apply])
  (fun j => by simp [Matrix.one_apply]) (by simp)

/-- **A missed block bounds every certificate**, `review/flat_spectrum.md`, *Patterns*: if `Ω`
misses `R × C`, `Y` vanishes off `Ω` with unit row and column sums, and `c (Y − Q)` is a
contraction, then `c² · #R · #C ≤ (N − #R)(N − #C)`. -/
theorem sq_mul_le_of_block {Ω : Set (n × n)} {R C : Finset n}
    (hRC : ∀ i ∈ R, ∀ j ∈ C, (i, j) ∉ Ω) {Y : Matrix n n ℂ}
    (hYΩ : ∀ i j, (i, j) ∉ Ω → Y i j = 0) (hrow : ∀ i, ∑ j, Y i j = 1)
    (hcol : ∀ j, ∑ i, Y i j = 1) {c : ℝ} (hW : IsContraction ((c : ℂ) • (Y - meanProj n))) :
    c ^ 2 * ((#R : ℝ) * #C) ≤ ((Fintype.card n : ℝ) - #R) * ((Fintype.card n : ℝ) - #C) := by
  have hkN : (#R : ℝ) ≤ Fintype.card n := by exact_mod_cast card_le_univ R
  have hlN : (#C : ℝ) ≤ Fintype.card n := by exact_mod_cast card_le_univ C
  rcases R.eq_empty_or_nonempty with hR | ⟨i, -⟩
  · rw [hR, card_empty, Nat.cast_zero, zero_mul, mul_zero]
    exact mul_nonneg (by linarith) (by linarith)
  have hN : (0 : ℝ) < Fintype.card n := by exact_mod_cast Fintype.card_pos_iff.mpr ⟨i⟩
  have hnorm (S : Finset n) : ∑ i, ‖(centeredIndicator S i : ℂ)‖ ^ 2 =
      #S * (Fintype.card n - #S) / Fintype.card n := by
    simp only [Complex.norm_real, Real.norm_eq_abs, sq_abs]
    exact sum_sq_centeredIndicator S hN.ne'
  have hpair : ∑ i, star (centeredIndicator R i : ℂ) *
      (((c : ℂ) • (Y - meanProj n)) *ᵥ fun j => (centeredIndicator C j : ℂ)) i =
      -((c * (#R * #C / Fintype.card n) : ℝ) : ℂ) := by
    simp only [Complex.star_def, Complex.conj_ofReal, smul_mulVec, Pi.smul_apply, smul_eq_mul,
      mul_left_comm _ (c : ℂ), ← mul_sum, sum_centeredIndicator_mul_mulVec hRC hYΩ hrow hcol hN.ne']
    push_cast
    ring
  have hcs := norm_sum_star_mul_le univ (fun i => (centeredIndicator R i : ℂ))
    (((c : ℂ) • (Y - meanProj n)) *ᵥ fun j => (centeredIndicator C j : ℂ))
  rw [hpair, norm_neg, Complex.norm_real, Real.norm_eq_abs, hnorm R] at hcs
  have hv := (hW.sum_norm_sq_mulVec_le fun j => (centeredIndicator C j : ℂ)).trans (hnorm C).le
  have hu0 : (0 : ℝ) ≤ #R * (Fintype.card n - #R) / Fintype.card n :=
    div_nonneg (mul_nonneg (by positivity) (by linarith)) hN.le
  rw [← Real.sqrt_mul hu0, Real.le_sqrt (abs_nonneg _)
    (mul_nonneg hu0 (sum_nonneg fun _ _ => sq_nonneg _)), sq_abs] at hcs
  have h3 := hcs.trans (mul_le_mul_of_nonneg_left hv hu0)
  set N : ℝ := (Fintype.card n : ℝ)
  set k : ℝ := (#R : ℝ)
  set l : ℝ := (#C : ℝ)
  have h4 : c ^ 2 * (k * l) * (k * l) ≤ (N - k) * (N - l) * (k * l) := by
    rw [← div_le_div_iff_of_pos_right (pow_pos hN 2)]
    convert h3 using 1 <;> ring
  rcases (mul_nonneg (Nat.cast_nonneg #R) (Nat.cast_nonneg #C) : (0 : ℝ) ≤ k * l).eq_or_lt
    with h0 | hpos
  · rw [← h0, mul_zero]
    exact mul_nonneg (by linarith) (by linarith)
  · exact le_of_mul_le_mul_right h4 hpos

example := sq_mul_le_of_block (n := Fin 2) (Ω := {ij | ij ≠ (0, 1)}) (R := {0}) (C := {1})
  (by simp) (Y := 1) (fun i j h => by simp_all) (fun i => by simp [Matrix.one_apply])
  (fun j => by simp [Matrix.one_apply]) (c := 0) (by simpa using isContraction_zero)

/-- **No certificate past a large missed block**, `review/flat_spectrum.md`, *Patterns*: if `Ω`
misses `R × C` with `R` and `C` nonempty and `N ≤ #R + #C`, and `Y` vanishes off `Ω` with unit row
and column sums, then no `c > 1` makes `c (Y − Q)` a contraction. -/
theorem not_isContraction_of_block {Ω : Set (n × n)} {R C : Finset n} (hR : R.Nonempty)
    (hC : C.Nonempty) (hN : Fintype.card n ≤ #R + #C) (hRC : ∀ i ∈ R, ∀ j ∈ C, (i, j) ∉ Ω)
    {Y : Matrix n n ℂ} (hYΩ : ∀ i j, (i, j) ∉ Ω → Y i j = 0) (hrow : ∀ i, ∑ j, Y i j = 1)
    (hcol : ∀ j, ∑ i, Y i j = 1) {c : ℝ} (hc : 1 < c) :
    ¬ IsContraction ((c : ℂ) • (Y - meanProj n)) := fun hW => by
  have h := sq_mul_le_of_block hRC hYΩ hrow hcol hW
  have hk : (0 : ℝ) < #R := by exact_mod_cast hR.card_pos
  have hl : (0 : ℝ) < #C := by exact_mod_cast hC.card_pos
  have hN' : (Fintype.card n : ℝ) ≤ #R + #C := by exact_mod_cast hN
  nlinarith [mul_lt_mul_of_pos_right (one_lt_pow₀ hc two_ne_zero) (mul_pos hk hl),
    mul_nonneg (Nat.cast_nonneg (α := ℝ) (Fintype.card n)) (sub_nonneg.mpr hN')]

example := not_isContraction_of_block (n := Fin 2) (Ω := {ij | ij ≠ (0, 1)}) (R := {0})
  (C := {1}) (singleton_nonempty 0) (singleton_nonempty 1) (by simp) (by simp) (Y := 1)
  (fun i j h => by simp_all) (fun i => by simp [Matrix.one_apply])
  (fun j => by simp [Matrix.one_apply]) (c := 2) (by norm_num)

/-- **A row with one cell admits no certificate**: if row `i₀` meets `Ω` only in column `j₀`,
`N ≥ 2`, and `Y` vanishes off `Ω` with unit row and column sums, then no `c > 1` makes
`c (Y − Q)` a contraction: `Ω` misses the block `{i₀} × {j₀}ᶜ`. -/
theorem not_isContraction_of_row {Ω : Set (n × n)} {i₀ j₀ : n} (hn : 2 ≤ Fintype.card n)
    (hi₀ : ∀ j ≠ j₀, (i₀, j) ∉ Ω) {Y : Matrix n n ℂ} (hYΩ : ∀ i j, (i, j) ∉ Ω → Y i j = 0)
    (hrow : ∀ i, ∑ j, Y i j = 1) (hcol : ∀ j, ∑ i, Y i j = 1) {c : ℝ} (hc : 1 < c) :
    ¬ IsContraction ((c : ℂ) • (Y - meanProj n)) := by
  refine not_isContraction_of_block (R := {i₀}) (C := {j₀}ᶜ) (singleton_nonempty i₀)
    (by rw [← card_pos, card_compl, card_singleton]; omega)
    (by rw [card_singleton, card_compl, card_singleton]; omega) ?_ hYΩ hrow hcol hc
  intro i hi j hj
  rw [mem_singleton.mp hi]
  exact hi₀ j (by simpa using hj)

example := not_isContraction_of_row (n := Fin 2) (Ω := {ij | ij ≠ (0, 1)}) (i₀ := 0) (j₀ := 0)
  (by simp) (by simp [Fin.forall_fin_two]) (Y := 1) (fun i j h => by simp_all)
  (fun i => by simp [Matrix.one_apply]) (fun j => by simp [Matrix.one_apply]) (c := 2) one_lt_two

end Transformer.Grokking.FlatSpectrum
