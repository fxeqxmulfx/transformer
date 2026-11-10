module

public import Transformer.Grokking.FlatSpectrum.FlatKernel
public import Transformer.Grokking.FlatSpectrum.Interpolation
public import Transformer.Grokking.FlatSpectrum.Training.Basic
public import Transformer.Grokking.FlatSpectrum.Unimodular

/-!
# A flat kernel has no cheap directions

Source: `review/flat_spectrum.md`, *Flat kernels of any shape* (added), item R3. Let `K` be flat
with squared singular value `c > 0` and `r₀ = min(|m|, |n|)`. Every matrix `X` of the shape of
`K` has

    c (r₀ − rank X) ≤ ‖K − X‖_S2²        (`IsFlat.mul_sub_rank_le_frobeniusSq_sub`).

Proof. `K / √c` is a contraction, so `Re tr(Kᴴ X) ≤ √c ‖X‖_S1` (`IsFlat.pairing_le`). The nonzero
singular values of `X` number `rank X`, so Cauchy–Schwarz gives `‖X‖_S1² ≤ rank X · ‖X‖_S2²`
(`nuclearNorm_sq_le_rank_mul_frobeniusSq`). With `‖K‖_S2² = c r₀`,

    ‖K − X‖_S2² ≥ c r₀ − 2 √c √(rank X) ‖X‖_S2 + ‖X‖_S2² ≥ c r₀ − c rank X.

So a product `A B` through width `k` misses a flat kernel by at least `c (r₀ − k)`
(`IsFlat.mul_sub_card_le_frobeniusSq_sub`): each unit of width buys the same `c`, the fraction
`1 / r₀` of `‖K‖_S2²`, and an exact factorization has `k ≥ r₀` (`IsFlat.card_le_of_eq_mul`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Training
open scoped ComplexOrder

variable {m n k : Type*} [Fintype m] [Fintype n] [Fintype k] [DecidableEq m] [DecidableEq n]

omit [DecidableEq m] in
/-- `‖X‖_S1² ≤ rank X · ‖X‖_S2²`: Cauchy–Schwarz over the nonzero singular values
(`review/flat_spectrum.md`, R3). -/
theorem nuclearNorm_sq_le_rank_mul_frobeniusSq (X : Matrix m n ℂ) :
    nuclearNorm X ^ 2 ≤ X.rank * frobeniusSq X := by
  have hA := isHermitian_conjTranspose_mul_self X
  have hcard : ((Finset.univ.filter fun i => hA.eigenvalues i ≠ 0).card : ℝ) = X.rank := by
    rw [← rank_conjTranspose_mul_self X, hA.rank_eq_card_non_zero_eigs, Fintype.card_subtype]
  have hsum : nuclearNorm X =
      ∑ i ∈ Finset.univ.filter fun i => hA.eigenvalues i ≠ 0, singularValues X i := by
    rw [nuclearNorm]
    refine (Finset.sum_filter_of_ne (p := fun i => hA.eigenvalues i ≠ 0)
      fun i _ hi h0 => hi ?_).symm
    rw [singularValues, h0, Real.sqrt_zero]
  have hsq : ∑ i ∈ Finset.univ.filter fun i => hA.eigenvalues i ≠ 0, singularValues X i ^ 2 ≤
      frobeniusSq X := by
    rw [← sum_singularValues_sq]
    exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _) fun _ _ _ => sq_nonneg _
  calc nuclearNorm X ^ 2
      = (∑ i ∈ Finset.univ.filter fun i => hA.eigenvalues i ≠ 0, singularValues X i) ^ 2 := by
        rw [hsum]
    _ ≤ (Finset.univ.filter fun i => hA.eigenvalues i ≠ 0).card *
          ∑ i ∈ Finset.univ.filter fun i => hA.eigenvalues i ≠ 0, singularValues X i ^ 2 :=
        sq_sum_le_card_mul_sum_sq
    _ ≤ X.rank * frobeniusSq X := by
        rw [hcard]
        exact mul_le_mul_of_nonneg_left hsq (Nat.cast_nonneg _)

omit [DecidableEq m] in
/-- `r_eff X ≤ rank X` for every matrix (`review/flat_spectrum.md`, R3). -/
theorem effectiveRank_le_rank (X : Matrix m n ℂ) : effectiveRank X ≤ X.rank := by
  rw [effectiveRank, div_pow, Real.sq_sqrt (frobeniusSq_nonneg X)]
  rcases (frobeniusSq_nonneg X).eq_or_lt with h | h
  · rw [← h, div_zero]
    exact Nat.cast_nonneg _
  · rw [div_le_iff₀ h]
    exact nuclearNorm_sq_le_rank_mul_frobeniusSq X

omit [DecidableEq m] in
/-- `r_eff X ≤ min(|m|, |n|)`: a flat kernel has the largest effective rank of its shape
(`review/flat_spectrum.md`, R3). -/
theorem effectiveRank_le_min (X : Matrix m n ℂ) :
    effectiveRank X ≤ (min (Fintype.card m) (Fintype.card n) : ℕ) :=
  (effectiveRank_le_rank X).trans
    (Nat.cast_le.2 (le_min (rank_le_card_height X) (rank_le_card_width X)))

/-- A flat kernel pairs with `X` at most `√c ‖X‖_S1` (`review/flat_spectrum.md`, R3). -/
theorem IsFlat.pairing_le {K : Matrix m n ℂ} {c : ℝ} (hc : 0 < c) (h : IsFlat K c)
    (X : Matrix m n ℂ) : pairing K X ≤ √c * nuclearNorm X := by
  have h1 := norm_trace_le_nuclearNorm (h.isContraction hc) X
  rw [conjTranspose_smul, Matrix.smul_mul, trace_smul, smul_eq_mul, norm_mul, star_inv₀,
    norm_inv, norm_star, Complex.norm_real, Real.norm_of_nonneg (Real.sqrt_nonneg c),
    inv_mul_le_iff₀ (Real.sqrt_pos.2 hc)] at h1
  exact ((le_abs_self _).trans (Complex.abs_re_le_norm _)).trans h1

/-- **A flat kernel has no cheap directions** (`review/flat_spectrum.md`, R3): every `X` misses
a flat `K` by at least `c` for each unit of rank below `min(|m|, |n|)`. -/
theorem IsFlat.mul_sub_rank_le_frobeniusSq_sub {K : Matrix m n ℂ} {c : ℝ} (hc : 0 < c)
    (h : IsFlat K c) (X : Matrix m n ℂ) :
    c * ((min (Fintype.card m) (Fintype.card n) : ℕ) - X.rank) ≤ frobeniusSq (K - X) := by
  have h1 := h.pairing_le hc X
  have h2 := nuclearNorm_sq_le_rank_mul_frobeniusSq X
  have hρ : (0 : ℝ) ≤ X.rank := Nat.cast_nonneg _
  have key : 2 * (√c * nuclearNorm X) ≤ c * X.rank + frobeniusSq X := by
    refine (le_abs_self _).trans (abs_le_of_sq_le_sq ?_
      (add_nonneg (mul_nonneg hc.le hρ) (frobeniusSq_nonneg X)))
    rw [mul_pow, mul_pow, Real.sq_sqrt hc.le]
    nlinarith [sq_nonneg (c * X.rank - frobeniusSq X), mul_le_mul_of_nonneg_left h2 hc.le]
  rw [frobeniusSq_sub, h.frobeniusSq_eq hc.ne']
  linarith

/-- The `a³ + a b` kernel for `p = 3` is at squared distance at least `3 (3 − rank X)` from
every `X`. -/
example (X : Matrix (ZMod 3) (ZMod 3) ℂ) :=
  (isFlat_cubicKernel 3).mul_sub_rank_le_frobeniusSq_sub (by norm_num) X

/-- **A product through width `k` misses a flat kernel by `c (min(|m|, |n|) − k)`**
(`review/flat_spectrum.md`, R3). -/
theorem IsFlat.mul_sub_card_le_frobeniusSq_sub {K : Matrix m n ℂ} {c : ℝ} (hc : 0 < c)
    (h : IsFlat K c) (A : Matrix m k ℂ) (B : Matrix k n ℂ) :
    c * ((min (Fintype.card m) (Fintype.card n) : ℕ) - Fintype.card k) ≤
      frobeniusSq (K - A * B) := by
  have h1 := h.mul_sub_rank_le_frobeniusSq_sub hc (A * B)
  have h2 : ((A * B).rank : ℝ) ≤ Fintype.card k :=
    Nat.cast_le.2 ((rank_mul_le_left A B).trans (rank_le_card_width A))
  nlinarith [mul_le_mul_of_nonneg_left h2 hc.le]

/-- A product through width one misses the `a³ + a b` kernel for `p = 3` by at least `6`, two
thirds of `‖K‖_S2² = 9`. -/
example (A : Matrix (ZMod 3) (Fin 1) ℂ) (B : Matrix (Fin 1) (ZMod 3) ℂ) :=
  (isFlat_cubicKernel 3).mul_sub_card_le_frobeniusSq_sub (by norm_num) A B

/-- **An exact factorization of a flat kernel has width at least `min(|m|, |n|)`**
(`review/flat_spectrum.md`, R3). -/
theorem IsFlat.card_le_of_eq_mul {K : Matrix m n ℂ} {c : ℝ}
    (hc : c ≠ 0) (h : IsFlat K c) {A : Matrix m k ℂ} {B : Matrix k n ℂ} (hAB : K = A * B) :
    min (Fintype.card m) (Fintype.card n) ≤ Fintype.card k := by
  rw [← h.rank_eq hc, hAB]
  exact (rank_mul_le_left A B).trans (rank_le_card_width A)

/-- The identity of size `2` factors through width `2`, and through no smaller width. -/
example := isFlat_one.card_le_of_eq_mul (n := Fin 2) one_ne_zero
  (A := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (B := 1) (by simp)

end Transformer.Grokking.FlatSpectrum
