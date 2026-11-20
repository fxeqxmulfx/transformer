module

public import Transformer.Grokking.FlatSpectrum.BilinearKernel
public import Transformer.Grokking.FlatSpectrum.TwoDiagonals
public import Transformer.Grokking.FlatSpectrum.Unimodular

/-!
# The price of a cut is exact at both ends

Source: `review/flat_spectrum.md`, *Functions of n variables* (added), item NV2.

R3 bounds from below the error of every matrix of rank at most `d` on a flat kernel:
`‖K − X‖_S2² ≥ c (min(m, n) − d)`. The bound is attained. If `Kᴴ K = c I`, every column has
squared length `c`; keeping `d` columns and zeroing the others gives a matrix of rank at most `d`
whose error is the squared length of the `|n| − d` dropped columns, `c (|n| − d)`
(`exists_rank_le_frobeniusSq_sub_eq`). The adjoint handles `K Kᴴ = c I`. So
`c (min(m, n) − d)` is the least error of rank `d` (`IsFlat.isLeast_frobeniusSq_sub`): a product
through width `d` on a flat table of `n` variables pays exactly `p^l (p^k − d)`.

The nuclear norm of an `N × N` table with entries of modulus one lies in `[N, N √N]`
(`card_le_nuclearNorm_le_of_unimodular`). For a table of `n` variables cut into two groups of
`k`, `N = p^k`, both ends are attained: the additive table `ψ(x₁ + … + x_n)` has rank one and
nuclear norm `N` (`isLeast_nuclearNorm_unimodular`), and `k` copies of `x³ + x y` have
`N √N` (`isGreatest_nuclearNorm_unimodular`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset
open scoped ComplexOrder

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]

omit [DecidableEq m] [DecidableEq n] in
/-- `‖Xᴴ‖_S2 = ‖X‖_S2`. -/
lemma frobeniusSq_conjTranspose (X : Matrix m n ℂ) : frobeniusSq Xᴴ = frobeniusSq X := by
  simp only [frobeniusSq, conjTranspose_apply, norm_star]
  exact Finset.sum_comm

omit [DecidableEq m] in
/-- **R3 is attained by dropping columns** (`review/flat_spectrum.md`, NV2): if `Kᴴ K = c I`,
keeping `d` columns of `K` and zeroing the others leaves rank at most `d` and the error
`c (|n| − d)`. -/
theorem exists_rank_le_frobeniusSq_sub_eq {K : Matrix m n ℂ} {c : ℝ} (h : Kᴴ * K = (c : ℂ) • 1)
    {d : ℕ} (hd : d ≤ Fintype.card n) :
    ∃ X : Matrix m n ℂ, X.rank ≤ d ∧ frobeniusSq (K - X) = c * ((Fintype.card n : ℝ) - d) := by
  obtain ⟨S, -, hS⟩ := exists_subset_card_eq (s := (univ : Finset n)) (by simpa using hd)
  set w : n → ℂ := fun j => if j ∈ S then 1 else 0 with hw
  refine ⟨K * diagonal w, (rank_mul_le_right _ _).trans ?_, ?_⟩
  · rw [rank_diagonal, Fintype.card_subtype, ← hS]
    exact card_le_card fun j hj => by simpa [hw] using hj
  · have hcol (j : n) : ∑ i, ‖K i j‖ ^ 2 = c := by
      have h1 := conjTranspose_mul_self_apply_self K j
      rw [h, Matrix.smul_apply, one_apply_eq, smul_eq_mul, mul_one] at h1
      exact_mod_cast h1.symm
    have h2 : frobeniusSq (K - K * diagonal w) = ∑ j, if j ∈ S then 0 else c := by
      rw [frobeniusSq, Finset.sum_comm]
      refine sum_congr rfl fun j _ => ?_
      by_cases hj : j ∈ S
      · simp [hw, hj]
      · simpa [hw, hj] using hcol j
    have e1 : ∑ j ∈ Sᶜ, (if j ∈ S then 0 else c) = ∑ _j ∈ Sᶜ, c :=
      sum_congr rfl fun j hj => ite_eq_right (mem_compl.1 hj)
    have e2 : ∑ j ∈ S, (if j ∈ S then 0 else c) = 0 := sum_eq_zero fun j hj => ite_eq_left hj
    rw [h2, ← sum_compl_add_sum S, e1, e2, add_zero, sum_const, card_compl, hS, nsmul_eq_mul,
      Nat.cast_sub hd, mul_comm]

/-- Dropping one of the two columns of the identity leaves the error `1`. -/
example := exists_rank_le_frobeniusSq_sub_eq (m := Fin 2) (n := Fin 2) (K := 1) (c := 1)
  (by simp) (d := 1) (by simp)

/-- **R3 is exact** (`review/flat_spectrum.md`, NV2): for a flat `K` and `d ≤ min(|m|, |n|)`, the
least error `‖K − X‖_S2²` over the matrices of rank at most `d` is `c (min(|m|, |n|) − d)`. -/
theorem IsFlat.isLeast_frobeniusSq_sub {K : Matrix m n ℂ} {c : ℝ} (hc : 0 < c) (h : IsFlat K c)
    {d : ℕ} (hd : d ≤ min (Fintype.card m) (Fintype.card n)) :
    IsLeast {e | ∃ X : Matrix m n ℂ, X.rank ≤ d ∧ e = frobeniusSq (K - X)}
      (c * (((min (Fintype.card m) (Fintype.card n) : ℕ) : ℝ) - d)) := by
  refine ⟨?_, fun e ⟨X, hX, he⟩ => he ▸ le_trans ?_ (h.mul_sub_rank_le_frobeniusSq_sub hc X)⟩
  · rcases h with h1 | h1
    · have hr := rank_eq_card_of_conjTranspose_mul_self hc.ne' h1
      have hmin : min (Fintype.card m) (Fintype.card n) = Fintype.card n :=
        min_eq_right (hr ▸ rank_le_card_height K)
      obtain ⟨X, hX, he⟩ := exists_rank_le_frobeniusSq_sub_eq h1 (hmin ▸ hd)
      exact ⟨X, hX, by rw [he, hmin]⟩
    · have hr := rank_eq_card_of_mul_conjTranspose_self hc.ne' h1
      have hmin : min (Fintype.card m) (Fintype.card n) = Fintype.card m :=
        min_eq_left (hr ▸ rank_le_card_width K)
      obtain ⟨X, hX, he⟩ := exists_rank_le_frobeniusSq_sub_eq (K := Kᴴ)
        (by rwa [conjTranspose_conjTranspose]) (hmin ▸ hd)
      refine ⟨Xᴴ, by rwa [rank_conjTranspose], ?_⟩
      rw [hmin, ← he, ← frobeniusSq_conjTranspose, conjTranspose_sub,
        conjTranspose_conjTranspose]
  · exact mul_le_mul_of_nonneg_left (by linarith [(Nat.cast_le (α := ℝ)).2 hX]) hc.le

/-- Rank one misses the identity of size two by exactly `1`. -/
example := isFlat_one.isLeast_frobeniusSq_sub (n := Fin 2) one_pos (d := 1) (by simp)

/-- **A cut of width `d` on two copies of `x³ + x y`** (`review/flat_spectrum.md`, NV2): the
least error is `9 (9 − d)`. -/
example (d : ℕ) (hd : d ≤ 9) :=
  (isFlat_bilinearKernel 3 (ι := Fin 2) (κ := Fin 2) (fun x => ∑ i, x i ^ 3) (fun _ => 0)
    (M := 1) (vecMul_injective_of_isUnit isUnit_one)).isLeast_frobeniusSq_sub
    (by positivity) (d := d) (by simpa using hd)

variable (p : ℕ) [NeZero p] {ι : Type*} [Fintype ι] [DecidableEq ι]

omit [DecidableEq ι] in
/-- The additive table `ψ(Σ uᵢ + Σ vⱼ)` is `M = 0`: the rank-one `ψ(Σ uᵢ) ψ(Σ vⱼ)ᵀ`. -/
lemma bilinearKernel_sum_sum :
    bilinearKernel p (fun u : ι → ZMod p => ∑ i, u i) (fun v : ι → ZMod p => ∑ i, v i) 0 =
      vecMulVec (fun u => ZMod.stdAddChar (∑ i, u i)) fun v => ZMod.stdAddChar (∑ i, v i) := by
  ext u v
  simp [bilinearKernel, vecMulVec_apply, AddChar.map_add_eq_mul]

/-- `|(ι → ℤ/p)| = p^{|ι|}`, as a real number. -/
lemma card_fun_zmod : ((Fintype.card (ι → ZMod p) : ℕ) : ℝ) = (p : ℝ) ^ Fintype.card ι := by
  simp [ZMod.card]

/-- **The least nuclear norm of a table of `2k` variables is `N = p^k`**
(`review/flat_spectrum.md`, NV2), over the tables with entries of modulus one, cut into two groups
of `k`; the additive table attains it. -/
theorem isLeast_nuclearNorm_unimodular :
    IsLeast {t | ∃ K : Matrix (ι → ZMod p) (ι → ZMod p) ℂ, (∀ u v, ‖K u v‖ = 1) ∧
      nuclearNorm K = t} ((p : ℝ) ^ Fintype.card ι) := by
  refine ⟨⟨bilinearKernel p (fun u : ι → ZMod p => ∑ i, u i) (fun v => ∑ i, v i) 0,
    fun u v => by simp [bilinearKernel], ?_⟩, ?_⟩
  · rw [bilinearKernel_sum_sum, nuclearNorm_vecMulVec_of_norm_eq_one (by simp) (by simp),
      card_fun_zmod]
  · rintro t ⟨K, hK, rfl⟩
    exact (card_fun_zmod p (ι := ι)) ▸ (card_le_nuclearNorm_le_of_unimodular hK).1

example := isLeast_nuclearNorm_unimodular 3 (ι := Fin 2)

/-- **The greatest nuclear norm of a table of `2k` variables is `N √N`**
(`review/flat_spectrum.md`, NV2), over the tables with entries of modulus one, cut into two groups
of `k`; `k` copies of `x³ + x y` attain it. -/
theorem isGreatest_nuclearNorm_unimodular :
    IsGreatest {t | ∃ K : Matrix (ι → ZMod p) (ι → ZMod p) ℂ, (∀ u v, ‖K u v‖ = 1) ∧
      nuclearNorm K = t} ((p : ℝ) ^ Fintype.card ι * √((p : ℝ) ^ Fintype.card ι)) := by
  refine ⟨⟨bilinearKernel p (fun x => ∑ i, x i ^ 3) 0 1, fun u v => by simp [bilinearKernel],
    nuclearNorm_bilinearKernel _ _ isUnit_one⟩, ?_⟩
  rintro t ⟨K, hK, rfl⟩
  have h := (card_le_nuclearNorm_le_of_unimodular hK).2
  rwa [card_fun_zmod, rpow_three_halves (by positivity)] at h

example := isGreatest_nuclearNorm_unimodular 3 (ι := Fin 2)

end Transformer.Grokking.FlatSpectrum
