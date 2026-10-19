module

public import Transformer.Homogenized.Simplex

/-!
# Centroid geometry of the original simplex

ArXiv:2604.01978v1, §7, thm:clustering_random_init and the
reference Gram matrix R(γ) in lem:Gram_stability.
The comparison argument needs an actual configuration with that
Gram matrix at every feasible scalar overlap. Here the original
simplex is decomposed into its centroid and its centered tokens.

The centroid has squared norm (1+(n-1)γ)/n. Every centered token
is orthogonal to the centroid, and their Gram matrix is
(1-γ)(I-11ᵀ/n). These are identities for the original vectors,
without any dimension enlargement or assumed Gram realization.
The next module uses them to construct the comparison simplex.
-/

@[expose] public section

open scoped BigOperators

namespace Transformer.Homogenized

/-- The actual token centroid used to realize the source's R(γ).
Source: arXiv:2604.01978v1, §7, lem:Gram_stability. -/
noncomputable def simplexCentroid {d n : ℕ} (X : Idx n → EucSpace d) : EucSpace d :=
  (n : ℝ)⁻¹ • ∑ i, X i

/-- The centroid weight of the simplex Gram matrix.
Source: arXiv:2604.01978v1, §7, R(γ)=(1-γ)I+γ11ᵀ. -/
noncomputable def simplexMeanSquare (n : ℕ) (γ : ℝ) : ℝ := (1 + ((n : ℝ) - 1) * γ) / n

/-- The original simplex Gram entries, with the unit diagonal retained.
Source: arXiv:2604.01978v1, §7, thm:clustering_random_init. -/
theorem inner_simplex_tokens {d n : ℕ} {γ : ℝ} {X : Idx n → EucSpace d}
    (hX : IsSimplexConfig γ X) (i j : Idx n) :
    inner (𝕜 := ℝ) (X i) (X j) = (if i = j then 1 - γ else 0) + γ := by
  split_ifs with hij
  · subst j
    rw [real_inner_self_eq_norm_sq, hX.1 i]
    ring
  · rw [hX.2 i j hij]
    ring

/-- Every token has the same pairing with the actual centroid.
Source: arXiv:2604.01978v1, §7, the reference R(γ) in lem:Gram_stability. -/
theorem inner_token_simplexCentroid {d n : ℕ} (hn : 0 < n) {γ : ℝ}
    {X : Idx n → EucSpace d} (hX : IsSimplexConfig γ X) (i : Idx n) :
    inner (𝕜 := ℝ) (X i) (simplexCentroid X) = simplexMeanSquare n γ := by
  have hnR : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  simp only [simplexCentroid, real_inner_smul_right, inner_sum,
    inner_simplex_tokens hX, Finset.sum_add_distrib]
  simp only [Finset.sum_ite_eq, Finset.mem_univ, ite_true, Finset.sum_const,
    Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  dsimp [simplexMeanSquare]
  field_simp
  ring

/-- The actual centroid has the squared norm dictated by R(γ).
Source: arXiv:2604.01978v1, §7, lem:Gram_stability. -/
theorem inner_simplexCentroid_self {d n : ℕ} (hn : 0 < n) {γ : ℝ}
    {X : Idx n → EucSpace d} (hX : IsSimplexConfig γ X) :
    inner (𝕜 := ℝ) (simplexCentroid X) (simplexCentroid X) = simplexMeanSquare n γ := by
  have hnR : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  conv_lhs => lhs; rw [simplexCentroid]
  simp only [real_inner_smul_left, sum_inner, inner_token_simplexCentroid hn hX,
    Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  exact inv_mul_cancel_left₀ hnR _

/-- Centering makes each original token orthogonal to the centroid.
Source: arXiv:2604.01978v1, §7, the simplex reference in lem:Gram_stability. -/
theorem inner_centered_simplexCentroid {d n : ℕ} (hn : 0 < n) {γ : ℝ}
    {X : Idx n → EucSpace d} (hX : IsSimplexConfig γ X) (i : Idx n) :
    inner (𝕜 := ℝ) (X i - simplexCentroid X) (simplexCentroid X) = 0 := by
  rw [inner_sub_left, inner_token_simplexCentroid hn hX, inner_simplexCentroid_self hn hX, sub_self]

/-- The centered original Gram matrix is exactly (1-γ)(I-11ᵀ/n).
Source: arXiv:2604.01978v1, §7, the reference R(γ) in lem:Gram_stability. -/
theorem inner_centered_simplex_tokens {d n : ℕ} (hn : 0 < n) {γ : ℝ}
    {X : Idx n → EucSpace d} (hX : IsSimplexConfig γ X) (i j : Idx n) :
    inner (𝕜 := ℝ) (X i - simplexCentroid X) (X j - simplexCentroid X) =
      (1 - γ) * ((if i = j then 1 else 0) - (n : ℝ)⁻¹) := by
  have hnR : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  rw [inner_sub_left, inner_sub_right, inner_sub_right, inner_simplex_tokens hX,
    inner_token_simplexCentroid hn hX, real_inner_comm (X j) (simplexCentroid X),
    inner_token_simplexCentroid hn hX, inner_simplexCentroid_self hn hX]
  dsimp [simplexMeanSquare]
  split_ifs <;> field_simp <;> ring

/-- The centroid weight is nonnegative on the full feasible interval.
Source: arXiv:2604.01978v1, §7, the admissible simplex overlaps. -/
theorem simplexMeanSquare_nonneg {n : ℕ} (hn : 2 ≤ n) {γ : ℝ}
    (hγ : -(1 / ((n : ℝ) - 1)) ≤ γ) : 0 ≤ simplexMeanSquare n γ := by
  have hnR : (2 : ℝ) ≤ n := by exact_mod_cast hn
  have hm : 0 < (n : ℝ) - 1 := by linarith
  have hcancel : -(1 / ((n : ℝ) - 1)) * ((n : ℝ) - 1) = -1 := by field_simp
  have hprod := mul_le_mul_of_nonneg_right hγ hm.le
  rw [hcancel] at hprod
  exact div_nonneg (by nlinarith) (by positivity)

/-- Interior admissibility makes the original centroid nonzero.
Source: arXiv:2604.01978v1, §7, the strict initial overlap interval
in thm:clustering_random_init. -/
theorem simplexMeanSquare_pos {n : ℕ} (hn : 2 ≤ n) {γ : ℝ}
    (hγ : -(1 / ((n : ℝ) - 1)) < γ) : 0 < simplexMeanSquare n γ := by
  have hnR : (2 : ℝ) ≤ n := by exact_mod_cast hn
  have hm : 0 < (n : ℝ) - 1 := by linarith
  have hcancel : -(1 / ((n : ℝ) - 1)) * ((n : ℝ) - 1) = -1 := by field_simp
  have hprod := mul_lt_mul_of_pos_right hγ hm
  rw [hcancel] at hprod
  exact div_pos (by nlinarith) (by positivity)

/-- The initial strict overlap interval really gives a nonzero original
centroid, so its direction can be normalized without a junk value.
Source: arXiv:2604.01978v1, §7, thm:clustering_random_init. -/
theorem simplexCentroid_ne_zero {d n : ℕ} (hn : 2 ≤ n) {γ : ℝ}
    (hγ : -(1 / ((n : ℝ) - 1)) < γ) {X : Idx n → EucSpace d}
    (hX : IsSimplexConfig γ X) : simplexCentroid X ≠ 0 := by
  intro hzero
  have hnorm := inner_simplexCentroid_self (by omega) hX
  rw [hzero, inner_zero_left] at hnorm
  exact (simplexMeanSquare_pos hn hγ).ne' hnorm.symm

/-- The actual centroid vanishes exactly at zero centroid weight.
Source: arXiv:2604.01978v1, §7, the simplex Gram matrix R(γ). -/
theorem simplexCentroid_eq_zero_iff {d n : ℕ} (hn : 0 < n) {γ : ℝ}
    {X : Idx n → EucSpace d} (hX : IsSimplexConfig γ X) :
    simplexCentroid X = 0 ↔ simplexMeanSquare n γ = 0 := by
  have hnorm := inner_simplexCentroid_self hn hX
  rw [real_inner_self_eq_norm_sq] at hnorm
  constructor
  · intro hzero
    simpa [hzero] using hnorm.symm
  · intro hzero
    exact norm_eq_zero.mp (by nlinarith [norm_nonneg (simplexCentroid X)])

/-- The original orthonormal pair satisfies all geometric hypotheses,
has a nonzero centroid, and has nonzero centered tokens.
Source: arXiv:2604.01978v1, §7, thm:clustering_random_init. -/
example : ∃ X : Idx 2 → EucSpace 2,
    0 < (2 : ℕ) ∧ 2 ≤ (2 : ℕ) ∧ IsSimplexConfig (0 : ℝ) X ∧
    -(1 / ((2 : ℝ) - 1)) < (0 : ℝ) ∧
    inner (𝕜 := ℝ) (simplexCentroid X) (simplexCentroid X) = 1 / 2 ∧
    inner (𝕜 := ℝ) (X 0 - simplexCentroid X) (X 0 - simplexCentroid X) = 1 / 2 := by
  let X : Idx 2 → EucSpace 2 := fun k => EuclideanSpace.single k 1
  have hX : IsSimplexConfig (0 : ℝ) X := by
    constructor
    · intro k; simp [X]
    · intro i j hij; simp [X, EuclideanSpace.inner_single_left, hij.symm]
  refine ⟨X, by norm_num, le_rfl, hX, by norm_num, ?_, ?_⟩
  · rw [inner_simplexCentroid_self (by norm_num) hX]
    norm_num [simplexMeanSquare]
  · rw [inner_centered_simplex_tokens (by norm_num) hX]
    norm_num

end Transformer.Homogenized
