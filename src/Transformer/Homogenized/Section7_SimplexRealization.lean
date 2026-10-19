module

public import Transformer.Homogenized.Section7_SimplexCentroid

/-!
# Realizing every feasible reference simplex in the original space

ArXiv:2604.01978v1, §7, the reference Gram matrix R(γ) in
lem:Gram_stability and thm:clustering_random_init.
Starting from the actual interior initial simplex, rescale its
centered part and centroid separately. This constructs unit tokens
with any overlap γ in [-1/(n-1),1], in the original EucSpace d.

The construction is continuous in γ, including both boundary
values, and returns the original tokens at their original overlap.
It requires no additional dimension hypothesis, orthonormal frame,
or assumed realization of an intermediate Gram matrix.
-/

@[expose] public section

open scoped BigOperators
open Set

namespace Transformer.Homogenized

/-- Rescale the two orthogonal parts of the actual initial simplex.
Source: arXiv:2604.01978v1, §7, R(γ) in lem:Gram_stability. -/
noncomputable def simplexRecenter {d n : ℕ} (X : Idx n → EucSpace d) (γ₀ γ : ℝ)
    (i : Idx n) : EucSpace d :=
  Real.sqrt ((1 - γ) / (1 - γ₀)) • (X i - simplexCentroid X) +
    Real.sqrt (simplexMeanSquare n γ / simplexMeanSquare n γ₀) • simplexCentroid X

/-- Independent rescaling of the centered tokens and the centroid gives
this exact Gram matrix. Source: arXiv:2604.01978v1, §7, R(γ). -/
theorem inner_rescaled_simplex_parts {d n : ℕ} (hn : 0 < n) {γ₀ : ℝ}
    {X : Idx n → EucSpace d} (hX : IsSimplexConfig γ₀ X) (a b : ℝ) (i j : Idx n) :
    inner (𝕜 := ℝ) (a • (X i - simplexCentroid X) + b • simplexCentroid X)
      (a • (X j - simplexCentroid X) + b • simplexCentroid X) =
        a ^ 2 * (1 - γ₀) * ((if i = j then 1 else 0) - (n : ℝ)⁻¹) +
          b ^ 2 * simplexMeanSquare n γ₀ := by
  simp only [inner_add_left, inner_add_right, real_inner_smul_left, real_inner_smul_right]
  rw [inner_centered_simplex_tokens hn hX, inner_centered_simplexCentroid hn hX,
    real_inner_comm (X j - simplexCentroid X) (simplexCentroid X),
    inner_centered_simplexCentroid hn hX, inner_simplexCentroid_self hn hX]
  ring

/-- The constructed vectors have the full desired Gram matrix, including
the unit diagonal. Source: arXiv:2604.01978v1, §7, lem:Gram_stability.
The reference simplex is constructed here, rather than assumed. -/
theorem inner_simplexRecenter {d n : ℕ} (hn : 2 ≤ n) {γ₀ : ℝ}
    (hγ₀ : γ₀ ∈ Ioo (-(1 / ((n : ℝ) - 1))) 1)
    {X : Idx n → EucSpace d} (hX : IsSimplexConfig γ₀ X) {γ : ℝ}
    (hγ : γ ∈ Icc (-(1 / ((n : ℝ) - 1))) 1) (i j : Idx n) :
    inner (𝕜 := ℝ) (simplexRecenter X γ₀ γ i) (simplexRecenter X γ₀ γ j) =
      if i = j then 1 else γ := by
  have hnR : (n : ℝ) ≠ 0 := by exact_mod_cast (show n ≠ 0 by omega)
  have hc : 0 < simplexMeanSquare n γ₀ := simplexMeanSquare_pos hn hγ₀.1
  have ha : 0 ≤ (1 - γ) / (1 - γ₀) := div_nonneg (by linarith [hγ.2]) (by linarith [hγ₀.2])
  have hb : 0 ≤ simplexMeanSquare n γ / simplexMeanSquare n γ₀ :=
    div_nonneg (simplexMeanSquare_nonneg hn hγ.1) hc.le
  rw [simplexRecenter, simplexRecenter, inner_rescaled_simplex_parts (by omega) hX,
    Real.sq_sqrt ha, Real.sq_sqrt hb]
  have hden : 1 - γ₀ ≠ 0 := by linarith [hγ₀.2]
  rw [div_mul_cancel₀ _ hden, div_mul_cancel₀ _ hc.ne']
  dsimp [simplexMeanSquare]
  split_ifs <;> field_simp <;> ring

/-- Every feasible overlap is realized by actual unit tokens in the original
ambient space. Source: arXiv:2604.01978v1, §7, thm:clustering_random_init. -/
theorem isSimplexConfig_simplexRecenter {d n : ℕ} (hn : 2 ≤ n) {γ₀ : ℝ}
    (hγ₀ : γ₀ ∈ Ioo (-(1 / ((n : ℝ) - 1))) 1)
    {X : Idx n → EucSpace d} (hX : IsSimplexConfig γ₀ X) {γ : ℝ}
    (hγ : γ ∈ Icc (-(1 / ((n : ℝ) - 1))) 1) :
    IsSimplexConfig γ (simplexRecenter X γ₀ γ) := by
  constructor
  · intro i
    have hi := inner_simplexRecenter hn hγ₀ hX hγ i i
    rw [ite_eq_left rfl, real_inner_self_eq_norm_sq] at hi
    nlinarith [norm_nonneg (simplexRecenter X γ₀ γ i)]
  · intro i j hij
    rw [inner_simplexRecenter hn hγ₀ hX hγ, ite_eq_right hij]

/-- At the initial overlap the explicit realization is the original
configuration itself. Source: arXiv:2604.01978v1, §7,
the initial simplex in thm:clustering_random_init. -/
theorem simplexRecenter_initial {d n : ℕ} (hn : 2 ≤ n) {γ₀ : ℝ}
    (hγ₀ : γ₀ ∈ Ioo (-(1 / ((n : ℝ) - 1))) 1) (X : Idx n → EucSpace d) :
    simplexRecenter X γ₀ γ₀ = X := by
  have hc := (simplexMeanSquare_pos hn hγ₀.1).ne'
  have ha : 1 - γ₀ ≠ 0 := by linarith [hγ₀.2]
  funext i
  simp [simplexRecenter, div_self ha, div_self hc]

/-- The reference configuration varies continuously even at the two
feasible boundary overlaps. Source: arXiv:2604.01978v1, §7,
the reference Gram line in lem:Gram_stability. -/
theorem continuous_simplexRecenter {d n : ℕ} (X : Idx n → EucSpace d) (γ₀ : ℝ) :
    Continuous (simplexRecenter X γ₀) := by
  apply continuous_pi
  intro i
  dsimp [simplexRecenter, simplexMeanSquare]
  fun_prop

/-- At overlap one all constructed tokens really coincide.
Source: arXiv:2604.01978v1, §7, the upper endpoint of R(γ).
This endpoint remains in the original space and is not excluded by
the square roots in the realization. -/
theorem simplexRecenter_one_eq {d n : ℕ} (hn : 2 ≤ n) {γ₀ : ℝ}
    (hγ₀ : γ₀ ∈ Ioo (-(1 / ((n : ℝ) - 1))) 1)
    {X : Idx n → EucSpace d} (hX : IsSimplexConfig γ₀ X) (i j : Idx n) :
    simplexRecenter X γ₀ 1 i = simplexRecenter X γ₀ 1 j := by
  have hinner (k l : Idx n) :
      inner (𝕜 := ℝ) (simplexRecenter X γ₀ 1 k) (simplexRecenter X γ₀ 1 l) = 1 := by
    simpa only [ite_self] using
      inner_simplexRecenter hn hγ₀ hX ⟨(hγ₀.1.trans hγ₀.2).le, le_rfl⟩ k l
  have hd : inner (𝕜 := ℝ)
      (simplexRecenter X γ₀ 1 i - simplexRecenter X γ₀ 1 j)
      (simplexRecenter X γ₀ 1 i - simplexRecenter X γ₀ 1 j) = 0 := by
    rw [inner_sub_left, inner_sub_right, inner_sub_right,
      hinner i i, hinner i j, hinner j i, hinner j j]
    ring
  rw [real_inner_self_eq_norm_sq] at hd
  have hnorm := norm_nonneg (simplexRecenter X γ₀ 1 i - simplexRecenter X γ₀ 1 j)
  exact sub_eq_zero.mp (norm_eq_zero.mp (by nlinarith))

/-- All reference Gram differences are bounded by the scalar overlap
difference. Source: arXiv:2604.01978v1, §7, lem:Gram_stability. -/
theorem abs_gram_simplexRecenter_sub_le {d n : ℕ} (hn : 2 ≤ n) {γ₀ : ℝ}
    (hγ₀ : γ₀ ∈ Ioo (-(1 / ((n : ℝ) - 1))) 1)
    {X : Idx n → EucSpace d} (hX : IsSimplexConfig γ₀ X) {γ η : ℝ}
    (hγ : γ ∈ Icc (-(1 / ((n : ℝ) - 1))) 1)
    (hη : η ∈ Icc (-(1 / ((n : ℝ) - 1))) 1) (i j : Idx n) :
    |inner (𝕜 := ℝ) (simplexRecenter X γ₀ γ i) (simplexRecenter X γ₀ γ j) -
      inner (𝕜 := ℝ) (simplexRecenter X γ₀ η i) (simplexRecenter X γ₀ η j)| ≤ |γ - η| := by
  rw [inner_simplexRecenter hn hγ₀ hX hγ, inner_simplexRecenter hn hγ₀ hX hη]
  split_ifs
  · simpa only [sub_self, abs_zero] using abs_nonneg (γ - η)
  · exact le_rfl

/-- An orthonormal pair realizes every feasible overlap in dimension two;
the same construction returns that pair at zero and is continuous.
All hypotheses above hold simultaneously with nonzero centered parts. -/
example : ∃ X : Idx 2 → EucSpace 2,
    2 ≤ (2 : ℕ) ∧ (0 : ℝ) ∈ Ioo (-(1 / ((2 : ℝ) - 1))) 1 ∧
    IsSimplexConfig (0 : ℝ) X ∧
    (∀ γ ∈ Icc (-1 : ℝ) 1, IsSimplexConfig γ (simplexRecenter X 0 γ)) ∧
    simplexRecenter X 0 0 = X ∧ Continuous (simplexRecenter X 0) := by
  let X : Idx 2 → EucSpace 2 := fun k => EuclideanSpace.single k 1
  have hX : IsSimplexConfig (0 : ℝ) X := by
    constructor
    · intro k; simp [X]
    · intro i j hij; simp [X, EuclideanSpace.inner_single_left, hij.symm]
  have hzero : (0 : ℝ) ∈ Ioo (-(1 / ((2 : ℝ) - 1))) 1 := by norm_num
  refine ⟨X, le_rfl, hzero, hX, ?_, simplexRecenter_initial le_rfl hzero X,
    continuous_simplexRecenter X 0⟩
  intro γ hγ
  exact isSimplexConfig_simplexRecenter le_rfl hzero hX (by norm_num at *; exact hγ)

end Transformer.Homogenized
