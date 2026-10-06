import Transformer.GPTMini.Sparsemax.CubeContentAttention

/-!
# Original shared values from selected content coefficients

New exact decoder following arXiv:1602.02068v2, Eq. (1). The actual
masked content sparsemax preserves every chosen bit-product feature.
Divide each learned response coefficient by its proved eigenvalue to
generate the original common value at any virtual state. A self floor
above one half bounds every divisor below by `2*floor-1`, uniformly in
the number of content bits and virtual memory slots.

The forward definition still evaluates genuine variational attention
times the generated original values. Its equality to the coefficient
response is proved. A separate local evaluation identity uses only the
query and its generated neighbors. Learned value storage is K*D for K
selected features and D output channels, not 2^r*D or a prototype table.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

variable {ι κ : Type*} [Fintype ι] [DecidableEq ι] [Fintype κ]

omit [DecidableEq ι] in
/-- Every selected content eigenvalue is bounded by the same actual routing budget.
Source: the nonnegative flip profile following sparsemax Eq. (1). -/
theorem cubeContentEigenvalue_bounds (floor : ℝ) (t : ι → ℝ)
    (ht : t ∈ cubeContentRouteDomain ι floor) (S : Finset ι) :
    2 * floor - 1 ≤ cubeContentEigenvalue t S ∧ cubeContentEigenvalue t S ≤ 1 := by
  have hn : 0 ≤ ∑ i ∈ S, t i := Finset.sum_nonneg (fun i hi => ht.1 i)
  have hs : (∑ i ∈ S, t i) ≤ ∑ i, t i :=
    Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ S) (fun i hi hnot => ht.1 i)
  unfold cubeContentEigenvalue
  constructor <;> linarith [ht.2]

/-- Two positive flips give a genuine uniformly invertible XOR feature. -/
example : 2 * (3 / 4 : ℝ) - 1 ≤ cubeContentEigenvalue (fun _ : Fin 2 => (1 / 8 : ℝ)) {0, 1} ∧
    cubeContentEigenvalue (fun _ : Fin 2 => (1 / 8 : ℝ)) {0, 1} ≤ 1 :=
  cubeContentEigenvalue_bounds (3 / 4) _
    (by constructor; intro i; norm_num; norm_num [Fin.sum_univ_two]) _

omit [DecidableEq ι] in
/-- A strict self floor makes every original content-value divisor positive.
Source: sufficient inverse conditioning following sparsemax Eq. (1). -/
theorem cubeContentEigenvalue_pos (floor : ℝ) (t : ι → ℝ) (hf : 1 / 2 < floor)
    (ht : t ∈ cubeContentRouteDomain ι floor) (S : Finset ι) : 0 < cubeContentEigenvalue t S := by
  have h := (cubeContentEigenvalue_bounds floor t ht S).1
  linarith

/-- Nonidentity content routing inhabits every positive-inverse premise. -/
example : 0 < cubeContentEigenvalue (fun _ : Fin 2 => (1 / 8 : ℝ)) {0, 1} :=
  cubeContentEigenvalue_pos (3 / 4) _ (by norm_num)
    (by constructor; intro i; norm_num; norm_num [Fin.sum_univ_two]) _

omit [DecidableEq ι] in
/-- Every content-feature inverse has the same dimension-independent amplification bound.
Source: the proved positive spectral floor following sparsemax Eq. (1). -/
theorem cubeContentInverse_bound (floor : ℝ) (t : ι → ℝ) (hf : 1 / 2 < floor)
    (ht : t ∈ cubeContentRouteDomain ι floor) (S : Finset ι) :
    1 / cubeContentEigenvalue t S ≤ 1 / (2 * floor - 1) := by
  exact one_div_le_one_div_of_le (by linarith) (cubeContentEigenvalue_bounds floor t ht S).1

/-- A positive two-bit interaction attains the uniform inverse bound. -/
example : 1 / cubeContentEigenvalue (fun _ : Fin 2 => (1 / 8 : ℝ)) {0, 1} ≤ 2 := by
  have h := cubeContentInverse_bound (3 / 4) (fun _ : Fin 2 => (1 / 8 : ℝ)) (by norm_num)
    (by constructor; intro i; norm_num; norm_num [Fin.sum_univ_two]) ({0, 1} : Finset (Fin 2))
  norm_num at h
  simpa only [one_div] using h

/-- Original common value coefficients compensate actual learned attention feature by feature.
Source: exact generated content inverse following sparsemax Eq. (1). -/
def cubeContentValueCoefficients {D : ℕ} (features : κ → Finset ι) (t : ι → ℝ)
    (W : Matrix κ (Fin D) ℝ) : Matrix κ (Fin D) ℝ :=
  fun k d => W k d / cubeContentEigenvalue t (features k)

/-- One shared original value function, generated on demand with no virtual-state rows stored.
Source: the content inverse coefficient chart following sparsemax Eq. (1). -/
def cubeContentValues {D : ℕ} (features : κ → Finset ι) (t : ι → ℝ)
    (W : Matrix κ (Fin D) ℝ) : Matrix (CubeContentState ι) (Fin D) ℝ :=
  cubeContentOutput features (cubeContentValueCoefficients features t W)

/-- Actual content queries use genuine sparsemax weights and the globally shared original values.
Source: original `attn @ v` after arXiv:1602.02068v2, Eq. (1), in the new feature chart. -/
def cubeContentForward {R D : ℕ} (code : Fin R → CubeContentState ι) (features : κ → Finset ι)
    (t : ι → ℝ) (W : Matrix κ (Fin D) ℝ) : Matrix (Fin R) (Fin D) ℝ :=
  fun r d => ∑ y, cubeContentAttention t (code r) y * cubeContentValues features t W y d

/-- Genuine sparsemax/common-value predictions equal the selected content-interaction response.
Source: proved diagonal feature action and positive inverse following sparsemax Eq. (1). -/
theorem cubeContentForward_eq {R D : ℕ} (floor : ℝ) (code : Fin R → CubeContentState ι)
    (features : κ → Finset ι) (t : ι → ℝ) (W : Matrix κ (Fin D) ℝ) (hf : 1 / 2 < floor)
    (ht : t ∈ cubeContentRouteDomain ι floor) :
    cubeContentForward code features t W = fun r d => cubeContentOutput features W (code r) d := by
  unfold cubeContentForward
  rw [cubeContentAttention_eq floor t (by linarith) ht]
  ext r d
  simp only [cubeContentValues, cubeContentOutput, Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro k hk
  simp_rw [← mul_assoc]
  rw [← Finset.sum_mul, cubeContentCharacter_action]
  unfold cubeContentValueCoefficients
  have hn : cubeContentEigenvalue t (features k) ≠ 0 :=
    ne_of_gt (cubeContentEigenvalue_pos floor t hf ht _)
  field_simp

/-- A real XOR-sign response inhabits the complete physical sparsemax/value forward premises. -/
example : cubeContentForward (fun _ : Fin 1 => fun i : Fin 2 => decide (i = 0))
    (fun _ : Fin 1 => ({0, 1} : Finset (Fin 2))) (fun _ => (1 / 8 : ℝ))
    (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ)) =
    fun _ d => cubeContentOutput (fun _ : Fin 1 => ({0, 1} : Finset (Fin 2)))
      (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ)) (fun i : Fin 2 => decide (i = 0)) d :=
  cubeContentForward_eq (3 / 4) _ _ _ _ (by norm_num)
    (by constructor; intro i; norm_num; norm_num [Fin.sum_univ_two])

/-- Local evaluation of the genuine forward reads only generated neighbor values.
Source: proved variational sparsemax equality and local routing action following Eq. (1). -/
theorem cubeContentForward_local {R D : ℕ} (floor : ℝ) (code : Fin R → CubeContentState ι)
    (features : κ → Finset ι) (t : ι → ℝ) (W : Matrix κ (Fin D) ℝ) (hf : 0 ≤ floor)
    (ht : t ∈ cubeContentRouteDomain ι floor) (r : Fin R) (d : Fin D) :
    cubeContentForward code features t W r d =
      cubeContentSelf t * cubeContentValues features t W (code r) d +
        ∑ i, t i * cubeContentValues features t W (cubeContentFlip i (code r)) d := by
  unfold cubeContentForward
  rw [cubeContentAttention_eq floor t hf ht]
  exact cubeContentRouting_action t (fun y => cubeContentValues features t W y d) (code r)

/-- Two positive neighbor values give an actual local-forward witness. -/
example : cubeContentForward (fun _ : Fin 1 => fun _ : Fin 2 => false)
    (fun _ : Fin 1 => ({0, 1} : Finset (Fin 2))) (fun _ => (1 / 8 : ℝ))
    (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ)) 0 0 =
      cubeContentSelf (fun _ : Fin 2 => (1 / 8 : ℝ)) *
        cubeContentValues (fun _ : Fin 1 => ({0, 1} : Finset (Fin 2))) (fun _ => (1 / 8 : ℝ))
          (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ)) (fun _ => false) 0 +
      ∑ i : Fin 2, (1 / 8 : ℝ) *
        cubeContentValues (fun _ : Fin 1 => ({0, 1} : Finset (Fin 2))) (fun _ => (1 / 8 : ℝ))
          (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ)) (cubeContentFlip i (fun _ => false)) 0 :=
  cubeContentForward_local (3 / 4) _ _ _ _ (by norm_num)
    (by constructor; intro i; norm_num; norm_num [Fin.sum_univ_two]) _ _

/-- Original values are genuinely inverse-adjusted for a two-bit interaction. -/
example : cubeContentValues (fun _ : Fin 1 => ({0, 1} : Finset (Fin 2)))
    (fun _ => (1 / 8 : ℝ)) (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ)) (fun _ => false) 0 = 2 := by
  norm_num [cubeContentValues, cubeContentOutput, cubeContentValueCoefficients,
    cubeContentEigenvalue, cubeContentCharacter, cubeContentSign]

end Transformer.GPTMini.Sparsemax
