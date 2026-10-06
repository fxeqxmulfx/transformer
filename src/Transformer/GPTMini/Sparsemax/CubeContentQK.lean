import Transformer.GPTMini.Sparsemax.CubeContentFeatures

/-!
# Actual learned content Q/K at width one plus the bit count

New physical embeddings for sparsemax arXiv:1602.02068v2, Eq. (1).
Keys and queries use a constant coordinate and the signs of the actual
content bits. A learned shared scale changes both embedding families.
The score coefficients realize the desired self mass and every separate
flip mass on the structural local mask. No exponential embedding table
or assumed Gram factorization is used.

Attention is over virtual content neighbors, not input-token occurrences.
The causal encoder supplies the observed state; virtual destination bits
are generated locally. This is a content-sensitive memory architecture,
not unrestricted QKNorm/RoPE attention or a jointly trained layer stack.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- A strictly positive learned physical scale for both content embedding families.
Source: new generated Q/K before sparsemax Eq. (1). -/
def cubeContentScale (t : ι → ℝ) : ℝ := 1 + (∑ i, t i) ^ 2

omit [DecidableEq ι] in
/-- Learned scalar products are well-defined at every finite routing point.
Source: the explicit physical scale before sparsemax Eq. (1). -/
theorem cubeContentScale_pos (t : ι → ℝ) : 0 < cubeContentScale t := by
  unfold cubeContentScale
  nlinarith [sq_nonneg (∑ i, t i)]

/-- One score coefficient per content coordinate generates its chosen flip mass.
Source: physical score construction before sparsemax Eq. (1). -/
def cubeContentScoreCoefficient (t : ι → ℝ) (i : ι) : ℝ := (cubeContentSelf t - t i) / 2

/-- Actual queries encode content signs and a shared learned scale.
Source: explicit physical Q before sparsemax Eq. (1). -/
def cubeContentQuery (t : ι → ℝ) (x : CubeContentState ι) : Option ι → ℝ
  | none => cubeContentScale t
  | some i => cubeContentScale t * cubeContentSign (x i)

/-- Shared original keys are generated from destination content and learned coefficients.
Source: explicit physical K before sparsemax Eq. (1). -/
def cubeContentKey (t : ι → ℝ) (y : CubeContentState ι) : Option ι → ℝ
  | none => (cubeContentSelf t - ∑ i, cubeContentScoreCoefficient t i) / cubeContentScale t
  | some i => cubeContentScoreCoefficient t i * cubeContentSign (y i) / cubeContentScale t

/-- Genuine Q/K dot products over one plus the observed bit count coordinates.
Source: physical content attention scores before sparsemax Eq. (1). -/
def cubeContentScore (t : ι → ℝ) (x y : CubeContentState ι) : ℝ :=
  ∑ d, cubeContentQuery t x d * cubeContentKey t y d

omit [DecidableEq ι] in
/-- The actual generated dot product has the stated content-dependent expansion.
Source: explicit finite Q/K algebra before sparsemax Eq. (1). -/
theorem cubeContentScore_formula (t : ι → ℝ) (x y : CubeContentState ι) :
    cubeContentScore t x y = cubeContentSelf t - ∑ i, cubeContentScoreCoefficient t i +
      ∑ i, cubeContentScoreCoefficient t i * (cubeContentSign (x i) * cubeContentSign (y i)) := by
  have hn : cubeContentScale t ≠ 0 := ne_of_gt (cubeContentScale_pos t)
  unfold cubeContentScore
  rw [Fintype.sum_option]
  simp only [cubeContentQuery, cubeContentKey]
  have h0 : cubeContentScale t *
      ((cubeContentSelf t - ∑ i, cubeContentScoreCoefficient t i) / cubeContentScale t) =
        cubeContentSelf t - ∑ i, cubeContentScoreCoefficient t i := by field_simp
  rw [h0]
  congr 1
  apply Finset.sum_congr rfl
  intro i hi
  field_simp

omit [DecidableEq ι] in
/-- Original scalar products give the exact desired self probability score.
Source: the content score construction before variational sparsemax Eq. (1). -/
theorem cubeContentScore_self (t : ι → ℝ) (x : CubeContentState ι) :
    cubeContentScore t x x = cubeContentSelf t := by
  rw [cubeContentScore_formula]
  have hs (i : ι) : cubeContentSign (x i) * cubeContentSign (x i) = 1 := by
    nlinarith [cubeContentSign_sq (x i)]
  simp_rw [hs, mul_one]
  ring

/-- Original scalar products give each independently learned neighbor score exactly.
Source: content-coordinate flip scores before variational sparsemax Eq. (1). -/
theorem cubeContentScore_flip (t : ι → ℝ) (x : CubeContentState ι) (i : ι) :
    cubeContentScore t x (cubeContentFlip i x) = t i := by
  rw [cubeContentScore_formula]
  have he (j : ι) : cubeContentScoreCoefficient t j *
      (cubeContentSign (x j) * cubeContentSign (cubeContentFlip i x j)) =
        cubeContentScoreCoefficient t j - 2 * (if i = j then cubeContentScoreCoefficient t j else 0) := by
    by_cases h : i = j
    · subst j
      rw [cubeContentFlip_same, cubeContentSign_not]
      simp only [ite_true]
      cases hx : x i <;> norm_num [cubeContentSign, hx] <;> ring
    · rw [cubeContentFlip_other i j x (Ne.symm h)]
      simp only [h, ite_false, mul_zero, sub_zero]
      cases hx : x j <;> norm_num [cubeContentSign, hx]
  simp_rw [he]
  rw [Finset.sum_sub_distrib, ← Finset.mul_sum, Fintype.sum_ite_eq]
  unfold cubeContentScoreCoefficient
  ring

omit [DecidableEq ι] in
/-- Physical width depends on input bits, not on the exponentially many virtual states.
Source: the explicit constant-plus-content Q/K type before sparsemax Eq. (1). -/
theorem cubeContentQK_width : Fintype.card (Option ι) = Fintype.card ι + 1 := by
  rw [Fintype.card_option]

omit [DecidableEq ι] in
/-- Actual query squared norm grows with content width, with no virtual-state dimension factor.
Source: explicit physical Q coordinates before sparsemax Eq. (1). -/
theorem cubeContentQuery_sq (t : ι → ℝ) (x : CubeContentState ι) :
    (∑ d, (cubeContentQuery t x d) ^ 2) = cubeContentScale t ^ 2 * (Fintype.card ι + 1) := by
  rw [Fintype.sum_option]
  simp only [cubeContentQuery, mul_pow, cubeContentSign_sq, mul_one]
  rw [Finset.sum_const]
  simp only [nsmul_eq_mul, Finset.card_univ]
  ring

/-- Two content bits give genuine width-three queries and four virtual states. -/
example : (∑ d, (cubeContentQuery (fun _ : Fin 2 => (1 / 8 : ℝ)) (fun _ => false) d) ^ 2) =
    cubeContentScale (fun _ : Fin 2 => (1 / 8 : ℝ)) ^ 2 * 3 := by
  rw [cubeContentQuery_sq, Fintype.card_fin]
  norm_num

/-- A query and two distinct possible neighbors have different genuine scores. -/
example : cubeContentScore (fun i : Fin 2 => if i = 0 then (1 / 8 : ℝ) else 1 / 16)
    (fun _ => false) (cubeContentFlip 0 (fun _ => false)) = 1 / 8 ∧
    cubeContentScore (fun i : Fin 2 => if i = 0 then (1 / 8 : ℝ) else 1 / 16)
      (fun _ => false) (cubeContentFlip 1 (fun _ => false)) = 1 / 16 := by
  constructor <;> rw [cubeContentScore_flip] <;> norm_num

/-- Learned routing changes actual queries, even before any value decoding. -/
example : cubeContentQuery (0 : Fin 1 → ℝ) (fun _ => false) none ≠
    cubeContentQuery (fun _ : Fin 1 => (1 / 8 : ℝ)) (fun _ => false) none := by
  norm_num [cubeContentQuery, cubeContentScale]

/-- The same routing change also changes actual shared keys. -/
example : cubeContentKey (0 : Fin 1 → ℝ) (fun _ => false) (some 0) ≠
    cubeContentKey (fun _ : Fin 1 => (1 / 8 : ℝ)) (fun _ => false) (some 0) := by
  norm_num [cubeContentKey, cubeContentScoreCoefficient, cubeContentSelf, cubeContentScale, cubeContentSign]

end Transformer.GPTMini.Sparsemax
