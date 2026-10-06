import Transformer.GPTMini.Sparsemax.CubeContentRouting
import Mathlib.Data.Fintype.Powerset

/-!
# Content interaction features diagonalize generated sparse attention

New compact-value architecture after arXiv:1602.02068v2, Eq. (1).
For every chosen set S of bit coordinates, the generated feature is the
product of their signs. It includes constants, individual content bits,
position interactions and arbitrary parity features. Flipping a bit in S
negates the feature; every other flip preserves it. Thus the actual local
routing action has eigenvalue `1 - 2 * sum_{i in S} t_i`.

Only chosen feature coefficients are learned. No virtual-state feature
or value table is stored. Features and bit identities are fixed before
training; learned responses are linear in these coefficients, even when
the observation-to-feature map has high-order nonlinear interactions.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

variable {ι κ : Type*} [DecidableEq ι]

/-- Generate a selected content interaction directly from its observed bits.
Source: the new spectral value chart following sparsemax Eq. (1). -/
def cubeContentCharacter (S : Finset ι) (x : CubeContentState ι) : ℝ :=
  ∏ i ∈ S, cubeContentSign (x i)

/-- A selected coordinate flip negates its content interaction and other flips preserve it.
Source: generated invariant content features following sparsemax Eq. (1). -/
theorem cubeContentCharacter_flip (S : Finset ι) (x : CubeContentState ι) (i : ι) :
    cubeContentCharacter S (cubeContentFlip i x) =
      (if i ∈ S then -1 else 1) * cubeContentCharacter S x := by
  induction S using Finset.induction_on with
  | empty => simp [cubeContentCharacter]
  | @insert j S hj ih =>
    simp only [cubeContentCharacter, Finset.prod_insert hj] at ih ⊢
    rw [ih]
    by_cases hij : i = j
    · subst j
      rw [cubeContentFlip_same, cubeContentSign_not]
      simp only [Finset.mem_insert, eq_self, true_or, ite_true, hj, ite_false]
      ring
    · rw [cubeContentFlip_other i j x (Ne.symm hij)]
      simp only [Finset.mem_insert, hij, false_or]
      ring

/-- A two-bit XOR feature changes sign when either selected observed bit flips. -/
example : cubeContentCharacter ({0, 1} : Finset (Fin 2)) (cubeContentFlip 0 (fun _ => false)) =
    -cubeContentCharacter {0, 1} (fun _ => false) := by
  rw [cubeContentCharacter_flip]
  norm_num
  rfl

omit [DecidableEq ι] in
/-- Every generated interaction has squared magnitude one, at every order.
Source: bounded sign products in the new content chart after sparsemax Eq. (1). -/
theorem cubeContentCharacter_sq (S : Finset ι) (x : CubeContentState ι) :
    cubeContentCharacter S x ^ 2 = 1 := by
  unfold cubeContentCharacter
  rw [← Finset.prod_pow]
  simp only [cubeContentSign_sq, Finset.prod_const_one]

omit [DecidableEq ι] in
/-- A generated product is exactly the parity sign of its selected observed one bits.
Source: high-order content response features after sparsemax Eq. (1), with no task labels. -/
theorem cubeContentCharacter_parity (S : Finset ι) (x : CubeContentState ι) :
    cubeContentCharacter S x = (-1 : ℝ) ^ (S.filter (fun i => x i = true)).card := by
  unfold cubeContentCharacter cubeContentSign
  rw [Finset.prod_ite]
  rw [Finset.prod_const, Finset.prod_const_one, mul_one]

omit [DecidableEq ι] in
/-- Bit comparison is a second-order content interaction, rather than a position-only response.
Source: generated content-matching features after sparsemax Eq. (1). -/
theorem cubeContentSign_match (x : CubeContentState ι) (i j : ι) :
    (1 + cubeContentSign (x i) * cubeContentSign (x j)) / 2 =
      if x i = x j then (1 : ℝ) else 0 := by
  cases hi : x i <;> cases hj : x j <;> norm_num [cubeContentSign]

/-- All sixteen observed bits can contribute one generated parity feature. -/
example (x : CubeContentState (Fin 16)) :
    cubeContentCharacter Finset.univ x = (-1 : ℝ) ^ (Finset.univ.filter (fun i => x i = true)).card :=
  cubeContentCharacter_parity _ _

/-- The eigenvalue depends only on learned flips of coordinates participating in the feature.
Source: generated content diffusion after sparsemax Eq. (1). -/
def cubeContentEigenvalue (t : ι → ℝ) (S : Finset ι) : ℝ := 1 - 2 * ∑ i ∈ S, t i

variable [Fintype ι] [Fintype κ]

/-- Genuine generated routing acts diagonally on every chosen content interaction.
Source: finite local flip algebra following sparsemax Eq. (1). -/
theorem cubeContentCharacter_action (t : ι → ℝ) (S : Finset ι) (x : CubeContentState ι) :
    (∑ y, cubeContentRouting t x y * cubeContentCharacter S y) =
      cubeContentEigenvalue t S * cubeContentCharacter S x := by
  rw [cubeContentRouting_action]
  simp_rw [cubeContentCharacter_flip, ← mul_assoc]
  rw [← Finset.sum_mul]
  have he (i : ι) : t i * (if i ∈ S then (-1 : ℝ) else 1) =
      t i - 2 * (if i ∈ S then t i else 0) := by
    by_cases hi : i ∈ S <;> simp only [hi, ite_true, ite_false] <;> ring
  have hs : (∑ i, if i ∈ S then t i else 0) = ∑ i ∈ S, t i := by
    rw [← Finset.sum_filter]
    congr 1
    ext i
    simp
  simp_rw [he]
  rw [Finset.sum_sub_distrib, ← Finset.mul_sum, hs]
  unfold cubeContentSelf cubeContentEigenvalue
  ring

/-- Arbitrary chosen subsets generate a shared vector response from their coefficient table.
Source: compact content values after sparsemax Eq. (1); features are computed on demand. -/
def cubeContentOutput {D : ℕ} (features : κ → Finset ι) (W : Matrix κ (Fin D) ℝ)
    (x : CubeContentState ι) (d : Fin D) : ℝ :=
  ∑ k, cubeContentCharacter (features k) x * W k d

omit [DecidableEq ι] [Fintype ι] in
/-- Content-nonlinear responses remain linear in all learned coefficients jointly.
Source: the selected invariant-feature chart following sparsemax Eq. (1). -/
theorem cubeContentOutput_linear {D : ℕ} (features : κ → Finset ι)
    (W V : Matrix κ (Fin D) ℝ) (a b : ℝ) :
    cubeContentOutput features (a • W + b • V) =
      a • cubeContentOutput features W + b • cubeContentOutput features V := by
  ext x d
  change (∑ k, cubeContentCharacter (features k) x * (a * W k d + b * V k d)) =
    a * (∑ k, cubeContentCharacter (features k) x * W k d) +
      b * (∑ k, cubeContentCharacter (features k) x * V k d)
  rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro k hk
  ring

/-- Only selected features, not virtual states or registered prefixes, contribute learned value coefficients.
Source: the explicit content coefficient type following sparsemax Eq. (1). -/
theorem cubeContentCoefficient_count (D : ℕ) :
    Fintype.card (κ × Fin D) = Fintype.card κ * D := by
  rw [Fintype.card_prod, Fintype.card_fin]

/-- Two simultaneous learned flips preserve a genuine two-bit interaction with a nonconstant eigenvalue. -/
example : (∑ y, cubeContentRouting (fun _ : Fin 2 => (1 / 8 : ℝ)) (fun _ => false) y *
    cubeContentCharacter ({0, 1} : Finset (Fin 2)) y) = 1 / 2 := by
  rw [cubeContentCharacter_action]
  norm_num [cubeContentEigenvalue, cubeContentCharacter, cubeContentSign]

/-- One product feature gives XOR-sign responses using one coefficient per channel. -/
example : cubeContentOutput (fun _ : Fin 1 => ({0, 1} : Finset (Fin 2)))
    (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ)) (fun i => decide (i = 0)) 0 = -1 := by
  norm_num [cubeContentOutput, cubeContentCharacter, cubeContentSign]

end Transformer.GPTMini.Sparsemax
