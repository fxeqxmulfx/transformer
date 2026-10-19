module

public import Transformer.Causal.Basic
public import Mathlib.Analysis.Calculus.ContDiff.RCLike
public import Mathlib.Analysis.InnerProductSpace.Calculus
public import Mathlib.Analysis.SpecialFunctions.ExpDeriv

/-!
# The original finite causal weights on the ambient token space

ArXiv:2411.04990v2, §2, `eqn:csa`, and §4, proof of `thm1`.
The original mask retains exactly the current token and its predecessors.
The current token makes every row partition strictly positive, even
before restricting tokens to the sphere. The actual row weights are
nonnegative, sum to one, and depend smoothly on the original tokens.

The original value average is therefore bounded by the operator norm
of the actual value matrix times the maximum token norm. Its exact
equality with the partition-inverse expression in the manuscript is
proved, including every original masked term. These properties are
the local smoothness and growth inputs for the causal flow construction.
No dynamics, convergence, or regularity is put into these definitions.
-/

@[expose] public section

open scoped BigOperators
open Real

namespace Transformer.Causal

variable {d n : ℕ}

/-- The original row partition, including its own current-token term.
Source: arXiv:2411.04990v2, §2, `eqn:csa`, and §4, proof of `thm1`. -/
noncomputable def csaPartition (β : ℝ) (Q K : ParamMatrix d)
    (Y : Idx n → EucSpace d) (i : Idx n) : ℝ :=
  ∑ j : Idx n, if (j : ℕ) ≤ (i : ℕ) then Real.exp (β * inner ℝ (Q (Y i)) (K (Y j))) else 0

/-- The actual normalized original causal row weight, zero above its causal mask.
Source: arXiv:2411.04990v2, §2, `eqn:csa`; no artificial unmasked scores are used. -/
noncomputable def csaTokenWeight (β : ℝ) (Q K : ParamMatrix d)
    (Y : Idx n → EucSpace d) (i j : Idx n) : ℝ :=
  if (j : ℕ) ≤ (i : ℕ) then
    Real.exp (β * inner ℝ (Q (Y i)) (K (Y j))) / csaPartition β Q K Y i else 0

/-- The original value average before its tangent projection.
Source: arXiv:2411.04990v2, §2, `eqn:csa`; the value matrix is retained exactly. -/
noncomputable def csaAverage (β : ℝ) (Q K V : ParamMatrix d)
    (Y : Idx n → EucSpace d) (i : Idx n) : EucSpace d :=
  ∑ j : Idx n, csaTokenWeight β Q K Y i j • V (Y j)

/-- Every original row partition is positive because its own original exponential is present.
Source: arXiv:2411.04990v2, §2, `eqn:csa`, and §4, proof of `thm1`. -/
theorem csaPartition_pos (β : ℝ) (Q K : ParamMatrix d) (Y : Idx n → EucSpace d) (i : Idx n) :
    0 < csaPartition β Q K Y i := by
  apply Finset.sum_pos'
  · intro j hj
    split_ifs <;> positivity
  · exact ⟨i, Finset.mem_univ i, by simp only [le_refl, ite_eq_left]; positivity⟩

/-- The original causal weight is nonnegative at every ambient tuple.
Source: arXiv:2411.04990v2, §2, `eqn:csa`. -/
theorem csaTokenWeight_nonneg (β : ℝ) (Q K : ParamMatrix d)
    (Y : Idx n → EucSpace d) (i j : Idx n) : 0 ≤ csaTokenWeight β Q K Y i j := by
  unfold csaTokenWeight
  split_ifs
  · exact div_nonneg (Real.exp_pos _).le (csaPartition_pos β Q K Y i).le
  · exact le_rfl

/-- Every predecessor and the current token has strictly positive original weight.
Source: arXiv:2411.04990v2, §2, `eqn:csa`, the positivity used in the proof of `thm1`. -/
theorem csaTokenWeight_pos (β : ℝ) (Q K : ParamMatrix d) (Y : Idx n → EucSpace d)
    (i j : Idx n) (hji : (j : ℕ) ≤ (i : ℕ)) : 0 < csaTokenWeight β Q K Y i j := by
  rw [csaTokenWeight, ite_eq_left hji]
  exact div_pos (Real.exp_pos _) (csaPartition_pos β Q K Y i)

/-- The original causal row weights sum to one with their actual partition function.
Source: arXiv:2411.04990v2, §2, `eqn:csa`. -/
theorem sum_csaTokenWeight (β : ℝ) (Q K : ParamMatrix d) (Y : Idx n → EucSpace d) (i : Idx n) :
    ∑ j, csaTokenWeight β Q K Y i j = 1 := by
  have h (j : Idx n) : csaTokenWeight β Q K Y i j =
      (if (j : ℕ) ≤ (i : ℕ) then Real.exp (β * inner ℝ (Q (Y i)) (K (Y j))) else 0) /
        csaPartition β Q K Y i := by
    by_cases hji : (j : ℕ) ≤ (i : ℕ) <;> simp [csaTokenWeight, hji]
  simp_rw [h]
  rw [← Finset.sum_div]
  exact div_self (csaPartition_pos β Q K Y i).ne'

/-- The original value average is exactly the manuscript's masked partition-inverse sum.
Source: arXiv:2411.04990v2, §2, `eqn:csa`, with its original normalization. -/
theorem csaAverage_eq_original (β : ℝ) (Q K V : ParamMatrix d)
    (Y : Idx n → EucSpace d) (i : Idx n) : csaAverage β Q K V Y i =
      (csaPartition β Q K Y i)⁻¹ • ∑ j : Idx n,
        if (j : ℕ) ≤ (i : ℕ) then Real.exp (β * inner ℝ (Q (Y i)) (K (Y j))) • V (Y j) else 0 := by
  rw [csaAverage, Finset.smul_sum]
  apply Finset.sum_congr rfl
  intro j hj
  by_cases hji : (j : ℕ) ≤ (i : ℕ)
  · simp only [csaTokenWeight, ite_eq_left hji, div_eq_mul_inv, smul_smul]
    rw [mul_comm]
  · simp only [csaTokenWeight, ite_eq_right hji, zero_smul, smul_zero]

/-- Each original unmasked exponential term is smooth in all ambient token coordinates.
Source: arXiv:2411.04990v2, §4, the smooth original field in the proof of `thm1`. -/
theorem contDiff_csaExponential (β : ℝ) (Q K : ParamMatrix d) (i j : Idx n) :
    ContDiff ℝ (⊤ : ℕ∞) (fun Y : Idx n → EucSpace d =>
      Real.exp (β * inner ℝ (Q (Y i)) (K (Y j)))) := by
  have hQ : ContDiff ℝ (⊤ : ℕ∞) (fun Y : Idx n → EucSpace d => Q (Y i)) := by fun_prop
  have hK : ContDiff ℝ (⊤ : ℕ∞) (fun Y : Idx n → EucSpace d => K (Y j)) := by fun_prop
  exact (contDiff_const.mul (hQ.inner ℝ hK)).exp

/-- The actual masked row partition is smooth in all original ambient tokens.
Source: arXiv:2411.04990v2, §4, the smooth original field in the proof of `thm1`. -/
theorem contDiff_csaPartition (β : ℝ) (Q K : ParamMatrix d) (i : Idx n) :
    ContDiff ℝ (⊤ : ℕ∞) (fun Y : Idx n → EucSpace d => csaPartition β Q K Y i) := by
  apply ContDiff.sum
  intro j hj
  split_ifs
  · exact contDiff_csaExponential β Q K i j
  · exact contDiff_const

/-- Each actual original weight is smooth in all ambient original token coordinates.
Source: arXiv:2411.04990v2, §4, the smooth autonomous system in the proof of `thm1`. -/
theorem contDiff_csaTokenWeight (β : ℝ) (Q K : ParamMatrix d) (i j : Idx n) :
    ContDiff ℝ (⊤ : ℕ∞) (fun Y : Idx n → EucSpace d => csaTokenWeight β Q K Y i j) := by
  unfold csaTokenWeight
  split_ifs
  · exact ContDiff.div (contDiff_csaExponential β Q K i j) (contDiff_csaPartition β Q K i)
      (fun Y => (csaPartition_pos β Q K Y i).ne')
  · exact contDiff_const

/-- The actual value average is smooth before projection to the sphere.
Source: arXiv:2411.04990v2, §4, the smooth original field of `thm1`. -/
theorem contDiff_csaAverage (β : ℝ) (Q K V : ParamMatrix d) (i : Idx n) :
    ContDiff ℝ (⊤ : ℕ∞) (fun Y : Idx n → EucSpace d => csaAverage β Q K V Y i) := by
  exact ContDiff.sum fun j hj => (contDiff_csaTokenWeight β Q K i j).smul (by fun_prop)

/-- The original masked average obeys the actual value-matrix operator-norm bound.
Source: arXiv:2411.04990v2, §2, `eqn:csa`, the convex row average in §4's proof of `thm1`. -/
theorem norm_csaAverage_le (β : ℝ) (Q K V : ParamMatrix d)
    (Y : Idx n → EucSpace d) (i : Idx n) : ‖csaAverage β Q K V Y i‖ ≤ ‖V‖ * ‖Y‖ := by
  refine (norm_sum_le _ _).trans ?_
  calc
    ∑ j, ‖csaTokenWeight β Q K Y i j • V (Y j)‖
        ≤ ∑ j, csaTokenWeight β Q K Y i j * (‖V‖ * ‖Y‖) := by
      apply Finset.sum_le_sum
      intro j hj
      rw [norm_smul, Real.norm_of_nonneg (csaTokenWeight_nonneg β Q K Y i j)]
      exact mul_le_mul_of_nonneg_left
        ((V.le_opNorm (Y j)).trans (mul_le_mul_of_nonneg_left (norm_le_pi_norm Y j) (norm_nonneg V)))
        (csaTokenWeight_nonneg β Q K Y i j)
    _ = ‖V‖ * ‖Y‖ := by rw [← Finset.sum_mul, sum_csaTokenWeight, one_mul]

/-- An actual current-token mask and a nonzero original token jointly realize every positive-weight premise. -/
example : (0 : ℕ) ≤ (0 : ℕ) ∧
    csaTokenWeight 1 (ContinuousLinearMap.id ℝ (EucSpace 2)) (ContinuousLinearMap.id ℝ _)
      (fun _ : Idx 1 => (basePoint 1 : EucSpace 2)) 0 0 = 1 := by
  constructor
  · exact le_rfl
  · simp [csaTokenWeight, csaPartition]

end Transformer.Causal
