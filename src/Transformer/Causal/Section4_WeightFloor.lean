/-
# Uniform positive weights in the original spherical causal system

ArXiv:2411.04990v2, §4, proof of `thm1`.
Every active original causal interaction has a fixed positive lower
bound on the original spherical state space. Cauchy--Schwarz and
the original operator norms bound all scores by
`abs(beta) * norm(Q) * norm(K)`. The numerator is therefore at least
the negative exponential of that bound and the actual masked
partition is at most `n` times its positive exponential.

The resulting explicit lower bound retains the original temperature,
query and key matrices and sample size. It applies for every real
temperature. The original mask is unchanged, so only predecessor
and current-token entries satisfy its positivity conclusion.
-/

import Transformer.Causal.Section4_ParticleWeights

open scoped BigOperators
open Real

namespace Transformer.Causal

variable {d n : ℕ}

/-- The actual uniform spherical score bound for the original query and key matrices.
Source: arXiv:2411.04990v2, §4, bounded positive original weights in the proof of `thm1`. -/
noncomputable def csaScoreBound (β : ℝ) (Q K : ParamMatrix d) : ℝ := |β| * ‖Q‖ * ‖K‖

/-- An explicit original active-weight lower bound from the original score and token-count scales.
Source: arXiv:2411.04990v2, §4, the positivity estimate for the actual `eqn:csa` field. -/
noncomputable def csaWeightFloor (β : ℝ) (Q K : ParamMatrix d) (n : ℕ) : ℝ :=
  Real.exp (-csaScoreBound β Q K) / ((n : ℝ) * Real.exp (csaScoreBound β Q K))

/-- The original score bound is nonnegative for every real temperature.
Source: arXiv:2411.04990v2, §4, the original exponential scores of `thm1`. -/
theorem csaScoreBound_nonneg (β : ℝ) (Q K : ParamMatrix d) : 0 ≤ csaScoreBound β Q K := by
  unfold csaScoreBound
  positivity

/-- The actual original spherical score is bounded by the actual original matrix operator norms.
Source: arXiv:2411.04990v2, §2, `eqn:csa`, and §4, proof of `thm1`. -/
theorem abs_csaScore_le (β : ℝ) (Q K : ParamMatrix d) (Y : Idx n → EucSpace d)
    (hY : ∀ j, ‖Y j‖ = 1) (i j : Idx n) :
    |β * inner ℝ (Q (Y i)) (K (Y j))| ≤ csaScoreBound β Q K := by
  have hQ : ‖Q (Y i)‖ ≤ ‖Q‖ := by simpa only [hY i, mul_one] using Q.le_opNorm (Y i)
  have hK : ‖K (Y j)‖ ≤ ‖K‖ := by simpa only [hY j, mul_one] using K.le_opNorm (Y j)
  rw [abs_mul]
  calc
    |β| * |inner ℝ (Q (Y i)) (K (Y j))|
        ≤ |β| * (‖Q (Y i)‖ * ‖K (Y j)‖) :=
      mul_le_mul_of_nonneg_left (abs_real_inner_le_norm _ _) (abs_nonneg _)
    _ ≤ |β| * (‖Q‖ * ‖K‖) :=
      mul_le_mul_of_nonneg_left (mul_le_mul hQ hK (norm_nonneg _) (norm_nonneg _)) (abs_nonneg _)
    _ = csaScoreBound β Q K := by unfold csaScoreBound; ring

/-- The original current token and predecessor exponentials have the original uniform two-sided bounds.
Source: arXiv:2411.04990v2, §4, the actual weights of `thm1`; no sign condition on beta is needed. -/
theorem csaExponential_bounds (β : ℝ) (Q K : ParamMatrix d) (Y : Idx n → EucSpace d)
    (hY : ∀ j, ‖Y j‖ = 1) (i j : Idx n) :
    Real.exp (-csaScoreBound β Q K) ≤ Real.exp (β * inner ℝ (Q (Y i)) (K (Y j))) ∧
      Real.exp (β * inner ℝ (Q (Y i)) (K (Y j))) ≤ Real.exp (csaScoreBound β Q K) := by
  have h := abs_le.mp (abs_csaScore_le β Q K Y hY i j)
  exact ⟨Real.exp_le_exp.mpr h.1, Real.exp_le_exp.mpr h.2⟩

/-- The actual masked original partition is at most the token count times the original score exponential bound.
Source: arXiv:2411.04990v2, §4, the original normalization in the proof of `thm1`. -/
theorem csaPartition_upper (β : ℝ) (Q K : ParamMatrix d) (Y : Idx n → EucSpace d)
    (hY : ∀ j, ‖Y j‖ = 1) (i : Idx n) :
    csaPartition β Q K Y i ≤ (n : ℝ) * Real.exp (csaScoreBound β Q K) := by
  calc
    csaPartition β Q K Y i ≤ ∑ _j : Idx n, Real.exp (csaScoreBound β Q K) := by
      apply Finset.sum_le_sum
      intro j hj
      split_ifs
      · exact (csaExponential_bounds β Q K Y hY i j).2
      · exact (Real.exp_pos _).le
    _ = (n : ℝ) * Real.exp (csaScoreBound β Q K) := by simp

/-- The explicit original active-weight lower bound is positive at every positive original sample size.
Source: arXiv:2411.04990v2, §4, the strict positivity used for `thm1`. -/
theorem csaWeightFloor_pos (β : ℝ) (Q K : ParamMatrix d) (hn : 0 < n) :
    0 < csaWeightFloor β Q K n := by
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  exact div_pos (Real.exp_pos _) (mul_pos hn' (Real.exp_pos _))

/-- The original self exponential also gives a uniform positive lower bound for the actual partition.
Source: arXiv:2411.04990v2, §4, the positive normalization of the original `eqn:csa` field. -/
theorem csaPartition_lower (β : ℝ) (Q K : ParamMatrix d) (Y : Idx n → EucSpace d)
    (hY : ∀ j, ‖Y j‖ = 1) (i : Idx n) :
    Real.exp (-csaScoreBound β Q K) ≤ csaPartition β Q K Y i := by
  calc
    _ ≤ Real.exp (β * inner ℝ (Q (Y i)) (K (Y i))) :=
      (csaExponential_bounds β Q K Y hY i i).1
    _ ≤ csaPartition β Q K Y i := by
      unfold csaPartition
      have h := Finset.single_le_sum (s := Finset.univ)
        (f := fun j : Idx n => if (j : ℕ) ≤ (i : ℕ) then
          Real.exp (β * inner ℝ (Q (Y i)) (K (Y j))) else 0)
        (fun j hj => by split_ifs <;> positivity) (Finset.mem_univ i)
      simpa only [le_refl, ite_true] using h

/-- Every actual original active weight is uniformly bounded below on the entire original spherical state space.
Source: arXiv:2411.04990v2, §4, the bounded positive weights in the proof of `thm1`. -/
theorem csaWeightFloor_le (β : ℝ) (Q K : ParamMatrix d) (Y : Idx n → EucSpace d)
    (hY : ∀ j, ‖Y j‖ = 1) (i j : Idx n) (hji : (j : ℕ) ≤ (i : ℕ)) :
    csaWeightFloor β Q K n ≤ csaTokenWeight β Q K Y i j := by
  have hn : 0 < n := Fin.pos_iff_nonempty.mpr ⟨i⟩
  have hnf : (n : ℝ) * Real.exp (csaScoreBound β Q K) ≠ 0 :=
    mul_ne_zero (Nat.cast_ne_zero.mpr hn.ne') (Real.exp_ne_zero _)
  rw [csaTokenWeight, ite_eq_left hji]
  apply (le_div_iff₀ (csaPartition_pos β Q K Y i)).mpr
  calc
    csaWeightFloor β Q K n * csaPartition β Q K Y i
        ≤ csaWeightFloor β Q K n * ((n : ℝ) * Real.exp (csaScoreBound β Q K)) :=
      mul_le_mul_of_nonneg_left (csaPartition_upper β Q K Y hY i) (csaWeightFloor_pos β Q K hn).le
    _ = Real.exp (-csaScoreBound β Q K) := by
      unfold csaWeightFloor
      field_simp
    _ ≤ Real.exp (β * inner ℝ (Q (Y i)) (K (Y j))) := (csaExponential_bounds β Q K Y hY i j).1

/-- Every original causal weight is at most one, including entries zeroed by the actual mask.
Source: arXiv:2411.04990v2, §4, the original probability rows of `thm1`. -/
theorem csaTokenWeight_le_one (β : ℝ) (Q K : ParamMatrix d)
    (Y : Idx n → EucSpace d) (i j : Idx n) : csaTokenWeight β Q K Y i j ≤ 1 := by
  have h := Finset.single_le_sum (s := Finset.univ)
    (fun l hl => csaTokenWeight_nonneg β Q K Y i l) (Finset.mem_univ j)
  rwa [sum_csaTokenWeight] at h

/-- All genuine predecessors together retain their original count times the uniform active-weight floor.
Source: arXiv:2411.04990v2, §4, the predecessor force in the induction proving `thm1`.
The actual last-row self weight remains in the original denominator. -/
theorem csaPredecessorWeight_lower (β : ℝ) (Q K : ParamMatrix d)
    (Y : Idx (n + 1) → EucSpace d) (hY : ∀ j, ‖Y j‖ = 1) :
    (n : ℝ) * csaWeightFloor β Q K (n + 1) ≤
      ∑ j : Idx n, csaTokenWeight β Q K Y (Fin.last n) j.castSucc := by
  calc
    _ = ∑ _j : Idx n, csaWeightFloor β Q K (n + 1) := by simp
    _ ≤ _ := by
      apply Finset.sum_le_sum
      intro j hj
      exact csaWeightFloor_le β Q K Y hY _ _ (Nat.le_of_lt j.isLt)

/-- The full spherical, active-mask and positive-size hypotheses hold together for two nonzero original tokens. -/
example : (0 : ℕ) < 2 ∧ (0 : ℕ) ≤ 1 ∧
    (∀ j : Idx 2, ‖(fun _ => (basePoint 1 : EucSpace 2)) j‖ = 1) :=
  ⟨by norm_num, by norm_num, fun _ => mem_sphere_zero_iff_norm.mp (basePoint 1).2⟩

end Transformer.Causal
