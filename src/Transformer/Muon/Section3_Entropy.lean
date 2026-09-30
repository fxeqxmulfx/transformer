/-
# Muon — spectral entropy

arXiv:2502.16982, §3.4, “Dynamics of Singular Spectrum”. Squared singular
values are normalized to probabilities. For at least two singular values
and positive energy the reported normalized entropy lies in `[0,1]`.
The comparisons of actual training checkpoints are empirical observations.
-/

import Transformer.Muon.Section2_Models
import Mathlib.Analysis.SpecialFunctions.Log.NegMulLog

open scoped BigOperators

noncomputable section

namespace Transformer.Muon

variable {n : ℕ}

/-- Normalized squared singular values, arXiv:2502.16982, §3.4. -/
def spectralWeights (s : Fin n → ℝ) (i : Fin n) : ℝ := s i ^ 2 / ∑ j, s j ^ 2

/-- SVD entropy as printed in arXiv:2502.16982, §3.4.
`negMulLog 0 = 0` implements the usual zero-weight convention. The
probabilistic interpretation requires positive energy and `n ≥ 2`. -/
def spectralEntropy (s : Fin n → ℝ) : ℝ :=
  (∑ i, Real.negMulLog (spectralWeights s i)) / Real.log n

/-- The spectral weights are probabilities at positive energy,
arXiv:2502.16982, §3.4. -/
theorem spectralWeights_probability (s : Fin n → ℝ) (hs : 0 < ∑ j, s j ^ 2) :
    (∀ i, 0 ≤ spectralWeights s i) ∧ (∑ i, spectralWeights s i) = 1 := by
  refine ⟨fun i => div_nonneg (sq_nonneg _) hs.le, ?_⟩
  simp only [spectralWeights, ← Finset.sum_div]
  exact div_self hs.ne'

/-- Positive spectral energy exists, arXiv:2502.16982, §3.4. -/
example : 0 < ∑ j : Fin 2, (fun _ : Fin 2 => (1 : ℝ)) j ^ 2 := by norm_num

/-- Scaling identity for entropy contributions, arXiv:2502.16982, §3.4. -/
theorem negMulLog_scale (c p : ℝ) (hc : 0 < c) (hp : 0 ≤ p) :
    Real.negMulLog (c * p) = c * Real.negMulLog p - c * p * Real.log c := by
  rcases eq_or_lt_of_le hp with h | h
  · subst p; simp
  · rw [Real.negMulLog, Real.negMulLog, Real.log_mul hc.ne' h.ne']
    ring

/-- The scaling hypotheses are satisfiable, arXiv:2502.16982, §3.4. -/
example : (0 : ℝ) < 2 ∧ (0 : ℝ) ≤ 1 := by norm_num

/-- Shannon entropy of a finite probability vector is between zero and
`log n`, arXiv:2502.16982, §3.4, the normalization in its entropy formula. -/
theorem finite_entropy_bounds (p : Fin n → ℝ) (hn : 0 < n)
    (hp : ∀ i, 0 ≤ p i) (hsum : ∑ i, p i = 1) :
    0 ≤ ∑ i, Real.negMulLog (p i) ∧ ∑ i, Real.negMulLog (p i) ≤ Real.log n := by
  have hn' : 0 < (n : ℝ) := by exact_mod_cast hn
  constructor
  · apply Finset.sum_nonneg
    intro i hi
    apply Real.negMulLog_nonneg (hp i)
    rw [← hsum]
    exact Finset.single_le_sum (fun j _ => hp j) (Finset.mem_univ i)
  · have h := Finset.sum_le_sum (s := Finset.univ) fun i _ =>
      Real.negMulLog_le_one_sub_self (mul_nonneg hn'.le (hp i))
    have heq : (∑ i, Real.negMulLog ((n : ℝ) * p i)) =
        (n : ℝ) * (∑ i, Real.negMulLog (p i)) - (n : ℝ) * Real.log n := by
      simp only [negMulLog_scale _ _ hn' (hp _), Finset.sum_sub_distrib]
      rw [← Finset.mul_sum]
      simp only [mul_assoc, ← Finset.mul_sum, ← Finset.sum_mul, hsum, one_mul]
    rw [heq] at h
    have hright : (∑ i : Fin n, (1 - (n : ℝ) * p i)) = 0 := by
      simp [Finset.sum_sub_distrib, ← Finset.mul_sum, hsum]
    rw [hright] at h
    nlinarith

/-- Finite probability vectors exist, arXiv:2502.16982, §3.4. -/
example : 0 < (2 : ℕ) ∧ (∀ i : Fin 2, (0 : ℝ) ≤ (fun _ : Fin 2 => (1 / 2 : ℝ)) i) ∧
    (∑ i : Fin 2, (fun _ : Fin 2 => (1 / 2 : ℝ)) i) = 1 := by norm_num

/-- The SVD entropy is in `[0,1]` on the meaningful domain of the paper's
formula. Source: arXiv:2502.16982, §3.4, “Dynamics of Singular Spectrum”. -/
theorem spectralEntropy_bounds (s : Fin n → ℝ) (hn : 1 < n) (hs : 0 < ∑ j, s j ^ 2) :
    0 ≤ spectralEntropy s ∧ spectralEntropy s ≤ 1 := by
  have hprob := spectralWeights_probability s hs
  have hb := finite_entropy_bounds (spectralWeights s) (by omega) hprob.1 hprob.2
  have hl : 0 < Real.log n := Real.log_pos (by exact_mod_cast hn)
  constructor
  · exact div_nonneg hb.1 hl.le
  · exact (div_le_one hl).mpr hb.2

/-- The meaningful entropy domain is nonempty, arXiv:2502.16982, §3.4. -/
example : 1 < (2 : ℕ) ∧ 0 < ∑ j : Fin 2, (fun _ : Fin 2 => (1 : ℝ)) j ^ 2 := by norm_num

/-- Uniform nonzero singular values give maximal normalized entropy one,
arXiv:2502.16982, §3.4, the interpretation of a flat spectrum. -/
theorem spectralEntropy_flat (t : ℝ) (hn : 1 < n) (ht : t ≠ 0) :
    spectralEntropy (fun _ : Fin n => t) = 1 := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast (by omega : n ≠ 0)
  have hl : Real.log n ≠ 0 := (Real.log_pos (by exact_mod_cast hn)).ne'
  have hw : spectralWeights (fun _ : Fin n => t) = fun _ => (n : ℝ)⁻¹ := by
    funext i
    simp only [spectralWeights, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
      nsmul_eq_mul]
    field_simp
  rw [spectralEntropy, hw]
  simp only [Real.negMulLog, Real.log_inv, neg_mul_neg, Finset.sum_const,
    Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  field_simp

/-- Nonzero flat spectra exist, arXiv:2502.16982, §3.4. -/
example : 1 < (2 : ℕ) ∧ (1 : ℝ) ≠ 0 := by norm_num

/-- Spectral probabilities, hence entropy, do not depend on a nonzero global
singular-value scale, arXiv:2502.16982, §3.4. -/
theorem spectralWeights_smul (s : Fin n → ℝ) (c : ℝ) (hc : c ≠ 0) :
    spectralWeights (fun i => c * s i) = spectralWeights s := by
  funext i
  simp only [spectralWeights, mul_pow, ← Finset.mul_sum]
  exact mul_div_mul_left _ _ (pow_ne_zero 2 hc)

/-- Nonzero scale factors exist, arXiv:2502.16982, §3.4. -/
example : (2 : ℝ) ≠ 0 := by norm_num

/-- SVD entropy is unchanged by a nonzero global scale,
arXiv:2502.16982, §3.4. -/
theorem spectralEntropy_smul (s : Fin n → ℝ) (c : ℝ) (hc : c ≠ 0) :
    spectralEntropy (fun i => c * s i) = spectralEntropy s := by
  rw [spectralEntropy, spectralWeights_smul s c hc, spectralEntropy]

/-- Nonzero scale factors exist, arXiv:2502.16982, §3.4. -/
example : (2 : ℝ) ≠ 0 := by norm_num

end Transformer.Muon
