/-
# DASH — convergence to the maximal eigenspace

arXiv:2602.02016v2, §3.4–3.5. The dominant eigenvalue may have
multiplicity greater than one. A nonzero projection onto its eigenspace
is required, rather than a unique largest eigenvector.
-/

import Transformer.DASH.Section3_PowerCoordinates
import Mathlib.Analysis.SpecificLimits.Basic

open scoped BigOperators Matrix

noncomputable section

namespace Transformer.DASH

variable {n : ℕ}

/-- Invariant coordinate quotient under nonzero vector scaling,
arXiv:2602.02016v2, §3.4, Rayleigh estimation after normalization. -/
theorem spectralRayleigh_smul (s z : Fin n → ℝ) (c : ℝ) (hc : c ≠ 0) :
    spectralRayleigh s (c • z) = spectralRayleigh s z := by
  have hQ : Orthogonal (1 : Matrix (Fin n) (Fin n) ℝ) := by
    simp [Orthogonal, Muon.OrthonormalColumns]
  calc
    spectralRayleigh s (c • z) =
        rayleigh (spectralMatrix 1 s) ((1 : Matrix (Fin n) (Fin n) ℝ) *ᵥ (c • z)) :=
      (rayleigh_spectral 1 s (c • z) hQ).symm
    _ = rayleigh (spectralMatrix 1 s) ((1 : Matrix (Fin n) (Fin n) ℝ) *ᵥ z) := by
      simp only [Matrix.one_mulVec]
      exact rayleigh_smul _ _ c hc
    _ = spectralRayleigh s z := rayleigh_spectral 1 s z hQ

/-- Scaling assumptions are satisfiable, arXiv:2602.02016v2, §3.4. -/
example : (2 : ℝ) ≠ 0 := by norm_num

/-- Rescaling all power coordinates by the dominant eigenvalue's power
leaves their Rayleigh quotient unchanged.
Source: arXiv:2602.02016v2, §3.4–3.5, the eigenvalue ratios governing PI. -/
theorem powerCoordinates_rescale (s z : Fin n → ℝ) (μ : ℝ) (hμ : μ ≠ 0) (k : ℕ) :
    spectralRayleigh s (powerCoordinates s z k) =
      spectralRayleigh s (powerCoordinates (fun i => s i / μ) z k) := by
  have heq : powerCoordinates s z k = μ ^ k • powerCoordinates (fun i => s i / μ) z k := by
    funext i
    simp only [powerCoordinates, Pi.smul_apply, smul_eq_mul, div_pow]
    field_simp
  rw [heq, spectralRayleigh_smul _ _ _ (pow_ne_zero _ hμ)]

/-- Dominant-eigenvalue scaling assumptions are satisfiable,
arXiv:2602.02016v2, §3.4. -/
example : (1 : ℝ) ≠ 0 := by norm_num

/-- Each rescaled power coordinate tends to its projection onto the maximal
eigenspace; repeated maximal eigenvalues are retained.
Source: arXiv:2602.02016v2, §3.4–3.5, Power Iteration convergence. -/
theorem powerCoordinates_tendsto (s z : Fin n → ℝ) (μ : ℝ)
    (hμ : 0 < μ) (hs : ∀ i, 0 ≤ s i) (hupper : ∀ i, s i ≤ μ) (i : Fin n) :
    Filter.Tendsto (fun k => powerCoordinates (fun j => s j / μ) z k i)
      Filter.atTop (nhds (if s i = μ then z i else 0)) := by
  by_cases hi : s i = μ
  · simp [powerCoordinates, hi, hμ.ne']
  · have hratio : s i / μ < 1 := (div_lt_one hμ).mpr (lt_of_le_of_ne (hupper i) hi)
    have hpow := tendsto_pow_atTop_nhds_zero_of_lt_one (div_nonneg (hs i) hμ.le) hratio
    simpa only [powerCoordinates, hi, ite_false, zero_mul] using
      hpow.mul (tendsto_const_nhds (x := z i))

/-- Projection-convergence assumptions are satisfiable,
arXiv:2602.02016v2, §3.4–3.5. -/
example : (0 : ℝ) < 1 ∧ (∀ i : Fin 1, (0 : ℝ) ≤ (fun _ => (1 : ℝ)) i) ∧
    (∀ i : Fin 1, (fun _ => (1 : ℝ)) i ≤ 1) := by norm_num

/-- The normalized implementation's Rayleigh estimate converges to the
largest eigenvalue when the starting vector has a nonzero maximal-eigenspace
component. This is the corrected form of the source's PI convergence claim;
without the component hypothesis the proved counterexample remains stuck.
Source: arXiv:2602.02016v2, §3.4–3.5. -/
theorem powerIteration_rayleigh_tendsto (Q : Matrix (Fin n) (Fin n) ℝ)
    (s z : Fin n → ℝ) (j : Fin n) (hQ : Orthogonal Q) (hsj : 0 < s j)
    (hs : ∀ i, 0 ≤ s i) (hupper : ∀ i, s i ≤ s j) (hzj : z j ≠ 0) :
    Filter.Tendsto
      (fun k => rayleigh (spectralMatrix Q s) (powerIterate (spectralMatrix Q s) (Q *ᵥ z) k))
      Filter.atTop (nhds (s j)) := by
  let w : Fin n → ℝ := fun i => if s i = s j then z i else 0
  have hw : w ≠ 0 := by
    intro h
    apply hzj
    simpa [w] using congrFun h j
  have hlim := fun i => powerCoordinates_tendsto s z (s j) hsj hs hupper i
  have hden := tendsto_finsetSum Finset.univ (fun i _ => (hlim i).pow 2)
  have hnum := tendsto_finsetSum Finset.univ
    (fun i _ => (tendsto_const_nhds (x := s i)).mul ((hlim i).pow 2))
  have hquot := hnum.div hden (coordinateEnergy_pos w hw).ne'
  have hvalue : (∑ i, s i * w i ^ 2) / (∑ i, w i ^ 2) = s j := by
    have hsum : (∑ i, s i * w i ^ 2) = s j * (∑ i, w i ^ 2) := by
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro i hi
      by_cases h : s i = s j <;> simp [w, h]
    rw [hsum]
    exact mul_div_cancel_right₀ _ (coordinateEnergy_pos w hw).ne'
  change Filter.Tendsto
    (fun k => spectralRayleigh s (powerCoordinates (fun i => s i / s j) z k))
    Filter.atTop (nhds ((∑ i, s i * w i ^ 2) / (∑ i, w i ^ 2))) at hquot
  rw [hvalue] at hquot
  convert hquot using 1
  funext k
  rw [powerIterate_rayleigh Q s z j hQ hsj hzj,
    powerCoordinates_rescale s z (s j) hsj.ne']

/-- All PI convergence hypotheses are satisfiable,
arXiv:2602.02016v2, §3.4–3.5. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    (0 : ℝ) < (fun _ : Fin 1 => (1 : ℝ)) 0 ∧
    (∀ i : Fin 1, (0 : ℝ) ≤ (fun _ => (1 : ℝ)) i) ∧
    (∀ i : Fin 1, (fun _ => (1 : ℝ)) i ≤ (fun _ : Fin 1 => (1 : ℝ)) 0) ∧
    (fun _ : Fin 1 => (1 : ℝ)) 0 ≠ 0 := by
  norm_num [Orthogonal, Muon.OrthonormalColumns]

end Transformer.DASH
