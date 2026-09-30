/-
# DASH — convergence and eventual safety of a pool of PI starts

arXiv:2602.02016v2, §3.4–3.5. One successful starting vector suffices
for convergence of the pool maximum. A fixed finite iteration count
still has no unconditional factor-two safety guarantee.
-/

import Transformer.DASH.Section3_PowerConvergence

open scoped BigOperators Matrix

noncomputable section

namespace Transformer.DASH

variable {n : ℕ}

/-- An upper spectral bound controls every vector's Rayleigh quotient,
including zero under the totalized quotient convention. The source defines
the quotient only for nonzero vectors; allowing zero needs `μ≥0`.
Source: arXiv:2602.02016v2, §3.4, the Rayleigh upper bound. -/
theorem rayleigh_le_spectral_bound (Q : Matrix (Fin n) (Fin n) ℝ)
    (s : Fin n → ℝ) (μ : ℝ) (hQ : Orthogonal Q) (hμ : 0 ≤ μ)
    (hupper : ∀ i, s i ≤ μ) (x : Fin n → ℝ) :
    rayleigh (spectralMatrix Q s) x ≤ μ := by
  have hx : Q *ᵥ (Q.transpose *ᵥ x) = x := by
    rw [Matrix.mulVec_mulVec, hQ.2, Matrix.one_mulVec]
  calc
    rayleigh (spectralMatrix Q s) x =
        rayleigh (spectralMatrix Q s) (Q *ᵥ (Q.transpose *ᵥ x)) := congrArg _ hx.symm
    _ = spectralRayleigh s (Q.transpose *ᵥ x) := rayleigh_spectral Q s _ hQ
    _ ≤ μ := by
      by_cases hz : Q.transpose *ᵥ x = 0
      · simpa [hz, spectralRayleigh, coordinateEnergy] using hμ
      · exact spectralRayleigh_le s _ μ hz hupper

/-- Universal-bound assumptions are satisfiable,
arXiv:2602.02016v2, §3.4. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧ (0 : ℝ) ≤ 1 ∧
    (∀ i : Fin 1, (fun _ => (1 : ℝ)) i ≤ 1) := by
  norm_num [Orthogonal, Muon.OrthonormalColumns]

/-- If one pool member has a nonzero maximal-eigenspace component, the maximum
Rayleigh estimate converges to the largest eigenvalue. Other starts may be
arbitrary, including zero or vectors orthogonal to that eigenspace. The
source's probabilistic motivation is separated from this deterministic result.
Source: arXiv:2602.02016v2, §3.5, multi-Power-Iteration. -/
theorem pooledRayleigh_tendsto {ι : Type} [Fintype ι] [Nonempty ι]
    (Q : Matrix (Fin n) (Fin n) ℝ) (s z : Fin n → ℝ)
    (starts : ι → Fin n → ℝ) (good : ι) (j : Fin n)
    (hQ : Orthogonal Q) (hsj : 0 < s j) (hs : ∀ i, 0 ≤ s i)
    (hupper : ∀ i, s i ≤ s j) (hzj : z j ≠ 0) (hgood : starts good = Q *ᵥ z) :
    Filter.Tendsto (pooledRayleigh (spectralMatrix Q s) starts) Filter.atTop (nhds (s j)) := by
  have hlow := powerIteration_rayleigh_tendsto Q s z j hQ hsj hs hupper hzj
  rw [← hgood] at hlow
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le hlow tendsto_const_nhds
  · intro k
    exact pooledRayleigh_ge_member _ starts k good
  · intro k
    exact pooledRayleigh_le _ starts k (s j)
      (fun t => rayleigh_le_spectral_bound Q s (s j) hQ hsj.le hupper _)

/-- A one-member successful pool witnesses every convergence assumption.
Source: arXiv:2602.02016v2, §3.5. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    (0 : ℝ) < (fun _ : Fin 1 => (1 : ℝ)) 0 ∧
    (∀ i : Fin 1, (0 : ℝ) ≤ (fun _ => (1 : ℝ)) i) ∧
    (∀ i : Fin 1, (fun _ => (1 : ℝ)) i ≤ (fun _ : Fin 1 => (1 : ℝ)) 0) ∧
    (fun _ : Fin 1 => (1 : ℝ)) 0 ≠ 0 ∧
    (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ)) 0 =
      (1 : Matrix (Fin 1) (Fin 1) ℝ) *ᵥ (fun _ => (1 : ℝ)) := by
  norm_num [Orthogonal, Muon.OrthonormalColumns]

/-- The factor-two rule is eventually safe under the corrected starting-vector
assumptions. The source claims safety of its finite estimate without these
conditions; this theorem neither supplies a universal stopping index nor
rescinds the previously proved finite-estimate counterexample.
Source: arXiv:2602.02016v2, §3.4–3.5, factor-two normalization. -/
theorem pooled_scaling_eventually_safe {ι : Type} [Fintype ι] [Nonempty ι]
    (Q : Matrix (Fin n) (Fin n) ℝ) (s z : Fin n → ℝ)
    (starts : ι → Fin n → ℝ) (good : ι) (j : Fin n)
    (hQ : Orthogonal Q) (hsj : 0 < s j) (hs : ∀ i, 0 ≤ s i)
    (hupper : ∀ i, s i ≤ s j) (hzj : z j ≠ 0) (hgood : starts good = Q *ᵥ z) :
    ∃ K : ℕ, ∀ k ≥ K,
      0 < pooledRayleigh (spectralMatrix Q s) starts k ∧
      s j / (2 * pooledRayleigh (spectralMatrix Q s) starts k) < 1 := by
  have hlim := pooledRayleigh_tendsto Q s z starts good j hQ hsj hs hupper hzj hgood
  have hev := hlim.eventually (lt_mem_nhds (show s j / 2 < s j by linarith))
  apply Filter.eventually_atTop.mp
  filter_upwards [hev] with k hk
  have hr : 0 < pooledRayleigh (spectralMatrix Q s) starts k := by linarith
  exact ⟨hr, factor_two_scaling_safe _ _ hr (by linarith)⟩

/-- Eventual-safety assumptions are satisfiable, arXiv:2602.02016v2, §3.4–3.5. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    (0 : ℝ) < (fun _ : Fin 1 => (1 : ℝ)) 0 ∧
    (∀ i : Fin 1, (0 : ℝ) ≤ (fun _ => (1 : ℝ)) i) ∧
    (∀ i : Fin 1, (fun _ => (1 : ℝ)) i ≤ (fun _ : Fin 1 => (1 : ℝ)) 0) ∧
    (fun _ : Fin 1 => (1 : ℝ)) 0 ≠ 0 ∧
    (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ)) 0 =
      (1 : Matrix (Fin 1) (Fin 1) ℝ) *ᵥ (fun _ => (1 : ℝ)) := by
  norm_num [Orthogonal, Muon.OrthonormalColumns]

end Transformer.DASH
