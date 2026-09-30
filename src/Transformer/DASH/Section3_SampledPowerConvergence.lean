/-
# DASH — connect the sampling event to the actual PI implementation

arXiv:2602.02016v2, §3.5. Bad candidates have zero projection onto the
entire maximal eigenspace. A pool containing any other candidate converges;
the finite uniform model counts pools missing every successful candidate.
-/

import Transformer.DASH.Section3_PoolProbability

open scoped Matrix

noncomputable section

namespace Transformer.DASH

variable {m n : ℕ}

/-- Exactly the candidates orthogonal to the maximal eigenspace, including
all of its coordinates when the largest eigenvalue is repeated.
Source: arXiv:2602.02016v2, §3.5, convergence to smaller eigenvalues. -/
def badPowerCandidates (s : Fin n → ℝ) (Z : Fin m → Fin n → ℝ) (μ : ℝ) :
    Finset (Fin m) := by
  classical
  exact Finset.univ.filter (fun candidate => ∀ i, s i = μ → Z candidate i = 0)

/-- A good candidate has a nonzero component in the maximal eigenspace.
Source: arXiv:2602.02016v2, §3.5, the missing starting-vector condition. -/
theorem not_mem_badPowerCandidates (s : Fin n → ℝ) (Z : Fin m → Fin n → ℝ)
    (μ : ℝ) (candidate : Fin m) :
    candidate ∉ badPowerCandidates s Z μ ↔ ∃ i, s i = μ ∧ Z candidate i ≠ 0 := by
  classical
  simp only [badPowerCandidates, Finset.mem_filter, Finset.mem_univ, true_and,
    not_forall, exists_prop]

/-- Every sampled pool containing a good candidate converges under the actual
normalized PI recurrence. This connects the counted bad-pool event to the
matrix estimator rather than merely naming an arbitrary subset “bad”.
Source: arXiv:2602.02016v2, §3.5, multi-Power-Iteration. -/
theorem sampled_pool_converges {N : ℕ} [Nonempty (Fin N)]
    (Q : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → ℝ) (Z : Fin m → Fin n → ℝ)
    (μ : ℝ) (draw : Fin N → Fin m) (hQ : Orthogonal Q) (hμ : 0 < μ)
    (hs : ∀ i, 0 ≤ s i) (hupper : ∀ i, s i ≤ μ)
    (hgood : ∃ position, draw position ∉ badPowerCandidates s Z μ) :
    Filter.Tendsto
      (pooledRayleigh (spectralMatrix Q s) (fun position => Q *ᵥ Z (draw position)))
      Filter.atTop (nhds μ) := by
  obtain ⟨position, hposition⟩ := hgood
  obtain ⟨i, hi, hzi⟩ := (not_mem_badPowerCandidates s Z μ _).1 hposition
  have hlim := pooledRayleigh_tendsto Q s (Z (draw position))
    (fun position => Q *ᵥ Z (draw position)) position i hQ
    (by simpa only [hi] using hμ) hs (fun j => by simpa only [hi] using hupper j) hzi rfl
  simpa only [hi] using hlim

/-- One good unit start satisfies the sampled-convergence assumptions.
Source: arXiv:2602.02016v2, §3.5. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧ (0 : ℝ) < 1 ∧
    (∀ i : Fin 1, (0 : ℝ) ≤ (fun _ => (1 : ℝ)) i) ∧
    (∀ i : Fin 1, (fun _ => (1 : ℝ)) i ≤ 1) ∧
    (∃ position : Fin 1, (fun _ : Fin 1 => (0 : Fin 1)) position ∉
      badPowerCandidates (fun _ : Fin 1 => 1) (fun _ : Fin 1 => fun _ : Fin 1 => 1) 1) := by
  refine ⟨?_, by norm_num, by norm_num, by norm_num, 0, ?_⟩
  · simp [Orthogonal, Muon.OrthonormalColumns]
  · exact (not_mem_badPowerCandidates _ _ _ _).2 ⟨0, rfl, by norm_num⟩

/-- Independent uniform draws miss the maximal eigenspace in every pool
position with the exact fraction `p^N`. Every pool outside this event has
the convergence established above. The statement concerns limiting PI, not
the accuracy or safety of a fixed finite iteration count.
Source: arXiv:2602.02016v2, §3.5, the probabilistic motivation of multiple starts. -/
theorem sampled_pool_miss_fraction (s : Fin n → ℝ) (Z : Fin m → Fin n → ℝ)
    (μ : ℝ) (N : ℕ) :
    poolMissFraction (badPowerCandidates s Z μ) N =
      (((badPowerCandidates s Z μ).card : ℝ) / m) ^ N :=
  poolMissFraction_eq_pow _ N

end Transformer.DASH
