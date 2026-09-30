/-
# DASH — two finite NDB calls for the inverse fourth root

arXiv:2602.02016v2, §3.3. Both calls use a growing finite iteration
count. Uniform convergence of the second inverse-root solver proves
convergence of their actual composition, including the changing
numerical square-root output of the first call.
-/

import Transformer.DASH.Section3_CNUniformConvergence
import Transformer.DASH.Section3_ChainedRoots

open scoped Matrix Topology
open Filter

noncomputable section

namespace Transformer.DASH

variable {n : ℕ}

/-- A finite NDB solve at a growing iteration count converges correctly
when its changing input converges inside `(0,2)`. Both the square-root
and inverse-square-root outputs include the input's numerical error.
Source: arXiv:2602.02016v2, §3.3, chaining numerical root calls. -/
theorem ndb_scalar_moving_input (a : ℕ → ℝ) (μ : ℝ) (hμ : 0 < μ) (hμ' : μ < 2)
    (ha : Tendsto a atTop (𝓝 μ)) :
    Tendsto (fun k : ℕ => ndbScalar (a k) (k + 1)) atTop
      (𝓝 (μ ^ (1 / 2 : ℝ), μ ^ (-(1 / 2 : ℝ)))) := by
  have hupper : μ < ((2 : ℝ) + 1) * (1 : ℝ) ^ (2 : ℕ) := by norm_num; linarith
  have hz := cnScalarX_moving_input 2 1 μ a (by norm_num) (by norm_num) hμ hupper ha
  norm_num only [Nat.cast_ofNat] at hz
  have hy := ha.mul hz
  have hpower : μ * μ ^ (-(1 / 2 : ℝ)) = μ ^ (1 / 2 : ℝ) := by
    have h := Real.rpow_add hμ 1 (-(1 / 2 : ℝ))
    norm_num at h
    exact h.symm
  rw [hpower] at hy
  have hsequence : (fun k : ℕ => ndbScalar (a k) (k + 1)) = fun k : ℕ =>
      (a k * cnScalarX 2 (a k) 1 (k + 1), cnScalarX 2 (a k) 1 (k + 1)) := by
    funext k
    exact ndbScalar_eq_cn _ _
  rw [hsequence]
  exact hy.prodMk_nhds hz

/-- Moving NDB inputs have satisfiable positive limit assumptions,
arXiv:2602.02016v2, §3.3. -/
example : (0 : ℝ) < 1 ∧ (1 : ℝ) < 2 ∧
    Tendsto (fun _ : ℕ => (1 : ℝ)) atTop (𝓝 1) :=
  ⟨by norm_num, by norm_num, tendsto_const_nhds⟩

/-- Two finite scalar NDB calls converge to the inverse fourth root when
both iteration counts increase together. The second input is the first
call's computed square-root iterate, not an exact square root.
Source: arXiv:2602.02016v2, §3.3, the two-call inverse-fourth-root procedure. -/
theorem ndb_scalar_finite_chain_convergence (a : ℝ) (ha : 0 < a) (ha' : a < 2) :
    Tendsto (fun k : ℕ => (ndbScalar (ndbScalar a (k + 1)).1 (k + 1)).2)
      atTop (𝓝 (a ^ (-(1 / 4 : ℝ)))) := by
  have hfirst : Tendsto (fun k : ℕ => (ndbScalar a (k + 1)).1)
      atTop (𝓝 (a ^ (1 / 2 : ℝ))) :=
    (tendsto_add_atTop_iff_nat 1).2 (ndb_scalar_convergence a ha ha').fst_nhds
  have hroot : 0 < a ^ (1 / 2 : ℝ) := Real.rpow_pos_of_pos ha _
  have hupper : a ^ (1 / 2 : ℝ) < ((2 : ℝ) + 1) * (1 : ℝ) ^ (2 : ℕ) := by
    norm_num
    rw [← Real.sqrt_eq_rpow]
    apply (Real.sqrt_lt ha.le (by norm_num : (0 : ℝ) ≤ 3)).2
    linarith
  have h := cnScalarX_moving_input 2 1 (a ^ (1 / 2 : ℝ))
    (fun k : ℕ => (ndbScalar a (k + 1)).1) (by norm_num) (by norm_num)
    hroot hupper hfirst
  norm_num only [Nat.cast_ofNat] at h
  rw [← Real.rpow_mul ha.le] at h
  norm_num only [show (1 / 2 : ℝ) * -(1 / 2) = -(1 / 4) by norm_num] at h
  have hsequence : (fun k : ℕ => (ndbScalar (ndbScalar a (k + 1)).1 (k + 1)).2) =
      fun k : ℕ => cnScalarX 2 (ndbScalar a (k + 1)).1 1 (k + 1) := by
    funext k
    rw [ndbScalar_eq_cn]
  rwa [hsequence]

/-- The finite scalar-chain domain is nonempty,
arXiv:2602.02016v2, §3.3. -/
example : (0 : ℝ) < 1 ∧ (1 : ℝ) < 2 := by norm_num

/-- The two finite matrix calls act on each eigenvalue by the two finite
scalar calls, preserving the original orthogonal eigenframe.
Source: arXiv:2602.02016v2, §3.3, NDB inverse-fourth-root chaining. -/
theorem ndb_finite_chain_spectrum (Q : Matrix (Fin n) (Fin n) ℝ)
    (s : Fin n → ℝ) (hQ : Orthogonal Q) (k m : ℕ) :
    (ndbIterate (ndbIterate (spectralMatrix Q s) k).1 m).2 =
      spectralMatrix Q (fun i => (ndbScalar (ndbScalar (s i) k).1 m).2) := by
  rw [ndbIterate_spectrum Q s hQ, ndbIterate_spectrum Q _ hQ]

/-- Finite chaining admits an orthogonal frame,
arXiv:2602.02016v2, §3.3. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [Orthogonal, Muon.OrthonormalColumns]

/-- The actual composition of two finite matrix NDB solves converges to
the inverse fourth root as their common iteration count tends to infinity.
This completes the chaining claim with the first call's numerical error
included. Source: arXiv:2602.02016v2, §3.3, the two-call procedure. -/
theorem ndb_matrix_finite_chain_convergence (Q : Matrix (Fin n) (Fin n) ℝ)
    (s : Fin n → ℝ) (hQ : Orthogonal Q) (hs : ∀ i, 0 < s i) (hupper : ∀ i, s i < 2) :
    Tendsto (fun k : ℕ =>
      (ndbIterate (ndbIterate (spectralMatrix Q s) (k + 1)).1 (k + 1)).2)
      atTop (𝓝 (spectralPower Q s (-(1 / 4)))) := by
  have h := spectralMatrix_tendsto Q
    (fun k i => (ndbScalar (ndbScalar (s i) (k + 1)).1 (k + 1)).2)
    (fun i => s i ^ (-(1 / 4 : ℝ)))
    (fun i => ndb_scalar_finite_chain_convergence (s i) (hs i) (hupper i))
  simpa only [← ndb_finite_chain_spectrum Q s hQ, spectralPower] using h

/-- Matrix-chain assumptions have a positive-definite example,
arXiv:2602.02016v2, §3.3. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    (∀ i : Fin 1, (0 : ℝ) < (fun _ => (1 : ℝ)) i) ∧
    (∀ i : Fin 1, (fun _ => (1 : ℝ)) i < 2) := by
  norm_num [Orthogonal, Muon.OrthonormalColumns]

end Transformer.DASH
