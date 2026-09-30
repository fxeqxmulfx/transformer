/-
# DASH — convergence of the preconditioner inverse-root solvers

arXiv:2602.02016v2, §3.2–3.3. These are the symmetric positive-definite
matrices arising from regularized Shampoo preconditioners. Eigenvalue
conditions express the source's convergence conditions in its EVD frame.
-/

import Transformer.DASH.Section3_NewtonSpectra
import Transformer.DASH.Section3_CNConvergence

open scoped BigOperators Matrix

noncomputable section

namespace Transformer.DASH

variable {n : ℕ}

/-- The CN matrices converge to the inverse root and identity, with the
source's endpoint error corrected. Positive eigenvalues, positive scaling,
and a positive integer order are required.
Source: arXiv:2602.02016v2, §3.2, `equation:CN-init`–`equation:CN-X-M`. -/
theorem cn_matrix_convergence (p : ℕ) (Q : Matrix (Fin n) (Fin n) ℝ)
    (s : Fin n → ℝ) (c : ℝ) (hQ : Orthogonal Q) (hp : 0 < p) (hc : 0 < c)
    (hs : ∀ i, 0 < s i) (hupper : ∀ i, s i < ((p : ℝ) + 1) * c ^ p) :
    Filter.Tendsto (cnIterate p (spectralMatrix Q s) c) Filter.atTop
      (nhds (spectralPower Q s (-(1 / (p : ℝ))), 1)) := by
  have hx := spectralMatrix_tendsto Q (fun k i => cnScalarX p (s i) c k)
    (fun i => s i ^ (-(1 / (p : ℝ))))
    (fun i => cnScalarX_tendsto p (s i) c hp (hs i) hc (hupper i))
  have hm := spectralMatrix_tendsto Q (fun k i => cnScalarM p (s i) c k)
    (fun _ => 1) (fun i => cnScalarM_tendsto p (s i) c hp (hs i) hc (hupper i))
  rw [show cnIterate p (spectralMatrix Q s) c = _ from
    funext (cnIterate_spectrum p Q s c hQ)]
  simpa only [spectralPower, spectralMatrix_one Q hQ] using
    hx.prodMk_nhds hm

/-- All matrix CN assumptions are satisfiable,
arXiv:2602.02016v2, §3.2: an identity frame and unit spectrum. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧ 0 < (2 : ℕ) ∧ (0 : ℝ) < 1 ∧
    (∀ i : Fin 1, (0 : ℝ) < (fun _ : Fin 1 => (1 : ℝ)) i) ∧
    (∀ i : Fin 1, (fun _ : Fin 1 => (1 : ℝ)) i < ((2 : ℝ) + 1) * (1 : ℝ) ^ (2 : ℕ)) := by
  norm_num [Orthogonal, Muon.OrthonormalColumns]

/-- Scalar NDB converges to the positive square and inverse square roots.
The hypothesis `0<a<2` is exactly the scalar version of `‖I-A‖₂<1`.
Source: arXiv:2602.02016v2, §3.3, `equation:NDB-Y-Z`. -/
theorem ndb_scalar_convergence (a : ℝ) (ha : 0 < a) (ha' : a < 2) :
    Filter.Tendsto (ndbScalar a) Filter.atTop
      (nhds (a ^ (1 / 2 : ℝ), a ^ (-(1 / 2 : ℝ)))) := by
  have hupper : a < ((2 : ℝ) + 1) * (1 : ℝ) ^ (2 : ℕ) := by norm_num; linarith
  have hz := cnScalarX_tendsto 2 a 1 (by norm_num) ha (by norm_num) hupper
  norm_num only [Nat.cast_ofNat] at hz
  have hy := (tendsto_const_nhds (x := a)).mul hz
  have heq : a * a ^ (-(1 / 2 : ℝ)) = a ^ (1 / 2 : ℝ) := by
    have h := Real.rpow_add ha 1 (-(1 / 2 : ℝ))
    norm_num at h
    exact h.symm
  rw [heq] at hy
  rw [show ndbScalar a = _ from funext (ndbScalar_eq_cn a)]
  exact hy.prodMk_nhds hz

/-- NDB's scalar convergence interval is nonempty,
arXiv:2602.02016v2, §3.3. -/
example : (0 : ℝ) < 1 ∧ (1 : ℝ) < 2 := by norm_num

/-- NDB converges to the square and inverse square roots of a symmetric
preconditioner. In an orthogonal eigenbasis `0<s_i<2` is the source's
condition `‖I-A‖₂<1`; the theorem makes this spectral domain explicit.
Source: arXiv:2602.02016v2, §3.3, `equation:NDB-init`–`equation:NDB-Y-Z`. -/
theorem ndb_matrix_convergence (Q : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → ℝ)
    (hQ : Orthogonal Q) (hs : ∀ i, 0 < s i) (hs' : ∀ i, s i < 2) :
    Filter.Tendsto (ndbIterate (spectralMatrix Q s)) Filter.atTop
      (nhds (spectralPower Q s (1 / 2), spectralPower Q s (-(1 / 2)))) := by
  have hscalar := fun i => ndb_scalar_convergence (s i) (hs i) (hs' i)
  have hy := spectralMatrix_tendsto Q (fun k i => (ndbScalar (s i) k).1)
    (fun i => s i ^ (1 / 2 : ℝ)) (fun i => (hscalar i).fst_nhds)
  have hz := spectralMatrix_tendsto Q (fun k i => (ndbScalar (s i) k).2)
    (fun i => s i ^ (-(1 / 2 : ℝ))) (fun i => (hscalar i).snd_nhds)
  rw [show ndbIterate (spectralMatrix Q s) = _ from
    funext (ndbIterate_spectrum Q s hQ)]
  simpa only [spectralPower] using hy.prodMk_nhds hz

/-- Matrix NDB assumptions are satisfiable, arXiv:2602.02016v2, §3.3. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    (∀ i : Fin 1, (0 : ℝ) < (fun _ : Fin 1 => (1 : ℝ)) i) ∧
    (∀ i : Fin 1, (fun _ : Fin 1 => (1 : ℝ)) i < 2) := by
  norm_num [Orthogonal, Muon.OrthonormalColumns]

end Transformer.DASH
