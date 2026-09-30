/-
# DASH — Newton solvers with automatic safe preprocessing

arXiv:2602.02016v2, §3.2–3.4. These algorithms include the scale
selection and output rescaling. The CN comparison scale is retained,
but the normalized spectrum lies strictly below its failed endpoint.
-/

import Transformer.DASH.Section3_GuardedScaling
import Transformer.DASH.Section3_CNUnitScaling
import Transformer.DASH.Section3_FiniteChaining

open scoped Matrix Topology
open Filter

noncomputable section

namespace Transformer.DASH

variable {n : ℕ}

/-- Corrected CN: guard the input scale, iterate in the source's unit
comparison regime, and undo normalization on the inverse root.
Source: arXiv:2602.02016v2, §3.2–3.4, CN and unit spectral scaling. -/
def guardedCnIterate (p : ℕ) (A : Matrix (Fin n) (Fin n) ℝ) (r : ℝ) (k : ℕ) :
    Matrix (Fin n) (Fin n) ℝ :=
  guardedScale A r ^ (-(1 / (p : ℝ))) •
    (cnIterate p (guardedNormalize A r) (cnUnitScale p) k).1

/-- Corrected NDB returns roots of the original input, with reciprocal
rescaling factors for its square and inverse square roots.
Source: arXiv:2602.02016v2, §3.3–3.4, NDB and matrix normalization. -/
def guardedNdbIterate (A : Matrix (Fin n) (Fin n) ℝ) (r : ℝ) (k : ℕ) :
    Matrix (Fin n) (Fin n) ℝ × Matrix (Fin n) (Fin n) ℝ :=
  let state := ndbIterate (guardedNormalize A r) k
  (guardedScale A r ^ (1 / 2 : ℝ) • state.1,
    guardedScale A r ^ (-(1 / 2 : ℝ)) • state.2)

/-- Corrected inverse-fourth-root solver uses two finite NDB calls and
rescales once at the end. The first call's numerical error is retained.
Source: arXiv:2602.02016v2, §3.3–3.4, inverse-root chaining. -/
def guardedInverseFourth (A : Matrix (Fin n) (Fin n) ℝ) (r : ℝ) (k : ℕ) :
    Matrix (Fin n) (Fin n) ℝ :=
  guardedScale A r ^ (-(1 / 4 : ℝ)) •
    (ndbIterate (ndbIterate (guardedNormalize A r) k).1 k).2

/-- The actual guarded CN solver converges for every candidate estimate.
The only mathematical input assumptions are a positive spectrum and order;
the scale positivity and strict CN domain are established by the algorithm.
Source: arXiv:2602.02016v2, §3.2–3.4, corrected unit-scaled CN procedure. -/
theorem guardedCnIterate_convergence (p : ℕ) (Q : Matrix (Fin n) (Fin n) ℝ)
    (s : Fin n → ℝ) (r : ℝ) (hQ : Orthogonal Q) (hs : ∀ i, 0 < s i) (hp : 0 < p) :
    Tendsto (guardedCnIterate p (spectralMatrix Q s) r) atTop
      (𝓝 (spectralPower Q s (-(1 / (p : ℝ))))) := by
  have hd := guarded_spectrum_bounds Q s r hQ hs
  have h := (cn_matrix_convergence p Q
    (fun i => s i / guardedScale (spectralMatrix Q s) r) (cnUnitScale p)
    hQ hp (cnUnitScale_pos p) (fun i => (hd i).1) (by
      intro i
      rw [cnUnitScale_power p hp]
      exact (hd i).2)).fst_nhds.const_smul
        (guardedScale (spectralMatrix Q s) r ^ (-(1 / (p : ℝ))))
  rw [spectralPower_rescale Q s _ _ (guardedScale_pos _ r) (fun i => (hs i).le)] at h
  change Tendsto (fun k => guardedCnIterate p (spectralMatrix Q s) r k) atTop _
  simpa only [guardedCnIterate, guardedNormalize_spectral] using h

/-- The guarded CN hypotheses are satisfiable, arXiv:2602.02016v2, §3.2–3.4. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    (∀ i : Fin 1, (0 : ℝ) < (fun _ => (1 : ℝ)) i) ∧ 0 < (4 : ℕ) := by
  norm_num [Orthogonal, Muon.OrthonormalColumns]

/-- Both outputs of the actual guarded NDB solver converge to roots of
the original matrix for every candidate estimate, including zero.
Source: arXiv:2602.02016v2, §3.3–3.4, corrected complete NDB procedure. -/
theorem guardedNdbIterate_convergence (Q : Matrix (Fin n) (Fin n) ℝ)
    (s : Fin n → ℝ) (r : ℝ) (hQ : Orthogonal Q) (hs : ∀ i, 0 < s i) :
    Tendsto (guardedNdbIterate (spectralMatrix Q s) r) atTop
      (𝓝 (spectralPower Q s (1 / 2), spectralPower Q s (-(1 / 2)))) := by
  have hd := guarded_spectrum_bounds Q s r hQ hs
  have h := ndb_matrix_convergence Q
    (fun i => s i / guardedScale (spectralMatrix Q s) r) hQ
    (fun i => (hd i).1) (fun i => lt_trans (hd i).2 (by norm_num))
  have hy := h.fst_nhds.const_smul (guardedScale (spectralMatrix Q s) r ^ (1 / 2 : ℝ))
  have hz := h.snd_nhds.const_smul (guardedScale (spectralMatrix Q s) r ^ (-(1 / 2 : ℝ)))
  rw [spectralPower_rescale Q s _ _ (guardedScale_pos _ r) (fun i => (hs i).le)] at hy hz
  change Tendsto (fun k => guardedNdbIterate (spectralMatrix Q s) r k) atTop _
  simpa only [guardedNdbIterate, guardedNormalize_spectral] using hy.prodMk_nhds hz

/-- The guarded NDB hypotheses are satisfiable, arXiv:2602.02016v2, §3.3–3.4. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    (∀ i : Fin 1, (0 : ℝ) < (fun _ => (1 : ℝ)) i) := by
  norm_num [Orthogonal, Muon.OrthonormalColumns]

/-- The complete guarded two-call inverse-fourth-root solver converges.
Both iteration budgets grow together; an exact first square root is never
substituted. No PI accuracy or external upper bound is assumed.
Source: arXiv:2602.02016v2, §3.3–3.4, corrected finite NDB chaining. -/
theorem guardedInverseFourth_convergence (Q : Matrix (Fin n) (Fin n) ℝ)
    (s : Fin n → ℝ) (r : ℝ) (hQ : Orthogonal Q) (hs : ∀ i, 0 < s i) :
    Tendsto (fun k : ℕ => guardedInverseFourth (spectralMatrix Q s) r (k + 1)) atTop
      (𝓝 (spectralPower Q s (-(1 / 4 : ℝ)))) := by
  have hd := guarded_spectrum_bounds Q s r hQ hs
  have h := (ndb_matrix_finite_chain_convergence Q
    (fun i => s i / guardedScale (spectralMatrix Q s) r) hQ
    (fun i => (hd i).1) (fun i => lt_trans (hd i).2 (by norm_num))).const_smul
      (guardedScale (spectralMatrix Q s) r ^ (-(1 / 4 : ℝ)))
  rw [spectralPower_rescale Q s _ _ (guardedScale_pos _ r) (fun i => (hs i).le)] at h
  simpa only [guardedInverseFourth, guardedNormalize_spectral] using h

/-- The finite-chain hypotheses are satisfiable, arXiv:2602.02016v2, §3.3–3.4. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    (∀ i : Fin 1, (0 : ℝ) < (fun _ => (1 : ℝ)) i) := by
  norm_num [Orthogonal, Muon.OrthonormalColumns]

end Transformer.DASH
