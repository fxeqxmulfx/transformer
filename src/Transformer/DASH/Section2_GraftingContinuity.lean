/-
# DASH — passing numerical root convergence through the parameter update

arXiv:2602.02016v2, §2.2–2.3. Continuity transfers the corrected
iterative root limits to preconditioning and grafted updates. Grafting
is continuous at nonzero directions; zero gradients are handled exactly
by the optimizer pipeline rather than by dividing by their norms.
-/

import Transformer.DASH.Section2_GraftingDomain

open scoped BigOperators Matrix Topology
open Filter

noncomputable section

namespace Transformer.DASH

variable {m n : ℕ}

/-- The norm used by the actual grafting algorithm is continuous.
Source: arXiv:2602.02016v2, §2.3, Frobenius norm ratio. -/
theorem continuous_frobeniusNorm :
    Continuous (frobeniusNorm : Matrix (Fin m) (Fin n) ℝ → ℝ) := by
  unfold frobeniusNorm Muon.squaredFrobenius
  fun_prop

/-- Rectangular left/right preconditioning is jointly continuous in
its two root buffers. Source: arXiv:2602.02016v2, §2.2–2.3, `U=L^(-1/4)GR^(-1/4)`. -/
theorem continuous_preconditionedGradient (G : Matrix (Fin m) (Fin n) ℝ) :
    Continuous (fun buffers : Matrix (Fin m) (Fin m) ℝ × Matrix (Fin n) (Fin n) ℝ =>
      preconditionedGradient buffers.1 G buffers.2) := by
  apply continuous_pi
  intro i
  apply continuous_pi
  intro j
  simp only [preconditionedGradient, Matrix.mul_apply]
  fun_prop

/-- Numerical root-buffer convergence passes to the actual rectangular
preconditioned gradient. Source: arXiv:2602.02016v2, §2.2–2.3, inverse-root update. -/
theorem preconditionedGradient_tendsto
    (P : ℕ → Matrix (Fin m) (Fin m) ℝ) (Q : ℕ → Matrix (Fin n) (Fin n) ℝ)
    (L : Matrix (Fin m) (Fin m) ℝ) (R : Matrix (Fin n) (Fin n) ℝ)
    (G : Matrix (Fin m) (Fin n) ℝ) (hP : Tendsto P atTop (𝓝 L))
    (hQ : Tendsto Q atTop (𝓝 R)) :
    Tendsto (fun k => preconditionedGradient (P k) G (Q k)) atTop
      (𝓝 (preconditionedGradient L G R)) := by
  apply tendsto_pi_nhds.mpr
  intro i
  apply tendsto_pi_nhds.mpr
  intro j
  simp only [preconditionedGradient, Matrix.mul_apply]
  apply tendsto_finsetSum
  intro a ha
  apply Tendsto.mul
  · apply tendsto_finsetSum
    intro b hb
    exact ((tendsto_pi_nhds.mp (tendsto_pi_nhds.mp hP i) b).mul tendsto_const_nhds)
  · exact tendsto_pi_nhds.mp (tendsto_pi_nhds.mp hQ a) j

/-- The buffer-convergence hypotheses are satisfiable,
arXiv:2602.02016v2, §2.2–2.3. -/
example : Tendsto (fun _ : ℕ => (1 : Matrix (Fin 1) (Fin 1) ℝ)) atTop (𝓝 1) ∧
    Tendsto (fun _ : ℕ => (1 : Matrix (Fin 1) (Fin 1) ℝ)) atTop (𝓝 1) :=
  ⟨tendsto_const_nhds, tendsto_const_nhds⟩

/-- The computed grafted parameter update converges when its limiting
Shampoo direction is nonzero. The denominator condition follows from
invertibility for regularized Shampoo and a nonzero gradient.
Source: arXiv:2602.02016v2, §2.3, norm matching and parameter update. -/
theorem graftedStep_tendsto (U : ℕ → Matrix (Fin m) (Fin n) ℝ)
    (V θ P : Matrix (Fin m) (Fin n) ℝ) (η : ℝ)
    (h : Tendsto U atTop (𝓝 V)) (hV : V ≠ 0) :
    Tendsto (fun k => graftedStep η θ (U k) P) atTop (𝓝 (graftedStep η θ V P)) := by
  have hnorm := (continuous_frobeniusNorm.tendsto V).comp h
  have hscale := (tendsto_const_nhds (x := frobeniusNorm P)).div hnorm
    (frobeniusNorm_pos V hV).ne'
  exact (tendsto_const_nhds (x := θ)).sub ((hscale.smul h).const_smul η)

/-- Grafting convergence has admissible nonzero directions,
arXiv:2602.02016v2, §2.3. -/
example : Tendsto (fun _ : ℕ => (1 : Matrix (Fin 1) (Fin 1) ℝ)) atTop (𝓝 1) ∧
    (1 : Matrix (Fin 1) (Fin 1) ℝ) ≠ 0 := by
  refine ⟨tendsto_const_nhds, ?_⟩
  intro h
  have := congrFun (congrFun h 0) 0
  norm_num at this

end Transformer.DASH
