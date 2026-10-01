/-
# Positive scalar MD directions cannot cross their sphere component

arXiv:2606.25971v2, §3.1 and Appendix A, Algorithm 2, training extension.
In one matrix entry the positive-radius sphere has two components. A
normalized gradient step of relative size less than one preserves the
positive component when the direction gradient is positive. Arbitrary
finite raw-gain optimizer outputs remain positive after softplus.
-/

import Transformer.MagnitudeDirection.SectionA_TrainingQuadratic

noncomputable section

namespace Transformer.MagnitudeDirection

variable {R C : Type*}

/-- The normalized scalar direction step is exactly `(1-eta) D` for
positive direction and positive genuine factor gradient. In particular
it cannot change sign when `eta < 1`.
Source: arXiv:2606.25971v2, §3.1, normalized-update training counterexample. -/
theorem normalizedGradientStep_single_positive
    (D G : Matrix (Fin 1) (Fin 1) ℝ) (eta : ℝ)
    (hD : 0 < D 0 0) (hG : 0 < G 0 0) (heta : eta < 1) :
    normalizedGradientStep D G eta 0 0 = (1 - eta) * D 0 0 ∧
      0 < normalizedGradientStep D G eta 0 0 := by
  have he : normalizedGradientStep D G eta 0 0 = (1 - eta) * D 0 0 := by
    dsimp only [normalizedGradientStep, Matrix.sub_apply, Matrix.smul_apply, smul_eq_mul]
    rw [single_frobeniusNorm D, single_frobeniusNorm G, abs_of_pos hD, abs_of_pos hG]
    field_simp
  exact ⟨he, he ▸ mul_pos (by linarith) hD⟩

/-- Positive entries and a positive small learning rate satisfy the premises,
arXiv:2606.25971v2, §3.1, training counterexample. -/
example : (0 : ℝ) < (1 : Matrix (Fin 1) (Fin 1) ℝ) 0 0 ∧
    (0 : ℝ) < (1 : Matrix (Fin 1) (Fin 1) ℝ) 0 0 ∧ (1 / 4 : ℝ) < 1 := by norm_num

/-- Projection of a positive one-entry candidate is the positive unit
direction. Source: arXiv:2606.25971v2, Appendix A, Algorithm 2, line 8. -/
theorem matrixProject_single_positive (W : Matrix (Fin 1) (Fin 1) ℝ)
    (hW : 0 < W 0 0) : matrixProject 1 W = 1 := by
  ext i j
  fin_cases i
  fin_cases j
  simp only [matrixProject, Matrix.smul_apply, smul_eq_mul, single_frobeniusNorm,
    abs_of_pos hW, Matrix.one_apply, ite_true]
  exact div_mul_cancel₀ 1 hW.ne'

/-- Positive projection inputs exist, arXiv:2606.25971v2, Appendix A. -/
example : (0 : ℝ) < (1 : Matrix (Fin 1) (Fin 1) ℝ) 0 0 := by norm_num

/-- Positive fused scalar storage has a positive recovered direction.
Source: arXiv:2606.25971v2, Appendix A, Algorithm 2, lines 1–2. -/
theorem fullDirection_single_positive (s : FullState 1 1) (hW : 0 < s.weight 0 0) :
    0 < fullDirection softplus s 0 0 :=
  div_pos hW (mul_pos (softplus_pos _) (softplus_pos _))

/-- Positive fused storage satisfies this premise,
arXiv:2606.25971v2, Appendix A, training counterexample. -/
example : (0 : ℝ) < (unitGainState (1 : Matrix (Fin 1) (Fin 1) ℝ)).weight 0 0 := by
  norm_num [unitGainState]

/-- With a positive current weight/gradient and `etaW < 1`, the actual
normalized-direction Algorithm 2 proposal has positive fused weight
and positive unit direction. This holds for every stateful raw-gain
optimizer, including Adam or AMSGrad: their finite raw outputs enter
the same strictly positive map. No gain reliability premise is imposed.
Source: arXiv:2606.25971v2, §3.1 and Appendix A, training counterexample. -/
theorem normalizedMDProposal_positive (rowOpt : StatefulGainStep R 1)
    (colOpt : StatefulGainStep C 1) (etaW etaG : ℝ)
    (memory : Unit × R × C) (s : FullState 1 1) (G : Matrix (Fin 1) (Fin 1) ℝ)
    (hW : 0 < s.weight 0 0) (hG : 0 < G 0 0) (heta : etaW < 1) :
    0 < (normalizedMDProposal rowOpt colOpt etaW etaG 1 memory s G).2.weight 0 0 ∧
      fullDirection softplus (normalizedMDProposal rowOpt colOpt etaW etaG 1 memory s G).2 = 1 := by
  let D := fullDirection softplus s
  let row := fun i => softplus (s.rawRow i)
  let col := fun j => softplus (s.rawCol j)
  have hDG : 0 < directionGradient row col G 0 0 := by
    exact mul_pos (mul_pos (softplus_pos _) hG) (softplus_pos _)
  have hC := (normalizedGradientStep_single_positive D (directionGradient row col G) etaW
    (fullDirection_single_positive s hW) hDG heta).2
  have hP := matrixProject_single_positive _ hC
  have he := statefulFullProposal_eq
    (fun _ D H eta => ((), normalizedGradientStep D H eta)) rowOpt colOpt
    etaW etaG 1 memory s G
  change (normalizedMDProposal rowOpt colOpt etaW etaG 1 memory s G).2 = _ at he
  rw [he]
  constructor
  · dsimp only [fullStep, fullCandidate]
    rw [hP]
    simp only [fuse, Matrix.one_apply, ite_true, mul_one]
    exact mul_pos (softplus_pos _) (softplus_pos _)
  · dsimp only [fullDirection, fullStep, fullCandidate]
    rw [unfuse_fuse _ _ _ (fun i => (softplus_pos _).ne') (fun j => (softplus_pos _).ne')]
    exact hP

/-- The complete proposal positivity assumptions are jointly satisfiable
with a positive direction LR and nonzero actual weight/gradient.
Source: arXiv:2606.25971v2, Appendix A, training counterexample. -/
example : (0 : ℝ) < (unitGainState (1 : Matrix (Fin 1) (Fin 1) ℝ)).weight 0 0 ∧
    (0 : ℝ) < (1 : Matrix (Fin 1) (Fin 1) ℝ) 0 0 ∧ (1 / 4 : ℝ) < 1 := by
  norm_num [unitGainState]

end Transformer.MagnitudeDirection
