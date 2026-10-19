module

public import Transformer.MDW.FreeDefs

/-!
# Schedule-Free MDW recurrence and storage guarantees

The local combinations retain the raw-variance or AMSGrad recurrence of
arXiv:2405.15682v4, §4.4 and arXiv:1904.03590v4, Algorithm 1. Physical
query-point decay is applied before averaging; the storage repair preserves
the resulting fused base and its positive-radius invariant when nonzero.
-/

@[expose] public section

noncomputable section

namespace Transformer.MDW

open MagnitudeDirection ScheduleFree

variable {ι : Type*} {m n d : ℕ}

/-- With the maximum enabled, the full factor callback is the existing AMSGrad step.
Source: local combination of arXiv:2405.15682v4, §4.4 and AMSGrad Algorithm 1. -/
theorem freeStepWith_amsgrad (project : (ι → ℝ) → (ι → ℝ)) (p : AdamParameters)
    (t : ℕ) (state : AMSGrad.State ι) (gradient : ι → ℝ) :
    freeStepWith project true p t state gradient =
      AMSGrad.stepWith project p t state gradient := rfl

/-- Adam stores the same iterates and variance; it needs no separate maximum buffer. -/
def eraseMaximum (state : AMSGrad.State (Fin d)) : AdamState d :=
  ⟨state.evaluation, state.base, state.second, state.total⟩

/-- With the maximum disabled, unprojected updates equal Algorithm 1's actual AdamW step.
Source: arXiv:2405.15682v4, §4.4, Algorithm 1, including its raw epsilon convention. -/
theorem freeStepWith_adam (p : AdamParameters) (t : ℕ)
    (state : AMSGrad.State (Fin d)) (gradient : Fin d → ℝ) :
    eraseMaximum (freeStepWith id false p t state gradient) =
      adamStep p t (eraseMaximum state) gradient := by
  by_cases h : adamLearningRate p t = 0
  · simp [eraseMaximum, freeStepWith, adamStep, h, ScheduleFree.AMSGrad.coefficient]
    rfl
  · simp only [eraseMaximum, freeStepWith, adamStep, ite_eq_right h]
    rfl

/-- Positive epsilon keeps either variant's actual adaptive denominator positive.
Source: the regularized denominator of arXiv:2405.15682v4, §4.4, Algorithm 1. -/
theorem free_denominator_pos (useMaximum : Bool) (p : AdamParameters)
    (state : AMSGrad.State ι) (gradient : ι → ℝ) (i : ι) (he : 0 < p.epsilon) :
    0 < Real.sqrt (freeVariance useMaximum p state gradient i) + p.epsilon :=
  adam_denominator_pos _ _ he

example : 0 < (⟨1, 0, 0, 1/2, 0, 1⟩ : AdamParameters).epsilon := by norm_num

/-- Repair and latent averaging preserve exactly the independently decayed fused base.
Source: local combination of Schedule-Free §4.4 and MD Appendix A, Algorithm 2. -/
theorem freeMDStep_base_weight (useMaximum : Bool) (p : AMSGrad.MDParameters) (decay : ℝ)
    (t : ℕ) (state : AMSGrad.MDState m n) (gradient : Matrix (Fin m) (Fin n) ℝ) :
    (freeBaseState (freeMDStep useMaximum p decay t state gradient)).weight =
      (freeBaseState (freeMDProposal useMaximum p t state gradient)).weight -
        (adamLearningRate (AMSGrad.mdDirectionParameters p) t * decay) •
          AMSGrad.mdQueryWeight p state := by
  let corrected := freeCorrectedBase useMaximum p decay t state gradient
  have hf := fuse_unfuse (fun i => softplus (corrected.rawRow i))
    (fun j => softplus (corrected.rawCol j)) corrected.weight
    (fun i => (softplus_pos _).ne') (fun j => (softplus_pos _).ne')
  change fuse (fun i => softplus (corrected.rawRow i))
    (fun j => softplus (corrected.rawCol j)) (fullDirection softplus corrected) = _
  rw [show fullDirection softplus corrected = unfuse
      (fun i => softplus (corrected.rawRow i))
      (fun j => softplus (corrected.rawCol j)) corrected.weight from rfl, hf]
  exact correctWeight_weight _ _ _ _ _

/-- The actual corrected base direction has its initial radius whenever storage is nonzero.
Source: local physical-decay extension of MD Appendix A and Schedule-Free §4.4. -/
theorem freeMDStep_base_norm (useMaximum : Bool) (p : AMSGrad.MDParameters) (decay : ℝ)
    (t : ℕ) (state : AMSGrad.MDState m n) (gradient : Matrix (Fin m) (Fin n) ℝ)
    (hr : 0 < state.radius)
    (hw : (freeBaseState (freeMDProposal useMaximum p t state gradient)).weight -
      (adamLearningRate (AMSGrad.mdDirectionParameters p) t * decay) •
        AMSGrad.mdQueryWeight p state ≠ 0) :
    frobeniusNorm (AMSGrad.mdBaseDirection
      (freeMDStep useMaximum p decay t state gradient)) = state.radius := by
  change frobeniusNorm (fullDirection softplus
    (freeCorrectedBase useMaximum p decay t state gradient)) = state.radius
  exact correctWeight_norm _ _ _ _ _ hr hw

example :
    let p : AMSGrad.MDParameters := ⟨0, 0, 0, 1/2, 0, 1⟩
    let state := AMSGrad.mdInitial (1 : Matrix (Fin 1) (Fin 1) ℝ)
    0 < state.radius ∧
      (freeBaseState (freeMDProposal true p 0 state 0)).weight -
        (adamLearningRate (AMSGrad.mdDirectionParameters p) 0 * 1) •
          AMSGrad.mdQueryWeight p state ≠ 0 := by
  constructor
  · norm_num [AMSGrad.mdInitial, frobeniusNorm_eq_sqrt]
  · intro h
    have hentry := congrFun (congrFun h 0) 0
    norm_num [freeBaseState, freeMDProposal, freeStepWith, adamLearningRate,
      AMSGrad.mdDirectionParameters, AMSGrad.mdGainParameters, AMSGrad.mdInitial,
      ScheduleFree.AMSGrad.initial, AMSGrad.factorWeight, fuse,
      softplus_unit_initialization] at hentry

/-- The physical correction changes no raw second moments, maxima or averaging totals.
Source: local MD extension of arXiv:2405.15682v4, §4.4, Algorithm 1. -/
theorem freeMDStep_histories (useMaximum : Bool) (p : AMSGrad.MDParameters) (decay : ℝ)
    (t : ℕ) (state : AMSGrad.MDState m n) (gradient : Matrix (Fin m) (Fin n) ℝ) :
    let next := freeMDStep useMaximum p decay t state gradient
    let adaptive := freeMDProposal useMaximum p t state gradient
    next.direction.second = adaptive.direction.second ∧
      next.direction.maximum = adaptive.direction.maximum ∧
      next.direction.total = adaptive.direction.total ∧
      next.row.second = adaptive.row.second ∧ next.row.maximum = adaptive.row.maximum ∧
      next.row.total = adaptive.row.total ∧ next.column.second = adaptive.column.second ∧
      next.column.maximum = adaptive.column.maximum ∧ next.column.total = adaptive.column.total :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

end Transformer.MDW
