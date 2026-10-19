module

public import Transformer.MDW.FreeBasic
public import Transformer.MagnitudeDirection.SectionA_TrainingQuadratic

/-!
# Kernel-checked MDW checkpoints

Shared scalar values distinguish native Adam, raw AMSGrad, raw Schedule-Free
variance and its running maximum. A zero-loss-gradient MDW step checks that
decay changes the actual physical weight while retaining the direction sphere.
Expected values are literal constants in the independent PyTorch tests.
-/

@[expose] public section

noncomputable section

namespace Transformer.MDW

open MagnitudeDirection ScheduleFree

/-- The native Adam callback has its completed-update bias corrections.
Source: specialization of the native callback used in MD Appendix A, Algorithm 2. -/
theorem adam_callback_checkpoint :
    (adamVectorCallback (1/2) (1/2) (3/4) (initialAdamBuffers (Fin 1))
      (fun _ => 2) (fun _ => 2) (1/10)).2 0 = 48/25 := by
  norm_num [adamVectorCallback, initialAdamBuffers]

/-- Physical auxiliary AdamW decay uses the old coordinate, outside the moments.
Source: local auxiliary-decay extension of MD Appendix A, Algorithm 2. -/
theorem adamw_auxiliary_checkpoint :
    (adamVectorCallback (1/2) (1/2) (3/4) (initialAdamBuffers (Fin 1))
      (fun _ => 2) (fun _ => 2) (1/10)).2 0 - (1/10) * (1/5) * 2 = 47/25 := by
  rw [adam_callback_checkpoint]
  norm_num

/-- The raw AMSGrad convention remains distinct from bias-corrected native Adam.
Source: arXiv:1904.03590v4, Algorithm 1, applied as the MD Appendix A callback. -/
theorem amsgrad_callback_checkpoint :
    (amsgradVectorCallback (1/2) (1/2) (3/4) (zeroAMSGradBuffers (Fin 1))
      (fun _ => 2) (fun _ => 2) (1/10)).2 0 = 29/15 := by
  norm_num [amsgradVectorCallback, zeroAMSGradBuffers]

/-- Retained raw history used by the decreasing-variance checkpoint. -/
def freeHistory : AMSGrad.State (Fin 1) :=
  ⟨fun _ => 2, fun _ => 2, fun _ => 4, fun _ => 4, 1⟩

/-- Unit effective rate with interpolation one half and no decay. -/
def freeParameters : AdamParameters := ⟨1, 0, 0, 1/2, 0, 1⟩

/-- A smaller new gradient distinguishes raw variance from the retained AMSGrad maximum.
Source: local AMSGrad extension of arXiv:2405.15682v4, §4.4, Algorithm 1. -/
theorem free_variance_checkpoint :
    (freeStepWith id false freeParameters 1 freeHistory (fun _ => 1)).base 0 = 3/2 ∧
      (freeStepWith id true freeParameters 1 freeHistory (fun _ => 1)).base 0 = 5/3 := by
  norm_num [freeStepWith, freeProposal, freeVariance, freeHistory, freeParameters,
    ScheduleFree.AMSGrad.nextSecond, ScheduleFree.AMSGrad.nextMaximum,
    adamLearningRate, warmupFactor]

private theorem scalar_projection (radius : ℝ) (direction : Fin 1 × Fin 1 → ℝ)
    (ij : Fin 1 × Fin 1) : AMSGrad.projectDirection radius direction ij =
    radius / |direction (0, 0)| * direction (0, 0) := by
  rcases ij with ⟨i, j⟩
  fin_cases i
  fin_cases j
  let weight : Matrix (Fin 1) (Fin 1) ℝ := fun i j => direction (i, j)
  have hn : frobeniusNorm weight = |direction (0, 0)| := single_frobeniusNorm weight
  change radius / frobeniusNorm weight * direction (0, 0) = _
  rw [hn]

/-- Nonzero scalar matrix with unit effective row/column gains. -/
def unitWeight : Matrix (Fin 1) (Fin 1) ℝ := fun _ _ => 2

/-- Positive direction and gain rates for the physical-decay checkpoint. -/
def freeMDParameters : AMSGrad.MDParameters := ⟨1/4, 1/8, 0, 1/2, 0, 1⟩

/-- With no loss gradient, physical decay still changes the fused base from 2 to 3/2.
Source: local physical-decay combination of Schedule-Free §4.4 and MD Appendix A. -/
theorem free_mdw_decay_checkpoint (useMaximum : Bool) :
    (freeBaseState (freeMDStep useMaximum freeMDParameters 1 0
      (AMSGrad.mdInitial unitWeight) 0)).weight 0 0 = 3/2 := by
  have half (x : ℝ) : 1/2 * x + 1/2 * x = x := by ring
  rw [freeMDStep_base_weight]
  cases useMaximum <;>
    norm_num [freeBaseState, freeMDProposal, freeStepWith, freeProposal,
      freeVariance, freeMDParameters, unitWeight, AMSGrad.mdInitial,
      ScheduleFree.AMSGrad.mdGradients, ScheduleFree.AMSGrad.mdQuery,
      AMSGrad.factorWeight, AMSGrad.mdQueryWeight, AMSGrad.mdDirectionParameters,
      AMSGrad.mdGainParameters, ScheduleFree.AMSGrad.initial,
      ScheduleFree.AMSGrad.query, ScheduleFree.AMSGrad.nextSecond,
      ScheduleFree.AMSGrad.nextMaximum, scalar_projection,
      adamLearningRate, warmupFactor, fullDirection, frobeniusNorm_eq_sqrt,
      directionGradient, rowGradient, colGradient, fuse,
      softplus_unit_initialization, half]

end Transformer.MDW
