import Transformer.ScheduleFree.SectionD_Optimistic
import Transformer.ScheduleFree.Section4_AdamW
import Transformer.Grokking.AdamW.ScalarRecurrence
import Transformer.MagnitudeDirection.SectionA_AMSGradCallbacks
import Transformer.AdamBeyond.Section5_AdamNC
import Transformer.AMSGrad.Section5_AdamX
import Transformer.DASH.Section2_Models
import Transformer.Magma.Section3_Algorithm
import Transformer.AdaFisher.Section3_Algorithm
import Transformer.Optimization.Basic

/-!
# Optimizer recurrence checkpoints

These rational specializations of the actual source definitions are checked
by Lean's kernel. `optimizers/tests/test_lean_cases.py` uses the same inputs
and outputs. Expected values are written explicitly, not exported from the
Python implementation. The full nonrational/matrix recurrences are audited
in optimizers/LEAN_CORRESPONDENCE.md and tested numerically there.
-/

open Transformer

noncomputable section

-- arXiv:1904.03590v4 Algorithm 1, via the actual AMSGrad callback.
example : (MagnitudeDirection.amsgradVectorCallback (1/2) (1/2) (3/4)
    (MagnitudeDirection.zeroAMSGradBuffers (Fin 1)) (fun _ => 2) (fun _ => 2) (1/10)).2 0 =
      (29/15 : ℝ) := by
  norm_num [MagnitudeDirection.amsgradVectorCallback, MagnitudeDirection.zeroAMSGradBuffers]

-- arXiv:1904.03590v4 Algorithm 2: rescale before taking the maximum.
example : AMSGrad.adamXRule (d := 1) (fun t => (1/2 : ℝ) / t) 2
    (fun _ => 4) (fun _ => 1) 0 = 9 := by
  norm_num [AMSGrad.adamXRule]

-- arXiv:1904.09237 Section 2 and Algorithm 3: the same mean-square update.
example : AdamBeyond.adagradRule (d := 1) 2 (fun _ => 4) (fun _ => 16) 0 = 10 := by
  norm_num [AdamBeyond.adagradRule]

example : AdamBeyond.adamNCRule (d := 1) (fun t => 1 - 1 / (t : ℝ)) 2
    (fun _ => 4) (fun _ => 16) 0 = 10 := by
  norm_num [AdamBeyond.adamNCRule]

-- Native AdamW's retained raw buffers and positive completed-update clock.
example : (Grokking.AdamW.scalarNativeStep (1/2) (3/4) (1/2) (1/5) (1/10)
    (Grokking.AdamW.seededScalarState 2) 2).parameter = (47/25 : ℝ) := by
  norm_num [Grokking.AdamW.scalarNativeStep, Grokking.AdamW.seededScalarState,
    Grokking.AdamW.nextBufferDirection]

-- arXiv:2502.16982 Section 2.1: one of the five polynomial iterations.
example : Muon.schulzPolynomial (34445/10000) (-47750/10000) (20315/10000) 1 =
    (701/1000 : ℝ) := by
  norm_num [Muon.schulzPolynomial]

-- arXiv:2405.16397v3 Section 3.2: gamma weights the FRESH factor.
example : AdaFisher.factorEMA (d := 1) (4/5) (fun _ => 2) (fun _ => 7) 0 = 6 := by
  norm_num [AdaFisher.factorEMA]

-- arXiv:2602.15322v1 Algorithm 1, the explicit zero-cosine convention.
example : Magma.damping 2 (1/2) (0 : ℝ) 0 = (1/2 : ℝ) := by
  norm_num [Magma.damping, Magma.cosine, Real.sigmoid]

-- The exact-real descent guard rejects an antiparallel candidate.
example : Optimization.descentGuard (1/2) (2 : ℝ) (-1) = 2 := by
  norm_num [Optimization.descentGuard]

-- arXiv:2405.15682v4 Algorithm 1, including decay at y and epsilon after sqrt(v).
example : (ScheduleFree.adamStep (d := 1) ⟨1/5, 3/10, 0, 3/5, 0, 1/2⟩ 0
    ⟨fun _ => 2, fun _ => 2, fun _ => 0, 0⟩ (fun _ => 2)).evaluation 0 = (43/25 : ℝ) := by
  norm_num [ScheduleFree.adamStep, ScheduleFree.adamQuery, ScheduleFree.adamLearningRate,
    ScheduleFree.warmupFactor]

-- Section 2: the initial base point is included in the uniform average.
example : ScheduleFree.average (fun _ => 1) (fun t => if t = 0 then (2 : ℝ) else 1) 2 =
    (3/2 : ℝ) := by
  norm_num [ScheduleFree.average, ScheduleFree.weightSum, Finset.sum_range_succ]

-- Appendix D.1: an all-zero prefix followed by the first nonzero gradient.
example : (ScheduleFree.optimisticRun 1 (fun t : ℕ => if t = 0 then (0 : ℝ) else 1) 2).z = -1 := by
  norm_num [ScheduleFree.optimisticRun, ScheduleFree.optimisticStep,
    ScheduleFree.clipGradient, ScheduleFree.projectBall]

-- Appendix D.1 includes the degenerate radius-zero ball.
example : (ScheduleFree.optimisticRun 0 (fun _ => (1 : ℝ)) 1).z = 0 := by
  norm_num [ScheduleFree.optimisticRun, ScheduleFree.optimisticStep,
    ScheduleFree.clipGradient, ScheduleFree.projectBall]

#print axioms ScheduleFree.optimistic_regret
