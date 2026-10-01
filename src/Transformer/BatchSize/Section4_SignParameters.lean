/-
# Smooth signed drift on its true positive-noise parameter domain

arXiv:2506.12543v1, Section 4.3, equation (3), Theorem 1.
The error-function response is evaluated on the actual gradient/noise pair.
The compact parameter range stays away from every zero denominator.
-/

import Transformer.BatchSize.Section4_ModelParameterBounds

open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- The signed drift as a function of its actual signal/noise
parameters, Section 4.3 (3), prior to state-dependent composition. -/
def signedParameterDrift (d B : ℕ) (p : EucSpace d × EucSpace d) : EucSpace d :=
  -WithLp.toLp 2 (fun k => signResponse B (p.2 k) (p.1 k))

/-- The true smooth domain of the signed Gaussian response,
Section 4.3 (3): every coordinate standard deviation is positive. -/
def positiveNoiseParameters (d : ℕ) : Set (EucSpace d × EucSpace d) :=
  {p | ∀ k, 0 < p.2 k}

/-- The bounded coordinate standard-deviation box from the model,
Section 4.3, Theorem 1. Its actual Euclidean coordinates lie in [c,L]. -/
def noiseParameterBox (d : ℕ) (c L : ℝ) : Set (EucSpace d) :=
  (WithLp.toLp 2) '' {z : Fin d → ℝ | ∀ k, z k ∈ Set.Icc c L}

/-- The signed response's true positive-noise parameter domain is
open, Section 4.3 (3), so ordinary state derivatives apply throughout. -/
theorem positiveNoiseParameters_isOpen (d : ℕ) : IsOpen (positiveNoiseParameters d) := by
  have heq : positiveNoiseParameters d = ⋂ k : Fin d,
      {p : EucSpace d × EucSpace d | 0 < p.2 k} := by ext p; simp [positiveNoiseParameters]
  rw [heq]
  exact isOpen_iInter_of_finite fun k => isOpen_lt continuous_const
    ((PiLp.continuous_apply 2 (fun _ : Fin d => ℝ) k).comp continuous_snd)

/-- The bounded actual coordinate box is compact, Section 4.3,
Theorem 1. Finite product compactness is preserved by the genuine
Euclidean coordinate identification. -/
theorem noiseParameterBox_isCompact (d : ℕ) (c L : ℝ) : IsCompact (noiseParameterBox d c L) :=
  (isCompact_pi_infinite (fun _ : Fin d => isCompact_Icc)).image
    (PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin d => ℝ)).symm.continuous

/-- The actual signed drift is C2 on positive-noise parameters,
Section 4.3 (3), with its true coordinatewise error function. -/
theorem signedParameterDrift_contDiffOn (d B : ℕ) :
    ContDiffOn ℝ 2 (signedParameterDrift d B) (positiveNoiseParameters d) := by
  apply ContDiffOn.neg
  apply contDiffOn_piLp' 2
  intro k
  have hx : ContDiffOn ℝ 2 (fun p : EucSpace d × EucSpace d => p.1 k)
      (positiveNoiseParameters d) :=
    ((contDiff_piLp_apply (𝕜 := ℝ) (i := k) 2).comp contDiff_fst).contDiffOn
  have hs : ContDiffOn ℝ 2 (fun p : EucSpace d × EucSpace d => p.2 k)
      (positiveNoiseParameters d) :=
    ((contDiff_piLp_apply (𝕜 := ℝ) (i := k) 2).comp contDiff_snd).contDiffOn
  exact (errorFunction_contDiff.of_le (by simp)).comp_contDiffOn
    ((contDiffOn_const.mul hx).div hs (fun p hp => (hp k).ne'))

/-- Nonvacuity of the positive-noise domain used by the signed
parameter derivative bounds, Section 4.3 (3). -/
example : (0 : ℝ) < 1 ∧
    (EuclideanSpace.single (0 : Fin 1) (1 : ℝ), WithLp.toLp 2 (fun _ : Fin 1 => (1 : ℝ))) ∈
      positiveNoiseParameters 1 := by
  refine ⟨by norm_num, fun k => ?_⟩
  fin_cases k
  norm_num

end Transformer.BatchSize
