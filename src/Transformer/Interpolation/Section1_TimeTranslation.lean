/-
# Exact translated and reversed clocks for the original weak equation

ArXiv:2411.04551v3, §1, `eq: vf` and `eq: cauchy.pb`,
and Appendix, the successive phases of `lem: tubular.mass.movement`.

A time translation evaluates every original parameter at the translated
clock. Subtracting the original test primitives at the two endpoints and
changing variables in their flux integral gives the translated weak law.
Interval integrability survives, and its initial law is the law at the offset.

Reversing the translated clock also negates both output matrices, as in
the original vector-field time-rescaling identity. This gives the exact
weak equation for `t ↦ μ(a-t)`, which will identify later autonomous
control phases by applying the proved autonomous-prefix transport theorem.
-/

import Transformer.Interpolation.Section1_TimeRescaling
import Transformer.Interpolation.Section3_TwoRidgeLaw

open Real MeasureTheory Set Filter
open scoped Topology

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Original parameters evaluated at the translated clock.
Source: arXiv:2411.04551v3, §1, `eq: vf`. -/
noncomputable def translateTimeParams (a : ℝ) (θ : TimeParams d) : TimeParams d :=
  fun t => θ (a + t)

/-- The full original field uses exactly the translated original time.
Source: arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem fullVF_translateTimeParams (a : ℝ) (θ : TimeParams d)
    (ρ : Perspective.ProbSphere d) (t : ℝ) (x : EucSpace d) :
    fullVF d (translateTimeParams a θ) ρ t x = fullVF d θ ρ (a + t) x := rfl

/-- Every original weak law translates with its true flux primitive and
interval integrability. Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem cauchyPB_translateTimeParams (a : ℝ) (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ) :
    cauchyPB d (translateTimeParams a θ) (fun t => μ (a + t)) := by
  intro φ hφ t
  let g (s : ℝ) := ∫ x : SSphere d, inner ℝ (gradient φ (x : EucSpace d))
    (fullVF d θ (μ s) s x) ∂(μ s : Measure (SSphere d))
  have heq : (fun s => ∫ x : SSphere d, inner ℝ (gradient φ (x : EucSpace d))
      (fullVF d (translateTimeParams a θ) (μ (a + s)) s x) ∂(μ (a + s) : Measure (SSphere d))) =
      (fun s => g (a + s)) := by
    funext s
    rfl
  rw [heq]
  have hI (r s : ℝ) : IntervalIntegrable g volume r s :=
    ((hμ φ hφ r).1.symm).trans (hμ φ hφ s).1
  have hi : IntervalIntegrable (fun s => g (a + s)) volume 0 t := by
    simpa only [sub_self, add_sub_cancel_left] using (hI a (a + t)).comp_add_left a
  refine ⟨hi, ?_⟩
  rw [intervalIntegral.integral_comp_add_left]
  simp only [add_zero]
  have hleft := (hμ φ hφ a).2
  have hright := (hμ φ hφ (a + t)).2
  have hsum := intervalIntegral.integral_add_adjacent_intervals (hI 0 a) (hI a (a + t))
  dsimp only [g] at hsum
  linarith

/-- Translation by zero preserves all original parameters.
Source: arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem translateTimeParams_zero (θ : TimeParams d) : translateTimeParams 0 θ = θ := by
  funext t
  simp only [translateTimeParams, zero_add]

/-- Successive translations add their actual clock offsets.
Source: arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem translateTimeParams_add (a b : ℝ) (θ : TimeParams d) :
    translateTimeParams a (translateTimeParams b θ) = translateTimeParams (b + a) θ := by
  funext t
  change θ (b + (a + t)) = θ ((b + a) + t)
  rw [add_assoc]

/-- The opposite clock offset recovers the original parameters.
Source: arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem translateTimeParams_inverse (a : ℝ) (θ : TimeParams d) :
    translateTimeParams (-a) (translateTimeParams a θ) = θ := by
  funext t
  change θ (a + (-a + t)) = θ t
  congr 1
  ring

/-- Every translated weak law yields an original weak law by the inverse
clock change. Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem cauchyPB_untranslateTimeParams (a : ℝ) (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d (translateTimeParams a θ) μ) :
    cauchyPB d θ (fun t => μ (-a + t)) := by
  simpa only [translateTimeParams_inverse] using
    cauchyPB_translateTimeParams (-a) (translateTimeParams a θ) μ hμ

/-- Actual original controls realizing the reversed clock `t ↦ a-t`.
Auxiliary for arXiv:2411.04551v3, Appendix, `lem: tubular.mass.movement`. -/
noncomputable def reverseTimeParams (a : ℝ) (θ : TimeParams d) : TimeParams d :=
  rescaleTimeParams (-1) (translateTimeParams a θ)

/-- Clock reversal negates the full original velocity at the reversed
time. Source: arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem fullVF_reverseTimeParams (a : ℝ) (θ : TimeParams d)
    (ρ : Perspective.ProbSphere d) (t : ℝ) (x : EucSpace d) :
    fullVF d (reverseTimeParams a θ) ρ t x = -fullVF d θ ρ (a - t) x := by
  simp only [reverseTimeParams, fullVF_rescaleTimeParams, fullVF_translateTimeParams,
    neg_one_mul, neg_one_smul, sub_eq_add_neg]

/-- The actual original weak equation holds for the reversed curve.
Auxiliary for arXiv:2411.04551v3, §1, `eq: cauchy.pb`, and Appendix,
the later phases in `lem: tubular.mass.movement`. -/
theorem cauchyPB_reverseTimeParams (a : ℝ) (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ) :
    cauchyPB d (reverseTimeParams a θ) (fun t => μ (a - t)) := by
  have hc := cauchyPB_rescaleTimeParams (-1) (by norm_num) (translateTimeParams a θ)
    (fun t => μ (a + t)) (cauchyPB_translateTimeParams a θ μ hμ)
  simpa only [reverseTimeParams, neg_one_mul, sub_eq_add_neg] using hc

/-- Reversing the same clock twice recovers all original controls.
Source: arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem reverseTimeParams_inverse (a : ℝ) (θ : TimeParams d) :
    reverseTimeParams a (reverseTimeParams a θ) = θ := by
  funext t
  simp [reverseTimeParams, rescaleTimeParams, translateTimeParams, smul_smul]

/-- The original and exactly reversed weak equations are equivalent.
Auxiliary for arXiv:2411.04551v3, Appendix, the clock reversal used
for the later phases of `lem: tubular.mass.movement`. -/
theorem cauchyPB_reverseTimeParams_iff (a : ℝ) (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) :
    cauchyPB d (reverseTimeParams a θ) (fun t => μ (a - t)) ↔ cauchyPB d θ μ := by
  constructor
  · intro hμ
    have hc := cauchyPB_reverseTimeParams a (reverseTimeParams a θ) (fun t => μ (a - t)) hμ
    have he : (fun t => μ (a - (a - t))) = μ := by
      funext t
      congr 1
      ring
    simpa only [reverseTimeParams_inverse, he] using hc
  · exact cauchyPB_reverseTimeParams a θ μ

/-- An actual nonzero two-neuron weak flow supplies both the original
and translated weak-solution hypotheses, for every initial probability. -/
example (ν : Perspective.ProbSphere 2) : ∃ (θ : TimeParams 2)
    (μ : ℝ → Perspective.ProbSphere 2),
    cauchyPB 2 θ μ ∧
    cauchyPB 2 (translateTimeParams 1 θ) (fun t => μ (1 + t)) ∧ μ 0 = ν := by
  let p : EucSpace 2 := basePoint 1
  let q : EucSpace 2 := EuclideanSpace.single 1 1
  let θ : TimeParams 2 := fun _ => twoRidgeParams 0 1 p 0 1 q 0 1
  let μ := twoRidgeLaw p 0 1 q 0 1 ν
  have hμ : cauchyPB 2 θ μ := cauchyPB_twoRidgeLaw 0 1 (by norm_num) _ _ _ _ _ _ _
  exact ⟨θ, μ, hμ, cauchyPB_translateTimeParams 1 θ μ hμ, twoRidgeLaw_zero _ _ _ _ _ _ _⟩

end Transformer.Interpolation
