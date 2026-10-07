import Transformer.GPTMini.Semantics.DepthEmbedding

/-!
# Quantitative original RMS multipliers for the depth construction

Source: GPTMini RMSNorm.forward at f11b6e2 and the original 64/128
residual widths. A genuine residual of norm at least one has positive
RMS multiplier, squared multiplier at most 128 and normalized norm
at most sixteen. At norm at most 4096 and epsilon in [0,1], its actual
multiplier is at least the fixed positive number r=1/4224.

The local norm hypotheses remain explicit here. The actual raw
embedding establishes them initially; subsequent block induction
must establish them again, rather than supply a prepared RMS scale.
The norm cap is a conservative proof domain, not an altered operator.

Finite shared threshold a=r^3/256 and ordinary FFN gain 1/(2a^2)
compensate the homogeneous plateau. A tied label gain 8/r^2 gives
the later readout margin. These real-arithmetic coefficients do not
assert floating-point equivalence or that AdamW learns these weights.
-/

namespace Transformer.GPTMini.Semantics

open Transformer.Basis

/-- The actual original RMS multiplier at this same residual vector and model width.
Source: rmsNormEps, with its full real denominator and original epsilon. -/
noncomputable def depthScale (mode : Mode) (eps : ℝ) (x : EucSpace (depthConfig mode).d_model) : ℝ :=
  Real.sqrt ((depthConfig mode).d_model : ℝ) /
    Real.sqrt (‖x‖ ^ 2 + ((depthConfig mode).d_model : ℝ) * eps)

/-- This scalar is exactly the genuine normalization operator's multiplier.
Source: original rmsNormEps, without a per-input correction in a parameter matrix. -/
theorem depthScale_rms (mode : Mode) (eps : ℝ) (x : EucSpace (depthConfig mode).d_model) :
    rmsNormEps eps x = depthScale mode eps x • x := by
  rfl

/-- A norm-at-least-one residual has a positive original RMS multiplier at every nonnegative epsilon.
Source: the true positive residual-norm denominator and positive original width. -/
theorem depthScale_pos (mode : Mode) (eps : ℝ) (heps : 0 ≤ eps)
    (x : EucSpace (depthConfig mode).d_model) (hx : 1 ≤ ‖x‖) : 0 < depthScale mode eps x := by
  unfold depthScale
  apply div_pos
  · exact Real.sqrt_pos.mpr (by exact_mod_cast (depthConfig mode).d_model_pos)
  · apply Real.sqrt_pos.mpr
    have ht := mul_nonneg (Nat.cast_nonneg (depthConfig mode).d_model) heps
    nlinarith

example : (0 : ℝ) ≤ 1 / 100000 ∧ 1 ≤ ‖depthAxis .easy 0‖ := by
  exact ⟨by norm_num, by rw [depthAxis_norm]⟩

/-- Both the actual RMS multiplier and its square have uniform upper bounds at the genuine norm lower bound.
Source: the original denominator is at least one, with d_model at most 128. -/
theorem depthScale_upper (mode : Mode) (eps : ℝ) (heps : 0 ≤ eps)
    (x : EucSpace (depthConfig mode).d_model) (hx : 1 ≤ ‖x‖) :
    depthScale mode eps x ≤ 16 ∧ (depthScale mode eps x) ^ 2 ≤ 128 := by
  have hD : (0 : ℝ) ≤ ((depthConfig mode).d_model : ℝ) := Nat.cast_nonneg _
  have hDu : ((depthConfig mode).d_model : ℝ) ≤ 128 := by exact_mod_cast (depthConfig_width mode).2
  have hden : 1 ≤ Real.sqrt (‖x‖ ^ 2 + ((depthConfig mode).d_model : ℝ) * eps) :=
    Real.le_sqrt_of_sq_le (by nlinarith [mul_nonneg hD heps])
  have hdenpos : 0 < Real.sqrt (‖x‖ ^ 2 + ((depthConfig mode).d_model : ℝ) * eps) := by linarith
  have hs : depthScale mode eps x ≤ Real.sqrt ((depthConfig mode).d_model : ℝ) := by
    rw [depthScale, div_le_iff₀ hdenpos]
    nlinarith [Real.sqrt_nonneg ((depthConfig mode).d_model : ℝ)]
  have hnum : Real.sqrt ((depthConfig mode).d_model : ℝ) ≤ 16 :=
    Real.sqrt_le_iff.mpr ⟨by norm_num, by nlinarith⟩
  have hp := depthScale_pos mode eps heps x hx
  constructor
  · exact hs.trans hnum
  · nlinarith [Real.sq_sqrt hD, Real.sqrt_nonneg ((depthConfig mode).d_model : ℝ)]

example : (0 : ℝ) ≤ 0 ∧ 1 ≤ ‖depthAxis .hard 0‖ := by
  exact ⟨by norm_num, by rw [depthAxis_norm]⟩

/-- Every genuine original RMS-normalized residual has norm at most sixteen for positive epsilon.
Source: original rmsNormEps_norm_le and the unchanged depth model widths. -/
theorem depthNormalized_norm (mode : Mode) (eps : ℝ) (heps : 0 < eps)
    (x : EucSpace (depthConfig mode).d_model) : ‖rmsNormEps eps x‖ ≤ 16 := by
  have hDu : ((depthConfig mode).d_model : ℝ) ≤ 128 := by exact_mod_cast (depthConfig_width mode).2
  exact (rmsNormEps_norm_le eps heps (depthConfig mode).d_model_pos x).trans
    (Real.sqrt_le_iff.mpr ⟨by norm_num, by nlinarith⟩)

example : (0 : ℝ) < 1 / 100000 := by norm_num

/-- A fixed positive lower scalar for bounded true residuals, shared by all layers and positions.
Source: the conservative norm-4096, width-at-most-128, epsilon-at-most-one denominator estimate. -/
noncomputable def depthScaleLower : ℝ := 1 / 4224

/-- The fixed lower scalar is strictly positive and no larger than one.
Source: its given finite rational value, needed by genuine propagated amplitude bounds. -/
theorem depthScaleLower_bounds : 0 < depthScaleLower ∧ depthScaleLower ≤ 1 := by
  norm_num [depthScaleLower]

/-- A bounded norm-at-least-one residual attains the same uniform positive lower RMS multiplier.
Source: the actual RMS numerator/denominator, with all local applicability hypotheses retained. -/
theorem depthScale_lower (mode : Mode) (eps : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1)
    (x : EucSpace (depthConfig mode).d_model) (hx : 1 ≤ ‖x‖) (hu : ‖x‖ ≤ 4096) :
    depthScaleLower ≤ depthScale mode eps x := by
  have hD : (0 : ℝ) ≤ ((depthConfig mode).d_model : ℝ) := Nat.cast_nonneg _
  have hDl : (64 : ℝ) ≤ ((depthConfig mode).d_model : ℝ) := by exact_mod_cast (depthConfig_width mode).1
  have hDu : ((depthConfig mode).d_model : ℝ) ≤ 128 := by exact_mod_cast (depthConfig_width mode).2
  have hterm : ((depthConfig mode).d_model : ℝ) * eps ≤ ((depthConfig mode).d_model : ℝ) := by
    simpa only [mul_one] using mul_le_mul_of_nonneg_left hclip hD
  have ht := hterm.trans hDu
  have hd : 0 < ‖x‖ ^ 2 + ((depthConfig mode).d_model : ℝ) * eps := by
    nlinarith [mul_nonneg hD heps]
  have hden : Real.sqrt (‖x‖ ^ 2 + ((depthConfig mode).d_model : ℝ) * eps) ≤ 4224 :=
    Real.sqrt_le_iff.mpr ⟨by norm_num, by nlinarith [norm_nonneg x]⟩
  have hnum : (1 : ℝ) ≤ Real.sqrt ((depthConfig mode).d_model : ℝ) :=
    Real.le_sqrt_of_sq_le (by nlinarith)
  unfold depthScaleLower depthScale
  calc (1 : ℝ) / 4224 ≤ 1 / Real.sqrt (‖x‖ ^ 2 + ((depthConfig mode).d_model : ℝ) * eps) :=
      div_le_div_of_nonneg_left (by norm_num) (Real.sqrt_pos.mpr hd) hden
    _ ≤ Real.sqrt ((depthConfig mode).d_model : ℝ) /
        Real.sqrt (‖x‖ ^ 2 + ((depthConfig mode).d_model : ℝ) * eps) :=
      div_le_div_of_nonneg_right hnum (Real.sqrt_nonneg _)

example : (0 : ℝ) ≤ 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧
    1 ≤ ‖depthAxis .easy 0‖ ∧ ‖depthAxis .easy 0‖ ≤ 4096 := by
  norm_num [depthAxis_norm]

/-- One ordinary finite presence threshold for all depth layers and positions.
Source: a genuine feature amplitude r^2, next RMS floor r and context cap 128. -/
noncomputable def depthThreshold : ℝ := depthScaleLower ^ 3 / 256

/-- This shared threshold is positive and twice it is exactly the required context presence floor.
Source: finite threshold arithmetic, with the actual positive scale floor kept explicit. -/
theorem depthThreshold_bounds : 0 < depthThreshold ∧ 2 * depthThreshold = depthScaleLower ^ 3 / 128 := by
  constructor
  · unfold depthThreshold
    have h := depthScaleLower_bounds.1
    positivity
  · unfold depthThreshold
    ring

/-- The ordinary shared output-matrix coefficient compensates the actual three-hinge plateau.
Source: depthStep's proved amplitude 2*a^2 at constant one. -/
noncomputable def depthDetectorGain : ℝ := 1 / (2 * depthThreshold ^ 2)

/-- The finite FFN gain is positive and its product with the genuine plateau is exactly one.
Source: the proved positive shared threshold, not cancellation of an externally supplied RMS scale. -/
theorem depthDetectorGain_plateau : 0 < depthDetectorGain ∧ depthDetectorGain * (2 * depthThreshold ^ 2) = 1 := by
  have ha := depthThreshold_bounds.1
  constructor
  · unfold depthDetectorGain
    positivity
  · unfold depthDetectorGain
    field_simp

/-- The ordinary tied label coefficient is shared across the vocabulary and every prefix.
Source: the planned strict readout at propagated squared scale at least r^2. -/
noncomputable def depthLabelGain : ℝ := 8 / depthScaleLower ^ 2

/-- The tied coefficient is positive and has fixed margin eight at the lowest genuine squared RMS scale.
Source: given finite readout arithmetic and the proved nonzero lower scalar. -/
theorem depthLabelGain_margin : 0 < depthLabelGain ∧ depthLabelGain * depthScaleLower ^ 2 = 8 := by
  have hr := depthScaleLower_bounds.1
  constructor
  · unfold depthLabelGain
    positivity
  · unfold depthLabelGain
    field_simp

end Transformer.GPTMini.Semantics
