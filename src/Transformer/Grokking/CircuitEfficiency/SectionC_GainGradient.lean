import Transformer.Grokking.CircuitEfficiency.SectionC_PhysicalGain

/-!+# True CE derivatives of the physical gained forward

Source: Varma et al., arXiv:2309.02390v1, section 3, Efficiency
and Slow vs fast learning, and appendix C, sim-overall-logits and
train-loss formula. Both circuits now produce logits through fixed
physical readout gains and their two actual trainable coordinates.
Differentiate the true finite-class CE in each coordinate, including
at zero products; no gain-dependent norm penalty is added to CE.

Every raw callback below is proved to be its actual coordinate
derivative. Gains affect both the score and its chain-rule factor.
Unit gains recover the source's previous actual plain-CE callback.
This degree-two fixed-table forward differs from the source's
assumed exponent-1.2 norm costs. The file does not claim native
convergence, rule selection, learned GPTMini circuits or a bridge to
floating-point autograd. Optimizer state enters only as current
parameters when evaluating these mathematical CE derivatives.
-/

namespace Transformer.Grokking.CircuitEfficiency

/-- Actual multiclass training CE of four physical parameters.
Source: appendix C's train loss, with section 3 efficiency implemented
by fixed readout gains rather than an assigned circuit-norm cost. -/
noncomputable def gainSubweightLoss (remaining : ℕ)
    (genGain memGain a b c d : ℝ) : ℝ :=
  tableTrainCE remaining (physicalCircuitScore genGain a b + physicalCircuitScore memGain c d)

/-- Fixed readout coefficient of each independently trained factor.
Source: appendix C's Gen/Mem pair order, with the explicit physical
forward deviation to section 3's Efficiency model. -/
def nativeFactorGain (genGain memGain : ℝ) : Fin 4 → ℝ :=
  ![genGain, genGain, memGain, memGain]

/-- Candidate actual CE callback at the current four parameters.
Source: appendix C's product chain rule; its derivative status is
proved below, not assumed as successful optimization feedback. -/
noncomputable def rawGainNativeGradient (remaining : ℕ) (genGain memGain : ℝ)
    (state : NativeSubweightState) (i : Fin 4) : ℝ :=
  nativeFactorGain genGain memGain i * (state (nativeFactorPartner i)).parameter *
    (-((remaining : ℝ) + 1) / (Real.exp (gainNativeTotalScore genGain memGain state) + (remaining : ℝ) + 1))

/-- True first-Gen coordinate derivative, retaining the gained score
of both circuits. Source: appendix C's train CE and section 3's
two-factor parameterization, with fixed physical readout gains. -/
theorem gain_subweight_first_deriv (remaining : ℕ) (genGain memGain a b c d : ℝ) :
    HasDerivAt (fun t => gainSubweightLoss remaining genGain memGain t b c d)
      (genGain * b * (-((remaining : ℝ) + 1) /
        (Real.exp (genGain * (a * b) + memGain * (c * d)) + (remaining : ℝ) + 1))) a := by
  have hp := (((hasDerivAt_id a).mul_const b).const_mul genGain).add_const (memGain * (c * d))
  have ht := (table_train_ce_deriv remaining (genGain * (a * b) + memGain * (c * d))).comp a hp
  convert ht using 1
  · funext t
    rfl
  · ring

/-- True second-Gen coordinate derivative, including the actual
gain in its chain factor. Source: appendix C's train CE and product
logits; the partner is the other trainable coordinate, not a teacher. -/
theorem gain_subweight_second_deriv (remaining : ℕ) (genGain memGain a b c d : ℝ) :
    HasDerivAt (fun t => gainSubweightLoss remaining genGain memGain a t c d)
      (genGain * a * (-((remaining : ℝ) + 1) /
        (Real.exp (genGain * (a * b) + memGain * (c * d)) + (remaining : ℝ) + 1))) b := by
  have hp := (((hasDerivAt_id b).const_mul a).const_mul genGain).add_const (memGain * (c * d))
  have ht := (table_train_ce_deriv remaining (genGain * (a * b) + memGain * (c * d))).comp b hp
  convert ht using 1
  · funext t
    rfl
  · ring

/-- True first-Mem coordinate derivative of the same joint CE.
Source: appendix C's training logit; both actual circuit outputs
remain in the denominator when differentiating a Mem parameter. -/
theorem gain_subweight_third_deriv (remaining : ℕ) (genGain memGain a b c d : ℝ) :
    HasDerivAt (fun t => gainSubweightLoss remaining genGain memGain a b t d)
      (memGain * d * (-((remaining : ℝ) + 1) /
        (Real.exp (genGain * (a * b) + memGain * (c * d)) + (remaining : ℝ) + 1))) c := by
  have hp := (((hasDerivAt_id c).mul_const d).const_mul memGain).const_add (genGain * (a * b))
  have ht := (table_train_ce_deriv remaining (genGain * (a * b) + memGain * (c * d))).comp c hp
  convert ht using 1
  · funext t
    rfl
  · ring

/-- True second-Mem derivative, valid at every finite parameter
point including the source's zero first-factor initialization.
Source: appendix C's product forward and true train CE formula. -/
theorem gain_subweight_fourth_deriv (remaining : ℕ) (genGain memGain a b c d : ℝ) :
    HasDerivAt (fun t => gainSubweightLoss remaining genGain memGain a b c t)
      (memGain * c * (-((remaining : ℝ) + 1) /
        (Real.exp (genGain * (a * b) + memGain * (c * d)) + (remaining : ℝ) + 1))) d := by
  have hp := (((hasDerivAt_id d).const_mul c).const_mul memGain).const_add (genGain * (a * b))
  have ht := (table_train_ce_deriv remaining (genGain * (a * b) + memGain * (c * d))).comp d hp
  convert ht using 1
  · funext t
    rfl
  · ring

/-- The complete callback vector consists of the actual four CE
partials, not a prescribed future gradient stream. Source: appendix C,
eq:dynamics with the explicit gained-forward/uncoupled-CE deviation;
all coordinates evaluate the same current parameter point. -/
theorem gain_native_raw_gradient_derivatives (remaining : ℕ) (genGain memGain : ℝ)
    (state : NativeSubweightState) :
    let p := nativeSubweightParameters state
    rawGainNativeGradient remaining genGain memGain state =
      ![deriv (fun t => gainSubweightLoss remaining genGain memGain t p.1.2 p.2.1 p.2.2) p.1.1,
        deriv (fun t => gainSubweightLoss remaining genGain memGain p.1.1 t p.2.1 p.2.2) p.1.2,
        deriv (fun t => gainSubweightLoss remaining genGain memGain p.1.1 p.1.2 t p.2.2) p.2.1,
        deriv (fun t => gainSubweightLoss remaining genGain memGain p.1.1 p.1.2 p.2.1 t) p.2.2] := by
  dsimp only
  rw [(gain_subweight_first_deriv remaining genGain memGain _ _ _ _).deriv,
    (gain_subweight_second_deriv remaining genGain memGain _ _ _ _).deriv,
    (gain_subweight_third_deriv remaining genGain memGain _ _ _ _).deriv,
    (gain_subweight_fourth_deriv remaining genGain memGain _ _ _ _).deriv]
  funext i
  fin_cases i <;> simp [rawGainNativeGradient, nativeFactorGain,
    nativeFactorPartner, nativeSubweightParameters, gainNativeTotalScore, physicalCircuitScore]

/-- Unit readouts recover the earlier true plain product CE.
Source: appendix C train-loss formula; the specialization removes
neither circuit and adds no source coupled norm penalty. -/
theorem gain_subweight_unit_loss (remaining : ℕ) (a b c d : ℝ) :
    gainSubweightLoss remaining 1 1 a b c d = tableTrainCE remaining (a * b + c * d) := by
  simp only [gainSubweightLoss, physicalCircuitScore, one_mul]

/-- At one-zero-factor source seeds, only the two first coordinates
receive actual CE gradients. Source: section 3, Slow vs fast learning,
and appendix C's zero-logit initialization; physical gains multiply
the seed-driven gradients rather than imposing a learning-speed flag. -/
theorem gain_native_raw_gradient_initial (remaining : ℕ) (genGain memGain genSeed memSeed : ℝ) :
    rawGainNativeGradient remaining genGain memGain
      (seededNativeSubweights ((0, genSeed), (0, memSeed))) =
      ![-(genGain * genSeed) * (((remaining : ℝ) + 1) / ((remaining : ℝ) + 2)),
        0, -(memGain * memSeed) * (((remaining : ℝ) + 1) / ((remaining : ℝ) + 2)), 0] := by
  funext i
  fin_cases i <;> simp [rawGainNativeGradient, nativeFactorGain, nativeFactorPartner,
    seededNativeSubweights, Transformer.Grokking.AdamW.seededScalarState,
    gainNativeTotalScore, physicalCircuitScore] <;> ring

/-- Unit-gain actual derivatives coincide with the earlier checked
callback at every parameter/moment state. Source: appendix C's
plain CE specialization; no equality of retained buffers is needed. -/
theorem gain_native_unit_raw_gradient (remaining : ℕ) (state : NativeSubweightState) :
    rawGainNativeGradient remaining 1 1 state = rawNativeSubweightGradient remaining state := by
  funext i
  rw [native_raw_gradient_partner]
  fin_cases i <;> simp [rawGainNativeGradient, nativeFactorGain, gain_native_unit_score]

end Transformer.Grokking.CircuitEfficiency
