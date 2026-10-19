module

public import Transformer.Homogenized.MvGenerator
public import Mathlib.MeasureTheory.Function.ConditionalExpectation.PullOut
public import Mathlib.Probability.Distributions.Gaussian.Real
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv

/-!
# Initial values and centered squares of the logistic angle martingale

ArXiv:2604.01978v1, §7, `sec: sde.logistic`.
The smooth-test limits first give the compensated angle increment
and the compensated difference of its squares. The source states
the angle itself as a martingale and identifies the bracket of
its centered increment. These forms require the initial terms.

An integrable initial-measurable random variable, held constant
in time, is a martingale on the original horizon. Sums and
differences preserve the actual conditional identities. A bounded
initial-measurable multiplier also preserves them, by the pull-out
property with the actual product integrability proved first.

Subtracting twice the initial angle times its martingale increment
from the square-test martingale gives the original centered square
with the same drift compensator. The initial angle may be random;
no deterministic-initial-value hypothesis is added.
Only times in [0,T] enter the conditional identities.
Initial measurability propagates to every such time by the
monotonicity of the actual filtration. Product integrability
is proved before applying the conditional pull-out property.

The examples realize the algebra on a genuine Gaussian probability
space using sine and cosine of its coordinate, and realize every
centered-square premise on a nonzero constant angle. These are
martingale-algebra witnesses and assert no additional SDE solution.
-/

@[expose] public section

open MeasureTheory Set Filter
open scoped Topology
namespace Transformer.Homogenized

/-- Add actual finite-horizon martingales without changing their filtration.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, restoring the initial angle. -/
theorem IsMartingaleOn.add {Ω : Type*} {m : MeasurableSpace Ω}
    {P : Measure Ω} {ℱ : Filtration ℝ m} {T : ℝ} {M N : ℝ → Ω → ℝ}
    (hM : IsMartingaleOn T ℱ P M) (hN : IsMartingaleOn T ℱ P N) :
    IsMartingaleOn T ℱ P (fun t ω => M t ω + N t ω) := by
  refine ⟨fun t ht => (hM.1 t ht).add (hN.1 t ht),
    fun t ht => (hM.2.1 t ht).add (hN.2.1 t ht), ?_⟩
  intro s hs t ht hst
  filter_upwards [condExp_add (hM.2.1 t ht) (hN.2.1 t ht) (ℱ s),
    hM.2.2 s hs t ht hst, hN.2.2 s hs t ht hst] with ω hp hm hn
  change P[fun ω => M t ω + N t ω | ℱ s] ω = P[M t | ℱ s] ω + P[N t | ℱ s] ω at hp
  rw [hp, hm, hn]

/-- Differences retain the full original conditional martingale identity.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, centering the square test. -/
theorem IsMartingaleOn.sub {Ω : Type*} {m : MeasurableSpace Ω}
    {P : Measure Ω} {ℱ : Filtration ℝ m} {T : ℝ} {M N : ℝ → Ω → ℝ}
    (hM : IsMartingaleOn T ℱ P M) (hN : IsMartingaleOn T ℱ P N) :
    IsMartingaleOn T ℱ P (fun t ω => M t ω - N t ω) := by
  refine ⟨fun t ht => (hM.1 t ht).sub (hN.1 t ht),
    fun t ht => (hM.2.1 t ht).sub (hN.2.1 t ht), ?_⟩
  intro s hs t ht hst
  filter_upwards [condExp_sub (hM.2.1 t ht) (hN.2.1 t ht) (ℱ s),
    hM.2.2 s hs t ht hst, hN.2.2 s hs t ht hst] with ω hp hm hn
  change P[fun ω => M t ω - N t ω | ℱ s] ω = P[M t | ℱ s] ω - P[N t | ℱ s] ω at hp
  rw [hp, hm, hn]

/-- An actual integrable initial random variable is a time-constant martingale.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, the initial angle term. -/
theorem isMartingaleOn_timeConstant {Ω : Type*} {m : MeasurableSpace Ω}
    (P : Measure Ω) [IsFiniteMeasure P] (ℱ : Filtration ℝ m) (T : ℝ)
    (W : Ω → ℝ) (hW : StronglyMeasurable[ℱ 0] W) (hiW : Integrable W P) :
    IsMartingaleOn T ℱ P (fun _ => W) := by
  have hm (t : ℝ) (ht : 0 ≤ t) : StronglyMeasurable[ℱ t] W := hW.mono (ℱ.mono ht)
  refine ⟨fun t ht => hm t ht.1, fun t ht => hiW, ?_⟩
  intro s hs t ht hst
  exact EventuallyEq.of_eq (condExp_of_stronglyMeasurable (ℱ.le s) (hm s hs.1) hiW)

/-- A bounded actual initial multiplier preserves the source martingale identity.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, the centered-square correction. -/
theorem IsMartingaleOn.mul_initial {Ω : Type*} {m : MeasurableSpace Ω}
    {P : Measure Ω} [IsFiniteMeasure P] {ℱ : Filtration ℝ m} {T : ℝ} {M : ℝ → Ω → ℝ}
    (hM : IsMartingaleOn T ℱ P M) (W : Ω → ℝ) (hW : StronglyMeasurable[ℱ 0] W)
    (C : ℝ) (hC : ∀ ω, ‖W ω‖ ≤ C) :
    IsMartingaleOn T ℱ P (fun t ω => W ω * M t ω) := by
  have hm (t : ℝ) (ht : 0 ≤ t) : StronglyMeasurable[ℱ t] W := hW.mono (ℱ.mono ht)
  have hi (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) : Integrable (fun ω => W ω * M t ω) P :=
    (hM.2.1 t ht).bdd_mul (hW.mono (ℱ.le 0)).aestronglyMeasurable (ae_of_all P hC)
  refine ⟨fun t ht => (hm t ht.1).mul (hM.1 t ht), hi, ?_⟩
  intro s hs t ht hst
  filter_upwards [condExp_mul_of_stronglyMeasurable_left (hm s hs.1) (hi t ht) (hM.2.1 t ht),
    hM.2.2 s hs t ht hst] with ω hp hmt
  change P[fun ω => W ω * M t ω | ℱ s] ω = W ω * P[M t | ℱ s] ω at hp
  rw [hp, hmt]

/-- The original square-test martingale gives the centered square with the
same actual bracket density. Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`.
The two martingale inputs are the forms supplied by the smooth-test limits. -/
theorem isMartingaleOn_centered_square {Ω : Type*} {m : MeasurableSpace Ω}
    {P : Measure Ω} [IsFiniteMeasure P] {ℱ : Filtration ℝ m} {T : ℝ}
    (θ q : ℝ → Ω → ℝ)
    (hInc : IsMartingaleOn T ℱ P (fun t ω => θ t ω - θ 0 ω))
    (hSq : IsMartingaleOn T ℱ P (fun t ω => θ t ω ^ 2 - θ 0 ω ^ 2 - ∫ s in (0 : ℝ)..t, q s ω))
    (hθ₀ : StronglyMeasurable[ℱ 0] (θ 0)) (C : ℝ) (hC : ∀ ω, ‖θ 0 ω‖ ≤ C) :
    IsMartingaleOn T ℱ P (fun t ω => (θ t ω - θ 0 ω) ^ 2 - ∫ s in (0 : ℝ)..t, q s ω) := by
  have hm := hInc.mul_initial (fun ω => 2 * θ 0 ω) (stronglyMeasurable_const.mul hθ₀)
    (2 * C) (fun ω => by
      rw [norm_mul, Real.norm_eq_abs, abs_of_pos (by norm_num : (0 : ℝ) < 2)]
      exact mul_le_mul_of_nonneg_left (hC ω) (by norm_num))
  have h := hSq.sub hm
  convert h using 1
  funext t ω
  ring

/-- Nonconstant Gaussian-coordinate sine and cosine realize both input
martingales, initial measurability, integrability and the multiplier bound. -/
example : ∃ (ℱ : Filtration ℝ (inferInstance : MeasurableSpace ℝ))
    (M N : ℝ → ℝ → ℝ) (W : ℝ → ℝ),
    IsMartingaleOn 1 ℱ (ProbabilityTheory.gaussianReal 0 1) M ∧
    IsMartingaleOn 1 ℱ (ProbabilityTheory.gaussianReal 0 1) N ∧
    StronglyMeasurable[ℱ 0] W ∧ Integrable W (ProbabilityTheory.gaussianReal 0 1) ∧
    (∀ ω, ‖W ω‖ ≤ 1) ∧ M 0 0 = 0 ∧ M 0 (Real.pi / 2) = 1 := by
  let ℱ := Filtration.const (ι := ℝ) (m := (inferInstance : MeasurableSpace ℝ)) inferInstance le_rfl
  have hs : StronglyMeasurable Real.sin := Real.continuous_sin.stronglyMeasurable
  have hc : StronglyMeasurable Real.cos := Real.continuous_cos.stronglyMeasurable
  have hsb (ω : ℝ) : ‖Real.sin ω‖ ≤ 1 := by simpa only [Real.norm_eq_abs] using Real.abs_sin_le_one ω
  have hcb (ω : ℝ) : ‖Real.cos ω‖ ≤ 1 := by simpa only [Real.norm_eq_abs] using Real.abs_cos_le_one ω
  have hsi : Integrable Real.sin (ProbabilityTheory.gaussianReal 0 1) :=
    Integrable.of_bound hs.aestronglyMeasurable 1 (ae_of_all _ hsb)
  have hci : Integrable Real.cos (ProbabilityTheory.gaussianReal 0 1) :=
    Integrable.of_bound hc.aestronglyMeasurable 1 (ae_of_all _ hcb)
  exact ⟨ℱ, fun _ => Real.sin, fun _ => Real.cos, Real.cos,
    isMartingaleOn_timeConstant _ ℱ 1 Real.sin hs hsi,
    isMartingaleOn_timeConstant _ ℱ 1 Real.cos hc hci,
    hc, hci, hcb, Real.sin_zero, Real.sin_pi_div_two⟩

/-- A nonzero actual constant angle with zero bracket density realizes both
compensated martingales and all initial-value inputs of the centered-square lemma. -/
example (ℱ : Filtration ℝ (inferInstance : MeasurableSpace Unit)) :
    IsMartingaleOn 1 ℱ (Measure.dirac ()) (fun _ _ => (1 : ℝ) - 1) ∧
    IsMartingaleOn 1 ℱ (Measure.dirac ())
      (fun t _ => (1 : ℝ) ^ 2 - 1 ^ 2 - ∫ _ in (0 : ℝ)..t, (0 : ℝ)) ∧
    StronglyMeasurable[ℱ 0] (fun _ : Unit => (1 : ℝ)) ∧
    ∀ _ω : Unit, ‖(1 : ℝ)‖ ≤ 1 := by
  have h := isMartingaleOn_const 1 ℱ (Measure.dirac ()) 0
  exact ⟨by simpa only [sub_self] using h,
    by simpa only [sub_self, intervalIntegral.integral_zero] using h,
    stronglyMeasurable_const, fun ω => by norm_num⟩

end Transformer.Homogenized
