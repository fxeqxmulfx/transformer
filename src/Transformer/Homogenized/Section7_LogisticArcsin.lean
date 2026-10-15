import Transformer.Homogenized.Section7_LogisticArcsinBounds
import Transformer.Homogenized.Section7_LogisticMartingaleLimits
import Transformer.Homogenized.Section7_LogisticMartingaleAlgebra

/-!
# The logistic angle martingale and its actual quadratic variation

ArXiv:2604.01978v1, §7, `sec: sde.logistic`.
The original logistic martingale problem implies that arcsin(u)
is a bounded martingale and that its centered square minus
the actual integral of 2(1-u²) is a martingale on each horizon.
This uses arcsin, correcting the source's printed sin substitution.

Globally smooth tests agree with arcsin(a u) near the entire
closed state interval. Their exact original generators have
the proved bounded residual drift and square-test diffusion term.
Dominated convergence gives the limiting conditional identities,
including the absorbing endpoints. Initial measurability then
restores the angle's initial value and centers its actual square.

The manuscript works on a probability space. The previous formal
statement allowed an arbitrary measure, which does not make a
bounded nonzero angle integrable. A finite-measure hypothesis
is explicit here and suffices; probability measures are included.
The original equation, state interval, filtration, initial random
angle and compensator are retained. Neither limiting martingale
identity nor quadratic variation is an input to the proof.
-/

open Real MeasureTheory Set Filter
open scoped Topology
namespace Transformer.Homogenized

/-- The source's bounded angle martingale and centered-square bracket identity.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`.
The source's probability law is generalized to any finite law, correcting
the previous omission of a finiteness hypothesis. The original martingale
problem and state range imply both conditional identities by smooth limits. -/
theorem logistic_arcsin_martingale {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω)
    [IsFiniteMeasure P] (ℱ : Filtration ℝ m) (u : ℝ → Ω → ℝ) (hu : IsLogisticSolution P ℱ u)
    (hrange : ∀ t, 0 ≤ t → ∀ ω, u t ω ∈ Icc (-1 : ℝ) 1) (T : ℝ) (hT : 0 ≤ T) :
    IsMartingaleOn T ℱ P (fun t ω => Real.arcsin (u t ω)) ∧
      IsMartingaleOn T ℱ P (fun t ω =>
        (Real.arcsin (u t ω) - Real.arcsin (u 0 ω)) ^ 2 -
          ∫ s in (0 : ℝ)..t, 2 * (1 - u s ω ^ 2)) := by
  choose φ hφ heq using fun k => exists_contDiff_arcsin_test (logisticArcsinScale k) (logisticArcsinScale_mem k)
  let F (k : ℕ) (t : ℝ) (ω : Ω) := φ k (u t ω)
  let D (k : ℕ) (t : ℝ) (ω : Ω) := logisticGenerator (φ k) (u t ω)
  let Q (k : ℕ) (t : ℝ) (ω : Ω) := regularizedLogisticVariance (logisticArcsinScale k) (u t ω)
  let D₂ (k : ℕ) (t : ℝ) (ω : Ω) := logisticGenerator (fun v => φ k v ^ 2) (u t ω)
  have hval (k : ℕ) (t : ℝ) (ht : 0 ≤ t) (ω : Ω) :
      F k t ω = Real.arcsin (logisticArcsinScale k * u t ω) := (heq k _ (hrange t ht ω)).eq_of_nhds
  have hD (k : ℕ) (t : ℝ) (ht : 0 ≤ t) (ω : Ω) :
      D k t ω = regularizedLogisticDrift (logisticArcsinScale k) (u t ω) := by
    obtain ⟨-, -, -, -, -, hau⟩ :=
      regularizedLogistic_parameter_bounds _ _ (logisticArcsinScale_mem k) (hrange t ht ω)
    exact logisticGenerator_arcsin_local _ _ hau _ (heq k _ (hrange t ht ω))
  have hFb (k : ℕ) (t : ℝ) (ht : 0 ≤ t) (ω : Ω) : |F k t ω| ≤ Real.pi / 2 := by
    rw [hval k t ht ω]
    exact abs_le.mpr (Real.arcsin_mem_Icc _)
  have hDb (k : ℕ) (t : ℝ) (ht : 0 ≤ t) (ω : Ω) : |D k t ω| ≤ 1 := by
    rw [hD k t ht ω]
    apply (abs_regularizedLogisticDrift_le _ _ (logisticArcsinScale_mem k) (hrange t ht ω)).trans
    calc
      _ ≤ Real.sqrt 1 := Real.sqrt_le_sqrt (by linarith [sq_nonneg (logisticArcsinScale k)])
      _ = 1 := Real.sqrt_one
  have hFlim (t : ℝ) (ht : 0 ≤ t) (ω : Ω) :
      Tendsto (fun k => F k t ω) atTop (𝓝 (Real.arcsin (u t ω))) := by
    have h := (tendsto_logisticArcsinScale.mul_const (u t ω)).arcsin
    apply (show Tendsto (fun k => Real.arcsin (logisticArcsinScale k * u t ω))
      atTop (𝓝 (Real.arcsin (u t ω))) by simpa only [one_mul] using h).congr'
    exact Eventually.of_forall (fun k => (hval k t ht ω).symm)
  have hDlim (t : ℝ) (ht : 0 ≤ t) (ω : Ω) : Tendsto (fun k => D k t ω) atTop (𝓝 0) :=
    (tendsto_regularizedLogisticDrift _ tendsto_logisticArcsinScale _ (hrange t ht ω)).congr'
      (Eventually.of_forall (fun k => (hD k t ht ω).symm))
  have hc (ω : Ω) : ContinuousOn (fun t => u t ω) (Icc (0 : ℝ) T) :=
    (hu.cont ω).mono (fun t ht => ht.1)
  have hM := isMartingaleOn_of_bounded_compensator_limits P ℱ T (Real.pi / 2) 1 hT zero_le_one
    F D (fun t ω => Real.arcsin (u t ω)) (fun _ _ => 0)
    (fun k => hu.mart _ (hφ k) T hT)
    (fun k t ht ω => by rw [Real.norm_eq_abs]; exact hFb k t ht.1 ω)
    (fun k t ht ω => by rw [Real.norm_eq_abs]; exact hDb k t ht.1 ω)
    (fun k ω => (continuous_logisticGenerator _ (hφ k)).comp_continuousOn (hc ω))
    (fun t ht ω => hFlim t ht.1 ω) (fun t ht ω => hDlim t ht.1 ω)
  have hInc : IsMartingaleOn T ℱ P (fun t ω => Real.arcsin (u t ω) - Real.arcsin (u 0 ω)) := by
    simpa only [intervalIntegral.integral_zero, sub_zero] using hM
  have hId (k : ℕ) (t : ℝ) (ht : 0 ≤ t) (ω : Ω) : D₂ k t ω = 2 * F k t ω * D k t ω + Q k t ω := by
    have h := logisticGenerator_square_arcsin_local _ _ (logisticArcsinScale_mem k)
      (hrange t ht ω) (φ k) (hφ k) (heq k _ (hrange t ht ω))
    rw [← hD k t ht ω] at h
    exact h
  have hQb (k : ℕ) (t : ℝ) (ht : 0 ≤ t) (ω : Ω) : Q k t ω ∈ Icc (0 : ℝ) 2 := by
    have h := regularizedLogisticVariance_mem_Icc _ _ (logisticArcsinScale_mem k) (hrange t ht ω)
    exact ⟨h.1, h.2.trans (by nlinarith [sq_nonneg (u t ω)])⟩
  have hD₂b (k : ℕ) (t : ℝ) (ht : 0 ≤ t) (ω : Ω) : |D₂ k t ω| ≤ Real.pi + 2 := by
    have hp : |2 * F k t ω * D k t ω| ≤ Real.pi := by
      rw [abs_mul, abs_mul, abs_of_pos (by norm_num : (0 : ℝ) < 2)]
      calc
        _ ≤ (2 * (Real.pi / 2)) * 1 :=
          mul_le_mul (mul_le_mul_of_nonneg_left (hFb k t ht ω) (by norm_num))
            (hDb k t ht ω) (abs_nonneg _) (by positivity)
        _ = Real.pi := by ring
    rw [hId k t ht ω]
    exact (abs_add_le _ _).trans (add_le_add hp (by rw [abs_of_nonneg (hQb k t ht ω).1]; exact (hQb k t ht ω).2))
  have hD₂lim (t : ℝ) (ht : 0 ≤ t) (ω : Ω) :
      Tendsto (fun k => D₂ k t ω) atTop (𝓝 (2 * (1 - u t ω ^ 2))) := by
    have h := (((hFlim t ht ω).mul (hDlim t ht ω)).const_mul 2).add
      (tendsto_regularizedLogisticVariance _ tendsto_logisticArcsinScale (u t ω))
    have hl : Tendsto (fun k => 2 * (F k t ω * D k t ω) + Q k t ω)
        atTop (𝓝 (2 * (1 - u t ω ^ 2))) := by simpa only [mul_zero, zero_add] using h
    apply hl.congr'
    exact Eventually.of_forall (fun k => by
      change 2 * (F k t ω * D k t ω) + Q k t ω = D₂ k t ω
      rw [hId k t ht ω]
      ring)
  have hSq := isMartingaleOn_of_bounded_compensator_limits P ℱ T ((Real.pi / 2) ^ 2) (Real.pi + 2)
    hT (by positivity) (fun k t ω => F k t ω ^ 2) D₂
    (fun t ω => Real.arcsin (u t ω) ^ 2) (fun t ω => 2 * (1 - u t ω ^ 2))
    (fun k => hu.mart _ ((hφ k).pow 2) T hT)
    (fun k t ht ω => by
      rw [Real.norm_eq_abs, abs_pow]
      exact pow_le_pow_left₀ (abs_nonneg _) (hFb k t ht.1 ω) 2)
    (fun k t ht ω => by rw [Real.norm_eq_abs]; exact hD₂b k t ht.1 ω)
    (fun k ω => (continuous_logisticGenerator _ ((hφ k).pow 2)).comp_continuousOn (hc ω))
    (fun t ht ω => (hFlim t ht.1 ω).pow 2) (fun t ht ω => hD₂lim t ht.1 ω)
  have hθ₀ : StronglyMeasurable[ℱ 0] (fun ω => Real.arcsin (u 0 ω)) :=
    Real.continuous_arcsin.comp_stronglyMeasurable (hu.adapted 0 le_rfl)
  have hbθ₀ (ω : Ω) : ‖Real.arcsin (u 0 ω)‖ ≤ Real.pi / 2 := by
    rw [Real.norm_eq_abs]
    exact abs_le.mpr (Real.arcsin_mem_Icc _)
  have hInit := isMartingaleOn_timeConstant P ℱ T _ hθ₀
    (Integrable.of_bound (hθ₀.mono (ℱ.le 0)).aestronglyMeasurable _ (ae_of_all P hbθ₀))
  refine ⟨?_, isMartingaleOn_centered_square _ _ hInc hSq hθ₀ _ hbθ₀⟩
  have hSum := hInc.add hInit
  convert hSum using 1
  funext t ω
  ring

/-- The actual absorbing solution u=1 realizes every premise, with a genuine
probability law, positive horizon, and nonzero angle π/2 at its singular endpoint. -/
example : ∃ (ℱ : Filtration ℝ (inferInstance : MeasurableSpace Unit)) (u : ℝ → Unit → ℝ),
    IsFiniteMeasure (Measure.dirac ()) ∧ IsProbabilityMeasure (Measure.dirac ()) ∧
    IsLogisticSolution (Measure.dirac ()) ℱ u ∧
    (∀ t, 0 ≤ t → ∀ ω, u t ω ∈ Icc (-1 : ℝ) 1) ∧
    (0 : ℝ) ≤ 1 ∧ u 0 () = 1 ∧ 0 < Real.arcsin (u 0 ()) := by
  let ℱ := Filtration.const (ι := ℝ) (m := (inferInstance : MeasurableSpace Unit)) ⊥ bot_le
  let u : ℝ → Unit → ℝ := fun _ _ => 1
  refine ⟨ℱ, u, inferInstance, inferInstance, isLogisticSolution_one (Measure.dirac ()) ℱ,
    fun t ht ω => by norm_num [u], zero_le_one, rfl, ?_⟩
  change 0 < Real.arcsin 1
  rw [Real.arcsin_one]
  positivity

end Transformer.Homogenized
