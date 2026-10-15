import Transformer.Homogenized.Section7_LogisticNegativePart
import Transformer.Homogenized.Section7_WeightedSupermartingale
import Mathlib.MeasureTheory.Integral.DominatedConvergence

/-!
# Invariance of the logistic state interval

ArXiv:2604.01978v1, §7, `sec: sde.logistic`, the state interval
underlying the source's bounded angle argument.
The globally smooth boundary tests have their actual compensated
martingales by the original equation. Their computed global drift
bound gives E Yε(t)≤exp(28t) E Yε(0), with E Yε(0)≤ε P(Ω).
Bounded dominated convergence as ε→0 forces the negative part of
the bounded boundary coordinate to have zero expectation.
Its sign then gives precisely the original interval [-1,1].

The finite-measure hypothesis restores the source's probability-law
setting (and slightly generalizes it). The earlier declaration omitted
measure finiteness, which is needed here for the bounded domination.
Neither a range property nor a martingale property of a nonsmooth
test or a weighted process is assumed. The solution is tested only
with the actual globally smooth functions of the preceding module.
-/

open Real MeasureTheory Set Filter
open scoped Topology
namespace Transformer.Homogenized

/-- The original smooth-test martingale gives the actual boundary-test expectation
bound. Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, interval invariance.
The integrating factor and conditional order are derived, not extra premises. -/
theorem integral_logisticBoundaryTest_le
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsFiniteMeasure P]
    (ℱ : Filtration ℝ m) (u : ℝ → Ω → ℝ) (hu : IsLogisticSolution P ℱ u)
    (hinit : ∀ ω, u 0 ω ∈ Icc (-1 : ℝ) 1) (ε : ℝ) (hε : ε ∈ Ioc (0 : ℝ) 1)
    (T : ℝ) (hT : 0 ≤ T) :
    Real.exp (-28 * T) * (∫ ω, logisticBoundaryTest ε (u T ω) ∂P) ≤
      ε * P.real univ := by
  let C : ℝ → Ω → ℝ := fun t _ => Real.exp (-28 * t)
  let V : ℝ → Ω → ℝ := fun t _ => -28 * Real.exp (-28 * t)
  let Y : ℝ → Ω → ℝ := fun t ω => logisticBoundaryTest ε (u t ω)
  let H : ℝ → Ω → ℝ := fun t ω => logisticGenerator (logisticBoundaryTest ε) (u t ω)
  have hφ := contDiff_logisticBoundaryTest ε hε.1
  have hY (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) : StronglyMeasurable[ℱ t] (Y t) :=
    hφ.continuous.comp_stronglyMeasurable (hu.adapted t ht.1)
  have hCb (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) (ω : Ω) : |C t ω| ≤ 1 := by
    rw [abs_of_pos (Real.exp_pos _)]
    exact Real.exp_le_one_iff.mpr (by nlinarith [ht.1])
  have hYb (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) (ω : Ω) : |Y t ω| ≤ 3 := by
    rw [abs_of_nonneg (logisticBoundaryTest_bounds ε (u t ω) hε).1]
    exact (logisticBoundaryTest_bounds ε (u t ω) hε).2.1
  have hHb (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) (ω : Ω) : |H t ω| ≤ 84 := by
    have hb := logisticBoundaryTest_bounds ε (u t ω) hε
    change |logisticGenerator (logisticBoundaryTest ε) (u t ω)| ≤ 84
    linarith [hb.2.1, hb.2.2.1]
  have hc (ω : Ω) := (hu.cont ω).mono (show Icc (0 : ℝ) T ⊆ Ici 0 from fun t ht => ht.1)
  have hce := condExp_mul_le_of_compensated_martingale P ℱ T C Y H V
    (fun t ht => stronglyMeasurable_const) hY
    (fun ω => ⟨by dsimp [C]; fun_prop,
      hφ.continuous.comp_continuousOn' (hc ω),
      (continuous_logisticGenerator _ hφ).comp_continuousOn' (hc ω),
      by dsimp [V]; fun_prop⟩)
    (fun ω x hx => by
      have hd := ((hasDerivAt_id x).const_mul (-28)).exp
      apply hd.congr_deriv
      dsimp [V, C]
      ring)
    (hu.mart _ hφ T hT) 1 3 84 28 (by norm_num) (by norm_num) hCb hYb hHb
    (fun t ht ω => by
      change |(-28 : ℝ) * Real.exp (-28 * t)| ≤ 28
      rw [abs_mul, abs_of_pos (Real.exp_pos _), show |(-28 : ℝ)| = 28 by norm_num]
      have hb := hCb t ht ω
      rw [abs_of_pos (Real.exp_pos _)] at hb
      linarith)
    (fun t ht ω => by
      have hb := (le_abs_self (H t ω)).trans (logisticBoundaryTest_bounds ε (u t ω) hε).2.2.1
      have hm := mul_le_mul_of_nonneg_left hb (Real.exp_nonneg (-28 * t))
      dsimp [C, V]
      nlinarith)
    0 T ⟨le_rfl, hT⟩ ⟨hT, le_rfl⟩ hT
  have hi0 : Integrable (Y 0) P := Integrable.of_bound
    ((hY 0 ⟨le_rfl, hT⟩).mono (ℱ.le 0)).aestronglyMeasurable 3
    (ae_of_all P (fun ω => by simpa only [Real.norm_eq_abs] using hYb 0 ⟨le_rfl, hT⟩ ω))
  have h0 : (fun ω => C 0 ω * Y 0 ω) = Y 0 := by funext ω; simp [C]
  have hi := integral_mono_ae integrable_condExp (by simpa only [h0] using hi0) hce
  rw [integral_condExp (ℱ.le 0), h0] at hi
  change (∫ ω, Real.exp (-28 * T) * Y T ω ∂P) ≤ ∫ ω, Y 0 ω ∂P at hi
  rw [integral_const_mul] at hi
  have hinit_le : (∫ ω, Y 0 ω ∂P) ≤ ε * P.real univ := by
    have hh := integral_mono_ae hi0 (integrable_const ε)
      (ae_of_all P (fun ω => (logisticBoundaryTest_bounds ε (u 0 ω) hε).2.2.2 (hinit ω)))
    simpa only [integral_const, smul_eq_mul, mul_comm] using hh
  exact hi.trans hinit_le

/-- **[-1,1] is invariant.** The source uses this as the bounded angle range
in `sec: sde.logistic`. This proves it from the full original smooth-test
martingale problem. The source works under a probability law; the missing
measure hypothesis in the earlier declaration is restored as finite measure.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`. -/
theorem logistic_range {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω)
    [IsFiniteMeasure P] (ℱ : Filtration ℝ m) (u : ℝ → Ω → ℝ) (hu : IsLogisticSolution P ℱ u)
    (hinit : ∀ ω, u 0 ω ∈ Icc (-1 : ℝ) 1) :
    ∀ t, 0 ≤ t → ∀ᵐ ω ∂P, u t ω ∈ Icc (-1 : ℝ) 1 := by
  intro T hT
  let ε : ℕ → ℝ := fun k => 1 / ((k : ℝ) + 1)
  have he (k : ℕ) : ε k ∈ Ioc (0 : ℝ) 1 := by
    refine ⟨by dsimp [ε]; positivity, ?_⟩
    dsimp [ε]
    exact (div_le_one (by positivity : (0 : ℝ) < (k : ℝ) + 1)).mpr
      (by linarith [show (0 : ℝ) ≤ (k : ℝ) from Nat.cast_nonneg k])
  have hlim : Tendsto ε atTop (𝓝 0) := tendsto_one_div_add_atTop_nhds_zero_nat
  let w : Ω → ℝ := fun ω => logisticBoundaryCoordinate (u T ω)
  let W : Ω → ℝ := fun ω => |w ω| - w ω
  have hwm : StronglyMeasurable w :=
    contDiff_logisticBoundaryCoordinate.continuous.comp_stronglyMeasurable
      ((hu.adapted T hT).mono (ℱ.le T))
  have hWm : StronglyMeasurable W := (continuous_abs.comp_stronglyMeasurable hwm).sub hwm
  have hW0 (ω : Ω) : 0 ≤ W ω := sub_nonneg.mpr (le_abs_self (w ω))
  have hWb (ω : Ω) : ‖W ω‖ ≤ 2 := by
    rw [Real.norm_eq_abs, abs_of_nonneg (hW0 ω)]
    have hb := (logisticBoundaryCoordinate_bounds (u T ω)).1
    change |w ω| ≤ 1 at hb
    dsimp [W]
    linarith [neg_le_abs (w ω)]
  have hiW : Integrable W P := Integrable.of_bound hWm.aestronglyMeasurable 2 (ae_of_all P hWb)
  have hD := tendsto_integral_of_dominated_convergence (μ := P)
    (F := fun k ω => logisticBoundaryTest (ε k) (u T ω)) (f := W) (fun _ => 3)
    (fun k => ((contDiff_logisticBoundaryTest (ε k) (he k).1).continuous.comp_stronglyMeasurable
      ((hu.adapted T hT).mono (ℱ.le T))).aestronglyMeasurable)
    (integrable_const 3)
    (fun k => ae_of_all P (fun ω => by
      rw [Real.norm_eq_abs, abs_of_nonneg (logisticBoundaryTest_bounds (ε k) (u T ω) (he k)).1]
      exact (logisticBoundaryTest_bounds (ε k) (u T ω) (he k)).2.1))
    (ae_of_all P (fun ω => by
      have hl : Tendsto (fun k => w ω ^ 2 + ε k ^ 2) atTop (𝓝 (w ω ^ 2 + 0 ^ 2)) :=
        tendsto_const_nhds.add (hlim.pow 2)
      simpa only [logisticBoundaryTest, smoothNegativePart, W, w, zero_pow (by decide : 2 ≠ 0),
        add_zero, Real.sqrt_sq_eq_abs] using hl.sqrt.sub tendsto_const_nhds))
  have hle := le_of_tendsto_of_tendsto (hD.const_mul (Real.exp (-28 * T)))
    (hlim.mul_const (P.real univ))
    (Eventually.of_forall (fun k => integral_logisticBoundaryTest_le P ℱ u hu hinit (ε k) (he k) T hT))
  simp only [zero_mul] at hle
  have hE : (∫ ω, W ω ∂P) = 0 := by
    have hnonneg := integral_nonneg_of_ae (ae_of_all P hW0)
    nlinarith [Real.exp_pos (-28 * T)]
  have hzero := (integral_eq_zero_iff_of_nonneg_ae (ae_of_all P hW0) hiW).mp hE
  filter_upwards [hzero] with ω hω
  apply (logisticBoundaryCoordinate_bounds (u T ω)).2.mp
  change |w ω| - w ω = 0 at hω
  change 0 ≤ w ω
  linarith [abs_nonneg (w ω)]

/-- The genuine absorbing solution realizes all equation, initial-state,
positive regularization, and nonzero horizon hypotheses simultaneously.
The regularized observable is nonzero, although its actual generator vanishes. -/
example : ∃ ℱ : Filtration ℝ (inferInstance : MeasurableSpace Unit),
    IsLogisticSolution (Measure.dirac ()) ℱ (fun _ _ => 1) ∧
      (∀ ω : Unit, (fun (_ : ℝ) (_ : Unit) => (1 : ℝ)) 0 ω ∈ Icc (-1 : ℝ) 1) ∧
        (1 : ℝ) ∈ Ioc (0 : ℝ) 1 ∧ (0 : ℝ) ≤ 1 ∧
          Real.exp (-28 * 1) * (∫ _ : Unit, logisticBoundaryTest 1 1 ∂Measure.dirac ()) ≤
            1 * (Measure.dirac ()).real univ := by
  let ℱ := Filtration.const (ι := ℝ) (m := (inferInstance : MeasurableSpace Unit)) ⊥ bot_le
  have hu := isLogisticSolution_one (Measure.dirac ()) ℱ
  have hinit (ω : Unit) : (1 : ℝ) ∈ Icc (-1 : ℝ) 1 := by norm_num
  refine ⟨ℱ, hu, hinit, by norm_num, zero_le_one, ?_⟩
  exact integral_logisticBoundaryTest_le (Measure.dirac ()) ℱ (fun _ _ => 1) hu hinit
    1 (by norm_num) 1 zero_le_one

end Transformer.Homogenized
