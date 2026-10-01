/-
# The actual continuous Euler limit satisfies the generator identity

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
Finite Brownian Euler identities pass to a genuine continuous modification
of their mean-square limits, with arbitrary bounded past weights.
-/

import Transformer.BatchSize.Section4_DyadicGeneratorIdentity
import Transformer.BatchSize.Section4_DyadicGeneratorLimit
import Transformer.BatchSize.Section4_BoundedIntervalLimits
import Transformer.BatchSize.Section4_WeightedTestLimits

open MeasureTheory Filter
open scoped NNReal Topology

noncomputable section

namespace Transformer.BatchSize

/-- The weighted integrated generator identity for a true continuous
modification of the actual Brownian Euler limits, Section 4.3 (2)--(3).
Every limit premise is a mean-square limit of the specified Euler
sampler; no martingale identity is included in the hypotheses. -/
theorem eulerLimit_weighted_generator_identity {d : ℕ}
    (b : EucSpace d → EucSpace d) (a : EucSpace d → Fin d → ℝ) (Kb Ka M A : NNReal)
    (hb : LipschitzWith Kb b) (ha : ∀ k, LipschitzWith Ka (fun x => a x k))
    (hbM : ∀ x, ‖b x‖ ≤ M) (haA : ∀ x k, |a x k| ≤ A)
    (x₀ : EucSpace d) (s T : NNReal) (hsT : s ≤ T)
    (Y : NNReal → BrownianSample d → EucSpace d)
    (hY : ∀ u, MemLp (Y u) 2 (brownianNoiseLaw d))
    (hlim : ∀ u, Tendsto (fun m => ∫ ω, ‖dyadicEuler b a x₀ u m ω - Y u ω‖ ^ 2
      ∂brownianNoiseLaw d) atTop (𝓝 0))
    (Z : BrownianSample d → DiffusionPath d)
    (hZY : ∀ u : NNReal, (fun ω => Z ω (u : ℝ)) =ᵐ[brownianNoiseLaw d] Y u)
    (F : BrownianSample d → ℝ) (hF : StronglyMeasurable[brownianFiltration d s] F)
    (hF1 : ∀ ω, |F ω| ≤ 1) (φ : EucSpace d → ℝ) (hφ : BoundedSmoothTest 2 φ) :
    (∫ ω, F ω * φ (Z ω T) ∂brownianNoiseLaw d) -
      (∫ ω, F ω * φ (Z ω s) ∂brownianNoiseLaw d) =
      ∫ u in (s : ℝ)..(T : ℝ), ∫ ω, F ω *
        frozenGaussianGenerator (b (Z ω u)) (a (Z ω u)) φ (Z ω u) ∂brownianNoiseLaw d := by
  let J (m : ℕ) := dyadicRightIndex s m
  let start (m : ℕ) : ℝ := dyadicGrid T m (J m)
  let G (m : ℕ) (u : ℝ) := ∫ ω, F ω * dyadicGeneratorValue b a x₀ T m φ u ω ∂brownianNoiseLaw d
  let g (u : ℝ) := ∫ ω, F ω * frozenGaussianGenerator (b (Z ω u)) (a (Z ω u)) φ (Z ω u)
    ∂brownianNoiseLaw d
  have hFm := (hF.mono ((brownianFiltration d).le s)).measurable
  have hG (m : ℕ) := dyadicGenerator_expectation_bounded b a hb.continuous
    (fun k => (ha k).continuous) M A hbM haA x₀ T m φ hφ F hFm hF1
  have hstart : Tendsto start atTop (𝓝 (s : ℝ)) := by
    have hsub : Tendsto (fun m => start m - s) atTop (𝓝 0) :=
      squeeze_zero (fun m => sub_nonneg.mpr (NNReal.coe_le_coe.mpr
        (dyadicRightTime_bounds s T hsT m).1))
        (fun m => (dyadicRightTime_bounds s T hsT m).2)
        (tendsto_pow_atTop_nhds_zero_of_lt_one (by norm_num) (by norm_num))
    simpa only [sub_add_cancel, zero_add] using hsub.add (tendsto_const_nhds (x := (s : ℝ)))
  have hGlim (u : ℝ) (hu : u ∈ Set.uIoc (s : ℝ) T) :
      Tendsto (fun m => G m u) atTop (𝓝 (g u)) := by
    rw [Set.uIoc_of_le (NNReal.coe_le_coe.mpr hsT)] at hu
    have hu0 : 0 ≤ u := s.coe_nonneg.trans hu.1.le
    have huT : u.toNNReal ≤ T := by
      apply NNReal.coe_le_coe.mp
      simpa only [Real.coe_toNNReal u hu0] using hu.2
    have h := dyadicGenerator_expectation_tendsto b a Kb Ka M A hb ha hbM haA x₀
      u.toNNReal T huT (Y u.toNNReal) (hY _) (hlim _)
      (fun ω => Z ω (u.toNNReal : ℝ)) (hZY _) F hFm hF1 φ hφ
    simpa only [Real.coe_toNNReal u hu0] using h
  have hR := (dyadicEuler_neighbors_meanSquare b a Kb Ka M A hb ha hbM haA x₀ s T hsT
    (Y s) (hY s) (hlim s)).2
  have hXT (m : ℕ) : MemLp (dyadicEuler b a x₀ T m) 2 (brownianNoiseLaw d) :=
    eulerChain_memLp b a Kb Ka hb ha _ (dyadicGrid_monotone T m) x₀ _
  have hEt := weighted_test_meanSquare_tendsto (brownianNoiseLaw d) _ hXT (Y T) (hY T)
    (hlim T) F hFm hF1 φ hφ
  have hEs := weighted_test_meanSquare_tendsto (brownianNoiseLaw d) _ hR.1 (Y s) (hY s)
    hR.2 F hFm hF1 φ hφ
  have hrep (u : NNReal) : (∫ ω, F ω * φ (Y u ω) ∂brownianNoiseLaw d) =
      ∫ ω, F ω * φ (Z ω u) ∂brownianNoiseLaw d := by
    apply integral_congr_ae
    filter_upwards [hZY u] with ω hω
    rw [hω]
  rw [hrep T] at hEt
  rw [hrep s] at hEs
  have hleft := hEt.sub hEs
  have hright := bounded_intervalIntegral_moving_start_tendsto G g
    (fun m => (hG m).1) ((M : ℝ) + (d : ℝ) * (A : ℝ) ^ 2 / 2)
    (fun m => (hG m).2) s T hGlim start hstart
  have heq (m : ℕ) : (∫ ω, F ω * φ (dyadicEuler b a x₀ T m ω) ∂brownianNoiseLaw d) -
      (∫ ω, F ω * φ (eulerChain b a (dyadicGrid T m) x₀ (J m) ω) ∂brownianNoiseLaw d) =
      ∫ u in start m..(T : ℝ), G m u := by
    have hFj := hF.mono ((brownianFiltration d).mono (dyadicRightTime_bounds s T hsT m).1)
    have h := dyadicEuler_weighted_generator_identity b a hb.continuous
      (fun k => (ha k).continuous) M A hbM haA x₀ T m (J m)
      (dyadicObservation_indices_le_endpoint s T hsT m).2 F hFj hF1 φ hφ
    have hI (X : BrownianSample d → EucSpace d) (hm : Measurable X) :
        Integrable (fun ω => F ω * φ (X ω)) (brownianNoiseLaw d) := by
      apply (integrable_const (1 : ℝ)).mono' (hFm.mul (hφ.1.continuous.measurable.comp hm)).aestronglyMeasurable
      exact Eventually.of_forall fun ω => by
        have hφ1 : |φ (X ω)| ≤ 1 := by
          simpa only [norm_iteratedFDeriv_zero, Real.norm_eq_abs] using hφ.2 0 (by norm_num) (X ω)
        simp only [Pi.mul_apply, Function.comp_def]
        rw [Real.norm_eq_abs, abs_mul]
        exact (mul_le_mul (hF1 ω) hφ1 (abs_nonneg _) (by norm_num)).trans_eq (one_mul _)
    simp_rw [mul_sub] at h
    rw [integral_sub
      (hI (dyadicEuler b a x₀ T m) (((eulerChain_adapted b a hb.continuous (fun k => (ha k).continuous) _
        (dyadicGrid_monotone T m) x₀ (dyadicEndpointIndex T m)).mono
        ((brownianFiltration d).le _)).measurable))
      (hI _ (((eulerChain_adapted b a hb.continuous (fun k => (ha k).continuous) _
        (dyadicGrid_monotone T m) x₀ (J m)).mono ((brownianFiltration d).le _)).measurable))] at h
    exact h
  have hright' := hright.congr' (Eventually.of_forall fun m => (heq m).symm)
  exact tendsto_nhds_unique hleft hright'

/-- Joint nonvacuity of all continuous Euler-limit generator
hypotheses, Section 4.3: bounded constant coefficients and a
nonzero constant continuous path, with actual constant Euler limits. -/
example : LipschitzWith 0 (fun _ : EucSpace 1 => (0 : EucSpace 1)) ∧
    (∀ _ : Fin 1, LipschitzWith 0 (fun _ : EucSpace 1 => (0 : ℝ))) ∧
    (∀ x : EucSpace 1, ‖(fun _ : EucSpace 1 => (0 : EucSpace 1)) x‖ ≤ (1 : NNReal)) ∧
    (∀ (x : EucSpace 1) (k : Fin 1), |(fun _ : EucSpace 1 => fun _ : Fin 1 => (0 : ℝ)) x k| ≤
      (1 : NNReal)) ∧ (0 : NNReal) ≤ 1 ∧
    ∃ (Y : NNReal → BrownianSample 1 → EucSpace 1) (Z : BrownianSample 1 → DiffusionPath 1),
      (∀ u, MemLp (Y u) 2 (brownianNoiseLaw 1)) ∧
      (∀ u, Tendsto (fun m => ∫ ω,
        ‖dyadicEuler (fun _ => (0 : EucSpace 1)) (fun _ _ => (0 : ℝ))
          (EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) u m ω - Y u ω‖ ^ 2 ∂brownianNoiseLaw 1)
        atTop (𝓝 0)) ∧
      (∀ u : NNReal, (fun ω => Z ω (u : ℝ)) =ᵐ[brownianNoiseLaw 1] Y u) ∧
      StronglyMeasurable[brownianFiltration 1 0] (fun _ : BrownianSample 1 => (1 : ℝ)) ∧
      |(1 : ℝ)| ≤ 1 ∧ BoundedSmoothTest 2 (fun _ : EucSpace 1 => (1 : ℝ)) := by
  let x₀ := EuclideanSpace.single (0 : Fin 1) (1 : ℝ)
  have hchain (t : ℕ → NNReal) (n : ℕ) (ω : BrownianSample 1) :
      eulerChain (fun _ => (0 : EucSpace 1)) (fun _ _ => (0 : ℝ)) t x₀ n ω = x₀ := by
    induction n with
    | zero => rfl
    | succ n ih =>
      have hz : WithLp.toLp 2 (fun _ : Fin 1 => (0 : ℝ)) = (0 : EucSpace 1) := rfl
      simp only [eulerChain, ih, smul_zero, zero_mul, hz, add_zero]
  refine ⟨LipschitzWith.const _, fun _ => LipschitzWith.const _, by simp, by simp,
    by norm_num, (fun _ _ => x₀), (fun _ => ContinuousMap.const ℝ x₀),
    (fun _ => memLp_const _), ?_, (fun _ => Eventually.of_forall fun _ => rfl),
    stronglyMeasurable_const, by norm_num, contDiff_const, ?_⟩
  · intro u
    change Tendsto (fun m => ∫ ω,
      ‖eulerChain (fun _ => (0 : EucSpace 1)) (fun _ _ => (0 : ℝ))
        (dyadicGrid u m) x₀ (dyadicEndpointIndex u m) ω - x₀‖ ^ 2 ∂brownianNoiseLaw 1)
      atTop (𝓝 (0 : ℝ))
    simp only [hchain, sub_self, norm_zero, zero_pow (by norm_num : (2 : ℕ) ≠ 0),
      integral_zero]
    exact tendsto_const_nhds
  · intro j hj y
    cases j with
    | zero => simp [norm_iteratedFDeriv_zero]
    | succ j => rw [iteratedFDeriv_const_of_ne (by omega)]; simp

end Transformer.BatchSize
