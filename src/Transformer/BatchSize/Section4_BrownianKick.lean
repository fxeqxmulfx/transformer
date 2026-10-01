/-
# Exact energy of an adapted diagonal Brownian update

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
This identity is the second-moment step in comparing Euler approximations.
The existing state and the noise coefficient may share their entire past.
-/

import Transformer.BatchSize.Section4_EulerChain

open MeasureTheory
open scoped NNReal BigOperators

noncomputable section

namespace Transformer.BatchSize

/-- A diagonal Brownian kick with random left-adapted coefficients,
Section 4.3 (2)--(3). -/
def brownianKick {d : ℕ} (H : BrownianSample d → EucSpace d)
    (s t : ℝ≥0) (ω : BrownianSample d) : EucSpace d :=
  WithLp.toLp 2 (fun k => H ω k * brownianIncrement k s t ω)

/-- A genuinely square-integrable diagonal Brownian update,
Section 4.3 (2)--(3). -/
theorem brownianKick_memLp {d : ℕ} (H : BrownianSample d → EucSpace d)
    (s t : ℝ≥0) (hst : s ≤ t)
    (hH : StronglyMeasurable[brownianFiltration d s] H)
    (hHL2 : MemLp H 2 (brownianNoiseLaw d)) :
    MemLp (brownianKick H s t) 2 (brownianNoiseLaw d) := by
  apply MemLp.of_eval_piLp
  intro k
  exact brownianIncrement_adapted_memLp k s t hst
    ((PiLp.continuous_apply 2 (fun _ : Fin d => ℝ) k).comp_stronglyMeasurable hH)
    (hHL2.eval_piLp k)

/-- Exact energy of a scalar adapted state plus its next random
Brownian kick, Section 4.3 (2)--(3). The mixed term vanishes because the
next increment is independent of the joint past. -/
theorem brownianIncrement_adapted_energy {d : ℕ} (k : Fin d)
    (Z H : BrownianSample d → ℝ) (s t : ℝ≥0) (hst : s ≤ t)
    (hZ : StronglyMeasurable[brownianFiltration d s] Z)
    (hH : StronglyMeasurable[brownianFiltration d s] H)
    (hZL2 : MemLp Z 2 (brownianNoiseLaw d)) (hHL2 : MemLp H 2 (brownianNoiseLaw d)) :
    (∫ ω, (Z ω + H ω * brownianIncrement k s t ω) ^ 2 ∂brownianNoiseLaw d) =
      (∫ ω, Z ω ^ 2 ∂brownianNoiseLaw d) +
        (∫ ω, H ω ^ 2 ∂brownianNoiseLaw d) * ((t : ℝ) - s) := by
  have hcross := brownianIncrement_adapted_mean k s t hst (hZ.mul hH)
    (hZL2.integrable_mul hHL2)
  have hsq := brownianIncrement_adapted_secondMoment k s t hst
    ((continuous_pow 2).comp_stronglyMeasurable hH) hHL2.integrable_sq
  have heq (ω : BrownianSample d) : (Z ω + H ω * brownianIncrement k s t ω) ^ 2 =
      Z ω ^ 2 + (2 * ((Z ω * H ω) * brownianIncrement k s t ω) +
        H ω ^ 2 * brownianIncrement k s t ω ^ 2) := by ring
  simp_rw [heq]
  rw [integral_add (f := fun ω => Z ω ^ 2)
    (g := fun ω => 2 * ((Z ω * H ω) * brownianIncrement k s t ω) +
      H ω ^ 2 * brownianIncrement k s t ω ^ 2)
    hZL2.integrable_sq ((hcross.1.const_mul 2).add hsq.1),
    integral_add (f := fun ω => 2 * ((Z ω * H ω) * brownianIncrement k s t ω))
      (g := fun ω => H ω ^ 2 * brownianIncrement k s t ω ^ 2)
      (hcross.1.const_mul 2) hsq.1, integral_const_mul]
  have hz : (∫ ω, (Z ω * H ω) * brownianIncrement k s t ω ∂brownianNoiseLaw d) = 0 := hcross.2
  rw [hz, hsq.2]
  ring

/-- Exact vector update energy, Section 4.3 (2)--(3). Random states
and amplitudes may depend on all coordinates; no independence between
the state and its amplitude is imposed. -/
theorem brownianKick_energy {d : ℕ} (Z H : BrownianSample d → EucSpace d)
    (s t : ℝ≥0) (hst : s ≤ t)
    (hZ : StronglyMeasurable[brownianFiltration d s] Z)
    (hH : StronglyMeasurable[brownianFiltration d s] H)
    (hZL2 : MemLp Z 2 (brownianNoiseLaw d)) (hHL2 : MemLp H 2 (brownianNoiseLaw d)) :
    (∫ ω, ‖Z ω + brownianKick H s t ω‖ ^ 2 ∂brownianNoiseLaw d) =
      (∫ ω, ‖Z ω‖ ^ 2 ∂brownianNoiseLaw d) +
        (∫ ω, ‖H ω‖ ^ 2 ∂brownianNoiseLaw d) * ((t : ℝ) - s) := by
  have hN := brownianKick_memLp H s t hst hH hHL2
  have hZk k := (PiLp.continuous_apply 2 (fun _ : Fin d => ℝ) k).comp_stronglyMeasurable hZ
  have hHk k := (PiLp.continuous_apply 2 (fun _ : Fin d => ℝ) k).comp_stronglyMeasurable hH
  simp only [EuclideanSpace.real_norm_sq_eq]
  rw [integral_finsetSum (f := fun k ω => (Z ω + brownianKick H s t ω) k ^ 2)
    _ (fun k _ => ((hZL2.add hN).eval_piLp k).integrable_sq)]
  have heq k : (∫ ω, (Z ω + brownianKick H s t ω) k ^ 2 ∂brownianNoiseLaw d) =
      (∫ ω, Z ω k ^ 2 ∂brownianNoiseLaw d) +
        (∫ ω, H ω k ^ 2 ∂brownianNoiseLaw d) * ((t : ℝ) - s) :=
    brownianIncrement_adapted_energy k (fun ω => Z ω k) (fun ω => H ω k)
      s t hst (hZk k) (hHk k) (hZL2.eval_piLp k) (hHL2.eval_piLp k)
  simp_rw [heq]
  rw [Finset.sum_add_distrib, ← Finset.sum_mul,
    integral_finsetSum _ (fun k _ => (hZL2.eval_piLp k).integrable_sq),
    integral_finsetSum _ (fun k _ => (hHL2.eval_piLp k).integrable_sq)]

/-- Joint nonvacuity of scalar energy hypotheses, Section 4.3:
both the state and the coefficient may be the same past Brownian value. -/
example : (1 : ℝ≥0) ≤ 2 ∧
    StronglyMeasurable[brownianFiltration 1 1] (coordinateBrownian (0 : Fin 1) 1) ∧
    MemLp (coordinateBrownian (0 : Fin 1) 1) 2 (brownianNoiseLaw 1) :=
  ⟨by norm_num, (coordinateBrownian_filtered (0 : Fin 1)).stronglyAdapted 1,
    ((coordinateBrownian_isBrownian (0 : Fin 1)).isGaussianProcess.hasGaussianLaw_eval 1).memLp_two⟩

/-- Joint nonvacuity of vector energy hypotheses, Section 4.3:
both the state and the coefficient may be the full past vector driver. -/
example : (1 : ℝ≥0) ≤ 2 ∧
    StronglyMeasurable[brownianFiltration 2 1] (vectorBrownian 2 1) ∧
    MemLp (vectorBrownian 2 1) 2 (brownianNoiseLaw 2) := by
  refine ⟨by norm_num, Filtration.stronglyAdapted_natural
    (fun t => (vectorBrownian_measurable 2 t).stronglyMeasurable) 1, ?_⟩
  apply MemLp.of_eval_piLp
  intro k
  exact ((coordinateBrownian_isBrownian k).isGaussianProcess.hasGaussianLaw_eval 1).memLp_two

end Transformer.BatchSize
