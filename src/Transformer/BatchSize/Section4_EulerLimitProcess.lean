/-
# The actual optimizer process obtained from Euler L2 limits

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
Each state is chosen from a proved existence theorem and is measurable
in the joint Brownian past. Continuous paths and the generator identity
are not assumed by the construction.
-/

import Transformer.BatchSize.Section4_EulerL2Limit

open MeasureTheory Filter
open scoped NNReal Topology

noncomputable section

namespace Transformer.BatchSize

/-- The optimizer's actual Euler-limit state at each nonnegative
time, Section 4.3 (2)--(3). The witness comes from the proved adapted
L2 convergence theorem, rather than an SDE existence hypothesis. -/
def optimizerEulerLimit {d : ℕ} (method : UpdateKind) (η : ℝ) (B : ℕ)
    (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hη : 0 ≤ η) (hB : 0 < B) (hmodel : RegularGaussianModel f σ)
    (x₀ : EucSpace d) (t : ℝ≥0) : BrownianSample d → EucSpace d :=
  Classical.choose (optimizerEuler_exists_adapted_limit method η B f σ hη hB hmodel x₀ t)

/-- Every optimizer Euler-limit state is genuinely measurable in
the whole vector driver's past, Section 4.3 (2)--(3). -/
theorem optimizerEulerLimit_adapted {d : ℕ} (method : UpdateKind) (η : ℝ) (B : ℕ)
    (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hη : 0 ≤ η) (hB : 0 < B) (hmodel : RegularGaussianModel f σ)
    (x₀ : EucSpace d) (t : ℝ≥0) :
    StronglyMeasurable[brownianFiltration d t]
      (optimizerEulerLimit method η B f σ hη hB hmodel x₀ t) :=
  (Classical.choose_spec (optimizerEuler_exists_adapted_limit method η B f σ hη hB hmodel x₀ t)).1

/-- Every constructed optimizer Euler-limit state has finite second
moment, Section 4.3 (2)--(3). -/
theorem optimizerEulerLimit_memLp {d : ℕ} (method : UpdateKind) (η : ℝ) (B : ℕ)
    (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hη : 0 ≤ η) (hB : 0 < B) (hmodel : RegularGaussianModel f σ)
    (x₀ : EucSpace d) (t : ℝ≥0) :
    MemLp (optimizerEulerLimit method η B f σ hη hB hmodel x₀ t) 2 (brownianNoiseLaw d) :=
  (Classical.choose_spec (optimizerEuler_exists_adapted_limit method η B f σ hη hB hmodel x₀ t)).2.1

/-- The actual optimizer dyadic states converge in mean square to
the constructed optimizer process, Section 4.3 (2)--(3). -/
theorem optimizerEulerLimit_meanSquare {d : ℕ} (method : UpdateKind) (η : ℝ) (B : ℕ)
    (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hη : 0 ≤ η) (hB : 0 < B) (hmodel : RegularGaussianModel f σ)
    (x₀ : EucSpace d) (t : ℝ≥0) :
    Tendsto (fun m => ∫ ω,
      ‖dyadicEuler (diffusionDrift method B f σ) (diffusionNoiseScale method η B f σ) x₀ t m ω -
        optimizerEulerLimit method η B f σ hη hB hmodel x₀ t ω‖ ^ 2 ∂brownianNoiseLaw d) atTop (𝓝 0) :=
  (Classical.choose_spec (optimizerEuler_exists_adapted_limit method η B f σ hη hB hmodel x₀ t)).2.2

/-- The constructed optimizer Euler-limit process starts at the
prescribed state almost surely, Section 4.3 (2)--(3). -/
theorem optimizerEulerLimit_zero {d : ℕ} (method : UpdateKind) (η : ℝ) (B : ℕ)
    (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hη : 0 ≤ η) (hB : 0 < B) (hmodel : RegularGaussianModel f σ) (x₀ : EucSpace d) :
    ∀ᵐ ω ∂brownianNoiseLaw d, optimizerEulerLimit method η B f σ hη hB hmodel x₀ 0 ω = x₀ := by
  let X (m : ℕ) := dyadicEuler (diffusionDrift method B f σ) (diffusionNoiseScale method η B f σ) x₀ 0 m
  let Y := optimizerEulerLimit method η B f σ hη hB hmodel x₀ 0
  have hXeq (m : ℕ) : X m = fun _ : BrownianSample d => x₀ := by
    funext ω
    simp [X, dyadicEuler, dyadicEndpointIndex, eulerChain]
  have hXL2 (m : ℕ) : MemLp (X m) 2 (brownianNoiseLaw d) := by
    rw [hXeq]
    exact memLp_const x₀
  have hYL2 := optimizerEulerLimit_memLp method η B f σ hη hB hmodel x₀ 0
  have hlim := meanSquare_tendsto_toLp (brownianNoiseLaw d) X hXL2 Y hYL2
    (optimizerEulerLimit_meanSquare method η B f σ hη hB hmodel x₀ 0)
  have hconst : MemLp (fun _ : BrownianSample d => x₀) 2 (brownianNoiseLaw d) := memLp_const x₀
  have heq (m : ℕ) : (hXL2 m).toLp (X m) = hconst.toLp (fun _ => x₀) :=
    (hXL2 m).toLp_congr hconst (Eventually.of_forall fun ω => congrFun (hXeq m) ω)
  simp_rw [heq] at hlim
  have hLp : hYL2.toLp Y = hconst.toLp (fun _ => x₀) :=
    tendsto_nhds_unique hlim tendsto_const_nhds
  exact (hYL2.toLp_eq_toLp_iff hconst).mp hLp

/-- Joint nonvacuity of constructed-process hypotheses, Section 4.3:
positive batch and rate, flat loss, unit Gaussian coordinate noise. -/
example : (0 : ℝ) ≤ 1 / 1000 ∧ 0 < (1 : ℕ) ∧
    RegularGaussianModel (fun _ : EucSpace 2 => (0 : ℝ))
      (fun _ => WithLp.toLp 2 (fun _ : Fin 2 => (1 : ℝ))) :=
  ⟨by norm_num, by norm_num, regularGaussianModel_flat 2⟩

end Transformer.BatchSize
