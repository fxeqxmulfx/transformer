/-
# Actual homogeneous power tests and their even moment expansion

ArXiv:2601.21366v2, §5.1, `eq:FH-basic` and `lem: quadpol` (i).
The original exponential kernel tested against a power of an actual
linear coordinate has a finite binomial expansion in the original
radial and perpendicular moments.

Every original test vector is its actual radial component plus its
actual perpendicular projection. Normalizing the latter gives an actual
unit perpendicular vector. If the projection is zero, an actual unit
perpendicular vector still exists in dimension at least two, and its
coefficient is the actual zero norm. Thus no test vector is discarded.

The original finite-measure integrability proves the termwise integral
expansion. Uniform reflection cancels every odd perpendicular term.
For an even term its original norm power is a power of the actual squared
norm, which equals the squared test norm minus its squared radial
coordinate. The next polynomial-degree argument uses exactly these
original terms. No moment, inverse or polynomial conclusion is assumed.
-/

import Transformer.Perceptron.Section5_PolynomialMoments

open Real MeasureTheory Set
open scoped BigOperators

namespace Transformer.Perceptron

/-- The actual original exponential kernel tested against a homogeneous
power of the original linear coordinate. Source: arXiv:2601.21366v2,
§5.1, `eq:FH-basic`, the finite polynomial tests of `lem: quadpol` (i). -/
noncomputable def powerMomentKernel {d : ℕ} (β : ℝ) (ν : Measure (SSphere d))
    (x b : EucSpace d) (n : ℕ) : ℝ :=
  ∫ y, exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)) * inner (𝕜 := ℝ) b (y : EucSpace d) ^ n ∂ν

/-- The actual original homogeneous power integrand is continuous.
Source: arXiv:2601.21366v2, §5.1, the polynomial tests of `lem: quadpol` (i).
Every actual finite original law therefore integrates it. -/
theorem continuous_powerMomentIntegrand {d : ℕ} (β : ℝ) (x b : EucSpace d) (n : ℕ) :
    Continuous (fun y : SSphere d => exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)) *
      inner (𝕜 := ℝ) b (y : EucSpace d) ^ n) :=
  (Real.continuous_exp.comp (continuous_const.mul
    (continuous_const.inner continuous_subtype_val))).mul
      ((continuous_const.inner continuous_subtype_val).pow n)

/-- Scaling the original test vector scales its original homogeneous
moment by the same actual scalar power. Source: arXiv:2601.21366v2,
§5.1, `eq:FH-basic`, homogeneous polynomial tests. -/
theorem powerMomentKernel_smul_test {d : ℕ} (β : ℝ) (ν : Measure (SSphere d))
    (x b : EucSpace d) (n : ℕ) (s : ℝ) :
    powerMomentKernel β ν x (s • b) n = s ^ n * powerMomentKernel β ν x b n := by
  unfold powerMomentKernel
  rw [← integral_const_mul]
  apply integral_congr_ae
  exact ae_of_all ν fun y => by dsimp; rw [real_inner_smul_left, mul_pow]; ring

/-- Every actual original test vector admits its actual radial and
normalized perpendicular decomposition at the original sphere point.
Source: arXiv:2601.21366v2, §5.1, `eq:FH-basic`, homogeneous polynomial tests.
At a zero perpendicular projection its coefficient is zero; the actual
unit witness is supplied by the proved original sphere geometry. -/
theorem exists_unit_perp_decomposition {d : ℕ} (hd : 2 ≤ d) (x : SSphere d) (b : EucSpace d) :
    ∃ u : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (u : EucSpace d) = 0 ∧
      b = inner (𝕜 := ℝ) (x : EucSpace d) b • (x : EucSpace d) +
        ‖proj d (x : EucSpace d) b‖ • (u : EucSpace d) := by
  have hb : b = inner (𝕜 := ℝ) (x : EucSpace d) b • (x : EucSpace d) + proj d (x : EucSpace d) b := by
    unfold proj
    abel
  by_cases hp : proj d (x : EucSpace d) b = 0
  · obtain ⟨u, hu⟩ := exists_unit_perp_sphere hd x
    exact ⟨u, hu, by simpa only [hp, norm_zero, zero_smul, add_zero] using hb⟩
  · have hn : ‖proj d (x : EucSpace d) b‖ ≠ 0 := norm_ne_zero_iff.mpr hp
    let u : SSphere d := ⟨‖proj d (x : EucSpace d) b‖⁻¹ • proj d (x : EucSpace d) b, by
      apply mem_sphere_zero_iff_norm.mpr
      rw [norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ hn]⟩
    refine ⟨u, ?_, ?_⟩
    · change inner (𝕜 := ℝ) (x : EucSpace d) (_ • proj d (x : EucSpace d) b) = 0
      rw [real_inner_smul_right, inner_proj_eq_zero (norm_eq_of_mem_sphere x), mul_zero]
    · change b = _ + ‖proj d (x : EucSpace d) b‖ • (_ • proj d (x : EucSpace d) b)
      rwa [smul_smul, mul_inv_cancel₀ hn, one_smul]

/-- The actual original power test at a radial/perpendicular linear
combination has the original finite binomial integral expansion.
Source: arXiv:2601.21366v2, §5.1, `eq:FH-basic`.
Every term is actually continuous and integrable under the finite law;
its original scalar coefficients and powers remain explicit. -/
theorem powerMomentKernel_radial_perp_expansion {d : ℕ} (β : ℝ)
    (ν : Measure (SSphere d)) [IsFiniteMeasure ν] (x u : EucSpace d) (c s : ℝ) (n : ℕ) :
    powerMomentKernel β ν x (c • x + s • u) n =
      ∑ m ∈ Finset.range (n + 1), (n.choose m : ℝ) * c ^ (n - m) * s ^ m *
        radialPerpMomentKernel β ν x u (n - m) m := by
  let K := fun y : SSphere d => exp (β * inner (𝕜 := ℝ) x (y : EucSpace d))
  let f := fun m (y : SSphere d) => ((n.choose m : ℝ) * c ^ (n - m) * s ^ m) *
    (K y * inner (𝕜 := ℝ) x (y : EucSpace d) ^ (n - m) * inner (𝕜 := ℝ) u (y : EucSpace d) ^ m)
  have hfun (y : SSphere d) : K y * inner (𝕜 := ℝ) (c • x + s • u) (y : EucSpace d) ^ n =
      ∑ m ∈ Finset.range (n + 1), f m y := by
    rw [inner_add_left, real_inner_smul_left, real_inner_smul_left, add_comm, add_pow, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro m _
    dsimp [f]
    rw [mul_pow, mul_pow]
    ring
  have hf (m : ℕ) : Integrable (f m) ν :=
    (Perspective.integrable_of_continuous_compact (continuous_radialPerpMomentIntegrand β x u (n - m) m) ν).const_mul _
  unfold powerMomentKernel
  rw [integral_congr_ae (ae_of_all ν hfun), integral_finsetSum _ (fun m _ => hf m)]
  apply Finset.sum_congr rfl
  intro m _
  exact integral_const_mul _ _

/-- Only actual even perpendicular terms remain under the original
uniform law. Their original squared projection norms are computed here.
Source: arXiv:2601.21366v2, §5.1, `eq:FH-basic`, polynomial tests.
The actual decomposition is supplied by the preceding geometric theorem;
odd terms vanish by the proved original reflection, including boundaries. -/
theorem powerMomentKernel_even_expansion {d : ℕ} (β : ℝ)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν) (x u : SSphere d)
    (b : EucSpace d) (n : ℕ) (hxu : inner (𝕜 := ℝ) (x : EucSpace d) (u : EucSpace d) = 0)
    (hb : b = inner (𝕜 := ℝ) (x : EucSpace d) b • (x : EucSpace d) +
      ‖proj d (x : EucSpace d) b‖ • (u : EucSpace d)) :
    powerMomentKernel β ν (x : EucSpace d) b n = ∑ m ∈ Finset.range (n + 1),
      if Even m then (n.choose m : ℝ) * inner (𝕜 := ℝ) (x : EucSpace d) b ^ (n - m) *
        (‖b‖ ^ 2 - inner (𝕜 := ℝ) (x : EucSpace d) b ^ 2) ^ (m / 2) *
          radialPerpMomentKernel β ν (x : EucSpace d) (u : EucSpace d) (n - m) m else 0 := by
  let _ : IsProbabilityMeasure ν := hν.1
  have hexp := powerMomentKernel_radial_perp_expansion β ν (x : EucSpace d) (u : EucSpace d)
    (inner (𝕜 := ℝ) (x : EucSpace d) b) ‖proj d (x : EucSpace d) b‖ n
  rw [← hb] at hexp
  rw [hexp]
  apply Finset.sum_congr rfl
  intro m _
  by_cases hm : Even m
  · rw [ite_eq_left hm]
    have hs : ‖proj d (x : EucSpace d) b‖ ^ m =
        (‖b‖ ^ 2 - inner (𝕜 := ℝ) (x : EucSpace d) b ^ 2) ^ (m / 2) := by
      calc
        _ = ‖proj d (x : EucSpace d) b‖ ^ (2 * (m / 2)) := by rw [Nat.two_mul_div_two_of_even hm]
        _ = (‖proj d (x : EucSpace d) b‖ ^ 2) ^ (m / 2) := pow_mul _ _ _
        _ = _ := by rw [quadratic_norm_proj_sq (x : EucSpace d) b (norm_eq_of_mem_sphere x)]
    rw [hs]
  · rw [ite_eq_right hm, radialPerpMomentKernel_eq_zero_of_odd β ν hν _ _ _ _ hxu
      (Nat.not_even_iff_odd.mp hm), mul_zero]

/-- The original dimension, uniform, perpendicular and decomposition
inputs hold jointly for the actual circle axes and a nonzero perpendicular
test vector. Source: arXiv:2601.21366v2, §5.1, `eq:FH-basic`.
The actual projection is the same original second-axis vector. -/
example : (2 : ℕ) ≤ 2 ∧ Metastability.IsUniformOn 2 (MeanField.uniformLaw 2) ∧
    ∃ u : SSphere 2, inner (𝕜 := ℝ) (basePoint 1 : EucSpace 2) (u : EucSpace 2) = 0 ∧
      secondAxis = inner (𝕜 := ℝ) (basePoint 1 : EucSpace 2) secondAxis • (basePoint 1 : EucSpace 2) +
        ‖proj 2 (basePoint 1 : EucSpace 2) secondAxis‖ • (u : EucSpace 2) :=
  ⟨le_rfl, Metastability.isUniformOn_uniformLaw (by norm_num),
    exists_unit_perp_decomposition le_rfl (basePoint 1) secondAxis⟩

end Transformer.Perceptron
