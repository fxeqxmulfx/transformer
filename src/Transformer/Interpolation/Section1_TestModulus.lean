module

public import Transformer.Interpolation.Section1_WeakContinuity
public import Mathlib.Geometry.Manifold.SmoothApprox

/-!
# A uniform time modulus for continuous tests of weak solutions

ArXiv:2411.04551v3, §1, `eq: cauchy.pb`, with applications to
§3.2, `lem: perturbation`, Part 2.

A uniform bound on the original vector field bounds the change of a
`C¹` test integral by elapsed time times its derivative bound.
Approximating a continuous ambient test by one smooth test gives a
time modulus chosen before the weak solution. Compactness of the
sphere supplies the derivative bound of the approximation.

The resulting radius applies to every actual weak solution and every
pair of times. No uniqueness or time regularity of the measure curve
is added to the definition of a solution.
-/

@[expose] public section

open Real MeasureTheory Filter Set

namespace Transformer.Interpolation

/-- A bounded field gives a quantitative time estimate for every `C¹`
test integral of an actual weak solution. The bound uses the ambient
Fréchet derivative on the original sphere.
The field bound is required along this solution. Both evaluation times
may lie on either side of the initial time.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem norm_cauchyPB_test_sub_le {d : ℕ} (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (φ : EucSpace d → ℝ) (hφ : ContDiff ℝ 1 φ) (B C : ℝ) (hC : 0 ≤ C)
    (hVF : ∀ (s : ℝ) (x : SSphere d), ‖fullVF d θ (μ s) s x‖ ≤ B)
    (hD : ∀ x : SSphere d, ‖fderiv ℝ φ (x : EucSpace d)‖ ≤ C) (s t : ℝ) :
    ‖(∫ x : SSphere d, φ x ∂(μ t : Measure (SSphere d))) -
      ∫ x : SSphere d, φ x ∂(μ s : Measure (SSphere d))‖ ≤ C * B * |t - s| := by
  let g (r : ℝ) := ∫ x : SSphere d,
    inner ℝ (gradient φ (x : EucSpace d)) (fullVF d θ (μ r) r x)
      ∂(μ r : Measure (SSphere d))
  have hg (r : ℝ) : ‖g r‖ ≤ C * B := by
    have hb : ∀ᵐ x : SSphere d ∂(μ r : Measure (SSphere d)),
        ‖inner ℝ (gradient φ (x : EucSpace d)) (fullVF d θ (μ r) r x)‖ ≤ C * B :=
      ae_of_all _ fun x => by
        rw [inner_gradient_left]
        exact ((fderiv ℝ φ (x : EucSpace d)).le_opNorm _).trans (mul_le_mul (hD x)
          (hVF r x) (norm_nonneg _) hC)
    simpa only [probReal_univ, mul_one] using norm_integral_le_of_norm_le_const hb
  have his : IntervalIntegrable g volume 0 s := (hμ φ hφ s).1
  have hit : IntervalIntegrable g volume 0 t := (hμ φ hφ t).1
  have hist : IntervalIntegrable g volume s t := his.symm.trans hit
  have heq : (∫ x : SSphere d, φ x ∂(μ t : Measure (SSphere d))) -
      (∫ x : SSphere d, φ x ∂(μ s : Measure (SSphere d))) = ∫ r in s..t, g r := by
    rw [(hμ φ hφ t).2, (hμ φ hφ s).2]
    have hi := intervalIntegral.integral_add_adjacent_intervals his hist
    linarith
  rw [heq]
  exact intervalIntegral.norm_integral_le_of_norm_le_const (fun r _ => hg r)

/-- For a uniformly bounded original field, continuous test integrals
have a time modulus uniform over every actual weak solution.
The radius is chosen before the solution and both evaluation times.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`; auxiliary for the
finite-time perturbation in §3.2, `lem: perturbation`, Part 2. -/
theorem exists_uniform_cauchyPB_test_radius {d : ℕ} (θ : TimeParams d)
    (B : ℝ) (hB : 0 ≤ B)
    (hVF : ∀ (ν : Perspective.ProbSphere d) (s : ℝ) (x : SSphere d),
      ‖fullVF d θ ν s x‖ ≤ B)
    (f : EucSpace d → ℝ) (hf : Continuous f) (ε : ℝ) (hε : 0 < ε) :
    ∃ δ : ℝ, 0 < δ ∧ ∀ (μ : ℝ → Perspective.ProbSphere d), cauchyPB d θ μ →
      ∀ s t : ℝ, |t - s| < δ →
        ‖(∫ x : SSphere d, f x ∂(μ t : Measure (SSphere d))) -
          ∫ x : SSphere d, f x ∂(μ s : Measure (SSphere d))‖ < ε := by
  obtain ⟨φ, hφ, happ, -⟩ := hf.exists_contDiff_approx 1
    (continuous_const : Continuous (fun _ : EucSpace d => ε / 4)) (fun _ => by positivity)
  obtain ⟨C₀, hC₀⟩ := (isCompact_univ : IsCompact (univ : Set (SSphere d))).exists_bound_of_continuousOn
    (((hφ.continuous_fderiv (by norm_num)).comp continuous_subtype_val).continuousOn)
  let C := |C₀|
  have hC : 0 ≤ C := abs_nonneg _
  have hD (x : SSphere d) : ‖fderiv ℝ φ (x : EucSpace d)‖ ≤ C :=
    (hC₀ x (mem_univ x)).trans (le_abs_self _)
  have herr (ν : Perspective.ProbSphere d) :
      ‖(∫ x : SSphere d, f x ∂(ν : Measure (SSphere d))) -
        ∫ x : SSphere d, φ x ∂(ν : Measure (SSphere d))‖ ≤ ε / 4 := by
    have hif : Integrable (fun x : SSphere d => f (x : EucSpace d)) (ν : Measure (SSphere d)) :=
      integrable_spherical_continuous ν _ (hf.comp continuous_subtype_val)
    have hiφ : Integrable (fun x : SSphere d => φ (x : EucSpace d)) (ν : Measure (SSphere d)) :=
      integrable_spherical_continuous ν _ (hφ.continuous.comp continuous_subtype_val)
    rw [← integral_sub hif hiφ]
    have hb : ∀ᵐ x : SSphere d ∂(ν : Measure (SSphere d)), ‖f x - φ x‖ ≤ ε / 4 :=
      ae_of_all _ fun x => by
        simpa only [dist_eq_norm, norm_sub_rev] using (happ (x : EucSpace d)).le
    simpa only [probReal_univ, mul_one] using norm_integral_le_of_norm_le_const hb
  let L := C * B
  have hL : 0 ≤ L := mul_nonneg hC hB
  have hden : 0 < 2 * (L + 1) := by positivity
  refine ⟨ε / (2 * (L + 1)), div_pos hε hden, fun μ hμ s t hst => ?_⟩
  have hm := norm_cauchyPB_test_sub_le θ μ hμ φ hφ B C hC (fun r => hVF (μ r) r) hD s t
  have heq : (∫ x : SSphere d, f x ∂(μ t : Measure (SSphere d))) -
      (∫ x : SSphere d, f x ∂(μ s : Measure (SSphere d))) =
      ((∫ x : SSphere d, f x ∂(μ t : Measure (SSphere d))) -
        (∫ x : SSphere d, φ x ∂(μ t : Measure (SSphere d)))) +
      ((∫ x : SSphere d, φ x ∂(μ t : Measure (SSphere d))) -
        (∫ x : SSphere d, φ x ∂(μ s : Measure (SSphere d)))) +
      ((∫ x : SSphere d, φ x ∂(μ s : Measure (SSphere d))) -
        (∫ x : SSphere d, f x ∂(μ s : Measure (SSphere d)))) := by ring
  rw [heq]
  calc
    _ ≤ (‖(∫ x : SSphere d, f x ∂(μ t : Measure (SSphere d))) -
        ∫ x : SSphere d, φ x ∂(μ t : Measure (SSphere d))‖ +
      ‖(∫ x : SSphere d, φ x ∂(μ t : Measure (SSphere d))) -
        ∫ x : SSphere d, φ x ∂(μ s : Measure (SSphere d))‖) +
      ‖(∫ x : SSphere d, φ x ∂(μ s : Measure (SSphere d))) -
        ∫ x : SSphere d, f x ∂(μ s : Measure (SSphere d))‖ := (norm_add_le _ _).trans
          (add_le_add (norm_add_le _ _) le_rfl)
    _ ≤ ε / 4 + L * |t - s| + ε / 4 := by
      exact add_le_add (add_le_add (herr (μ t)) hm)
        (by rw [norm_sub_rev]; exact herr (μ s))
    _ < ε := by
      have h := (lt_div_iff₀ hden).mp hst
      nlinarith [abs_nonneg (t - s)]

/-- Nonzero stationary data for the original pure-attention equation,
a genuine uniform field bound over all probability laws, and a smooth
nonzero test jointly satisfy the premises of both estimates. -/
example :
    cauchyPB 1 (pureAttentionParams 1 0) (fun _ => Perspective.diracProb 1 (basePoint 0)) ∧
    (0 : ℝ) ≤ 2 ∧ (0 : ℝ) ≤ 1 ∧
    (∀ (ν : Perspective.ProbSphere 1) (s : ℝ) (x : SSphere 1),
      ‖fullVF 1 (pureAttentionParams 1 0) ν s x‖ ≤ 2) ∧
    ContDiff ℝ 1 (fun _ : EucSpace 1 => (1 : ℝ)) ∧
    Continuous (fun _ : EucSpace 1 => (1 : ℝ)) ∧
    (∀ x : SSphere 1, ‖fderiv ℝ (fun _ : EucSpace 1 => (1 : ℝ)) (x : EucSpace 1)‖ ≤ 1) ∧
    (0 : ℝ) < 1 := by
  refine ⟨cauchyPB_pureAttention_dirac 0 (basePoint 0), by norm_num, zero_le_one,
    ?_, contDiff_const, continuous_const, by simp, zero_lt_one⟩
  intro ν s x
  rw [fullVF_pureAttention_zero]
  let m := ∫ y : SSphere 1, (y : EucSpace 1) ∂(ν : Measure (SSphere 1))
  have hm : ‖m‖ ≤ 1 := norm_spherical_mean_le_one ν
  have hx : ‖(x : EucSpace 1)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  have hi : |inner ℝ (x : EucSpace 1) m| ≤ 1 := by
    have h := abs_real_inner_le_norm (x : EucSpace 1) m
    simpa only [hx, one_mul] using h.trans (by simpa only [hx, one_mul] using hm)
  calc
    ‖proj 1 x m‖ ≤ ‖m‖ + ‖inner ℝ (x : EucSpace 1) m • (x : EucSpace 1)‖ := norm_sub_le _ _
    _ = ‖m‖ + |inner ℝ (x : EucSpace 1) m| := by rw [norm_smul, Real.norm_eq_abs, hx, mul_one]
    _ ≤ 2 := by linarith

end Transformer.Interpolation
