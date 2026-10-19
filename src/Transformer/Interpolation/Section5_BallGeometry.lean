module

public import Transformer.Perceptron.Analytic
public import Transformer.Perspective.SphereInvariant
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
public import Mathlib.Analysis.Real.Pi.Bounds

/-!
# Convex spherical caps in the ball decomposition

ArXiv:2411.04551v3, §5, proof of `cl: balls`, the monotonicity of
`s ↦ ν(B(γ(s),r))` on the small support. A small ambient ball on the
unit sphere is a positive inner-product cap. Its homogeneous cone is
convex, so a shorter great-circle segment stays in the cap whenever
both endpoints do. This proves nesting on the carrying set without
claiming that the actual equal-radius balls are globally nested.

The great circle is the existing explicit sine/cosine curve of
`Perceptron.Analytic`; the identities below concern its actual unit
points, independently of any measure or flow.
-/

@[expose] public section

open Real Metric Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The endpoint of the source's geodesic has a genuine orthogonal
unit tangent in every dimension at least two. Source:
arXiv:2411.04551v3, §5, `cl: balls`, choosing the initial center. -/
theorem exists_unit_tangent (hd : 2 ≤ d) (p : SSphere d) :
    ∃ v : SSphere d, inner (𝕜 := ℝ) (p : EucSpace d) (v : EucSpace d) = 0 := by
  let e₀ : SSphere d := ⟨EuclideanSpace.single ⟨0, by omega⟩ 1, by simp⟩
  let e₁ : SSphere d := ⟨EuclideanSpace.single ⟨1, by omega⟩ 1, by simp⟩
  obtain ⟨U, hU⟩ := Perspective.exists_sphereMap_apply_eq d e₀ p
  have hp : U (e₀ : EucSpace d) = (p : EucSpace d) := congrArg Subtype.val hU
  refine ⟨Perspective.sphereMap d U e₁, ?_⟩
  change inner (𝕜 := ℝ) (p : EucSpace d) (U (e₁ : EucSpace d)) = 0
  rw [← hp, U.inner_map_map]
  simp [e₀, e₁, EuclideanSpace.inner_single_left]

example : (2 : ℕ) ≤ 2 := le_rfl

/-- An actual positive-radius ball on the unit sphere is the actual
inner-product cap with threshold `1-r²/2`. Source:
arXiv:2411.04551v3, §5, the balls in `cl: balls` (ambient radii). -/
theorem mem_sphere_ball_iff_inner (x p : SSphere d) {r : ℝ} (hr : 0 < r) :
    x ∈ ball p r ↔
      1 - r ^ 2 / 2 < inner (𝕜 := ℝ) (x : EucSpace d) (p : EucSpace d) := by
  change ‖(x : EucSpace d) - (p : EucSpace d)‖ < r ↔ _
  rw [← sq_lt_sq₀ (norm_nonneg _) hr.le, norm_sub_sq_real,
    norm_eq_of_mem_sphere, norm_eq_of_mem_sphere]
  constructor <;> intro h <;> nlinarith

example : (0 : ℝ) < 1 := one_pos

/-- The homogeneous cone of an actual small spherical ball is
convex: a nonnegative linear combination that is itself unit length
remains in that same ball. Source: arXiv:2411.04551v3, §5,
the monotonicity step of `cl: balls`; both actual endpoint memberships
are retained, and the radius bound gives a nonnegative cap threshold. -/
theorem mem_sphere_ball_of_nonneg_combination (x p q z : SSphere d)
    {r a b : ℝ} (hr : 0 < r) (hr1 : r ≤ 1) (ha : 0 ≤ a) (hb : 0 ≤ b)
    (hz : (z : EucSpace d) = a • (p : EucSpace d) + b • (q : EucSpace d))
    (hp : x ∈ ball p r) (hq : x ∈ ball q r) : x ∈ ball z r := by
  let c := 1 - r ^ 2 / 2
  have hc : 0 ≤ c := by dsimp [c]; nlinarith
  have hp' := (mem_sphere_ball_iff_inner x p hr).mp hp
  have hq' := (mem_sphere_ball_iff_inner x q hr).mp hq
  have hn : 1 ≤ a + b := by
    have h := norm_add_le (a • (p : EucSpace d)) (b • (q : EucSpace d))
    rw [← hz, norm_smul, norm_smul, norm_eq_of_mem_sphere, norm_eq_of_mem_sphere,
      norm_eq_of_mem_sphere, Real.norm_of_nonneg ha, Real.norm_of_nonneg hb,
      mul_one, mul_one] at h
    exact h
  have hpos : 0 < a * (inner (𝕜 := ℝ) (x : EucSpace d) (p : EucSpace d) - c) +
      b * (inner (𝕜 := ℝ) (x : EucSpace d) (q : EucSpace d) - c) := by
    have hp0 : 0 < inner (𝕜 := ℝ) (x : EucSpace d) (p : EucSpace d) - c := sub_pos.mpr hp'
    have hq0 : 0 < inner (𝕜 := ℝ) (x : EucSpace d) (q : EucSpace d) - c := sub_pos.mpr hq'
    rcases lt_or_eq_of_le ha with ha | ha
    · exact add_pos_of_pos_of_nonneg (mul_pos ha hp0) (mul_nonneg hb hq0.le)
    · have hbpos : 0 < b := by linarith
      rw [← ha, zero_mul, zero_add]
      exact mul_pos hbpos hq0
  apply (mem_sphere_ball_iff_inner x z hr).mpr
  rw [hz, inner_add_right, real_inner_smul_right, real_inner_smul_right]
  have hcn : c ≤ c * (a + b) := by nlinarith [mul_nonneg hc (sub_nonneg.mpr hn)]
  dsimp [c] at hpos hcn ⊢
  nlinarith

/-- Actual unit endpoint data realize every conic-cap input, including
the boundary coefficient zero. Source: arXiv:2411.04551v3, §5, `cl: balls`. -/
example : (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 0 ∧
    (basePoint 1 : EucSpace 2) = (1 : ℝ) • (basePoint 1 : EucSpace 2) +
      (0 : ℝ) • (basePoint 1 : EucSpace 2) ∧
    basePoint 1 ∈ ball (basePoint 1 : SSphere 2) 1 := by simp

/-- The actual great circle as a sphere-valued path of centers.
Source: arXiv:2411.04551v3, §5, the minimizing geodesic `γ` in `cl: balls`.
Its ambient formula is `Perceptron.greatCircle`, introduced for
arXiv:2601.21366v2, §5.2. -/
noncomputable def sweepPoint (p v : SSphere d)
    (hperp : inner (𝕜 := ℝ) (p : EucSpace d) (v : EucSpace d) = 0) (s : ℝ) : SSphere d :=
  ⟨Perceptron.greatCircle (p : EucSpace d) (v : EucSpace d) s,
    mem_sphere_zero_iff_norm.mpr (Perceptron.norm_greatCircle
      (norm_eq_of_mem_sphere p) (norm_eq_of_mem_sphere v) hperp s)⟩

/-- The actual sweep starts at the carrying cap's center.
Source: arXiv:2411.04551v3, §5, the endpoint of `γ` in `cl: balls`. -/
@[simp] theorem sweepPoint_zero (p v : SSphere d)
    (hperp : inner (𝕜 := ℝ) (p : EucSpace d) (v : EucSpace d) = 0) :
    sweepPoint p v hperp 0 = p := by
  apply Subtype.ext
  simp [sweepPoint, Perceptron.greatCircle]

/-- The source's actual center path is continuous in its angular time.
Source: arXiv:2411.04551v3, §5, `cl: balls`, continuity of `γ`. -/
theorem continuous_sweepPoint (p v : SSphere d)
    (hperp : inner (𝕜 := ℝ) (p : EucSpace d) (v : EucSpace d) = 0) :
    Continuous (sweepPoint p v hperp) := by
  apply Continuous.subtype_mk
  exact (Real.continuous_cos.smul continuous_const).add
    (Real.continuous_sin.smul continuous_const)

/-- A point of a shorter great-circle segment is an actual
nonnegative linear combination of its two endpoint vectors.
Source: arXiv:2411.04551v3, §5, `cl: balls`, monotonicity of the sweep. -/
theorem greatCircle_sin_interpolation (p v : EucSpace d) (s t : ℝ) (ht : Real.sin t ≠ 0) :
    Perceptron.greatCircle p v s = (Real.sin (t - s) / Real.sin t) • p +
      (Real.sin s / Real.sin t) • Perceptron.greatCircle p v t := by
  have hc : Real.sin (t - s) / Real.sin t + Real.sin s / Real.sin t * Real.cos t = Real.cos s := by
    rw [Real.sin_sub]
    field_simp
    ring
  have hs : Real.sin s / Real.sin t * Real.sin t = Real.sin s := by field_simp
  symm
  calc
    _ = (Real.sin (t - s) / Real.sin t + Real.sin s / Real.sin t * Real.cos t) • p +
        (Real.sin s / Real.sin t * Real.sin t) • v := by unfold Perceptron.greatCircle; module
    _ = _ := by rw [hc, hs]; rfl

/-- On a small support inside the endpoint ball, the actual balls
along a shorter segment grow toward the endpoint. Source:
arXiv:2411.04551v3, §5, `cl: balls`, the monotonicity used for equal masses. -/
theorem sweep_ball_nested_on_endpoint_ball (p v : SSphere d)
    (hperp : inner (𝕜 := ℝ) (p : EucSpace d) (v : EucSpace d) = 0)
    {r s t : ℝ} (hr : 0 < r) (hr1 : r ≤ 1) (hs : 0 ≤ s) (hst : s ≤ t)
    (ht : t < Real.pi) (x : SSphere d) (hp : x ∈ ball p r)
    (hx : x ∈ ball (sweepPoint p v hperp t) r) : x ∈ ball (sweepPoint p v hperp s) r := by
  by_cases ht0 : t = 0
  · have hs0 : s = 0 := by linarith
    simpa [hs0, ht0] using hx
  have htpos : 0 < t := lt_of_le_of_ne (hs.trans hst) (Ne.symm ht0)
  have hsin := Real.sin_pos_of_pos_of_lt_pi htpos ht
  apply mem_sphere_ball_of_nonneg_combination x p (sweepPoint p v hperp t)
    (sweepPoint p v hperp s) hr hr1
    (div_nonneg (Real.sin_nonneg_of_nonneg_of_le_pi (by linarith) (by linarith)) hsin.le)
    (div_nonneg (Real.sin_nonneg_of_nonneg_of_le_pi hs (hst.trans ht.le)) hsin.le)
    (greatCircle_sin_interpolation _ _ s t hsin.ne') hp hx

/-- Two actual unit axes satisfy every geometric frame and segment
input, for a nonzero segment and a positive ball radius. -/
example : ∃ (p v : SSphere 2)
    (hperp : inner (𝕜 := ℝ) (p : EucSpace 2) (v : EucSpace 2) = 0),
    (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 / 4 ∧ (1 / 4 : ℝ) ≤ 1 / 2 ∧
      (1 / 2 : ℝ) < Real.pi ∧ p ∈ ball p 1 ∧
      p ∈ ball (sweepPoint p v hperp (1 / 2)) 1 := by
  obtain ⟨v, hv⟩ := exists_unit_tangent le_rfl (basePoint 1)
  refine ⟨basePoint 1, v, hv, one_pos, le_rfl, by norm_num, by norm_num,
    by linarith [Real.pi_gt_three], by simp, ?_⟩
  apply (mem_sphere_ball_iff_inner _ _ one_pos).mpr
  change 1 - (1 : ℝ) ^ 2 / 2 < inner (𝕜 := ℝ) (basePoint 1 : EucSpace 2)
    (Perceptron.greatCircle (basePoint 1 : EucSpace 2) (v : EucSpace 2) (1 / 2))
  rw [Perceptron.inner_greatCircle (norm_eq_of_mem_sphere _) hv]
  have h := (abs_le.mp (Real.cos_bound (x := 1 / 2) (by norm_num))).1
  norm_num at h ⊢
  linarith

end Transformer.Interpolation
