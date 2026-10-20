module

public import Transformer.Causal.Section4_SubspaceCovectors
public import Transformer.GPTMini.Bridge.SphereResidence

/-!
# The direction of a negative leading projection

ArXiv:2411.04990v2, §4, geometry for the exceptional branch of `thm2`.
The off-axis leading part is the orthogonal projection onto the part of
`L` perpendicular to its leading unit axis. Convergence of every fixed
transverse leading covector gives norm convergence of this part in finite
dimension. A uniformly negative longitudinal component then makes the
normalized leading projection converge to the negative axis. The final
normalization estimate reuses the repository's RMSNorm bridge estimate.
-/

@[expose] public section

open Real Filter
open scoped Topology

namespace Transformer.Causal

variable {d : ℕ}

/-- The leading projection with its component along `u` removed.
For a unit `u ∈ L`, this is the projection onto `L ∩ uᗮ`.
Source: arXiv:2411.04990v2, §4, the leading splitting in `thm2`. -/
noncomputable def leadingOffAxis (L : Submodule ℝ (EucSpace d))
    (u x : EucSpace d) : EucSpace d :=
  L.starProjection x - inner ℝ u x • u

/-- Removing a leading-axis component preserves membership in the
leading subspace. Source: arXiv:2411.04990v2, §4, `thm2`. -/
theorem leadingOffAxis_mem (L : Submodule ℝ (EucSpace d)) (u x : EucSpace d)
    (hu : u ∈ L) : leadingOffAxis L u x ∈ L :=
  L.sub_mem (L.starProjection_apply_mem x) (L.smul_mem _ hu)

/-- The off-axis projection is self-adjoint, even before imposing the
unit-axis hypothesis. Source: arXiv:2411.04990v2, §4, the orthogonal
leading splitting of `thm2`. -/
theorem inner_leadingOffAxis (L : Submodule ℝ (EucSpace d)) (u v x : EucSpace d) :
    inner ℝ v (leadingOffAxis L u x) = inner ℝ (leadingOffAxis L u v) x := by
  simp only [leadingOffAxis, inner_sub_right, inner_sub_left, real_inner_smul_right,
    real_inner_smul_left, L.inner_starProjection_left_eq_right]
  rw [real_inner_comm u v]
  ring

/-- Removing the component along a leading unit axis makes the result
orthogonal to that axis. Source: arXiv:2411.04990v2, §4, `thm2`. -/
theorem leadingOffAxis_inner_axis (L : Submodule ℝ (EucSpace d)) (u : SSphere d)
    (hu : (u : EucSpace d) ∈ L) (x : EucSpace d) :
    inner ℝ (leadingOffAxis L u x) u = 0 := by
  rw [real_inner_comm (u : EucSpace d) (leadingOffAxis L u x)]
  simp only [leadingOffAxis, inner_sub_right, real_inner_smul_right,
    ← L.inner_starProjection_left_eq_right, L.starProjection_eq_self_iff.mpr hu,
    real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp u.2, one_pow, mul_one, sub_self]

/-- Every fixed transverse leading covector tending to zero gives norm
convergence of the off-axis leading projection. The finite-dimensional
coordinate test is explicit. Source: arXiv:2411.04990v2, §4, geometry
for the exceptional-trajectory reduction of `thm2`. -/
theorem tendsto_leadingOffAxis (L : Submodule ℝ (EucSpace d)) (u : SSphere d)
    (hu : (u : EucSpace d) ∈ L) (x : ℝ → EucSpace d)
    (hx : ∀ v ∈ L, inner ℝ v u = 0 →
      Tendsto (fun t => inner ℝ v (x t)) atTop (nhds 0)) :
    Tendsto (fun t => leadingOffAxis L u (x t)) atTop (nhds 0) := by
  have hcoord (j : Fin d) : Tendsto (fun t => (leadingOffAxis L u (x t)) j)
      atTop (nhds 0) := by
    let e : EucSpace d := EuclideanSpace.single j 1
    have h := hx (leadingOffAxis L u e) (leadingOffAxis_mem L u e hu)
      (leadingOffAxis_inner_axis L u hu e)
    convert! h using 1
    ext t
    rw [← inner_leadingOffAxis, EuclideanSpace.inner_single_left]
    simp only [map_one, one_mul]
  have hp : Tendsto (fun t => (leadingOffAxis L u (x t)).ofLp) atTop
      (nhds (fun _ : Fin d => (0 : ℝ))) := tendsto_pi_nhds.mpr hcoord
  have h := ((PiLp.continuous_toLp 2 (fun _ : Fin d => ℝ)).tendsto
    (fun _ : Fin d => (0 : ℝ))).comp hp
  convert! h using 1

/-- A uniformly negative leading overlap bounds the leading projection
away from zero, so its normalized direction is defined on that tail.
Source: arXiv:2411.04990v2, §4, the exceptional branch of `thm2`. -/
theorem norm_starProjection_ge_of_negative (L : Submodule ℝ (EucSpace d))
    (u : SSphere d) (hu : (u : EucSpace d) ∈ L) (x : EucSpace d) (δ : ℝ)
    (hq : inner ℝ (u : EucSpace d) x ≤ -δ) : δ ≤ ‖L.starProjection x‖ := by
  have he : inner ℝ (u : EucSpace d) (L.starProjection x) = inner ℝ (u : EucSpace d) x := by
    rw [← L.inner_starProjection_left_eq_right, L.starProjection_eq_self_iff.mpr hu]
  have hc := abs_real_inner_le_norm (u : EucSpace d) (L.starProjection x)
  rw [mem_sphere_zero_iff_norm.mp u.2, one_mul, he] at hc
  exact (show δ ≤ -inner ℝ (u : EucSpace d) x by linarith).trans ((neg_le_abs _).trans hc)

/-- A uniformly negative overlap controls the distance of the normalized
leading projection from the negative axis by the off-axis component.
Source: arXiv:2411.04990v2, §4, geometry for `thm2`; the normalization
bound is `GPTMini.Bridge.norm_inv_smul_sub_le`. -/
theorem norm_normalized_projection_neg_le (L : Submodule ℝ (EucSpace d))
    (u : SSphere d) (x : EucSpace d) (δ : ℝ) (hδ : 0 < δ)
    (hq : inner ℝ (u : EucSpace d) x ≤ -δ) :
    ‖‖L.starProjection x‖⁻¹ • L.starProjection x - (-(u : EucSpace d))‖ ≤
      2 * ‖leadingOffAxis L u x‖ / δ := by
  let q := inner ℝ (u : EucSpace d) x
  have hqneg : q < 0 := by dsimp [q]; linarith
  have hu0 : (u : EucSpace d) ≠ 0 := by
    exact norm_ne_zero_iff.mp (by rw [mem_sphere_zero_iff_norm.mp u.2]; norm_num)
  have ha : q • (u : EucSpace d) ≠ 0 := smul_ne_zero (ne_of_lt hqneg) hu0
  have hnorm : ‖q • (u : EucSpace d)‖ = -q := by
    rw [norm_smul, Real.norm_eq_abs, abs_of_neg hqneg,
      mem_sphere_zero_iff_norm.mp u.2, mul_one]
  have hdir : ‖q • (u : EucSpace d)‖⁻¹ • (q • (u : EucSpace d)) = -(u : EucSpace d) := by
    rw [hnorm, smul_smul]
    have he : (-q)⁻¹ * q = -1 := by field_simp [ne_of_lt hqneg]
    rw [he, neg_one_smul]
  have h := GPTMini.Bridge.norm_inv_smul_sub_le (L.starProjection x) ha
  rw [hdir, hnorm] at h
  apply h.trans
  exact div_le_div_of_nonneg_left (by positivity) hδ (by dsimp [q]; linarith)

/-- A vanishing off-axis leading projection and a uniformly negative
longitudinal overlap imply convergence of the normalized leading
projection to the negative axis. No limit of the full token is assumed.
Source: arXiv:2411.04990v2, §4, the exceptional branch of `thm2`. -/
theorem tendsto_normalized_projection_neg (L : Submodule ℝ (EucSpace d))
    (u : SSphere d) (x : ℝ → EucSpace d)
    (hx : Tendsto (fun t => leadingOffAxis L u (x t)) atTop (nhds 0))
    (δ : ℝ) (hδ : 0 < δ)
    (hq : ∀ᶠ t in atTop, inner ℝ (u : EucSpace d) (x t) ≤ -δ) :
    Tendsto (fun t => ‖L.starProjection (x t)‖⁻¹ • L.starProjection (x t))
      atTop (nhds (-(u : EucSpace d))) := by
  apply tendsto_iff_norm_sub_tendsto_zero.mpr
  apply squeeze_zero' (Eventually.of_forall fun _ => norm_nonneg _) (hq.mono fun t ht =>
    norm_normalized_projection_neg_le L u (x t) δ hδ ht)
  simpa only [norm_zero, mul_zero, zero_div] using (hx.norm.const_mul (2 : ℝ)).div_const δ

/-- A negative leading unit vector jointly realizes the subspace,
orthogonality, vanishing-covector and uniformly negative hypotheses.
Source: arXiv:2411.04990v2, §4, the antipodal geometry of `thm2`. -/
example : ∃ (u : SSphere 2) (x : ℝ → EucSpace 2),
    (u : EucSpace 2) ∈ (⊤ : Submodule ℝ (EucSpace 2)) ∧
    (∀ v ∈ (⊤ : Submodule ℝ (EucSpace 2)), inner ℝ v u = 0 →
      Tendsto (fun t => inner ℝ v (x t)) atTop (nhds 0)) ∧
    Tendsto (fun t => leadingOffAxis ⊤ u (x t)) atTop (nhds 0) ∧
    (0 : ℝ) < 1 ∧ (∀ᶠ t in atTop, inner ℝ (u : EucSpace 2) (x t) ≤ -(1 : ℝ)) := by
  let u : SSphere 2 := basePoint 1
  refine ⟨u, fun _ => -(u : EucSpace 2), Submodule.mem_top, ?_, ?_, one_pos, ?_⟩
  · intro v _ hv
    simpa only [inner_neg_right, hv, neg_zero] using
      (tendsto_const_nhds : Tendsto (fun _ : ℝ => (0 : ℝ)) atTop (nhds 0))
  · have he : leadingOffAxis ⊤ u (-(u : EucSpace 2)) = 0 := by
      simp [leadingOffAxis, Submodule.starProjection_top, mem_sphere_zero_iff_norm.mp u.2]
    simpa only [he] using
      (tendsto_const_nhds : Tendsto (fun _ : ℝ => (0 : EucSpace 2)) atTop (nhds 0))
  · exact Eventually.of_forall fun _ => by simp [mem_sphere_zero_iff_norm.mp u.2]

end Transformer.Causal
