module

public import Transformer.Causal.SingleToken
public import Mathlib.Analysis.InnerProductSpace.Projection.Basic
public import Mathlib.Topology.Order.Compact

/-!
# The spectral splitting in the leading-subspace conjecture

ArXiv:2411.04990v2, §4, `thm2`, and §3, `lemma1`.
The manuscript assumes a scalar value map on the leading eigenspace,
invariance of its orthogonal complement, and strict quadratic decay on
that complement. These hypotheses give the actual projection identity
and a uniform positive transverse gap. No symmetry of the value map,
diagonalization, convergence, or quantitative gap is assumed.
-/

@[expose] public section

open Real

namespace Transformer.Causal

variable {d : ℕ}

/-- The leading projection of the original value field is its leading
eigenvalue times the original projection. Source: arXiv:2411.04990v2,
§4, `thm2`, the invariant orthogonal splitting used after `lemma1`. -/
theorem starProjection_value_eq (V : ParamMatrix d) (L : Submodule ℝ (EucSpace d))
    (lam : ℝ) (heig : ∀ v ∈ L, V v = lam • v)
    (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ) (x : EucSpace d) :
    L.starProjection (V x) = lam • L.starProjection x := by
  have hp := L.starProjection_apply_mem x
  have hz := L.sub_starProjection_mem_orthogonal x
  have hzero := L.starProjection_apply_eq_zero_iff.mpr (hinv _ hz)
  rw [map_sub, heig _ hp, map_sub, map_smul,
    L.starProjection_eq_self_iff.mpr hp] at hzero
  exact sub_eq_zero.mp hzero

/-- The strict transverse quadratic inequality in `thm2` has a uniform
positive gap in finite dimension. The proof also covers a zero orthogonal
complement, without assuming its unit sphere is nonempty.
Source: arXiv:2411.04990v2, §4, `thm2`. -/
theorem exists_transverse_gap (V : ParamMatrix d) (L : Submodule ℝ (EucSpace d))
    (lam : ℝ)
    (hdecay : ∀ z ∈ Lᗮ, z ≠ 0 → inner ℝ (V z) z < lam * ‖z‖ ^ 2) :
    ∃ c : ℝ, 0 < c ∧ ∀ z ∈ Lᗮ,
      inner ℝ (V z) z ≤ (lam - c) * ‖z‖ ^ 2 := by
  let S : Set (EucSpace d) := (Lᗮ : Set (EucSpace d)) ∩ Metric.sphere 0 1
  have hcompact : IsCompact S := (isCompact_sphere (0 : EucSpace d) 1).inter_left
    L.isClosed_orthogonal
  have hcont : Continuous (fun z : EucSpace d => lam - inner ℝ (V z) z) :=
    continuous_const.sub (V.continuous.inner continuous_id)
  obtain ⟨c, hc, hgap⟩ := hcompact.exists_forall_le' hcont.continuousOn
    (a := (0 : ℝ)) (fun z hz => by
      have hn : ‖z‖ = 1 := mem_sphere_zero_iff_norm.mp hz.2
      have h := hdecay z hz.1 (norm_ne_zero_iff.mp (by rw [hn]; norm_num))
      rw [hn, one_pow, mul_one] at h
      exact sub_pos.mpr h)
  refine ⟨c, hc, fun z hz => ?_⟩
  by_cases hz0 : z = 0
  · simp [hz0]
  have hn : 0 < ‖z‖ := norm_pos_iff.mpr hz0
  have hu : ‖z‖⁻¹ • z ∈ S := by
    refine ⟨Lᗮ.smul_mem _ hz, mem_sphere_zero_iff_norm.mpr ?_⟩
    rw [norm_smul, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hn), inv_mul_cancel₀ hn.ne']
  have h := hgap _ hu
  rw [map_smul, real_inner_smul_left, real_inner_smul_right] at h
  have hm := mul_le_mul_of_nonneg_left h (sq_nonneg ‖z‖)
  have hcancel : ‖z‖ ^ 2 * (‖z‖⁻¹ * (‖z‖⁻¹ * inner ℝ (V z) z)) =
      inner ℝ (V z) z := by field_simp
  rw [mul_sub, hcancel] at hm
  nlinarith

/-- A genuine non-scalar value map realizes the projection, invariance,
and strict transverse hypotheses on a nonzero orthogonal complement.
Source: arXiv:2411.04990v2, §4, the spectral assumptions of `thm2`.
Here the value map is the orthogonal projection itself. -/
example : ∃ (V : ParamMatrix 3) (L : Submodule ℝ (EucSpace 3)),
    (∀ v ∈ L, V v = (1 : ℝ) • v) ∧
    (∀ z ∈ Lᗮ, V z ∈ Lᗮ) ∧
    (∀ z ∈ Lᗮ, z ≠ 0 → inner ℝ (V z) z < (1 : ℝ) * ‖z‖ ^ 2) := by
  let L : Submodule ℝ (EucSpace 3) :=
    (ℝ ∙ EuclideanSpace.single (0 : Fin 3) (1 : ℝ))ᗮ
  refine ⟨L.starProjection, L, ?_, ?_, ?_⟩
  · intro v hv
    simpa only [one_smul] using L.starProjection_eq_self_iff.mpr hv
  · intro z hz
    rw [L.starProjection_apply_eq_zero_iff.mpr hz]
    exact Lᗮ.zero_mem
  · intro z hz hz0
    rw [L.starProjection_apply_eq_zero_iff.mpr hz, inner_zero_left, one_mul]
    exact sq_pos_of_pos (norm_pos_iff.mpr hz0)

end Transformer.Causal
