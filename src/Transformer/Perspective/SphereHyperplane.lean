/-
# Rotation-invariant measures do not charge great spheres

`Perspective.UniformTuple` pins the uniform law on `𝕊^{d-1}` down by rotation
invariance.  `Perspective.SphereInvariant` draws from that description that
points are null; this file draws the next fact, that great spheres are: a
rotation-invariant finite measure `σ` gives `{x ∈ 𝕊^{d-1} : ⟨x, u⟩ = 0}` mass
`0` for every `u ≠ 0` (`measure_greatSphere_eq_zero`), hence every proper
subspace of `ℝ^d` meets the sphere in a null set
(`measure_mem_submodule_eq_zero`).

The argument is the one for points, with great spheres in place of points.  A
reflection carries any unit vector to any other, and with it the great sphere
orthogonal to it, so all great spheres carry one mass `c`.  The vectors
`(1, t, t², …, t^{d-1})`, `t = 0, 1, …, N - 1`, are in general position: a
nonzero `x` orthogonal to one of them has that `t` as a root of the nonzero
polynomial `Σ_k x_k t^k` of degree `< d`, so `x` lies on at most `d - 1` of
their great spheres.  Integrating the count, `N c ≤ (d - 1) σ(𝕊^{d-1})` for
every `N`, and `c = 0`.

Source: arXiv:2312.10794v5, §6.1 (the footnote to `thm: d.infty`: that `n ≤ d`
uniform points lie in an open hemisphere almost surely "is easy to see
directly").
-/

import Transformer.Perspective.SphereInvariant
import Mathlib.Algebra.Polynomial.Roots
import Mathlib.MeasureTheory.Integral.Lebesgue.Add

open scoped BigOperators ENNReal
open MeasureTheory Polynomial

namespace Transformer
namespace Perspective

variable (d : ℕ)

/-- The great sphere of `𝕊^{d-1}` orthogonal to `u`. -/
def greatSphere (u : EucSpace d) : Set (SSphere d) :=
  {x | inner (𝕜 := ℝ) (x : EucSpace d) u = 0}

theorem measurableSet_greatSphere (u : EucSpace d) : MeasurableSet (greatSphere d u) :=
  (isClosed_eq (continuous_subtype_val.inner continuous_const) continuous_const).measurableSet

/-- Rescaling `u` does not move its great sphere. -/
theorem greatSphere_smul (u : EucSpace d) {c : ℝ} (hc : c ≠ 0) :
    greatSphere d (c • u) = greatSphere d u := by
  ext x
  simp [greatSphere, real_inner_smul_right, hc]

/-- **All great spheres carry one mass** under a rotation-invariant measure:
an isometry `U` with `U x = y` pulls the great sphere orthogonal to `y` back
to the one orthogonal to `x`. -/
theorem measure_greatSphere_eq (σ : Measure (SSphere d))
    (hσ : ∀ U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d, σ.map (sphereMap d U) = σ)
    (x y : SSphere d) : σ (greatSphere d x) = σ (greatSphere d y) := by
  obtain ⟨U, hU⟩ := exists_sphereMap_apply_eq d x y
  have hpre : sphereMap d U ⁻¹' greatSphere d y = greatSphere d x := by
    ext z
    simp only [Set.mem_preimage, greatSphere, Set.mem_ofPred_eq]
    rw [← hU]
    show inner (𝕜 := ℝ) (U (z : EucSpace d)) (U (x : EucSpace d)) = 0 ↔ _
    rw [LinearIsometryEquiv.inner_map_map]
  have := Measure.map_apply (μ := σ) (measurable_sphereMap d U) (measurableSet_greatSphere d y)
  rw [hσ U, hpre] at this
  exact this.symm

/-- The hypothesis of `measure_greatSphere_eq` is satisfiable: the zero
measure is invariant under every `sphereMap`. -/
example : ∀ U : EucSpace 1 ≃ₗᵢ[ℝ] EucSpace 1,
    (0 : Measure (SSphere 1)).map (sphereMap 1 U) = 0 := fun _ => Measure.map_zero _

/-! ### Great spheres in general position -/

/-- The point `(1, t, t², …, t^{d-1})` of the moment curve. -/
noncomputable def momentVec (t : ℝ) : EucSpace d :=
  WithLp.toLp 2 fun k : Fin d => t ^ (k : ℕ)

/-- The polynomial `Σ_k x_k X^k` with the coordinates of `x` as coefficients. -/
noncomputable def coordPoly (x : EucSpace d) : ℝ[X] :=
  ∑ k : Fin d, C (x k) * X ^ (k : ℕ)

/-- `⟨x, (1, t, …, t^{d-1})⟩` is `coordPoly x` evaluated at `t`. -/
theorem inner_momentVec (x : EucSpace d) (t : ℝ) :
    inner (𝕜 := ℝ) x (momentVec d t) = (coordPoly d x).eval t := by
  simp only [momentVec, coordPoly, PiLp.inner_apply, Real.inner_apply, eval_finsetSum, eval_mul,
    eval_C, eval_pow, eval_X]

theorem coeff_coordPoly (x : EucSpace d) (k : Fin d) : (coordPoly d x).coeff k = x k := by
  simp [coordPoly, finsetSum_coeff, Fin.val_inj]

theorem natDegree_coordPoly_le (x : EucSpace d) : (coordPoly d x).natDegree ≤ d - 1 :=
  natDegree_sum_le_of_forall_le _ _ fun k _ =>
    (natDegree_C_mul_X_pow_le _ _).trans (by omega)

/-- **A nonzero vector is orthogonal to at most `d - 1` of the points
`momentVec m`, `m < N`**: each such `m` is a root of `coordPoly x`, which is
nonzero and of degree at most `d - 1`. -/
theorem card_filter_inner_momentVec_le {x : EucSpace d} (hx : x ≠ 0) (N : ℕ) :
    ((Finset.range N).filter fun m : ℕ => inner (𝕜 := ℝ) x (momentVec d m) = 0).card ≤ d - 1 := by
  have hp : coordPoly d x ≠ 0 := by
    intro hp
    refine hx (PiLp.ext fun k => ?_)
    rw [← coeff_coordPoly, hp, coeff_zero]
    rfl
  refine le_trans ?_ (natDegree_coordPoly_le d x)
  have h := card_le_degree_of_subset_roots (p := coordPoly d x)
    (Z := ((Finset.range N).filter fun m : ℕ => inner (𝕜 := ℝ) x (momentVec d m) = 0).image
      (Nat.cast : ℕ → ℝ))
    (fun z hz => by
      rw [Finset.mem_val] at hz
      obtain ⟨m, hm, rfl⟩ := Finset.mem_image.1 hz
      rw [mem_roots hp, IsRoot, ← inner_momentVec]
      exact (Finset.mem_filter.1 hm).2)
  rwa [Finset.card_image_of_injective _ Nat.cast_injective] at h

/-! ### Great spheres are null -/

/-- The great sphere of a nonzero `v` is that of the unit vector along it. -/
theorem exists_greatSphere_eq {v : EucSpace d} (hv : v ≠ 0) :
    ∃ y : SSphere d, greatSphere d v = greatSphere d y := by
  have hnv : ‖v‖ ≠ 0 := norm_ne_zero_iff.mpr hv
  refine ⟨⟨‖v‖⁻¹ • v, ?_⟩, (greatSphere_smul d v (inv_ne_zero hnv)).symm⟩
  rw [mem_sphere_zero_iff_norm, norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ hnv]

/-- **A rotation-invariant finite measure on `𝕊^{d-1}` does not charge great
spheres.**  All of them carry one mass `c` (`measure_greatSphere_eq`); a point
of the sphere lies on at most `d - 1` of the great spheres orthogonal to
`momentVec 0, …, momentVec (N - 1)` (`card_filter_inner_momentVec_le`), so
integrating the count gives `N c ≤ (d - 1) σ(𝕊^{d-1})` for every `N`. -/
theorem measure_greatSphere_eq_zero (σ : Measure (SSphere d)) [IsFiniteMeasure σ]
    (hσ : ∀ U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d, σ.map (sphereMap d U) = σ)
    {u : EucSpace d} (hu : u ≠ 0) : σ (greatSphere d u) = 0 := by
  obtain ⟨y₀, hy₀⟩ := exists_greatSphere_eq d hu
  rw [hy₀]
  have hd : 0 < d := by
    rcases Nat.eq_zero_or_pos d with rfl | h
    · exact absurd (PiLp.ext fun k => k.elim0) hu
    · exact h
  -- the first coordinate of `momentVec d t` is `t^0 = 1`
  have hmom : ∀ t : ℝ, momentVec d t ≠ 0 := fun t h => by
    have := congrArg (fun v : EucSpace d => v ⟨0, hd⟩) h
    simp [momentVec] at this
  have hmass : ∀ m : ℕ, σ (greatSphere d (momentVec d m)) = σ (greatSphere d y₀) := fun m => by
    obtain ⟨y, hy⟩ := exists_greatSphere_eq d (hmom m)
    rw [hy]
    exact measure_greatSphere_eq d σ hσ y y₀
  have hsum : ∀ N : ℕ,
      (N : ℝ≥0∞) * σ (greatSphere d y₀) ≤ ((d - 1 : ℕ) : ℝ≥0∞) * σ Set.univ := by
    intro N
    calc (N : ℝ≥0∞) * σ (greatSphere d y₀)
        = ∑ m ∈ Finset.range N, σ (greatSphere d (momentVec d m)) := by
          rw [Finset.sum_congr rfl fun m _ => hmass m, Finset.sum_const, Finset.card_range,
            nsmul_eq_mul]
      _ = ∫⁻ x, ∑ m ∈ Finset.range N, (greatSphere d (momentVec d m)).indicator 1 x ∂σ := by
          rw [lintegral_finsetSum _ fun m _ =>
            measurable_one.indicator (measurableSet_greatSphere d _)]
          exact Finset.sum_congr rfl fun m _ =>
            (lintegral_indicator_one (measurableSet_greatSphere d _)).symm
      _ ≤ ∫⁻ _, ((d - 1 : ℕ) : ℝ≥0∞) ∂σ := lintegral_mono fun x => ?_
      _ = ((d - 1 : ℕ) : ℝ≥0∞) * σ Set.univ := lintegral_const _
    have hx : (x : EucSpace d) ≠ 0 := by
      intro h
      have := mem_sphere_zero_iff_norm.mp x.2
      rw [h, norm_zero] at this
      exact zero_ne_one this
    classical
    simp only [Set.indicator_apply, Pi.one_apply, greatSphere, Set.mem_ofPred_eq]
    rw [Finset.sum_boole]
    exact_mod_cast card_filter_inner_momentVec_le d hx N
  by_contra hne
  obtain ⟨N, hN⟩ := ENNReal.exists_nat_mul_gt hne
    (ENNReal.mul_ne_top (ENNReal.natCast_ne_top _) (measure_ne_top σ _))
  exact absurd (hsum N) (not_le.mpr hN)

/-- The hypotheses of `measure_greatSphere_eq_zero` are satisfiable: the zero
measure is finite and invariant under every `sphereMap`, and the first basis
vector of `ℝ^1` is nonzero. -/
example : IsFiniteMeasure (0 : Measure (SSphere 1)) ∧
    (∀ U : EucSpace 1 ≃ₗᵢ[ℝ] EucSpace 1, (0 : Measure (SSphere 1)).map (sphereMap 1 U) = 0) ∧
    EuclideanSpace.single (0 : Fin 1) (1 : ℝ) ≠ 0 :=
  ⟨inferInstance, fun _ => Measure.map_zero _, by simp⟩

/-- **A proper subspace meets the sphere in a null set** under a
rotation-invariant finite measure: it lies in the great sphere of any nonzero
vector orthogonal to it. -/
theorem measure_mem_submodule_eq_zero (σ : Measure (SSphere d)) [IsFiniteMeasure σ]
    (hσ : ∀ U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d, σ.map (sphereMap d U) = σ)
    {V : Submodule ℝ (EucSpace d)} (hV : V ≠ ⊤) :
    σ {x : SSphere d | (x : EucSpace d) ∈ V} = 0 := by
  obtain ⟨u, huV, hu⟩ := (Submodule.ne_bot_iff _).1 (mt Submodule.orthogonal_eq_bot_iff.1 hV)
  exact measure_mono_null (fun x hx => Submodule.inner_right_of_mem_orthogonal hx huV)
    (measure_greatSphere_eq_zero d σ hσ hu)

/-- The hypothesis `V ≠ ⊤` of `measure_mem_submodule_eq_zero` is satisfiable:
the zero subspace of `ℝ^1`. -/
example : (⊥ : Submodule ℝ (EucSpace 1)) ≠ ⊤ := bot_ne_top

end Perspective
end Transformer
