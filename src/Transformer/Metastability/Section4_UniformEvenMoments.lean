/-
# Exact even-moment recurrence for the original uniform sphere law

ArXiv:2410.06833v1, §4, `prop: concentration unif`, and
arXiv:2312.10794v5, Appendix D, Step 2, almost orthogonality.
The original uniform linear-coordinate moments supply the scalar
tail estimate needed for projecting a uniform sample onto a
previously selected finite-dimensional subspace.

The actual mixed-moment identity from uniform rotations is combined
with the actual unit-sphere identity `Σ xᵢ²=1`. Summing over the
original coordinate axes gives `(d+2k) M_(k+1)=(2k+1) M_k`,
with `M_0=1`. Orthogonal transitivity proves that the same moments
hold for every original unit direction. All laws, coordinates,
indices and factors remain actual integrals of the original sample.
-/

import Transformer.Metastability.Section4_UniformPowerIdentity

open Real MeasureTheory
open scoped BigOperators

namespace Transformer.Metastability

/-- The original even moment in its original ambient direction.
Source: arXiv:2410.06833v1, §4, the uniform projection concentration
input for `prop: concentration unif`. The index is half the degree. -/
noncomputable def uniformEvenMoment {d : ℕ} (ν : Measure (SSphere d)) (u : EucSpace d) (k : ℕ) : ℝ :=
  ∫ x : SSphere d, inner (𝕜 := ℝ) u (x : EucSpace d) ^ (2 * k) ∂ν

/-- The original probability law has zeroth moment one.
Source: arXiv:2410.06833v1, §4, the normalization of uniform initialization. -/
@[simp] theorem uniformEvenMoment_zero {d : ℕ} (ν : Measure (SSphere d)) [IsProbabilityMeasure ν]
    (u : EucSpace d) : uniformEvenMoment ν u 0 = 1 := by
  simp [uniformEvenMoment]

/-- Every original even coordinate moment is nonnegative.
Source: arXiv:2410.06833v1, §4, the original nonnegative moment bounds. -/
theorem uniformEvenMoment_nonneg {d : ℕ} (ν : Measure (SSphere d)) (u : EucSpace d) (k : ℕ) :
    0 ≤ uniformEvenMoment ν u k :=
  integral_nonneg fun x => (even_two_mul k).pow_nonneg (inner (𝕜 := ℝ) u (x : EucSpace d))

/-- An actual orthogonal map preserves the original even directional
moment. Source: arXiv:2410.06833v1, §4, rotational invariance of
the original initialization used for `prop: concentration unif`. -/
theorem uniformEvenMoment_isometry {d : ℕ} (ν : Measure (SSphere d)) (hν : IsUniformOn d ν)
    (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) (u : EucSpace d) (k : ℕ) :
    uniformEvenMoment ν (U u) k = uniformEvenMoment ν u k := by
  have hf : Continuous (fun x : SSphere d => inner (𝕜 := ℝ) (U u) (x : EucSpace d) ^ (2 * k)) :=
    (continuous_const.inner continuous_subtype_val).pow (2 * k)
  have h := integral_map (μ := ν) (Perspective.measurable_sphereMap d U).aemeasurable
    hf.aestronglyMeasurable
  rw [hν.2 U] at h
  simpa only [uniformEvenMoment, Perspective.sphereMap, LinearIsometryEquiv.inner_map_map] using h

/-- Every actual unit direction has the same original even moments.
Source: arXiv:2410.06833v1, §4, the directional concentration estimate
for the original uniform law. The actual orthogonal transport is constructed. -/
theorem uniformEvenMoment_eq_of_unit {d : ℕ} (ν : Measure (SSphere d)) (hν : IsUniformOn d ν)
    {u v : EucSpace d} (hu : ‖u‖ = 1) (hv : ‖v‖ = 1) (k : ℕ) :
    uniformEvenMoment ν u k = uniformEvenMoment ν v k := by
  obtain ⟨U, hU⟩ := Perspective.exists_sphereMap_apply_eq d
    ⟨u, mem_sphere_zero_iff_norm.mpr hu⟩ ⟨v, mem_sphere_zero_iff_norm.mpr hv⟩
  have huv : U u = v := congrArg Subtype.val hU
  have h := uniformEvenMoment_isometry ν hν U u k
  rw [huv] at h
  exact h.symm

/-- Summing the original mixed-coordinate moments gives the exact even
moment recurrence. Source: arXiv:2410.06833v1, §4, the uniform
projection concentration estimate underlying `prop: concentration unif`.
The sphere identity and the actual differentiated rotation integral supply
all inputs. Positive dimension suffices, including the two-point sphere. -/
theorem uniformEvenMoment_recurrence {d : ℕ} (hd : 0 < d)
    (ν : Measure (SSphere d)) (hν : IsUniformOn d ν) {u : EucSpace d}
    (hu : ‖u‖ = 1) (k : ℕ) :
    ((d : ℝ) + 2 * k) * uniformEvenMoment ν u (k + 1) =
      (2 * (k : ℝ) + 1) * uniformEvenMoment ν u k := by
  classical
  let : IsProbabilityMeasure ν := hν.1
  let e : Fin d := ⟨0, hd⟩
  let a : EucSpace d := EuclideanSpace.single e 1
  have ha : ‖a‖ = 1 := by simp [a]
  have heq (j : ℕ) : uniformEvenMoment ν u j = uniformEvenMoment ν a j :=
    uniformEvenMoment_eq_of_unit ν hν hu ha j
  rw [heq (k + 1), heq k]
  let f : Fin d → SSphere d → ℝ := fun i x => inner (𝕜 := ℝ) a (x : EucSpace d) ^ (2 * k) *
    (x : EucSpace d) i ^ 2
  have hf (i : Fin d) : Integrable (f i) ν := by
    have hc : Continuous (f i) := by
      change Continuous (fun x : SSphere d => inner (𝕜 := ℝ) a (x : EucSpace d) ^ (2 * k) *
        (x : EucSpace d) i ^ 2)
      fun_prop
    exact hc.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
  have hsum : (∑ i : Fin d, ∫ x, f i x ∂ν) = uniformEvenMoment ν a k := by
    rw [← integral_finsetSum _ (fun i _ => hf i)]
    apply integral_congr_ae
    filter_upwards with x
    dsimp only [f]
    rw [← Finset.mul_sum, ← EuclideanSpace.real_norm_sq_eq, norm_eq_of_mem_sphere x,
      one_pow, mul_one]
  have hterm (i : Fin d) : (2 * (k : ℝ) + 1) * (∫ x, f i x ∂ν) =
      uniformEvenMoment ν a (k + 1) +
        if i = e then 2 * (k : ℝ) * uniformEvenMoment ν a (k + 1) else 0 := by
    by_cases hi : i = e
    · rw [ite_eq_left hi]
      have hdiag : (∫ x, f i x ∂ν) = uniformEvenMoment ν a (k + 1) := by
        apply integral_congr_ae
        filter_upwards with x
        simp only [f, a, hi, EuclideanSpace.inner_single_left, map_one, one_mul]
        rw [← pow_add]
        exact congrArg (fun m : ℕ => (x : EucSpace d) e ^ m) (by omega)
      rw [hdiag]
      ring
    · rw [ite_eq_right hi, add_zero]
      have hb : ‖(EuclideanSpace.single i 1 : EucSpace d)‖ = 1 := by simp
      have hab : inner (𝕜 := ℝ) a (EuclideanSpace.single i 1 : EucSpace d) = 0 := by
        simp [a, EuclideanSpace.inner_single_left, Ne.symm hi]
      have h := integral_uniform_power_eq_mixed ν hν ha hb hab (2 * (k + 1)) (by omega)
      have h₁ : 2 * (k + 1) - 1 = 2 * k + 1 := by omega
      have h₂ : 2 * (k + 1) - 2 = 2 * k := by omega
      simpa [uniformEvenMoment, f, h₁, h₂, EuclideanSpace.inner_single_left] using h.symm
  have hrec : (2 * (k : ℝ) + 1) * uniformEvenMoment ν a k =
      ((d : ℝ) + 2 * k) * uniformEvenMoment ν a (k + 1) := by
    rw [← hsum, Finset.mul_sum]
    simp_rw [hterm]
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
      nsmul_eq_mul]
    simp only [Finset.sum_ite_eq', Finset.mem_univ, ite_true]
    ring
  exact hrec.symm

/-- The original uniform directional variance is exactly `1/d`.
Source: arXiv:2410.06833v1, §4, the second-moment normalization
in the original uniform projection estimate. It follows from the
proved exact recurrence at its original zero half-degree. -/
theorem uniformEvenMoment_one {d : ℕ} (hd : 0 < d)
    (ν : Measure (SSphere d)) (hν : IsUniformOn d ν) {u : EucSpace d}
    (hu : ‖u‖ = 1) : uniformEvenMoment ν u 1 = 1 / (d : ℝ) := by
  let : IsProbabilityMeasure ν := hν.1
  have h := uniformEvenMoment_recurrence hd ν hν hu 0
  have hdpos : 0 < (d : ℝ) := by exact_mod_cast hd
  apply (eq_div_iff hdpos.ne').mpr
  simpa [mul_comm] using h

/-- The original uniform circle and a nonzero original unit coordinate
jointly witness positive dimension, actual probability, uniformity and
unit-direction hypotheses of the exact recurrence. -/
example : (0 : ℕ) < 2 ∧ IsUniformOn 2 (MeanField.uniformLaw 2) ∧
    ‖(EuclideanSpace.single (0 : Fin 2) 1 : EucSpace 2)‖ = 1 :=
  ⟨by decide, isUniformOn_uniformLaw (by norm_num), by simp⟩

end Transformer.Metastability
