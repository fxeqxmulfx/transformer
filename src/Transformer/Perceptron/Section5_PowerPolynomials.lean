/-
# Original homogeneous kernel powers preserve their polynomial degree

ArXiv:2601.21366v2, §5.1, `eq:FH-basic` and `lem: quadpol` (i).
The actual even-moment expansion of an original homogeneous power test
has coefficients computed at one original orthonormal pair. The proved
two-reflection identity makes these coefficients independent of the
original kernel point and its chosen perpendicular vector.

Each actual even term has degree at most its original power: its radial
factor has degree `n-m`, and its squared perpendicular factor has degree
`2(m/2)=m`. The polynomial witnesses and actual degree inequalities are
proved below. Summation therefore preserves the original degree bound.

This establishes degree preservation for every original homogeneous
power and every actual finite weighted family of them. It retains the
original kernel, uniform law, test vectors, coefficients and exponents.
The original polynomial characterization still requires their spanning
and the actual finite-dimensional inverse; neither is assumed here.
-/

import Transformer.Perceptron.Section5_PowerMoments
import Transformer.Perceptron.Section5_QuadraticPolynomials

open Real MeasureTheory Set
open scoped BigOperators

namespace Transformer.Perceptron

/-- The actual finite expression with original moments at a fixed
original pair. Source: arXiv:2601.21366v2, §5.1, `eq:FH-basic`.
Its identity with the actual kernel integral and degree are proved below. -/
noncomputable def powerMomentExpansion {d : ℕ} (β : ℝ) (ν : Measure (SSphere d))
    (o v b : EucSpace d) (n : ℕ) (x : SSphere d) : ℝ :=
  ∑ m ∈ Finset.range (n + 1), if Even m then
    ((n.choose m : ℝ) * radialPerpMomentKernel β ν o v (n - m) m) *
      (inner (𝕜 := ℝ) b (x : EucSpace d) ^ (n - m) *
        (‖b‖ ^ 2 - inner (𝕜 := ℝ) b (x : EucSpace d) ^ 2) ^ (m / 2)) else 0

/-- The actual finite moment expression is an original spherical
polynomial of degree at most its original homogeneous power.
Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` (i).
Evenness gives the actual equality `2(m/2)=m`, retaining the original
degree rather than bounding every term by twice the original power. -/
theorem isPolyOfDegreeLE_powerMomentExpansion {d : ℕ} (β : ℝ) (ν : Measure (SSphere d))
    (o v b : EucSpace d) (n : ℕ) : IsPolyOfDegreeLE n (powerMomentExpansion β ν o v b n) := by
  have hc : IsPolyOfDegreeLE 2 (fun _ : SSphere d => ‖b‖ ^ 2) :=
    isPolyOfDegreeLE_mono (isPolyOfDegreeLE_const _) (by omega)
  have hs : IsPolyOfDegreeLE 2 (fun x : SSphere d => ‖b‖ ^ 2 - inner (𝕜 := ℝ) b (x : EucSpace d) ^ 2) := by
    have h := isPolyOfDegreeLE_add hc
      (isPolyOfDegreeLE_const_mul (-1 : ℝ) (isPolyOfDegreeLE_pow (isPolyOfDegreeLE_inner b) 2))
    simpa only [neg_one_mul, ← sub_eq_add_neg] using h
  apply isPolyOfDegreeLE_finsetSum
  intro m hm
  have hmn : m ≤ n := by have := Finset.mem_range.mp hm; omega
  by_cases heven : Even m
  · simp only [ite_eq_left heven]
    have hprod := isPolyOfDegreeLE_mul
      (isPolyOfDegreeLE_pow (isPolyOfDegreeLE_inner b) (n - m))
      (isPolyOfDegreeLE_pow hs (m / 2))
    have hdeg : (n - m) * 1 + m / 2 * 2 ≤ n := by
      have := Nat.two_mul_div_two_of_even heven
      omega
    exact isPolyOfDegreeLE_mono
      (isPolyOfDegreeLE_const_mul ((n.choose m : ℝ) * radialPerpMomentKernel β ν o v (n - m) m) hprod) hdeg
  · simp only [ite_eq_right heven]
    exact isPolyOfDegreeLE_mono (isPolyOfDegreeLE_const 0) (Nat.zero_le n)

/-- The actual original power integral equals the actual polynomial
moment expression at every original sphere point.
Source: arXiv:2601.21366v2, §5.1, `eq:FH-basic` and `lem: quadpol` (i).
The original perpendicular vector is supplied geometrically; transporting
its actual moments to the fixed original pair gives the coefficients. -/
theorem powerMomentKernel_eq_expansion {d : ℕ} (hd : 2 ≤ d) (β : ℝ)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν) (o v : SSphere d)
    (hov : inner (𝕜 := ℝ) (o : EucSpace d) (v : EucSpace d) = 0) (b : EucSpace d) (n : ℕ) (x : SSphere d) :
    powerMomentKernel β ν (x : EucSpace d) b n = powerMomentExpansion β ν (o : EucSpace d) (v : EucSpace d) b n x := by
  obtain ⟨u, hxu, hb⟩ := exists_unit_perp_decomposition hd x b
  rw [powerMomentKernel_even_expansion β ν hν x u b n hxu hb]
  unfold powerMomentExpansion
  apply Finset.sum_congr rfl
  intro m _
  by_cases hm : Even m
  · rw [ite_eq_left hm, ite_eq_left hm,
      radialPerpMomentKernel_eq_of_orthogonal_pairs β ν hν (x : EucSpace d) (u : EucSpace d)
        (o : EucSpace d) (v : EucSpace d) (n - m) m
        ((norm_eq_of_mem_sphere x).trans (norm_eq_of_mem_sphere o).symm)
        ((norm_eq_of_mem_sphere u).trans (norm_eq_of_mem_sphere v).symm) hxu hov,
      real_inner_comm (x : EucSpace d) b]
    ring
  · rw [ite_eq_right hm, ite_eq_right hm]

/-- The original exponential transform of every actual homogeneous
power test preserves its original degree bound.
Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` (i).
An actual original coordinate axis and a proved perpendicular sphere
point fix the actual coefficients; no spectral-preservation input is used. -/
theorem isPolyOfDegreeLE_powerMomentKernel {d : ℕ} (hd : 2 ≤ d) (β : ℝ)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν) (b : EucSpace d) (n : ℕ) :
    IsPolyOfDegreeLE n (fun x : SSphere d => powerMomentKernel β ν (x : EucSpace d) b n) := by
  let o : SSphere d := ⟨EuclideanSpace.single (⟨0, by omega⟩ : Fin d) (1 : ℝ), by simp⟩
  obtain ⟨v, hov⟩ := exists_unit_perp_sphere hd o
  have heq : (fun x : SSphere d => powerMomentKernel β ν (x : EucSpace d) b n) =
      powerMomentExpansion β ν (o : EucSpace d) (v : EucSpace d) b n :=
    funext (powerMomentKernel_eq_expansion hd β ν hν o v hov b n)
  rw [heq]
  exact isPolyOfDegreeLE_powerMomentExpansion β ν (o : EucSpace d) (v : EucSpace d) b n

/-- Every original finite weighted family of homogeneous power tests
preserves its common original polynomial degree under the actual kernel.
Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` (i).
Actual continuous integrability proves the sum of original integrals;
the polynomial witnesses above give the resulting actual degree bound. -/
theorem isPolyOfDegreeLE_integral_exp_mul_finset_powers {d k : ℕ} {ι : Type*}
    (hd : 2 ≤ d) (β : ℝ) (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν)
    (S : Finset ι) (w : ι → ℝ) (b : ι → EucSpace d) (n : ι → ℕ) (hn : ∀ i ∈ S, n i ≤ k) :
    IsPolyOfDegreeLE k (fun x : SSphere d => ∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) *
      (∑ i ∈ S, w i * inner (𝕜 := ℝ) (b i) (y : EucSpace d) ^ n i) ∂ν) := by
  let _ : IsProbabilityMeasure ν := hν.1
  have heq : (fun x : SSphere d => ∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) *
      (∑ i ∈ S, w i * inner (𝕜 := ℝ) (b i) (y : EucSpace d) ^ n i) ∂ν) =
      fun x : SSphere d => ∑ i ∈ S, w i * powerMomentKernel β ν (x : EucSpace d) (b i) (n i) := by
    funext x
    have hK : Continuous (fun y : SSphere d => exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d))) :=
      Real.continuous_exp.comp (continuous_const.mul (continuous_const.inner continuous_subtype_val))
    have hint (i : ι) : Integrable (fun y : SSphere d =>
        exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) *
          (w i * inner (𝕜 := ℝ) (b i) (y : EucSpace d) ^ n i)) ν :=
      Perspective.integrable_of_continuous_compact
        (hK.mul (continuous_const.mul ((continuous_const.inner continuous_subtype_val).pow (n i)))) ν
    simp_rw [Finset.mul_sum]
    rw [integral_finsetSum S (fun i _ => hint i)]
    apply Finset.sum_congr rfl
    intro i _
    unfold powerMomentKernel
    rw [← integral_const_mul]
    apply integral_congr_ae
    exact ae_of_all ν fun y => by ring
  rw [heq]
  exact isPolyOfDegreeLE_finsetSum S _ fun i hi => isPolyOfDegreeLE_mono
    (isPolyOfDegreeLE_const_mul (w i) (isPolyOfDegreeLE_powerMomentKernel hd β ν hν (b i) (n i))) (hn i hi)

/-- Every dimensional, uniform, fixed-pair and degree hypothesis holds
jointly on the actual original circle with actual quadratic coordinate
powers. Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` (i).
The actual second axis supplies the original fixed perpendicular point. -/
example : (2 : ℕ) ≤ 2 ∧ Metastability.IsUniformOn 2 (MeanField.uniformLaw 2) ∧
    (∃ v : SSphere 2, inner (𝕜 := ℝ) (basePoint 1 : EucSpace 2) (v : EucSpace 2) = 0) ∧
    ∀ i ∈ (Finset.univ : Finset (Fin 2)), (fun _ : Fin 2 => (2 : ℕ)) i ≤ 2 :=
  ⟨le_rfl, Metastability.isUniformOn_uniformLaw (by norm_num),
    ⟨⟨secondAxis, mem_sphere_zero_iff_norm.mpr norm_secondAxis⟩, inner_basePoint_secondAxis⟩,
    fun _ _ => le_rfl⟩

end Transformer.Perceptron
