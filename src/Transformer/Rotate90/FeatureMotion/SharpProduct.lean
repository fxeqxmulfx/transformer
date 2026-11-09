module

public import Transformer.Rotate90.FeatureMotion.Product

/-!
# Checkpoint pair amplitudes in the complex-product bound

Source: `review/flat_spectrum.md`, *Pairwise encoder amplification*, PA1.
The kernel is pointwise in logical complex pairs. Its old amplification is
the largest pair amplitude, which can be smaller than the full vector norm.
The finite mixed term still uses both unrestricted displacement radii.
-/

@[expose] public section
noncomputable section
namespace Transformer.Rotate90.FeatureMotion

open Landscape.Neural Finset
open scoped BigOperators
variable {K : Type*} [Fintype K]

/-- PA1: each logical pair's squared amplitude is part of the total energy. -/
theorem pair_energy_le_norm_sq (x : EuclideanSpace ℝ (Channels K)) (k : K) :
    x (k, 0)^2 + x (k, 1)^2 ≤ ‖x‖^2 := by
  rw [EuclideanSpace.real_norm_sq_eq, Fintype.sum_prod_type]
  simp only [Fin.sum_univ_two]
  exact single_le_sum (f := fun j => x (j, 0)^2 + x (j, 1)^2)
    (fun j _ => by positivity) (mem_univ k)

/-- PA1: the exact squared norm of the calibrated source product. -/
theorem product_norm_sq (g u : EuclideanSpace ℝ (Channels K)) :
    ‖pairedProduct g u‖^2 =
      (∑ k, (g (k, 0)^2 + g (k, 1)^2) * (u (k, 0)^2 + u (k, 1)^2)) / 2 := by
  have hs2 : (Real.sqrt 2)^2 = 2 := Real.sq_sqrt (by norm_num)
  rw [EuclideanSpace.real_norm_sq_eq, Fintype.sum_prod_type, sum_div]
  apply sum_congr rfl
  intro k hk
  simp only [Fin.sum_univ_two, pairedProduct, ComplexProduct.realProduct,
    Fin.isValue, ↓reduceIte, Fin.reduceEq]
  change ((g (k, 0)*u (k, 0)-g (k, 1)*u (k, 1))/Real.sqrt 2)^2 +
    ((g (k, 0)*u (k, 1)+g (k, 1)*u (k, 0))/Real.sqrt 2)^2 = _
  rw [div_pow, div_pow, hs2]
  ring

omit [Fintype K] in
/-- PA1: commutativity of the actual complex product, including calibration. -/
theorem pairedProduct_comm (g u : EuclideanSpace ℝ (Channels K)) :
    pairedProduct g u = pairedProduct u g := by
  ext ⟨k, j⟩
  fin_cases j <;> simp [pairedProduct, ComplexProduct.realProduct] <;> ring

/-- PA1: a bound on each old value pair replaces its full vector norm. -/
theorem product_norm_le_pair_right (g u : EuclideanSpace ℝ (Channels K)) (U : ℝ)
    (hU : 0 ≤ U) (hu : ∀ k, u (k, 0)^2 + u (k, 1)^2 ≤ U^2) :
    ‖pairedProduct g u‖ ≤ ‖g‖ * U / Real.sqrt 2 := by
  have hs : 0 < Real.sqrt 2 := Real.sqrt_pos.mpr (by norm_num)
  have hs2 : (Real.sqrt 2)^2 = 2 := Real.sq_sqrt (by norm_num)
  have hg : ‖g‖^2 = ∑ k, (g (k, 0)^2 + g (k, 1)^2) := by
    rw [EuclideanSpace.real_norm_sq_eq, Fintype.sum_prod_type]
    simp only [Fin.sum_univ_two]
  have hsum : (∑ k, (g (k, 0)^2 + g (k, 1)^2) *
      (u (k, 0)^2 + u (k, 1)^2)) ≤ ‖g‖^2 * U^2 := by
    rw [hg, sum_mul]
    exact sum_le_sum (fun k _ => mul_le_mul_of_nonneg_left (hu k) (by positivity))
  have hp := product_norm_sq g u
  apply (le_div_iff₀ hs).mpr
  nlinarith [norm_nonneg (pairedProduct g u), norm_nonneg g,
    mul_nonneg (norm_nonneg g) hU,
    mul_nonneg (norm_nonneg (pairedProduct g u)) hs.le]

/-- PA1: the symmetric old-gate bound. -/
theorem product_norm_le_pair_left (g u : EuclideanSpace ℝ (Channels K)) (G : ℝ)
    (hG : 0 ≤ G) (hg : ∀ k, g (k, 0)^2 + g (k, 1)^2 ≤ G^2) :
    ‖pairedProduct g u‖ ≤ G * ‖u‖ / Real.sqrt 2 := by
  rw [pairedProduct_comm]
  simpa only [mul_comm] using product_norm_le_pair_right u g G hG hg

/-- PA1: full finite product motion using only old pair amplitudes and
the two moving-input radii. The mixed product remains unrestricted. -/
theorem product_motion_pair (g u g' u' : EuclideanSpace ℝ (Channels K))
    (G U rg ru : ℝ) (hG : 0 ≤ G) (hU : 0 ≤ U)
    (hgp : ∀ k, g (k, 0)^2 + g (k, 1)^2 ≤ G^2)
    (hup : ∀ k, u (k, 0)^2 + u (k, 1)^2 ≤ U^2)
    (hg : ‖g' - g‖ ≤ rg) (hu : ‖u' - u‖ ≤ ru) :
    ‖pairedProduct g' u' - pairedProduct g u‖ ≤
      (rg * U + G * ru + rg * ru) / Real.sqrt 2 := by
  have hrg := (norm_nonneg (g' - g)).trans hg
  have h1 := (product_norm_le_pair_right (g' - g) u U hU hup).trans
    (div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right hg hU)
      (Real.sqrt_nonneg _))
  have h2 := (product_norm_le_pair_left g (u' - u) G hG hgp).trans
    (div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hu hG)
      (Real.sqrt_nonneg _))
  have h3 := (product_norm_le (g' - g) (u' - u)).trans
    (div_le_div_of_nonneg_right (mul_le_mul hg hu (norm_nonneg _) hrg) (Real.sqrt_nonneg _))
  rw [product_difference]
  have h := (norm_add_le (pairedProduct (g' - g) u + pairedProduct g (u' - u))
    (pairedProduct (g' - g) (u' - u))).trans (add_le_add (norm_add_le _ _) le_rfl)
  rw [add_div, add_div]
  linarith

/-- Two nonzero logical pairs, each of amplitude one. -/
def pairWitness : EuclideanSpace ℝ (Channels (Fin 2)) :=
  WithLp.toLp 2 (fun j => if j.2 = 0 then (1 : ℝ) else 0)

example := product_norm_le_pair_right pairWitness pairWitness 1
  (by norm_num) (by intro k; simp [pairWitness])
example := product_norm_le_pair_left pairWitness pairWitness 1
  (by norm_num) (by intro k; simp [pairWitness])
example := product_motion_pair pairWitness pairWitness (pairWitness + pairWitness)
  (pairWitness + pairWitness) 1 1 ‖pairWitness‖ ‖pairWitness‖
  (by norm_num) (by norm_num) (by intro k; simp [pairWitness])
  (by intro k; simp [pairWitness]) (by simp) (by simp)

end Transformer.Rotate90.FeatureMotion
