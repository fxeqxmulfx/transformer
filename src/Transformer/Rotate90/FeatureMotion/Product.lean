module

public import Transformer.Rotate90.FeatureMotion.Basic
public import Transformer.Rotate90.ComplexProduct.Basic

/-!
# Motion through the source quarter-turn and complex product

Source: `review/flat_spectrum.md`, *Full encoder motion*, EM1 and EM3.
The hidden product has both scalar cross terms and its sqrt(2) divisor.
The preceding and following matrices need not be complex-linear.
-/

@[expose] public section
noncomputable section
namespace Transformer.Rotate90.FeatureMotion

open Landscape.Neural Finset
open scoped BigOperators
variable {K : Type*} [Fintype K]

/-- EM1: the real source quarter-turn on logical pairs. -/
def turn (x : EuclideanSpace ℝ (Channels K)) : EuclideanSpace ℝ (Channels K) :=
  WithLp.toLp 2 (rotate90 x)

/-- EM1: a quarter-turn preserves Euclidean length exactly. -/
theorem turn_norm (x : EuclideanSpace ℝ (Channels K)) : ‖turn x‖ = ‖x‖ := by
  have h : ‖turn x‖ ^ 2 = ‖x‖ ^ 2 := by
    simp only [EuclideanSpace.real_norm_sq_eq, Channels, Fintype.sum_prod_type,
      Fin.sum_univ_two, turn, rotate90, Fin.isValue,
      ↓reduceIte, Fin.reduceEq, neg_sq]
    apply sum_congr rfl
    intro k hk
    ring
  nlinarith [norm_nonneg (turn x), norm_nonneg x]

/-- EM1: both moving embedding rows contribute, including repeated inputs. -/
theorem input_motion (a b a' b' : EuclideanSpace ℝ (Channels K)) (ra rb : ℝ)
    (ha : ‖a' - a‖ ≤ ra) (hb : ‖b' - b‖ ≤ rb) :
    ‖(a' + turn b') - (a + turn b)‖ ≤ ra + rb := by
  have he : (a' + turn b') - (a + turn b) = (a' - a) + turn (b' - b) := by
    ext ⟨k, j⟩
    fin_cases j <;> simp [turn, rotate90] <;> ring
  rw [he]
  exact (norm_add_le _ _).trans (add_le_add ha (by simpa only [turn_norm] using hb))

/-- EM3: the independent real four-product kernel, in Euclidean coordinates. -/
def pairedProduct (g u : EuclideanSpace ℝ (Channels K)) : EuclideanSpace ℝ (Channels K) :=
  WithLp.toLp 2 (ComplexProduct.realProduct g u)

omit [Fintype K] in
/-- EM3: finite product motion retains the quadratic mixed product. -/
theorem product_difference (g u g' u' : EuclideanSpace ℝ (Channels K)) :
    pairedProduct g' u' - pairedProduct g u =
      pairedProduct (g' - g) u + pairedProduct g (u' - u) +
      pairedProduct (g' - g) (u' - u) := by
  ext ⟨k, j⟩
  fin_cases j <;> simp [pairedProduct, ComplexProduct.realProduct] <;> ring

/-- EM3: summing the scalar complex-modulus identity gives an L2 product
bound with the exact calibration; it does not assume coordinate independence. -/
theorem product_norm_le (g u : EuclideanSpace ℝ (Channels K)) :
    ‖pairedProduct g u‖ ≤ ‖g‖ * ‖u‖ / Real.sqrt 2 := by
  have hs : 0 < Real.sqrt 2 := Real.sqrt_pos.mpr (by norm_num)
  have hs2 : (Real.sqrt 2)^2 = 2 := Real.sq_sqrt (by norm_num)
  have hg : ‖g‖ ^ 2 = ∑ k, (g (k, 0)^2 + g (k, 1)^2) := by
    rw [EuclideanSpace.real_norm_sq_eq, Fintype.sum_prod_type]
    simp only [Fin.sum_univ_two]
  have hu : ‖u‖ ^ 2 = ∑ k, (u (k, 0)^2 + u (k, 1)^2) := by
    rw [EuclideanSpace.real_norm_sq_eq, Fintype.sum_prod_type]
    simp only [Fin.sum_univ_two]
  have hp : ‖pairedProduct g u‖ ^ 2 =
      (∑ k, (g (k, 0)^2 + g (k, 1)^2) * (u (k, 0)^2 + u (k, 1)^2)) / 2 := by
    rw [EuclideanSpace.real_norm_sq_eq, Fintype.sum_prod_type, sum_div]
    apply sum_congr rfl
    intro k hk
    simp only [Fin.sum_univ_two, pairedProduct, ComplexProduct.realProduct,
      Fin.isValue, ↓reduceIte, Fin.reduceEq]
    change ((g (k, 0)*u (k, 0)-g (k, 1)*u (k, 1))/Real.sqrt 2)^2 +
      ((g (k, 0)*u (k, 1)+g (k, 1)*u (k, 0))/Real.sqrt 2)^2 = _
    rw [div_pow, div_pow, hs2]
    ring
  have hb : ∀ k, u (k, 0)^2 + u (k, 1)^2 ≤ ‖u‖^2 := by
    intro k
    rw [hu]
    exact single_le_sum (f := fun j => u (j, 0)^2 + u (j, 1)^2)
      (fun j _ => by positivity) (mem_univ k)
  have hsum : (∑ k, (g (k, 0)^2 + g (k, 1)^2) *
      (u (k, 0)^2 + u (k, 1)^2)) ≤ ‖g‖^2 * ‖u‖^2 := by
    rw [hg, sum_mul]
    exact sum_le_sum (fun k _ => mul_le_mul_of_nonneg_left (hb k) (by positivity))
  apply (le_div_iff₀ hs).mpr
  nlinarith [norm_nonneg (pairedProduct g u), norm_nonneg g, norm_nonneg u,
    mul_nonneg (norm_nonneg g) (norm_nonneg u),
    mul_nonneg (norm_nonneg (pairedProduct g u)) hs.le]

/-- EM3: a product radius uses only the old gate/value and their input
radii. In particular the gate/value mixed radius cannot be discarded. -/
theorem product_motion (g u g' u' : EuclideanSpace ℝ (Channels K)) (rg ru : ℝ)
    (hg : ‖g' - g‖ ≤ rg) (hu : ‖u' - u‖ ≤ ru) :
    ‖pairedProduct g' u' - pairedProduct g u‖ ≤
      (rg * ‖u‖ + ‖g‖ * ru + rg * ru) / Real.sqrt 2 := by
  have hrg := (norm_nonneg (g' - g)).trans hg
  have h1 := (product_norm_le (g' - g) u).trans
    (div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right hg (norm_nonneg u))
      (Real.sqrt_nonneg _))
  have h2 := (product_norm_le g (u' - u)).trans
    (div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hu (norm_nonneg g))
      (Real.sqrt_nonneg _))
  have h3 := (product_norm_le (g' - g) (u' - u)).trans
    (div_le_div_of_nonneg_right (mul_le_mul hg hu (norm_nonneg _) hrg) (Real.sqrt_nonneg _))
  rw [product_difference]
  have h := (norm_add_le (pairedProduct (g' - g) u + pairedProduct g (u' - u))
    (pairedProduct (g' - g) (u' - u))).trans (add_le_add (norm_add_le _ _) le_rfl)
  rw [add_div, add_div]
  linarith

example : let v := WithLp.toLp 2 (fun j : Channels (Fin 1) => if j.2 = 0 then (1 : ℝ) else 0)
    ‖(v + turn v) - (0 + turn 0)‖ ≤ ‖v‖ + ‖v‖ := by
  dsimp
  exact input_motion 0 0 _ _ _ _ (by simp) (by simp)
example : let v := WithLp.toLp 2 (fun j : Channels (Fin 1) => if j.2 = 0 then (1 : ℝ) else 0)
    ‖pairedProduct v v - pairedProduct 0 0‖ ≤ ‖v‖ * ‖v‖ / Real.sqrt 2 := by
  dsimp
  simpa using product_motion 0 0
    (WithLp.toLp 2 (fun j : Channels (Fin 1) => if j.2 = 0 then (1 : ℝ) else 0))
    (WithLp.toLp 2 (fun j : Channels (Fin 1) => if j.2 = 0 then (1 : ℝ) else 0))
    _ _ (by simp) (by simp)

end Transformer.Rotate90.FeatureMotion
