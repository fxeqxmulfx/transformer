module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.SharedAtoms.Scaling.Queries
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.SharedAtoms.Witness

/-!
# Dimension-based bias scaling retains every learned coordinate

Source: `review/flat_spectrum.md`, CM93. Scale only each always-present bias
address by 1/sqrt(V), leaving every original object, position and pair address
at scale one. This changes the capacity geometry without deleting directions.
The raw class-bias response is then 1/V, which the query reservation must use.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.SharedAtoms.Scaling

open scoped BigOperators
open Identification

variable {O : Type*} [Fintype O] {T : ℕ}

/-- Full coordinate contraction cannot increase complete energy, CM93. -/
theorem energy_le (w r : RealCoordinate O T → ℝ) (hw : ∀ z, w z ^ 2 ≤ 1) :
    Capacity.energy (Updates.diagonal w r) ≤ Capacity.energy r := by
  apply Finset.sum_le_sum
  intro z hz
  change (w z * r z) ^ 2 ≤ r z ^ 2
  rw [mul_pow]
  simpa using mul_le_mul_of_nonneg_right (hw z) (sq_nonneg (r z))

/-- A contracting scaled atomic domain retains the CM91 energy bound,
CM93. No converse inclusion of capacity domains is presumed. -/
theorem domain_energy (a : QueryCoordinate O → ℝ) (b : ContextCoordinate O T → ℝ)
    (B : ℝ) (r : RealCoordinate O T → ℝ) (hw : ∀ z, weight a b z ^ 2 ≤ 1)
    (hr : r ∈ domain a b B) : Capacity.energy r ≤ B ^ 2 := by
  obtain ⟨s, hs, rfl⟩ := hr
  exact (energy_le _ s hw).trans (SharedAtoms.domain_energy B s hs)

/-- The dimension-only scale does not inspect object IDs or labels, CM93. -/
noncomputable def dimension (n : ℕ) : ℝ := (Real.sqrt n)⁻¹

/-- Positive original vocabulary size gives an invertible contraction,
CM93. A zero vocabulary is excluded by the native object interface. -/
theorem dimension_bounds (n : ℕ) (hn : 0 < n) : 0 < dimension n ∧ dimension n ≤ 1 := by
  have hp : 0 < (n : ℝ) := by exact_mod_cast hn
  have h1 : 1 ≤ (n : ℝ) := by exact_mod_cast hn
  exact ⟨inv_pos.mpr (Real.sqrt_pos.mpr hp), inv_le_one_of_one_le₀ (Real.one_le_sqrt.mpr h1)⟩

/-- The joint class-bias response is exactly reciprocal vocabulary size,
CM93. This controls finite-query cost as well as learned capacity. -/
theorem dimension_squared (n : ℕ) : dimension n ^ 2 = (n : ℝ)⁻¹ := by
  rw [dimension, inv_pow, Real.sq_sqrt (Nat.cast_nonneg n)]

/-- Bias is the unique `none` address; all original object/position/pair
addresses remain at positive scale one, CM93. -/
def biasScale {A : Type*} (s : ℝ) : Option A → ℝ
  | none => s
  | some _ => 1

/-- No coordinate direction is removed by positive bias scaling, CM93. -/
theorem biasScale_bounds {A : Type*} (s : ℝ) (hs : 0 < s) (hs1 : s ≤ 1) (a : Option A) :
    0 < biasScale s a ∧ biasScale s a ≤ 1 := by
  cases a
  · exact ⟨hs, hs1⟩
  · norm_num [biasScale]

omit [Fintype O] in
/-- The full joint bias geometry is positive and contracting on every
class and both complex components, CM93. -/
theorem bias_weight_bounds (s : ℝ) (hs : 0 < s) (hs1 : s ≤ 1) (z : RealCoordinate O T) :
    0 < weight (biasScale s) (biasScale s) z ∧ weight (biasScale s) (biasScale s) z ^ 2 ≤ 1 := by
  have hq := biasScale_bounds s hs hs1 z.2.2.1
  have hd := biasScale_bounds s hs hs1 z.2.2.2
  have hp := mul_pos hq.1 hd.1
  have hh : biasScale s z.2.2.1 * biasScale s z.2.2.2 ≤ 1 := by
    have h := mul_le_mul_of_nonneg_right hq.2 hd.1.le
    linarith [hd.2]
  exact ⟨hp, by unfold weight; nlinarith⟩

omit [Fintype O] in
/-- Native dimension scaling has the corrected class-query response,
CM93, for every original output object. -/
theorem bias_response (n : ℕ) (c : O) :
    weight (biasScale (dimension n)) (biasScale (dimension n))
      (Updates.biasCoordinate c : RealCoordinate O T) = (n : ℝ)⁻¹ := by
  change dimension n * dimension n = (n : ℝ)⁻¹
  simpa only [pow_two] using dimension_squared n

example : 0 < dimension 97 ∧ dimension 97 ≤ 1 := dimension_bounds 97 (by decide)

example (z : RealCoordinate Bool 1) :
    0 < weight (biasScale (1 / 2)) (biasScale (1 / 2)) z ∧
      weight (biasScale (1 / 2)) (biasScale (1 / 2)) z ^ 2 ≤ 1 :=
  bias_weight_bounds (1 / 2) (by norm_num) (by norm_num) z

example : Capacity.energy (Updates.diagonal (weight (biasScale (1 / 2)) (biasScale (1 / 2)))
    ((8 : ℝ) • encode (joint Witness.component))) ≤ (8 : ℝ) ^ 2 :=
  domain_energy _ _ 8 _ (fun z => (bias_weight_bounds (1 / 2) (by norm_num) (by norm_num) z).2)
    ⟨_, SharedAtoms.generator_mem 8 Witness.component Witness.unit, rfl⟩

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.SharedAtoms.Scaling
