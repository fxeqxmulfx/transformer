/-
# Genuine independent fair coordinate splits

Source: arXiv:2604.01978v1, §6.3, `lem:Hanson_wright`.
Each coordinate is assigned independently to one of two blocks by
a fair Boolean variable. Its indicator and complementary indicator
have mean one half. Distinct ordered pairs therefore enter a cross
block with probability one quarter, while diagonal pairs never enter.
The actual cross-matrix polynomial is written with these indicators,
preparing the averaging and Jensen steps of the decoupling argument.
The product law and its finiteness include the empty coordinate family.
-/

import Transformer.Homogenized.Section6_IndependentBilinearMGF
import Mathlib.Probability.Distributions.Uniform
import Mathlib.MeasureTheory.Integral.Bochner.SumMeasure
import Mathlib.MeasureTheory.MeasurableSpace.Instances

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal ENNReal

namespace Transformer.Homogenized

/-- The actual fair Boolean law used for coordinate decoupling.
Source: arXiv:2604.01978v1, §6.3, the random split in `lem:Hanson_wright`. -/
noncomputable def fairCoinLaw : Measure Bool := (PMF.uniformOfFintype Bool).toMeasure

instance : IsProbabilityMeasure fairCoinLaw := by unfold fairCoinLaw; infer_instance

/-- The genuine zero-or-one value of a Boolean outcome. Source:
arXiv:2604.01978v1, §6.3, the coordinate indicator in `lem:Hanson_wright`. -/
def coinValue (b : Bool) : ℝ := if b then 1 else 0

/-- An original indicator and its complement never both contribute.
Source: arXiv:2604.01978v1, §6.3, exclusion of diagonal pairs
from the decoupling cross term in `lem:Hanson_wright`. -/
theorem coinValue_mul_complement_self (b : Bool) : coinValue b * (1 - coinValue b) = 0 := by
  cases b <;> norm_num [coinValue]

/-- The actual independent product of the fair coordinate laws. Source:
arXiv:2604.01978v1, §6.3, the random split for `lem:Hanson_wright`. -/
noncomputable def randomSplitLaw (n : ℕ) : Measure (Fin n → Bool) :=
  Measure.pi (fun _ : Fin n => fairCoinLaw)

instance (n : ℕ) : IsProbabilityMeasure (randomSplitLaw n) := by unfold randomSplitLaw; infer_instance

/-- The actual subset of coordinates assigned to the first block.
Source: arXiv:2604.01978v1, §6.3, the subset in `lem:Hanson_wright`. -/
def randomCoordinateSubset {n : ℕ} (σ : Fin n → Bool) : Finset (Fin n) :=
  Finset.univ.filter (fun i => σ i = true)

/-- Subset membership is exactly the original Boolean outcome.
Source: arXiv:2604.01978v1, §6.3, the random coordinate mask. -/
theorem mem_randomCoordinateSubset {n : ℕ} (σ : Fin n → Bool) (i : Fin n) :
    i ∈ randomCoordinateSubset σ ↔ σ i = true := by
  simp only [randomCoordinateSubset, Finset.mem_filter, Finset.mem_univ, true_and]

/-- Both genuine Boolean atoms have their actual probability one half.
Source: arXiv:2604.01978v1, §6.3, the fair split for `lem:Hanson_wright`. -/
theorem fairCoinLaw_real_singleton (b : Bool) : fairCoinLaw.real {b} = 1 / 2 := by
  rw [Measure.real, fairCoinLaw,
    PMF.toMeasure_apply_singleton (PMF.uniformOfFintype Bool) b (measurableSet_singleton b)]
  norm_num [PMF.uniformOfFintype_apply, Fintype.card_bool]

/-- The original fair indicator has mean one half. Source:
arXiv:2604.01978v1, §6.3, the coordinate averaging in `lem:Hanson_wright`. -/
theorem integral_fairCoin_value : (∫ b : Bool, coinValue b ∂fairCoinLaw) = 1 / 2 := by
  rw [integral_fintype Integrable.of_finite, Fintype.sum_bool]
  norm_num [fairCoinLaw_real_singleton, coinValue]

/-- Every function of the finite split has a genuine finite real
integral. Source: arXiv:2604.01978v1, §6.3, the finite auxiliary
expectation before applying Jensen in `lem:Hanson_wright`. -/
theorem integrable_randomSplit_function (n : ℕ) (f : (Fin n → Bool) → ℝ) :
    Integrable f (randomSplitLaw n) := Integrable.of_finite

/-- Original fair indicators are mutually independent under their
actual product law. Source: arXiv:2604.01978v1, §6.3,
the decoupling coordinate indicators for `lem:Hanson_wright`. -/
theorem iIndepFun_split_coins (n : ℕ) :
    iIndepFun (fun i : Fin n => fun σ : Fin n → Bool => coinValue (σ i)) (randomSplitLaw n) := by
  exact iIndepFun_pi (μ := fun _ : Fin n => fairCoinLaw) (X := fun _ => coinValue)
    (fun i => (measurable_of_countable coinValue).aemeasurable)

/-- Each original indicator retains its actual mean under the genuine
product split law. Source: arXiv:2604.01978v1, §6.3, `lem:Hanson_wright`. -/
theorem integral_split_coin {n : ℕ} (i : Fin n) :
    (∫ σ : Fin n → Bool, coinValue (σ i) ∂randomSplitLaw n) = 1 / 2 := by
  have hl : HasLaw (fun σ : Fin n → Bool => σ i) fairCoinLaw (randomSplitLaw n) :=
    ⟨(measurable_pi_apply i).aemeasurable, (measurePreserving_eval (fun _ : Fin n => fairCoinLaw) i).map_eq⟩
  simpa only [Function.comp_def, integral_fairCoin_value] using
    hl.integral_comp (f := coinValue) Integrable.of_finite.aestronglyMeasurable

/-- The original complementary indicator also has actual mean one half.
Source: arXiv:2604.01978v1, §6.3, the second block in `lem:Hanson_wright`. -/
theorem integral_split_complement_coin {n : ℕ} (i : Fin n) :
    (∫ σ : Fin n → Bool, 1 - coinValue (σ i) ∂randomSplitLaw n) = 1 / 2 := by
  rw [integral_sub Integrable.of_finite Integrable.of_finite, integral_split_coin]
  norm_num

/-- Every distinct ordered coordinate pair has the actual cross-block
indicator expectation one quarter. Source: arXiv:2604.01978v1,
§6.3, the factor four in the decoupling proof of `lem:Hanson_wright`. -/
theorem integral_split_pair_of_ne {n : ℕ} (i j : Fin n) (hij : i ≠ j) :
    (∫ σ : Fin n → Bool, coinValue (σ i) * (1 - coinValue (σ j)) ∂randomSplitLaw n) = 1 / 4 := by
  have h : IndepFun (fun σ : Fin n → Bool => coinValue (σ i))
      (fun σ : Fin n → Bool => 1 - coinValue (σ j)) (randomSplitLaw n) := by
    simpa only [Function.comp_def, id_eq] using ((iIndepFun_split_coins n).indepFun hij).comp
      measurable_id (continuous_const.fun_sub continuous_id).measurable
  have he := h.integral_fun_mul_eq_mul_integral
    Integrable.of_finite.aestronglyMeasurable Integrable.of_finite.aestronglyMeasurable
  rw [integral_split_coin, integral_split_complement_coin] at he
  norm_num at he
  exact he

/-- Diagonal pairs contribute zero and distinct pairs one quarter
under the original split law. Source: arXiv:2604.01978v1, §6.3,
the original off-diagonal averaging in `lem:Hanson_wright`. -/
theorem integral_split_pair {n : ℕ} (i j : Fin n) :
    (∫ σ : Fin n → Bool, coinValue (σ i) * (1 - coinValue (σ j)) ∂randomSplitLaw n) =
      if i = j then 0 else 1 / 4 := by
  by_cases hij : i = j
  · subst j
    have he : (fun σ : Fin n → Bool => coinValue (σ i) * (1 - coinValue (σ i))) = (fun _ => (0 : ℝ)) := by
      funext σ
      exact coinValue_mul_complement_self (σ i)
    rw [he]
    simp
  · rw [ite_eq_right hij, integral_split_pair_of_ne i j hij]

/-- The actual original cross-matrix polynomial has exactly the split
indicator weights. Source: arXiv:2604.01978v1, §6.3,
the random cross form used in the proof of `lem:Hanson_wright`. -/
theorem random_cross_quadratic_eq_coin_sum {n : ℕ}
    (B : Matrix (Fin n) (Fin n) ℝ) (σ : Fin n → Bool) (z : EucSpace n) :
    hansonWrightQuadraticForm (coordinateMask (randomCoordinateSubset σ) * B *
      coordinateMask (randomCoordinateSubset σ)ᶜ) z =
      ∑ i : Fin n, ∑ j : Fin n, (B i j * z i * z j) * (coinValue (σ i) * (1 - coinValue (σ j))) := by
  simp only [hansonWrightQuadraticForm, cross_mask_apply, mem_randomCoordinateSubset]
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  cases σ i <;> cases σ j <;> simp [coinValue]

/-- Two distinct genuine coordinates witness the non-diagonal pair
hypothesis and a split with both blocks nonempty. -/
example : (0 : Fin 2) ≠ 1 ∧
    (0 : Fin 2) ∈ randomCoordinateSubset (fun i : Fin 2 => i == 0) ∧
    (1 : Fin 2) ∉ randomCoordinateSubset (fun i : Fin 2 => i == 0) := by
  simp [mem_randomCoordinateSubset]

end Transformer.Homogenized
