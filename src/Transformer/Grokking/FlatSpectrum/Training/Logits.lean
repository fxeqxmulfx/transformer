module

public import Mathlib.Analysis.Convex.Jensen
public import Mathlib.Analysis.Convex.SpecificFunctions.Basic
public import Mathlib.Probability.Moments.SubGaussian
public import Mathlib.Probability.ProbabilityMassFunction.Integrals
public import Transformer.Grokking.GradientEvidence.RelativeLoss

/-!
# The cross-entropy of finite logits is convex and lies below a quadratic

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), items XE1 and XE2,
their facts about one sample. For logits `z` of a finite class set, their softmax `p` and a
target `t`, the cross-entropy `log Σ_c exp z_c − z_t` changes under an increment `v` by
`log Σ_c p_c exp v_c − v_t` (`GradientEvidence.crossEntropy_change_weighted`). So

* it is convex with gradient `p − e_t` (`crossEntropy_first_order`): by Jensen's inequality
  `Σ_c p_c v_c ≤ log Σ_c p_c exp v_c` (`sum_probability_mul_le_log_weightedExp`);
* it lies below a quadratic model (`crossEntropy_le_add`): if `|v_c| ≤ r` for every class, the
  excess over the linear model is at most `r² / 2` (`log_weightedExp_le`). This is Hoeffding's
  lemma for the distribution `p` (`sum_mul_exp_sub_le_of_mem_Icc`, from
  `ProbabilityTheory.hasSubgaussianMGF_of_mem_Icc`): a variable with values in `[a, b]` has
  `E exp (d − E d) ≤ exp ((b − a)² / 8)`;
* at equal logits it is `log |C|` (`crossEntropy_zero`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Landscape.Neural (crossEntropy)
open Geometry (probability probability_pos sum_probability)
open GradientEvidence (weightedExp weightedExp_pos crossEntropy_change_weighted)

variable {C : Type*} [Fintype C]

/-- Jensen's inequality for the softmax `p` of `z`: `Σ_c p_c v_c ≤ log Σ_c p_c exp v_c`. -/
lemma sum_probability_mul_le_log_weightedExp (z v : C → ℝ) (t : C) :
    ∑ c, probability z c * v c ≤ Real.log (weightedExp z v) := by
  rw [Real.le_log_iff_exp_le (weightedExp_pos z v t)]
  have h := convexOn_exp.map_sum_le (t := Finset.univ) (w := probability z) (p := v)
    (fun c _ => (probability_pos z c).le) (sum_probability z t) fun c _ => Set.mem_univ _
  simpa only [smul_eq_mul, weightedExp] using h

/-- **The cross-entropy is convex in the logits, with gradient `p − e_t`**
(`review/flat_spectrum.md`, item XE1): for the softmax `p` of `z`,
`ce z + Σ_c p_c (z'_c − z_c) − (z'_t − z_t) ≤ ce z'`. -/
theorem crossEntropy_first_order (z z' : C → ℝ) (t : C) :
    crossEntropy z t + (∑ c, probability z c * (z' c - z c) - (z' t - z t)) ≤
      crossEntropy z' t := by
  have h := crossEntropy_change_weighted z (fun c => z' c - z c) t
  rw [show (fun c => z c + (z' c - z c)) = z' from funext fun c => by ring] at h
  linarith [sum_probability_mul_le_log_weightedExp z (fun c => z' c - z c) t]

/-- **Hoeffding's lemma for a finite distribution** (`review/flat_spectrum.md`, item XE2): if
`p` is a probability vector and every `d_c` lies in `[a, b]`, then
`Σ_c p_c exp (d_c − Σ p d) ≤ exp ((b − a)² / 8)`. -/
theorem sum_mul_exp_sub_le_of_mem_Icc {p d : C → ℝ} (hp : ∀ c, 0 ≤ p c) (hsum : ∑ c, p c = 1)
    {a b : ℝ} (hd : ∀ c, d c ∈ Set.Icc a b) :
    ∑ c, p c * Real.exp (d c - ∑ c', p c' * d c') ≤ Real.exp ((b - a) ^ 2 / 8) := by
  classical
  let _ : MeasurableSpace C := ⊤
  have : DiscreteMeasurableSpace C := ⟨fun _ => MeasurableSpace.measurableSet_top⟩
  have hone : ∑ c, ENNReal.ofReal (p c) = 1 := by
    rw [← ENNReal.ofReal_sum_of_nonneg fun c _ => hp c, hsum, ENNReal.ofReal_one]
  have hint (f : C → ℝ) : ∫ c, f c ∂(PMF.ofFintype _ hone).toMeasure = ∑ c, p c * f c := by
    rw [PMF.integral_eq_sum]
    exact Finset.sum_congr rfl fun c _ => by
      rw [PMF.ofFintype_apply, ENNReal.toReal_ofReal (hp c), smul_eq_mul]
  obtain ⟨c₀⟩ : Nonempty C := by
    by_contra hC
    rw [not_nonempty_iff] at hC
    simp at hsum
  have hab : a ≤ b := (hd c₀).1.trans (hd c₀).2
  have h := (ProbabilityTheory.hasSubgaussianMGF_of_mem_Icc
    (μ := (PMF.ofFintype _ hone).toMeasure) (X := d) (a := a) (b := b)
    Measurable.of_discrete.aemeasurable (Filter.Eventually.of_forall hd)).mgf_le 1
  simp only [ProbabilityTheory.mgf, hint, one_mul] at h
  refine h.trans (le_of_eq ?_)
  congr 1
  simp only [NNReal.coe_pow, NNReal.coe_div, coe_nnnorm, Real.norm_eq_abs,
    abs_of_nonneg (sub_nonneg.mpr hab), NNReal.coe_ofNat]
  ring

/-- The uniform distribution on two points with values `0` and `1`. -/
example := sum_mul_exp_sub_le_of_mem_Icc (p := fun _ : Bool => (1 / 2 : ℝ))
  (d := fun c => if c then 1 else 0) (a := 0) (b := 1) (fun _ => by norm_num) (by norm_num)
  fun c => by cases c <;> norm_num

/-- The exponential moment of increments bounded by `r` under the softmax of `z`:
`log Σ_c p_c exp v_c ≤ Σ_c p_c v_c + r² / 2`. -/
lemma log_weightedExp_le (z v : C → ℝ) (t : C) {r : ℝ} (hv : ∀ c, |v c| ≤ r) :
    Real.log (weightedExp z v) ≤ ∑ c, probability z c * v c + r ^ 2 / 2 := by
  rw [Real.log_le_iff_le_exp (weightedExp_pos z v t), Real.exp_add]
  have h := sum_mul_exp_sub_le_of_mem_Icc (p := probability z) (d := v) (a := -r) (b := r)
    (fun c => (probability_pos z c).le) (sum_probability z t) fun c => abs_le.mp (hv c)
  have e : weightedExp z v = Real.exp (∑ c, probability z c * v c) *
      ∑ c, probability z c * Real.exp (v c - ∑ c', probability z c' * v c') := by
    rw [weightedExp, Finset.mul_sum]
    refine Finset.sum_congr rfl fun c _ => ?_
    rw [Real.exp_sub]
    field_simp
  rw [e]
  exact mul_le_mul_of_nonneg_left (h.trans (Real.exp_le_exp.mpr (le_of_eq (by ring))))
    (Real.exp_pos _).le

example := log_weightedExp_le (fun _ : Bool => (0 : ℝ)) (fun c => if c then 1 else -1) true
  (r := 1) fun c => by cases c <;> norm_num

/-- **The cross-entropy lies below its quadratic model** (`review/flat_spectrum.md`, item
XE2): if the logits move by at most `r` in every class, the cross-entropy exceeds its linear
model at `z`, with gradient `p − e_t`, by at most `r² / 2`. -/
theorem crossEntropy_le_add (z z' : C → ℝ) (t : C) {r : ℝ} (hr : ∀ c, |z' c - z c| ≤ r) :
    crossEntropy z' t ≤ crossEntropy z t +
      (∑ c, probability z c * (z' c - z c) - (z' t - z t)) + r ^ 2 / 2 := by
  have h := crossEntropy_change_weighted z (fun c => z' c - z c) t
  rw [show (fun c => z c + (z' c - z c)) = z' from funext fun c => by ring] at h
  linarith [log_weightedExp_le z (fun c => z' c - z c) t hr]

example := crossEntropy_le_add (fun _ : Bool => (0 : ℝ)) (fun c => if c then 1 else -1) true
  (r := 1) fun c => by cases c <;> norm_num

/-- With every logit equal to zero the cross-entropy is `log |C|`. -/
lemma crossEntropy_zero (t : C) :
    crossEntropy (fun _ : C => (0 : ℝ)) t = Real.log (Fintype.card C) := by
  simp [crossEntropy]

end Transformer.Grokking.FlatSpectrum.Training
