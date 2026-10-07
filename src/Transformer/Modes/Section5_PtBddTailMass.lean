import Transformer.Modes.Section5_PtBddDecayFalse
import Transformer.Modes.Section3_Edgeworth

/-
# The number of modes of a Gaussian KDE — mass near the origin

The proof of `lem: pt.bdd` in arXiv:2412.09080v3, §5.5, overlooks Gaussian
samples far from the observation point. Both coordinates of `(G, G')` then
approach zero exponentially. If all `n` samples lie in the same remote unit
interval, their normalized sum lies in a small rectangle around the origin.
Independence gives a lower bound on the rectangle's probability.

This module supplies that lower bound, without assuming that a density
exists. `Section5_PtBddDensity.lean` compares it with the quadratic bound
imposed by a continuous planar density and refutes `lem: pt.bdd`.

Source: arXiv:2412.09080v3, §2.2, `eq: Gt`; §5.5, `lem: pt.bdd`.
-/

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators ENNReal

namespace Transformer.Modes

/-- A rectangle of side length `2r` centered at the origin, used to test
the density claimed in §5.5, `lem: pt.bdd`. -/
def centeredBox (r : ℝ) : Set (ℝ × ℝ) := Set.Icc (-r) r ×ˢ Set.Icc (-r) r

/-- Both coordinates of the rectangle are measurable.
Source: arXiv:2412.09080v3, §5.5, the planar density in `lem: pt.bdd`. -/
theorem measurableSet_centeredBox (r : ℝ) : MeasurableSet (centeredBox r) :=
  measurableSet_Icc.prod measurableSet_Icc

/-- Membership records the two coordinate bounds.
Source: arXiv:2412.09080v3, §5.5, the origin of the joint law. -/
theorem mem_centeredBox (r : ℝ) (v : ℝ × ℝ) :
    v ∈ centeredBox r ↔ |v.1| ≤ r ∧ |v.2| ≤ r := by
  simp only [centeredBox, Set.mem_prod, Set.mem_Icc, abs_le]

/-- Away from the observation point, `G` obeys the same bound as `G'`.
This is the first-coordinate companion to `abs_bigG'_le_far`.
Source: arXiv:2412.09080v3, §2.2, `eq: Gt`; §5.5, `lem: pt.bdd`. -/
theorem abs_bigG_le_far {β : ℝ} (hβ : 2 < β) (t : ℝ) {x z : ℝ}
    (hz : 1 ≤ z) (hx : z ≤ |t - x|) :
    |bigG β t x| ≤ (2 * β / (β - 2)) * Real.exp (-((β + 2) / 4) * z ^ 2) := by
  have hu : 1 ≤ |t - x| := hz.trans hx
  have hu2 : 1 ≤ (t - x) ^ 2 := by nlinarith [sq_abs (t - x)]
  have hsign : 1 - β * (t - x) ^ 2 ≤ 0 := by nlinarith
  have hcompare : |t - x| ≤ β * (t - x) ^ 2 - 1 := by
    nlinarith [sq_nonneg (|t - x| - 1), sq_abs (t - x)]
  calc
    |bigG β t x| ≤ |bigG' β t x| := by
      rw [bigG, bigG', abs_mul, abs_mul, Real.abs_exp, abs_of_nonpos hsign]
      exact mul_le_mul_of_nonneg_left (by linarith) (Real.exp_pos _).le
    _ ≤ _ := abs_bigG'_le_far hβ t hz hx

/-- A remote sample at `β = 10` satisfies the first-coordinate bound. -/
example : |bigG 10 0 (-2)| ≤ (2 * 10 / (10 - 2)) * Real.exp (-((10 + 2) / 4) * 1 ^ 2) :=
  abs_bigG_le_far (by norm_num) 0 le_rfl (by norm_num)

/-- Independence multiplies the mass of a common sampling interval.
Source: arXiv:2412.09080v3, §5.5, the convolution `ν_t^{*n}`. -/
theorem gaussianSample_interval_mass (n : ℕ) (a b : ℝ) :
    (gaussianSample n).real (Set.pi Set.univ fun _ => Set.Icc a b) =
      (gaussianReal 0 1).real (Set.Icc a b) ^ n := by
  simp only [measureReal_def, gaussianSample, Measure.pi_pi, Fin.prod_const,
    ENNReal.toReal_pow]

/-- The normalized sum of the coordinates is measurable in the sample.
Source: arXiv:2412.09080v3, §2.2, the display after `eq: Gt`. -/
theorem measurable_sumGG' (n : ℕ) (β t : ℝ) : Measurable (sumGG' n β t) := by
  have hc : Continuous (sumGG' n β t) := by
    unfold sumGG' bigG bigG'
    fun_prop
  exact hc.measurable

/-- A bound on every summand bounds its normalized sum.
Source: arXiv:2412.09080v3, §2.2, normalization in `eq:Fn`. -/
theorem abs_normalized_sum_le {n : ℕ} {u : Fin n → ℝ} {r : ℝ}
    (hu : ∀ i, |u i| ≤ r) :
    |(Real.sqrt n)⁻¹ * ∑ i, u i| ≤ (Real.sqrt n)⁻¹ * (n * r) := by
  rw [abs_mul, abs_of_nonneg (by positivity : 0 ≤ (Real.sqrt n)⁻¹)]
  apply mul_le_mul_of_nonneg_left _ (by positivity)
  calc
    |∑ i, u i| ≤ ∑ i, |u i| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _i : Fin n, r := Finset.sum_le_sum fun i _ => hu i
    _ = n * r := by simp

/-- Zero summands satisfy the deterministic normalization bound. -/
example : ∀ i : Fin 5, |(fun _ : Fin 5 => (0 : ℝ)) i| ≤ 1 := by
  intro i
  norm_num

/-- If all samples lie in a remote unit interval, their normalized pair
lies in a small rectangle. No density assumption is used.
Source: arXiv:2412.09080v3, §5.5, `lem: pt.bdd`. -/
theorem remote_samples_subset_box {β : ℝ} (hβ : 2 < β) (n : ℕ) (t : ℝ)
    {z : ℝ} (hz : 1 ≤ z) :
    (Set.pi Set.univ fun _ : Fin n => Set.Icc (t - z - 1) (t - z)) ⊆
      sumGG' n β t ⁻¹' centeredBox
        ((Real.sqrt n)⁻¹ * (n * (2 * β / (β - 2))) *
          Real.exp (-((β + 2) / 4) * z ^ 2)) := by
  intro X hX
  have hx (i : Fin n) : z ≤ |t - X i| := by
    have hi := hX i (Set.mem_univ i)
    rw [abs_of_nonneg (by linarith [hi.2])]
    linarith [hi.2]
  rw [Set.mem_preimage, mem_centeredBox]
  dsimp only [sumGG']
  constructor
  · simpa only [mul_assoc] using
      abs_normalized_sum_le (fun i => abs_bigG_le_far hβ t hz (hx i))
  · simpa only [mul_assoc] using
      abs_normalized_sum_le (fun i => abs_bigG'_le_far hβ t hz (hx i))

/-- The remote sampling event is nonempty at the counterexample parameters. -/
example : (2 : ℝ) < 10 ∧ (1 : ℝ) ≤ 2 ∧
    (fun _ : Fin 5 => (-2 : ℝ)) ∈
      Set.pi Set.univ (fun _ => Set.Icc (0 - 2 - 1) (0 - 2)) := by
  refine ⟨by norm_num, by norm_num, ?_⟩
  intro i hi
  norm_num

/-- The rectangle's probability is at least the product of the Gaussian
tail lower bounds. This will exceed every `O(r²)` bound when `β + 2 > n`.
Source: arXiv:2412.09080v3, §5.5, counterexample to `lem: pt.bdd`. -/
theorem gaussianSample_box_mass_lower {β : ℝ} (hβ : 2 < β) (n : ℕ) (t : ℝ)
    {z : ℝ} (hz : 1 ≤ z) (hzt : t ≤ z) :
    ((Real.sqrt (2 * π))⁻¹) ^ n * Real.exp (-n * (z + 1 - t) ^ 2 / 2) ≤
      (gaussianSample n).real (sumGG' n β t ⁻¹' centeredBox
        ((Real.sqrt n)⁻¹ * (n * (2 * β / (β - 2))) *
          Real.exp (-((β + 2) / 4) * z ^ 2))) := by
  have hpow := pow_le_pow_left₀ (by positivity)
    (gaussian_Icc_lower hzt) n
  rw [mul_pow, ← Real.exp_nat_mul] at hpow
  have htail := measureReal_mono (μ := gaussianSample n) (remote_samples_subset_box hβ n t hz)
    (measure_ne_top _ _)
  rw [gaussianSample_interval_mass] at htail
  have hexp : Real.exp ((n : ℝ) * (-(z + 1 - t) ^ 2 / 2)) =
      Real.exp (-n * (z + 1 - t) ^ 2 / 2) := by congr 1; ring
  rw [hexp] at hpow
  exact hpow.trans htail

/-- The probability lower bound applies at `n = 5`, `β = 10`, `t = 0`. -/
example : (2 : ℝ) < 10 ∧ (1 : ℝ) ≤ 2 ∧ (0 : ℝ) ≤ 2 := by norm_num

/-- The rectangle has actual sample points at every remote radius, so the
tail event in the probability lower bound is never empty. -/
example (n : ℕ) (t z : ℝ) :
    (fun _ : Fin n => t - z) ∈
      Set.pi Set.univ (fun _ => Set.Icc (t - z - 1) (t - z)) := by
  intro i hi
  constructor <;> linarith

end Transformer.Modes
