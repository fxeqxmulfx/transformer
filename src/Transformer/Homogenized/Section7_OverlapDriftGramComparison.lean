module

public import Transformer.Homogenized.Section7_BarycenterGramComparison
public import Transformer.Homogenized.SimplexWellDefined
public import Transformer.Homogenized.OverlapDrift

/-!
# The original overlap drift is Lipschitz in the Gram matrix

ArXiv:2604.01978v1, §7, eq:Dij_explicit_clean and the final
polynomial substitution in lem:Gram_stability.
For dimensions at least two, the cross coefficient is bounded by
one and is one-Lipschitz in the overlap; the self coefficient is at
most one half. Splitting the actual products then gives factor four
times the common correlation Lipschitz constant.

The proved original Gaussian barycenter comparison supplies that
constant. The complete original drift therefore has the source's
dimension and variance scaling, with universal constant 80.
-/

@[expose] public section

open MeasureTheory
open scoped NNReal

namespace Transformer.Homogenized

/-- Split a product perturbation into its two actual factors.
Source: arXiv:2604.01978v1, §7, the final substitution after eq:Lispchitz_bound_softmax. -/
theorem abs_mul_sub_mul_le_split (a b x y : ℝ) :
    |a * x - b * y| ≤ |a| * |x - y| + |a - b| * |y| := by
  rw [show a * x - b * y = a * (x - y) + (a - b) * y by ring]
  simpa only [abs_mul] using abs_add_le (a * (x - y)) ((a - b) * y)

/-- The original cross-correlation coefficient has absolute value at most one.
Source: arXiv:2604.01978v1, §7, eq:Dij_explicit_clean. -/
theorem abs_overlap_gram_coefficient_le_one (d r : ℝ) (hd : 2 ≤ d) (hr : |r| ≤ 1) :
    |(1 / d) * (d - 2 + r ^ 2)| ≤ 1 := by
  have hd0 : 0 < d := by linarith
  have hr2 := (sq_le_one_iff_abs_le_one r).2 hr
  have hp0 : 0 ≤ (1 / d) * (d - 2 + r ^ 2) :=
    mul_nonneg (by positivity) (by nlinarith [sq_nonneg r])
  rw [abs_of_nonneg hp0]
  calc
    _ = (d - 2 + r ^ 2) / d := by ring
    _ ≤ 1 := (div_le_iff₀ hd0).2 (by linarith)

/-- The original cross-correlation coefficient is one-Lipschitz on unit overlaps.
Source: arXiv:2604.01978v1, §7, the polynomial substitution after eq:Lispchitz_bound_softmax. -/
theorem abs_sub_overlap_gram_coefficient_le (d r s ε : ℝ) (hd : 2 ≤ d)
    (hr : |r| ≤ 1) (hs : |s| ≤ 1) (hR : |r - s| ≤ ε) :
    |(1 / d) * (d - 2 + r ^ 2) - (1 / d) * (d - 2 + s ^ 2)| ≤ ε := by
  have hd0 : 0 < d := by linarith
  have hi0 : 0 ≤ (1 / d) := by positivity
  have hi1 : (1 / d) ≤ 1 / 2 := one_div_le_one_div_of_le (by norm_num) hd
  have hsum : |r + s| ≤ 2 := (abs_add_le r s).trans (by linarith)
  have hcs : |(1 / d) * (r + s)| ≤ 1 := by
    rw [abs_mul, abs_of_nonneg hi0]
    exact (mul_le_mul_of_nonneg_left hsum hi0).trans (by linarith)
  rw [show (1 / d) * (d - 2 + r ^ 2) - (1 / d) * (d - 2 + s ^ 2) =
    ((1 / d) * (r + s)) * (r - s) by ring, abs_mul]
  simpa only [one_mul] using mul_le_mul hcs hR (abs_nonneg _) zero_le_one

/-- The original self-correlation coefficient lies between zero and one half.
Source: arXiv:2604.01978v1, §7, eq:Dij_explicit_clean. -/
theorem overlap_self_coefficient_mem_Icc (d : ℝ) (hd : 2 ≤ d) :
    (d - 1) / (2 * d) ∈ Set.Icc (0 : ℝ) (1 / 2) := by
  have hd0 : 0 < d := by linarith
  constructor
  · exact div_nonneg (by linarith) (by positivity)
  · exact (div_le_iff₀ (by positivity : 0 < 2 * d)).2 (by linarith)

/-- The actual scalar drift formula preserves Lipschitz control with factor four.
Source: arXiv:2604.01978v1, §7, the final polynomial substitution in lem:Gram_stability. -/
theorem abs_overlapDrift_scalar_sub_le (d r s u v a b c e L ε : ℝ) (hd : 2 ≤ d)
    (hr : |r| ≤ 1) (hs : |s| ≤ 1) (hv : |v| ≤ 1) (hce : |c + e| ≤ 2)
    (hL : 1 ≤ L) (hε : 0 ≤ ε) (hR : |r - s| ≤ ε)
    (hu : |u - v| ≤ L * ε) (ha : |a - c| ≤ L * ε) (hb : |b - e| ≤ L * ε) :
    |((1 / d) * (d - 2 + r ^ 2) * u - ((d - 1) / (2 * d)) * r * (a + b)) -
      ((1 / d) * (d - 2 + s ^ 2) * v - ((d - 1) / (2 * d)) * s * (c + e))| ≤
        4 * L * ε := by
  let P (z : ℝ) := (1 / d) * (d - 2 + z ^ 2)
  let Q := (d - 1) / (2 * d)
  have hP := abs_overlap_gram_coefficient_le_one d r hd hr
  have hPd := abs_sub_overlap_gram_coefficient_le d r s ε hd hr hs hR
  have hQ := overlap_self_coefficient_mem_Icc d hd
  have hfirst : |P r * u - P s * v| ≤ L * ε + ε := by
    apply (abs_mul_sub_mul_le_split (P r) (P s) u v).trans
    exact add_le_add
      ((mul_le_mul hP hu (abs_nonneg _) zero_le_one).trans_eq (one_mul _))
      ((mul_le_mul hPd hv (abs_nonneg _) hε).trans_eq (mul_one _))
  have hab : |(a + b) - (c + e)| ≤ 2 * L * ε := by
    rw [show (a + b) - (c + e) = (a - c) + (b - e) by ring]
    exact (abs_add_le _ _).trans ((add_le_add ha hb).trans_eq (by ring))
  have hprod : |r * (a + b) - s * (c + e)| ≤ 2 * L * ε + 2 * ε := by
    apply (abs_mul_sub_mul_le_split r s (a + b) (c + e)).trans
    apply add_le_add
    · exact (mul_le_mul hr hab (abs_nonneg _) zero_le_one).trans_eq (one_mul _)
    · exact (mul_le_mul hR hce (abs_nonneg _) hε).trans_eq (by ring)
  have hsecond : |Q * r * (a + b) - Q * s * (c + e)| ≤ L * ε + ε := by
    rw [show Q * r * (a + b) - Q * s * (c + e) = Q * (r * (a + b) - s * (c + e)) by ring,
      abs_mul, abs_of_nonneg hQ.1]
    calc
      _ ≤ Q * (2 * L * ε + 2 * ε) := mul_le_mul_of_nonneg_left hprod hQ.1
      _ ≤ (1 / 2 : ℝ) * (2 * L * ε + 2 * ε) :=
        mul_le_mul_of_nonneg_right hQ.2 (by positivity)
      _ = L * ε + ε := by ring
  calc
    _ = |(P r * u - P s * v) + -(Q * r * (a + b) - Q * s * (c + e))| := by
      congr 1
      dsimp [P, Q]
      ring
    _ ≤ |P r * u - P s * v| + |Q * r * (a + b) - Q * s * (c + e)| := by
      simpa only [abs_neg] using abs_add_le (P r * u - P s * v) (-(Q * r * (a + b) - Q * s * (c + e)))
    _ ≤ (L * ε + ε) + (L * ε + ε) := add_le_add hfirst hsecond
    _ ≤ 4 * L * ε := by nlinarith [mul_nonneg (sub_nonneg.mpr hL) hε]

/-- The original drift has a universal Gram Lipschitz bound with constant 80.
Source: arXiv:2604.01978v1, §7, lem:Gram_stability, comparing the two actual
drifts as corrected in GramStability. This stronger assertion allows any
two unit configurations, any temperature and coincident token indices. -/
theorem abs_overlapDrift_sub_le_gram {d n : ℕ} (hd : 2 ≤ d) (β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)) (hρ : IsGaussianHeadLaw d σV σA ρ)
    (X Y : Idx n → EucSpace d) (hX : ∀ a, ‖X a‖ = 1) (hY : ∀ a, ‖Y a‖ = 1)
    (i j : Idx n) (ε : ℝ)
    (hR : ∀ a b, |inner (𝕜 := ℝ) (X a) (X b) - inner (𝕜 := ℝ) (Y a) (Y b)| ≤ ε) :
    |overlapDrift β ρ X i j - overlapDrift β ρ Y i j| ≤
      80 * (1 + (d : ℝ) * (σA : ℝ) ^ 4 * β ^ 2 * (n : ℝ)) * ε := by
  have : IsProbabilityMeasure ρ := isProbabilityMeasure_of_isGaussianHeadLaw hρ
  let U (a b : Idx n) := baryCorr β ρ (empMeasure X) (X a) (X b)
  let V (a b : Idx n) := baryCorr β ρ (empMeasure Y) (Y a) (Y b)
  let L := 1 + 20 * (d : ℝ) * (σA : ℝ) ^ 4 * β ^ 2 * (n : ℝ)
  have hL : 1 ≤ L := by
    have hnon : 0 ≤ 20 * (d : ℝ) * (σA : ℝ) ^ 4 * β ^ 2 * (n : ℝ) := by positivity
    dsimp [L]
    linarith
  have hε : 0 ≤ ε := (abs_nonneg _).trans (hR i i)
  have hΔ (a b : Idx n) : |U a b - V a b| ≤ L * ε :=
    abs_baryCorr_sub_le_gram i.pos β σV σA ρ hρ X Y hX hY a b ε hR
  have hV (a b : Idx n) : |V a b| ≤ 1 := abs_baryCorr_empMeasure_le_one β ρ Y hY a b
  have hsum : |V i i + V j j| ≤ 2 := (abs_add_le _ _).trans (by linarith [hV i i, hV j j])
  have hgramX : |inner (𝕜 := ℝ) (X i) (X j)| ≤ 1 := by
    simpa [hX i, hX j] using abs_real_inner_le_norm (X i) (X j)
  have hgramY : |inner (𝕜 := ℝ) (Y i) (Y j)| ≤ 1 := by
    simpa [hY i, hY j] using abs_real_inner_le_norm (Y i) (Y j)
  have h := abs_overlapDrift_scalar_sub_le (d : ℝ) _ _ (U i j) (V i j) (U i i) (U j j) (V i i) (V j j) L ε
    (by exact_mod_cast hd) hgramX hgramY (hV i j) hsum hL hε (hR i j) (hΔ i j) (hΔ i i) (hΔ j j)
  change |overlapDrift β ρ X i j - overlapDrift β ρ Y i j| ≤ 4 * L * ε at h
  apply h.trans
  dsimp [L]
  nlinarith
/-- Positive scalar drift data instantiate every coefficient and perturbation
hypothesis, with a nonzero overlap change. -/
example : |((1 / (2 : ℝ)) * (2 - 2 + (1 / 2) ^ 2) * (3 / 4) -
      ((2 - 1) / (2 * 2)) * (1 / 2) * (1 + 1)) -
    ((1 / 2) * (2 - 2 + 0 ^ 2) * (1 / 2) - ((2 - 1) / (2 * 2)) * 0 * (1 + 1))| ≤ 2 := by
  convert abs_overlapDrift_scalar_sub_le 2 (1 / 2) 0 (3 / 4) (1 / 2) 1 1 1 1 1 (1 / 2)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) using 1
  norm_num

/-- Original positive-variance Gaussian heads, orthonormal unit tokens and a
singular unit reference satisfy every actual drift-comparison hypothesis. -/
example : ∃ X Y : Idx 2 → EucSpace 2,
    (2 : ℕ) ≤ 2 ∧ IsGaussianHeadLaw 2 1 1 (gaussHeadLaw 2 1 1) ∧
    (∀ a, ‖X a‖ = 1) ∧ (∀ a, ‖Y a‖ = 1) ∧
    (∀ a b, |inner (𝕜 := ℝ) (X a) (X b) - inner (𝕜 := ℝ) (Y a) (Y b)| ≤ 1) := by
  let X : Idx 2 → EucSpace 2 := fun a => EuclideanSpace.single a 1
  let Y : Idx 2 → EucSpace 2 := fun _ => EuclideanSpace.single (0 : Fin 2) 1
  refine ⟨X, Y, le_rfl, isGaussianHeadLaw_gaussHeadLaw 2 1 1,
    fun a => by simp [X], fun a => by simp [Y], ?_⟩
  intro a b
  fin_cases a <;> fin_cases b <;> norm_num [X, Y, EuclideanSpace.inner_single_left, PiLp.single_apply]

end Transformer.Homogenized
