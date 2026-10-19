module

public import Transformer.Metastability.Section4_UniformEvenMoments
public import Mathlib.Data.Nat.Factorial.Basic

/-!
# Quantitative even-moment bounds for the original uniform sphere law

ArXiv:2410.06833v1, §4, `prop: concentration unif`, and
arXiv:2312.10794v5, Appendix D, Step 2, almost orthogonality.
The proved original recurrence `(d+2k)M_(k+1)=(2k+1)M_k`
gives Gaussian even-moment upper bounds in each original unit
direction. The exact original moments remain in the conclusions.

The comparison coefficients are `(2k)!/(2^k k! d^k)`. Their
actual recurrence and original zeroth value are computed here;
they are then proved to dominate the original uniform moments.
The factorial inequality also gives the bound `(k/d)^k`, useful
for Markov's inequality with a degree selected from `log d`.
These estimates assume neither a concentration bound nor a
Gaussian representation of the original uniform measure.
-/

@[expose] public section

open Real MeasureTheory
open scoped Nat

namespace Transformer.Metastability

/-- The Gaussian comparison coefficient for the original half-degree.
Source: arXiv:2410.06833v1, §4, the scalar moment bound used in
the original uniform projection-concentration estimate. -/
noncomputable def gaussianEvenMomentBound (d k : ℕ) : ℝ :=
  ((2 * k).factorial : ℝ) / ((2 : ℝ) ^ k * (k.factorial : ℝ) * (d : ℝ) ^ k)

/-- The comparison's zeroth coefficient has the original normalization.
Source: arXiv:2410.06833v1, §4, the probability normalization of the moment estimate. -/
@[simp] theorem gaussianEvenMomentBound_zero (d : ℕ) : gaussianEvenMomentBound d 0 = 1 := by
  simp [gaussianEvenMomentBound]

/-- The actual Gaussian comparison coefficients have their original
even-moment recurrence. Source: arXiv:2410.06833v1, §4,
the quantitative moment bound used for `prop: concentration unif`. -/
theorem gaussianEvenMomentBound_succ {d : ℕ} (hd : 0 < d) (k : ℕ) :
    gaussianEvenMomentBound d (k + 1) = ((2 * (k : ℝ) + 1) / (d : ℝ)) *
      gaussianEvenMomentBound d k := by
  have hdne : (d : ℝ) ≠ 0 := by exact_mod_cast hd.ne'
  have hkne : (k.factorial : ℝ) ≠ 0 := by exact_mod_cast Nat.factorial_ne_zero k
  have hdegree : 2 * (k + 1) = (2 * k + 1) + 1 := by omega
  unfold gaussianEvenMomentBound
  rw [hdegree, Nat.factorial_succ (2 * k + 1), Nat.factorial_succ (2 * k), Nat.factorial_succ k,
    pow_succ, pow_succ]
  push_cast
  field_simp
  ring

/-- The exact original recurrence bounds one moment by the preceding
moment divided by the original dimension.
Source: arXiv:2410.06833v1, §4, the uniform directional moment estimate. -/
theorem uniformEvenMoment_succ_le {d : ℕ} (hd : 0 < d)
    (ν : Measure (SSphere d)) (hν : IsUniformOn d ν) {u : EucSpace d}
    (hu : ‖u‖ = 1) (k : ℕ) :
    uniformEvenMoment ν u (k + 1) ≤ (2 * (k : ℝ) + 1) * uniformEvenMoment ν u k / (d : ℝ) := by
  have hdpos : 0 < (d : ℝ) := by exact_mod_cast hd
  apply (le_div_iff₀ hdpos).mpr
  calc
    _ = (d : ℝ) * uniformEvenMoment ν u (k + 1) := by ring
    _ ≤ ((d : ℝ) + 2 * k) * uniformEvenMoment ν u (k + 1) :=
      mul_le_mul_of_nonneg_right (le_add_of_nonneg_right (by positivity))
        (uniformEvenMoment_nonneg ν u (k + 1))
    _ = _ := uniformEvenMoment_recurrence hd ν hν hu k

/-- Every original uniform even directional moment is bounded by its
actual Gaussian comparison coefficient.
Source: arXiv:2410.06833v1, §4, the scalar tail input for
`prop: concentration unif`. The original law and direction are retained. -/
theorem uniformEvenMoment_le_gaussianBound {d : ℕ} (hd : 0 < d)
    (ν : Measure (SSphere d)) (hν : IsUniformOn d ν) {u : EucSpace d}
    (hu : ‖u‖ = 1) (k : ℕ) : uniformEvenMoment ν u k ≤ gaussianEvenMomentBound d k := by
  let : IsProbabilityMeasure ν := hν.1
  have hdpos : 0 < (d : ℝ) := by exact_mod_cast hd
  induction k with
  | zero => simp
  | succ k ih =>
    calc
      _ ≤ (2 * (k : ℝ) + 1) * uniformEvenMoment ν u k / (d : ℝ) :=
        uniformEvenMoment_succ_le hd ν hν hu k
      _ = ((2 * (k : ℝ) + 1) / (d : ℝ)) * uniformEvenMoment ν u k := by ring
      _ ≤ ((2 * (k : ℝ) + 1) / (d : ℝ)) * gaussianEvenMomentBound d k :=
        mul_le_mul_of_nonneg_left ih (by positivity)
      _ = _ := (gaussianEvenMomentBound_succ hd k).symm

/-- The actual comparison coefficient is bounded by the simple original
half-degree power. Source: arXiv:2410.06833v1, §4,
the moment degree chosen for the original projection-concentration bound. -/
theorem gaussianEvenMomentBound_le_pow {d : ℕ} (hd : 0 < d) (k : ℕ) :
    gaussianEvenMomentBound d k ≤ ((k : ℝ) / (d : ℝ)) ^ k := by
  have hdpos : 0 < (d : ℝ) := by exact_mod_cast hd
  have hkne : (k.factorial : ℝ) ≠ 0 := by exact_mod_cast Nat.factorial_ne_zero k
  have hfac : ((2 * k).factorial : ℝ) ≤ (2 * (k : ℝ)) ^ k * (k.factorial : ℝ) := by
    exact_mod_cast Nat.factorial_two_mul_le k
  calc
    _ ≤ ((2 * (k : ℝ)) ^ k * (k.factorial : ℝ)) /
        ((2 : ℝ) ^ k * (k.factorial : ℝ) * (d : ℝ) ^ k) :=
      div_le_div_of_nonneg_right hfac (by positivity)
    _ = _ := by
      rw [mul_pow, div_pow]
      field_simp

/-- The original even moment has a quantitative bound in the original
dimension and half-degree, uniformly over original unit directions.
Source: arXiv:2410.06833v1, §4, the uniform directional concentration input. -/
theorem uniformEvenMoment_le_pow {d : ℕ} (hd : 0 < d)
    (ν : Measure (SSphere d)) (hν : IsUniformOn d ν) {u : EucSpace d}
    (hu : ‖u‖ = 1) (k : ℕ) : uniformEvenMoment ν u k ≤ ((k : ℝ) / (d : ℝ)) ^ k :=
  (uniformEvenMoment_le_gaussianBound hd ν hν hu k).trans (gaussianEvenMomentBound_le_pow hd k)

/-- Scaling the actual direction scales its original moment by the
original even power. Source: arXiv:2410.06833v1, §4, directional moments. -/
theorem uniformEvenMoment_smul {d : ℕ} (ν : Measure (SSphere d)) (u : EucSpace d)
    (c : ℝ) (k : ℕ) : uniformEvenMoment ν (c • u) k = c ^ (2 * k) * uniformEvenMoment ν u k := by
  unfold uniformEvenMoment
  rw [← integral_const_mul]
  apply integral_congr_ae
  exact ae_of_all ν fun x => by dsimp only; rw [real_inner_smul_left, mul_pow]

/-- The original moment estimate also holds for directions of norm at
most one. Source: arXiv:2410.06833v1, §4, the finite-direction
projection estimate needed for `prop: concentration unif`. -/
theorem uniformEvenMoment_le_pow_of_norm_le_one {d : ℕ} (hd : 0 < d)
    (ν : Measure (SSphere d)) (hν : IsUniformOn d ν) (u : EucSpace d)
    (hu : ‖u‖ ≤ 1) (k : ℕ) : uniformEvenMoment ν u k ≤ ((k : ℝ) / (d : ℝ)) ^ k := by
  let : IsProbabilityMeasure ν := hν.1
  by_cases hz : u = 0
  · subst u
    simp only [uniformEvenMoment, inner_zero_left, integral_const, probReal_univ, one_smul]
    by_cases hk : k = 0
    · simp [hk]
    · rw [zero_pow (by omega : 2 * k ≠ 0)]
      positivity
  · have hn : ‖u‖ ≠ 0 := norm_ne_zero_iff.mpr hz
    let v : EucSpace d := ‖u‖⁻¹ • u
    have hv : ‖v‖ = 1 := by
      dsimp only [v]
      rw [norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ hn]
    have heq : u = ‖u‖ • v := by
      dsimp only [v]
      rw [smul_smul, mul_inv_cancel₀ hn, one_smul]
    have hp : ‖u‖ ^ (2 * k) ≤ 1 := by
      simpa only [one_pow] using pow_le_pow_left₀ (norm_nonneg u) hu (2 * k)
    rw [heq, uniformEvenMoment_smul]
    exact (mul_le_mul_of_nonneg_right hp (uniformEvenMoment_nonneg ν v k)).trans
      (by simpa only [one_mul] using uniformEvenMoment_le_pow hd ν hν hv k)

/-- The original fourth directional moment is computed exactly.
Source: arXiv:2410.06833v1, §4, the original uniform moment recurrence,
here checking both the dimension factors and the Gaussian numerator three. -/
theorem uniformEvenMoment_two {d : ℕ} (hd : 0 < d)
    (ν : Measure (SSphere d)) (hν : IsUniformOn d ν) {u : EucSpace d}
    (hu : ‖u‖ = 1) : uniformEvenMoment ν u 2 = 3 / ((d : ℝ) * ((d : ℝ) + 2)) := by
  have hdpos : 0 < (d : ℝ) := by exact_mod_cast hd
  have h := uniformEvenMoment_recurrence hd ν hν hu 1
  rw [uniformEvenMoment_one hd ν hν hu] at h
  norm_num only [Nat.cast_one, mul_one, one_add_one_eq_two] at h
  apply (eq_div_iff (by positivity : (d : ℝ) * ((d : ℝ) + 2) ≠ 0)).mpr
  have hm := congrArg (fun z : ℝ => (d : ℝ) * z) h
  field_simp at hm
  nlinarith

/-- The comparison coefficient is positive in every original positive
dimension. Source: arXiv:2410.06833v1, §4, the scalar moment comparison. -/
theorem gaussianEvenMomentBound_pos {d : ℕ} (hd : 0 < d) (k : ℕ) :
    0 < gaussianEvenMomentBound d k := by
  have hdpos : 0 < (d : ℝ) := by exact_mod_cast hd
  unfold gaussianEvenMomentBound
  positivity

/-- Positive dimension, the actual uniform law and a nonzero unit
direction jointly realize every hypothesis of the quantitative bounds. -/
example : (0 : ℕ) < 2 ∧ IsUniformOn 2 (MeanField.uniformLaw 2) ∧
    ‖(EuclideanSpace.single (0 : Fin 2) 1 : EucSpace 2)‖ = 1 :=
  ⟨by decide, isUniformOn_uniformLaw (by norm_num), by simp⟩

/-- The actual uniform circle coordinate has fourth moment `3/8`,
checking the original law and the exact recurrence on a nonempty sphere. -/
example : uniformEvenMoment (MeanField.uniformLaw 2)
    (EuclideanSpace.single (0 : Fin 2) 1) 2 = 3 / 8 := by
  have h := uniformEvenMoment_two (d := 2) (by decide) _
    (isUniformOn_uniformLaw (by norm_num)) (by simp : ‖(EuclideanSpace.single (0 : Fin 2) 1 : EucSpace 2)‖ = 1)
  norm_num at h ⊢
  exact h

end Transformer.Metastability
