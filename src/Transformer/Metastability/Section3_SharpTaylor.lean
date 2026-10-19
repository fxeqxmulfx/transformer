module

public import Transformer.Metastability.Section3_ScalarTaylor
public import Transformer.Perceptron.KernelSup

/-!
# Sharp Taylor bounds for the normalized angular interaction

ArXiv:2410.06833v1, §3.2, `eq: otto.attention` and the kernel `g`
in the proof of `lem: PL.borjan`. The actual normalized interaction
satisfies `|g|≤1` at every positive temperature. This follows from
the already proved bound for the same exponential kernel in
arXiv:2601.21366v2, `rem:strictSOPD-perceptron`.

Keeping the factor `1/2` in Taylor's theorem is essential for the
Euclidean lower bound in (H1) when there are two tokens.
-/

@[expose] public section

open scoped NNReal
open Set Real

namespace Transformer.Metastability

/-- **The normalized interaction derivative has absolute bound one.**
Source: arXiv:2410.06833v1, §3.2, the kernel `g`; the equivalent
unnormalized bound is arXiv:2601.21366v2, `rem:strictSOPD-perceptron`.
The positive factors `β exp β` cancel exactly. -/
theorem interaction_derivative_abs_le_one (β : ℝ) (hβ : 0 < β) (x : ℝ) :
    |Causal.g_pot β x| ≤ 1 := by
  have heq : Perceptron.kernelK2 β x = -(β * Real.exp β) * Causal.g_pot β x := by
    unfold Perceptron.kernelK2 Causal.g_pot
    rw [show β * (Real.cos x - 1) = β * Real.cos x - β by ring,
      Real.exp_sub]
    field_simp
    ring
  have h := Perceptron.abs_kernelK2_le β hβ x
  rw [heq, abs_mul, abs_neg, abs_of_pos (mul_pos hβ (Real.exp_pos β))] at h
  exact le_of_mul_le_mul_left (by simpa using h) (mul_pos hβ (Real.exp_pos β))

/-- **The normalized force kernel is globally one-Lipschitz.**
Source: arXiv:2410.06833v1, §3.2, `g=h'` in `lem: PL.borjan`. -/
theorem interaction_sub_abs_le_one (β : ℝ) (hβ : 0 < β) (x y : ℝ) :
    |Causal.h_pot β x - Causal.h_pot β y| ≤ |x - y| := by
  have hlip : LipschitzWith 1 (Causal.h_pot β) := by
    apply lipschitzWith_of_nnnorm_deriv_le
      (fun u => (Causal.hasDerivAt_h_pot β u).differentiableAt)
    intro u
    rw [(Causal.hasDerivAt_h_pot β u).deriv]
    exact_mod_cast interaction_derivative_abs_le_one β hβ u
  simpa only [Real.norm_eq_abs, NNReal.coe_one, one_mul] using hlip.norm_sub_le x y

/-- The temperature restriction is realized at `β=1`. -/
example : (0 : ℝ) < 1 := one_pos

/-- **The global derivative bound is attained at a zero angular gap.**
Source: arXiv:2410.06833v1, §3.2, the displayed kernel `g`.
At this point the sine term vanishes and the normalized exponential
and cosine both equal one, at every temperature. -/
theorem interaction_derivative_zero (β : ℝ) : Causal.g_pot β 0 = 1 := by
  simp [Causal.g_pot]

/-- A positive-temperature witness also attains the absolute bound,
so the constant one cannot be reduced. -/
example : (0 : ℝ) < 1 ∧ |Causal.g_pot 1 0| = 1 := by
  rw [interaction_derivative_zero]
  norm_num

/-- **Taylor's upper bound with the sharp quadratic coefficient `L/2`.**
Apply the mean value theorem to the remainder after subtracting
`(L/2)t²(y-x)²` along the segment. Source: arXiv:2410.06833v1, §3.2,
the energy-discrepancy estimates leading to `eq: otto.attention`. -/
theorem quadratic_upper_half_of_derivative_bound (f df : ℝ → ℝ) (L : ℝ)
    (hf : ∀ u : ℝ, HasDerivAt f (df u) u)
    (hlip : ∀ u v : ℝ, |df u - df v| ≤ L * |u - v|) (x y : ℝ) :
    f y - f x ≤ df x * (y - x) + (L / 2) * (y - x) ^ 2 := by
  let F (t : ℝ) := f (x + t * (y - x)) - df x * t * (y - x) -
    (L / 2) * t ^ 2 * (y - x) ^ 2
  have hd (t : ℝ) : HasDerivAt F
      ((df (x + t * (y - x)) - df x) * (y - x) - L * t * (y - x) ^ 2) t := by
    have hpath := (hf (x + t * (y - x))).comp t
      (((hasDerivAt_id t).mul_const (y - x)).const_add x)
    have hlin := ((hasDerivAt_id t).const_mul (df x)).mul_const (y - x)
    have hquad := ((hasDerivAt_pow 2 t).const_mul (L / 2)).mul_const ((y - x) ^ 2)
    convert hpath.sub hlin |>.sub hquad using 1
    · rfl
    · simp only [one_mul, Nat.cast_ofNat, mul_one]
      ring
  have hb (t : ℝ) (ht : 0 ≤ t) :
      (df (x + t * (y - x)) - df x) * (y - x) ≤ L * t * (y - x) ^ 2 := by
    have h := mul_le_mul_of_nonneg_right (hlip (x + t * (y - x)) x)
      (abs_nonneg (y - x))
    rw [show x + t * (y - x) - x = t * (y - x) by ring,
      abs_mul, abs_of_nonneg ht] at h
    calc
      _ ≤ |(df (x + t * (y - x)) - df x) * (y - x)| := le_abs_self _
      _ ≤ L * (t * |y - x|) * |y - x| := by rwa [abs_mul]
      _ = L * t * (y - x) ^ 2 := by
        rw [show L * (t * |y - x|) * |y - x| = L * t * |y - x| ^ 2 by ring, sq_abs]
  have h := (convex_Icc (0 : ℝ) 1).image_sub_le_mul_sub_of_deriv_le
    (continuous_iff_continuousAt.mpr fun t => (hd t).continuousAt).continuousOn
    (fun t _ => (hd t).differentiableAt.differentiableWithinAt)
    (C := 0) (fun t ht => by
      rw [(hd t).deriv]
      rw [interior_Icc] at ht
      linarith [hb t ht.1.le])
    0 ⟨le_rfl, zero_le_one⟩ 1 ⟨zero_le_one, le_rfl⟩ zero_le_one
  dsimp only [F] at h
  have hzero : x + 0 * (y - x) = x := by ring
  have hone : x + 1 * (y - x) = y := by ring
  rw [hzero, hone] at h
  norm_num at h
  linarith

/-- **The sharp absolute Taylor remainder.** Applying the upper bound
to both signs gives the lower bound as well. Source: arXiv:2410.06833v1,
§3.2, the energy comparison in `eq: otto.attention`. -/
theorem quadratic_remainder_half_abs_le (f df : ℝ → ℝ) (L : ℝ)
    (hf : ∀ u : ℝ, HasDerivAt f (df u) u)
    (hlip : ∀ u v : ℝ, |df u - df v| ≤ L * |u - v|) (x y : ℝ) :
    |f y - f x - df x * (y - x)| ≤ (L / 2) * (y - x) ^ 2 := by
  have hup := quadratic_upper_half_of_derivative_bound f df L hf hlip x y
  have hlow := quadratic_upper_half_of_derivative_bound (fun u => -f u) (fun u => -df u)
    L (fun u => (hf u).neg) (fun u v => by
      rw [show -df u - -df v = -(df u - df v) by ring, abs_neg]
      exact hlip u v) x y
  apply abs_le.mpr
  constructor <;> linarith

/-- Both sharp scalar estimates have their full inputs realized by
`f(u)=u²`, `df(u)=2u`, `L=2`. -/
example : (∀ u : ℝ, HasDerivAt (fun v : ℝ => v ^ 2) (2 * u) u) ∧
    (∀ u v : ℝ, |2 * u - 2 * v| ≤ 2 * |u - v|) := by
  refine ⟨fun u => ?_, fun u v => ?_⟩
  · simpa using hasDerivAt_pow 2 u
  · rw [← mul_sub, abs_mul, abs_of_pos (by norm_num : (0 : ℝ) < 2)]

/-- **The true pair energy has remainder at most half the squared gap.**
Source: arXiv:2410.06833v1, §3.2, `𝖤_β`, `g=h'`, and
the projection-energy comparison in `eq: otto.attention`. -/
theorem angularPairPotential_remainder_half (β : ℝ) (hβ : 0 < β) (x y : ℝ) :
    |angularPairPotential β y - angularPairPotential β x + Causal.h_pot β x * (y - x)|
      ≤ (1 / 2 : ℝ) * (y - x) ^ 2 := by
  have h := quadratic_remainder_half_abs_le (angularPairPotential β)
    (fun u => -Causal.h_pot β u) 1 (hasDerivAt_angularPairPotential β hβ.ne')
    (fun u v => by
      rw [show -Causal.h_pot β u - -Causal.h_pot β v =
        -(Causal.h_pot β u - Causal.h_pot β v) by ring, abs_neg, one_mul]
      exact interaction_sub_abs_le_one β hβ u v) x y
  simpa only [neg_mul, sub_neg_eq_add] using h

/-- The actual-potential estimate has admissible temperature `β=1`. -/
example : (0 : ℝ) < 1 := one_pos

end Transformer.Metastability
