module

public import Mathlib.Analysis.Real.Pi.Bounds
public import Transformer.Grokking.FlatSpectrum.TablePart.Cyclotomic
public import Transformer.Grokking.FlatSpectrum.TablePart.E6Lines

/-!
# PT6: certified lower bounds on `cos(2π r / 97)`

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. A rational point
`ẑ` of the unit disc lies within `3 · 10⁻⁶` of `ζ = ψ(1)`, by the Taylor bounds on the cosine and
the sine and the bounds on `π`. So `‖ζ^r − ẑ^r‖ ≤ 3 r · 10⁻⁶`. The kernel computes `ẑ^r` exactly
and checks a table of lower bounds on `10⁶ cos(2π r / 97)` (`re_stdAddChar_ge_cosLow`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart.E6

open Complex

attribute [local instance] fact_prime_97

/-- The value `ψ(1)` of the character. -/
local notation "ζ" => (ZMod.stdAddChar (1 : ZMod 97) : ℂ)

/-- The real part of `ẑ`, over `10⁹`. -/
def zx : ℤ := 997901792

/-- The imaginary part of `ẑ`, over `10⁹`. -/
def zy : ℤ := 64729508

/-- One multiplication by `zx + i zy`. -/
def zStep (q : ℤ × ℤ) : ℤ × ℤ := (q.1 * zx - q.2 * zy, q.1 * zy + q.2 * zx)

/-- The real and imaginary parts of `(zx + i zy)^r`. -/
def zPow : ℕ → ℤ × ℤ
  | 0 => (1, 0)
  | r + 1 => zStep (zPow r)

/-- The table of lower bounds, in fields of 21 bits, each shifted up by `1000600`. -/
def cosPack : ℕ := fromWords 336 [
  0xbef98e2ad9b2db07a24b6d681e6dbd73802d9c85cde7df9f58d7fb7743e045af1e9f79983bcfd3de86d6,
  0x466e926d985549cab9861647e6b62881d0c9af834783ffe45ed7a4e8bf369f4a2d8a551af6c57e7709be,
  0x009d400d1740ca7a0a61407b4e855a0038ab0242ad166fc0d9a908181a4bd822bd2199061ce24ee7ec34,
  0x3206c95e684979d840b831b3130b36184848e1c4fe0ab1903d96c14bbc0651301a1f804e580067800342,
  0xaa2a7d166ca6d13d273158bd1041fb29f0442e85336c489323b2d72ed6aa3fb4dac1a33400fd6dc711bd,
  0xf31f078ec73c0489db997eb0b0f3e807907cfbff92db6c76b39eb4460196bd9c54e45f767ee107162a79,
  0x1e7c64]

/-- A lower bound on `10⁶ cos(2π r / 97)`. -/
def cosLow (r : ℕ) : ℤ := (((cosPack >>> (21 * r)) &&& 2097151 : ℕ) : ℤ) - 1000600

/-- **The kernel checks the table against the exact powers of `ẑ`.** -/
theorem cosLow_check : ∀ r, r < 97 → (cosLow r + 3 * r) * 10 ^ (9 * r) ≤ 1000000 * (zPow r).1 := by
  decide +kernel

/-- The rational point `ẑ` next to `ζ`. -/
noncomputable def zHat : ℂ := ⟨(zx : ℝ) / 10 ^ 9, (zy : ℝ) / 10 ^ 9⟩

lemma norm_zHat_le : ‖zHat‖ ≤ 1 := by
  have h : ‖zHat‖ ^ 2 ≤ 1 := by
    rw [Complex.sq_norm, Complex.normSq_apply]
    simp only [zHat, zx, zy]
    norm_num
  exact (sq_le_one_iff₀ (norm_nonneg _)).1 h

lemma zeta_re_im : (ζ).re = Real.cos (2 * Real.pi / 97) ∧ (ζ).im = Real.sin (2 * Real.pi / 97) := by
  have h := ZMod.stdAddChar_coe (N := 97) 1
  have harg : (2 * Real.pi * I * ((1 : ℤ) : ℂ) / (97 : ℕ) : ℂ) =
      ((2 * Real.pi / 97 : ℝ) : ℂ) * I := by
    push_cast
    ring
  rw [harg, Int.cast_one] at h
  constructor
  · rw [h, Complex.exp_ofReal_mul_I_re]
  · rw [h, Complex.exp_ofReal_mul_I_im]

/-- The Taylor bounds at `θ = 2π / 97`, with the bounds on `π`. -/
lemma near_cos_sin :
    |Real.cos (2 * Real.pi / 97) - 997901792 / 10 ^ 9| ≤ 13 / 10 ^ 7 ∧
      |Real.sin (2 * Real.pi / 97) - 64729508 / 10 ^ 9| ≤ 13 / 10 ^ 7 := by
  have h1 := Real.pi_gt_d6
  have h2 := Real.pi_lt_d6
  set t := 2 * Real.pi / 97 with ht
  have hl : (0.0647750927 : ℝ) < t := by
    rw [ht]
    linarith
  have hu : t < 0.0647751135 := by
    rw [ht]
    linarith
  have h0 : (0 : ℝ) ≤ t := by linarith
  have habs : |t| = t := abs_of_nonneg h0
  have hone : |t| ≤ 1 := by
    rw [habs]
    linarith
  have hc := Real.cos_bound hone
  have hs := Real.sin_bound hone
  rw [habs] at hc hs
  have ht2l : (0.0647750927 : ℝ) ^ 2 ≤ t ^ 2 := pow_le_pow_left₀ (by norm_num) hl.le 2
  have ht2u : t ^ 2 ≤ (0.0647751135 : ℝ) ^ 2 := pow_le_pow_left₀ h0 hu.le 2
  have ht3l : (0.0647750927 : ℝ) ^ 3 ≤ t ^ 3 := pow_le_pow_left₀ (by norm_num) hl.le 3
  have ht3u : t ^ 3 ≤ (0.0647751135 : ℝ) ^ 3 := pow_le_pow_left₀ h0 hu.le 3
  have ht4u : t ^ 4 ≤ (0.0647751135 : ℝ) ^ 4 := pow_le_pow_left₀ h0 hu.le 4
  have ht5u : t ^ 5 ≤ (0.0647751135 : ℝ) ^ 5 := pow_le_pow_left₀ h0 hu.le 5
  obtain ⟨hc1, hc2⟩ := abs_le.1 hc
  obtain ⟨hs1, hs2⟩ := abs_le.1 hs
  norm_num at ht2l ht2u ht3l ht3u ht4u ht5u
  refine ⟨abs_le.2 ⟨?_, ?_⟩, abs_le.2 ⟨?_, ?_⟩⟩ <;> norm_num <;> linarith

lemma norm_zeta_sub_le : ‖ζ - zHat‖ ≤ 3 / 10 ^ 6 := by
  obtain ⟨hc, hs⟩ := near_cos_sin
  obtain ⟨hre, him⟩ := zeta_re_im
  have h1 : (ζ - zHat).re = Real.cos (2 * Real.pi / 97) - 997901792 / 10 ^ 9 := by
    rw [Complex.sub_re, hre]
    simp [zHat, zx]
  have h2 : (ζ - zHat).im = Real.sin (2 * Real.pi / 97) - 64729508 / 10 ^ 9 := by
    rw [Complex.sub_im, him]
    simp [zHat, zy]
  calc ‖ζ - zHat‖ ≤ |(ζ - zHat).re| + |(ζ - zHat).im| := Complex.norm_le_abs_re_add_abs_im _
    _ ≤ 13 / 10 ^ 7 + 13 / 10 ^ 7 := by
      rw [h1, h2]
      exact add_le_add hc hs
    _ ≤ 3 / 10 ^ 6 := by norm_num

/-- Powers of two points of the unit disc stay within `n` times their distance. -/
lemma norm_pow_sub_pow_le {a b : ℂ} (ha : ‖a‖ ≤ 1) (hb : ‖b‖ ≤ 1) (n : ℕ) :
    ‖a ^ n - b ^ n‖ ≤ n * ‖a - b‖ := by
  induction n with
  | zero => simp
  | succ n ih =>
    have e : a ^ (n + 1) - b ^ (n + 1) = (a ^ n - b ^ n) * a + b ^ n * (a - b) := by ring
    have hbn : ‖b‖ ^ n ≤ 1 := pow_le_one₀ (norm_nonneg _) hb
    have hd := norm_nonneg (a - b)
    calc ‖a ^ (n + 1) - b ^ (n + 1)‖
        ≤ ‖a ^ n - b ^ n‖ * ‖a‖ + ‖b‖ ^ n * ‖a - b‖ := by
          rw [e]
          refine (norm_add_le _ _).trans ?_
          rw [norm_mul, norm_mul, norm_pow]
      _ ≤ (n * ‖a - b‖) * 1 + 1 * ‖a - b‖ := by
          gcongr
      _ = ((n + 1 : ℕ) : ℝ) * ‖a - b‖ := by
          push_cast
          ring

example : ‖(1 : ℂ) ^ 3 - (1 : ℂ) ^ 3‖ ≤ (3 : ℕ) * ‖(1 : ℂ) - 1‖ :=
  norm_pow_sub_pow_le (by simp) (by simp) 3

/-- The exact powers of `ẑ`, by the integer recursion. -/
lemma zHat_pow (r : ℕ) : (zHat ^ r).re = ((zPow r).1 : ℝ) / 10 ^ (9 * r) ∧
    (zHat ^ r).im = ((zPow r).2 : ℝ) / 10 ^ (9 * r) := by
  induction r with
  | zero => simp [zPow]
  | succ r ih =>
    obtain ⟨h1, h2⟩ := ih
    have e : (10 : ℝ) ^ (9 * (r + 1)) = 10 ^ (9 * r) * 10 ^ 9 := by ring
    have hp : (10 : ℝ) ^ (9 * r) ≠ 0 := by positivity
    rw [pow_succ, Complex.mul_re, Complex.mul_im, h1, h2, e]
    simp only [zPow, zStep, zHat]
    push_cast
    constructor <;> field_simp

/-- **The table bounds the cosine from below.** -/
theorem cosLow_le {r : ℕ} (hr : r < 97) : (cosLow r : ℝ) / 10 ^ 6 ≤ (ζ ^ r).re := by
  have h1 : (((cosLow r + 3 * r) * 10 ^ (9 * r) : ℤ) : ℝ) ≤ ((1000000 * (zPow r).1 : ℤ) : ℝ) :=
    Int.cast_le.2 (cosLow_check r hr)
  push_cast at h1
  have hp : (0 : ℝ) < 10 ^ (9 * r) := by positivity
  have h2 : ((cosLow r : ℝ) + 3 * r) / 10 ^ 6 ≤ (zHat ^ r).re := by
    rw [(zHat_pow r).1, div_le_div_iff₀ (by positivity) hp]
    linarith
  have hz : ‖ζ‖ ≤ 1 := by simp
  have h3 := norm_pow_sub_pow_le hz norm_zHat_le r
  have h4 : (r : ℝ) * ‖ζ - zHat‖ ≤ r * (3 / 10 ^ 6) :=
    mul_le_mul_of_nonneg_left norm_zeta_sub_le (Nat.cast_nonneg r)
  have h5 := (abs_le.1 (Complex.abs_re_le_norm (ζ ^ r - zHat ^ r))).1
  rw [Complex.sub_re] at h5
  have h6 : ((cosLow r : ℝ) + 3 * r) / 10 ^ 6 = (cosLow r : ℝ) / 10 ^ 6 + r * (3 / 10 ^ 6) := by
    ring
  linarith

example : (cosLow 0 : ℝ) / 10 ^ 6 ≤ ((ZMod.stdAddChar (1 : ZMod 97) : ℂ) ^ 0).re :=
  cosLow_le (by norm_num)

/-- **A lower bound on the real part of every value of the character.** -/
theorem re_stdAddChar_ge_cosLow (x : ZMod 97) :
    (cosLow x.val : ℝ) / 10 ^ 6 ≤ (ZMod.stdAddChar x : ℂ).re := by
  have h := cosLow_le (ZMod.val_lt x)
  rwa [← stdAddChar_natCast, ZMod.natCast_zmod_val] at h

example : (cosLow 0 : ℝ) / 10 ^ 6 ≤ (ZMod.stdAddChar (0 : ZMod 97) : ℂ).re :=
  re_stdAddChar_ge_cosLow 0

end Transformer.Grokking.FlatSpectrum.TablePart.E6
