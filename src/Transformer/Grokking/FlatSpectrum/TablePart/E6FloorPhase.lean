module

public import Transformer.Grokking.FlatSpectrum.TablePart.E6CosTable

/-!
# PT6: a table of the powers of `ζ`, in units of `2⁻²⁰`

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. The table holds the
powers of the rational point `ẑ`, rounded to `2⁻²⁰`, in 194 fields of 22 bits. The kernel computes
`ẑ^r` exactly and checks each field, so the entry `r` lies within `4 (r + 1) / 2²⁰` of `ζ^r`
(`norm_pow_sub_phaseHat_le`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart.E6

open Complex

attribute [local instance] fact_prime_97

/-- The value `ψ(1)` of the character. -/
local notation "ζ" => (ZMod.stdAddChar (1 : ZMod 97) : ℂ)

/-- The rounded powers of `ẑ`: 97 real parts, then 97 imaginary parts, in fields of 22 bits, each
shifted up by `2²¹`. -/
def phasePack : ℕ := fromWords 368 [
  0x2627a417ca9dc0caa9e42b67a3b06de2cc24cb56dbade638b98856ece81bcac1ef7743becb96fddabbfdda300000,
  0x38584d2864e6cda2d856c9e5dc26de74a919e30b97ce47e04259852d36252dd8d5fbe459699558e664d4d9cf6768,
  0x74009710025b404e19035cb41a40d0acf9440591165e847686925d7d4bbe1138f9250f3f14f70656f3a968dd35da,
  0x939ad613491768df5a37495bcea53dc1d43cfd4e3e512ef874976011da1d4597ad1016742b3f10690540d7310138,
  0x6f26905b4a097ba73d689934725560c916506357ca894af214b30810901f390878c299d2a36709b45b278d68b5f5,
  0xf388449e10922800002ff6febf750afb27ebddb7af2aa4bb3886e61b8b7977ad5b17b307e6c1b29ad9d62aa749a7,
  0xaf1558bad346e447fb7136ed366fb2674abefaeace2e2a75b6a69de28cedd9fb33270239984082517839065be316,
  0xe56dec465bb3a74ad7f6bb81a1ae7e92bb9c36f3f89be1e96fbf14bf966effb10bffd72ff27bbf51eafa57abd96b,
  0x704719dadd6baf79cae1f7b6b8212b618d618a51aa398ae92645a5948e9a28227787da17bb693c5fa83f9ead8a2a,
  0x7d0788d4302a1119224606711f9a64a02fd316564ee72d46a6e549e095f03a5b0f057a132621e819761469ae35b6,
  0x5629b94c76f5041f13665b4b270523b5646eed114b6343ab7509a88416af102bbc4036ed000da40149501a9a4104,
  0x7bdb31deece73a421be6936ba2119efed63f7658134a5cc4f9658a7]

/-- The real part of the entry `r`, in units of `2⁻²⁰`. -/
def phC (r : ℕ) : ℤ := (((phasePack >>> (22 * r)) &&& 4194303 : ℕ) : ℤ) - 2097152

/-- The imaginary part of the entry `r`, in units of `2⁻²⁰`. -/
def phS (r : ℕ) : ℤ := (((phasePack >>> (22 * (97 + r))) &&& 4194303 : ℕ) : ℤ) - 2097152

/-- **The kernel checks the real parts against the exact powers of `ẑ`.** -/
theorem phC_check : ∀ r, r < 97 →
    -(10 ^ (9 * r) : ℤ) ≤ phC r * 10 ^ (9 * r) - 2 ^ 20 * (zPow r).1 ∧
      phC r * 10 ^ (9 * r) - 2 ^ 20 * (zPow r).1 ≤ 10 ^ (9 * r) := by
  decide +kernel

/-- **The kernel checks the imaginary parts against the exact powers of `ẑ`.** -/
theorem phS_check : ∀ r, r < 97 →
    -(10 ^ (9 * r) : ℤ) ≤ phS r * 10 ^ (9 * r) - 2 ^ 20 * (zPow r).2 ∧
      phS r * 10 ^ (9 * r) - 2 ^ 20 * (zPow r).2 ≤ 10 ^ (9 * r) := by
  decide +kernel

/-- The entry `r` of the table, a point next to `ζ ^ r`. -/
noncomputable def phaseHat (r : ℕ) : ℂ := ⟨(phC r : ℝ) / 2 ^ 20, (phS r : ℝ) / 2 ^ 20⟩

/-- A rounding in integers is a rounding of the quotients. -/
lemma abs_div_sub_div_le {A C P : ℝ} (hP : 0 < P) (h1 : -P ≤ C * P - 2 ^ 20 * A)
    (h2 : C * P - 2 ^ 20 * A ≤ P) : |A / P - C / 2 ^ 20| ≤ 1 / 2 ^ 20 := by
  have e : A / P - C / 2 ^ 20 = -(C * P - 2 ^ 20 * A) / P / 2 ^ 20 := by
    field_simp
    ring
  rw [e, abs_div, abs_div, abs_neg, abs_of_pos hP, abs_of_pos (by positivity : (0 : ℝ) < 2 ^ 20)]
  refine div_le_div_of_nonneg_right ?_ (by positivity)
  exact (div_le_one hP).2 (abs_le.2 ⟨h1, h2⟩)

example : |(1 : ℝ) / 1 - 1048576 / 2 ^ 20| ≤ 1 / 2 ^ 20 :=
  abs_div_sub_div_le (by norm_num) (by norm_num) (by norm_num)

/-- **The table is within `4 (r + 1) / 2²⁰` of `ζ^r`.** -/
theorem norm_pow_sub_phaseHat_le {r : ℕ} (hr : r < 97) :
    ‖ζ ^ r - phaseHat r‖ ≤ 4 * (r + 1) / 2 ^ 20 := by
  have hP : (0 : ℝ) < 10 ^ (9 * r) := by positivity
  obtain ⟨c1, c2⟩ := phC_check r hr
  obtain ⟨s1, s2⟩ := phS_check r hr
  have hc1 : -((10 : ℝ) ^ (9 * r)) ≤ (phC r : ℝ) * 10 ^ (9 * r) - 2 ^ 20 * ((zPow r).1 : ℝ) := by
    exact_mod_cast c1
  have hc2 : (phC r : ℝ) * 10 ^ (9 * r) - 2 ^ 20 * ((zPow r).1 : ℝ) ≤ 10 ^ (9 * r) := by
    exact_mod_cast c2
  have hs1 : -((10 : ℝ) ^ (9 * r)) ≤ (phS r : ℝ) * 10 ^ (9 * r) - 2 ^ 20 * ((zPow r).2 : ℝ) := by
    exact_mod_cast s1
  have hs2 : (phS r : ℝ) * 10 ^ (9 * r) - 2 ^ 20 * ((zPow r).2 : ℝ) ≤ 10 ^ (9 * r) := by
    exact_mod_cast s2
  have hre : |(zHat ^ r - phaseHat r).re| ≤ 1 / 2 ^ 20 := by
    rw [Complex.sub_re, (zHat_pow r).1]
    exact abs_div_sub_div_le hP hc1 hc2
  have him : |(zHat ^ r - phaseHat r).im| ≤ 1 / 2 ^ 20 := by
    rw [Complex.sub_im, (zHat_pow r).2]
    exact abs_div_sub_div_le hP hs1 hs2
  have h5 : ‖zHat ^ r - phaseHat r‖ ≤ 1 / 2 ^ 20 + 1 / 2 ^ 20 :=
    (Complex.norm_le_abs_re_add_abs_im _).trans (add_le_add hre him)
  have hz : ‖ζ‖ ≤ 1 := by simp
  have h3 := norm_pow_sub_pow_le hz norm_zHat_le r
  have h4 : (r : ℝ) * ‖ζ - zHat‖ ≤ r * (3 / 10 ^ 6) :=
    mul_le_mul_of_nonneg_left norm_zeta_sub_le (Nat.cast_nonneg r)
  have hd := norm_sub_le_norm_sub_add_norm_sub (ζ ^ r) (zHat ^ r) (phaseHat r)
  have h6 : (r : ℝ) * (3 / 10 ^ 6) ≤ r * (4 / 2 ^ 20) :=
    mul_le_mul_of_nonneg_left (by norm_num) (Nat.cast_nonneg r)
  have h7 : 4 * ((r : ℝ) + 1) / 2 ^ 20 = r * (4 / 2 ^ 20) + 4 / 2 ^ 20 := by ring
  have h8 : (1 : ℝ) / 2 ^ 20 + 1 / 2 ^ 20 ≤ 4 / 2 ^ 20 := by norm_num
  rw [h7]
  linarith

example : ‖(ZMod.stdAddChar (1 : ZMod 97) : ℂ) ^ 5 - phaseHat 5‖ ≤ 4 * ((5 : ℕ) + 1) / 2 ^ 20 :=
  norm_pow_sub_phaseHat_le (by norm_num)

end Transformer.Grokking.FlatSpectrum.TablePart.E6
