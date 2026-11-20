module

public import Transformer.RoPE.RoFormer.Section3_Properties
public import Transformer.RoPE.YaRN.Section3_Ramp

/-!
# YaRN §2: RoPE in complex coordinates, wavelengths and Position Interpolation

Peng et al., *YaRN*, arXiv:2309.00071v3, §§2.1–2.3 and Appendix A.1. A vector of a head is
`x : Fin 2 × Fin p → ℝ`, its pair `l` made of `x (0, l)` and `x (1, l)`; RoFormer's `rotary θ m`
is RoPE at position `m`, and `frequency b |D| d = b^{−2d/|D|}` is Eq. (6)'s `θ_d`.

* `toComplex`, `dotProduct_eq_re`: Eqs. (3)–(4) identify `ℝ^{|D|}` with `ℂ^{|D|/2}`, "where the
  inner product `qᵀk` becomes the real part of the standard Hermitian inner product `Re(q*k)`";
* `toComplex_rotary`: Eq. (5), `f_W(x_m, m, θ) = e^{imθ} W x_m`: RoPE multiplies the pair `l` by
  `e^{imθ_l}`;
* `rotary_dotProduct_eq_re`: "the dot product between the query vector and the key vector only
  depends on the relative distance `m − n`": it is `Re Σ_l conj(q_l) k_l e^{i(n−m)θ_l}`;
* `rotary_div`: Eq. (7) with `g(m) = m/s`, `h(θ) = θ` and with `g(m) = m`, `h(θ) = θ/s` give the
  same embedding, so the `θ_d/s` of Eq. (13) is Position Interpolation;
* `pi_position_lt`, `printed_pi_position`: Eq. (16), PI's `g(m) = mL/L' = m/s`, keeps every
  position of the extended window `[0, L')` inside the pretrained window `[0, L)`. Eq. (9) prints
  `g(m) = s·m`, which sends the position `L` of the extended window to `L' > L`;
* `period_frequency`, `rotation_period`: Eq. (8), `λ_d = 2π/θ_d = 2πb^{2d/|D|}`, is "the length of
  tokens needed in order for the rotary position embedding at dimension `d` to perform a full
  rotation": the pair repeats after `λ_d` tokens and after no shorter positive shift;
* `rotations_frequency`: Eq. (10), `r(d) = L/λ_d = L/(2πb^{2d/|D|})`.
-/

@[expose] public section

open Matrix ComplexConjugate

namespace Transformer.RoPE.YaRN

open RoFormer

variable {p : ℕ}

/-- Eqs. (3)–(4): the pair `l` of `x` as the complex number `x_{0,l} + i x_{1,l}`. -/
def toComplex (x : Fin 2 × Fin p → ℝ) (l : Fin p) : ℂ :=
  ⟨x (0, l), x (1, l)⟩

/-- §2.1: under Eqs. (3)–(4) "the inner product `qᵀk` becomes the real part of the standard
Hermitian inner product `Re(q*k)`". -/
theorem dotProduct_eq_re (q k : Fin 2 × Fin p → ℝ) :
    q ⬝ᵥ k = (∑ l, conj (toComplex q l) * toComplex k l).re := by
  simp only [dotProduct, Fintype.sum_prod_type, Fin.sum_univ_two, ← Finset.sum_add_distrib,
    Complex.re_sum, Complex.mul_re, Complex.conj_re, Complex.conj_im, toComplex]
  refine Finset.sum_congr rfl fun l _ => ?_
  ring

/-- Eq. (5): `f_W(x_m, m, θ) = e^{imθ} W x_m`; RoPE at position `m` multiplies the pair `l` by
`e^{imθ_l}`. -/
theorem toComplex_rotary (θ : Fin p → ℝ) (m : ℝ) (x : Fin 2 × Fin p → ℝ) (l : Fin p) :
    toComplex (rotary θ m *ᵥ x) l = Complex.exp ((m * θ l : ℝ) * Complex.I) * toComplex x l := by
  rw [rotary_mulVec]
  apply Complex.ext <;>
    simp only [toComplex, Complex.mul_re, Complex.mul_im, Complex.exp_ofReal_mul_I_re,
      Complex.exp_ofReal_mul_I_im] <;>
    simp [rotateHalf] <;> ring

/-- §2.1: "the dot product between the query vector and the key vector only depends on the
relative distance `m − n`": `(R^m q)ᵀ(R^n k) = Re Σ_l conj(q_l) k_l e^{i(n−m)θ_l}`. -/
theorem rotary_dotProduct_eq_re (θ : Fin p → ℝ) (q k : Fin 2 × Fin p → ℝ) (m n : ℝ) :
    (rotary θ m *ᵥ q) ⬝ᵥ (rotary θ n *ᵥ k) =
      (∑ l, conj (toComplex q l) * toComplex k l *
        Complex.exp (((n - m) * θ l : ℝ) * Complex.I)).re := by
  rw [rotary_dotProduct, dotProduct_eq_re]
  congr 1
  refine Finset.sum_congr rfl fun l _ => ?_
  rw [toComplex_rotary]
  ring

/-- Eq. (7) for Position Interpolation: the position `g(m) = m/s` at the frequencies `θ` is the
position `m` at the frequencies `h(θ) = θ/s`. -/
theorem rotary_div (θ : Fin p → ℝ) (m s : ℝ) :
    rotary θ (m / s) = rotary (fun l => θ l / s) m := by
  unfold rotary
  congr 1
  funext l
  rw [div_mul_eq_mul_div, mul_div_assoc]

/-- Eq. (16): PI moves the position `m` to `mL/L' = m/s`, `s = L'/L`, and every position of the
extended window `[0, L')` lands in the pretrained window `[0, L)`. -/
theorem pi_position_lt {L L' m : ℝ} (hL : 0 < L) (hm0 : 0 ≤ m) (hm : m < L') :
    m * L / L' = m / (L' / L) ∧ m * L / L' < L := by
  have hL' : 0 < L' := lt_of_le_of_lt hm0 hm
  refine ⟨(div_div_eq_mul_div m L' L).symm, ?_⟩
  rw [div_lt_iff₀ hL']
  nlinarith

example : (1000 : ℝ) * 2048 / 4096 = 1000 / (4096 / 2048) ∧ (1000 : ℝ) * 2048 / 4096 < 2048 :=
  pi_position_lt (by norm_num) (by norm_num) (by norm_num)

/-- Eq. (9) prints PI as `g(m) = s·m`, `s = L'/L`. That map sends the position `L` of the
extended window `[0, L')` out of the pretrained window `[0, L)`; Eq. (16)'s `m/s` keeps it in. -/
theorem printed_pi_position {L L' : ℝ} (hL : 0 < L) (hLL : L < L') :
    ¬ L' / L * L < L ∧ L / (L' / L) < L := by
  have hL' : 0 < L' := hL.trans hLL
  refine ⟨by rw [div_mul_cancel₀ _ hL.ne']; exact not_lt.mpr hLL.le, ?_⟩
  rw [div_div_eq_mul_div, div_lt_iff₀ hL']
  nlinarith

example : ¬ (4096 : ℝ) / 2048 * 2048 < 2048 ∧ (2048 : ℝ) / (4096 / 2048) < 2048 :=
  printed_pi_position (by norm_num) (by norm_num)

/-- Eq. (8): `λ_d = 2π/θ_d = 2πb^{2d/|D|}`. -/
theorem period_frequency {base : ℝ} (hbase : 0 < base) (D d : ℕ) :
    period (frequency base D d) = 2 * Real.pi * base ^ (2 * (d : ℝ) / D) := by
  rw [period, frequency, neg_div, Real.rpow_neg hbase.le, div_inv_eq_mul]

example : period (frequency 10000 128 3) = 2 * Real.pi * 10000 ^ (2 * ((3 : ℕ) : ℝ) / 128) :=
  period_frequency (by norm_num) 128 3

/-- §2.2: the rotation of a pair of frequency `ω > 0` repeats after `λ = 2π/ω` tokens, and after no
shorter positive shift. -/
theorem rotation_period {ω : ℝ} (hω : 0 < ω) (m : ℝ) :
    rotation ((m + period ω) * ω) = rotation (m * ω) ∧
      ∀ t, 0 < t → t < period ω → rotation (t * ω) ≠ 1 := by
  refine ⟨?_, fun t ht htp h => ?_⟩
  · rw [add_mul, period, div_mul_cancel₀ _ hω.ne']
    simp [rotation]
  · have hc : Real.cos (t * ω) = 1 := by
      simpa [rotation] using congr_fun (congr_fun h 0) 0
    have hlt : t * ω < 2 * Real.pi := by
      rwa [period, lt_div_iff₀ hω] at htp
    have h0 := (Real.cos_eq_one_iff_of_lt_of_lt (by nlinarith [Real.pi_pos]) hlt).mp hc
    nlinarith

example : rotation ((5 + period 1) * 1) = rotation (5 * 1) ∧
    ∀ t, 0 < t → t < period 1 → rotation (t * 1) ≠ 1 :=
  rotation_period (by norm_num) 5

/-- Eq. (10): `r(d) = L/λ_d = L/(2πb^{2d/|D|})`, the rotations of the pair `d` over the
pretrained context. -/
theorem rotations_frequency {base : ℝ} (hbase : 0 < base) (L : ℝ) (D d : ℕ) :
    rotations L (frequency base D d) = L / period (frequency base D d) ∧
      rotations L (frequency base D d) = L / (2 * Real.pi * base ^ (2 * (d : ℝ) / D)) := by
  have h : rotations L (frequency base D d) = L / period (frequency base D d) := by
    rw [rotations, period, div_div_eq_mul_div]
  exact ⟨h, by rw [h, period_frequency hbase]⟩

example : rotations 2048 (frequency 10000 128 3) = 2048 / period (frequency 10000 128 3) ∧
    rotations 2048 (frequency 10000 128 3) =
      2048 / (2 * Real.pi * 10000 ^ (2 * ((3 : ℕ) : ℝ) / 128)) :=
  rotations_frequency (by norm_num) 2048 128 3

end Transformer.RoPE.YaRN
