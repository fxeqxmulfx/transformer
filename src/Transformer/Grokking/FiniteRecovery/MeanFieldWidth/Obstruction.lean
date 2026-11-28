module

public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Literal
public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.ErrorBound

/-! The bound at the division width. The division run has width 64, so by
`run_literal` its floor in mean-field units is 64 · 10⁻⁸. At the peak rate
3/1000 one step of the Gronwall recursion multiplies the coupled distance
by at least 56251, and even at the floor rate 10⁻⁸ by at least 19/16.
`widthError` at width N is then at least 2 LΓ Lχ LG k Cφ (H+1) 56251^(H+2)/√N:
at N = 64 the bound is vacuous for every horizon unless one of these
constants vanishes. The factor 1/eps is not an artifact of the proof: from
zero moments the first direction is g/(|g| + eps), whose Lipschitz
constant in the gradient is at least 1/(2 eps). -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.MeanFieldWidth

open Transformer.AMSGradW.Kernite

/-- At the peak rate and the floor of width 64 one Gronwall step is at least 56251. -/
theorem growth_width64_peak {LG Lφ : ℝ} (hLG : 0 ≤ LG) (hLφ : 0 ≤ Lφ) :
    56251 ≤ growth (64 / 100000000) (3 / 1000) LG Lφ := by
  have : 0 ≤ LG * (1 + Lφ) := mul_nonneg hLG (by linarith)
  unfold growth
  norm_num
  linarith

/-- At the floor rate 10⁻⁸ and the floor of width 64 one Gronwall step is at least 19/16. -/
theorem growth_width64_floor {LG Lφ : ℝ} (hLG : 0 ≤ LG) (hLφ : 0 ≤ Lφ) :
    19 / 16 ≤ growth (64 / 100000000) (1 / 100000000) LG Lφ := by
  have : 0 ≤ LG * (1 + Lφ) := mul_nonneg hLG (by linarith)
  unfold growth
  norm_num
  nlinarith

theorem forcing_width64_peak (LG : ℝ) : forcing (64 / 100000000) (3 / 1000) LG = 56251 * LG := by
  unfold forcing
  norm_num

/-- The width error under the peak rate and the floor of width 64. -/
theorem widthError_width64_ge {LG Lφ Cφ LΓ Lχ Mχ Cχ : ℝ} (hLG : 0 ≤ LG) (hLφ : 0 ≤ Lφ)
    (hCφ : 0 ≤ Cφ) (hLΓ : 0 ≤ LΓ) (hLχ : 0 ≤ Lχ) (hMχ : 0 ≤ Mχ) (hCχ : 0 ≤ Cχ) (k j N B H n : ℕ) :
    2 * LΓ * Lχ * LG * k * Cφ * (H + 1) * 56251 ^ (H+2) / Real.sqrt N ≤
      widthError (64 / 100000000) (3 / 1000) LG Lφ Cφ LΓ Lχ Mχ Cχ k j N B H n := by
  have hG : (56251:ℝ) ^ (H+1) ≤ growth (64 / 100000000) (3 / 1000) LG Lφ ^ (H+1) :=
    pow_le_pow_left₀ (by norm_num) (growth_width64_peak hLG hLφ) _
  have hx : 0 ≤ ((H:ℝ) + 1) * (k * Cφ / Real.sqrt N) := by positivity
  have h1 : 2 * Lχ * ((56251:ℝ) ^ (H+1) * (56251 * LG) * ((H + 1) * (k * Cφ / Real.sqrt N))) ≤
      2 * Lχ * (growth (64 / 100000000) (3 / 1000) LG Lφ ^ (H+1) * forcing (64 / 100000000)
        (3 / 1000) LG * ((H + 1) * (k * Cφ / Real.sqrt N))) := by
    rw [forcing_width64_peak]
    gcongr
  have h2 : 0 ≤ ((H + 1 - B : ℕ) : ℝ) * ((n + 1) * (j * Cχ / Real.sqrt N)) := by positivity
  have h3 : 0 ≤ 2 * Mχ / (n:ℝ) := by positivity
  calc 2 * LΓ * Lχ * LG * k * Cφ * (H + 1) * 56251 ^ (H+2) / Real.sqrt N =
        LΓ * (2 * Lχ * ((56251:ℝ) ^ (H+1) * (56251 * LG) * ((H + 1) * (k * Cφ / Real.sqrt N)))) := by
        ring
    _ ≤ _ := by
        unfold widthError
        exact mul_le_mul_of_nonneg_left (by linarith) hLΓ

/-- At N = 64 the bound is at least LΓ Lχ LG k Cφ (H+1) 56251^(H+2)/4. -/
theorem widthError_width64 {LG Lφ Cφ LΓ Lχ Mχ Cχ : ℝ} (hLG : 0 ≤ LG) (hLφ : 0 ≤ Lφ)
    (hCφ : 0 ≤ Cφ) (hLΓ : 0 ≤ LΓ) (hLχ : 0 ≤ Lχ) (hMχ : 0 ≤ Mχ) (hCχ : 0 ≤ Cχ) (k j B H n : ℕ) :
    LΓ * Lχ * LG * k * Cφ * (H + 1) * 56251 ^ (H+2) / 4 ≤
      widthError (64 / 100000000) (3 / 1000) LG Lφ Cφ LΓ Lχ Mχ Cχ k j 64 B H n := by
  have h := widthError_width64_ge hLG hLφ hCφ hLΓ hLχ hMχ hCχ k j 64 B H n
  have h8 : Real.sqrt ((64:ℕ):ℝ) = 8 := by
    rw [show ((64:ℕ):ℝ) = 8^2 by norm_num]
    exact Real.sqrt_sq (by norm_num)
  rw [h8] at h
  calc _ = 2 * LΓ * Lχ * LG * k * Cφ * (H + 1) * 56251 ^ (H+2) / 8 := by ring
    _ ≤ _ := h

/-- From zero moments the first direction is g/(|g| + eps). -/
theorem first_direction (eps g : ℝ) : direction eps 0 (g / 10) (g^2 / 20) = g / (|g| + eps) := by
  have h20 : Real.sqrt 20 ≠ 0 := (Real.sqrt_pos.2 (by norm_num)).ne'
  have hs : Real.sqrt (g^2 / 20) / Real.sqrt (1 - (95/100 : ℝ)^(0+1)) = |g| := by
    rw [show (1 - (95/100 : ℝ)^(0+1)) = 1/20 by norm_num, Real.sqrt_div' _ (by norm_num : (0:ℝ) ≤ 20),
      Real.sqrt_sq_eq_abs, Real.sqrt_div' _ (by norm_num : (0:ℝ) ≤ 20), Real.sqrt_one]
    field_simp
  unfold direction denominator
  rw [hs, firstBias, show (1 - (9/10 : ℝ)^(0+1)) * (|g| + eps) = (|g| + eps) / 10 by ring,
    div_div_div_cancel_right₀ (by norm_num : (10:ℝ) ≠ 0)]

/-- Every Lipschitz constant of the first direction in the gradient is at least 1/(2 eps). -/
theorem first_direction_lipschitz {eps : ℝ} (he : 0 < eps) {L : ℝ}
    (hL : ∀ g g' : ℝ,
      |direction eps 0 (g / 10) (g^2 / 20) - direction eps 0 (g' / 10) (g'^2 / 20)| ≤ L * |g - g'|) :
    1 / (2 * eps) ≤ L := by
  have h := hL eps 0
  have h1 : eps / (|eps| + eps) = 1 / 2 := by
    rw [abs_of_pos he]
    field_simp
    ring
  rw [first_direction, first_direction, h1, abs_zero, zero_div, sub_zero, sub_zero, abs_of_pos he,
    abs_of_pos (by norm_num : (0:ℝ) < 1/2)] at h
  rw [div_le_iff₀ (by positivity)]
  linarith

section Examples

example := growth_width64_peak (LG := 1) (Lφ := 0) zero_le_one le_rfl
example := growth_width64_floor (LG := 1) (Lφ := 0) zero_le_one le_rfl
example := widthError_width64_ge (LG := 1) (Lφ := 0) (Cφ := 1) (LΓ := 1) (Lχ := 1) (Mχ := 0)
  (Cχ := 0) zero_le_one le_rfl zero_le_one zero_le_one zero_le_one le_rfl le_rfl 1 1 64 1 3 4
example := widthError_width64 (LG := 1) (Lφ := 0) (Cφ := 1) (LΓ := 1) (Lχ := 1) (Mχ := 0)
  (Cχ := 0) zero_le_one le_rfl zero_le_one zero_le_one zero_le_one le_rfl le_rfl 1 1 1 3 4
example := first_direction_lipschitz (eps := 1) one_pos (L := 4) fun g g' => by
  have hd := direction_lipschitz (eps := 1) one_pos 0 (m := g / 10) (v := g^2 / 20) (m' := g' / 10)
    (v' := g'^2 / 20) (by nlinarith [sq_nonneg g])
  have hs := sqrt_second_lipschitz (v := 0) (v' := 0) (g := g) (g' := g') le_rfl le_rfl
  rw [show (95/100 : ℝ) * 0 + 5/100 * g^2 = g^2 / 20 by ring,
    show (95/100 : ℝ) * 0 + 5/100 * g'^2 = g'^2 / 20 by ring, Real.sqrt_zero, sub_self, abs_zero,
    zero_add] at hs
  have hm : |g / 10 - g' / 10| = |g - g'| / 10 := by
    rw [← sub_div, abs_div]
    norm_num
  rw [hm, div_one] at hd
  linarith

end Examples

end Transformer.Grokking.FiniteRecovery.MeanFieldWidth
