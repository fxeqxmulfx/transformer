module

public import Transformer.Phasor.TwoCopies.Constants

/-!
# Two copies: the margin of the construction

Source: `review/flat_spectrum.md`, *Functions of n variables*, NV8 (added), task T3c.

Between two tokens the logit's own-token term moves by at most `3 κ_emb`, and the answer's term
drops by at least `A (1 − cos(2π/p))`, which is positive for `p ≥ 2` (`one_sub_cos_pos`). The bias
`b = 5/(s c)`, `c = 1 − cos(2π/p)`, keeps `κ_emb < √D/b = s c √D/5`, and the gain of the second
branch makes its product `G = √(Dε) + 1`, so that the answer's amplitude
`A = s √D G/√(G² + Dε) > (3/5) s √D`. Together, `3 κ_emb < A c` (`three_kEmb_lt`), at every
`ε ≥ 0`, depth scale `s > 0` and width `D > 0`.
-/

@[expose] public section

open Real

namespace Transformer.Phasor.TwoCopies

variable {D p : ℕ}

/-- Two different tokens are apart on the circle: `1 − cos(2π/p) > 0` for `p ≥ 2`. -/
theorem one_sub_cos_pos (hp : 2 ≤ p) : 0 < 1 - cos (2 * π / p) := by
  have hp' : (2 : ℝ) ≤ p := by exact_mod_cast hp
  have hle : 2 * π / p ≤ π := by
    rw [div_le_iff₀ (by linarith)]; nlinarith [pi_pos]
  have h := cos_lt_cos_of_nonneg_of_le_pi le_rfl hle (div_pos (by positivity) (by linarith))
  rw [cos_zero] at h
  linarith

example : 0 < 1 - cos (2 * π / (29 : ℕ)) := one_sub_cos_pos (by norm_num)

/-- **The margin of the construction**: with the bias `b = 5/(s c)` and the gain that makes the
second branch's product `√(D ε) + 1`, the answer's amplitude beats three embeddings,
`3 κ_emb < A c`. -/
theorem three_kEmb_lt (hD : 0 < D) {eps s c : ℝ} (heps : 0 ≤ eps) (hs : 0 < s) (hc : 0 < c) :
    3 * kEmb D eps (5 / (s * c)) < ampOut D eps (5 / (s * c)) s
      ((Real.sqrt (D * eps) + 1) / ampProd D eps (5 / (s * c)) s) * c := by
  have hb : 0 < 5 / (s * c) := div_pos (by norm_num) (mul_pos hs hc)
  have hDpos : (0 : ℝ) < D := Nat.cast_pos.2 hD
  have hDe : 0 ≤ (D : ℝ) * eps := mul_nonneg hDpos.le heps
  have hr0 : 0 ≤ Real.sqrt (D * eps) := Real.sqrt_nonneg _
  have hr2 : Real.sqrt (D * eps) ^ 2 = D * eps := Real.sq_sqrt hDe
  have hG : (Real.sqrt (D * eps) + 1) / ampProd D eps (5 / (s * c)) s *
      ampProd D eps (5 / (s * c)) s = Real.sqrt (D * eps) + 1 :=
    div_mul_cancel₀ _ (ampProd_pos hD heps hs).ne'
  simp only [ampOut, kEmb, rmsScale, hG]
  set r := Real.sqrt (D * eps)
  set S := Real.sqrt D
  have hS : 0 < S := Real.sqrt_pos.2 hDpos
  have hX : 5 / (s * c) < Real.sqrt ((5 / (s * c)) ^ 2 + 2 + D * eps) := by
    rw [Real.lt_sqrt hb.le]; linarith
  have hY : 0 < Real.sqrt ((r + 1) ^ 2 + D * eps) :=
    Real.sqrt_pos.2 (add_pos_of_pos_of_nonneg (pow_pos (by linarith) 2) hDe)
  have hY5 : Real.sqrt ((r + 1) ^ 2 + D * eps) < 5 / 3 * (r + 1) := by
    rw [Real.sqrt_lt' (by positivity)]; nlinarith [sq_nonneg r]
  have h1 : 3 * (S / Real.sqrt ((5 / (s * c)) ^ 2 + 2 + D * eps)) < 3 / 5 * (s * c * S) := by
    have h := div_lt_div_of_pos_left hS hb hX
    have h' : S / (5 / (s * c)) = s * c * S / 5 := by field_simp
    linarith
  have h35 : 3 / 5 < (r + 1) / Real.sqrt ((r + 1) ^ 2 + D * eps) := by
    rw [lt_div_iff₀ hY]; linarith
  have h2 := mul_lt_mul_of_pos_left h35 (mul_pos (mul_pos hs hc) hS)
  calc 3 * (S / Real.sqrt ((5 / (s * c)) ^ 2 + 2 + D * eps)) < 3 / 5 * (s * c * S) := h1
    _ = s * c * S * (3 / 5) := by ring
    _ < s * c * S * ((r + 1) / Real.sqrt ((r + 1) ^ 2 + D * eps)) := h2
    _ = s * (S / Real.sqrt ((r + 1) ^ 2 + D * eps)) * (r + 1) * c := by ring

/-- Main's width `D = 128`, `ε = 2⁻²³` and depth scale `1/√8`, at `p = 29`. -/
example : 3 * kEmb 128 (1 / 8388608) (5 / (1 / Real.sqrt 8 * (1 - cos (2 * π / (29 : ℕ))))) <
    ampOut 128 (1 / 8388608) (5 / (1 / Real.sqrt 8 * (1 - cos (2 * π / (29 : ℕ)))))
        (1 / Real.sqrt 8)
      ((Real.sqrt ((128 : ℕ) * (1 / 8388608)) + 1) / ampProd 128 (1 / 8388608)
        (5 / (1 / Real.sqrt 8 * (1 - cos (2 * π / (29 : ℕ))))) (1 / Real.sqrt 8)) *
      (1 - cos (2 * π / (29 : ℕ))) :=
  three_kEmb_lt (by norm_num) (by norm_num) (by positivity) (one_sub_cos_pos (by norm_num))

end Transformer.Phasor.TwoCopies
