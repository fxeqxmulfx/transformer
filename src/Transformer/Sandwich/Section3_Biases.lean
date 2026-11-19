module

public import Transformer.Sandwich.Section3_Transformer
public import Transformer.BAM.Section2_Priors

/-!
# §3.2–3.3 ALiBi and windowed attention

Chi, Fan, Rudnicky, Ramadge, arXiv:2212.10356v2, §3.2 (Eq. 2), §3.3 and Figures 1–2.

* ALiBi (Eq. 2) subtracts `(m − n)/2^h` from the logit of the head with exponent `h` before the
  scaling of Eq. (1); the exponents are `h = n · 8/H` for the heads `n = 1, …, H` (Figure 1).
  After the scaling its bias is `−(m − n)/(2^h √k)` (`alibiBias`, `alibi_logit`), the bias
  `Transformer.BAM.alibi` of slope `1/(2^h √k)` (`alibiBias_eq_alibi`). The slopes `1/2^h` of
  consecutive heads have the ratio `2^{−8/H}`, the "geometric sequence" of §4.2
  (`slope_exponent_succ`).
* Windowed attention of size `w` (§3.3, Figure 2) keeps the key `n` of the query `m` when
  `n + w > m ≥ n` and masks it otherwise (`windowBias`, `window_logit`).
* `IsWindowed w b`: the bias masks every distance `≥ w`. Windowed attention is windowed; so is
  any bias cut off at `w`, such as ALiBi or Sandwich with the cache window of Appendix B.
-/

@[expose] public section

namespace Transformer.Sandwich

open Finset Matrix

variable {d k : ℕ}

/-- ALiBi's slope `1/2^h` (§1, Figure 1). -/
noncomputable def slope (h : ℝ) : ℝ := 1 / 2 ^ h

/-- ALiBi's exponent `h = n · 8/H` of the head `n = j + 1` (Figure 1, §3.2). -/
noncomputable def exponent (H : ℕ) (j : Fin H) : ℝ := ((j : ℝ) + 1) * 8 / H

/-- ALiBi's bias after the scaling of Eq. (1): `−(m − n)/(2^h √k)` at the distance `m − n`. -/
noncomputable def alibiBias (k : ℕ) (h : ℝ) (t : ℕ) : EReal :=
  ((-(slope h * t) / Real.sqrt k : ℝ) : EReal)

/-- ALiBi with `H` heads of dimension `k` (§3.2): head `j` has the exponent `exponent H j`. -/
noncomputable def alibiHeads (k H : ℕ) : Fin H → ℕ → EReal := fun j => alibiBias k (exponent H j)

/-- Windowed attention of size `w` (§3.3): the bias `0` at the distances `m − n < w`, `−∞`
beyond. -/
noncomputable def windowBias (w t : ℕ) : EReal := if t < w then 0 else ⊥

/-- The bias masks every distance `≥ w`. -/
def IsWindowed (w : ℕ) (b : ℕ → EReal) : Prop := ∀ t, w ≤ t → b t = ⊥

/-- Windowed attention of size `w` masks every distance `≥ w`. -/
theorem isWindowed_windowBias (w : ℕ) : IsWindowed w (windowBias w) := fun t ht => by
  simp [windowBias, not_lt.mpr ht]

/-- **Eq. (2) with Eq. (1)**: ALiBi's logit `⟨q_m, k_n⟩ − (m − n)/2^h`, scaled by `1/√k`, is the
score with the bias `alibiBias k h`. -/
theorem alibi_logit (A : Head d k) (h : ℝ) (x : Embs d) (m n : ℕ) :
    ((((A.Wq *ᵥ x m) ⬝ᵥ (A.Wk *ᵥ x n) - slope h * (m - n : ℕ)) / Real.sqrt k : ℝ) : EReal) =
      score A (alibiBias k h) x m n := by
  simp only [score, alibiBias, ← EReal.coe_add]
  congr 1
  ring

/-- ALiBi's bias is `Transformer.BAM.alibi` with the slope `1/(2^h √k)`. -/
theorem alibiBias_eq_alibi (k : ℕ) (h : ℝ) (t : ℕ) :
    alibiBias k h t = Transformer.BAM.alibi (slope h / Real.sqrt k) t := by
  simp only [alibiBias, Transformer.BAM.alibi, Nat.abs_cast]
  congr 1
  ring

/-- **The slopes form a geometric sequence** (§4.2): the head `n + 1` has the slope of the head
`n` times `2^{−8/H}`. -/
theorem slope_exponent_succ {H : ℕ} (j : ℕ) (hj : j + 1 < H) :
    slope (exponent H ⟨j + 1, hj⟩) = slope (exponent H ⟨j, by omega⟩) * slope (8 / H) := by
  have h : exponent H ⟨j + 1, hj⟩ = exponent H ⟨j, by omega⟩ + 8 / H := by
    simp only [exponent, Nat.cast_add, Nat.cast_one]
    ring
  rw [h, slope, slope, slope, Real.rpow_add two_pos, one_div_mul_one_div]

/-- The hypothesis of `slope_exponent_succ` holds for the first two of `H = 8` heads, whose slopes
are `1/2` and `1/4`. -/
example :
    slope (exponent 8 ⟨1, by norm_num⟩) = slope (exponent 8 ⟨0, by norm_num⟩) * slope (8 / 8) :=
  slope_exponent_succ 0 (by norm_num)

/-- **Windowed attention** (§3.3): the key `n ≤ m` keeps its scaled logit when `n + w > m`, and is
masked by `−∞` otherwise. -/
theorem window_logit (A : Head d k) (w : ℕ) (x : Embs d) {m n : ℕ} (hn : n ≤ m) :
    score A (windowBias w) x m n =
      if m < n + w then (((A.Wq *ᵥ x m) ⬝ᵥ (A.Wk *ᵥ x n) / Real.sqrt k : ℝ) : EReal) else ⊥ := by
  unfold score windowBias
  by_cases h : m < n + w
  · rw [ite_eq_left (by omega : m - n < w), ite_eq_left h, add_zero]
  · rw [ite_eq_right (by omega : ¬m - n < w), ite_eq_right h, EReal.add_bot]

/-- The hypothesis of `window_logit` holds for the query and key `0`. -/
example (A : Head d k) (w : ℕ) (x : Embs d) :
    score A (windowBias w) x 0 0 =
      if 0 < 0 + w then (((A.Wq *ᵥ x 0) ⬝ᵥ (A.Wk *ᵥ x 0) / Real.sqrt k : ℝ) : EReal) else ⊥ :=
  window_logit A w x le_rfl

end Transformer.Sandwich
