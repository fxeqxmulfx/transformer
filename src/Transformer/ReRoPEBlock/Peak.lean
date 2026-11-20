module

public import Transformer.ReRoPEBlock.PiHead

/-!
# ReRoPE's previous-token head keeps its weight at every length

Source: the blocks of `Transformer.ReRoPEBlock.Defs` (DiffGPT `5a37c51`, ReRoPE of
kexue.fm/archives/9708). The peak head is one ReRoPE head of the main model with the window `W`,
every frequency `φ = π/(W − 1)` (`peakFrequency`), `β = 0`, the query `q` and the key `q` turned
by `1` for every token (`peakHead`). It scores the distance `δ` by
`κ ln(i + 1) cos((1 − min(δ, W)) φ)`, `κ = qᵀq/d` (`headScore_peak`): `+κ ln N` at the
distance `1` and `−κ ln N` at every distance `≥ W`, `N = i + 1` keys.

* `peakHead_sub_permute_swap`: swapping the tokens of two keys changes the head by
  `(e^{s_a} − e^{s_b}) / ∑_j e^{s_j}` times the difference of their values;
* `le_norm_peakHead_sub`: at a query `D ≥ W`, `κ ≥ 1/2`, swapping the keys at the distances
  `1` and `D` changes it by at least `1/(2(W + 1))` times the difference of their values,
  whatever `D`. SSMax's gain `ln N` keeps the weight of the distance `1` against the `N` far
  keys: at most `W` keys lie in the window, none scoring above it, and each far key weighs
  `N^{−2κ} ≤ 1/N` of it.
-/

@[expose] public section

open Matrix
open Transformer.AttentionResiduals Transformer.RoPE.ReRoPE Transformer.RoPE.RoFormer

namespace Transformer.ReRoPEBlock

variable {X E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] {p : ℕ}

/-- The peak head's frequency for the window `W`: `π/(W − 1)`, half a turn from the distance `1`
to the distance `W`. -/
noncomputable def peakFrequency (W : ℕ) : ℝ := Real.pi / ((W : ℝ) - 1)

/-- **ReRoPE's peak head**: one head with every frequency `π/(W − 1)`, `β = 0`, the query `q` and
the key `q` turned by `1` for every token, at ReRoPE's positions with the window `W`, reading the
values `u`; the block returns its output. -/
noncomputable def peakHead (W : ℕ) (d : ℝ) (q : Fin 2 × Fin p → ℝ) (u : X → E) :
    (ℕ → X) → ℕ → E :=
  block (rePosition W) (fun _ a => a 0)
    (fun (_ : Fin 1) => headScore (fun _ => peakFrequency W) 0 d (fun _ => q)
      (fun _ => rotary (fun _ => peakFrequency W) 1 *ᵥ q))
    (fun _ => u)

/-- The peak head's score of the position `t`: `ln(i + 1) κ cos((1 − t) φ)`, `κ = qᵀq/d`. -/
theorem headScore_peak (φ d : ℝ) (q : Fin 2 × Fin p → ℝ) (i : ℕ) (x y : X) (t : ℝ) :
    headScore (fun _ => φ) 0 d (fun _ => q) (fun _ => rotary (fun _ => φ) 1 *ᵥ q) i x y t =
      Real.log ((i : ℝ) + 1) * (q ⬝ᵥ q / d) * Real.cos ((1 - t) * φ) := by
  simp only [headScore, phaseScore, zero_mul, zero_div, add_zero, mulVec_mulVec, ← rotary_add,
    dotProduct_rotary_eq_sum_cos, ← Finset.sum_mul]
  have hq : ∑ l, (q (0, l) ^ 2 + q (1, l) ^ 2) = q ⬝ᵥ q := by
    simp only [dotProduct, Fintype.sum_prod_type, Fin.sum_univ_two, ← Finset.sum_add_distrib, sq]
  rw [hq, show -t + 1 = 1 - t by ring]
  ring

/-- **Swapping two keys** `a ≠ b` before the query `D` changes the peak head by
`(e^{s(D−a)} − e^{s(D−b)}) / ∑_j e^{s(D−j)}` times `u(x_a) − u(x_b)`, where
`s δ = ln(D + 1) κ cos((1 − min(δ, W)) φ)`. -/
theorem peakHead_sub_permute_swap (W : ℕ) (d : ℝ) (q : Fin 2 × Fin p → ℝ) (u : X → E)
    (x : ℕ → X) {D : ℕ} {a b : Fin (D + 1)} (hab : a ≠ b) :
    peakHead W d q u x D - peakHead W d q u (permute (Equiv.swap a b) x) D =
      ((Real.exp (Real.log ((D : ℝ) + 1) * (q ⬝ᵥ q / d) *
            Real.cos ((1 - rePosition W (D - a)) * peakFrequency W)) -
          Real.exp (Real.log ((D : ℝ) + 1) * (q ⬝ᵥ q / d) *
            Real.cos ((1 - rePosition W (D - b)) * peakFrequency W))) /
        ∑ j : Fin (D + 1), Real.exp (Real.log ((D : ℝ) + 1) * (q ⬝ᵥ q / d) *
          Real.cos ((1 - rePosition W (D - j)) * peakFrequency W))) • (u (x a) - u (x b)) := by
  simp only [peakHead, block, headScore_peak, permute_apply]
  exact attend_sub_attend_swap _ (fun j : Fin (D + 1) => u (x j)) hab

/-- Inside the window at most `W` keys of the query `D`. -/
theorem card_filter_sub_lt_le (D W : ℕ) :
    (Finset.univ.filter fun j : Fin (D + 1) => D - (j : ℕ) < W).card ≤ W := by
  have h := Finset.card_le_card_of_injOn (fun j : Fin (D + 1) => D - (j : ℕ))
    (s := Finset.univ.filter fun j : Fin (D + 1) => D - (j : ℕ) < W) (t := Finset.range W)
    (fun j hj => by simpa using hj) (fun j _ j' _ h => by
      have := j.2; have := j'.2; exact Fin.ext (by simp only at h; omega))
  rwa [Finset.card_range] at h

/-- **The peak head keeps its weight at every length**: at a query `D ≥ W ≥ 2`, `0 < d`,
`κ = qᵀq/d ≥ 1/2`, swapping the keys at the distances `1` and `D` changes the peak head by at
least `1/(2(W + 1))` times the difference of their values. -/
theorem le_norm_peakHead_sub {W D : ℕ} (hW : 2 ≤ W) (hD : W ≤ D) {d : ℝ} (hd : 0 < d)
    (q : Fin 2 × Fin p → ℝ) (hq : d ≤ 2 * (q ⬝ᵥ q)) (u : X → E) (x : ℕ → X) :
    ‖u (x (D - 1)) - u (x 0)‖ / (2 * ((W : ℝ) + 1)) ≤
      ‖peakHead W d q u x D -
        peakHead W d q u (permute (Equiv.swap (⟨D - 1, by omega⟩ : Fin (D + 1)) 0) x) D‖ := by
  have hab : (⟨D - 1, by omega⟩ : Fin (D + 1)) ≠ 0 := fun h => by
    have := congrArg Fin.val h
    simp only [Fin.val_zero] at this
    omega
  rw [peakHead_sub_permute_swap W d q u x hab, norm_smul, Real.norm_eq_abs, div_eq_inv_mul,
    ← one_div]
  refine mul_le_mul_of_nonneg_right ?_ (norm_nonneg _)
  have hW1 : (1 : ℝ) < W := by exact_mod_cast hW
  have hcos1 : Real.cos ((1 - rePosition W (D - (D - 1))) * peakFrequency W) = 1 := by
    rw [show D - (D - 1) = 1 by omega, rePosition_of_lt (by omega), lePosition]
    simp
  have hcosW : ∀ δ : ℕ, W ≤ δ → Real.cos ((1 - rePosition W δ) * peakFrequency W) = -1 :=
    fun δ hδ => by
      rw [rePosition_of_le hδ, peakFrequency, show (1 - (W : ℝ)) * (Real.pi / ((W : ℝ) - 1)) =
        -Real.pi by field_simp; ring, Real.cos_neg, Real.cos_pi]
  simp only [Fin.val_zero, Nat.sub_zero, hcos1, hcosW D hD]
  set N : ℝ := (D : ℝ) + 1 with hN
  set g : ℝ := Real.log N * (q ⬝ᵥ q / d) with hg
  have hN2 : (2 : ℝ) ≤ N := by
    have : (2 : ℝ) ≤ D := by exact_mod_cast hW.trans hD
    linarith
  have hlog : 0 ≤ Real.log N := Real.log_nonneg (by linarith)
  have hg0 : 0 ≤ g := mul_nonneg hlog (div_nonneg (by linarith) hd.le)
  have h2g : Real.log N ≤ 2 * g := by
    rw [hg, show 2 * (Real.log N * (q ⬝ᵥ q / d)) = Real.log N * (2 * (q ⬝ᵥ q) / d) by ring]
    exact le_mul_of_one_le_right hlog ((one_le_div hd).2 hq)
  set A := Real.exp g
  set B := Real.exp (-g)
  have hAB : A * B = 1 := by rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]
  have hA2 : N ≤ A * A := by
    rw [← Real.exp_add, ← Real.exp_log (by linarith : (0 : ℝ) < N)]
    exact Real.exp_le_exp.2 (by linarith)
  have hA : 0 < A := Real.exp_pos g
  have hB : 0 < B := Real.exp_pos (-g)
  set S := ∑ j : Fin (D + 1), Real.exp (g * Real.cos ((1 - rePosition W (D - j)) *
    peakFrequency W))
  have hS0 : 0 < S := Finset.sum_pos (fun j _ => Real.exp_pos _) Finset.univ_nonempty
  have hS : S ≤ N * B + W * A := by
    calc S ≤ ∑ j : Fin (D + 1), (B + if D - (j : ℕ) < W then A else 0) :=
          Finset.sum_le_sum fun j _ => by
            split_ifs with h
            · exact (Real.exp_le_exp.2 (mul_le_of_le_one_right hg0 (Real.cos_le_one _))).trans
                (le_add_of_nonneg_left hB.le)
            · rw [hcosW _ (not_lt.1 h), mul_neg_one, add_zero]
      _ ≤ N * B + W * A := by
          rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
            nsmul_eq_mul, Finset.sum_ite, Finset.sum_const_zero, add_zero, Finset.sum_const,
            nsmul_eq_mul, hN]
          push_cast
          gcongr
          exact_mod_cast card_filter_sub_lt_le D W
  have hNB : N * B ≤ A := by nlinarith
  have hBA : 2 * B ≤ A := by nlinarith
  rw [mul_one, mul_neg_one, abs_of_nonneg (div_nonneg (by linarith) hS0.le)]
  calc 1 / (2 * ((W : ℝ) + 1)) = (A / 2) / (((W : ℝ) + 1) * A) := by field_simp
    _ ≤ (A - B) / S := div_le_div₀ (by linarith) (by linarith) hS0 (by linarith)

/-- The window `2`, the query `5`, `q = 1` of width `2`, so `κ = 1`. -/
example (x : ℕ → ℝ) :=
  le_norm_peakHead_sub (W := 2) (D := 5) le_rfl (by norm_num) (d := 2) two_pos
    (fun _ : Fin 2 × Fin 1 => (1 : ℝ)) (by norm_num [dotProduct]) (fun t : ℝ => t) x

end Transformer.ReRoPEBlock
