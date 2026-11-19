module

public import Transformer.Sandwich.Section6_Sandwich

/-!
# Appendix E: the reference implementation of Sandwich

Chi, Fan, Rudnicky, Ramadge, arXiv:2212.10356v2, Appendix E, "Python Implementation of Sandwich":

    i = np.arange(bar_d // 2)
    pos_embs = np.concatenate([np.sin(positions / base ** (2 * i / bar_d)),
                               np.cos(positions / base ** (2 * i / bar_d))], axis=-1)
    sandwich = np.matmul(pos_embs, pos_embs.T)
    compression_ratio = np.arange(1, heads + 1) * 8 / heads
    multi_head_sandwich = sandwich[None, ...] / compression_ratio[..., None, None]

with `base = 1e4`. The rows of `pos_embs` are Sinusoidal's `sinusoidalAPE` of §5.3, with the
frequencies `apeFreq` for `i = 0, …, d̄/2 − 1`; the compression ratios are §6.1's
(`exponent`). The code differs from Eq. (6) twice:

* It does not subtract `d̄/2`. Adding a constant to every bias does not change the attention
  (`attnWeight_add_const`): Softmax is shift-invariant.
* Its frequencies are `1/10000^{2i/d̄}` for `i = 0, …, d̄/2 − 1`, while §6.1 sums over
  `i = 1, …, d̄/2`: the code has the frequency `1` in place of `1/10000`. Its bias is Eq. (6)'s plus
  `(d̄/2 + cos t − cos(t/10000))/h` at the distance `t` (`codeSandwich_sub`), which is not a
  constant, and it changes the attention of every head on every input (`attnWeight_code_ne`).
-/

@[expose] public section

namespace Transformer.Sandwich

open Finset Matrix Transformer.BAM

variable {d k D : ℕ}

/-- **Appendix E**'s bias at the distance `t` with the compression ratio `h`:
`Σ_{i=0}^{d̄/2−1} cos(t/10000^{2i/d̄}) / h`. -/
noncomputable def codeSandwich (D : ℕ) (h t : ℝ) : ℝ := (∑ i, Real.cos (t * apeFreq D i)) / h

/-- **Appendix E's bias is `p_m^⊤ p_n / h`** for Sinusoidal's `p_m` of Eq. (5), the rows of
`pos_embs`, sines first. -/
theorem codeSandwich_eq (D : ℕ) (h : ℝ) (m n : ℕ) :
    codeSandwich D h ((m : ℝ) - n) = sinusoidalAPE D m ⬝ᵥ sinusoidalAPE D n / h := by
  rw [codeSandwich, ← sinusoid_dot]
  congr 1
  simp only [dotProduct, sinusoidalAPE]
  exact (finSumFinEquiv.symm.sum_comp
    fun s => sinusoid (apeFreq D) m s * sinusoid (apeFreq D) n s).symm

/-- **Appendix E's bias is Eq. (6)'s plus `(d̄/2 + cos t − cos(t/10000))/h`**: the code omits
`−d̄/2`, and its frequencies start at `1` where §6.1's end at `1/10000`. -/
theorem codeSandwich_sub (hD : 1 ≤ D) (h t : ℝ) :
    codeSandwich D h t - sandwich D h t = (D + Real.cos t - Real.cos (t / 10000)) / h := by
  obtain ⟨D, rfl⟩ : ∃ D', D = D' + 1 := ⟨D - 1, by omega⟩
  have hsucc : ∀ i : Fin D, Real.cos (t * apeFreq (D + 1) i.succ) =
      Real.cos (t * freq (D + 1) i.castSucc) := fun i => by
    simp [apeFreq, freq, Fin.val_succ]
  have h0 : apeFreq (D + 1) 0 = 1 := by simp [apeFreq]
  have hlast : freq (D + 1) (Fin.last D) = 1 / 10000 := by
    have hD1 : (D : ℝ) + 1 ≠ 0 := by positivity
    simp [freq, div_self hD1]
  rw [codeSandwich, sandwich, div_sub_div_same, Fin.sum_univ_succ, Fin.sum_univ_castSucc, h0,
    hlast, sum_congr rfl fun i _ => hsucc i, mul_one, mul_one_div]
  push_cast
  ring_nf

/-- The hypothesis of `codeSandwich_sub` holds for `d̄ = 128`. -/
example (h t : ℝ) :
    codeSandwich 64 h t - sandwich 64 h t = ((64 : ℕ) + Real.cos t - Real.cos (t / 10000)) / h :=
  codeSandwich_sub (by norm_num) h t

/-- **Adding a constant to the bias does not change the attention**: Softmax is shift-invariant,
so the `−d̄/2` that Appendix E omits is harmless. -/
theorem attnWeight_add_const (A : Head d k) (b : ℕ → EReal) (c : ℝ) (x : Embs d) (m n : ℕ) :
    attnWeight A (fun t => b t + c) x m n = attnWeight A b x m n := by
  have hs : ∀ i, score A (fun t => b t + c) x m i = score A b x m i + c := fun i => by
    simp only [score, add_assoc]
  simp only [attnWeight, hs, weight_add, weight_coe, ← sum_mul]
  exact mul_div_mul_right _ _ (Real.exp_pos c).ne'

/-- Two keys: the weight of the key `0` for the query `1`, with real biases `r`. -/
theorem attnWeight_one_zero (A : Head d k) (r : ℕ → ℝ) (x : Embs d) :
    attnWeight A (fun t => (r t : EReal)) x 1 0 =
      Real.exp ((A.Wq *ᵥ x 1) ⬝ᵥ (A.Wk *ᵥ x 0) / Real.sqrt k + r 1) /
        (Real.exp ((A.Wq *ᵥ x 1) ⬝ᵥ (A.Wk *ᵥ x 0) / Real.sqrt k + r 1) +
          Real.exp ((A.Wq *ᵥ x 1) ⬝ᵥ (A.Wk *ᵥ x 1) / Real.sqrt k + r 0)) := by
  simp [attnWeight, score, sum_range_succ, ← EReal.coe_add]

/-- **Appendix E's code changes the attention** of every head on every input: the weight of the
key `0` for the query `1` differs from Eq. (6)'s, since `cos 1 ≠ cos(1/10000)`. -/
theorem attnWeight_code_ne (hD : 1 ≤ D) {h : ℝ} (hh : 0 < h) (A : Head d k) (x : Embs d) :
    attnWeight A (fun t => (codeSandwich D h t : EReal)) x 1 0 ≠
      attnWeight A (fun t => (sandwich D h t : EReal)) x 1 0 := by
  rw [attnWeight_one_zero A (fun t => codeSandwich D h t),
    attnWeight_one_zero A (fun t => sandwich D h t)]
  intro heq
  rw [div_eq_div_iff (by positivity) (by positivity)] at heq
  set c₀ := (A.Wq *ᵥ x 1) ⬝ᵥ (A.Wk *ᵥ x 0) / Real.sqrt k
  set c₁ := (A.Wq *ᵥ x 1) ⬝ᵥ (A.Wk *ᵥ x 1) / Real.sqrt k
  have hexp : Real.exp (c₀ + codeSandwich D h (1 : ℕ)) * Real.exp (c₁ + sandwich D h (0 : ℕ)) =
      Real.exp (c₀ + sandwich D h (1 : ℕ)) * Real.exp (c₁ + codeSandwich D h (0 : ℕ)) := by
    linear_combination heq
  rw [← Real.exp_add, ← Real.exp_add] at hexp
  have h1 := codeSandwich_sub hD h 1
  have h0 := codeSandwich_sub hD h 0
  have hcos : Real.cos 1 < Real.cos (1 / 10000) :=
    Real.cos_lt_cos_of_nonneg_of_le_pi (by norm_num) (by linarith [Real.two_le_pi])
      (by norm_num)
  have := Real.exp_injective hexp
  push_cast at this
  rw [Real.cos_zero, zero_div, Real.cos_zero] at h0
  have key : (D + Real.cos 1 - Real.cos (1 / 10000)) / h = (D + 1 - 1) / h := by linarith
  rw [div_left_inj₀ hh.ne'] at key
  linarith

/-- The hypotheses of `attnWeight_code_ne` hold for `d̄ = 128` and `h = 8`. -/
example (A : Head d k) (x : Embs d) :
    attnWeight A (fun t => (codeSandwich 64 8 t : EReal)) x 1 0 ≠
      attnWeight A (fun t => (sandwich 64 8 t : EReal)) x 1 0 :=
  attnWeight_code_ne (by norm_num) (by norm_num) A x

end Transformer.Sandwich
