import Transformer.AMSGrad.Section4_Counter
import Transformer.AMSGrad.Section4_Lemmas

/-
# AMSGrad — the run of the counterexample to Corollary 4.5

§4 of arXiv:1904.03590v4, Corollary 4.5: the run of `signSetup`.  With
`β_{1,t} = 0` and `g_t = s_t`, the step is `x_{t+1} = Π(x_t - η_t s_t)` with
`η_t = α_t/√v̂_t ≥ 1/√t`, so inside a block `[2^k, 2^{k+1})` the iterate
reaches the endpoint `-s_t` after at most `2√(2^{k+1})` steps and stays there.

Source: arXiv:1904.03590v4, §4, Corollary 4.5.
-/

open Finset

namespace Transformer
namespace AMSGrad

/-- `s_t² = 1`.  arXiv:1904.03590v4, §4, Corollary 4.5. -/
theorem blockSign_sq (t : ℕ) : blockSign t ^ 2 = 1 := by
  unfold blockSign; split_ifs <;> norm_num

/-- `0 ≤ v_t ≤ 1` and `0 ≤ v̂_t ≤ 1`, and `1/2 ≤ v̂_t` once `t ≥ 1`.
arXiv:1904.03590v4, §4, Corollary 4.5. -/
theorem sign_vhat (n : ℕ) :
    (0 ≤ (signSetup.state amsgradRule n).v 0 ∧ (signSetup.state amsgradRule n).v 0 ≤ 1) ∧
      (0 ≤ (signSetup.state amsgradRule n).vhat 0 ∧
        (signSetup.state amsgradRule n).vhat 0 ≤ 1) ∧
      (1 ≤ n → 1 / 2 ≤ (signSetup.state amsgradRule n).vhat 0) := by
  induction n with
  | zero => simp [Setup.state]
  | succ n ih =>
    obtain ⟨⟨h1, h2⟩, ⟨h3, h4⟩, -⟩ := ih
    have hv : (signSetup.state amsgradRule (n + 1)).v 0 =
        (signSetup.state amsgradRule n).v 0 / 2 + 1 / 2 := by
      simp only [Setup.state, Setup.step, Pi.add_apply, Pi.smul_apply, Pi.pow_apply,
        smul_eq_mul]
      rw [show signSetup.β₂ = 1 / 2 from rfl, show signSetup.f (n + 1) =
        fun x => blockSign (n + 1) * x 0 from rfl, grad_linear, blockSign_sq]
      ring
    have hvh : (signSetup.state amsgradRule (n + 1)).vhat 0 =
        (signSetup.state amsgradRule n).vhat 0 ⊔ (signSetup.state amsgradRule (n + 1)).v 0 :=
      rfl
    rw [hvh, hv]
    refine ⟨⟨by linarith, by linarith⟩, ⟨le_sup_of_le_left h3, sup_le h4 (by linarith)⟩,
      fun _ => le_sup_of_le_right (by linarith)⟩

/-- The step: `x_{t+1} = Π(x_t - η_t s_t)` with `η_t ≥ 1/√t`.
arXiv:1904.03590v4, §4, Corollary 4.5. -/
theorem sign_step (n : ℕ) : ∃ η : ℝ, 1 / Real.sqrt ((n + 1 : ℕ) : ℝ) ≤ η ∧
    signSetup.x amsgradRule (n + 2) 0 =
      max (-1) (min 1 (signSetup.x amsgradRule (n + 1) 0 - η * blockSign (n + 1))) := by
  obtain ⟨-, ⟨-, h4⟩, h5⟩ := sign_vhat (n + 1)
  have h5 := h5 (by omega)
  have hs : 0 < Real.sqrt ((signSetup.state amsgradRule (n + 1)).vhat 0) :=
    Real.sqrt_pos.2 (by linarith)
  have hs1 : Real.sqrt ((signSetup.state amsgradRule (n + 1)).vhat 0) ≤ 1 :=
    Real.sqrt_le_one.2 h4
  refine ⟨1 / Real.sqrt ((n + 1 : ℕ) : ℝ) /
    Real.sqrt ((signSetup.state amsgradRule (n + 1)).vhat 0), le_div_self (by positivity) hs hs1,
    ?_⟩
  show (signSetup.step amsgradRule (n + 1) (signSetup.state amsgradRule n)).x 0 = _
  simp only [Setup.step]
  rw [show signSetup.proj = fun _ => boxProj (-1) 1 from rfl,
    show signSetup.f (n + 1) = fun x => blockSign (n + 1) * x 0 from rfl, grad_linear,
    show signSetup.α (n + 1) = 1 / Real.sqrt ((n + 1 : ℕ) : ℝ) from rfl,
    show signSetup.β₁ (n + 1) = 0 from rfl]
  simp only [boxProj]
  congr 3
  change _ = _ - _ / Real.sqrt ((signSetup.step amsgradRule (n + 1)
    (signSetup.state amsgradRule n)).vhat 0) * _
  simp only [Setup.step, Pi.sub_apply, Pi.smul_apply, Pi.div_apply, Pi.add_apply, smul_eq_mul,
    vsqrt]
  rw [show signSetup.x amsgradRule (n + 1) = (signSetup.state amsgradRule n).x from rfl,
    show signSetup.f (n + 1) = fun x => blockSign (n + 1) * x 0 from rfl, grad_linear]
  ring

/-- Clamping commutes with the sign: `s Π(x - ηs) ≤ max(-1, sx - η)` for `s = ±1`.
arXiv:1904.03590v4, §4, Corollary 4.5. -/
theorem clamp_sign {s x η : ℝ} (hs : s = 1 ∨ s = -1) :
    s * max (-1) (min 1 (x - η * s)) ≤ max (-1) (s * x - η) := by
  rcases hs with rfl | rfl
  · simp only [one_mul, mul_one]; exact max_le_max le_rfl (min_le_right _ _)
  · simp only [max_def, min_def]; split_ifs <;> linarith

/-- On the block `[2^k, 2^{k+1})`, `s_t = (-1)^k`.
arXiv:1904.03590v4, §4, Corollary 4.5. -/
theorem blockSign_block {k t : ℕ} (h1 : 2 ^ k ≤ t) (h2 : t < 2 ^ (k + 1)) :
    blockSign t = (-1) ^ k := by
  unfold blockSign
  rw [Nat.log_eq_of_pow_le_of_lt_pow h1 h2]
  split_ifs with h
  · exact (Even.neg_one_pow (Nat.even_iff.2 h)).symm
  · exact (Odd.neg_one_pow (Nat.odd_iff.2 (by omega))).symm

/-- `x_t ∈ [-1, 1]`.  arXiv:1904.03590v4, §4, Corollary 4.5. -/
theorem sign_x_mem (t : ℕ) :
    -1 ≤ signSetup.x amsgradRule t 0 ∧ signSetup.x amsgradRule t 0 ≤ 1 :=
  ⟨(x_mem isOnlineConvex_sign amsgradRule t).1 0, (x_mem isOnlineConvex_sign amsgradRule t).2 0⟩

/-- Inside the block `[2^k, 2^{k+1})`, `n` steps after its start,
`(-1)^k x ≤ max(-1, 1 - n/√(2^{k+1}))`.
arXiv:1904.03590v4, §4, Corollary 4.5. -/
theorem sign_block (k n : ℕ) (hn : 2 ^ k + n ≤ 2 ^ (k + 1)) :
    (-1 : ℝ) ^ k * signSetup.x amsgradRule (2 ^ k + n) 0 ≤
      max (-1) (1 - n / Real.sqrt ((2 : ℝ) ^ (k + 1))) := by
  have hc : 0 < Real.sqrt ((2 : ℝ) ^ (k + 1)) := Real.sqrt_pos.2 (by positivity)
  induction n with
  | zero =>
    obtain ⟨h1, h2⟩ := sign_x_mem (2 ^ k)
    refine le_max_of_le_right ?_
    rcases neg_one_pow_eq_or ℝ k with h | h <;> rw [h] <;> simp <;> linarith
  | succ n ih =>
    have ih := ih (by omega)
    obtain ⟨m, hm⟩ : ∃ m, 2 ^ k + n = m + 1 := ⟨2 ^ k + n - 1, by
      have := Nat.one_le_two_pow (n := k); omega⟩
    obtain ⟨η, hη, hx⟩ := sign_step m
    have hb : blockSign (m + 1) = (-1) ^ k := blockSign_block (by omega) (by omega)
    rw [show 2 ^ k + (n + 1) = m + 2 by omega, hx, hb]
    rw [hm] at ih
    refine (clamp_sign (neg_one_pow_eq_or ℝ k)).trans (max_le (le_max_left _ _) ?_)
    have hη' : 1 / Real.sqrt ((2 : ℝ) ^ (k + 1)) ≤ η := by
      refine le_trans ?_ hη
      refine one_div_le_one_div_of_le (Real.sqrt_pos.2 (by positivity)) (Real.sqrt_le_sqrt ?_)
      exact_mod_cast (show m + 1 ≤ 2 ^ (k + 1) by omega)
    push_cast
    rcases le_max_iff.1 ih with h | h
    · refine le_max_of_le_left ?_
      have : 0 < η := lt_of_lt_of_le (by positivity) hη'
      linarith
    · refine le_max_of_le_right ?_
      rw [add_div, one_div] at *
      linarith

/-- `s_t x_t ≤ -1` on the block `[2^k, 2^{k+1})` from `2^k + 2^{⌊k/2⌋+2}` on,
and `≤ 1` before.
arXiv:1904.03590v4, §4, Corollary 4.5. -/
theorem sign_term {k t : ℕ} (h1 : 2 ^ k ≤ t) (h2 : t < 2 ^ (k + 1)) :
    blockSign t * signSetup.x amsgradRule t 0 ≤
      -1 + 2 * (if t < 2 ^ k + 2 ^ (k / 2 + 2) then 1 else 0) := by
  rw [blockSign_block h1 h2]
  split_ifs with h
  · obtain ⟨h1, h2⟩ := sign_x_mem t
    rcases neg_one_pow_eq_or ℝ k with h | h <;> rw [h] <;> linarith
  · have hb := sign_block k (t - 2 ^ k) (by omega)
    rw [show 2 ^ k + (t - 2 ^ k) = t by omega] at hb
    have hN : 4 * 2 ^ (k + 1) ≤ (t - 2 ^ k) ^ 2 :=
      calc 4 * 2 ^ (k + 1) = 2 ^ (k + 3) := by ring
        _ ≤ 2 ^ (2 * (k / 2 + 2)) := Nat.pow_le_pow_right two_pos (by omega)
        _ = (2 ^ (k / 2 + 2)) ^ 2 := by rw [pow_mul']
        _ ≤ (t - 2 ^ k) ^ 2 := Nat.pow_le_pow_left (by omega) 2
    have hc : Real.sqrt ((2 : ℝ) ^ (k + 1)) ≤ ((t - 2 ^ k : ℕ) : ℝ) / 2 := by
      rw [Real.sqrt_le_iff]
      refine ⟨by positivity, ?_⟩
      have : ((4 * 2 ^ (k + 1) : ℕ) : ℝ) ≤ (((t - 2 ^ k) ^ 2 : ℕ) : ℝ) := by exact_mod_cast hN
      push_cast at this
      nlinarith
    have hc0 : 0 < Real.sqrt ((2 : ℝ) ^ (k + 1)) := Real.sqrt_pos.2 (by positivity)
    have : 2 ≤ ((t - 2 ^ k : ℕ) : ℝ) / Real.sqrt ((2 : ℝ) ^ (k + 1)) := by
      rw [le_div_iff₀ hc0]; linarith
    refine hb.trans (max_le (by norm_num) (by linarith))

/-- The hypotheses of `clamp_sign`, `blockSign_block`, `sign_block` and `sign_term` are
satisfiable: `s = 1`, `k = 0`, `t = 1`, `n = 0`. -/
example : ((1 : ℝ) = 1 ∨ (1 : ℝ) = -1) ∧ 2 ^ 0 ≤ 1 ∧ 1 < 2 ^ (0 + 1) ∧ 2 ^ 0 + 0 ≤ 2 ^ (0 + 1) := by
  norm_num

end AMSGrad
end Transformer
