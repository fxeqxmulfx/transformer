module

public import Mathlib.Analysis.Complex.ExponentialBounds
public import Mathlib.Analysis.SpecialFunctions.Pow.Asymptotics
public import Transformer.KERPLE.Section4_Attention

/-!
# Appendix A.5: the attenuation of a bias

Chi, Fan, Ramadge, Rudnicky, arXiv:2205.09921v2, Appendix A.5: the effective length is the
distance at which `k̃` falls below `−2`, "because `k̃` is a bias and is followed by the Softmax
normalization", with the footnote "Since Softmax is an exponentiated function, a −2 bias in the
Softmax's argument roughly gives an attenuation of `exp(−2) ≈ 0.135`".

* A bias `b` in one argument of the Softmax multiplies that weight's ratio to every other weight by
  `exp(b)` (`softmax_div_softmax`), and `0.135 < exp(−2) < 0.136` (`exp_neg_two_mem_Ioo`).
* Eq. 1 divides the bias by `√d` before the Softmax: for two keys with equal content scores
  `q_mᵀk_n = q_mᵀk_{n'}`, `a_{m,n} / a_{m,n'} = exp((k̃(m, n) − k̃(m, n'))/√d)`
  (`attention_div_attention`). §5.1 sets `d = 64` in Eq. 1 ("Hidden size 64 means that `d = 64` in
  Eq. (1)"), and there biases differing by `−2` give the ratio `exp(−1/4) > 3/4`, not `exp(−2)`
  (`attention_div_attention_of_sub_two`): under Eq. 1 as printed, A.5's effective lengths mark the
  bias `−1/4` of the Softmax's argument, and the attenuation `exp(−2)` needs `k̃` to fall by
  `2√d = 16`.
* A.5's reading holds when the bias is added after the scaling, in the Softmax of
  `q_mᵀk_i/√d + k̃(m, i)` (`softmax_scaled_add_div_softmax`).
* Characteristics Learned by Kernels: "logarithm (`−a log(1 + b|m − n|)`) decays much slower than
  power (`−a|m − n|^p`) does": `a log(1 + b x) = o(x^p)` for `b, p > 0`
  (`isLittleO_mul_log_one_add_rpow`), so from some distance on the logarithmic variant's bias
  attenuates less than every power variant's (`eventually_powerKernel_lt_logKernel`).
-/

@[expose] public section

namespace Transformer.KERPLE

variable {ι : Type*} [Fintype ι]

/-- **A bias `b` in the Softmax's argument attenuates by `exp(b)`** (A.5, footnote): the ratio of
two Softmax weights is `exp(z_n − z_{n'})`. -/
theorem softmax_div_softmax (z : ι → ℝ) (n n' : ι) :
    softmax z n / softmax z n' = Real.exp (z n - z n') := by
  have hS : (∑ i, Real.exp (z i)) ≠ 0 :=
    (Finset.sum_pos (fun i _ => Real.exp_pos (z i)) ⟨n, Finset.mem_univ n⟩).ne'
  rw [softmax, softmax, div_div_div_cancel_right₀ hS, Real.exp_sub]

/-- **The footnote's value**: `exp(−2) ≈ 0.135`, with `0.135 < exp(−2) < 0.136`. -/
theorem exp_neg_two_mem_Ioo : Real.exp (-2) ∈ Set.Ioo 0.135 0.136 := by
  have he1 := Real.exp_one_gt_d9
  have he2 := Real.exp_one_lt_d9
  have ht0 := Real.exp_pos (-1)
  have h1 : Real.exp (-1) * Real.exp 1 = 1 := by rw [← Real.exp_add]; norm_num
  have hlo : 0.3678 < Real.exp (-1) := by nlinarith [mul_pos ht0 (sub_pos.mpr he2)]
  have hhi : Real.exp (-1) < 0.3679 := by nlinarith [mul_pos ht0 (sub_pos.mpr he1)]
  have h2 : Real.exp (-2) = Real.exp (-1) ^ 2 := by rw [sq, ← Real.exp_add]; norm_num
  rw [h2]
  constructor <;> nlinarith

/-- `√64 = 8`, the scale of Eq. 1 at `d = 64`. -/
theorem sqrt_sixtyfour : Real.sqrt ((64 : ℕ) : ℝ) = 8 := by
  rw [show ((64 : ℕ) : ℝ) = 8 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]

/-- **The attenuation due to the bias in Eq. 1**: for two keys with equal content scores,
`a_{m,n} / a_{m,n'} = exp((k̃(m, n) − k̃(m, n'))/√d)`. -/
theorem attention_div_attention {L d : ℕ} (q k : Fin L → EuclideanSpace ℝ (Fin d))
    (kt : ℝ → ℝ → ℝ) (m n n' : Fin L) (h : inner ℝ (q m) (k n) = inner ℝ (q m) (k n')) :
    attention q k kt m n / attention q k kt m n' =
      Real.exp ((kt (m : ℕ) (n : ℕ) - kt (m : ℕ) (n' : ℕ)) / Real.sqrt d) := by
  rw [attention, attention, softmax_div_softmax]
  simp only [h]
  congr 1
  ring

/-- The hypothesis of `attention_div_attention` holds for equal queries and keys; at `d = 64` the
attenuation `exp(−2)` needs biases differing by `−2√d = −16`. -/
example (kt : ℝ → ℝ → ℝ) (m n n' : Fin 3) (v : EuclideanSpace ℝ (Fin 64))
    (hkt : kt (m : ℕ) (n : ℕ) = kt (m : ℕ) (n' : ℕ) - 16) :
    attention (fun _ => v) (fun _ => v) kt m n / attention (fun _ => v) (fun _ => v) kt m n' =
      Real.exp (-2) := by
  rw [attention_div_attention _ _ kt m n n' rfl, hkt, sqrt_sixtyfour]
  congr 1
  ring

/-- **A.5's attenuation fails under Eq. 1 at `d = 64`** (§5.1): for two keys with equal content
scores whose biases differ by `−2`, the ratio of their weights is `exp(−1/4) > 3/4`, not the
attenuation `exp(−2) ≈ 0.135` that A.5's footnote gives "a −2 bias in the Softmax's argument". -/
theorem attention_div_attention_of_sub_two {L : ℕ} (q k : Fin L → EuclideanSpace ℝ (Fin 64))
    (kt : ℝ → ℝ → ℝ) (m n n' : Fin L) (h : inner ℝ (q m) (k n) = inner ℝ (q m) (k n'))
    (hkt : kt (m : ℕ) (n : ℕ) = kt (m : ℕ) (n' : ℕ) - 2) :
    attention q k kt m n / attention q k kt m n' = Real.exp (-1 / 4) ∧
      3 / 4 < Real.exp (-1 / 4) := by
  refine ⟨?_, ?_⟩
  · rw [attention_div_attention q k kt m n n' h, hkt, sqrt_sixtyfour]
    congr 1
    ring
  · have := Real.add_one_lt_exp (by norm_num : (-1 / 4 : ℝ) ≠ 0)
    linarith

/-- The hypotheses of `attention_div_attention_of_sub_two` hold for equal queries and keys and
ALiBi's kernel `−|m − n|` at the distances `2` and `0`. -/
example (v : EuclideanSpace ℝ (Fin 64)) :
    attention (fun _ : Fin 3 => v) (fun _ => v) (powerKernel 1 1) 0 2 /
        attention (fun _ : Fin 3 => v) (fun _ => v) (powerKernel 1 1) 0 0 = Real.exp (-1 / 4) ∧
      3 / 4 < Real.exp (-1 / 4) :=
  attention_div_attention_of_sub_two _ _ _ 0 2 0 rfl (by norm_num [powerKernel])

/-- **A.5's reading, with the bias added after the scaling**: in the Softmax of
`q_mᵀk_i/√d + k̃(m, i)`, two keys with equal content scores have the weight ratio
`exp(k̃(m, n) − k̃(m, n'))`, and biases differing by `−2` attenuate by `exp(−2)`. -/
theorem softmax_scaled_add_div_softmax {L d : ℕ} (q k : Fin L → EuclideanSpace ℝ (Fin d))
    (kt : ℝ → ℝ → ℝ) (m n n' : Fin L) (h : inner ℝ (q m) (k n) = inner ℝ (q m) (k n')) :
    softmax (fun i : Fin L => inner ℝ (q m) (k i) / Real.sqrt d + kt (m : ℕ) (i : ℕ)) n /
        softmax (fun i : Fin L => inner ℝ (q m) (k i) / Real.sqrt d + kt (m : ℕ) (i : ℕ)) n' =
      Real.exp (kt (m : ℕ) (n : ℕ) - kt (m : ℕ) (n' : ℕ)) := by
  rw [softmax_div_softmax]
  simp only [h]
  congr 1
  ring

/-- The hypothesis of `softmax_scaled_add_div_softmax` holds for equal queries and keys. -/
example (kt : ℝ → ℝ → ℝ) (m n n' : Fin 3) (v : EuclideanSpace ℝ (Fin 64)) :
    softmax (fun i : Fin 3 => inner ℝ v v / Real.sqrt ((64 : ℕ) : ℝ) + kt (m : ℕ) (i : ℕ)) n /
        softmax (fun i : Fin 3 => inner ℝ v v / Real.sqrt ((64 : ℕ) : ℝ) + kt (m : ℕ) (i : ℕ)) n' =
      Real.exp (kt (m : ℕ) (n : ℕ) - kt (m : ℕ) (n' : ℕ)) :=
  softmax_scaled_add_div_softmax (fun _ => v) (fun _ => v) kt m n n' rfl

open Filter Asymptotics

/-- **"Logarithm decays much slower than power"** (A.5, Characteristics Learned by Kernels): for
`b, p > 0`, `a log(1 + b x) = o(x^p)`. -/
theorem isLittleO_mul_log_one_add_rpow (a : ℝ) {b p : ℝ} (hb : 0 < b) (hp : 0 < p) :
    (fun x : ℝ => a * Real.log (1 + b * x)) =o[atTop] fun x => x ^ p := by
  have hg : Tendsto (fun x : ℝ => 1 + b * x) atTop atTop :=
    tendsto_atTop_add_const_left _ 1 (tendsto_id.const_mul_atTop hb)
  have h1 : (fun x : ℝ => Real.log (1 + b * x)) =o[atTop] fun x => (1 + b * x) ^ p :=
    (isLittleO_log_rpow_atTop hp).comp_tendsto hg
  have h2 : (fun x : ℝ => (1 + b * x) ^ p) =O[atTop] fun x => x ^ p := by
    refine IsBigO.of_bound ((1 + b) ^ p) ?_
    filter_upwards [eventually_ge_atTop 1] with x hx
    have hx0 : (0 : ℝ) ≤ x := by linarith
    rw [Real.norm_of_nonneg (by positivity), Real.norm_of_nonneg (by positivity),
      ← Real.mul_rpow (by positivity) hx0]
    exact Real.rpow_le_rpow (by positivity) (by nlinarith) hp.le
  exact (h1.trans_isBigO h2).const_mul_left a

/-- The hypotheses of `isLittleO_mul_log_one_add_rpow` hold for the square root. -/
example : (fun x : ℝ => 1 * Real.log (1 + 1 * x)) =o[atTop] fun x => x ^ (1 / 2 : ℝ) :=
  isLittleO_mul_log_one_add_rpow 1 one_pos (by norm_num)

/-- **From some distance on, the logarithmic variant lies above every power variant** (A.5): for
`r₂ > 0` and `s₁, s₂ > 0`, eventually `−s₁ d^{s₂} < −r₁ log(1 + r₂ d)`. -/
theorem eventually_powerKernel_lt_logKernel (r₁ : ℝ) {r₂ s₁ s₂ : ℝ} (hr₂ : 0 < r₂) (hs₁ : 0 < s₁)
    (hs₂ : 0 < s₂) : ∀ᶠ d : ℕ in atTop, powerKernel s₁ s₂ 0 d < logKernel r₁ r₂ 0 d := by
  filter_upwards [tendsto_natCast_atTop_atTop.eventually
    ((isLittleO_mul_log_one_add_rpow r₁ hr₂ hs₂).bound (half_pos hs₁)),
    eventually_ge_atTop 1] with d hd hd1
  have hpos : 0 < (d : ℝ) ^ s₂ := Real.rpow_pos_of_pos (Nat.cast_pos.mpr (by omega)) _
  rw [Real.norm_of_nonneg hpos.le] at hd
  have habs := Real.le_norm_self (r₁ * Real.log (1 + r₂ * d))
  simp only [powerKernel, logKernel, zero_sub, abs_neg, Nat.abs_cast]
  nlinarith [mul_pos hs₁ hpos]

/-- The hypotheses of `eventually_powerKernel_lt_logKernel` hold for ALiBi's `−|m − n|` against
`−log(1 + |m − n|)`. -/
example : ∀ᶠ d : ℕ in atTop, powerKernel 1 1 0 d < logKernel 1 1 0 d :=
  eventually_powerKernel_lt_logKernel 1 one_pos one_pos one_pos

end Transformer.KERPLE
