module

public import Transformer.BAM.Section2_LongContext

/-!
# §2.5 The shapes of BAM's priors, attention fading and SSMax

Bianchessi, Aguirre, Barros, Kupssinskü, arXiv:2505.22842v4, §2.5.

* "The effect of the location parameter `μ` over the GGD is to move the peak": for `β > 0` the
  weight `e^{−|(x − μ)/α|^β}` is largest at `x = μ` (`ggdWeight_lt_self`) and falls with the
  distance from `μ` (`ggdWeight_anti`). "If `μ > 0`, it moves the peak beyond the causal mask":
  the visible positions `x ≤ 0` peak at the query, below the peak at `μ` (`ggdWeight_causal`).
  ALiBi is the case `μ = 0`, `β = 1` (`ggdBias_alibi`): its peak is at the query.
* "`β < 0` is not a valid parametrization of the GGD in a strict probability sense": for `β ≤ 0`
  the weight is not integrable (`not_integrable_ggdWeight`), and the GGD formula integrates to
  `0` (`integral_ggd_of_nonpos`). For `β < 0` the prior is `0` at `μ` and grows with the distance
  from it (`weight_ggdBias_mono`), "concentrated at the far end of the context window".
* `β ∈ (0, 1)` has a heavier tail than Laplace: `tendsto_ggdWeight_div` (Appendix A).
* Attention fading: without a positional bias, content scores bounded by `C` give every key at
  most `e^{2C}/(i + 1)` (`relAttention_noPE_le`), "the denominator in the Softmax computation
  increases with context size `n`, while the numerator for each token remains constant".
* Scalable softmax: `n^{s z_i} / Σ_j n^{s z_j} = e^{(s log n) z_i} / Σ_j e^{(s log n) z_j}`
  (`ssmax_eq`).
-/

@[expose] public section

namespace Transformer.BAM

open Finset MeasureTheory

variable {L : ℕ}

/-- **`μ` is the peak of the GGD prior** for `β > 0` (§2.5): away from `μ` its weight is below
its value `1` at `μ`. -/
theorem ggdWeight_lt_self {μ α β x : ℝ} (hα : α ≠ 0) (hβ : 0 < β) (hx : x ≠ μ) :
    Real.exp (-|(x - μ) / α| ^ β) < Real.exp (-|(μ - μ) / α| ^ β) := by
  rw [sub_self, zero_div, abs_zero, Real.zero_rpow hβ.ne', neg_zero, Real.exp_lt_exp, neg_lt_zero]
  exact Real.rpow_pos_of_pos (abs_pos.2 (div_ne_zero (sub_ne_zero.2 hx) hα)) β

/-- The hypotheses of `ggdWeight_lt_self` hold for `α = 1`, `β = 2`, `x = μ + 1`. -/
example (μ : ℝ) : Real.exp (-|(μ + 1 - μ) / 1| ^ (2 : ℝ)) < Real.exp (-|(μ - μ) / 1| ^ (2 : ℝ)) :=
  ggdWeight_lt_self one_ne_zero two_pos (by linarith)

/-- For `β > 0` the GGD weight falls with the distance from `μ` (§2.5). -/
theorem ggdWeight_anti {μ α β x y : ℝ} (hβ : 0 < β) (h : |x - μ| ≤ |y - μ|) :
    Real.exp (-|(y - μ) / α| ^ β) ≤ Real.exp (-|(x - μ) / α| ^ β) := by
  rw [Real.exp_le_exp, neg_le_neg_iff]
  refine Real.rpow_le_rpow (abs_nonneg _) ?_ hβ.le
  rw [abs_div, abs_div]
  exact div_le_div_of_nonneg_right h (abs_nonneg α)

/-- The hypotheses of `ggdWeight_anti` hold for `β = 1/2` and the positions `−1`, `−3`. -/
example (α : ℝ) : Real.exp (-|(-3 - 0) / α| ^ (1 / 2 : ℝ)) ≤
    Real.exp (-|(-1 - 0) / α| ^ (1 / 2 : ℝ)) :=
  ggdWeight_anti (by norm_num) (by norm_num)

/-- **If `μ > 0`, the GGD's peak is beyond the causal mask** (§2.5): on the visible relative
positions `x ≤ 0` the weight (`β > 0`) is largest at the query, `x = 0`, where it is below its
peak at `x = μ`. -/
theorem ggdWeight_causal {μ α β x : ℝ} (hα : α ≠ 0) (hβ : 0 < β) (hμ : 0 < μ) (hx : x ≤ 0) :
    Real.exp (-|(x - μ) / α| ^ β) ≤ Real.exp (-|(0 - μ) / α| ^ β) ∧
      Real.exp (-|(0 - μ) / α| ^ β) < Real.exp (-|(μ - μ) / α| ^ β) :=
  ⟨ggdWeight_anti hβ (by rw [abs_of_neg (by linarith), abs_of_neg (by linarith)]; linarith),
    ggdWeight_lt_self hα hβ (by linarith)⟩

/-- The hypotheses of `ggdWeight_causal` hold for `α = 1`, `β = 1`, `μ = 2`, `x = −1`. -/
example : Real.exp (-|(-1 - 2) / 1| ^ (1 : ℝ)) ≤ Real.exp (-|(0 - 2) / 1| ^ (1 : ℝ)) ∧
    Real.exp (-|(0 - 2) / 1| ^ (1 : ℝ)) < Real.exp (-|(2 - 2) / 1| ^ (1 : ℝ)) :=
  ggdWeight_causal one_ne_zero one_pos two_pos (by norm_num)

/-- **For `β < 0` the prior grows with the distance from `μ`** (§2.5): it is `0` at `μ`,
"effectively zero in the vicinity of the query position and sharply concentrated at the far
end". -/
theorem weight_ggdBias_mono {μ α β x y : ℝ} (hα : α ≠ 0) (hβ : β < 0) (h : |x - μ| ≤ |y - μ|) :
    weight (ggdBias μ α β x) ≤ weight (ggdBias μ α β y) := by
  rcases eq_or_ne x μ with rfl | hx
  · rw [ggdBias_self_of_neg hβ, weight_bot]
    exact weight_nonneg _
  have hx' : 0 < |(x - μ) / α| := abs_pos.2 (div_ne_zero (sub_ne_zero.2 hx) hα)
  have hxy : |(x - μ) / α| ≤ |(y - μ) / α| := by
    rw [abs_div, abs_div]
    exact div_le_div_of_nonneg_right h (abs_nonneg α)
  rw [weight_ggdBias (Or.inl (abs_pos.1 hx')),
    weight_ggdBias (Or.inl (abs_pos.1 (hx'.trans_le hxy))), Real.exp_le_exp, neg_le_neg_iff]
  exact Real.rpow_le_rpow_of_nonpos hx' hxy hβ.le

/-- The hypotheses of `weight_ggdBias_mono` hold for `α = 1`, `β = −1`, `x = 1`, `y = 5`. -/
example : weight (ggdBias 0 1 (-1) 1) ≤ weight (ggdBias 0 1 (-1) 5) :=
  weight_ggdBias_mono one_ne_zero (by norm_num) (by norm_num)

/-- For `β ≤ 0` the GGD weight is not integrable: it is at least `e^{−1}` at every
`x ≥ μ + |α|`. -/
theorem not_integrable_ggdWeight {μ α β : ℝ} (hβ : β ≤ 0) :
    ¬ Integrable (fun x => Real.exp (-|(x - μ) / α| ^ β)) := by
  intro h
  have hsub : Set.Ici (μ + |α|) ⊆ {x | Real.exp (-1) ≤ ‖Real.exp (-|(x - μ) / α| ^ β)‖} := by
    intro x hx
    have hx : μ + |α| ≤ x := hx
    have hpow : |(x - μ) / α| ^ β ≤ 1 := by
      rcases eq_or_ne α 0 with rfl | hα
      · simpa using Real.zero_rpow_le_one β
      refine Real.rpow_le_one_of_one_le_of_nonpos ?_ hβ
      rw [abs_div, le_div_iff₀ (abs_pos.2 hα), one_mul,
        abs_of_nonneg (by linarith [abs_nonneg α] : (0 : ℝ) ≤ x - μ)]
      linarith
    change Real.exp (-1) ≤ ‖Real.exp (-|(x - μ) / α| ^ β)‖
    rw [Real.norm_eq_abs, Real.abs_exp]
    exact Real.exp_le_exp.2 (neg_le_neg hpow)
  exact (h.measure_norm_ge_lt_top (Real.exp_pos (-1))).ne (measure_mono_top hsub Real.volume_Ici)

/-- The hypothesis of `not_integrable_ggdWeight` holds for `β = −1`. -/
example (μ α : ℝ) : ¬ Integrable (fun x => Real.exp (-|(x - μ) / α| ^ (-1 : ℝ))) :=
  not_integrable_ggdWeight (by norm_num)

/-- **`β ≤ 0` is not a valid parametrization of the GGD** (§2.5): the GGD formula of Appendix A
integrates to `0`, not `1`. -/
theorem integral_ggd_of_nonpos {μ α β : ℝ} (hβ : β ≤ 0) : ∫ x, ggd μ α β x = 0 := by
  simp only [ggd]
  rw [integral_const_mul, integral_undef (not_integrable_ggdWeight hβ), mul_zero]

/-- The hypothesis of `integral_ggd_of_nonpos` holds for `β = −1`. -/
example (μ α : ℝ) : ∫ x, ggd μ α (-1) x = 0 := integral_ggd_of_nonpos (by norm_num)

/-- **Attention fading** (§2.5): without a positional bias (NoPE) and with content scores bounded
by `C`, query `i` gives every key at most `e^{2C}/(i + 1)`, which vanishes as the context grows. -/
theorem relAttention_noPE_le {C : ℝ} {f : Fin L → ℝ} (hf : ∀ z, |f z| ≤ C) (i j : Fin L) :
    relAttention (fun _ => 0) f i j ≤ Real.exp (2 * C) / ((i : ℕ) + 1) := by
  simpa using relAttention_le_fade (K := 0) hf (fun _ => le_rfl) one_pos (fun _ _ => by simp)
    (Nat.zero_le _) j

/-- The hypothesis of `relAttention_noPE_le` holds for zero content, `C = 0`. -/
example (i j : Fin 4) :
    relAttention (fun _ => 0) (fun _ => 0) i j ≤ Real.exp (2 * 0) / ((i : ℕ) + 1) :=
  relAttention_noPE_le (fun _ => by simp) i j

/-- **Scalable softmax** (§2.5, after Nakanishi): over `n > 0` tokens,
`n^{s z_i} / Σ_j n^{s z_j} = e^{(s log n) z_i} / Σ_j e^{(s log n) z_j}`. -/
theorem ssmax_eq {ι : Type*} [Fintype ι] {n : ℕ} (hn : 0 < n) (s : ℝ) (z : ι → ℝ) (i : ι) :
    (n : ℝ) ^ (s * z i) / ∑ j, (n : ℝ) ^ (s * z j) =
      Real.exp (s * Real.log n * z i) / ∑ j, Real.exp (s * Real.log n * z j) := by
  have h : ∀ x, (n : ℝ) ^ (s * x) = Real.exp (s * Real.log n * x) := fun x => by
    rw [Real.rpow_def_of_pos (by exact_mod_cast hn)]
    ring_nf
  simp only [h]

/-- The hypothesis of `ssmax_eq` holds for `n = 8` tokens. -/
example (s : ℝ) (z : Fin 8 → ℝ) (i : Fin 8) :
    ((8 : ℕ) : ℝ) ^ (s * z i) / ∑ j, ((8 : ℕ) : ℝ) ^ (s * z j) =
      Real.exp (s * Real.log (8 : ℕ) * z i) / ∑ j, Real.exp (s * Real.log (8 : ℕ) * z j) :=
  ssmax_eq (by norm_num) s z i

end Transformer.BAM
