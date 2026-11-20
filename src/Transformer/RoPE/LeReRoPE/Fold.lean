module

public import Transformer.RoPE.LeRoPE.Section3_Surrogate
public import Transformer.RoPE.ReRoPE.Section2_Fusion

/-!
# LeReRoPE: the gradient of a frequency under ReRoPE's clamp

In no paper. LeRoPE's Eq. (6) and Lemma 3.1 (Karypis et al., arXiv:2607.10134v1, §3.2;
`Transformer.RoPE.LeRoPE.hasDerivAt_loss`,
`Transformer.RoPE.LeRoPE.hasDerivAt_loss_deriv_surrogate`) rotate band `m`'s pair `(s, t)` by
its offset `δ = s − t`. ReRoPE (Su, kexue.fm/archives/9708) rotates it by `ψ(δ) = min(δ, w)`,
Leaky ReRoPE by `w + (δ − w)/k` beyond the window `w` (`Transformer.RoPE.ReRoPE.rectifiedPosition`,
`Transformer.RoPE.ReRoPE.leakyPosition`). For every position map `ψ` with `ψ(0) = 0`:

* `hasDerivAt_positioned_loss`: Eq. (6) becomes
  `∂L/∂θ = −Σ_s Σ_{t<s} ψ(s − t) ‖q_s‖ ‖k_t‖ sin(φ_st + ψ(s − t) θ) ∂L/∂z_st`;
* `hasDerivAt_positioned_loss_deriv_surrogate`: Lemma 3.1 becomes `∂L/∂θ = P_ψ'(θ)` for the
  surrogate `P_ψ(θ) = Re Σ_{δ=1}^{L−1} C(δ) e^{iψ(δ)θ}` of LeRoPE's offset signal `C`;
  `positionedSurrogate_natCast`: `ψ(δ) = δ` gives LeRoPE's `P`;
* `positionedSurrogate_rectified`: under the clamp the far offsets fold into one tap,
  `P_w(θ) = Re (Σ_{δ=1}^{w−1} C(δ) e^{iδθ} + (Σ_{δ=w}^{L−1} C(δ)) e^{iwθ})`. The frequency sees the
  far pairs only through their summed phasor, at the single offset `w`; Leaky ReRoPE keeps every
  far offset, at the phase `(w + (δ − w)/k) θ`.
-/

@[expose] public section

open scoped ComplexConjugate
open Finset

namespace Transformer.RoPE.LeReRoPE

open Transformer.RoPE.LeRoPE Transformer.RoPE.ReRoPE

variable {n : ℕ}

/-- The logits `z_st = r_st + D_st` with band `m`'s pair `(s, t)` rotated by `ψ(s − t) θ`: the
query chunk at the position `ψ(s − t)`, the key chunk at `0`. -/
noncomputable def positionedLogits (ψ : ℕ → ℝ) (r : Fin n × Fin n → ℝ) (q k : Fin n → ℂ)
    (θ : ℝ) : Fin n × Fin n → ℝ :=
  fun p => r p + bandProduct θ (ψ ((p.1 : ℕ) - p.2)) 0 (q p.1) (k p.2)

/-- Eq. (6) **at the positions `ψ`**: for a loss `ℓ` of the logits, differentiable at `z(θ)` with
the derivative `ℓ'` and whose partials at the masked pairs vanish,
`∂L/∂θ = −Σ_s Σ_{t<s} ψ(s − t) ‖q_s‖ ‖k_t‖ sin(φ_st + ψ(s − t) θ) ∂L/∂z_st`. -/
theorem hasDerivAt_positioned_loss {ψ : ℕ → ℝ} (hψ : ψ 0 = 0) {ℓ : (Fin n × Fin n → ℝ) → ℝ}
    {ℓ' : (Fin n × Fin n → ℝ) →L[ℝ] ℝ} {r : Fin n × Fin n → ℝ} {q k : Fin n → ℂ} {θ : ℝ}
    (hℓ : HasFDerivAt ℓ ℓ' (positionedLogits ψ r q k θ))
    (hmask : ∀ s t : Fin n, s < t → logitPartial ℓ' (s, t) = 0) :
    HasDerivAt (fun θ => ℓ (positionedLogits ψ r q k θ))
      (-∑ s : Fin n, ∑ t ∈ Iio s, ψ ((s : ℕ) - t) * ‖q s‖ * ‖k t‖ *
        Real.sin (Complex.arg (q s * conj (k t)) + ψ ((s : ℕ) - t) * θ) *
          logitPartial ℓ' (s, t)) θ := by
  have hz : HasDerivAt (positionedLogits ψ r q k) (fun p => -(ψ ((p.1 : ℕ) - p.2) *
      (‖q p.1‖ * ‖k p.2‖ * Real.sin (Complex.arg (q p.1 * conj (k p.2)) +
        ψ ((p.1 : ℕ) - p.2) * θ)))) θ :=
    hasDerivAt_pi.2 fun p =>
      ((hasDerivAt_bandProduct _ _ _ _ θ).const_add (r p)).congr_deriv (by rw [sub_zero])
  refine (hℓ.comp_hasDerivAt θ hz).congr_deriv ?_
  rw [apply_eq_sum_logitPartial, ← sum_neg_distrib]
  refine sum_congr rfl fun s _ => ?_
  rw [← sum_subset (subset_univ (Iio s)), ← sum_neg_distrib]
  · exact sum_congr rfl fun t _ => by ring
  · intro t _ ht
    rcases (not_lt.1 (mem_Iio.not.1 ht)).lt_or_eq with hst | rfl
    · rw [hmask s t hst, mul_zero]
    · simp [hψ]

/-- The hypotheses of `hasDerivAt_positioned_loss` hold for ReRoPE's positions with the window
`1` and the loss that reads the logit `z_10` of two tokens. -/
example (r : Fin 2 × Fin 2 → ℝ) (q k : Fin 2 → ℂ) (θ : ℝ) :=
  hasDerivAt_positioned_loss (ψ := fun δ => rectifiedPosition 1 δ) (by norm_num [rectifiedPosition])
    (ℓ := fun z => z (1, 0)) (r := r) (q := q) (k := k) (θ := θ)
    (hasFDerivAt_apply (1, 0) _) fun _ _ hst => logitPartial_proj_masked (by decide) hst

/-- LeRoPE's surrogate at the positions `ψ`, `P_ψ(θ) = Re Σ_{δ=1}^{L−1} C(δ) e^{iψ(δ)θ}`, of a
context of `L = n` tokens. -/
noncomputable def positionedSurrogate (ψ : ℕ → ℝ) (q k : Fin n → ℂ) (g : Fin n × Fin n → ℝ)
    (θ : ℝ) : ℝ :=
  (∑ δ ∈ Ico 1 n, offsetSignal q k g δ * Complex.exp ((ψ δ * θ : ℝ) * Complex.I)).re

/-- At the offsets themselves, `ψ(δ) = δ`, the surrogate is LeRoPE's. -/
theorem positionedSurrogate_natCast :
    positionedSurrogate (fun δ : ℕ => (δ : ℝ)) = surrogate (n := n) :=
  rfl

/-- Regrouped by pairs, `P_ψ(θ) = Σ_s Σ_{t<s} g_st D_st`, with the pair `(s, t)` at `ψ(s − t)`. -/
theorem positionedSurrogate_eq (ψ : ℕ → ℝ) (q k : Fin n → ℂ) (g : Fin n × Fin n → ℝ) (θ : ℝ) :
    positionedSurrogate ψ q k g θ =
      ∑ s : Fin n, ∑ t ∈ Iio s, g (s, t) * bandProduct θ (ψ ((s : ℕ) - t)) 0 (q s) (k t) := by
  have h := sum_offset fun p : Fin n × Fin n =>
    g p * bandProduct θ (ψ ((p.1 : ℕ) - p.2)) 0 (q p.1) (k p.2)
  dsimp only at h
  rw [← h, positionedSurrogate, Complex.re_sum]
  refine sum_congr rfl fun δ _ => ?_
  rw [offsetSignal, sum_mul, Complex.re_sum]
  refine sum_congr rfl fun p hp => ?_
  have hδ : δ = (p.1 : ℕ) - p.2 := by
    have := (mem_filter.1 hp).2
    omega
  rw [hδ, re_ofReal_mul_exp_mul_exp, bandProduct_eq, sub_zero]
  ring

/-- The surrogate's derivative,
`P_ψ'(θ) = −Σ_s Σ_{t<s} ψ(s − t) ‖q_s‖ ‖k_t‖ sin(φ_st + ψ(s − t) θ) g_st`. -/
theorem hasDerivAt_positionedSurrogate (ψ : ℕ → ℝ) (q k : Fin n → ℂ) (g : Fin n × Fin n → ℝ)
    (θ : ℝ) :
    HasDerivAt (positionedSurrogate ψ q k g)
      (-∑ s : Fin n, ∑ t ∈ Iio s, ψ ((s : ℕ) - t) * ‖q s‖ * ‖k t‖ *
        Real.sin (Complex.arg (q s * conj (k t)) + ψ ((s : ℕ) - t) * θ) * g (s, t)) θ := by
  rw [funext (positionedSurrogate_eq ψ q k g)]
  refine (HasDerivAt.fun_sum fun s _ => HasDerivAt.fun_sum fun t _ =>
    (hasDerivAt_bandProduct _ _ (q s) (k t) θ).const_mul (g (s, t))).congr_deriv ?_
  simp only [sub_zero]
  rw [← sum_neg_distrib]
  refine sum_congr rfl fun s _ => ?_
  rw [← sum_neg_distrib]
  exact sum_congr rfl fun t _ => by ring

/-- Lemma 3.1 **at the positions `ψ`**: with the weights `g_st = ∂L/∂z_st` at `θ`, frozen,
`∂L/∂θ = P_ψ'(θ)`. -/
theorem hasDerivAt_positioned_loss_deriv_surrogate {ψ : ℕ → ℝ} (hψ : ψ 0 = 0)
    {ℓ : (Fin n × Fin n → ℝ) → ℝ} {ℓ' : (Fin n × Fin n → ℝ) →L[ℝ] ℝ} {r : Fin n × Fin n → ℝ}
    {q k : Fin n → ℂ} {θ : ℝ} (hℓ : HasFDerivAt ℓ ℓ' (positionedLogits ψ r q k θ))
    (hmask : ∀ s t : Fin n, s < t → logitPartial ℓ' (s, t) = 0) :
    HasDerivAt (fun θ => ℓ (positionedLogits ψ r q k θ))
      (deriv (positionedSurrogate ψ q k (logitPartial ℓ')) θ) θ := by
  rw [(hasDerivAt_positionedSurrogate ψ q k _ θ).deriv]
  exact hasDerivAt_positioned_loss hψ hℓ hmask

/-- The hypotheses of `hasDerivAt_positioned_loss_deriv_surrogate` hold for Leaky ReRoPE's
positions with the window `1` and `k = 16`, and the loss that reads the logit `z_10` of two
tokens. -/
example (r : Fin 2 × Fin 2 → ℝ) (q k : Fin 2 → ℂ) (θ : ℝ) :=
  hasDerivAt_positioned_loss_deriv_surrogate (ψ := fun δ => leakyPosition 1 16 δ)
    (by norm_num [leakyPosition]) (ℓ := fun z => z (1, 0)) (r := r) (q := q) (k := k) (θ := θ)
    (hasFDerivAt_apply (1, 0) _) fun _ _ hst => logitPartial_proj_masked (by decide) hst

/-- **Under ReRoPE's clamp the far offsets fold into one tap**: for a window `1 ≤ w ≤ L`,
`P_w(θ) = Re (Σ_{δ=1}^{w−1} C(δ) e^{iδθ} + (Σ_{δ=w}^{L−1} C(δ)) e^{iwθ})`. -/
theorem positionedSurrogate_rectified {w : ℕ} (hw : 1 ≤ w) (hwn : w ≤ n) (q k : Fin n → ℂ)
    (g : Fin n × Fin n → ℝ) (θ : ℝ) :
    positionedSurrogate (fun δ => rectifiedPosition w δ) q k g θ =
      (∑ δ ∈ Ico 1 w, offsetSignal q k g δ * Complex.exp ((δ * θ : ℝ) * Complex.I) +
        (∑ δ ∈ Ico w n, offsetSignal q k g δ) * Complex.exp ((w * θ : ℝ) * Complex.I)).re := by
  rw [positionedSurrogate, ← sum_Ico_consecutive _ hw hwn, sum_mul]
  congr 2
  · refine sum_congr rfl fun δ hδ => ?_
    rw [rectifiedPosition, min_eq_left (by exact_mod_cast (mem_Ico.1 hδ).2.le)]
  · refine sum_congr rfl fun δ hδ => ?_
    rw [rectifiedPosition, min_eq_right (by exact_mod_cast (mem_Ico.1 hδ).1)]

/-- The hypotheses of `positionedSurrogate_rectified` hold for the window `1024` of `2048`
tokens. -/
example (q k : Fin 2048 → ℂ) (g : Fin 2048 × Fin 2048 → ℝ) (θ : ℝ) :=
  positionedSurrogate_rectified (w := 1024) (by norm_num) (by norm_num) q k g θ

end Transformer.RoPE.LeReRoPE
